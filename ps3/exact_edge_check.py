"""Exact collision checking of a straight edge in configuration space, after
Schwarzer, Saha and Latombe: clearance at the two ends of an edge, and a bound
on how far any point of the arm can travel, certify everything in between."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
from itertools import pairwise, product
from typing import Callable

import numpy as np
from pydrake.all import Box

# The scene, the planning problem and the planners are the RRT problem's.
from rrt_planning import (
    EDGE_STEP,
    GRIPPER_SETPOINT,
    LEFT_DOOR_ANGLE,
    NUM_JOINTS,
    RIGHT_DOOR_ANGLE,
    ManipulationStationSim,
    PlanningProblem,
    first_collision,
    how_long,
    iiwa_configuration_space,
    interpolate,
    make_problem,
    report,
    rrt_connect_planning,
)
from utils.viz import keep_meshcat_open

######################################################################
## Support code
######################################################################

# Distances beyond this, in meters, are reported as this.  A clearance only
# has to be a lower bound, and capping it keeps the distance query cheap.
CLEARANCE_CAP = 0.5


def robot_bodies(station: ManipulationStationSim) -> list:
    """The bodies of the arm and its gripper."""
    plant = station.plant
    return [
        plant.get_body(index)
        for model in (station.iiwa, station.wsg)
        for index in plant.GetBodyIndices(model)
    ]


def robot_geometry_ids(station: ManipulationStationSim) -> set:
    """The collision geometries of the arm and its gripper."""
    return {
        geometry_id
        for body in robot_bodies(station)
        for geometry_id in station.plant.GetCollisionGeometriesForBody(body)
    }


def box_corners(station: ManipulationStationSim, body) -> list[np.ndarray]:
    """World positions of the corners of every collision box on `body`, with
    the station in whatever configuration it was last set to."""
    inspector = station.scene_graph.model_inspector()
    query_object = station.query_output_port.Eval(station.context_scene_graph)
    corners = []
    for geometry_id in station.plant.GetCollisionGeometriesForBody(body):
        shape = inspector.GetShape(geometry_id)
        assert isinstance(shape, Box), "the bound below is written for boxes"
        X_WBox = query_object.GetPoseInWorld(geometry_id)
        half = np.array([shape.width(), shape.depth(), shape.height()]) / 2
        for signs in product([-1, 1], repeat=3):
            corners.append(X_WBox @ (half * signs))
    return corners


def distance_to_line(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Distance from point p to the line through `origin` along the unit
    vector `direction`."""
    offset = p - origin
    return float(np.linalg.norm(offset - (offset @ direction) * direction))


def joint_radii(station: ManipulationStationSim) -> np.ndarray:
    """r[j]: a bound for every arm configuration at the fixed gripper opening.

    It bounds the distance from joint j+1's axis to every point of the arm or
    gripper that it moves, using this serial arm's box collision geometries.

    A box is farthest from a line at one of its corners.  Corners on the link
    the joint drives directly are measured from the axis.  Corners on a later
    link k are bounded by straightening the arm: the distance from the axis
    to the next joint's origin, then origin to origin along the chain, then
    from link k's joint origin to the corner.  Each of those lengths is fixed
    by the arm's geometry, so the configuration they are measured in does not
    matter.
    """
    plant = station.plant
    station.SetStationConfiguration(
        np.zeros(NUM_JOINTS), GRIPPER_SETPOINT, LEFT_DOOR_ANGLE, RIGHT_DOOR_ANGLE
    )

    origins, axes, corners = [], [], []
    for j in range(NUM_JOINTS):
        joint = plant.GetJointByName(f"iiwa_joint_{j + 1}")
        X_WJ = joint.frame_on_child().CalcPoseInWorld(station.context_plant)
        origins.append(X_WJ.translation())
        axes.append(X_WJ.rotation() @ joint.revolute_axis())
        corners.append(box_corners(station, joint.child_body()))
    # With the finger opening fixed, all gripper geometries move with link 7.
    for body in robot_bodies(station):
        if body.model_instance() == station.wsg:
            corners[-1] += box_corners(station, body)

    # reach_from_origin[j]: how far any point on link j+1 or later can be
    # from the origin of joint j+1.
    reach_from_origin = np.zeros(NUM_JOINTS)
    radii = np.zeros(NUM_JOINTS)
    for j in reversed(range(NUM_JOINTS)):
        reach_from_origin[j] = max(np.linalg.norm(c - origins[j]) for c in corners[j])
        radii[j] = max(distance_to_line(c, origins[j], axes[j]) for c in corners[j])
        if j + 1 < NUM_JOINTS:
            link_length = np.linalg.norm(origins[j + 1] - origins[j])
            reach_from_origin[j] = max(
                reach_from_origin[j], link_length + reach_from_origin[j + 1]
            )
            radii[j] = max(
                radii[j],
                distance_to_line(origins[j + 1], origins[j], axes[j])
                + reach_from_origin[j + 1],
            )
    return radii


######################################################################
## Code for students to be aware of
######################################################################

# Reject sampled configurations with obstacle clearance at most this (meters).
# The certificate proves positive clearance between samples, not this margin.
DELTA = 0.005


class ClearanceChecker:
    """Distance from the arm to the static obstacles, as a function of q."""

    def __init__(self, station: ManipulationStationSim) -> None:
        self.station = station
        self.robot_ids = robot_geometry_ids(station)
        self.num_calls = 0

    def clearance(self, q: tuple) -> float:
        """eta(q): the smallest distance, in meters, between a geometry of the
        arm or gripper and a geometry of anything else.  It is negative when
        they overlap, and never more than CLEARANCE_CAP.

        Robot-robot pairs are left out; Drake's scene collision filters also
        apply.  Capping the clearance gives a safe lower bound for certification.
        """
        self.num_calls += 1
        self.station.SetStationConfiguration(
            np.array(q), GRIPPER_SETPOINT, LEFT_DOOR_ANGLE, RIGHT_DOOR_ANGLE
        )
        query_object = self.station.query_output_port.Eval(
            self.station.context_scene_graph
        )
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(CLEARANCE_CAP)
        distances = [
            pair.distance
            for pair in pairs
            if (pair.id_A in self.robot_ids) != (pair.id_B in self.robot_ids)
        ]
        return min(distances, default=CLEARANCE_CAP)


class ExactPlanningProblem(PlanningProblem):
    """The RRT planning problem, with edges checked by exact_edge_is_free
    instead of at a fixed EDGE_STEP."""

    def __init__(self, problem: PlanningProblem, rho: float, delta: float = DELTA):
        super().__init__(
            problem.cspace, problem.start, problem.goal, problem.collision_checker
        )
        self.checker = ClearanceChecker(problem.collision_checker)
        self.rho = rho
        self.delta = delta

    def edge_is_free(self, q_from: tuple, q_to: tuple) -> bool:
        return exact_edge_is_free(
            self.checker.clearance, self.collide, q_from, q_to, self.rho, self.delta
        )


######################################################################
## Code for students to write
######################################################################


def motion_bound(radii: np.ndarray) -> float:
    """rho: no point of the arm travels farther than rho * ||q - q'||_2, in
    meters, when the configuration moves in a straight line from q to q'."""
    raise NotImplementedError("your code here")


def is_certified(
    q_a: tuple, eta_a: float, q_b: tuple, eta_b: float, rho: float
) -> bool:
    """The theorem: whether the clearances at the two ends are enough to
    prove the whole segment free."""
    raise NotImplementedError("your code here")


def exact_edge_is_free(
    clearance: Callable[[tuple], float],
    collide: Callable[[tuple], bool],
    q_from: tuple,
    q_to: tuple,
    rho: float,
    delta: float,
) -> bool:
    """Certify positive clearance from the static obstacles by bisection.

    Reject sampled configurations with clearance <= delta.  The certificate
    does not guarantee clearance > delta between those configurations.

    `collide` is also asked about every configuration that gets tested, which
    detects self-collision at those samples only.  Once a clearance query
    rejects a sample, no additional collision query is needed for it.
    """

    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def test1() -> None:
    """The motion bound for this arm, the clearance at the two ends of the
    problem, and the padding that would make fixed-step checking sound."""
    problem = make_problem(is_visualizing=False)
    radii = joint_radii(problem.collision_checker)
    rho = motion_bound(radii)
    print(f"joint radii r_j (m): {np.round(radii, 3)}")
    print(f"rho = {rho:.3f} m/rad")

    checker = ClearanceChecker(problem.collision_checker)
    print(f"clearance at q_start: {100 * checker.clearance(problem.start):.1f} cm")
    print(f"clearance at q_goal:  {100 * checker.clearance(problem.goal):.1f} cm")

    # interpolate() lets every joint move EDGE_STEP at once.
    step_length = np.sqrt(NUM_JOINTS) * EDGE_STEP
    print(
        f"an EDGE_STEP is up to {step_length:.4f} rad long, and is sound "
        f"if samples keep {100 * rho * step_length / 2:.1f} cm clear"
    )


def test2() -> None:
    """Plan with the fixed-step planner, then put each edge of the path it
    returns through the exact check."""
    problem = make_problem(is_visualizing=False)
    rho = motion_bound(joint_radii(problem.collision_checker))
    path, num_iter = rrt_connect_planning(problem)
    report("RRT-Connect", path, num_iter, problem)
    if path is None:
        return

    checker = ClearanceChecker(problem.collision_checker)
    rejected = 0
    collision_calls = 0

    def counted_collide(q: tuple) -> bool:
        nonlocal collision_calls
        collision_calls += 1
        return problem.collide(q)

    started = time.perf_counter()
    for q_from, q_to in pairwise(path):
        rejected += not exact_edge_is_free(
            checker.clearance, counted_collide, q_from, q_to, rho, DELTA
        )
    adaptive_seconds = time.perf_counter() - started
    fixed_step_calls = sum(
        len(interpolate(q_from, q_to, EDGE_STEP)) for q_from, q_to in pairwise(path)
    )
    # Time the same full recheck whose query count is reported below.
    started = time.perf_counter()
    for q_from, q_to in pairwise(path):
        for q in interpolate(q_from, q_to, EDGE_STEP):
            problem.collide(q)
    fixed_seconds = time.perf_counter() - started
    print(
        f"returned-path check times: adaptive {adaptive_seconds:.3f} s; fixed-step {fixed_seconds:.3f} s"
    )
    print(
        f"adaptive checker rejected {rejected} of {len(path) - 1} edges "
        f"(sample rejection threshold: {1000 * DELTA:.0f} mm)"
    )
    print(
        f"returned-path adaptive check: {checker.num_calls} clearance queries "
        f"and {collision_calls} collision queries; a full fixed-step recheck "
        f"would use {fixed_step_calls} collision queries"
    )


def test3(trials: int = 5) -> None:
    """Plan with the exact check inside RRT-Connect."""
    problem = make_problem(is_visualizing=False)
    rho = motion_bound(joint_radii(problem.collision_checker))
    exact_problem = ExactPlanningProblem(problem, rho)
    returned, failed_recheck = 0, 0
    for _ in range(trials):
        t0 = time.time()
        path, num_iter = rrt_connect_planning(exact_problem)
        print(f"RRT-Connect with the exact check took {how_long(time.time() - t0)}")
        passed = report("RRT-Connect", path, num_iter, exact_problem)
        if path is not None:
            returned += 1
            failed_recheck += not passed
    if returned == 0:
        print(f"no returned paths to evaluate in {trials} trials")
    else:
        print(
            f"finer recheck failed for {failed_recheck} of {returned} returned paths ({trials} trials)"
        )


if __name__ == "__main__":
    # Every test needs motion_bound, so there is nothing to run until it is
    # written: uncomment one then.
    # test1()
    # test2()
    # test3()
    keep_meshcat_open()
