"""Sampling-based motion planning for the iiwa: RRT and RRT-Connect."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
from itertools import pairwise
from random import random
from typing import Literal, Protocol

import numpy as np
from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    RRT,
    TreeNode,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    ConstantVectorSource,
    ContactResults,
    DiagramBuilder,
    EventStatus,
    MultibodyPlant,
    PiecewisePolynomial,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    TrajectorySource,
)

from utils.viz import (
    HEADLESS,
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    sim_duration,
    start_recording,
)

######################################################################
## Support code
######################################################################

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
NUM_JOINTS = 7

# The arm is welded a quarter meter behind the world origin.
X_WL0 = RigidTransform([-0.25, 0, 0])

# Rigid transform from link 7 to the gripper
GRIPPER_OFFSET = 0.114
X_L7G = RigidTransform(RollPitchYaw(np.pi / 2, 0, np.pi / 2), [0, 0, GRIPPER_OFFSET])

# The cupboard doors are held open throughout; the gripper is open too.
GRIPPER_SETPOINT = 0.1
LEFT_DOOR_ANGLE = -np.pi / 2
RIGHT_DOOR_ANGLE = np.pi / 2

SCENARIO_YAML = f"""directives:
- add_model:
    name: iiwa
    file: {IIWA_URL}
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.5]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.9]
        iiwa_joint_5: [0]
        iiwa_joint_6: [0.65]
        iiwa_joint_7: [1.7]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [{X_WL0.translation()[0]}, 0, 0]
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, {GRIPPER_OFFSET}]
        rotation: !Rpy {{ deg: [90, 0, 90] }}
- add_model:
    name: table
    file: package://drake_models/manipulation_station/amazon_table_simplified.sdf
- add_weld:
    parent: world
    child: table::amazon_table
    X_PC:
        translation: [0.3257, 0, -0.0127]
- add_model:
    name: cupboard
    file: package://manipulation/hydro/cupboard.sdf
    default_joint_positions:
        left_door_hinge: [{LEFT_DOOR_ANGLE}]
        right_door_hinge: [{RIGHT_DOOR_ANGLE}]
- add_weld:
    parent: world
    child: cupboard::cupboard_body
    X_PC:
        translation: [0.9057, 0, 0.4148]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: bin
    file: package://manipulation/hydro/bin.sdf
- add_weld:
    parent: world
    child: bin::bin_base
    X_PC:
        translation: [0.2, 0, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: mustard
    file: package://manipulation/hydro/006_mustard_bottle.sdf
    default_free_body_pose:
        base_link_mustard:
            base_frame: world
            translation: [0.43, 0, 0.215]
model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""

# A returned path is re-checked this finely: a tenth of the EDGE_STEP the
# planner itself used.
CHECK_STEP = np.radians(0.1)


def first_collision(
    path: list[tuple], problem: "PlanningProblem", max_step: float = CHECK_STEP
) -> tuple | None:
    """The first configuration along the path that is in collision, or None if
    the path is free.
    """
    for q_from, q_to in pairwise(path):
        for q in interpolate(q_from, q_to, max_step):
            if problem.collide(q):
                return q
    return None


def report(
    name: str, path: list[tuple] | None, num_iter: int, problem: "PlanningProblem"
) -> bool:
    """Report a path; return whether it exists and passes the finer check.

    Passing this sampled check does not certify continuous collision freedom.
    """
    if path is None:
        print(f"{name}: no path found in {num_iter} iterations")
        return False
    print(
        f"{name}: {len(path)} waypoints after {num_iter} iterations, "
        f"path length {problem.path_distance(path):.2f} rad"
    )
    assert path[0] == problem.start and path[-1] == problem.goal
    q_bad = first_collision(path, problem)
    check = f"checked every {np.degrees(CHECK_STEP)} deg"
    if q_bad is None:
        print(f"{name}: passed finer sampled collision check, {check}")
    else:
        print(f"{name}: IN COLLISION at {np.round(q_bad, 3)}, {check}")
    return q_bad is None


######################################################################
## Code for students to be aware of
######################################################################

# Seconds between frames when a path is drawn rather than simulated.
FRAME_DELAY = 0.2


class ManipulationStationSim:
    """The station, with its context held open so poses can be set and queried.

    This is the collision checker: `ExistsCollision` puts the arm at `q_iiwa`
    and asks SceneGraph whether anything is penetrating anything else.
    `iiwa_url` picks the arm model, and with it the arm's collision geometry.
    """

    def __init__(self, is_visualizing: bool = False, iiwa_url: str = IIWA_URL) -> None:
        builder = DiagramBuilder()
        scenario = LoadScenario(data=SCENARIO_YAML.replace(IIWA_URL, iiwa_url))
        self.station = builder.AddSystem(
            MakeHardwareStation(
                scenario, meshcat=get_meshcat() if is_visualizing else None
            )
        )
        self.plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.wsg = self.plant.GetModelInstanceByName("wsg")
        self.is_visualizing = is_visualizing

        self.query_output_port = self.scene_graph.GetOutputPort("query")

        self.diagram = builder.Build()

        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram
        )
        self.station.GetInputPort("iiwa.position").FixValue(
            self.context_station, np.zeros(NUM_JOINTS)
        )
        self.station.GetInputPort("wsg.position").FixValue(
            self.context_station, [GRIPPER_SETPOINT]
        )
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station
        )
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station
        )

        self.q0 = self.plant.GetPositions(self.context_plant, self.iiwa)
        if is_visualizing:
            self.DrawStation(
                self.q0, GRIPPER_SETPOINT, LEFT_DOOR_ANGLE, RIGHT_DOOR_ANGLE
            )

    def SetStationConfiguration(
        self,
        q_iiwa: np.ndarray,
        gripper_setpoint: float,
        left_door_angle: float,
        right_door_angle: float,
    ) -> None:
        self.plant.SetPositions(self.context_plant, self.iiwa, q_iiwa)
        # The gripper's two fingers each move half the opening.
        self.plant.SetPositions(
            self.context_plant, self.wsg, [-gripper_setpoint / 2, gripper_setpoint / 2]
        )

        # The left hinge opens the other way.
        if left_door_angle > 0:
            left_door_angle *= -1
        self.plant.GetJointByName("left_door_hinge").set_angle(
            context=self.context_plant, angle=left_door_angle
        )
        self.plant.GetJointByName("right_door_hinge").set_angle(
            context=self.context_plant, angle=right_door_angle
        )

    def DrawStation(
        self,
        q_iiwa: np.ndarray,
        gripper_setpoint: float,
        q_door_left: float,
        q_door_right: float,
    ) -> None:
        if not self.is_visualizing:
            return
        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right
        )
        self.diagram.ForcedPublish(self.context_diagram)

    def ExistsCollision(
        self,
        q_iiwa: np.ndarray,
        gripper_setpoint: float,
        q_door_left: float,
        q_door_right: float,
    ) -> bool:
        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint, q_door_left, q_door_right
        )
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        return len(query_object.ComputePointPairPenetration()) > 0

    def collide(self, configuration: tuple) -> bool:
        """Whether the arm at this configuration touches anything."""
        return self.ExistsCollision(
            np.array(configuration),
            GRIPPER_SETPOINT,
            LEFT_DOOR_ANGLE,
            RIGHT_DOOR_ANGLE,
        )

    def gripper_pose(self, q_iiwa: np.ndarray) -> RigidTransform:
        """Where the gripper frame G sits with the arm at `q_iiwa`."""
        self.SetStationConfiguration(
            q_iiwa, GRIPPER_SETPOINT, LEFT_DOOR_ANGLE, RIGHT_DOOR_ANGLE
        )
        X_WL7 = self.plant.EvalBodyPoseInWorld(
            self.context_plant, self.plant.GetBodyByName("iiwa_link_7", self.iiwa)
        )
        return X_WL7 @ X_L7G

    def visualize_path(
        self, path: list[tuple] | None, delay: float = FRAME_DELAY
    ) -> None:
        """Draw the arm at each configuration along a path, in turn, pausing
        `delay` seconds on each."""
        if path is None:
            return
        for q in path:
            self.DrawStation(
                np.array(q), GRIPPER_SETPOINT, LEFT_DOOR_ANGLE, RIGHT_DOOR_ANGLE
            )
            if not HEADLESS:
                time.sleep(delay)


# Where the arm should end up: the gripper pointing straight down at about
# [0.64, -0.19, 0.36], just in front of the bottle.
Q_GOAL = (-0.7137, 1.5313, 1.5567, -0.8832, -2.9634, 0.1729, 1.4317)


# PlanningProblem is quoted in this signature because it is defined below, in
# the section on the planning problem itself.
def make_problem(is_visualizing: bool) -> "PlanningProblem":
    station = ManipulationStationSim(is_visualizing)
    q_start = station.q0
    q_goal = np.array(Q_GOAL)
    if is_visualizing:
        meshcat = get_meshcat()
        if meshcat is not None:
            AddMeshcatTriad(
                meshcat, "goal pose", X_PT=station.gripper_pose(q_goal), opacity=0.5
            )
    cspace = iiwa_configuration_space(station.plant)
    return PlanningProblem(cspace, q_start, q_goal, station)


def how_long(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds >= 1.0 else f"{1000 * seconds:.0f} ms"


def body_name(plant: MultibodyPlant, inspector, geometry_id) -> str:
    """The name of the body a piece of collision geometry belongs to."""
    return plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id)).name()


def robot_contacts(
    results: ContactResults, plant: MultibodyPlant, inspector
) -> dict[tuple[str, str], float]:
    """Every contact involving the arm or its gripper, as {(body, body): newtons}.

    The plant's contact model is hydroelastic with a point-pair fallback, there are
    two types of contacts: a point-pair contact names its two bodies
    outright, while a hydroelastic one names geometries, which the scene
    graph's inspector turns back into bodies.
    """
    robot_bodies = {
        plant.get_body(index).name()
        for model in ("iiwa", "wsg")
        for index in plant.GetBodyIndices(plant.GetModelInstanceByName(model))
    }

    contacts = []
    for i in range(results.num_point_pair_contacts()):
        info = results.point_pair_contact_info(i)
        contacts.append(
            (
                plant.get_body(info.bodyA_index()).name(),
                plant.get_body(info.bodyB_index()).name(),
                np.linalg.norm(info.contact_force()),
            )
        )
    for i in range(results.num_hydroelastic_contacts()):
        info = results.hydroelastic_contact_info(i)
        surface = info.contact_surface()
        contacts.append(
            (
                body_name(plant, inspector, surface.id_M()),
                body_name(plant, inspector, surface.id_N()),
                np.linalg.norm(info.F_Ac_W().translational()),
            )
        )

    # Each pair is keyed with the robot's body first, so that several contacts
    # between the same two bodies collapse to the worst one.
    forces: dict[tuple[str, str], float] = {}
    for name_a, name_b, force in contacts:
        if name_a not in robot_bodies and name_b not in robot_bodies:
            continue
        pair = (name_a, name_b) if name_a in robot_bodies else (name_b, name_a)
        forces[pair] = max(forces.get(pair, 0.0), float(force))
    return forces


def free_body_positions(plant: MultibodyPlant, context) -> dict[str, np.ndarray]:
    """Where each body that is free to move --- here, only the bottle --- is."""
    return {
        plant.get_body(index).name(): plant.EvalBodyPoseInWorld(
            context, plant.get_body(index)
        ).translation()
        for index in plant.GetFloatingBaseBodies()
    }


# Seconds between waypoints when a path is simulated.
WAYPOINT_DT = 0.2

# How long the simulation keeps running after the last waypoint, so that
# anything the arm knocked over comes to rest instead of freezing mid-fall.
SETTLE_TIME = 2.0

# Contact quieter than this, in newtons, is the near miss that the plant's
# millimeter of penetration allowance turns into a whisper of force.  Real
# contact with the arm moving is tens of newtons.
GRAZE = 1.0

# A free body that ends up within this of where it started, in meters, only
# settled onto its shelf; anything further was knocked there by the arm.
NUDGE = 0.01


def simulate_path(path: list[tuple] | None) -> None:
    """Play the path through the station, with the physics running."""
    if path is None:
        return

    builder = DiagramBuilder()
    scenario = LoadScenario(data=SCENARIO_YAML)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=get_meshcat()))

    times = [WAYPOINT_DT * i for i in range(len(path))]
    traj = PiecewisePolynomial.FirstOrderHold(times, np.column_stack(path))
    iiwa_src = builder.AddSystem(TrajectorySource(traj))
    wsg_src = builder.AddSystem(ConstantVectorSource([GRIPPER_SETPOINT]))
    builder.Connect(iiwa_src.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(wsg_src.get_output_port(), station.GetInputPort("wsg.position"))
    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant = station.GetSubsystemByName("plant")
    inspector = station.GetSubsystemByName("scene_graph").model_inspector()
    context_plant = plant.GetMyContextFromRoot(context)
    started_at = free_body_positions(plant, context_plant)

    contact_port = station.GetOutputPort("contact_results")
    peak: dict[tuple[str, str], float] = {}
    lasted: dict[tuple[str, str], float] = {}
    previous_t = 0.0

    def watch_contact(root_context) -> EventStatus:
        nonlocal previous_t
        t = root_context.get_time()
        step, previous_t = t - previous_t, t
        results = contact_port.Eval(station.GetMyContextFromRoot(root_context))
        for pair, force in robot_contacts(results, plant, inspector).items():
            peak[pair] = max(peak.get(pair, 0.0), force)
            lasted[pair] = lasted.get(pair, 0.0) + step
        return EventStatus.Succeeded()

    simulator.set_monitor(watch_contact)

    diagram.ForcedPublish(context)
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    start_recording()
    simulator.AdvanceTo(sim_duration(traj.end_time() + SETTLE_TIME))
    publish_recording()

    hits = [(pair, force) for pair, force in peak.items() if force > GRAZE]
    if hits:
        print("playback: the arm made contact")
        for (robot_body, other), force in sorted(hits, key=lambda kv: -kv[1]):
            print(
                f"   {robot_body} against {other}, peaking at {force:.0f} N "
                f"over {how_long(lasted[(robot_body, other)])}"
            )
    else:
        print("playback: the arm touched nothing")
    ended_at = free_body_positions(plant, context_plant)
    for name, p_start in started_at.items():
        moved = float(np.linalg.norm(ended_at[name] - p_start))
        if moved > NUDGE:
            print(f"   {name} ended up {100 * moved:.0f} cm from where it started")


######################################################################
## Code for students to study
######################################################################

# A straight line between two configurations is checked for collision at
# this spacing, per joint
EDGE_STEP = np.radians(1.0)  # This is *one degree*, converted to radians


def interpolate(q_from: tuple, q_to: tuple, max_step: float) -> list[tuple]:
    """The straight line from q_from to q_to, sampled so that no joint moves
    more than max_step from one configuration to the next.  Both ends are
    included."""
    dq = np.array(q_to) - np.array(q_from)
    steps = max(1, int(np.ceil(np.abs(dq).max() / max_step)))
    # q_to itself is the last element, rather than a value computed from it:
    # callers test whether they arrived with ==.
    inner = [tuple(np.array(q_from) + dq * (i / steps)) for i in range(steps)]
    return inner + [q_to]


def iiwa_configuration_space(plant: MultibodyPlant) -> ConfigurationSpace:
    """Store some useful properties of the robot: joint ranges and a function
    for measuring the distance between two configurations."""
    ranges = []
    for i in range(NUM_JOINTS):
        joint = plant.GetJointByName(f"iiwa_joint_{i + 1}")
        lower = joint.position_lower_limits().item()
        upper = joint.position_upper_limits().item()
        ranges.append(Range(lower, upper))

    def l2_distance(dq: tuple) -> float:
        return float(np.sqrt(sum(dq_i**2 for dq_i in dq)))

    return ConfigurationSpace(ranges, l2_distance, NUM_JOINTS * [EDGE_STEP])


class CollisionChecker(Protocol):
    """Signature for a collision checker"""

    def collide(self, configuration: tuple) -> bool: ...


class PlanningProblem:
    """a configuration space, a start and a goal, and a collision checker."""

    def __init__(
        self,
        cspace: ConfigurationSpace,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        collision_checker: CollisionChecker,
    ) -> None:
        self.cspace = cspace
        self.start = tuple(q_start)
        self.goal = tuple(q_goal)
        self.collision_checker = collision_checker

        for q in (self.start, self.goal):
            assert self.cspace.valid_configuration(q), "outside the joint limits"
            assert not self.collide(q), "starts or ends in collision"

    def collide(self, configuration: tuple) -> bool:
        return self.collision_checker.collide(configuration)

    def edge_is_free(self, q_from: tuple, q_to: tuple) -> bool:
        """Whether the straight line from q_from to q_to is collision-free,
        checked every EDGE_STEP."""
        return not any(self.collide(q) for q in interpolate(q_from, q_to, EDGE_STEP))

    def path_distance(self, path: list[tuple]) -> float:
        """Total joint-space length of a path."""
        return sum(
            self.cspace.distance(path[i], path[i + 1]) for i in range(len(path) - 1)
        )


# An extension moves at most this many EDGE_STEPs toward its target.
EXTEND_STEPS = 10


class RRT_tools:
    """Basic operations for an RRT.

    A configuration is a tuple `q`; a node in a tree is a `TreeNode`, whose
    configuration is `node.value` and whose parent is `node.parent`.  The
    methods that touch a tree take it as an argument, since RRT-Connect has
    two.
    """

    def __init__(self, problem: PlanningProblem) -> None:
        self.problem = problem

    def new_tree(self, q_root: tuple) -> RRT:
        """A tree with a single node, at q_root."""
        return RRT(TreeNode(q_root), self.problem.cspace)

    def sample_configuration(self) -> tuple:
        """A configuration drawn uniformly from within the joint limits."""
        return self.problem.cspace.sample()

    def nearest_node(self, tree: RRT, q: tuple) -> TreeNode:
        """The node of `tree` closest to q."""
        return tree.nearest(q)

    def step_toward(self, q_from: tuple, q_target: tuple) -> tuple:
        """The configuration EXTEND_STEPS interpolation steps along the
        straight line from q_from to q_target, or q_target itself if it is
        closer than that."""
        line = interpolate(q_from, q_target, EDGE_STEP)
        return line[min(EXTEND_STEPS, len(line) - 1)]

    def edge_is_free(self, q_from: tuple, q_to: tuple) -> bool:
        """Whether the straight line from q_from to q_to is collision-free."""
        return self.problem.edge_is_free(q_from, q_to)

    def add_node(self, tree: RRT, parent_node: TreeNode, q: tuple) -> TreeNode:
        """Add q to `tree` as a child of parent_node; return its node."""
        return tree.add_configuration(parent_node, q)

    def path_from_root(self, node: TreeNode) -> list[tuple]:
        """The configurations from the root down to node, following parents."""
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

    @staticmethod
    def join_paths(path_a: list[tuple], path_b: list[tuple]) -> list[tuple]:
        """Join two paths, dropping the duplicate where they meet."""
        if path_a and path_b and path_a[-1] == path_b[0]:
            return path_a + path_b[1:]
        return path_a + path_b


# How an extension ended: blocked before it could move, a node added short of
# the target, or the target itself added.
Status = Literal["Trapped", "Advanced", "Reached"]


######################################################################
## Code for students to write
######################################################################

MAX_ITERATIONS = 1000

# RRT: how often the goal itself is used as the sample.
GOAL_BIAS = 0.15


def extend(
    tools: RRT_tools, tree: RRT, q_target: tuple
) -> tuple[Status, TreeNode | None]:
    """EXTEND: grow `tree` one step from its nearest node toward q_target.

    Returns the status and the node added, which is None when Trapped.
    """
    raise NotImplementedError("your code here")


def rrt_planning(
    problem: PlanningProblem,
    max_iterations: int = MAX_ITERATIONS,
    prob_sample_q_goal: float = GOAL_BIAS,
) -> tuple[list[tuple] | None, int]:
    """BUILD_RRT, with the goal as the sample some of the time, stopping as
    soon as the tree reaches it.

    Returns (path, iterations): path is [q_start, ..., q_goal] as a list of
    configurations, or None if the goal was not reached in max_iterations.
    """
    raise NotImplementedError("your code here")


def rrt_connect_planning(
    problem: PlanningProblem, max_iterations: int = MAX_ITERATIONS
) -> tuple[list[tuple] | None, int]:
    """RRT_CONNECT_PLANNER: grow trees from both ends, alternating, until one
    reaches the other.

    Returns (path, iterations) exactly as rrt_planning does.
    """
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


# Seconds a single configuration is held when it is shown on its own.
POSE_DELAY = 2.0


def test1() -> None:
    """Show the scene and the goal frame, with the arm first at q_start and
    then, after a pause, at q_goal."""
    problem = make_problem(is_visualizing=True)
    print(f"q_start = {np.round(problem.start, 3)}")
    print(f"q_goal  = {np.round(problem.goal, 3)}")
    problem.collision_checker.visualize_path(
        [problem.start, problem.goal], delay=POSE_DELAY
    )


def test2(max_iterations: int = MAX_ITERATIONS) -> None:
    """Plan with RRT and play the path.  Expect to run it more than once."""
    problem = make_problem(is_visualizing=False)
    t0 = time.time()
    path, num_iter = rrt_planning(problem, max_iterations, prob_sample_q_goal=GOAL_BIAS)
    print(f"RRT took {time.time() - t0:.1f} s")
    report("RRT", path, num_iter, problem)
    simulate_path(path)


def test3(max_iterations: int = MAX_ITERATIONS) -> None:
    """Plan with RRT-Connect and play the path.  It seldom needs a second run."""
    problem = make_problem(is_visualizing=False)
    t0 = time.time()
    path, num_iter = rrt_connect_planning(problem, max_iterations)
    print(f"RRT-Connect took {time.time() - t0:.1f} s")
    report("RRT-Connect", path, num_iter, problem)
    simulate_path(path)


def test4(trials: int = 20) -> None:
    """Iterations to solve, RRT against RRT-Connect, over a few trials, and
    how many of the paths found survive the finer collision check.

    Success and collision-recheck rates vary with random samples; no fixed
    rate is an expected answer.  An iteration is an outer planner iteration,
    so RRT-Connect may make several extensions within one iteration.
    """
    problem = make_problem(is_visualizing=False)
    for name, planner in [("RRT", rrt_planning), ("RRT-Connect", rrt_connect_planning)]:
        iters, seconds, solved, free = [], [], 0, 0
        for _ in range(trials):
            started = time.perf_counter()
            path, num_iter = planner(problem, max_iterations=MAX_ITERATIONS)
            seconds.append(round(time.perf_counter() - started, 3))
            iters.append(num_iter if path is not None else None)
            if path is not None:
                solved += 1
                free += first_collision(path, problem) is None
        print(f"{name:12s} outer iterations per trial: {iters}")
        print(f"{'':12s} planning seconds per trial (excluding recheck): {seconds}")
        print(
            f"{'':12s} passed finer sampled check at {np.degrees(CHECK_STEP)} deg: "
            f"{free} of the {solved} paths found"
        )


if __name__ == "__main__":
    test1()
    # test2(1000)
    # test3()
    # test4(20)
    keep_meshcat_open()
