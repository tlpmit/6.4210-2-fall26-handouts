"""Inverse kinematics for an arm on a mobile base."""

import sys
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from manipulation import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    DiscreteContactApproximation,
    InverseKinematics,
    Joint,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Solve,
    WeldJoint,
    eq,
)

# The pose constraints are the ones written for the door-opening problem.
from door_opening import add_pose_constraint
from utils.viz import HEADLESS, get_meshcat, keep_meshcat_open

######################################################################
## Support code
######################################################################


def draw_configuration(diagram: Diagram, plant: MultibodyPlant, q: np.ndarray) -> None:
    """Put the robot at q in a fresh context and publish it to Meshcat."""
    context = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(context), q)
    diagram.ForcedPublish(context)


######################################################################
## Code for students to be aware of
######################################################################

SCENE_URL = "package://manipulation/pr2_shelves.dmd.yaml"
TIME_STEP = 0.01

# The PR2's first three positions are the base's (x, y, theta); the joints on
# top of it follow.
BASE = slice(0, 3)

# Joints that do not affect the left gripper pose.  They are welded shut so
# that IK has fewer variables to search over.
FROZEN_JOINTS = [
    "head_pan_joint",
    "head_tilt_joint",
    "r_gripper_l_finger_joint",
    "r_gripper_r_finger_joint",
    "r_gripper_l_finger_tip_joint",
    "r_gripper_r_finger_tip_joint",
    "l_gripper_l_finger_joint",
    "l_gripper_r_finger_joint",
    "l_gripper_l_finger_tip_joint",
    "l_gripper_r_finger_tip_joint",
]

# The frame the goal pose applies to.
GRIPPER_FRAME = "l_gripper_palm_link"


def freeze_joint(plant: MultibodyPlant, joint: Joint) -> None:
    """Replace a joint with a weld, and remove whatever used to drive it."""
    actuators = [plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()]
    for actuator in actuators:
        if actuator.joint() == joint:
            plant.RemoveJointActuator(actuator)

    weld = WeldJoint(
        joint.name(), joint.frame_on_parent(), joint.frame_on_child(), RigidTransform()
    )
    plant.RemoveJoint(joint)
    plant.AddJoint(weld)


def build_scene() -> tuple[Diagram, MultibodyPlant, SceneGraph]:
    """Load the PR2 and the shelves, and build the diagram around them."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(SCENE_URL)

    # The finger joints are coupled by mimic constraints, which have to go
    # before the joints they refer to can be replaced.
    for id in plant.GetConstraintIds():
        plant.RemoveConstraint(id)
    for name in FROZEN_JOINTS:
        freeze_joint(plant, plant.GetJointByName(name))
    plant.Finalize()

    meshcat = get_meshcat()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph.get_query_output_port(),
            meshcat,
            MeshcatVisualizerParams(delete_on_initialization_event=False),
        )

    return builder.Build(), plant, scene_graph


######################################################################
## Code for students to study
######################################################################

# Target pose inside the shelves, with the palm facing down.
GOAL_POSE = RigidTransform(
    RotationMatrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    np.array([-0.83, 0.18, 1.4]),
)

# IK tolerances: 1 mm per axis and one degree.
POSITION_TOLERANCE = 0.001
ORIENTATION_TOLERANCE = np.radians(1.0)

# Candidate collision pairs must be separated by at least MIN_DISTANCE.
# Drake filters excluded pairs. The influence threshold is the lower bound
# plus INFLUENCE_DISTANCE_OFFSET, hence 0.11 m with the values below.
MIN_DISTANCE = 0.01
INFLUENCE_DISTANCE_OFFSET = 0.1

# Initial guesses are drawn uniformly within the joint limits; a joint with no
# limit (the base's x, y and theta) is drawn from +/- this instead.
UNBOUNDED_RANGE = np.pi

# How many random initial guesses to try before giving up.
MAX_TRIES = 10

# The random-restart seed.  It makes a run repeat on your own machine; leave
# it alone.  It does not make one repeat on someone else's: the solver's
# arithmetic differs between platforms, so the same guesses can succeed here
# and fail there.
IK_SEED = 16


def solve_ik(
    X_WG: RigidTransform,
    max_tries: int = MAX_TRIES,
    fix_base: bool = False,
    base_pose: np.ndarray | None = None,
    seed: int = IK_SEED,
    return_first: bool = False,
) -> list[np.ndarray] | np.ndarray | None:
    """Configurations putting the gripper frame at X_WG, cheapest first.

    If fix_base is true, they must have base pose `base_pose`.  With
    return_first, stop at the first one and return it alone, or None.
    """
    if base_pose is None:
        base_pose = np.zeros(3)
    np.random.seed(seed)

    diagram, plant, _ = build_scene()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # The minimum-distance constraint needs a context to query geometry in.
    ik = InverseKinematics(plant, plant_context)
    q_variables = ik.q()
    prog = ik.prog()

    # Stay near a nominal configuration: the base at base_pose, every joint at zero.
    q_nominal = np.zeros(len(q_variables))
    q_nominal[BASE] = base_pose
    prog.AddQuadraticErrorCost(np.eye(len(q_variables)), q_nominal, q_variables)

    add_ik_constraints(plant, ik, X_WG, fix_base=fix_base, base_pose=base_pose)

    # The problem is nonconvex, so the solver can get stuck: restart it from
    # random configurations and keep every attempt that satisfies every
    # constraint.  Which attempts those are is not the same from one machine
    # to the next -- the solver's arithmetic differs -- so count them rather
    # than reading anything into which attempt number succeeded.
    solutions = []
    for attempt in range(1, max_tries + 1):
        prog.SetInitialGuess(q_variables, random_initial_guess(plant))
        result = Solve(prog)

        if not HEADLESS and not return_first:
            # Show where the solver ended up, whether or not it succeeded, and
            # wait: the failures are the point, and they flash past otherwise.
            # A return_first caller only wants the verdict, so it skips this.
            draw_configuration(diagram, plant, result.GetSolution(q_variables))
            verdict = "solved" if result.is_success() else "failed"
            input(f"  attempt {attempt} of {max_tries}: {verdict} -- Enter to go on ")

        if result.is_success():
            solutions.append(
                (result.get_optimal_cost(), result.GetSolution(q_variables))
            )
            if return_first:
                print(f"Succeeded on attempt {attempt} of {max_tries}.")
                return solutions[0][1]

    if return_first:
        print(f"No solution in {max_tries} attempts.")
        return None

    solutions.sort(key=lambda pair: pair[0])
    print(f"{len(solutions)} of {max_tries} attempts succeeded.")
    if solutions:
        print("costs: " + ", ".join(f"{cost:.2f}" for cost, _ in solutions))
    return [q for _, q in solutions]


######################################################################
## Code for students to write
######################################################################


def add_ik_constraints(
    plant: MultibodyPlant,
    ik: InverseKinematics,
    X_WG: RigidTransform,
    fix_base: bool = False,
    base_pose: np.ndarray | None = None,
) -> None:
    """Constrain the gripper frame to X_WG and keep the robot out of collision.

    If fix_base is true, also pin the base's (x, y, theta) to base_pose.
    """
    raise NotImplementedError("your code here")


def random_initial_guess(plant: MultibodyPlant) -> np.ndarray:
    """A configuration drawn uniformly from the robot's joint limits.

    `plant.GetPositionLowerLimits()` and `plant.GetPositionUpperLimits()` each
    return one number per position, in the same order as the configuration
    vector.  A position that has no limit --- the base's x, y and theta --- has
    an infinite entry
    """
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################

# Candidate base poses for the multiple-choice part, as (x, y, theta).
CANDIDATE_BASE_POSES = [
    np.array([-1.23, 0.05, 0.0]),
    np.array([-1.5, -0.2, 0.0]),
    np.array([-1.8, 0.2, 0.0]),
    np.array([-1.4, 0.3, 0.0]),
]

# A base pose from which the reference run found feasible configurations.
TEST1_BASE_POSE = np.array([-1.3, 0.0, 0.0])

# Target positions for the multiple-choice part, all with GOAL_POSE's
# palm-down orientation.  Some are reachable from TEST1_BASE_POSE, some only
# if the base pose may vary. Failure within the restart budget does not prove
# that a target is unreachable.
CANDIDATE_TARGETS = [
    np.array([-0.83, 0.18, 1.40]),
    np.array([-0.83, 0.50, 1.40]),
    np.array([-0.83, 1.30, 1.40]),
    np.array([-0.83, 0.18, 2.30]),
]


def test1() -> None:
    """Run every restart from one fixed base: how many succeed, at what cost."""
    solve_ik(GOAL_POSE, max_tries=20, fix_base=True, base_pose=TEST1_BASE_POSE)


def test2(max_tries: int = 20) -> None:
    """Each target, tried from the fixed base and then with the base free."""
    for position in CANDIDATE_TARGETS:
        X_WG = RigidTransform(GOAL_POSE.rotation(), position)
        fixed = solve_ik(
            X_WG,
            max_tries=max_tries,
            fix_base=True,
            base_pose=TEST1_BASE_POSE,
            return_first=True,
        )
        free = solve_ik(X_WG, max_tries=max_tries, fix_base=False, return_first=True)
        no_solution = f"no solution found in {max_tries} attempts"
        print(
            f"  {position} -> fixed base "
            f"{no_solution if fixed is None else 'solved'}, free base "
            f"{no_solution if free is None else 'solved'}"
        )


def test3(max_tries: int = 20) -> None:
    """Try a few fixed base poses"""
    for base_pose in CANDIDATE_BASE_POSES:
        q = solve_ik(
            GOAL_POSE,
            max_tries=max_tries,
            fix_base=True,
            base_pose=base_pose,
            return_first=True,
        )
        verdict = (
            f"no solution found in {max_tries} attempts" if q is None else "solved"
        )
        print(f"  {base_pose} -> {verdict}")


if __name__ == "__main__":
    # Every test needs add_ik_constraints and random_initial_guess, so there is
    # nothing to run until those are written: uncomment one then.
    # test1()
    # test2()
    # test3()
    keep_meshcat_open()
