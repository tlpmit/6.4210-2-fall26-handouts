"""Open a cupboard door using optimization-based inverse kinematics."""

import sys
from collections.abc import Callable
from pathlib import Path

# Add the repository root so utils can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from manipulation.utils import FindResource
from pydrake.all import (
    ConstantVectorSource,
    Diagram,
    DiagramBuilder,
    Frame,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    Solve,
    Trajectory,
    TrajectorySource,
)
from pydrake.multibody import inverse_kinematics

from utils.viz import (
    HEADLESS,
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    start_recording,
)

SCENARIO_FILE = "models/cupboard.scenario.yaml"


def draw_home_scene() -> None:
    """Put the robot and cupboard into Meshcat at the configuration the motion
    starts from, so the nominal trajectory's triads have a scene around them."""
    builder = DiagramBuilder()
    station, _ = add_station(builder)

    iiwa_position = builder.AddSystem(ConstantVectorSource(np.zeros(7)))
    builder.Connect(
        iiwa_position.get_output_port(), station.GetInputPort("iiwa.position")
    )
    # The fingers start open.
    wsg_position = builder.AddSystem(ConstantVectorSource([0.06]))
    builder.Connect(
        wsg_position.get_output_port(), station.GetInputPort("wsg.position")
    )

    simulator = Simulator(builder.Build())
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)


def draw_nominal_trajectory(
    InterpolatePose: Callable[[float], RigidTransform], t_lst: np.ndarray
) -> None:
    meshcat = get_meshcat()
    if meshcat is None:
        return
    for t in t_lst:
        AddMeshcatTriad(meshcat, path=str(t), X_PT=InterpolatePose(t), opacity=0.2)


######################################################################
## Code for students to be aware of
######################################################################

## Eleven-second motion:
##
##   0 <= t < 5    reach: interpolate from wherever the gripper starts to the
##                 pose that grips the handle, fingers open
##   5 <= t < 6    hold still and close the fingers
##   6 <= t <= 11  follow the arc the handle sweeps as the door opens
##
## The reach is linear; the opening follows an arc about the hinge.

# Handle geometry in the closed door's reference frame R.
p_WR = np.array([0.7477, -0.1445, 0.4148])

p_Rhandle = np.array([-0.033, 0.1245, 0])
p_Whandle = p_WR + p_Rhandle

p_Rhinge = np.array([0.008, -0.1395, 0])
p_Whinge = p_WR + p_Rhinge

p_Rhinge_handle = p_Rhandle - p_Rhinge
r_Rhinge_handle = np.linalg.norm(p_Rhinge_handle)
theta_Rhinge_handle = np.arctan2(p_Rhinge_handle[1], p_Rhinge_handle[0])

# The handle's angle about the hinge at the end of the motion, measured in the
# door frame.  It starts at theta_Rhinge_handle (about 99 degrees), so the door
# itself sweeps angle_end - theta_Rhinge_handle: about 81 degrees for np.pi.
angle_end = np.pi

# The hand is facing the handle, and 0.1 m out along the
# handle frame's y axis.
X_handleG = RigidTransform(
    RollPitchYaw(0, np.pi, np.pi).ToRotationMatrix(), np.array([0.0, 0.1, 0.0])
)

# The hand pose that grasps the handle with the door still shut: the handle
# frame at its rest angle, with that grip applied.
X_WG_grasp = RigidTransform(
    RollPitchYaw(0, 0, theta_Rhinge_handle).ToRotationMatrix(), p_Whandle
).multiply(X_handleG)

# The phase boundaries, in seconds: reach until T_GRASP, close the fingers
# until T_CLOSED, then swing the door until TOTAL_TIME.
T_GRASP = 5.0
T_CLOSED = 6.0
TOTAL_TIME = 11.0

# Finger trajectory: 0.02 m open, 0.0 closed on the handle.  The (1, 4) shape
# is one row per gripper position input.
GRIPPER_TIMES = np.array([0.0, T_GRASP, T_CLOSED, TOTAL_TIME])
GRIPPER_KNOTS = np.array([0.02, 0.02, 0.0, 0.0]).reshape(1, 4)

# More keyframes give a smoother trajectory but take longer to solve.
N_KEYFRAMES = 30


def get_gripper_frame(plant: MultibodyPlant) -> Frame:
    """Frame G: the WSG's palm, which its SDF calls "body"."""
    return plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg"))


def add_station(builder: DiagramBuilder) -> tuple[Diagram, MultibodyPlant]:
    """The cupboard scenario, with a triad drawn on the gripper body."""
    scenario = LoadScenario(filename=FindResource(SCENARIO_FILE))
    station = builder.AddSystem(MakeHardwareStation(scenario, get_meshcat()))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(get_gripper_frame(plant), scene_graph)
    return station, plant


def initial_gripper_pose() -> RigidTransform:
    """X_WG with the arm at zero, which is where the reach starts from.

    Read off the plant's default context
    """
    _, plant = add_station(DiagramBuilder())
    context = plant.CreateDefaultContext()
    return get_gripper_frame(plant).CalcPoseInWorld(context)


def InterpolatePoseOpen(fraction: float) -> RigidTransform:
    """The gripper pose part way through opening the door.

    `fraction` runs 0 (shut, gripping the handle) to 1 (open).  It is a
    fraction of the opening phase, not seconds.
    """
    theta = theta_Rhinge_handle + (angle_end - theta_Rhinge_handle) * fraction
    p_Whandle_t = (
        r_Rhinge_handle * np.array([np.cos(theta), np.sin(theta), 0]) + p_Whinge
    )
    R_Whandle = RollPitchYaw(0, 0, theta).ToRotationMatrix()
    X_Whandle = RigidTransform(R_Whandle, p_Whandle_t)
    return X_Whandle.multiply(X_handleG)


def make_pose_interpolator(
    initial_pose: RigidTransform,
) -> Callable[[float], RigidTransform]:
    """X_WG(t) over the whole eleven seconds."""
    entry = PiecewisePose.MakeLinear([0.0, T_GRASP], [initial_pose, X_WG_grasp])

    def InterpolatePose(t: float) -> RigidTransform:
        if t < T_GRASP:
            return entry.GetPose(t)
        elif t < T_CLOSED:
            # Hold at the grasp while the fingers close.
            return X_WG_grasp
        else:
            # Circular arc for opening the door
            opening = (t - T_CLOSED) / (TOTAL_TIME - T_CLOSED)
            return InterpolatePoseOpen(opening)

    return InterpolatePose


def CreateIiwaControllerPlant() -> tuple[MultibodyPlant, list[int]]:
    """Create the robot-only plant used for IK."""
    scenario = LoadScenario(filename=FindResource(SCENARIO_FILE))
    plant_robot = MakeMultibodyPlant(
        scenario=scenario, model_instance_names=["iiwa", "wsg"]
    )
    link_frame_indices = [
        plant_robot.GetFrameByName(f"iiwa_link_{i}").index() for i in range(8)
    ]
    return plant_robot, link_frame_indices


def BuildAndSimulateTrajectory(
    q_traj: Trajectory, g_traj: Trajectory, duration: float = 0.01
) -> tuple[Simulator, MultibodyPlant]:
    """Play a joint trajectory and a finger trajectory through the station."""
    builder = DiagramBuilder()
    station, plant = add_station(builder)

    q_traj_system = builder.AddSystem(TrajectorySource(q_traj))
    g_traj_system = builder.AddSystem(TrajectorySource(g_traj))

    builder.Connect(
        q_traj_system.get_output_port(), station.GetInputPort("iiwa.position")
    )
    builder.Connect(
        g_traj_system.get_output_port(), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    start_recording(set_visualizations_while_recording=False)
    simulator.AdvanceTo(duration)
    publish_recording()

    return simulator, plant


## IK tolerances: 1 mm per axis and 0.01 pi radians.
POSITION_TOLERANCE = 0.001
ORIENTATION_TOLERANCE = 0.01 * np.pi

# Nominal configuration; the final two entries are finger joints.
Q_NOMINAL = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0, 0.0, 0.0])

######################################################################
## Code for students to write
######################################################################


def add_pose_constraint(
    ik: inverse_kinematics.InverseKinematics,
    plant: MultibodyPlant,
    frame: Frame,
    X_WF: RigidTransform,
    pos_tol: float,
    theta_bound: float,
) -> None:
    """Ask an IK program for `frame` to sit at the pose X_WF in the world.

    Two constraints on the one frame: the position constraint holds its origin
    within +/- pos_tol along each world axis about the target. The orientation
    constraint bounds the relative rotation angle by theta_bound.

    The two IK problems later in this pset import this function, so keep it
    about a frame and a pose.  Nothing here should know about the door.
    """
    raise NotImplementedError("your code here")


def create_q_knots(pose_lst: list[RigidTransform]) -> list[np.ndarray]:
    """Solve IK for each gripper keyframe.

    Args:
        pose_lst: Gripper poses X_WG.

    Returns:
        Joint configurations corresponding to the poses.
    """
    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def keyframes() -> tuple[
    np.ndarray, list[RigidTransform], Callable[[float], RigidTransform]
]:
    """The nominal poses to solve IK at, and the times they happen."""
    initial_pose = initial_gripper_pose()
    InterpolatePose = make_pose_interpolator(initial_pose)
    t_lst = np.linspace(0, TOTAL_TIME, N_KEYFRAMES)
    return t_lst, [InterpolatePose(t) for t in t_lst], InterpolatePose


def test1():
    """Draw the nominal end-effector trajectory."""
    t_lst, pose_lst, InterpolatePose = keyframes()
    draw_home_scene()
    draw_nominal_trajectory(InterpolatePose, t_lst)
    print(f"{len(pose_lst)} keyframes over {TOTAL_TIME} s")


def test2():
    """Solve the IK and open the door."""
    t_lst, pose_lst, InterpolatePose = keyframes()
    draw_nominal_trajectory(InterpolatePose, t_lst)

    q_knots = np.array(create_q_knots(pose_lst))
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_knots[:, 0:7].T)
    g_traj = PiecewisePolynomial.FirstOrderHold(GRIPPER_TIMES, GRIPPER_KNOTS)

    simulator, station_plant = BuildAndSimulateTrajectory(q_traj, g_traj, TOTAL_TIME)

    door_angle = station_plant.GetPositions(
        station_plant.GetMyContextFromRoot(simulator.get_context()),
        station_plant.GetModelInstanceByName("cupboard"),
    )
    print(f"door opened {np.rad2deg(door_angle[1]):.1f} degrees")


def check_final_configuration(q_iiwa: np.ndarray) -> bool:
    """Check seven submitted joint angles, allowing for four-decimal rounding.

    This checks feasibility, not global optimality of the nonconvex IK cost.
    It is supplied support code, not another function for students to write.
    """
    q_iiwa = np.asarray(q_iiwa, dtype=float)
    if q_iiwa.shape != (7,) or not np.all(np.isfinite(q_iiwa)):
        return False

    plant, _ = CreateIiwaControllerPlant()
    context = plant.CreateDefaultContext()
    q = Q_NOMINAL.copy()
    q[:7] = q_iiwa
    plant.SetPositions(context, q)
    actual = plant.CalcRelativeTransform(
        context, plant.world_frame(), get_gripper_frame(plant)
    )
    _, poses, _ = keyframes()
    desired = poses[-1]

    # Rounding each angle to four decimals can change it by at most 0.00005 rad.
    joint_slack = 0.00005
    position_slack = 0.0002  # m, to accommodate the resulting pose roundoff
    orientation_slack = 0.0004  # rad
    within_limits = np.all(
        q >= plant.GetPositionLowerLimits() - joint_slack
    ) and np.all(q <= plant.GetPositionUpperLimits() + joint_slack)
    position_error = np.max(np.abs(actual.translation() - desired.translation()))
    rotation_error = (
        (desired.rotation().inverse() @ actual.rotation()).ToAngleAxis().angle()
    )
    return bool(
        within_limits
        and position_error <= POSITION_TOLERANCE + position_slack
        and rotation_error <= ORIENTATION_TOLERANCE + orientation_slack
    )


def test3():
    """The answer to hand in: the arm configuration at the last keyframe."""
    _, pose_lst, _ = keyframes()
    q_knots = np.array(create_q_knots(pose_lst))
    answer = np.round(q_knots[-1][:7], 4)
    print(f"q at the final pose = {answer}")
    print(
        f"rounded answer satisfies final pose and joint limits: {check_final_configuration(answer)}"
    )


if __name__ == "__main__":
    test1()
    # test2()
    # test3()
    keep_meshcat_open()
