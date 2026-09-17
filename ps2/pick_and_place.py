import sys
from pathlib import Path
from typing import NamedTuple

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from manipulation.station import (
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
    MakeMultibodyPlant,
    Scenario,
)
from pydrake.all import (
    AddFrameTriadIllustration,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    InverseKinematics,
    JacobianWrtVariable,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    TrajectorySource,
)

from utils.dtsystems import DTSystem
from utils.viz import HEADLESS, get_meshcat, keep_meshcat_open

######################################################################
## A scene to pick from: an iiwa, a table, and three blocks
######################################################################

# Where to put the blocks and what color to paint them, as
# (name, x, y, yaw_degrees, rgb).  Each one rests on the table in front of the
# arm and inside its reach.  The yaws are multiples of 90 degrees, so every
# block face stays aligned with the world axes.
BLOCK_LAYOUT = [
    ("red", 0.60, -0.25, 0, (0.85, 0.25, 0.20)),
    ("blue", 0.70, 0.00, 90, (0.20, 0.45, 0.80)),
    ("yellow", 0.60, 0.25, 0, (0.95, 0.75, 0.15)),
]

# Blocks are 150 x 50 x 50 mm.  The long axis is deliberately wider than the
# gripper's 110 mm maximum opening, so a block cannot be grasped across it: the
# gripper has to be turned to close on the 50 mm short axis instead, which is
# what makes each block's yaw matter when you plan the grasp.
BLOCK_SIZE = (0.15, 0.05, 0.05)
BLOCK_MASS = 0.1

# The table the arm stands on.  It is welded at z = -height so that its top
# surface lands exactly at z = 0, which is the height everything else assumes.
TABLE_SIZE = (2.0, 2.0, 0.1)
TABLE_COLOR = (0.55, 0.50, 0.45)
TABLE_MASS = 20.0

# Both the table and the blocks are boxes written out by write_box_sdf, rather
# than models checked into assets/.
BOX_BODY = "base_link"
ASSET_DIR = Path(__file__).parent.resolve() / "assets"

# The arm and the gripper welded to its last link.  The table and the blocks
# are added on top of this by make_scenario.
ROBOT = """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.4]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.6]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.2]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""


def write_box_sdf(
    name: str,
    size: tuple[float, float, float],
    rgb: tuple[float, float, float],
    mass: float,
) -> Path:
    """Write an SDF for a colored box, and return the file it went to.

    The body frame is at the center of the box's *bottom* face, so setting that
    frame's height to the height of a surface rests the box on that surface.

    Contact is compliant hydroelastic.  Under plain point contact two boxes
    touch only at their corners, and a block left sitting on the table buzzes
    and slowly creeps instead of holding the pose we placed it at.
    """
    sx, sy, sz = size
    r, g, b = rgb
    # Inertia of a uniform box, about its own center.
    ixx = mass * (sy**2 + sz**2) / 12
    iyy = mass * (sx**2 + sz**2) / 12
    izz = mass * (sx**2 + sy**2) / 12
    # The box, its mass and its inertia all sit half a height above the frame.
    center = f"0 0 {sz / 2} 0 0 0"
    box = f"<box><size>{sx} {sy} {sz}</size></box>"

    path = ASSET_DIR / f"{name}.sdf"
    path.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <link name="{BOX_BODY}">
      <inertial>
        <pose>{center}</pose>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <pose>{center}</pose>
        <geometry>{box}</geometry>
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>1.0e6</drake:hydroelastic_modulus>
        </drake:proximity_properties>
      </collision>
      <visual name="visual">
        <pose>{center}</pose>
        <geometry>{box}</geometry>
        <material><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
""")
    return path


def table_directive() -> str:
    """The directive that welds the table down with its top surface at z = 0."""
    sdf = write_box_sdf("table", TABLE_SIZE, TABLE_COLOR, TABLE_MASS)
    return f"""directives:
- add_model:
    name: table
    file: file://{sdf}
- add_weld:
    parent: world
    child: table::{BOX_BODY}
    X_PC:
        translation: [0, 0, {-TABLE_SIZE[2]}]
"""


def block_directive(name: str, x: float, y: float, yaw: float, rgb) -> str:
    """The directive that puts one block on the table at (x, y, yaw)."""
    sdf = write_box_sdf(name, BLOCK_SIZE, rgb, BLOCK_MASS)
    return f"""directives:
- add_model:
    name: {name}
    file: file://{sdf}
    default_free_body_pose:
        {BOX_BODY}:
            translation: [{x}, {y}, 0]
            rotation: !Rpy {{ deg: [0, 0, {yaw}] }}
"""


def add_all_triads(station):
    def add_triad(body_name: str, model_name: str, length: float):
        model = plant.GetModelInstanceByName(model_name)
        AddFrameTriadIllustration(
            scene_graph=scene_graph,
            body=plant.GetBodyByName(body_name, model),
            length=length,
        )

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    add_triad("body", "wsg", 0.15)
    for name, *_ in BLOCK_LAYOUT:
        add_triad(BOX_BODY, name, 0.1)


def make_scenario() -> Scenario:
    """The robot, the table it stands on, and one block per BLOCK_LAYOUT entry."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    scenario = LoadScenario(data=ROBOT)
    scenario = AppendDirectives(scenario, data=table_directive())
    for block in BLOCK_LAYOUT:
        scenario = AppendDirectives(scenario, data=block_directive(*block))
    return scenario


# How far apart to hold the fingers, just under the 110 mm they can manage.
FINGER_OPEN = 0.107

# How much narrower than the block to ask for when gripping it, so that the
# fingers press into it rather than merely touching.  The command is what sets
# the grip force, and this much of one is worth a few times the block's weight
# -- enough to carry it, and not so much that it squirts out sideways.
FINGER_SQUEEZE = 0.02
FINGER_CLOSED = BLOCK_SIZE[1] - FINGER_SQUEEZE


class Waypoint(NamedTuple):
    """A place for the hand to be, and how wide to hold the fingers there.

    Opening and closing the gripper is a waypoint like any other: same pose as
    the one before it, different fingers.  Most waypoints are travel, and the
    fingers stay where the plan last put them, so the opening has a default.
    """

    X_WG: RigidTransform
    opening: float = FINGER_OPEN


def add_constant_torques(builder, plant, station):
    # Hold the command at the configuration the scenario starts in
    iiwa = plant.GetModelInstanceByName("iiwa")
    hold_arm = builder.AddSystem(ConstantVectorSource(plant.GetDefaultPositions(iiwa)))
    builder.Connect(hold_arm.get_output_port(), station.GetInputPort("iiwa.position"))


def start_the_fingers_open(plant):
    """Have the fingers already be apart when the simulation starts."""
    plant.SetDefaultPositions(
        plant.GetModelInstanceByName("wsg"), [-FINGER_OPEN / 2, FINGER_OPEN / 2]
    )


def open_the_fingers(builder, plant, station):
    """Start the fingers open, and keep commanding them open."""
    start_the_fingers_open(plant)
    command = builder.AddSystem(ConstantVectorSource([FINGER_OPEN]))
    builder.Connect(command.get_output_port(), station.GetInputPort("wsg.position"))


def set_robot_conf(q, station):
    """Start the arm at joint angles `q`.

    This sets the plant's *default* positions, which every context built from
    it afterwards inherits -- so it has to happen before add_constant_torques,
    which reads those same defaults back out to use as the holding command.
    """
    plant = station.GetSubsystemByName("plant")
    plant.SetDefaultPositions(plant.GetModelInstanceByName("iiwa"), q)


######################################################################
## Code for students to be aware of
######################################################################


# What the integrator is called in the diagram, so that show_scene can find it
# again to start it off.
JOINT_INTEGRATOR = "JointIntegrator"


def start_the_integrator(system, context):
    """Plant the arm's own joint angles in Drake's Integrator.

    An Integrator starts its state at zero and offers no way to say otherwise
    when the diagram is built, so the one place to set it is the context that
    is about to be run.

    A scene with no controller in it has no integrator, and needs nothing.
    """
    integrator = next(
        (s for s in system.GetSystems() if s.get_name() == JOINT_INTEGRATOR), None
    )
    if integrator is None:
        return
    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetDefaultPositions(plant.GetModelInstanceByName("iiwa")),
    )


def show_scene(system, duration: float = 2.0):
    """Simulate a system in real time, so it can be watched in Meshcat."""
    simulator = Simulator(system)
    start_the_integrator(system, simulator.get_mutable_context())
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)


def inv_kin(X_WG: RigidTransform) -> np.ndarray:
    """Arm joint angles that put the gripper's body frame G at X_WG.

    The IK runs on a plant holding only the arm and the gripper -- the blocks
    are free bodies, and letting the solver move them around would just make
    the problem harder for no reason.
    """
    plant = MakeMultibodyPlant(LoadScenario(data=ROBOT))
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()

    ik = InverseKinematics(plant)
    ik.AddPositionConstraint(
        G, [0, 0, 0], plant.world_frame(), X_WG.translation(), X_WG.translation()
    )
    ik.AddOrientationConstraint(
        G, RotationMatrix(), plant.world_frame(), X_WG.rotation(), 0.0
    )
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), plant.GetDefaultPositions())
    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError(f"no IK solution for gripper pose\n{X_WG}")

    context = plant.CreateDefaultContext()
    plant.SetPositions(context, result.GetSolution(ik.q()))
    return plant.GetPositions(context, plant.GetModelInstanceByName("iiwa"))


######################################################################
## Code for students to study
######################################################################


def build_static_system(q=None):
    """The scene, with the arm parked at `q` (or at the scenario's own pose)."""
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(make_scenario(), meshcat=get_meshcat())
    )
    plant = station.GetSubsystemByName("plant")
    # For visualizing everybody's coordinate frame
    add_all_triads(station)
    open_the_fingers(builder, plant, station)
    if q is not None:
        set_robot_conf(q, station)
    # To keep the robot from falling down initially
    add_constant_torques(builder, plant, station)
    return builder.Build()


def block_pose(name: str) -> RigidTransform:
    """Where make_scenario puts the named block, as a pose in the world."""
    for block_name, x, y, yaw, _ in BLOCK_LAYOUT:
        if block_name == name:
            R_WO = RotationMatrix.MakeZRotation(np.radians(yaw))
            return RigidTransform(R_WO, [x, y, 0])
    raise ValueError(f"there is no block named {name!r}")


def build_system_tracking_waypoints(waypoints, speed=0.05):
    """Walk the gripper through a list of Waypoints, in order.

    Returns the system and how long its plan runs for.
    """
    return compose_jacobian_control(waypoints, speed)


def arm_jacobian_columns(plant) -> slice:
    """The columns of the plant's Jacobian that belong to the arm.

    The plant holds the blocks as well as the robot, so a gripper Jacobian
    comes back 6 x 27; only these seven columns are joints we can command.
    """
    return slice(
        plant.GetJointByName("iiwa_joint_1").velocity_start(),
        plant.GetJointByName("iiwa_joint_7").velocity_start() + 1,
    )


GRIPPER_RADIUS = 0.1


def pose_distance(error: np.ndarray) -> float:
    """How big a gripper motion `error` is, as one number, in meters"""
    scaled = np.concatenate([GRIPPER_RADIUS * error[:3], error[3:]])
    return float(np.linalg.norm(scaled))


def pose_delta(X_WG: RigidTransform, X_WG_target: RigidTransform) -> np.ndarray:
    """The gripper motion, with the rotation in axis-angle space, that would move from
    X_WG to X_WG_target."""
    R_error = X_WG_target.rotation() @ X_WG.rotation().inverse()
    angle_axis = R_error.ToAngleAxis()
    return np.concatenate(
        [
            angle_axis.angle() * angle_axis.axis(),
            X_WG_target.translation() - X_WG.translation(),
        ]
    )


# How far the middle of the finger pads sits from the gripper's body frame G,
# along the axis the fingers reach out on.  Measured off the model: the pads
# are 20 mm long and centered 108 mm out.
GRIPPER_REACH = 0.108


def trajectory_stops(waypoints, X_WG_start) -> list[Waypoint]:
    """The waypoints a trajectory is actually built from.

    Two more than the plan asks for.  One at `X_WG_start`, because the
    controller below is a *feedforward* one: it integrates a velocity and never
    looks at where the gripper has got to, so a trajectory that does not begin
    where the gripper begins is wrong by that difference for ever after.  And
    one at the back, a repeat of the last, so that the trajectory ends standing
    still.
    """
    waypoints = list(waypoints)
    return [Waypoint(X_WG_start)] + waypoints + [waypoints[-1]]


# How long every leg of a plan is given, whatever it has to do in it.
KEYFRAME_SECONDS = 4.0


def waypoint_times(stops, speed: float) -> list[float]:
    """When to be at each of `stops`, starting from zero."""
    return [KEYFRAME_SECONDS * i for i in range(len(stops))]


######################################################################
## Code for students to write
######################################################################
##
## Three groups, one for each of the tests at the bottom of the file that
## exercises them.


## ---- test2: a pose to hold the gripper at -------------------------


def make_overhead_box_grasp() -> RigidTransform:
    """Where to hold the gripper, in a block's frame, to grasp it from above.

    In the gripper's own frame G the fingers close along +/-x and reach out
    along +y, meeting at the middle of the pads GRIPPER_REACH away.  So to take
    a block from above, G's y axis has to point straight down, and G's x axis
    has to lie across the block's *short* axis -- the long axis is wider than
    the gripper can open.
    """
    # The columns of a rotation matrix are the axes of the frame it describes,
    # written in the frame it maps into.  So we can just name them.
    Gx_O = np.array([0.0, 1.0, 0.0])  # close across the block's short (y) axis
    Gy_O = np.array([0.0, 0.0, -1.0])  # reach down onto the block
    R_OG = RotationMatrix(np.column_stack([Gx_O, Gy_O, np.cross(Gx_O, Gy_O)]))

    # Put the middle of the pads at the center of the block.  The block's frame
    # is at the center of its bottom face, so its center is half a height up,
    # and G itself sits GRIPPER_REACH back along the direction it reaches.
    p_OG = np.array([0, 0, BLOCK_SIZE[2] / 2]) - GRIPPER_REACH * Gy_O
    return RigidTransform(R_OG, p_OG)


def target_conf_to_pick(block_name) -> np.ndarray:
    """Arm joint angles that put the open gripper around the red block."""
    X_OG = make_overhead_box_grasp()
    X_WO = block_pose(block_name)
    X_WG = X_WO @ X_OG
    return inv_kin(X_WG)


## ---- test3: a timed trajectory, and following it ------------------


## note for the TAs: nothing calls what follows.  It is one answer to "the
## timing above is not very good -- do better", which is worth having written
## down somewhere even though the students should arrive at it themselves.
##
## Giving every leg the same two seconds means the speed of the gripper is
## whatever the geometry happens to make it.  Over the sixteen-waypoint plan
## the legs run from 0 m to 0.54 m long, so the hand is asked to crawl on some
## and to move at 0.27 m/s on others, and five of the seventeen legs are the
## arm standing still for two seconds while the fingers, which take about a
## third of a second, close.  Nothing in the plan says how fast to go, and
## nothing checks that what has been asked for is possible: at 2 s a keyframe
## the stack still gets built, at 1 s it still does, and at 0.5 s the arm
## throws the last block across the table.
##
## The rule below fixes the speed instead of the duration.  A leg takes as
## long as the slower of its two jobs, which happen at the same time: carrying
## the hand there at `speed`, and moving the fingers.


# How long to allow for the fingers to travel, whenever a leg opens or closes
# them.  They move on their own time, not the arm's, and take only a fraction
# of a second, so this is mostly margin.
FINGER_SETTLE = 1.0

# No leg is shorter than this, so that a repeated waypoint -- which is how a
# plan is made to stand still -- still gets a segment of its own.
MINIMUM_LEG = 1.0


def waypoint_times_at_a_fixed_speed(stops, speed: float) -> list[float]:
    """When to be at each of `stops`, if the gripper is to travel at `speed`."""
    times = [0.0]
    for a, b in zip(stops, stops[1:]):
        travel = pose_distance(pose_delta(a.X_WG, b.X_WG)) / speed
        fingers = FINGER_SETTLE if a.opening != b.opening else 0.0
        times.append(times[-1] + max(travel, fingers, MINIMUM_LEG))
    return times


def timed_trajectories(waypoints, X_WG_start, speed: float):
    """Turn a plan into the trajectories the controller executes.

    Returns (V_G, wsg): how fast the gripper should be moving and how wide the
    fingers should be, each as a function of time.  Where the gripper should
    *be* is worked out on the way and then thrown away -- a controller that
    integrates velocity has no use for it.

    The poses are interpolated -- a first-order hold on the translation and a
    slerp on the rotation, which is what PiecewisePose.MakeLinear is -- so the
    gripper travels in a straight line from one waypoint to the next.
    Differentiating that *pose* trajectory is what gives V_G: a 6-vector
    spatial velocity, angular then translational, which is what the Jacobian
    expects.  Differentiating the translation alone would give three of the
    six.

    When to be where is waypoint_times' business, and it is worth a look: the
    trajectory is only as good as the times handed to it.

    The opening is a zero-order hold instead.  There is nothing to interpolate:
    the fingers have a controller of their own, and what we owe them is a width
    to go to, held from the moment the plan asks for it.
    """
    stops = trajectory_stops(waypoints, X_WG_start)
    times = waypoint_times(stops, speed)

    traj_X_G = PiecewisePose.MakeLinear(times, [stop.X_WG for stop in stops])
    traj_V_G = traj_X_G.MakeDerivative()
    traj_wsg = PiecewisePolynomial.ZeroOrderHold(
        times, np.array([[stop.opening for stop in stops]])
    )
    return traj_V_G, traj_wsg


def JointVelocity(plant):
    """Which joint velocities produce a wanted gripper velocity.

    Input ports: V_WG, iiwa.position
    Output: iiwa.velocity
    """
    plant_context = plant.CreateDefaultContext()
    iiwa = plant.GetModelInstanceByName("iiwa")
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()
    W = plant.world_frame()
    arm_columns = arm_jacobian_columns(plant)

    def solve(_state, inputs):
        # Compute the gripper jacobian
        # The jacobian is 6 x N, with N being the number of DOFs.
        # We only want the 6 x 7 submatrix corresponding to the IIWA --
        # `arm_columns` is that slice.

        # compute `v` by mapping the gripper velocity to the joint space
        V_WG, q = inputs
        plant.SetPositions(plant_context, iiwa, q)
        J_G = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, G, [0, 0, 0], W, W
        )

        # note from lpk to TAs: I would probably give them all but the line below
        return np.linalg.pinv(J_G[:, arm_columns]) @ V_WG

    return DTSystem(
        [6, 7],
        0,
        7,
        None,
        solve,
        output_depends_on_input=True,
        input_port_name=["V_WG", "iiwa.position"],
        output_port_name="iiwa.velocity",
        name="JointVelocity",
    )


def compose_jacobian_control(waypoints, speed):
    """Wire the trajectories for `waypoints` up to the arm and the hand.
    Returns the diagram and how long its plan runs for.  Those two belong
    together: the trajectory is built in here, and it is the only thing that
    knows when it ends.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(make_scenario(), meshcat=get_meshcat())
    )
    plant = station.GetSubsystemByName("plant")
    add_all_triads(station)
    start_the_fingers_open(plant)

    X_WG_start = plant.EvalBodyPoseInWorld(
        plant.CreateDefaultContext(),
        plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")),
    )
    traj_V_G, traj_wsg = timed_trajectories(waypoints, X_WG_start, speed)
    gripper_velocity = builder.AddSystem(TrajectorySource(traj_V_G))
    gripper_velocity.set_name("GripperVelocity")
    finger_command = builder.AddSystem(TrajectorySource(traj_wsg))
    finger_command.set_name("FingerCommand")
    joint_velocity = builder.AddSystem(JointVelocity(plant))
    command = builder.AddSystem(Integrator(7))
    command.set_name(JOINT_INTEGRATOR)

    builder.Connect(
        gripper_velocity.get_output_port(), joint_velocity.GetInputPort("V_WG")
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        joint_velocity.GetInputPort("iiwa.position"),
    )
    builder.Connect(joint_velocity.get_output_port(), command.get_input_port())
    builder.Connect(command.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        finger_command.get_output_port(), station.GetInputPort("wsg.position")
    )
    return builder.Build(), traj_V_G.end_time()


## ---- test4: walking a whole plan ----------------------------------


def X_AB_for_A_exactly_on_B() -> RigidTransform:
    """Where B goes to rest squarely on top of A, keeping its resting faces."""
    return RigidTransform(p=[0, 0, BLOCK_SIZE[2]])


def X_AB_for_A_standing_upright_on_B() -> RigidTransform:
    """Where B goes to stand on end, centered on top of A."""
    length, _, height = BLOCK_SIZE
    return RigidTransform(
        RotationMatrix.MakeYRotation(-np.pi / 2), [height / 2, 0, height + length / 2]
    )


def pregrasp_X(offset=0.10) -> RigidTransform:
    """A transform that will put the hand offset above a given pose"""
    return RigidTransform(p=[0, 0, offset])


def pick_and_prepick(X_WO):
    pick = X_WO @ make_overhead_box_grasp()
    prepick = pregrasp_X() @ pick
    return pick, prepick


def plan(a, b, c):
    """Waypoints to stack b squarely on a, then stand c upright on b.
    Each waypoint includes a gripper pose and a finger opening.
    """
    X_WA_init = block_pose(a)
    X_WB_init = block_pose(b)
    X_WC_init = block_pose(c)
    X_WG_pickB, X_WG_prepickB = pick_and_prepick(X_WB_init)
    X_WG_pickC, X_WG_prepickC = pick_and_prepick(X_WC_init)
    X_WB_place = X_WA_init @ X_AB_for_A_exactly_on_B()
    X_WC_place = X_WB_place @ X_AB_for_A_standing_upright_on_B()
    X_WG_placeB, X_WG_preplaceB = pick_and_prepick(X_WB_place)
    X_WG_placeC, X_WG_preplaceC = pick_and_prepick(X_WC_place)

    def pick_up(pregrasp, grasp):
        """Reach in with the fingers open, close them, and lift what is held."""
        return [
            Waypoint(pregrasp),
            Waypoint(grasp),
            Waypoint(grasp, FINGER_CLOSED),
            Waypoint(pregrasp, FINGER_CLOSED),
        ]

    def put_down(preplace, place):
        """Carry it in still gripped, let go, and back away empty-handed."""
        return [
            Waypoint(preplace, FINGER_CLOSED),
            Waypoint(place, FINGER_CLOSED),
            Waypoint(place),
            Waypoint(preplace),
        ]

    return (
        pick_up(X_WG_prepickB, X_WG_pickB)
        + put_down(X_WG_preplaceB, X_WG_placeB)
        + pick_up(X_WG_prepickC, X_WG_pickC)
        + put_down(X_WG_preplaceC, X_WG_placeC)
    )


######################################################################
## Testing
######################################################################


def test1():
    show_scene(build_static_system())


def test2(block_name="red"):
    show_scene(build_static_system(target_conf_to_pick(block_name)))


def test3(block_name="red", speed=0.05):
    """Drive the gripper onto a block, and watch it get there."""
    X_WG_target = block_pose(block_name) @ make_overhead_box_grasp()
    system, duration = build_system_tracking_waypoints([Waypoint(X_WG_target)], speed)
    show_scene(system, duration + 2.0)


def test4(a="red", b="blue", c="yellow", speed=0.05):
    """Execute a whole plan."""
    system, duration = build_system_tracking_waypoints(plan(a, b, c), speed)
    show_scene(system, duration + 2.0)


if __name__ == "__main__":
    test1()
    # test2("blue")
    # test3("blue")
    # test4()
    keep_meshcat_open()
