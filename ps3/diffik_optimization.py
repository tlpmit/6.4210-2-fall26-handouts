"""Differential inverse kinematics written as an optimization problem."""

import sys
from collections.abc import Callable
from pathlib import Path

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    Box,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    MathematicalProgram,
    MultibodyPlant,
    Rgba,
    RigidTransform,
    Simulator,
    SnoptSolver,
    Sphere,
    ge,
    le,
)

from utils.dtsystems import DTSystem
from utils.viz import (
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    sim_duration,
    start_recording,
)

######################################################################
## Code for students to be aware of
######################################################################


# The arm, the hand, and a floor to see them against.  The notebook this came
# from used `clutter.dmd.yaml`, which is this plus the two bins of the
# pick-and-place scenes; nothing here picks anything up, the arm model carries
# no collision geometry, and the hand passes a good 0.25 m above where the bins
# would be, so they were scenery suggesting a task that is not happening.
SCENARIO = """
directives:
- add_directives:
    file: package://manipulation/iiwa_and_wsg.dmd.yaml
- add_model:
    name: floor
    file: package://manipulation/floor.sdf
- add_weld:
    parent: world
    child: floor::box
    X_PC:
        translation: [0, 0, -0.5]
model_drivers:
    iiwa: !IiwaDriver
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""


def build_and_simulate(
    diffik_fun: Callable, V_d: np.ndarray, duration: float
) -> Simulator:
    """
    This test-rig inputs a constant desired hand velocity. It goes into the
    controller which emits joint
    velocities, then into an Integrator turns them into joint angles, and those are the
    position command to the station.  The station's measured position and
    velocity are observed by the controller.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(LoadScenario(data=SCENARIO), meshcat=get_meshcat())
    )
    plant = station.GetSubsystemByName("plant")

    controller = builder.AddSystem(DifferentialIKSystem(plant, diffik_fun))
    integrator = builder.AddSystem(Integrator(7))
    desired_vel = builder.AddSystem(ConstantVectorSource(V_d))

    builder.Connect(desired_vel.get_output_port(), controller.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"), controller.get_input_port(1)
    )
    builder.Connect(
        station.GetOutputPort("iiwa.velocity_estimated"), controller.get_input_port(2)
    )
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))

    diagram = builder.Build()
    diagram.set_name("diagram")

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    station.GetInputPort("iiwa.torque").FixValue(station_context, np.zeros((7, 1)))
    station.GetInputPort("wsg.position").FixValue(station_context, [0.1])

    plant_context = plant.GetMyContextFromRoot(context)

    # Same reason as in ps2: an Integrator starts its state at zero and offers
    # no way to say otherwise when the diagram is built, so the arm's own joint
    # angles have to be planted in the context that is about to be run.
    integrator.set_integral_value(
        integrator.GetMyMutableContextFromRoot(context),
        plant.GetPositions(plant_context, plant.GetModelInstanceByName("iiwa")),
    )

    # The hand's pose comes from the plant's own state --- the scenario's
    # default joint positions --- which is also what was just copied into the
    # integrator.  The copy goes plant -> integrator, so this reads the same
    # X_WG wherever it sits relative to the block above.
    show_target_ball(
        plant.CalcRelativeTransform(
            plant_context, plant.world_frame(), plant.GetBodyByName("body").body_frame()
        ),
        V_d,
        duration,
    )

    start_recording()
    simulator.AdvanceTo(duration)
    publish_recording()
    return simulator


def show_end_effector_box() -> None:
    """Draw the box the hand is meant to stay inside, so the wall is visible."""
    meshcat = get_meshcat()
    if meshcat is None:
        return
    meshcat.SetObject("/end_effector_box", Box(0.6, 2.0, 1.0), Rgba(0.1, 0.5, 0.1, 0.2))
    meshcat.SetTransform("/end_effector_box", RigidTransform([0.0, 0.0, 0.5]))


def hide_end_effector_box() -> None:
    meshcat = get_meshcat()
    if meshcat is not None:
        meshcat.Delete("/end_effector_box")


# How far the middle of the finger pads sits from the gripper's body frame G,
# along the axis the fingers reach out on.  Measured off the model in ps2,
# where the same number places a grasp: the pads are centered 108 mm out on +y.
GRIPPER_REACH = 0.108

TARGET_BALL_RADIUS = 0.03


def show_target_ball(
    X_WG_start: RigidTransform, V_d: np.ndarray, duration: float
) -> None:
    """Draw a ball where the fingers are being asked to end up.

    V_d is a constant command, so the target is simply where tracking it
    perfectly for `duration` seconds would leave the hand.  Nothing reaches
    it: 15 s at 0.1 m/s is 1.5 m, which puts the ball about 1.8 m from the
    base against this arm's 0.8 m reach (the scene welds an iiwa7 -- an
    R800 -- to the world origin).  The ball marks the command, not an
    attainable goal.

    The ball is a Meshcat object rather than a body in the plant, so there is
    nothing for the gripper to collide with or push around -- it marks the
    goal without changing the problem.

    Where to put it: the controller aims G's *origin*, which sits inside the
    gripper body, so a ball there would be swallowed by the hand rather than
    held between the fingers.  GRIPPER_REACH out along G's +y is the middle of
    the pads.  The angular part of V_d is zero in all four problems, so the
    hand is meant to arrive in the orientation it started in, and that is the
    orientation the offset is applied in.
    """
    meshcat = get_meshcat()
    if meshcat is None:
        return
    X_WG_target = RigidTransform(
        X_WG_start.rotation(), X_WG_start.translation() + duration * V_d[3:6]
    )
    meshcat.SetObject(
        "/target_ball", Sphere(TARGET_BALL_RADIUS), Rgba(0.9, 0.1, 0.1, 0.8)
    )
    meshcat.SetTransform(
        "/target_ball",
        RigidTransform(X_WG_target @ np.array([0.0, GRIPPER_REACH, 0.0])),
    )


def hide_target_ball() -> None:
    meshcat = get_meshcat()
    if meshcat is not None:
        meshcat.Delete("/target_ball")


######################################################################
## Code for students to study
######################################################################


def DifferentialIKSystem(plant: MultibodyPlant, diffik_fun: Callable) -> DTSystem:
    """

    Input ports: spatial_velocity, iiwa.position_measured, iiwa.velocity_measured
    Output: iiwa_velocity_command
    """
    plant_context = (
        plant.CreateDefaultContext()
    )  # Plant used for hypothetical reasoning
    iiwa = plant.GetModelInstanceByName("iiwa")
    G = plant.GetBodyByName("body").body_frame()
    W = plant.world_frame()

    def solve(_state, inputs):
        V_G_desired, q_now, v_now = inputs

        plant.SetPositions(plant_context, iiwa, q_now)
        J_G = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kQDot, G, [0, 0, 0], W, W
        )
        # Just get the joint columns from a more complex plant
        J_G = J_G[:, 0:7]

        p_now = plant.CalcRelativeTransform(plant_context, W, G).translation()

        return diffik_fun(J_G, V_G_desired, q_now, v_now, p_now)

    return DTSystem(
        [6, 7, 7],
        0,
        7,
        None,
        solve,
        output_depends_on_input=True,
        input_port_name=[
            "spatial_velocity",
            "iiwa.position_measured",
            "iiwa.velocity_measured",
        ],
        output_port_name="iiwa_velocity_command",
        name="DifferentialIKSystem",
    )


def DiffIK_Zero(
    J_G: np.ndarray,
    V_G_desired: np.ndarray,
    q_now: np.ndarray,
    v_now: np.ndarray,
    p_now: np.ndarray,
) -> np.ndarray:
    """Command nothing"""
    return np.zeros(7)


def DiffIKPseudoInverse(
    J_G: np.ndarray,
    V_G_desired: np.ndarray,
    q_now: np.ndarray,
    v_now: np.ndarray,
    p_now: np.ndarray,
) -> np.ndarray:
    """v = J^+ V_d"""
    return np.linalg.pinv(J_G).dot(V_G_desired)


# First-order prediction horizon in seconds, shared with the wall checkpoint.
# This is not the controller's sampling period.
WALL_H = 4e-3


######################################################################
## Code for students to write
######################################################################


def DiffIKQP(
    J_G: np.ndarray,
    V_G_desired: np.ndarray,
    q_now: np.ndarray,
    v_now: np.ndarray,
    p_now: np.ndarray,
) -> np.ndarray:
    """Joint velocities that best track V_G_desired subject to |v| <= v_max."""
    prog = MathematicalProgram()
    v = prog.NewContinuousVariables(7, "v")
    v_max = 3.0  # do not modify

    raise NotImplementedError("your code here")


def DiffIKQP_Wall(
    J_G: np.ndarray,
    V_G_desired: np.ndarray,
    q_now: np.ndarray,
    v_now: np.ndarray,
    p_now: np.ndarray,
) -> np.ndarray:
    """Like DiffIKQP, but also keep the hand's predicted next position, one
    step of h seconds ahead, inside the box [lower_bound, upper_bound]."""
    prog = MathematicalProgram()
    v = prog.NewContinuousVariables(7, "joint_velocities")
    v_max = 3.0  # do not modify
    h = WALL_H  # do not modify
    lower_bound = np.array([-0.3, -1.0, 0.0])  # do not modify
    upper_bound = np.array([0.3, 1.0, 1.0])  # do not modify

    raise NotImplementedError("your code here")


######################################################################
## Testing
######################################################################


def test1():
    """Nothing should move"""
    build_and_simulate(DiffIK_Zero, np.zeros(6), sim_duration(5.0))


def test2():
    """Drive the hand at 0.1 m/s in +x"""
    V_d = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    build_and_simulate(DiffIKPseudoInverse, V_d, sim_duration(15.0))


def test3():
    """The same command, with joint velocities bounded"""
    V_d = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    build_and_simulate(DiffIKQP, V_d, sim_duration(15.0))


def test4():
    """A command that would move the hand straight through the wall"""
    show_end_effector_box()
    V_d = np.array([0.0, 0.0, 0.0, 0.1, 0.05, 0.0])
    build_and_simulate(DiffIKQP_Wall, V_d, sim_duration(15.0))


J_G_GIVEN = np.array(
    [
        [0.0, 9.02435102e-01, 2.54531849e-02, -6.52856354e-01,
         7.26280879e-01, 6.83118291e-01, 2.12647126e-04],
        [0.0, 4.30825819e-01, -5.33158564e-02, -7.57108057e-01,
         -6.34577338e-01, 7.09128073e-01, -2.39276758e-01],
        [1.0, 1.29333062e-17, 9.98253252e-01, -2.37901464e-02,
         2.64249288e-01, -1.74604626e-01, -9.70951383e-01],
        [3.15610018e-01, 1.45973504e-01, 2.96994113e-01, 3.84093679e-02,
         1.15833004e-01, -1.24517465e-01, -6.93889390e-18],
        [3.00354015e-01, -3.05765366e-01, 2.91205259e-01, -4.60616310e-02,
         1.20243355e-01, 1.13081997e-01, -8.91756287e-18],
        [0.0, -4.14217824e-01, 7.98035139e-03, 4.11844127e-01,
         -2.96068464e-02, -2.78946736e-02, 2.19635643e-18],
    ]
)  # fmt: skip

V_G_DESIRED_GIVEN = np.array([0.0, 0.0, 0.0, 0.1, 0.05, 0.0])

Q_NOW_GIVEN = np.array(
    [-1.12538865, 0.05911441, 0.41443944, -1.24917317,
     -0.20671306, 1.67556706, 0.81804794]
)  # fmt: skip

V_NOW_GIVEN = np.array(
    [0.1628158, 0.09226121, 0.12800451, 0.0873851, -0.09184795, 0.05699749, 0.27648769]
)

P_NOW_GIVEN = np.array([0.30035402, -0.31561002, 0.67932255])

V_G_DESIRED_FAST = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 0.0])

V_MAX_GIVEN = 3.0  # the limit inside DiffIKQP, repeated here to count against


def test5():
    """The velocity-limited QP checkpoint; the wall controller is not called."""
    fast = (J_G_GIVEN, V_G_DESIRED_FAST, Q_NOW_GIVEN, V_NOW_GIVEN, P_NOW_GIVEN)

    v = DiffIKQP(*fast)
    at_limit = int(np.sum(np.abs(np.abs(v) - V_MAX_GIVEN) < 1e-4))
    print(f"DiffIKQP      v[:2] = {np.round(v[:2], 4)}")
    print(f"              {at_limit} of 7 joints at the {V_MAX_GIVEN} rad/s limit")
    print(f"              max |v| = {np.abs(v).max():.4f}")


def test6():
    """The wall checkpoint: unique hand motion, not nonunique joint velocities."""
    args = (J_G_GIVEN, V_G_DESIRED_GIVEN, Q_NOW_GIVEN, V_NOW_GIVEN, P_NOW_GIVEN)
    v = DiffIKQP_Wall(*args)
    spatial_velocity = J_G_GIVEN @ v
    p_next = P_NOW_GIVEN + WALL_H * spatial_velocity[3:6]
    print(f"DiffIKQP_Wall hand x velocity = {spatial_velocity[3]:.4f} m/s")
    print(f"              predicted next x = {p_next[0]:.4f} m")


if __name__ == "__main__":
    test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    keep_meshcat_open()
