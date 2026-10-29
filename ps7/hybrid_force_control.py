"""Planar book-dragging control for PS7.

Adapted from Russ Tedrake's Manipulation, force/exercises/hybrid_force_position.ipynb.
See THIRD_PARTY_NOTICES.txt for the upstream BSD license.
The supplied wrapper, TorqueController, maps the end-effector wrench from
compute_ctrl to joint torques, and the station adds gravity compensation.
The simulation runs when build_and_simulate is called. hybrid_force_demo.py
calls it, and also prints the wrench for the verification inputs.
"""

# Code for nobody but Python to read
from typing import Callable

import numpy as np
from pydrake.all import (
    BasicVector,
    Context,
    DiagramBuilder,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    RollPitchYaw,
    Simulator,
    VectorLog,
    VectorLogSink,
)

from manipulation.station import AppendDirectives, LoadScenario, MakeHardwareStation
from manipulation.utils import FindResource

# Code for students to write
def compute_ctrl(p_pxz_now, v_pxz_now, x_des, f_des):
    """Return the planar end-effector wrench [tau_y, f_x, f_z].

    Inputs:
        p_pxz_now: shape (3,), [pitch (rad), x (m), z (m)].
        v_pxz_now: shape (3,), [pitch rate (rad/s), x velocity (m/s), z velocity (m/s)].
        x_des: desired horizontal position (m); desired pitch is zero.
        f_des: positive downward force magnitude on the book (N).
    Output:
        NumPy array (3,), [pitch torque (N m), x force (N), z force (N)].

    Use kp_pitch=10, kd_pitch=3, kp_x=50, kd_x=15. PD feedback regulates
    pitch to zero and x to x_des; the desired pitch rate and horizontal
    velocity are zero. There is no position feedback in z: command only the
    force f_des pressing down on the book. World z points up, so that force
    has a negative z component. The station supplies gravity compensation;
    return only this additional wrench.
    """
    raise NotImplementedError("your code here")


# Code for students to study
class TorqueController(LeafSystem):
    """Wrapper system that commands pure torques to the planar iiwa.
    @param plant MultibodyPlant of the simulated plant.
    @param ctrl_fun control law, such as compute_ctrl, returning the
        end-effector wrench that this system maps to joint torques.
    @param vx Velocity of the x target after t = 2 s (m/s). build_and_simulate
        passes -0.125 by default, which drags the book toward the robot.
    """

    def __init__(self, plant: MultibodyPlant, ctrl_fun: Callable, vx: float) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._ctrl_fun = ctrl_fun
        self._vx = vx
        self._joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]

        self.DeclareVectorInputPort("iiwa_position_measured", 3)
        self.DeclareVectorInputPort("iiwa_velocity_measured", 3)

        # Adding this input port would allow closed-loop force control in z.
        # self.DeclareVectorInputPort("iiwa_torque_external", 3)

        self.DeclareVectorOutputPort(
            "iiwa_position_command", 3, self.CalcPositionOutput
        )
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 3, self.CalcTorqueOutput)
        # The gripper body's [pitch, x, z], logged so that the notebook and
        # hybrid_force_demo.py can check the finger-tip height.
        self.DeclareVectorOutputPort("wsg_position", 3, self.CalcWsgPositionOutput)

    def CalcPositionOutput(self, context: Context, output: BasicVector) -> None:
        """Set q_d = q_now, so that the position-control torques from the
        station's InverseDynamicsController are zero and the iiwa runs in pure
        torque mode. This works only in simulation.
        """
        q_now = self.get_input_port(0).Eval(context)
        output.SetFromVector(q_now)

    def CalcTorqueOutput(self, context: Context, output: BasicVector) -> None:
        # Hard-coded position and force profiles. Can be connected from Trajectory class.
        if context.get_time() < 2.0:
            px_des = 0.65
        else:
            px_des = 0.65 + self._vx * (context.get_time() - 2.0)

        f_des = 10.0  # downward force magnitude (N), passed to compute_ctrl as f_des

        # Read inputs
        q_now = self.get_input_port(0).Eval(context)
        v_now = self.get_input_port(1).Eval(context)
        # tau_now = self.get_input_port(2).Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

        # 1. Convert joint space quantities to Cartesian quantities.
        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

        rpy_now = RollPitchYaw(X_now.rotation()).vector()
        p_xyz_now = X_now.translation()

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )

        # Only select relevant terms. We end up with J_G of shape (3,3).
        # Rows correspond to (pitch, x, z).
        # Columns correspond to (q0, q1, q2).
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]
        v_pxz_now = J_G.dot(v_now)

        p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

        # 2. Apply ctrl_fun
        F_pxz = self._ctrl_fun(p_pxz_now, v_pxz_now, px_des, f_des)

        # 3. Convert back to joint coordinates
        tau_cmd = J_G.T.dot(F_pxz)
        output.SetFromVector(tau_cmd)

    def CalcWsgPositionOutput(self, context: Context, output: BasicVector) -> None:
        """
        Compute Forward kinematics. Needed to log the position trajectory for grading.  TODO(russt): Could use MultibodyPlant's body_poses output port for this.
        """
        q_now = self.get_input_port(0).Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

        rpy_now = RollPitchYaw(X_now.rotation()).vector()
        p_xyz_now = X_now.translation()
        p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

        output.SetFromVector(p_pxz_now)


def build_and_simulate(
    ctrl_fun: Callable, velocity: float = -0.125, duration: float = 6.5,
    *, meshcat=None
) -> tuple[VectorLog, MultibodyPlant, Context]:
    """Simulate book dragging, recording to Meshcat if one is given.

    The horizontal target is 0.65 + velocity * max(t - 2, 0) meters,
    with a constant 10 N downward force command. Return the gripper-pose
    log [pitch, x, z], plant, and final plant context for inspection.
    """
    if meshcat is not None:
        meshcat.Set2dRenderMode(xmin=-0.25, xmax=1.5, ymin=-0.1, ymax=1.3)
    builder = DiagramBuilder()

    scenario = LoadScenario(
        filename=FindResource("models/planar_manipulation_station.scenario.yaml")
    )
    book_directive = """
directives:
- add_model:
    name: book
    file: package://manipulation/book.sdf
    default_free_body_pose:
        book:
            translation: [0.65, 0, 0]
"""
    scenario = AppendDirectives(scenario, data=book_directive)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")

    controller = builder.AddSystem(TorqueController(plant, ctrl_fun, velocity))

    logger = builder.AddSystem(VectorLogSink(3))

    builder.Connect(
        controller.get_output_port(0), station.GetInputPort("iiwa.position")
    )
    builder.Connect(
        controller.get_output_port(1),
        station.GetInputPort("iiwa.torque"),
    )
    builder.Connect(controller.get_output_port(2), logger.get_input_port(0))

    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.get_input_port(0),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.velocity_estimated"),
        controller.get_input_port(1),
    )

    diagram = builder.Build()

    # Initialize default positions for plant.
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    plant.SetPositions(
        plant_context,
        plant.GetModelInstanceByName("iiwa"),
        np.array([np.pi / 4, -np.pi / 3, np.pi / 3]),
    )

    station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
    station.GetInputPort("wsg.position").FixValue(station_context, [0.02])

    if meshcat is not None:
        meshcat.StartRecording(set_visualizations_while_recording=False)
        simulator.AdvanceTo(duration)
        meshcat.PublishRecording()
    else:
        simulator.AdvanceTo(duration)

    # Return these so that we can check the pose of each object.
    return logger.FindLog(simulator.get_context()), plant, plant_context
