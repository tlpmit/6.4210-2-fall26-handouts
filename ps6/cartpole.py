"""Small, headless-by-default Drake cart-pole for the PS6 feasibility pilot.

Use cartpole.ipynb for the homework experiment.
The stock Drake model has a 10 kg cart and a 1 kg point mass on a 0.5 m pole.
Angles exposed here are relative to upright (the SDF uses downward zero).
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


@dataclass(frozen=True)
class Config:
    dt: float = 0.02
    integration_step: float = 0.01
    horizon: int = 500
    force: float = 100.0
    x_limit: float = 2.4
    theta_limit: float = float(np.deg2rad(12))


class CartPole(gym.Env):
    """Binary force control, +1 per transition until failure or the time limit.

    Observation order: x, theta (relative to upright), xdot, thetadot.
    reset(options={"state": ...}) supports controlled diagnostic starts.
    The pilot maximizes capped episode length: its horizon is part of the
    objective, so the training code uses zero continuation at BOTH endings.
    Gym's truncation flag still identifies the time limit separately.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None, *, meshcat=None):
        super().__init__()
        self.config = config or Config()
        cfg = self.config
        if min(cfg.dt, cfg.integration_step, cfg.horizon, cfg.force,
               cfg.x_limit, cfg.theta_limit) <= 0:
            raise ValueError("All configuration values must be positive")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.meshcat = None
        if render_mode:
            from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, StartMeshcat
            builder = DiagramBuilder()
            self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.)
        else:
            self.plant = MultibodyPlant(0.)
        Parser(self.plant).AddModels(
            url="package://drake/examples/multibody/cart_pole/cart_pole.sdf")
        self.plant.Finalize()
        if render_mode:
            self.meshcat = meshcat if meshcat is not None else StartMeshcat()
            params = MeshcatVisualizerParams()
            params.publish_period = cfg.dt
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat, params)
            self.system = builder.Build()
        else:
            self.system = self.plant
        self.simulator = Simulator(self.system)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.input_port = self.plant.get_actuation_input_port()
        self.input_port.FixValue(self.plant_context, [0.])
        integrator = self.simulator.get_mutable_integrator()
        integrator.set_maximum_step_size(cfg.integration_step)
        integrator.set_fixed_step_mode(True)
        self.simulator.set_target_realtime_rate(0.)
        self.ended = True
        self.steps = 0

    def state(self):
        state = self.plant.GetPositionsAndVelocities(self.plant_context).copy()
        state[1] -= np.pi
        return state

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.asarray((options or {}).get(
            "state", self.np_random.uniform(-.05, .05, 4)), dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be four finite numbers")
        state = state.copy()
        state[1] += np.pi
        self.context.SetTime(0.)
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        self.input_port.FixValue(self.plant_context, [0.])
        self.simulator.Initialize()
        self.steps = 0
        self.ended = False
        return self.state().astype(np.float32), {}

    def step(self, action):
        if self.ended:
            raise RuntimeError("Call reset() before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 or 1")
        self.input_port.FixValue(self.plant_context,
                                 [(2 * int(action) - 1) * self.config.force])
        self.steps += 1
        self.simulator.AdvanceTo(self.steps * self.config.dt)
        state = self.state()
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Nonfinite simulator state")
        terminated = bool(abs(state[0]) > self.config.x_limit or
                          abs(state[1]) > self.config.theta_limit)
        truncated = self.steps >= self.config.horizon and not terminated
        self.ended = terminated or truncated
        reason = ('pole_angle' if abs(state[1]) > self.config.theta_limit else
                  'cart_position' if abs(state[0]) > self.config.x_limit else
                  'horizon' if truncated else None)
        return state.astype(np.float32), 1., terminated, truncated, {
            "success": truncated, "elapsed": self.steps * self.config.dt,
            "end_reason": reason}

    def render(self):
        if self.meshcat is not None:
            self.system.ForcedPublish(self.context)


def feedback_action(state):
    """Bang-bang version of stabilizing linear state feedback; diagnostic only."""
    return int(np.dot([1., -20., 2., -5.], state) > 0.)
