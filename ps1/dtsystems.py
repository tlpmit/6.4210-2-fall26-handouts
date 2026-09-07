######################################################################
## Code for nobody but Python to read
######################################################################

import sys
from pathlib import Path

# The shared modules live in utils/, a sibling of this directory, and these
# scripts are run from inside their own pset directory -- so the repository
# root has to go on the path before anything imports from it.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from pydrake.all import (
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    DiscreteValues,
    LeafSystem,
    LogVectorOutput,
    MultibodyPlant,
    Simulator,
    System,
    plot_system_graphviz,
)

from utils.plotting import plt

# ps1's own names for these: the meshcat handling is shared with ps2 and ps3,
# but ps1's files import it from here, where it used to live.
from utils.viz import (  # noqa: F401
    HEADLESS,
    IN_COLAB,
    current_meshcat,
    get_meshcat,
    keep_meshcat_open,
    publish_recording,
    show_meshcat,
    start_recording,
)

_NO_STATE = np.zeros(0)

# Frame rate that simulate() records at, for show_meshcat() to play back.
RECORDING_FPS = 32

# Seconds between the samples series_composition's loggers keep for plot_log().
# PLOT_PERIOD = 0.1


######################################################################
## Code for students to be aware of
######################################################################

class DTVectorSystem(LeafSystem):
    """
    A simple, generic discrete-time system abstraction in Drake: input, state,
    and output are vectors, and the behavior is given by two functions,
    next_state_fun(s, inp) and output_fun(s, inp).
    """

    def __init__(
        self,
        input_d: int | Sequence[int],
        state_d: int,
        output_d: int,
        next_state_fun: Callable | None,
        output_fun: Callable,
        dt: float = 1,
        output_depends_on_input: bool = False,
        input_port_name: str | Sequence[str] = "input_port",
        name: str | None = None,
        verbose: bool = False,
    ):
        LeafSystem.__init__(self)
        # Shows up in Drake's error messages and graphviz output, so the
        # constructor functions below each pass their own.
        if name:
            self.set_name(name)
        self.next_state_fun = next_state_fun
        self.output_fun = output_fun
        self.dt = dt
        self.output_depends_on_input = output_depends_on_input
        self.verbose = verbose
        # input_d is a single dimension or a list of
        # them, one port each.  0 means none.
        self.multi_input = isinstance(input_d, (list, tuple))
        dims = list(input_d) if self.multi_input else ([input_d] if input_d else [])
        names = (
            list(input_port_name)
            if isinstance(input_port_name, (list, tuple))
            else [
                f"{input_port_name}{i}" if len(dims) > 1 else input_port_name
                for i in range(len(dims))
            ]
        )
        self.input_ports = [
            self.DeclareVectorInputPort(n, d) for n, d in zip(names, dims)
        ]
        # The s0 list passed to
        # simulate() only has to account for the systems that actually have state
        if state_d:
            self.DeclareDiscreteState(state_d)
        self.DeclareVectorOutputPort(
            "output_port",
            BasicVector(output_d),
            self.GetOutput,
            prerequisites_of_calc=(
                {self.all_sources_ticket()}
                if output_depends_on_input
                else {self.xd_ticket()}
            ),
        )
        if state_d:
            self.DeclarePeriodicDiscreteUpdateEvent(
                period_sec=dt, offset_sec=0.0, update=self.Update
            )

    def GetInput(self, context: Context) -> np.ndarray | list[np.ndarray] | None:
        inps = [p.Eval(context) for p in self.input_ports]
        if self.multi_input:
            return inps
        return inps[0] if inps else None

    def GetState(self, context: Context) -> np.ndarray:
        ds = context.get_discrete_state()
        return ds.get_vector(0).get_value() if ds.num_groups() else _NO_STATE

    def GetOutput(self, context: Context, output: BasicVector) -> None:
        # Only touch the input port if we declared a dependency on it.
        inp = self.GetInput(context) if self.output_depends_on_input else None
        output.set_value(self.output_fun(self.GetState(context), inp))

    def Update(self, context: Context, state: DiscreteValues) -> None:
        s = self.GetState(context)
        inp = self.GetInput(context)
        s_next = self.next_state_fun(s, inp)
        if self.verbose:
            print(f"Update computing new values: {s}, {inp} ->  {s_next}")
        state.get_mutable_vector().SetFromVector(s_next)


def simulate(
    system: System, s0: Sequence[ArrayLike], T: float, strtr: bool = False
) -> Simulator:
    """
    Simulate the system for T seconds and return the simulator, so callers can
    read the final state.  s0 is a list of initial values, one per discrete
    state group, in the order the systems were added to the builder; pass []
    to leave a group at its default.
    """
    simulator = Simulator(system)
    context_state = simulator.get_mutable_context().get_mutable_discrete_state()
    for i, g0 in enumerate(s0):
        if len(g0) > 0:
            context_state.set_value(i, g0)
    # Pacing to wall-clock only makes sense with a live window to watch.
    if strtr and not IN_COLAB and not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    # Recording costs nothing when nobody is watching live, and it is what
    # gives show_meshcat() an animation rather than a final-frame snapshot.
    # Every frame is baked into the HTML that show_meshcat() emits, so the rate
    # is a size knob: Drake's default 64 is more than the eye needs, and a 20 s
    # iiwa run at 32 halves the frames for no visible difference.
    start_recording(frames_per_second=RECORDING_FPS)
    simulator.AdvanceTo(T)
    publish_recording()
    return simulator


def get_positions(plant: MultibodyPlant, simulator: Simulator) -> np.ndarray:
    """
    The plant's positions out of the context that was actually simulated.
    """
    return plant.GetPositions(plant.GetMyContextFromRoot(simulator.get_context()))


def get_state(system: DTVectorSystem, simulator: Simulator) -> np.ndarray:
    """
    A system's own discrete state out of the context that was actually
    simulated.
    """
    return system.GetState(system.GetMyContextFromRoot(simulator.get_context()))


def _subsystems(system: Diagram) -> Iterator[System]:
    """
    Unnest/flatten the subsystems of a diagram.
    """
    for s in system.GetSystems():
        yield s
        if isinstance(s, Diagram):
            yield from _subsystems(s)


def plot_path(
    diagram: Diagram, simulator: Simulator, name: str, indices: Sequence[int] = (0, 1)
) -> None:
    """
    Plot the path traced in the plane by two of a logged signal's elements,
    and report its length.
    """
    loggers = {s.get_name(): s for s in _subsystems(diagram)}
    log = loggers[name].FindLog(simulator.get_context())
    xy = log.data()[list(indices)]
    length = np.sum(np.linalg.norm(np.diff(xy, axis=1), axis=0))
    print(f"   Path length:             {length:.4f} m")
    plt.scatter(xy[0], xy[1], c=log.sample_times(), cmap="viridis", s=4)
    plt.colorbar(label="time (s)")
    plt.title(f"path length {length:.4f} m")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True)
    if not HEADLESS:
        plt.show()


def plot_log(diagram: Diagram, simulator: Simulator, *names: str) -> None:
    """
    Plot each named logger's signal against time.  series_composition names
    each of its loggers log_<system name>.
    """
    subsystems = {s.get_name(): s for s in _subsystems(diagram)}
    # verify that the names are valid, and print the valid names of subsystems
    for name in names:
        if name not in subsystems:
            raise ValueError(f"Invalid subsystem name: {name}. Valid subsystem names: {list(subsystems.keys())}")
    for name in names:
        log = subsystems[name].FindLog(simulator.get_context())
        for i, row in enumerate(log.data()):
            plt.plot(log.sample_times(), row, marker=".", label=f"{name}[{i}]")
    plt.xlabel("time (s)")
    plt.legend()
    plt.grid(True)
    if not HEADLESS:
        plt.show()


######################################################################
## Code for students to study
######################################################################


def series_composition(a: System, b: System) -> Diagram:
    """
    Series composition of two systems: a's output is wired into b's input.
    a's input (if it has one) and b's output are exported, and each system's
    output gets a logger named log_<system name>.
    Takes in two systems, and returns a Diagram.
    """
    builder = DiagramBuilder()
    asys = builder.AddSystem(a)
    bsys = builder.AddSystem(b)
    builder.Connect(asys.get_output_port(), bsys.get_input_port())
    if asys.num_input_ports():
        builder.ExportInput(asys.get_input_port(), "input_port")
    builder.ExportOutput(bsys.get_output_port(), "output_port")
    for s in (asys, bsys):
        logger = LogVectorOutput(s.get_output_port(), builder)
        logger.set_name(f"log_{s.get_name()}")
    diagram = builder.Build()
    return diagram


def Counter(increment: float, verbose: bool = False) -> DTVectorSystem:
    """
    No input; one state, which is emitted as the output and stepped by
    `increment` on every tick.
    """
    return DTVectorSystem(
        0,
        1,
        1,
        lambda s, inp: s + increment,
        lambda s, inp: s,
        name="Counter",
        verbose=verbose,
    )


def Constant(c: ArrayLike) -> DTVectorSystem:
    """
    No input and no state; always outputs the vector c.
    """
    return DTVectorSystem(
        0, 0, len(c), lambda s, inp: s, lambda s, inp: np.array(c), name="Constant"
    )

def PDController(
    dim: int, set_point: ArrayLike, gain: float = 1, d_gain: float = 1, dt: float = 0.01
) -> DTVectorSystem:
    """
    Proportional-derivative controller.  Estimates velocity by differencing
    successive measurements: the state is the previous measured position.
    `dt` is the controller's own update period and is what the derivative term
    divides by, so it must match the rate you want this controller to run at.
    """
    return DTVectorSystem(
        dim,
        dim,
        dim,
        lambda _s, inp: inp,
        lambda s, inp: gain * (np.asarray(set_point) - inp) - d_gain * (inp - s) / dt,
        dt=dt,
        output_depends_on_input=True,
        name="PDController",
    )


def PDController2(
    dim: int, gain: float = 1, d_gain: float = 1, dt: float = 0.01
) -> DTVectorSystem:
    """
    Two-input-port PD controller: like PDController, but the set point arrives
    on a "target" input port alongside the "actual" measurement.
    """
    return DTVectorSystem(
        [dim, dim],
        dim,
        dim,
        lambda _s, inp: inp[1],
        lambda s, inp: gain * (inp[0] - inp[1]) - d_gain * (inp[1] - s) / dt,
        dt=dt,
        output_depends_on_input=True,
        input_port_name=["target", "actual"],
        name="PDController2",
    )


######################################################################
##  Code for students to write
######################################################################


def feedback_composition(a: System, b: System) -> Diagram:
    """
    Feedback composition of two systems.
    Takes in two systems, and returns a Diagram.
    """
    raise NotImplementedError("your code here")


def Body1D(mass: float = 1, dt: float = 1) -> DTVectorSystem:
    """
    Point mass under a force input.  State is (position, velocity); only the
    position is output.
    Takes in mass and dt, and returns a DTVectorSystem.
    """
    raise NotImplementedError("your code here")


def PController(dim: int, set_point: ArrayLike, gain: float = 1) -> DTVectorSystem:
    """
    Proportional position controller: the input is the current value, and the
    output is gain * (set_point - current value).
    Takes in the dimension, the set point, and the gain, and returns a
    stateless DTVectorSystem.
    """
    raise NotImplementedError("your code here")


def Observer(d_in: int, indices: Sequence[int]) -> DTVectorSystem:
    """
    Adapter from a full plant state to the subset a controller can observe:
    the output is the elements of the input selected by indices.
    Takes in the input dimension and the indices to keep, and returns a
    stateless DTVectorSystem.
    """
    raise NotImplementedError("your code here")


def SimpleTrajectoryFollower(
    waypoints: list[np.ndarray], epsilon: float, dt: float = 0.1
) -> DTVectorSystem:
    """
    Follow a sequence of waypoints: the input is the current position x, and
    the output is a target x (the waypoint currently being chased).  The state
    keeps track of which waypoint that is; once the input is within epsilon of
    it, move on to the next.
    """
    raise NotImplementedError("your code here")


def PController2(dim: int, gain: float = 1) -> DTVectorSystem:
    """
    Like PController, but with two input ports, "target" and "actual"; the
    output is gain * (target - actual).
    """
    raise NotImplementedError("your code here")


######################################################################
### Test rigs
######################################################################

def main0_without_logging(init_val: float = 0) -> None:
    system = Counter(2, verbose=True)
    s0 = [[init_val]]
    simulate(system, s0, T=5)

def main0(init_val: float = 0) -> None:
    """
    Exactly the same as main0_without_logging, but with logging.
    """
    builder = DiagramBuilder()
    counter = builder.AddSystem(Counter(2, verbose=True))
    logger = LogVectorOutput(counter.get_output_port(), builder)
    logger.set_name("log_Counter")
    diagram = builder.Build()
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()

    s0 = [[init_val]]
    simulator = simulate(diagram, s0, T=5)
    plot_log(diagram, simulator, "log_Counter")


if __name__ == "__main__":
    # main0_without_logging(4)
    main0(4)
