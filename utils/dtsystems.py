######################################################################
## Code for nobody but Python to read
######################################################################

"""Discrete-time systems described by two functions, as in ps1 -- but for
values that are not vectors as well.

`ps1/dtsystems.py` has `DTVectorSystem`: rather than writing a `LeafSystem`
subclass, you say how wide the input, the state and the output are, and hand
over `next_state_fun(s, inp)` and `output_fun(s, inp)`.  Everything there is a
vector of floats.

Several of the systems in `pick_and_place.py` want to pass a `RigidTransform`
from one to the next, and a pose is not naturally a vector -- writing it as one
would mean packing and unpacking a quaternion at both ends of every wire.  So
`DTSystem` below generalises the same idea: anywhere `DTVectorSystem` wants a
width, `DTSystem` also accepts `Abstract(prototype)`, and the port, the state or
the output then carries whatever type `prototype` is.

The two functions keep the same signatures, so a system's *description* does not
change when one of its wires stops being a vector.
"""

import numpy as np
from pydrake.all import AbstractValue, BasicVector, LeafSystem

_NO_STATE = np.zeros(0)


class Abstract:
    """Marks a port or a state as holding `prototype`'s type, not a vector.

    The prototype is only ever used as a model -- Drake needs an instance to
    know what type will flow through -- so any default-constructed value of the
    right type will do, e.g. Abstract(RigidTransform()).
    """

    def __init__(self, prototype):
        self.prototype = prototype


def _present(spec) -> bool:
    """Whether a spec asks for anything at all.

    0, None and an empty list all mean "no port" / "no state".  This cannot
    just be `bool(spec)`, because a spec is allowed to be an array of initial
    state values, and numpy refuses to be asked whether an array is truthy.
    """
    if spec is None or isinstance(spec, Abstract):
        return spec is not None
    if isinstance(spec, (int, np.integer)):
        return spec > 0
    return len(spec) > 0


class DTSystem(LeafSystem):
    """A discrete-time system built from a state-update and an output function.

    Each of `input_spec`, `state_spec` and `output_spec` is one of:

      - an int, for that many floats;
      - `Abstract(prototype)`, for a value of the prototype's type;
      - for the state only, a sequence of numbers, which both sizes the state
        and gives its *initial* value -- handy when a system has to start
        somewhere other than zero, and the alternative is reaching into a
        built diagram's context afterwards to plant a starting value;
      - 0 or None, for no input port / no state.

    `input_spec` may also be a list of specs, for one port each, in which case
    `next_state_fun` and `output_fun` receive a list of the inputs.  So may
    `output_spec`, in which case `output_fun` returns a list of the outputs.
    """

    def __init__(
        self,
        input_spec,
        state_spec,
        output_spec,
        next_state_fun,
        output_fun,
        dt=1,
        output_depends_on_input=False,
        input_port_name="input_port",
        output_port_name="output_port",
        name=None,
        verbose=False,
    ):
        LeafSystem.__init__(self)
        # Shows up in Drake's error messages and graphviz output, so the
        # constructor functions that use this each pass their own.
        if name:
            self.set_name(name)
        self.next_state_fun = next_state_fun
        self.output_fun = output_fun
        self.dt = dt
        self.output_depends_on_input = output_depends_on_input
        self.verbose = verbose

        # -- inputs: one spec, or a list of them, one port each.
        self.multi_input = isinstance(input_spec, (list, tuple))
        specs = (
            list(input_spec)
            if self.multi_input
            else ([input_spec] if _present(input_spec) else [])
        )
        names = (
            list(input_port_name)
            if isinstance(input_port_name, (list, tuple))
            else [
                f"{input_port_name}{i}" if len(specs) > 1 else input_port_name
                for i in range(len(specs))
            ]
        )
        self.input_ports = [
            self._declare_input(n, s) for n, s in zip(names, specs)
        ]

        # -- state: discrete if it is a vector, abstract otherwise.
        self.has_state = _present(state_spec)
        self.abstract_state = isinstance(state_spec, Abstract)
        if self.has_state:
            if self.abstract_state:
                self.DeclareAbstractState(AbstractValue.Make(state_spec.prototype))
            elif isinstance(state_spec, (int, np.integer)):
                self.DeclareDiscreteState(int(state_spec))
            else:
                self.DeclareDiscreteState(np.asarray(state_spec, dtype=float))

        # -- outputs: one spec, or a list of them, one port each.  Declaring
        # the wider dependency is always safe; it just recomputes more often
        # than it strictly has to.
        prerequisites = {
            self.all_sources_ticket()
            if output_depends_on_input
            else self.all_state_ticket()
        }
        self.multi_output = isinstance(output_spec, (list, tuple))
        specs = list(output_spec) if self.multi_output else [output_spec]
        names = (
            list(output_port_name)
            if isinstance(output_port_name, (list, tuple))
            else [
                f"{output_port_name}{i}" if len(specs) > 1 else output_port_name
                for i in range(len(specs))
            ]
        )
        for i, (n, spec) in enumerate(zip(names, specs)):
            self._declare_output(n, spec, i, prerequisites)

        if self.has_state:
            # An unrestricted update is the one kind that may touch abstract
            # state; discrete state gets the cheaper discrete update.
            update = self.UpdateAbstract if self.abstract_state else self.Update
            declare = (
                self.DeclarePeriodicUnrestrictedUpdateEvent
                if self.abstract_state
                else self.DeclarePeriodicDiscreteUpdateEvent
            )
            declare(period_sec=dt, offset_sec=0.0, update=update)

    def _declare_input(self, name, spec):
        if isinstance(spec, Abstract):
            return self.DeclareAbstractInputPort(
                name, AbstractValue.Make(spec.prototype)
            )
        return self.DeclareVectorInputPort(name, spec)

    def _declare_output(self, name, spec, i, prerequisites):
        # One calc callback per port, each picking its own value out of what
        # `output_fun` returned -- so a multi-output system's output_fun runs
        # once per port rather than once per step.  The systems here are cheap
        # enough that keeping the description simple is worth more.
        def calc(context, output):
            self.GetOutput(context, output, i)

        if isinstance(spec, Abstract):
            model = spec.prototype
            return self.DeclareAbstractOutputPort(
                name,
                lambda: AbstractValue.Make(model),
                calc,
                prerequisites_of_calc=prerequisites,
            )
        return self.DeclareVectorOutputPort(
            name, BasicVector(spec), calc, prerequisites_of_calc=prerequisites
        )

    def GetInput(self, context):
        inps = [p.Eval(context) for p in self.input_ports]
        if self.multi_input:
            return inps
        return inps[0] if inps else None

    def GetState(self, context):
        if not self.has_state:
            return _NO_STATE
        if self.abstract_state:
            return context.get_abstract_state(0).get_value()
        return context.get_discrete_state().get_vector(0).get_value()

    def GetOutput(self, context, output, port=0):
        # Only touch the input port if we declared a dependency on it.
        inp = self.GetInput(context) if self.output_depends_on_input else None
        out = self.output_fun(self.GetState(context), inp)
        output.set_value(out[port] if self.multi_output else out)

    def _next_state(self, context):
        s = self.GetState(context)
        inp = self.GetInput(context)
        s_next = self.next_state_fun(s, inp)
        if self.verbose:
            print(f"Update computing new values: {s}, {inp} ->  {s_next}")
        return s_next

    def Update(self, context, discrete_state):
        discrete_state.get_mutable_vector().SetFromVector(self._next_state(context))

    def UpdateAbstract(self, context, state):
        state.get_mutable_abstract_state(0).set_value(self._next_state(context))


# ps1 describes everything by width alone, which DTSystem already accepts, so
# the vector-only name is just the general one under its old spelling.
DTVectorSystem = DTSystem
