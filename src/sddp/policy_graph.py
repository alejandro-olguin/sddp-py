# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Policy graphs, nodes, subproblems, and the user-facing modelling API.

Ported from ``src/user_interface.jl`` (Noise, State, Node, PolicyGraph,
parameterize, set_stage_objective, objective states, belief states) and
``src/JuMP.jl`` (the ``SDDP.State`` variable extension).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, TypeVar

from sddp.graph import Graph, LinearGraph, MarkovianGraph
from sddp.solver.model import HiGHS, Constraint, Model, OptimizerFactory, Sense, Variable

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class Noise(Generic[T]):
    """An atom of a discrete random variable: ``term`` with probability ``probability``."""

    term: Any
    probability: float


class State(NamedTuple):
    """A state variable: the incoming (``in_``) and outgoing (``out``) solver variables."""

    in_: Variable
    out: Variable

    @property
    def incoming(self) -> Variable:
        return self.in_

    @property
    def outgoing(self) -> Variable:
        return self.out


class StateValue(NamedTuple):
    """The primal value of a state variable (returned by :func:`sddp.simulate`)."""

    in_: float
    out: float

    @property
    def incoming(self) -> float:
        return self.in_

    @property
    def outgoing(self) -> float:
        return self.out


@dataclass
class ObjectiveState:
    """Storage for objective states (``add_objective_state``)."""

    update: Callable[..., Any]
    initial_value: tuple[float, ...]
    state: tuple[float, ...]
    lower_bound: tuple[float, ...]
    upper_bound: tuple[float, ...]
    mu: tuple[Variable, ...]


@dataclass
class BeliefState:
    """Storage for belief states (partially observable policy graphs)."""

    partition_index: int
    belief: dict[Any, float]
    mu: dict[Any, Variable]
    updater: Callable[..., dict[Any, float]]


class Node(Generic[T]):
    """A node of the policy graph: its subproblem, children, noise, states, and Bellman function."""

    def __init__(self, index: T, subproblem: Subproblem, policy_graph: PolicyGraph):
        self.index: T = index
        self.subproblem: Subproblem = subproblem
        self.policy_graph = policy_graph
        self.children: list[Noise] = []
        self.noise_terms: list[Noise] = []
        self.parameterize_fn: Callable[[Any], Any] = lambda w: None
        self.states: dict[str, State] = {}
        self.stage_objective: Any = 0.0
        self.stage_objective_set: bool = False
        self.bellman_function: Any = None
        self.objective_state: ObjectiveState | None = None
        self.belief_state: BeliefState | None = None
        self.pre_optimize_hook: Callable[..., Any] | None = None
        self.post_optimize_hook: Callable[..., Any] | None = None
        self.has_integrality: bool = False
        self.optimizer: OptimizerFactory | None = None
        self.ext: dict[str, Any] = {}
        self.incoming_state_bounds: dict[str, tuple[float, float, bool]] = {}

    @property
    def model(self) -> Model:
        return self.subproblem.model

    def __repr__(self) -> str:
        return (
            f"Node {self.index}\n  # State variables : {len(self.states)}\n"
            f"  # Children        : {len(self.children)}\n"
            f"  # Noise terms     : {len(self.noise_terms)}\n"
        )


class Subproblem:
    """The object handed to the user's builder function. Wraps a solver :class:`Model`.

    All modelling calls (``add_state``, ``add_variable``, ``add_constraint``,
    ``parameterize``, ``set_stage_objective``, ...) go through this object.
    """

    def __init__(self, model: Model, policy_graph: PolicyGraph):
        self.model = model
        self.policy_graph = policy_graph
        self.node: Node = None  # type: ignore[assignment]  # set by PolicyGraph
        self.ext: dict[str, Any] = {}

    # --------------------------------------------------------------- modelling
    def add_state(
        self,
        name: str,
        *,
        initial_value: float,
        lb: float = -math.inf,
        ub: float = math.inf,
        integer: bool = False,
        binary: bool = False,
    ) -> State:
        """Add a state variable (``@variable(sp, lb <= x <= ub, SDDP.State, initial_value=...)``).

        The incoming copy ``x.in_`` has no bounds (it is fixed by the algorithm); the
        outgoing copy ``x.out`` carries ``lb``/``ub``/integrality.
        """
        if name in self.node.states:
            raise ValueError(f"A state variable named {name!r} already exists.")
        if initial_value is None or (isinstance(initial_value, float) and math.isnan(initial_value)):
            raise ValueError(
                "When creating a state variable, you must set the `initial_value` keyword "
                "to the value of the state variable at the root node."
            )
        in_ = self.model.add_variable(name + "_in")
        out = self.model.add_variable(name + "_out", lb=lb, ub=ub, integer=integer, binary=binary)
        state = State(in_, out)
        self.node.states[name] = state
        self.policy_graph.initial_root_state[name] = float(initial_value)
        self.model.names[name] = state
        return state

    def add_variable(
        self,
        name: str | None = None,
        *,
        lb: float = -math.inf,
        ub: float = math.inf,
        integer: bool = False,
        binary: bool = False,
    ) -> Variable:
        """Add a control variable."""
        return self.model.add_variable(name, lb=lb, ub=ub, integer=integer, binary=binary)

    def add_variables(self, names: Sequence[str], **kwargs: Any) -> list[Variable]:
        return [self.add_variable(n, **kwargs) for n in names]

    def add_constraint(self, con: Any, name: str | None = None) -> Constraint:
        """Add a linear constraint built with operators, e.g. ``x.out == x.in_ + u``."""
        return self.model.add_constraint(con, name=name)

    def add_constraints(self, cons: Sequence[Any]) -> list[Constraint]:
        return [self.add_constraint(c) for c in cons]

    def set_stage_objective(self, stage_objective: Any) -> None:
        """Set the stage objective (``@stageobjective``). Accepts a number, variable, or expression."""
        self.node.stage_objective = stage_objective
        self.node.stage_objective_set = False

    def parameterize(
        self,
        modify_or_realizations: Any,
        realizations: Sequence[Any] | None = None,
        probability: Sequence[float] | None = None,
    ) -> Any:
        """Add stagewise-independent noise.

        Two forms::

            sp.parameterize(modify, realizations, probability=None)

            @sp.parameterize(realizations, probability=None)
            def modify(omega): ...
        """
        if callable(modify_or_realizations):
            if realizations is None:
                raise TypeError("parameterize(modify, realizations, probability) requires realizations")
            self._parameterize(modify_or_realizations, realizations, probability)
            return None
        support = modify_or_realizations
        probs = realizations if probability is None else probability  # positional shift
        if realizations is not None and probability is None:
            probs = realizations

        def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self._parameterize(fn, support, probs)
            return fn

        return decorator

    def _parameterize(
        self, modify: Callable[[Any], Any], realizations: Sequence[Any], probability: Sequence[float] | None
    ) -> None:
        node = self.node
        if node.noise_terms:
            raise ValueError("Duplicate calls to parameterize detected.")
        realizations = list(realizations)
        if probability is None:
            probability = [1.0 / len(realizations)] * len(realizations)
        if len(probability) != len(realizations):
            raise ValueError("realizations and probability must have the same length.")
        for w, p in zip(realizations, probability):
            node.noise_terms.append(Noise(w, float(p)))
        node.parameterize_fn = modify

    def add_objective_state(
        self,
        update: Callable[..., Any],
        *,
        initial_value: float | Sequence[float],
        lipschitz: float | Sequence[float],
        lower_bound: float | Sequence[float] = -math.inf,
        upper_bound: float | Sequence[float] = math.inf,
    ) -> None:
        """Add an objective state (``SDDP.add_objective_state``)."""
        node = self.node
        if node.objective_state is not None:
            raise ValueError("add_objective_state can only be called once.")
        iv = _to_tuple(initial_value)
        N = len(iv)
        lb = _to_tuple(lower_bound, N)
        ub = _to_tuple(upper_bound, N)
        lip = _to_tuple(lipschitz, N)
        mu = tuple(self.model.add_variable(lb=-lip[i], ub=lip[i]) for i in range(N))
        node.objective_state = ObjectiveState(update, iv, iv, lb, ub, mu)

    def objective_state(self) -> Any:
        """Return the current objective state (scalar if 1-D, else a tuple)."""
        os = self.node.objective_state
        if os is None:
            raise ValueError("No objective state defined.")
        return os.state[0] if len(os.state) == 1 else os.state

    # ------------------------------------------------------- modifications
    def fix(self, v: Variable, value: float) -> None:
        self.model.fix(v, value)

    def unfix(self, v: Variable) -> None:
        self.model.unfix(v)

    def set_lower_bound(self, v: Variable, value: float) -> None:
        self.model.set_lower_bound(v, value)

    def set_upper_bound(self, v: Variable, value: float) -> None:
        self.model.set_upper_bound(v, value)

    def set_normalized_rhs(self, c: Constraint, value: float) -> None:
        self.model.set_normalized_rhs(c, value)

    def set_normalized_coefficient(self, c: Constraint, v: Variable, value: float) -> None:
        self.model.set_normalized_coefficient(c, v, value)

    # ---------------------------------------------------------------- query
    def value(self, x: Any) -> Any:
        """Primal value of a variable/expression, or a :class:`StateValue` for a state."""
        if isinstance(x, State):
            return StateValue(self.model.value(x.in_), self.model.value(x.out))
        return self.model.value(x)

    def dual(self, c: Constraint) -> float:
        """Dual of a constraint: d objective / d rhs (see solver docs for the sign contract)."""
        return self.model.dual(c)

    def __getitem__(self, name: str) -> Any:
        return self.model.names[name]

    def __contains__(self, name: str) -> bool:
        return name in self.model.names

    @property
    def objective_sense(self) -> Sense:
        return self.model.objective_sense


def _to_tuple(x: Any, N: int | None = None) -> tuple[float, ...]:
    if isinstance(x, (int, float)):
        return tuple(float(x) for _ in range(N or 1))
    t = tuple(float(v) for v in x)
    if N is not None and len(t) != N:
        raise ValueError(
            f"Invalid dimension in the input to `add_objective_state`. Got: `{x}`, but "
            f"expected it to have length `{N}`."
        )
    return t


@dataclass
class Log:
    iteration: int
    bound: float
    simulation_value: float
    time: float
    pid: int
    total_solves: int
    duality_key: str
    serious_numerical_issue: bool


@dataclass
class TrainingResults:
    status: str
    log: list[Log] = field(default_factory=list)


class PolicyGraph(Generic[T]):
    """A policy graph: the root node, its children, and a :class:`Node` per graph node."""

    def __init__(
        self,
        builder: Callable[[Subproblem, Any], Any],
        graph: Graph,
        *,
        sense: str | Sense = "Min",
        lower_bound: float = -math.inf,
        upper_bound: float = math.inf,
        optimizer: OptimizerFactory | None = None,
        bellman_function: Any = None,
    ):
        from sddp.plugins.bellman_functions import BellmanFunction, initialize_bellman_function

        graph.validate()
        self.objective_sense: Sense = Sense.parse(sense)
        self.root_node: T = graph.root_node
        self.root_children: list[Noise] = []
        self.initial_root_state: dict[str, float] = {}
        self.nodes: dict[T, Node] = {}
        self.belief_partition: list[set] = []
        self.most_recent_training_results: TrainingResults | None = None
        self.ext: dict[str, Any] = {}
        self.optimizer: OptimizerFactory = optimizer if optimizer is not None else HiGHS
        if bellman_function is None:
            if self.objective_sense is Sense.MIN and lower_bound == -math.inf:
                raise ValueError(
                    "You must specify a finite lower bound on the objective value using the "
                    "`lower_bound = value` keyword argument."
                )
            if self.objective_sense is Sense.MAX and upper_bound == math.inf:
                raise ValueError(
                    "You must specify a finite upper bound on the objective value using the "
                    "`upper_bound = value` keyword argument."
                )
            bellman_function = BellmanFunction(lower_bound=lower_bound, upper_bound=upper_bound)
        # Initialize nodes.
        for node_index in graph.nodes:
            if node_index == graph.root_node:
                continue
            model = Model(self.optimizer)
            sp = Subproblem(model, self)
            node: Node = Node(node_index, sp, self)
            sp.node = node
            node.optimizer = self.optimizer
            self.nodes[node_index] = node
            model.set_objective_sense(self.objective_sense)
            builder(sp, node_index)
            if not node.noise_terms:
                node.noise_terms.append(Noise(None, 1.0))
            node.has_integrality = model.has_integrality()
        # Loop back through and add the arcs/children.
        for node_index, children in graph.nodes.items():
            if node_index == graph.root_node:
                continue
            node = self.nodes[node_index]
            for child, probability in children:
                node.children.append(Noise(child, probability))
            node.bellman_function = initialize_bellman_function(bellman_function, self, node)
        # Add root nodes and check the initial point is feasible w.r.t. bounds (SDDP.jl#387).
        for child, probability in graph.nodes[graph.root_node]:
            self.root_children.append(Noise(child, probability))
            for k, v in self.initial_root_state.items():
                x_out = self[child].states[k].out
                m = self[child].model
                if m.has_lower_bound(x_out) and m.lower_bound(x_out) > v:
                    raise ValueError(f"Initial point {v} violates lower bound on state {k}")
                if m.has_upper_bound(x_out) and m.upper_bound(x_out) < v:
                    raise ValueError(f"Initial point {v} violates upper bound on state {k}")
        if graph.belief_partition:
            _initialize_belief_states(self, graph)

    def __getitem__(self, index: T) -> Node:
        return self.nodes[index]

    def __repr__(self) -> str:
        nodes = list(self.nodes)
        try:
            nodes = sorted(nodes)
        except TypeError:
            pass
        if len(nodes) < 10:
            s = ", ".join(str(n) for n in nodes)
        else:
            s = f"{nodes[0]}, ..., {nodes[-1]}"
        return f"A policy graph with {len(self.nodes)} nodes.\n Node indices: {s}\n"

    @property
    def is_minimization(self) -> bool:
        return self.objective_sense is Sense.MIN


def LinearPolicyGraph(
    builder: Callable[[Subproblem, int], Any], *, stages: int, **kwargs: Any
) -> PolicyGraph[int]:
    """A linear policy graph with ``stages`` nodes labelled ``1..stages``."""
    if stages < 1:
        raise ValueError("You must create a LinearPolicyGraph with `stages >= 1`.")
    return PolicyGraph(builder, LinearGraph(stages), **kwargs)


def MarkovianPolicyGraph(
    builder: Callable[[Subproblem, tuple[int, int]], Any],
    *,
    transition_matrices: Sequence[Any],
    **kwargs: Any,
) -> PolicyGraph[tuple[int, int]]:
    """A Markovian policy graph; nodes are ``(stage, markov_state)`` tuples."""
    return PolicyGraph(builder, MarkovianGraph(transition_matrices), **kwargs)


# ---------------------------------------------------------------------------
# Objective-state helpers
# ---------------------------------------------------------------------------
def get_objective_state_component(node: Node) -> list[tuple[Variable, float]]:
    """Terms of ``<y, μ>`` for the objective."""
    os = node.objective_state
    if os is None:
        return []
    return [(mu, y) for y, mu in zip(os.state, os.mu)]


def initialize_objective_state(first_node: Node) -> tuple[tuple[float, ...] | None, int]:
    os = first_node.objective_state
    if os is not None:
        return os.initial_value, len(os.initial_value)
    return None, 0


def update_objective_state(obj_state: ObjectiveState | None, current_state: Any, noise: Any) -> Any:
    if obj_state is None:
        return None
    if len(current_state) == 1:
        obj_state.state = (float(obj_state.update(current_state[0], noise)),)
    else:
        obj_state.state = tuple(float(v) for v in obj_state.update(tuple(current_state), noise))
    return obj_state.state


# ---------------------------------------------------------------------------
# Belief-state helpers
# ---------------------------------------------------------------------------
def build_phi(graph: PolicyGraph) -> dict[tuple[Any, Any], float]:
    phi: dict[tuple[Any, Any], float] = {}
    for i, node in graph.nodes.items():
        for child in node.children:
            phi[(i, child.term)] = child.probability
    for child in graph.root_children:
        phi[(graph.root_node, child.term)] = child.probability
    return phi


def _noise_key(term: Any) -> Any:
    try:
        hash(term)
        return term
    except TypeError:
        return repr(term)


def construct_belief_update(graph: PolicyGraph, partition: list[set]) -> Callable[..., dict[Any, float]]:
    """Bayes update of the belief. See ``construct_belief_update`` in SDDP.jl."""
    phi = build_phi(graph)
    omega: dict[Any, dict[Any, float]] = {}
    for index, node in graph.nodes.items():
        omega[index] = {}
        for noise in node.noise_terms:
            omega[index][_noise_key(noise.term)] = noise.probability

    def belief_updater(
        outgoing_belief: dict[Any, float],
        incoming_belief: dict[Any, float],
        observed_partition: int,
        observed_noise: Any,
    ) -> dict[Any, float]:
        key = _noise_key(observed_noise)
        PY = 0.0
        for node_i, belief in incoming_belief.items():
            probability = 0.0
            for node_j, omega_j in omega.items():
                p_ij = phi.get((node_i, node_j), 0.0)
                p_w = omega_j.get(key, 0.0)
                probability += p_ij * p_w
            PY += belief * probability
        if math.isclose(PY, 0.0, rel_tol=1.4901161193847656e-08, abs_tol=0.0):
            raise ValueError(
                f"Unable to update belief in partition {observed_partition} after observing "
                f"{observed_noise}. The incoming belief is:\n  {incoming_belief}"
            )
        for node_i in list(incoming_belief):
            PX = sum(b * phi.get((node_j, node_i), 0.0) for node_j, b in incoming_belief.items())
            PY_X = 0.0
            if node_i in partition[observed_partition]:
                PY_X += omega[node_i].get(key, 0.0)
            outgoing_belief[node_i] = PY_X * PX / PY
        if len(outgoing_belief) == 2:
            for node_i, belief in list(incoming_belief.items()):
                if belief < 1e-6:
                    incoming_belief[node_i] = 0.0
                elif belief > 1 - 1e-6:
                    incoming_belief[node_i] = 1.0
        return outgoing_belief

    return belief_updater


def get_belief_state_component(node: Node) -> list[tuple[Variable, float]]:
    bs = node.belief_state
    if bs is None:
        return []
    return [(mu, bs.belief[key]) for key, mu in bs.mu.items()]


def _initialize_belief_states(policy_graph: PolicyGraph, graph: Graph) -> None:
    from sddp.plugins.bellman_functions import bellman_term

    partition_sets = [set(p) for p in graph.belief_partition]
    belief_updater = construct_belief_update(policy_graph, partition_sets)
    belief = {k: 0.0 for k in graph.nodes if k != graph.root_node}
    for partition_index, (part, lips) in enumerate(zip(graph.belief_partition, graph.belief_lipschitz)):
        policy_graph.belief_partition.append(set(part))
        for node_index in part:
            node = policy_graph[node_index]
            mu: dict[Any, Variable] = {}
            for node_name, L in zip(part, lips):
                mu[node_name] = node.model.add_variable(lb=-L, ub=L)
            # add_initial_bounds: bound <b, μ> + θ at the corners of the simplex.
            theta = bellman_term(node.bellman_function)
            m = node.model
            lb = m.lower_bound(theta) if m.has_lower_bound(theta) else -math.inf
            ub = m.upper_bound(theta) if m.has_upper_bound(theta) else math.inf
            for variable in mu.values():
                if lb > -math.inf:
                    m.add_constraint(variable + theta >= lb)
                if ub < math.inf:
                    m.add_constraint(variable + theta <= ub)
            node.belief_state = BeliefState(partition_index, dict(belief), mu, belief_updater)
            node.bellman_function.global_theta.belief_states = mu
            for local_theta in node.bellman_function.local_thetas:
                local_theta.belief_states = mu


def _get_incoming_domain(model: PolicyGraph) -> dict[Any, dict[str, tuple[float, float, bool] | None]]:
    """Bounds of the incoming state at each node (used by Lagrangian duality)."""
    from sddp.algorithm import parameterize

    def _bounds(m: Model, x: Variable) -> tuple[float, float, bool]:
        l, u = -math.inf, math.inf
        if m.has_lower_bound(x):
            l = m.lower_bound(x)
        if m.has_upper_bound(x):
            u = m.upper_bound(x)
        if m.is_fixed(x):
            l = u = m.fix_value(x)
        is_int = m.is_integer(x)
        if m.is_binary(x):
            l, u = max(l, 0.0), min(u, 1.0)
            is_int = True
        return l, u, is_int

    outgoing: dict[tuple[Any, str], tuple[float, float, bool] | None] = {
        (k, s): None for k, node in model.nodes.items() for s in node.states
    }
    for k, node in model.nodes.items():
        for noise in node.noise_terms:
            parameterize(node, noise.term)
            for state_name, state in node.states.items():
                domain = outgoing[(k, state_name)]
                l_new, u_new, i_new = _bounds(node.model, state.out)
                if domain is None:
                    outgoing[(k, state_name)] = (l_new, u_new, i_new)
                else:
                    l, u, i = domain
                    outgoing[(k, state_name)] = (min(l, l_new), max(u, u_new), i and i_new)
    incoming: dict[Any, dict[str, tuple[float, float, bool] | None]] = {
        k: {s: None for s in node.states} for k, node in model.nodes.items()
    }
    for parent_name, parent in model.nodes.items():
        for state_name in parent.states:
            l_new, u_new, i_new = outgoing[(parent_name, state_name)]  # type: ignore[misc]
            for child in parent.children:
                domain = incoming[child.term][state_name]
                if domain is None:
                    incoming[child.term][state_name] = (l_new, u_new, i_new)
                else:
                    l, u, i = domain
                    incoming[child.term][state_name] = (min(l, l_new), max(u, u_new), i and i_new)
    for state_name in model.initial_root_state:
        for child in model.root_children:
            incoming[child.term][state_name] = (-math.inf, math.inf, False)
    return incoming
