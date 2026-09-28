"""Python port of the SDDP.jl "Introductory theory" explanation page.

Source: ``docs/src/explanation/theory_intro.jl`` (https://sddp.dev/stable/explanation/theory_intro/).

The page teaches how SDDP works by implementing a *vanilla* version of the algorithm
from scratch. This module does the same: nothing here uses ``sddp.train`` or
``sddp.PolicyGraph``; only the LP layer (:class:`sddp.solver.model.Model`, the same
thin pyoptinterface wrapper the package uses) plays the role JuMP + HiGHS play on the
page. The structures, function names and the order of the code follow the Julia page.

Differences from the Julia page (all deliberate):

* ``ForwardDiff.gradient`` is replaced by central finite differences
  (:func:`finite_difference_gradient`).
* Randomness goes through an explicit ``random.Random`` instance (``rng``) so runs are
  reproducible; the page calls the global ``rand()``.
* Printing goes to ``io``; pass ``io=None`` to silence a function (the page passes
  ``devnull``).
* Node indices are 1-based like the page (``arcs[i - 1][j]`` is the probability of the
  arc ``i => j``); :func:`get_node` hides the ``- 1``.
* Each :class:`Node` additionally records the cuts it receives (:attr:`Node.cuts`) so
  the cost-to-go approximation can be evaluated at any point
  (:func:`cost_to_go_approximation`). The page does not need this because it never
  queries the approximation directly; the tests use it to compare against the package.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from sddp.solver.model import HiGHS, Model, OptimizerFactory, Sense, Variable

# ---------------------------------------------------------------------------
# Small helpers replacing Julia conveniences
# ---------------------------------------------------------------------------


def _println(io: TextIO | None, *args: Any) -> None:
    """``println(io, args...)``; a ``None`` io is ``devnull``."""
    if io is not None:
        print("".join(str(a) for a in args), file=io)


def finite_difference_gradient(
    f: Callable[[list[float]], float], x: Sequence[float], h: float = 1e-6
) -> list[float]:
    """Central finite-difference gradient of ``f`` at ``x`` (stands in for ForwardDiff)."""
    x = [float(v) for v in x]
    grad = []
    for i in range(len(x)):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        grad.append((f(xp) - f(xm)) / (2 * h))
    return grad


# ---------------------------------------------------------------------------
# Preliminaries: Kelley's cutting plane algorithm
# ---------------------------------------------------------------------------


@dataclass
class KelleyResult:
    x: list[float]
    lower_bound: float
    upper_bound: float
    iterations: int
    status: str  # "converged" or "iteration limit"


def kelleys_cutting_plane(
    # The function to be minimized.
    f: Callable[[list[float]], float],
    # The gradient of `f`. By default we use central finite differences (the page uses
    # automatic differentiation) so the user doesn't have to provide it.
    dfdx: Callable[[list[float]], Sequence[float]] | None = None,
    *,
    # The number of arguments to `f`.
    input_dimension: int,
    # A lower bound for the function `f` over its domain.
    lower_bound: float,
    # The number of iterations to run Kelley's algorithm for before stopping.
    iteration_limit: int,
    # The absolute tolerance ϵ to use for convergence.
    tolerance: float = 1e-6,
    optimizer: OptimizerFactory = HiGHS,
    io: TextIO | None = sys.stdout,
) -> KelleyResult:
    """Kelley's cutting plane algorithm for minimising a convex function.

    Returns the last candidate ``x_K`` together with the final bounds (the page only
    prints them).
    """
    if dfdx is None:

        def dfdx(x: list[float]) -> Sequence[float]:
            return finite_difference_gradient(f, x)

    # Step (1):
    K = 0
    model = Model(optimizer)  # silent by default
    theta = model.add_variable("θ", lb=lower_bound)
    x = [model.add_variable(f"x[{i + 1}]") for i in range(input_dimension)]
    model.set_objective(1.0 * theta, Sense.MIN)
    x_k = [math.nan] * input_dimension
    lower_bound, upper_bound = -math.inf, math.inf
    status = "iteration limit"
    while True:
        # Step (2):
        model.optimize()
        x_k = [model.value(xi) for xi in x]
        # Step (3):
        lower_bound = model.objective_value()
        upper_bound = min(upper_bound, f(x_k))
        _println(io, f"K = {K} : {lower_bound} <= f(x*) <= {upper_bound}")
        # Step (4): θ >= f(x_k) + dfdx(x_k)' * (x - x_k)
        g = [float(v) for v in dfdx(x_k)]
        model.add_constraint(
            theta >= f(x_k) + sum(g[i] * (x[i] - x_k[i]) for i in range(input_dimension))
        )
        # Step (5):
        K += 1
        # Step (6):
        if K == iteration_limit:
            _println(io, "-- Termination status: iteration limit --")
            break
        elif abs(upper_bound - lower_bound) < tolerance:
            status = "converged"
            _println(io, "-- Termination status: converged --")
            break
    _println(io, "Found solution: x_K = ", x_k)
    return KelleyResult(x_k, lower_bound, upper_bound, K, status)


def kelleys_example(io: TextIO | None = sys.stdout) -> KelleyResult:
    """The page's example: minimise ``(x1 - 1)^2 + (x2 + 2)^2 + 1``."""
    return kelleys_cutting_plane(
        lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2 + 1.0,
        input_dimension=2,
        lower_bound=0.0,
        iteration_limit=20,
        io=io,
    )


# ---------------------------------------------------------------------------
# Implementation: modeling
# ---------------------------------------------------------------------------


@dataclass
class State:
    """An incoming and an outgoing state variable."""

    in_: Variable
    out: Variable


@dataclass
class Uncertainty:
    """A finite discrete random variable: ``Ω[i]`` occurs w.p. ``P[i]``.

    ``parameterize(ω)`` updates the subproblem for the realization ``ω``.
    """

    parameterize: Callable[[Any], None]
    Ω: list[Any]
    P: list[float]


@dataclass
class Cut:
    """``cost_to_go >= intercept + Σ coefficients[k] * x_out[k]`` (recorded for inspection)."""

    intercept: float
    coefficients: dict[str, float]


@dataclass
class Node:
    subproblem: Model
    states: dict[str, State]
    uncertainty: Uncertainty
    cost_to_go: Variable
    cuts: list[Cut] = field(default_factory=list)


@dataclass
class PolicyGraph:
    """``nodes[i - 1]`` is node ``i``; ``arcs[i - 1][j]`` is P(i => j) (1-based, like Julia).

    The root node transitions to node 1 with probability 1 and there are no other
    incoming arcs to node 1. Cyclic graphs are allowed.
    """

    nodes: list[Node]
    arcs: list[dict[int, float]]

    def __repr__(self) -> str:
        lines = [f"A policy graph with {len(self.nodes)} nodes", "Arcs:"]
        for from_, arcs in enumerate(self.arcs, start=1):
            for to, probability in arcs.items():
                lines.append(f"  {from_} => {to} w.p. {probability}")
        return "\n".join(lines) + "\n"


def get_node(model: PolicyGraph, t: int) -> Node:
    """Node ``t`` (1-based, as on the page)."""
    return model.nodes[t - 1]


def subproblem_builder(subproblem: Model, t: int) -> tuple[dict[str, State], Uncertainty]:
    """The hydro-thermal example of the page, built on a bare :class:`Model`."""
    # Define the state variables. Note how we fix the incoming state to the initial state
    # value regardless of `t`! This isn't strictly necessary; it only matters for node 1.
    volume_in = subproblem.add_variable("volume_in")
    subproblem.fix(volume_in, 200.0)
    volume_out = subproblem.add_variable("volume_out", lb=0.0, ub=200.0)
    states = {"volume": State(volume_in, volume_out)}
    # Define the control variables.
    thermal_generation = subproblem.add_variable("thermal_generation", lb=0.0)
    hydro_generation = subproblem.add_variable("hydro_generation", lb=0.0)
    hydro_spill = subproblem.add_variable("hydro_spill", lb=0.0)
    inflow = subproblem.add_variable("inflow")
    # Define the constraints
    subproblem.add_constraint(
        volume_out == volume_in + inflow - hydro_generation - hydro_spill, name="balance"
    )
    subproblem.add_constraint(
        thermal_generation + hydro_generation == 150.0, name="demand_constraint"
    )
    # Define the objective for each stage `t` (t = 1, 2, 3).
    fuel_cost = [50.0, 100.0, 150.0]
    subproblem.set_objective(fuel_cost[t - 1] * thermal_generation, Sense.MIN)
    # Finally, the uncertainty object: the user only modifies constraints (here, a bound),
    # never the objective.
    uncertainty = Uncertainty(lambda ω: subproblem.fix(inflow, ω), [0.0, 50.0, 100.0], [1 / 3] * 3)
    return states, uncertainty


def build_policy_graph(
    subproblem_builder: Callable[[Model, int], tuple[dict[str, State], Uncertainty]],
    *,
    graph: list[dict[int, float]],
    lower_bound: float,
    optimizer: OptimizerFactory = HiGHS,
) -> PolicyGraph:
    """The page's ``PolicyGraph(subproblem_builder; graph, lower_bound, optimizer)``."""
    nodes: list[Node] = []
    for t in range(1, len(graph) + 1):
        # Create a model.
        model = Model(optimizer)
        # The user's function returns the states and an `Uncertainty` object.
        states, uncertainty = subproblem_builder(model, t)
        # Now add the cost-to-go term:
        cost_to_go = model.add_variable("cost_to_go", lb=lower_bound)
        obj = model.objective_function()
        model.set_objective(obj + cost_to_go, Sense.MIN)
        # If there are no outgoing arcs, the cost-to-go is 0.0.
        if len(graph[t - 1]) == 0:
            model.fix(cost_to_go, 0.0)
        nodes.append(Node(model, states, uncertainty, cost_to_go))
    return PolicyGraph(nodes, graph)


FINITE_GRAPH: list[dict[int, float]] = [{2: 1.0}, {3: 1.0}, {}]
INFINITE_GRAPH: list[dict[int, float]] = [{2: 1.0}, {3: 1.0}, {2: 0.5}]


def build_finite_model(optimizer: OptimizerFactory = HiGHS) -> PolicyGraph:
    return build_policy_graph(
        subproblem_builder, graph=FINITE_GRAPH, lower_bound=0.0, optimizer=optimizer
    )


def build_infinite_model(optimizer: OptimizerFactory = HiGHS) -> PolicyGraph:
    return build_policy_graph(
        subproblem_builder, graph=INFINITE_GRAPH, lower_bound=0.0, optimizer=optimizer
    )


# ---------------------------------------------------------------------------
# Implementation: helpful samplers
# ---------------------------------------------------------------------------


def sample_uncertainty(uncertainty: Uncertainty, rng: random.Random) -> Any:
    r = rng.random()  # uniform in [0, 1)
    for p, ω in zip(uncertainty.P, uncertainty.Ω):
        r -= p
        if r < 0.0:
            return ω
    raise RuntimeError("We should never get here because P should sum to 1.0.")


def sample_next_node(model: PolicyGraph, current: int, rng: random.Random) -> int | None:
    arcs = model.arcs[current - 1]
    if len(arcs) == 0:
        # No outgoing arcs!
        return None
    r = rng.random()
    for to, probability in arcs.items():
        r -= probability
        if r < 0.0:
            return to
    # We looped through the outgoing arcs and still have probability left over! This
    # means we've hit an implicit "zero" node.
    return None


# ---------------------------------------------------------------------------
# Implementation: the forward pass
# ---------------------------------------------------------------------------

Trajectory = list[tuple[int, dict[str, float]]]


def forward_pass(
    model: PolicyGraph, rng: random.Random, io: TextIO | None = sys.stdout
) -> tuple[Trajectory, float]:
    _println(io, "| Forward Pass")
    # First, get the value of the state at the root node (e.g., x_R).
    node1 = get_node(model, 1)
    incoming_state = {k: node1.subproblem.fix_value(v.in_) for k, v in node1.states.items()}
    # `simulation_cost` accumulates the stage-costs incurred over the forward pass.
    simulation_cost = 0.0
    # Record the nodes visited and the resultant outgoing states for the backward pass.
    trajectory: Trajectory = []
    t: int | None = 1
    while t is not None:
        node = get_node(model, t)
        _println(io, f"| | Visiting node {t}")
        # Sample the uncertainty:
        ω = sample_uncertainty(node.uncertainty, rng)
        _println(io, "| | | ω = ", ω)
        # Parameterize the subproblem using the user-provided function:
        node.uncertainty.parameterize(ω)
        _println(io, "| | | x = ", incoming_state)
        # Update the incoming state variable:
        for k, v in incoming_state.items():
            node.subproblem.fix(node.states[k].in_, v)
        # Now solve the subproblem and check we found an optimal solution:
        node.subproblem.optimize()
        if not node.subproblem.has_primal_solution():
            raise RuntimeError("Something went terribly wrong!")
        # Compute the outgoing state variables:
        outgoing_state = {k: node.subproblem.value(v.out) for k, v in node.states.items()}
        _println(io, "| | | x′ = ", outgoing_state)
        # The stage cost is the objective less the cost-to-go term:
        stage_cost = node.subproblem.objective_value() - node.subproblem.value(node.cost_to_go)
        simulation_cost += stage_cost
        _println(io, "| | | C(x, u, ω) = ", stage_cost)
        # Set the outgoing state of stage t as the incoming state of the next stage, and
        # add the node to the trajectory.
        incoming_state = outgoing_state
        trajectory.append((t, outgoing_state))
        # Finally, sample a new node to step to (`None` ends the walk).
        t = sample_next_node(model, t, rng)
    return trajectory, simulation_cost


# ---------------------------------------------------------------------------
# Implementation: the backward pass
# ---------------------------------------------------------------------------


def add_cut(
    node: Node, intercept: float, coefficients: dict[str, float], io: TextIO | None
) -> None:
    """Add ``cost_to_go >= intercept + Σ coefficients[k] * x_out[k]`` to ``node``."""
    node.subproblem.add_constraint(
        node.cost_to_go >= intercept + sum(coefficients[k] * x.out for k, x in node.states.items())
    )
    node.cuts.append(Cut(intercept, dict(coefficients)))
    _println(io, "| | | Adding cut : cost_to_go >= ", intercept, " + ", coefficients, " ⋅ x′")


def backward_pass(
    model: PolicyGraph, trajectory: Trajectory, io: TextIO | None = sys.stdout
) -> None:
    _println(io, "| Backward pass")
    # For the backward pass, we walk back up the nodes.
    for index, outgoing_states in reversed(trajectory):
        node = get_node(model, index)
        _println(io, f"| | Visiting node {index}")
        if len(model.arcs[index - 1]) == 0:
            # If there are no children, the cost-to-go is 0.
            _println(io, "| | | Skipping node because the cost-to-go is 0")
            continue
        # Build up the cut  θ >= Σ_j Σ_φ P_ij p_φ [V + dVdx′ᵀ (x′ - x′_k)]
        intercept = 0.0
        coefficients = {k: 0.0 for k in node.states}
        # For each node j ∈ i⁺
        for j, P_ij in model.arcs[index - 1].items():
            next_node = get_node(model, j)
            # Set the incoming states of node j to the outgoing states of node i
            for k, v in outgoing_states.items():
                next_node.subproblem.fix(next_node.states[k].in_, v)
            # Then for each realization of φ ∈ Ωⱼ
            for pφ, φ in zip(next_node.uncertainty.P, next_node.uncertainty.Ω):
                # Setup and solve for the realization of φ
                _println(io, "| | | Solving φ = ", φ)
                next_node.uncertainty.parameterize(φ)
                next_node.subproblem.optimize()
                # Then prepare the cut `P_ij * pφ * [V + dVdxᵀ(x - x_k)]`
                V = next_node.subproblem.objective_value()
                _println(io, "| | | | V = ", V)
                # The subgradient is the reduced cost of the fixed incoming state variable
                # (d objective / d fixed value, the same convention the package uses).
                dVdx = {
                    k: next_node.subproblem.reduced_cost(v.in_) for k, v in next_node.states.items()
                }
                _println(io, "| | | | dVdx′ = ", dVdx)
                intercept += P_ij * pφ * (V - sum(dVdx[k] * outgoing_states[k] for k in dVdx))
                for k in coefficients:
                    coefficients[k] += P_ij * pφ * dVdx[k]
        # And then refine the cost-to-go variable by adding the cut:
        add_cut(node, intercept, coefficients, io)


# ---------------------------------------------------------------------------
# Implementation: bounds
# ---------------------------------------------------------------------------


def lower_bound(model: PolicyGraph) -> float:
    """``E_{ω ∈ Ω_1}[V_1^K(x_R, ω)]`` (assumes the root has a single arc to node 1)."""
    node = get_node(model, 1)
    bound = 0.0
    for p, ω in zip(node.uncertainty.P, node.uncertainty.Ω):
        node.uncertainty.parameterize(ω)
        node.subproblem.optimize()
        bound += p * node.subproblem.objective_value()
    return bound


def upper_bound(
    model: PolicyGraph, *, replications: int, rng: random.Random
) -> tuple[float, float]:
    """Monte Carlo estimate ``(μ, 1.96 σ / √n)`` of the policy's expected cost."""
    # Pipe the output to devnull (io=None) so we don't print too much!
    simulations = [forward_pass(model, rng, None) for _ in range(replications)]
    z = [s[1] for s in simulations]
    μ = statistics.mean(z)
    tσ = 1.96 * statistics.stdev(z) / math.sqrt(replications)
    return μ, tσ


# ---------------------------------------------------------------------------
# Implementation: the training loop
# ---------------------------------------------------------------------------


def train(
    model: PolicyGraph,
    *,
    iteration_limit: int,
    replications: int,
    rng: random.Random | None = None,
    io: TextIO | None = sys.stdout,
) -> tuple[float, float]:
    """Forward/backward passes, then a simulation; returns the final ``(μ, tσ)``."""
    rng = rng if rng is not None else random.Random()
    for i in range(1, iteration_limit + 1):
        _println(io, f"Starting iteration {i}")
        outgoing_states, _ = forward_pass(model, rng, io)
        backward_pass(model, outgoing_states, io)
        _println(io, "| Finished iteration")
        _println(io, "| | lower_bound = ", lower_bound(model))
    _println(io, "Termination status: iteration limit")
    μ, tσ = upper_bound(model, replications=replications, rng=rng)
    _println(io, f"Upper bound = {μ} ± {tσ}")
    return μ, tσ


# ---------------------------------------------------------------------------
# Implementation: evaluating the policy
# ---------------------------------------------------------------------------


def evaluate_policy(
    model: PolicyGraph, *, node: int, incoming_state: dict[str, float], random_variable: Any
) -> dict[str, float]:
    """Solve ``node`` at ``incoming_state`` and ``random_variable`` (may be out-of-sample).

    Returns every named variable's value plus each named constraint's primal value, like
    ``JuMP.value.(object_dictionary)`` on the page.
    """
    the_node = get_node(model, node)
    the_node.uncertainty.parameterize(random_variable)
    for k, v in incoming_state.items():
        the_node.subproblem.fix(the_node.states[k].in_, v)
    the_node.subproblem.optimize()
    m = the_node.subproblem
    out: dict[str, float] = {}
    for name, obj in m.names.items():
        out[name] = m.value(obj) if isinstance(obj, Variable) else m.constraint_primal(obj)
    return out


def cost_to_go_approximation(node: Node, outgoing_state: dict[str, float]) -> float:
    """Height of the current cut polyhedron ``V^K`` of ``node`` at ``outgoing_state``.

    Not on the page; used to inspect the approximation (``max(M, max_k cut_k(x′))``).
    """
    height = node.subproblem.lower_bound(node.cost_to_go)
    for cut in node.cuts:
        height = max(
            height,
            cut.intercept + sum(c * outgoing_state[k] for k, c in cut.coefficients.items()),
        )
    return height


# ---------------------------------------------------------------------------
# main: reproduce the page
# ---------------------------------------------------------------------------


def main(seed: int = 1234, io: TextIO | None = sys.stdout) -> None:
    rng = random.Random(seed)
    # Kelley's cutting plane algorithm
    kelleys_example(io)
    # Finite-horizon model
    model = build_finite_model()
    _println(io, model)
    for _ in range(3):
        _println(io, "ω = ", sample_uncertainty(get_node(model, 1).uncertainty, rng))
    for i in range(1, 4):
        _println(io, f"Next node from {i} = ", repr(sample_next_node(model, i, rng)))
    forward_pass(model, rng, io)
    _println(io, "lower_bound(model) = ", lower_bound(model))
    train(model, iteration_limit=3, replications=100, rng=rng, io=io)
    _println(
        io,
        evaluate_policy(model, node=1, incoming_state={"volume": 150.0}, random_variable=75),
    )
    # Infinite horizon
    model = build_infinite_model()
    train(model, iteration_limit=3, replications=100, rng=rng, io=io)
    _println(
        io,
        evaluate_policy(model, node=3, incoming_state={"volume": 100.0}, random_variable=10.0),
    )


if __name__ == "__main__":
    main()
