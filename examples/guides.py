"""Python translations of the SDDP.jl how-to guides (docs/src/guides/*.md).

One function per code block, in the order of the Julia pages. ``tests/test_guides.py``
checks them against SDDP.jl (``reference/oracle/guides.json``) and hand-computed values.
"""

from __future__ import annotations

import itertools
import math
import random
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import sddp
from examples.hydro_thermal import subproblem_builder as _hydro_thermal_builder

KW = {"print_level": 0, "run_numerical_stability_report": False}


# =========================================================== access_previous_variables
def build_first_stage_capacity_model(w: Sequence[float] | None = None) -> sddp.PolicyGraph:
    """Access a first-stage decision in a future stage (the guide uses ``rand(4)`` inflows)."""
    if w is None:
        w = [random.Random(0).random() for _ in range(4)]

    def builder(sp: sddp.Subproblem, t: int) -> None:
        # Capacity of the generator. Decided in the first stage.
        capacity = sp.add_state("capacity", lb=0.0, initial_value=0.0)
        # Quantity of water stored.
        reservoir = sp.add_state("reservoir", lb=0.0, initial_value=0.0)
        # Quantity of water to use for electricity generation in current stage.
        generation = sp.add_variable("generation", lb=0.0)
        if t == 1:
            # No constraints in the first stage, but push the reservoir to the next stage.
            sp.add_constraint(reservoir.out == reservoir.in_)
            # Since we're maximizing profit, subtract cost of capacity.
            sp.set_stage_objective(-capacity.out)
        else:
            # Water balance constraint.
            balance = sp.add_constraint(
                reservoir.out - reservoir.in_ + generation == 0, name="balance"
            )
            # Generation limit.
            sp.add_constraint(generation <= capacity.in_)
            # Push capacity to the next stage.
            sp.add_constraint(capacity.out == capacity.in_)
            # Maximize generation.
            sp.set_stage_objective(generation)
            # Random inflow in balance constraint.
            sp.parameterize(lambda omega: sp.set_normalized_rhs(balance, omega), list(w))

    return sddp.LinearPolicyGraph(
        builder, stages=10, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS
    )


def build_pipeline_lag_model() -> sddp.PolicyGraph:
    """Access a decision from N stages ago: a 5-stage inventory pipeline (deterministic)."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        # Current inventory on hand.
        inventory = sp.add_state("inventory", lb=0.0, initial_value=0.0)
        # Inventory pipeline: pipeline[1].out are orders placed today, pipeline[5].in are
        # orders that arrive today. Stock moves up one slot each stage.
        pipeline = [sp.add_state(f"pipeline[{i}]", initial_value=0.0) for i in range(1, 6)]
        buy = sp.add_variable("buy", lb=0.0, ub=10.0)
        sell = sp.add_variable("sell", lb=0.0)
        # Buy orders get placed in the pipeline.
        sp.add_constraint(pipeline[0].out == buy)
        # Stock moves up one slot in the pipeline each stage.
        for i in range(1, 5):
            sp.add_constraint(pipeline[i].out == pipeline[i - 1].in_)
        # Stock balance constraint.
        sp.add_constraint(inventory.out == inventory.in_ - sell + pipeline[4].in_)
        # Maximize quantity of sold items.
        sp.set_stage_objective(sell)

    return sddp.LinearPolicyGraph(
        builder, stages=10, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS
    )


def build_stochastic_lead_time_model(T: int = 10, corrected: bool = False) -> sddp.PolicyGraph:
    """Stochastic lead times via ``set_normalized_coefficient`` inside ``parameterize``.

    The guide's code sets the *normalized* coefficient of ``u_buy`` to ``+1``, which (because
    JuMP/the port normalise ``x_pipeline[i].out == x_pipeline[i+1].in + 1 * u_buy`` to
    ``x_pipeline[i].out - x_pipeline[i+1].in - u_buy == 0``) flips the sign of the order and
    makes buying useless (optimal value 0). ``corrected=True`` sets ``-1`` instead, which is
    what the prose describes (orders arrive ``ω`` stages later; optimal value 145 for T=10).
    """

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x_inventory = sp.add_state("x_inventory", lb=0.0, initial_value=0.0)
        x_pipeline = [sp.add_state(f"x_pipeline[{i}]", initial_value=0.0) for i in range(1, T + 2)]
        u_buy = sp.add_variable("u_buy", lb=0.0, ub=10.0)
        u_sell = sp.add_variable("u_sell", lb=0.0)
        sp.fix(x_pipeline[T].out, 0.0)
        sp.set_stage_objective(u_sell)
        # Shift the orders one stage.
        c_pipeline = [
            sp.add_constraint(
                x_pipeline[i].out == x_pipeline[i + 1].in_ + 1 * u_buy, name=f"c_pipeline[{i + 1}]"
            )
            for i in range(T)
        ]
        # x_pipeline[1].in are arriving on the inventory.
        sp.add_constraint(x_inventory.out == x_inventory.in_ - u_sell + x_pipeline[0].in_)

        def modify(omega: int) -> None:
            # Rewrite c_pipeline[i] indicating how many stages ahead the order arrives (ω).
            for i in range(T):
                v = 1 if omega == i + 1 else 0
                sp.set_normalized_coefficient(c_pipeline[i], u_buy, -v if corrected else v)

        sp.parameterize(modify, range(1, T + 1))

    return sddp.LinearPolicyGraph(
        builder, stages=20, sense="Max", upper_bound=1000.0, optimizer=sddp.HiGHS
    )


def stochastic_lead_time_optimal_value(T: int = 10, stages: int = 20) -> float:
    """Closed form for the corrected model: buy 10 every stage, sell when the order arrives."""
    return 10.0 * sum(
        sum(1.0 / T for omega in range(1, T + 1) if t + omega <= stages)
        for t in range(1, stages + 1)
    )


# ==================================================================== add_a_custom_cut
def create_custom_cut_model() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=100.0, integer=True, initial_value=0.0)
        u_p = sp.add_variable("u_p", lb=0.0, ub=200.0, integer=True)
        u_o = sp.add_variable("u_o", lb=0.0, integer=True)
        w = sp.add_variable("w")
        sp.add_constraint(x.out == x.in_ + u_p + u_o - w)
        sp.set_stage_objective(100 * u_p + 300 * u_o + 50 * x.out)
        omega = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        sp.parameterize(lambda w_: sp.fix(w, w_), omega[t - 1])

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


CUSTOM_CUT_JSON = """
[{
    "node": "1",
    "single_cuts": [{
        "state": {"x": 10.0},
        "coefficients": {"x": -200.0},
        "intercept": 55500.0
    }],
    "risk_set_cuts": [],
    "multi_cuts": []
}]
"""


def write_cuts_after_one_iteration(filename: str, seed: int | None = 1) -> str:
    """Train for one iteration and write the cuts to ``filename``; returns the file's text."""
    model = create_custom_cut_model()
    sddp.train(model, iteration_limit=1, seed=seed, **KW)
    sddp.write_cuts_to_file(model, filename)
    with open(filename) as io:
        return io.read()


def bound_with_custom_cut(filename: str) -> tuple[float, float]:
    """Write ``θ ≥ 55500 − 200 (x − 10)`` for node 1 to ``filename`` and read it back.

    Returns ``(bound before, bound after)`` = ``(10000.0, 62500.0)``.
    """
    model = create_custom_cut_model()
    before = sddp.calculate_bound(model)
    with open(filename, "w") as io:
        io.write(CUSTOM_CUT_JSON)
    sddp.read_cuts_from_file(model, filename)
    return before, sddp.calculate_bound(model)


# ================================================= add_a_multidimensional_state_variable
def build_multidimensional_state_model(
    record: list[float] | None = None, printer: Callable[[str], Any] = print
) -> sddp.PolicyGraph:
    """Scalar, vector and 2-D containers of state variables (a 1-stage model)."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        # A scalar state variable.
        x = sp.add_state("x", lb=0.0, initial_value=0.0)
        lbs = [sp.lower_bound(x.out)]
        printer(f"Lower bound of outgoing x is: {lbs[-1]}")
        # A vector of state variables.
        y = {i: sp.add_state(f"y[{i}]", lb=float(i), initial_value=float(i)) for i in (1, 2)}
        lbs.append(sp.lower_bound(y[1].out))
        printer(f"Lower bound of outgoing y[1] is: {lbs[-1]}")
        # A 2-D container of state variables (Julia: a DenseAxisArray).
        z = {
            (i, j): sp.add_state(f"z[{i},{j}]", lb=float(i), initial_value=float(i))
            for i in (3, 4)
            for j in ("A", "B")
        }
        lbs.append(sp.lower_bound(z[3, "B"].out))
        printer(f"Lower bound of outgoing z[3, :B] is: {lbs[-1]}")
        if record is not None:
            record.extend(lbs)

    return sddp.LinearPolicyGraph(builder, stages=1, lower_bound=0.0, optimizer=sddp.HiGHS)


# ================================================================== add_a_risk_measure
def build_risk_measure_model() -> sddp.PolicyGraph:
    """The "first steps" hydro-thermal model, on which the guide trains risk-averse policies."""
    return sddp.LinearPolicyGraph(
        _hydro_thermal_builder, stages=3, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def train_with_risk_measure(
    model: sddp.PolicyGraph, iteration_limit: int = 10, seed: int | None = None
) -> None:
    """The same risk measure at every node."""
    sddp.train(
        model, risk_measure=sddp.WorstCase(), iteration_limit=iteration_limit, seed=seed, **KW
    )


def train_with_risk_measure_dict(
    model: sddp.PolicyGraph, iteration_limit: int = 10, seed: int | None = None
) -> None:
    """A dictionary with one risk measure per node (the guide's ``Dict(1 => ..., 2 => ...)``)."""
    sddp.train(
        model,
        risk_measure={1: sddp.Expectation(), 2: sddp.WorstCase(), 3: sddp.WorstCase()},
        iteration_limit=iteration_limit,
        seed=seed,
        **KW,
    )


def train_with_risk_measure_function(
    model: sddp.PolicyGraph, iteration_limit: int = 10, seed: int | None = None
) -> None:
    """A function ``node_index -> risk measure``."""

    def risk_measure(node_index: int) -> sddp.Expectation | sddp.WorstCase:
        if node_index == 1:
            return sddp.Expectation()
        return sddp.WorstCase()

    sddp.train(model, risk_measure=risk_measure, iteration_limit=iteration_limit, seed=seed, **KW)


def supported_risk_measures() -> list[Any]:
    """One instance of every risk measure listed in the guide."""
    return [
        sddp.Expectation(),
        sddp.AVaR(0.5),
        sddp.ConvexCombination((0.5, sddp.Expectation()), (0.5, sddp.AVaR(0.5))),
        sddp.CVaR(0.5),
        sddp.EAVaR(lambda_=0.5, beta=0.5),
        sddp.Entropic(1.0),
        sddp.ModifiedChiSquared(0.5),
        sddp.Wasserstein(lambda x, y: abs(x - y), sddp.HiGHS, alpha=0.5),
        sddp.WorstCase(),
    ]


# =================================================================== add_integrality
def build_integrality_model() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=100.0, integer=True, initial_value=0.0)
        u = sp.add_variable("u", lb=0.0, ub=200.0, integer=True)
        v = sp.add_variable("v", lb=0.0)
        sp.add_constraint(x.out == x.in_ + u + v - 150)
        sp.set_stage_objective(2 * u + 6 * v + x.out)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


def train_with_bandit_duality(
    model: sddp.PolicyGraph,
    iteration_limit: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> None:
    sddp.train(
        model,
        duality_handler=sddp.BanditDuality(),
        log_every_iteration=True,
        iteration_limit=iteration_limit,
        seed=seed,
        **kwargs,
    )


def train_with_conic_duality_optimizer(
    model: sddp.PolicyGraph,
    optimizer: sddp.OptimizerFactory | None = None,
    iteration_limit: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> None:
    """``ContinuousConicDuality(Ipopt.Optimizer)`` in the guide; Ipopt is not available in the
    port, so a HiGHS factory with different options stands in for the relaxed solves."""
    if optimizer is None:
        optimizer = sddp.HiGHS.with_options(presolve="off")
    sddp.train(
        model,
        duality_handler=sddp.ContinuousConicDuality(optimizer),
        iteration_limit=iteration_limit,
        seed=seed,
        **kwargs,
    )


# ========================================================== add_multidimensional_noise
class Realization(NamedTuple):
    value: float
    coefficient: float


def build_simple_multidimensional_noise_model() -> sddp.PolicyGraph:
    """Cartesian product of two marginals with product probabilities."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=0.0)
        omega = [Realization(v, c) for v in [1, 2] for c in [3, 4, 5]]
        P = [v * c for v in [0.5, 0.5] for c in [0.3, 0.5, 0.2]]

        def modify(w: Realization) -> None:
            sp.fix(x.out, w.value)
            sp.set_stage_objective(w.coefficient * x.out)

        sp.parameterize(modify, omega, P)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


def _binomial_pmf(n: int, p: float, k: int) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def finite_product_distribution() -> tuple[list[list[int]], list[float]]:
    """``Ω``/``P`` of ``Product([Binomial(10, .5), Bernoulli(.5), truncated(Poisson(5), 2, 8)])``
    computed with :mod:`math` (Distributions.jl is not available). The enumeration order is
    Julia's ``Base.product``: the first factor varies fastest."""
    supports = [list(range(0, 11)), [0, 1], list(range(2, 9))]
    pois = {k: _poisson_pmf(5.0, k) for k in supports[2]}
    total = sum(pois.values())
    pmfs = [
        {k: _binomial_pmf(10, 0.5, k) for k in supports[0]},
        {0: 0.5, 1: 0.5},
        {k: v / total for k, v in pois.items()},
    ]
    omega = [[b, be, p] for p, be, b in itertools.product(supports[2], supports[1], supports[0])]
    P = [pmfs[0][w[0]] * pmfs[1][w[1]] * pmfs[2][w[2]] for w in omega]
    return omega, P


def sampled_product_distribution(
    N: int = 100, seed: int | None = None
) -> tuple[list[list[int]], list[float]]:
    """``N`` samples of ``Product([Binomial(100, 0.5), Geometric(1/20), Poisson(20)])`` with
    uniform probabilities. ``Geometric`` counts failures (support ``0, 1, ...``), like
    Distributions.jl; ``numpy``'s counts trials, hence the ``- 1``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    omega = [
        [int(rng.binomial(100, 0.5)), int(rng.geometric(1 / 20)) - 1, int(rng.poisson(20))]
        for _ in range(N)
    ]
    return omega, [1.0 / N] * N


def build_vector_noise_model(
    omega: Sequence[Sequence[float]], P: Sequence[float]
) -> sddp.PolicyGraph:
    """``fix(x.out, ω[1]); @stageobjective(ω[2] * x.out + ω[3])`` with vector-valued ``ω``."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=0.0)

        def modify(w: Sequence[float]) -> None:
            sp.fix(x.out, w[0])
            sp.set_stage_objective(w[1] * x.out + w[2])

        sp.parameterize(modify, omega, P)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


# ================================================== add_noise_in_the_constraint_matrix
def build_constraint_matrix_noise_model() -> sddp.PolicyGraph:
    """``set_normalized_coefficient(emissions, x.out, ω)`` rewrites ``1 x.out <= 1`` to
    ``ω x.out <= 1``."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=0.0)
        emissions = sp.add_constraint(1 * x.out <= 1, name="emissions")
        sp.parameterize(
            lambda omega: sp.set_normalized_coefficient(emissions, x.out, omega), [0.2, 0.5, 1.0]
        )
        sp.set_stage_objective(-x.out)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


# ============================================================== choose_a_stopping_rule
def train_iteration_limit(model: sddp.PolicyGraph, seed: int | None = None) -> str:
    sddp.train(model, iteration_limit=10, seed=seed, **KW)
    return sddp.termination_status(model)


def train_time_limit(
    model: sddp.PolicyGraph, time_limit: float = 2.0, seed: int | None = None
) -> str:
    sddp.train(model, time_limit=time_limit, seed=seed, **KW)
    return sddp.termination_status(model)


def train_bound_stalling(model: sddp.PolicyGraph, seed: int | None = None) -> str:
    """Terminate if BoundStalling becomes true."""
    sddp.train(model, stopping_rules=[sddp.BoundStalling(10, rtol=1e-4)], seed=seed, **KW)
    return sddp.termination_status(model)


def train_time_limit_or_bound_stalling(
    model: sddp.PolicyGraph, time_limit: float = 100.0, seed: int | None = None
) -> str:
    """Terminate if TimeLimit OR BoundStalling becomes true."""
    sddp.train(
        model,
        stopping_rules=[sddp.TimeLimit(time_limit), sddp.BoundStalling(10, rtol=1e-4)],
        seed=seed,
        **KW,
    )
    return sddp.termination_status(model)


def train_time_limit_and_bound_stalling(
    model: sddp.PolicyGraph, time_limit: float = 100.0, seed: int | None = None
) -> str:
    """Terminate if TimeLimit AND BoundStalling becomes true."""
    sddp.train(
        model,
        stopping_rules=[
            sddp.StoppingChain(sddp.TimeLimit(time_limit), sddp.BoundStalling(10, rtol=1e-4))
        ],
        seed=seed,
        **KW,
    )
    return sddp.termination_status(model)


def supported_stopping_rules() -> list[Any]:
    """One instance of every stopping rule listed in the guide."""
    return [
        sddp.IterationLimit(1),
        sddp.TimeLimit(1.0),
        sddp.Statistical(num_replications=1, disable_warning=True),
        sddp.BoundStalling(1),
        sddp.StoppingChain(sddp.IterationLimit(1), sddp.TimeLimit(1.0)),
        sddp.SimulationStoppingRule(),
        sddp.FirstStageStoppingRule(),
    ]


# =============================================================== create_a_belief_state
def create_belief_graph() -> sddp.Graph:
    """A Markovian graph with an ambiguity set over the nodes of each stage."""
    G = sddp.MarkovianGraph([[0.5, 0.5], [[0.2, 0.8], [0.8, 0.2]]])
    for t in (1, 2):
        G.add_ambiguity_set([(t, 1), (t, 2)])
    return G


def build_belief_model(G: sddp.Graph | None = None) -> sddp.PolicyGraph:
    """A tiny model on the belief graph (not in the guide; checks the graph is usable)."""
    if G is None:
        G = create_belief_graph()

    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, i = node
        x = sp.add_state("x", lb=0.0, ub=10.0, initial_value=5.0)
        u = sp.add_variable("u", lb=0.0)
        sp.add_constraint(x.out == x.in_ - u)
        sp.parameterize(
            lambda w: sp.set_stage_objective(w * u), [1.0, 2.0] if i == 1 else [2.0, 3.0]
        )

    return sddp.PolicyGraph(builder, G, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS)


# ======================================================= create_a_general_policy_graph
def linear_graph_example() -> tuple[str, sddp.Graph]:
    """``LinearGraph(3)`` (returned as its repr) then ``add_node``/``add_edge`` to close a cycle."""
    graph = sddp.LinearGraph(3)
    before = repr(graph)
    graph.add_node(4)
    graph.add_edge(3, 4, 1.0)
    graph.add_edge(4, 1, 0.9)
    return before, graph


def unicyclic_graph_example() -> sddp.Graph:
    return sddp.UnicyclicGraph(0.95, num_nodes=2)


def markovian_graph_example() -> sddp.Graph:
    return sddp.MarkovianGraph([[1.0], [0.4, 0.6]])


def general_graph_example() -> tuple[str, sddp.Graph]:
    """``Graph(:root_node)`` (returned as its repr) then a self-looping ``decision_node``."""
    graph = sddp.Graph("root_node")
    before = repr(graph)
    graph.add_node("decision_node")
    graph.add_edge("root_node", "decision_node", 1.0)
    graph.add_edge("decision_node", "decision_node", 0.9)
    return before, graph


def create_policy_graph_from_graph(
    printer: Callable[[str], Any] = print,
) -> tuple[sddp.PolicyGraph, list[Any]]:
    """``PolicyGraph(graph)``: the builder is called once per node (not for the root)."""
    graph = sddp.Graph.from_edges(
        "root_node",
        ["decision_node"],
        [(("root_node", "decision_node"), 1.0), (("decision_node", "decision_node"), 0.9)],
    )
    called: list[Any] = []

    def builder(sp: sddp.Subproblem, node: str) -> None:
        printer(f"Called from node: {node}")
        called.append(node)

    model = sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=sddp.HiGHS)
    return model, called


def build_cyclic_model(graph: sddp.Graph | None = None) -> sddp.PolicyGraph:
    """A small model on the cyclic ``LinearGraph(3) + 4 -> 1`` graph (for the simulate block)."""
    if graph is None:
        _, graph = linear_graph_example()

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=1.0)
        sp.add_constraint(x.out <= x.in_)
        sp.set_stage_objective(x.out)

    return sddp.PolicyGraph(builder, graph, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS)


def simulate_fixed_depth(
    model: sddp.PolicyGraph, n: int = 1, max_depth: int = 10, seed: int | None = None
) -> list[list[dict[str, Any]]]:
    """Simulate exactly ``max_depth`` stages of a cyclic policy graph."""
    return sddp.simulate(
        model,
        n,
        sampling_scheme=sddp.InSampleMonteCarlo(max_depth=max_depth, terminate_on_dummy_leaf=False),
        seed=seed,
    )


def simulator(rng: random.Random | None = None) -> list[float]:
    """A random walk with 5 steps (``scenario[i] = scenario[i-1] + rand() - 0.5``)."""
    r = rng if rng is not None else random
    scenario = [0.0] * 5
    for i in range(1, 5):
        scenario[i] = scenario[i - 1] + r.random() - 0.5
    return scenario


def build_simulator_markovian_model(
    budget: int = 10, scenarios: int = 100, seed: int | None = None
) -> sddp.PolicyGraph:
    """``MarkovianGraph(simulator; budget, scenarios)`` with ``(stage, price)`` node indices."""
    rng = random.Random(seed)
    graph = sddp.markovian_graph_from_simulator(
        lambda: simulator(rng), budget=budget, scenarios=scenarios
    )

    def builder(sp: sddp.Subproblem, node: tuple[int, float]) -> None:
        stage, price = node
        x = sp.add_state("x", lb=0.0, initial_value=1.0)
        sp.add_constraint(x.out <= x.in_)
        sp.set_stage_objective(price * x.out)

    return sddp.PolicyGraph(builder, graph, sense="Max", upper_bound=1e3, optimizer=sddp.HiGHS)


# ============================================================= deterministic_equivalent
def build_deterministic_equivalent_model() -> sddp.PolicyGraph:
    """The 2-stage model of the guide (also used by write_subproblems_to_file)."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=1.0)
        y = sp.add_variable("y")
        sp.add_constraint(x.in_ == x.out + y, name="balance")

        def modify(omega: float) -> None:
            sp.set_stage_objective(omega * x.out)
            sp.fix(y, omega)

        sp.parameterize(modify, [1.1, 2.2])

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0, optimizer=sddp.HiGHS)


def solve_deterministic_equivalent(model: sddp.PolicyGraph) -> tuple[sddp.Model, float]:
    det_equiv = sddp.deterministic_equivalent(model, sddp.HiGHS)
    det_equiv.optimize()
    return det_equiv, det_equiv.objective_value()


# ================================================================ implement_a_par_model
ALPHA = [0.5, 0.3, 0.2]
Y_HISTORY = [1.0, 1.0, 1.0]
PAR_OMEGA = [-0.25, 0.0, 0.25]


def build_par_model(
    alpha: Sequence[float] = ALPHA, omega: Sequence[float] = PAR_OMEGA, stages: int = 12
) -> sddp.PolicyGraph:
    """``y_t = Σ_p α_p y_{t-p} + ω_t`` with ``y[1:P]`` states and shift constraints."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        P = len(alpha)
        y = [sp.add_state(f"y[{p}]", initial_value=1.0) for p in range(1, P + 1)]
        w = sp.add_variable("omega")
        # The PAR constraint.
        sp.add_constraint(y[0].out == sum(alpha[p] * y[p].in_ for p in range(P)) + w)
        # "Shift" constraints to move the values of `y` between stages.
        for p in range(1, P):
            sp.add_constraint(y[p].out == y[p - 1].in_)
        # Parameterize the additive noise term.
        sp.parameterize(lambda v: sp.fix(w, v), omega)

    return sddp.LinearPolicyGraph(
        builder, stages=stages, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def simulate_par_model(
    model: sddp.PolicyGraph, sampling_scheme: Any = None, seed: int | None = None
) -> list[float]:
    """``[stage[:y][1].out for stage in simulations[1]]``."""
    sims = sddp.simulate(model, 1, ["y[1]"], sampling_scheme=sampling_scheme, seed=seed)
    return [stage["y[1]"].out for stage in sims[0]]


def par_historical_scheme(values: Sequence[float] = (0.1, -0.1, 0.2)) -> sddp.Historical:
    """``SDDP.Historical(tuple.(1:3, [0.1, -0.1, 0.2]))``."""
    return sddp.Historical([(t, v) for t, v in zip(range(1, len(values) + 1), values)])


def par_recursion(
    alpha: Sequence[float], y_history: Sequence[float], noise: Sequence[float]
) -> list[float]:
    """Hand-computed ``y_t`` for a sequence of additive noises (most recent history first)."""
    hist = list(y_history)
    out = []
    for w in noise:
        y = sum(a * h for a, h in zip(alpha, hist)) + w
        out.append(y)
        hist = [y] + hist[:-1]
    return out


@dataclass(frozen=True)
class NoiseTerm:
    is_noise: bool
    value: float


def build_par_model_with_noise_terms(
    alpha: Sequence[float] = ALPHA, values: Sequence[float] = PAR_OMEGA, stages: int = 12
) -> sddp.PolicyGraph:
    """A noise term that either adds ``ω`` or fixes ``y[1].out`` to an observed value."""
    omega = [NoiseTerm(True, v) for v in values]

    def builder(sp: sddp.Subproblem, t: int) -> None:
        P = len(alpha)
        y = [sp.add_state(f"y[{p}]", initial_value=1.0) for p in range(1, P + 1)]
        w = sp.add_variable("omega")
        sp.add_constraint(y[0].out == sum(alpha[p] * y[p].in_ for p in range(P)) + w)
        for p in range(1, P):
            sp.add_constraint(y[p].out == y[p - 1].in_)

        def modify(term: NoiseTerm) -> None:
            if term.is_noise:
                sp.fix(w, term.value)
            else:
                sp.unfix(w)
                sp.fix(y[0].out, term.value)

        sp.parameterize(modify, omega)

    return sddp.LinearPolicyGraph(
        builder, stages=stages, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def par_noise_term_scheme(values: Sequence[float] = (1.0, 1.2, 1.4)) -> sddp.Historical:
    """``SDDP.Historical(tuple.(1:3, NoiseTerm.(false, [1.0, 1.2, 1.4])))``."""
    return sddp.Historical(
        [(t, NoiseTerm(False, v)) for t, v in zip(range(1, len(values) + 1), values)]
    )


# ==================================================== improve_computational_performance
def build_two_stage_model() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=1.0)
        sp.set_stage_objective(x.out)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0, optimizer=sddp.HiGHS)


def train_single_cut(model: sddp.PolicyGraph, iteration_limit: int = 10, **kwargs: Any) -> None:
    sddp.train(model, cut_type=sddp.SINGLE_CUT, iteration_limit=iteration_limit, **kwargs)


def train_multi_cut(model: sddp.PolicyGraph, iteration_limit: int = 10, **kwargs: Any) -> None:
    sddp.train(model, cut_type=sddp.MULTI_CUT, iteration_limit=iteration_limit, **kwargs)


def train_and_simulate_threaded(
    model: sddp.PolicyGraph, iteration_limit: int = 10, replications: int = 10, **kwargs: Any
) -> list[list[dict[str, Any]]]:
    sddp.train(model, iteration_limit=iteration_limit, parallel_scheme=sddp.Threaded(), **kwargs)
    return sddp.simulate(model, replications, parallel_scheme=sddp.Threaded())


# ============================================ simulate_using_a_different_sampling_scheme
class HydroRealization(NamedTuple):
    inflow: float
    fuel_multiplier: float


OMEGA = [
    HydroRealization(0.0, 1.5),
    HydroRealization(50.0, 1.0),
    HydroRealization(100.0, 0.75),
]


def _markov_probability(markov_state: int) -> list[float]:
    if markov_state == 1:  # wet climate state
        return [1 / 6, 1 / 3, 1 / 2]
    return [1 / 2, 1 / 3, 1 / 6]  # dry climate state


def build_markov_hydro_model() -> sddp.PolicyGraph:
    """The model of the "Markovian policy graphs" tutorial."""

    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, markov_state = node
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal_generation = sp.add_variable("thermal_generation", lb=0.0)
        hydro_generation = sp.add_variable("hydro_generation", lb=0.0)
        hydro_spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.add_constraint(volume.out == volume.in_ + inflow - hydro_generation - hydro_spill)
        sp.add_constraint(thermal_generation + hydro_generation == 150.0)
        fuel_cost = [50.0, 100.0, 150.0]

        def modify(w: HydroRealization) -> None:
            sp.fix(inflow, w.inflow)
            sp.set_stage_objective(w.fuel_multiplier * fuel_cost[t - 1] * thermal_generation)

        sp.parameterize(modify, OMEGA, _markov_probability(markov_state))

    return sddp.MarkovianPolicyGraph(
        builder,
        transition_matrices=[[[1.0]], [[0.75, 0.25]], [[0.75, 0.25], [0.25, 0.75]]],
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def simulate_in_sample(
    model: sddp.PolicyGraph, n: int = 20, seed: int | None = None
) -> list[list[dict[str, Any]]]:
    return sddp.simulate(model, n, sampling_scheme=sddp.InSampleMonteCarlo(), seed=seed)


def out_of_sample_scheme(model: sddp.PolicyGraph) -> sddp.OutOfSampleMonteCarlo:
    """New transition probabilities (50% switching) and a deterministic final-stage noise."""

    def f(node: tuple[int, int]) -> Any:
        stage, markov_state = node
        if stage == 0:
            # Called from the root node: only the list of children.
            return [sddp.Noise((1, 1), 1.0)]
        elif stage == 3:
            # Final node: no children, one deterministic noise realization.
            children: list[sddp.Noise] = []
            noise_terms = [sddp.Noise(HydroRealization(75.0, 1.2), 1.0)]
            return children, noise_terms
        else:
            noise_terms = [
                sddp.Noise(w, p) for w, p in zip(OMEGA, _markov_probability(markov_state))
            ]
            children = [sddp.Noise((stage + 1, 1), 0.5), sddp.Noise((stage + 1, 2), 0.5)]
            return children, noise_terms

    return sddp.OutOfSampleMonteCarlo(f, model)


def out_of_sample_scheme_insample_transition(model: sddp.PolicyGraph) -> sddp.OutOfSampleMonteCarlo:
    """Only the noise terms change; transitions are the in-sample ones."""

    def f(node: tuple[int, int]) -> list[sddp.Noise]:
        stage, markov_state = node
        if stage == 3:
            return [sddp.Noise(HydroRealization(65.0, 1.1), 1.0)]
        return [sddp.Noise(w, p) for w, p in zip(OMEGA, _markov_probability(markov_state))]

    return sddp.OutOfSampleMonteCarlo(f, model, use_insample_transition=True)


def simulate_historical(model: sddp.PolicyGraph) -> list[dict[str, Any]]:
    """One particular sequence of nodes and noises."""
    sims = sddp.simulate(
        model,
        sampling_scheme=sddp.Historical(
            [((1, 1), OMEGA[0]), ((2, 2), OMEGA[2]), ((3, 1), OMEGA[1])]
        ),
    )
    return sims[0]


HISTORICAL_SCENARIOS = [
    [
        ((1, 1), HydroRealization(65.0, 1.1)),
        ((2, 2), HydroRealization(10.0, 1.4)),  # Can be out-of-sample
        ((3, 1), HydroRealization(65.0, 1.1)),
    ],
    [
        ((1, 1), HydroRealization(65.0, 1.1)),
        ((2, 2), HydroRealization(100.0, 0.75)),
        ((3, 1), HydroRealization(0.0, 1.5)),
    ],
]


def historical_sequential() -> sddp.Historical:
    return sddp.Historical(HISTORICAL_SCENARIOS)


def historical_probabilistic() -> sddp.Historical:
    return sddp.Historical(HISTORICAL_SCENARIOS, [0.3, 0.7])


# ================================================================== use_multithreading
def build_twelve_stage_model() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=1.0)
        sp.set_stage_objective(x.out)

    return sddp.LinearPolicyGraph(builder, stages=12, lower_bound=0.0, optimizer=sddp.HiGHS)


def train_threaded(
    model: sddp.PolicyGraph,
    num_threads: int | None = None,
    iteration_limit: int = 10,
    **kwargs: Any,
) -> None:
    sddp.train(
        model,
        iteration_limit=iteration_limit,
        log_every_iteration=True,
        parallel_scheme=sddp.Threaded(num_threads),
        **kwargs,
    )


def simulate_threaded_with_thread_ids(
    model: sddp.PolicyGraph, n: int = 100, num_threads: int | None = None
) -> list[list[dict[str, Any]]]:
    """``custom_recorders = Dict(:thread_id => sp -> Threads.threadid())``."""
    return sddp.simulate(
        model,
        n,
        parallel_scheme=sddp.Threaded(num_threads),
        custom_recorders={"thread_id": lambda sp: threading.get_ident()},
    )


# ============================================================ write_subproblems_to_file
def write_parameterized_subproblem(model: sddp.PolicyGraph, omega: float, filename: str) -> str:
    """``SDDP.parameterize(model[1], ω); SDDP.write_subproblem_to_file(model[1], filename)``."""
    sddp.parameterize(model[1], omega)
    sddp.write_subproblem_to_file(model[1], filename)
    with open(filename) as io:
        return io.read()


_TERM = re.compile(r"([+-]?\s*(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)?\s*([A-Za-z_][\w\[\],.]*)")


def _parse_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for coef, var in _TERM.findall(text):
        coef = coef.replace(" ", "")
        if coef in ("", "+"):
            c = 1.0
        elif coef == "-":
            c = -1.0
        else:
            c = float(coef)
        terms[var] = terms.get(var, 0.0) + c
    return terms


def parse_lp_file(text: str) -> dict[str, Any]:
    """A small parser for the LP files HiGHS writes: objective sense and coefficients, the
    rows (``name -> (coefficients, sense, rhs)``) and the variable bounds (``lb, ub``)."""
    sense = None
    objective: dict[str, float] = {}
    rows: dict[str, tuple[dict[str, float], str, float]] = {}
    bounds: dict[str, tuple[float, float]] = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("\\"):
            continue
        key = line.lower()
        if key in ("min", "minimize", "minimise", "max", "maximize", "maximise"):
            sense, section = key[:3], "obj"
            continue
        if key in ("st", "s.t.", "subject to", "such that"):
            section = "rows"
            continue
        if key == "bounds":
            section = "bounds"
            continue
        if key in ("general", "integer", "binary", "end"):
            section = key
            continue
        if section == "obj":
            objective.update(_parse_terms(line.split(":", 1)[1] if ":" in line else line))
        elif section == "rows":
            name, body = line.split(":", 1)
            m = re.match(r"(.*?)\s*(<=|>=|=)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$", body.strip())
            assert m is not None, line
            rows[name.strip()] = (_parse_terms(m.group(1)), m.group(2), float(m.group(3)))
        elif section == "bounds":
            num = r"[+-]?(?:\d+\.?\d*|\.\d+|inf(?:inity)?)(?:[eE][+-]?\d+)?"
            if line.endswith(" free"):
                bounds[line[:-5].strip()] = (-math.inf, math.inf)
            elif m := re.match(rf"({num})\s*<=\s*(\S+)\s*<=\s*({num})$", line):
                bounds[m.group(2)] = (float(m.group(1)), float(m.group(3)))
            elif m := re.match(rf"(\S+)\s*=\s*({num})$", line):
                bounds[m.group(1)] = (float(m.group(2)), float(m.group(2)))
            elif m := re.match(rf"(\S+)\s*>=\s*({num})$", line):
                bounds[m.group(1)] = (float(m.group(2)), math.inf)
            elif m := re.match(rf"(\S+)\s*<=\s*({num})$", line):
                bounds[m.group(1)] = (-math.inf, float(m.group(2)))
            elif m := re.match(rf"({num})\s*<=\s*(\S+)$", line):
                bounds[m.group(2)] = (float(m.group(1)), math.inf)
    return {"sense": sense, "objective": objective, "rows": rows, "bounds": bounds}


# ================================================================================ main
def main() -> None:  # pragma: no cover - reproduces the pages' printed output
    import os
    import tempfile

    tmp = tempfile.mkdtemp()
    print("# access_previous_variables")
    for build in (build_first_stage_capacity_model, build_pipeline_lag_model):
        model = build()
        sddp.train(model, iteration_limit=20, seed=1, **KW)
        print(build.__name__, sddp.calculate_bound(model))
    model = build_stochastic_lead_time_model()
    sddp.train(model, iteration_limit=20, seed=1, **KW)
    print("stochastic lead time (as written)", sddp.calculate_bound(model))
    print("# add_a_custom_cut")
    print(write_cuts_after_one_iteration(os.path.join(tmp, "cuts.json")))
    print(bound_with_custom_cut(os.path.join(tmp, "new_cuts.json")))
    print("# add_a_multidimensional_state_variable")
    build_multidimensional_state_model()
    print("# add_a_risk_measure")
    for train in (
        train_with_risk_measure,
        train_with_risk_measure_dict,
        train_with_risk_measure_function,
    ):
        model = build_risk_measure_model()
        train(model, seed=1)
        print(train.__name__, sddp.calculate_bound(model))
    print("# add_integrality")
    model = build_integrality_model()
    train_with_bandit_duality(model, iteration_limit=10, seed=1)
    model = build_integrality_model()
    train_with_conic_duality_optimizer(model, iteration_limit=10, seed=1, **KW)
    print("conic with optimizer", sddp.calculate_bound(model))
    print("# add_multidimensional_noise")
    for s in sddp.simulate(build_simple_multidimensional_noise_model(), 1, seed=1)[0]:
        print("ω is:", s["noise_term"])
    for s in sddp.simulate(build_vector_noise_model(*finite_product_distribution()), 1, seed=1)[0]:
        print("ω is:", s["noise_term"])
    for s in sddp.simulate(
        build_vector_noise_model(*sampled_product_distribution(seed=1)), 1, seed=1
    )[0]:
        print("ω is:", s["noise_term"])
    print("# add_noise_in_the_constraint_matrix")
    print(build_constraint_matrix_noise_model())
    print("# choose_a_stopping_rule")
    for train_fn in (
        train_iteration_limit,
        train_bound_stalling,
        train_time_limit_or_bound_stalling,
    ):
        print(train_fn.__name__, train_fn(build_risk_measure_model(), seed=1))
    print("# create_a_belief_state")
    print(create_belief_graph())
    print("# create_a_general_policy_graph")
    before, graph = linear_graph_example()
    print(before, graph, unicyclic_graph_example(), markovian_graph_example(), sep="\n")
    before, graph = general_graph_example()
    print(before, graph, sep="\n")
    create_policy_graph_from_graph()
    print(len(simulate_fixed_depth(build_cyclic_model(), seed=1)[0]), "stages simulated")
    model = build_simulator_markovian_model(seed=1)
    sddp.train(model, iteration_limit=5, seed=1, **KW)
    print("simulator graph bound", sddp.calculate_bound(model))
    print("# deterministic_equivalent")
    det, obj = solve_deterministic_equivalent(build_deterministic_equivalent_model())
    print(obj, det.num_variables(), det.num_constraints())
    print("# implement_a_par_model")
    model = build_par_model()
    print(simulate_par_model(model, seed=1))
    print(simulate_par_model(model, par_historical_scheme()))
    model = build_par_model_with_noise_terms()
    print(simulate_par_model(model, seed=1))
    print(simulate_par_model(model, par_noise_term_scheme()))
    print("# improve_computational_performance")
    train_single_cut(build_two_stage_model(), print_level=0)
    train_multi_cut(build_two_stage_model(), print_level=0)
    print(len(train_and_simulate_threaded(build_two_stage_model(), print_level=0)), "simulations")
    print("# simulate_using_a_different_sampling_scheme")
    model = build_markov_hydro_model()
    sddp.train(model, iteration_limit=10, seed=1, **KW)
    sims = simulate_in_sample(model, seed=1)
    print(sorted({s["noise_term"] for sim in sims for s in sim}))
    print(
        sddp.simulate(model, 1, sampling_scheme=out_of_sample_scheme(model), seed=1)[0][2][
            "noise_term"
        ]
    )
    print(
        sddp.simulate(
            model, 1, sampling_scheme=out_of_sample_scheme_insample_transition(model), seed=1
        )[0][2]["noise_term"]
    )
    print([s["node_index"] for s in simulate_historical(model)])
    print(historical_sequential())
    print(historical_probabilistic())
    print("# use_multithreading")
    model = build_twelve_stage_model()
    train_threaded(model, 4, print_level=0)
    sims = simulate_threaded_with_thread_ids(model, num_threads=4)
    print([sims[i][0]["thread_id"] for i in (0, 25, 50, 75)])
    print("# write_subproblems_to_file")
    model = build_deterministic_equivalent_model()
    for omega in (1.1, 3.3):
        print(write_parameterized_subproblem(model, omega, os.path.join(tmp, "subproblem.lp")))


if __name__ == "__main__":
    main()
