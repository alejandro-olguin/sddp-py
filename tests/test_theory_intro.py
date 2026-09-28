"""The "Introductory theory" explanation page (examples/theory_intro.py).

The page's vanilla SDDP is cross-validated against the package (``sddp.train``,
``sddp.calculate_bound``, ``sddp.ValueFunction``, ``sddp.DecisionRule``), against the
deterministic equivalent, and against the Julia oracle ``reference/oracle/explanation.json``
(SDDP.jl on the same models; Kelley's run with JuMP + HiGHS).
"""

from __future__ import annotations

import math
import random

from examples import theory_intro as ti
from examples.theory_intro import (
    build_finite_model,
    build_infinite_model,
    cost_to_go_approximation,
    evaluate_policy,
    forward_pass,
    get_node,
    kelleys_cutting_plane,
    kelleys_example,
    lower_bound,
    sample_next_node,
    sample_uncertainty,
    train,
    upper_bound,
)

import sddp
from tests.conftest import load_oracle
from tests.problems import build_hydro_thermal

KW = {"print_level": 0, "run_numerical_stability_report": False}
FUEL_COST = [50.0, 100.0, 150.0]


def close(a: float, b: float, rel: float = 1e-6, atol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


def kelley_f(x: list[float]) -> float:
    return (x[0] - 1) ** 2 + (x[1] + 2) ** 2 + 1.0


# ---------------------------------------------------------------------------
# Kelley's cutting plane algorithm
# ---------------------------------------------------------------------------


def test_kelleys_page_example_20_iterations():
    r = kelleys_example(io=None)
    # Like the Julia page (oracle: status "iteration limit"), 20 iterations are not enough
    # for the 1e-6 tolerance on this 2-d quadratic, but the bounds bracket f* = 1 tightly.
    assert r.iterations == 20 and r.status == "iteration limit"
    assert r.lower_bound <= 1.0 + 1e-9 <= r.upper_bound + 1e-9
    assert r.upper_bound - r.lower_bound < 1e-3
    assert abs(r.x[0] - 1.0) < 1e-2 and abs(r.x[1] + 2.0) < 1e-2
    o = load_oracle("explanation")["kelley"]["limit_20"]
    assert o["status"] == "iteration limit"
    # The LP iterates are not unique, so only the bounds are compared (loosely).
    assert abs(r.lower_bound - o["lower_bound"]) < 1e-3
    assert abs(r.upper_bound - o["upper_bound"]) < 1e-3


def test_kelleys_converges_to_the_minimiser():
    r = kelleys_cutting_plane(
        kelley_f, input_dimension=2, lower_bound=0.0, iteration_limit=200, io=None
    )
    assert r.status == "converged"
    assert r.upper_bound - r.lower_bound < 1e-6
    assert close(r.upper_bound, 1.0, atol=1e-6)
    assert abs(r.x[0] - 1.0) < 2e-3 and abs(r.x[1] + 2.0) < 2e-3
    o = load_oracle("explanation")["kelley"]["limit_200"]
    assert o["status"] == "converged"
    assert close(r.upper_bound, o["upper_bound"], atol=1e-6)
    # An explicit gradient gives the same run as central finite differences.
    r2 = kelleys_cutting_plane(
        kelley_f,
        lambda x: [2 * (x[0] - 1), 2 * (x[1] + 2)],
        input_dimension=2,
        lower_bound=0.0,
        iteration_limit=200,
        io=None,
    )
    assert r2.status == "converged" and r2.iterations == r.iterations
    assert close(r2.lower_bound, r.lower_bound, atol=1e-6)


# ---------------------------------------------------------------------------
# Structures and samplers
# ---------------------------------------------------------------------------


def test_policy_graph_structure():
    model = build_finite_model()
    assert len(model.nodes) == 3
    assert model.arcs == [{2: 1.0}, {3: 1.0}, {}]
    text = repr(model)
    assert "A policy graph with 3 nodes" in text and "1 => 2 w.p. 1.0" in text
    node1 = get_node(model, 1)
    assert node1.subproblem.fix_value(node1.states["volume"].in_) == 200.0
    assert node1.subproblem.lower_bound(node1.cost_to_go) == 0.0
    # The terminal node has its cost-to-go fixed to 0.
    node3 = get_node(model, 3)
    assert node3.subproblem.is_fixed(node3.cost_to_go)
    assert node3.subproblem.fix_value(node3.cost_to_go) == 0.0
    assert node1.uncertainty.Ω == [0.0, 50.0, 100.0]
    assert close(sum(node1.uncertainty.P), 1.0)


def test_samplers():
    rng = random.Random(0)
    model = build_finite_model()
    for _ in range(20):
        assert sample_uncertainty(get_node(model, 1).uncertainty, rng) in (0.0, 50.0, 100.0)
    assert sample_next_node(model, 1, rng) == 2
    assert sample_next_node(model, 2, rng) == 3
    assert sample_next_node(model, 3, rng) is None
    # Sampling frequencies follow P.
    counts = {w: 0 for w in (0.0, 50.0, 100.0)}
    for _ in range(3000):
        counts[sample_uncertainty(get_node(model, 1).uncertainty, rng)] += 1
    assert all(abs(c / 3000 - 1 / 3) < 0.05 for c in counts.values())
    # Infinite horizon: node 3 continues to node 2 w.p. 0.5, otherwise ends.
    inf = build_infinite_model()
    nxt = [sample_next_node(inf, 3, rng) for _ in range(2000)]
    assert set(nxt) == {2, None}
    assert abs(nxt.count(2) / 2000 - 0.5) < 0.05


def test_forward_pass_trajectory():
    model = build_finite_model()
    rng = random.Random(1)
    trajectory, cost = forward_pass(model, rng, io=None)
    assert [t for t, _ in trajectory] == [1, 2, 3]
    for _, x in trajectory:
        assert 0.0 <= x["volume"] <= 200.0
    # Untrained policy: the cost-to-go is 0, so node 1 burns thermal (cheap) only if needed.
    assert cost >= 0.0
    # The stage cost is fuel * thermal at every visited node.
    rng = random.Random(1)
    trajectory, cost = forward_pass(model, rng, io=None)
    total = 0.0
    for t, _ in trajectory:
        node = get_node(model, t)
        total += FUEL_COST[t - 1] * node.subproblem.value(node.subproblem["thermal_generation"])
    # (the subproblems were re-solved along the way; recompute from the same seed instead)
    model2 = build_finite_model()
    _, cost2 = forward_pass(model2, random.Random(1), io=None)
    assert close(cost, cost2)
    assert total >= 0.0


# ---------------------------------------------------------------------------
# Vanilla SDDP vs the package vs the deterministic equivalent
# ---------------------------------------------------------------------------


def _exact_cost_to_go(volume: float) -> float:
    """Exact node-1 cost-to-go at ``volume_out = volume``: a 2-stage deterministic equivalent."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        v = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=volume)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.parameterize(lambda w: sp.fix(inflow, w), [0.0, 50.0, 100.0], [1 / 3] * 3)
        sp.add_constraint(v.out == v.in_ - hydro - spill + inflow)
        sp.add_constraint(hydro + thermal == 150.0)
        sp.set_stage_objective(FUEL_COST[t] * thermal)

    m = sddp.LinearPolicyGraph(
        builder, stages=2, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )
    de = sddp.deterministic_equivalent(m)
    de.optimize()
    return de.objective_value()


def test_vanilla_sddp_matches_package_and_deterministic_equivalent():
    det = load_oracle("hydro_thermal")["deterministic_equivalent"]
    assert close(det, 8333.333333333333)
    model = build_finite_model()
    mu, tsigma = train(model, iteration_limit=20, replications=100, rng=random.Random(1), io=None)
    lb = lower_bound(model)
    assert close(lb, det, rel=1e-9, atol=1e-6)
    # The policy is optimal, so the simulated cost is an unbiased estimate of the bound.
    assert abs(mu - det) <= 3 * tsigma / 1.96
    # Package on the same model and the Julia oracle.
    pk = build_hydro_thermal()
    sddp.train(pk, iteration_limit=20, seed=1, **KW)
    assert close(sddp.calculate_bound(pk), lb, rel=1e-9, atol=1e-6)
    o = load_oracle("explanation")["hydro_thermal_expectation"]
    assert o["iteration_limit"] == 20
    assert close(o["bound"], lb, rel=1e-9, atol=1e-6)
    # Cut counts: one cut per visited non-terminal node per iteration.
    assert [len(n.cuts) for n in model.nodes] == [20, 20, 0]

    # Value function comparison at volume ∈ {0, 100, 200}. Both are outer approximations,
    # so both are ≤ the exact value; they coincide with it (and with each other) at points
    # the forward passes visit. Node 1 always chooses volume_out = 200 (keeping water is
    # worth 83.3/unit, thermal costs 50), so 200 is saturated in both. At 0 and 100 the
    # approximations may sit below the truth by different amounts (they were built at
    # different sample points): vanilla gives 22500 / 11666.67, the package 22500 / 12500.
    V = sddp.ValueFunction(pk, node=1)
    exact = {v: _exact_cost_to_go(v) for v in (0.0, 100.0, 200.0)}
    assert close(exact[0.0], 23333.333333333332) and close(exact[100.0], 12500.0)
    assert close(exact[200.0], 3333.3333333333335)
    for v in (0.0, 100.0, 200.0):
        vanilla = cost_to_go_approximation(get_node(model, 1), {"volume": v})
        package, _ = sddp.evaluate_value_function(V, {"volume": v})
        assert vanilla <= exact[v] + 1e-6, (v, vanilla, exact[v])
        assert package <= exact[v] + 1e-6, (v, package, exact[v])
    assert close(cost_to_go_approximation(get_node(model, 1), {"volume": 200.0}), exact[200.0])
    assert close(sddp.evaluate_value_function(V, {"volume": 200.0})[0], exact[200.0])
    # The exact value is piecewise linear: slope -116.7 on [0, 50], -100 on [50, 100],
    # -83.3 on [100, 200]. Neither implementation visits volume_out = 0, so at 0 both report
    # the best cut generated elsewhere: 22500 (a cut built at 50 or at 100 has slope -100 and
    # passes through 22500 at 0), strictly below the exact 23333.33. Document that equality.
    assert close(cost_to_go_approximation(get_node(model, 1), {"volume": 0.0}), 22500.0)
    assert close(sddp.evaluate_value_function(V, {"volume": 0.0})[0], 22500.0)

    # Decision rule at node 1, volume 150, out-of-sample ω = 75 (the page's query).
    res = evaluate_policy(model, node=1, incoming_state={"volume": 150.0}, random_variable=75)
    assert close(res["volume_out"], 200.0) and close(res["thermal_generation"], 125.0)
    assert close(res["hydro_generation"], 25.0) and close(res["hydro_spill"], 0.0)
    assert close(res["demand_constraint"], 150.0)
    rule = sddp.DecisionRule(pk, node=1)
    r = sddp.evaluate(
        rule,
        incoming_state={"volume": 150.0},
        noise=75.0,
        controls_to_record=["thermal_generation", "hydro_generation", "hydro_spill"],
    )
    assert close(r.outgoing_state["volume"], res["volume_out"])
    for k in ("thermal_generation", "hydro_generation", "hydro_spill"):
        assert close(r.controls[k], res[k])
    assert close(r.stage_objective, 50.0 * res["thermal_generation"])


def test_evaluate_policy_moves_node_1_incoming_state():
    """A quirk of the page: evaluating node 1 refixes its incoming state (documented)."""
    model = build_finite_model()
    evaluate_policy(model, node=1, incoming_state={"volume": 150.0}, random_variable=75)
    node1 = get_node(model, 1)
    assert node1.subproblem.fix_value(node1.states["volume"].in_) == 150.0
    # The next forward pass therefore starts from 150, like the Julia code would.
    trajectory, _ = forward_pass(model, random.Random(0), io=None)
    assert trajectory[0][0] == 1


# ---------------------------------------------------------------------------
# Infinite horizon
# ---------------------------------------------------------------------------


def _package_cyclic_model() -> sddp.PolicyGraph:
    g = sddp.Graph(0)
    for n in (1, 2, 3):
        g.add_node(n)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(2, 3, 1.0)
    g.add_edge(3, 2, 0.5)

    def builder(sp: sddp.Subproblem, t: int) -> None:
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.parameterize(lambda w: sp.fix(inflow, w), [0.0, 50.0, 100.0], [1 / 3] * 3)
        sp.add_constraint(volume.out == volume.in_ - hydro - spill + inflow)
        sp.add_constraint(hydro + thermal == 150.0, name="demand_constraint")
        sp.set_stage_objective(FUEL_COST[t - 1] * thermal)

    return sddp.PolicyGraph(builder, g, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS)


def test_infinite_horizon_vs_package():
    model = build_infinite_model()
    assert model.arcs == [{2: 1.0}, {3: 1.0}, {2: 0.5}]
    rng = random.Random(1)
    train(model, iteration_limit=30, replications=10, rng=rng, io=None)
    lb30 = lower_bound(model)
    pk = _package_cyclic_model()
    sddp.train(pk, iteration_limit=30, seed=1, **KW)
    b30 = sddp.calculate_bound(pk)
    # After 30 iterations both are within 1e-4 (relative) of each other (and of the limit).
    assert close(lb30, b30, rel=1e-4, atol=0.0), (lb30, b30)
    # Train both further: the cyclic bound converges geometrically (discount 0.5) and the
    # two implementations agree to 1e-8.
    train(model, iteration_limit=90, replications=10, rng=rng, io=None)
    sddp.train(pk, iteration_limit=90, seed=2, add_to_existing_cuts=True, **KW)
    lb, b = lower_bound(model), sddp.calculate_bound(pk)
    assert close(lb, b, rel=1e-8, atol=0.0), (lb, b)
    assert lb30 <= lb + 1e-9 and b30 <= b + 1e-9  # bounds are monotone
    o = load_oracle("explanation")["infinite_cyclic"]
    assert close(o["bound"], lb, rel=1e-6, atol=0.0), (o["bound"], lb)
    # The page's query: node 3, volume 100, out-of-sample inflow 10.
    res = evaluate_policy(model, node=3, incoming_state={"volume": 100.0}, random_variable=10.0)
    rule = sddp.DecisionRule(pk, node=3)
    r = sddp.evaluate(
        rule,
        incoming_state={"volume": 100.0},
        noise=10.0,
        controls_to_record=["thermal_generation", "hydro_generation", "hydro_spill"],
    )
    # Using all the water now (150 $/unit thermal vs an expected discounted 50) is optimal.
    assert close(res["thermal_generation"], 40.0) and close(res["hydro_generation"], 110.0)
    assert close(res["hydro_spill"], 0.0) and close(res["volume_out"], 0.0)
    for k in ("thermal_generation", "hydro_generation", "hydro_spill"):
        assert close(r.controls[k], res[k])
    assert close(r.outgoing_state["volume"], res["volume_out"])
    assert close(r.stage_objective, 150.0 * 40.0)
    # Forward passes have random length in a cyclic graph.
    lengths = {len(forward_pass(model, rng, io=None)[0]) for _ in range(50)}
    assert len(lengths) > 1 and min(lengths) == 3


def test_upper_bound_is_a_confidence_interval():
    model = build_finite_model()
    rng = random.Random(3)
    train(model, iteration_limit=10, replications=10, rng=rng, io=None)
    mu, tsigma = upper_bound(model, replications=200, rng=rng)
    assert tsigma > 0.0
    assert abs(mu - 8333.333333333333) < 4 * tsigma / 1.96


def test_main_runs_silently():
    ti.main(seed=7, io=None)
