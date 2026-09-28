"""Tutorial page "Example: the milk producer" (``examples/tutorial_milk_producer.py``) vs the
Julia oracle ``reference/oracle/tutorials_b.json`` (key ``milk_producer``)."""

from __future__ import annotations

import math
import random

import pytest
from examples import tutorial_milk_producer as mp

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


def test_simulator():
    sim = mp.make_simulator(random.Random(1))
    assert len(mp.RESIDUALS) == 19 and close(sum(mp.RESIDUALS), 0.0)
    for _ in range(50):
        s = sim()
        assert len(s) == 12 and all(3.0 <= p <= 9.0 for p in s)
    # Mean reversion: the average price drifts up from 4.5 towards 6.
    means = [sum(sim()[t] for _ in range(300)) / 300 for t in (0, 11)]
    assert 4.3 < means[0] < 4.9 and means[0] < means[1] < 6.0


@pytest.fixture(scope="module")
def fitted():
    rng = random.Random(1)
    sim = mp.make_simulator(rng)
    graph = mp.build_graph(sim, budget=30, scenarios=10_000)
    return sim, graph


def test_markovian_graph_from_simulator(fitted):
    _, graph = fitted
    assert graph.root_node == (0, 0.0)
    nodes = [n for n in graph.nodes if n != graph.root_node]
    assert len(nodes) == 30  # budget = number of nodes
    assert {t for t, _ in nodes} == set(range(1, 13))
    assert all(3.0 <= price <= 9.0 for _, price in nodes)
    for node, arcs in graph.nodes.items():
        if arcs:
            assert close(sum(p for _, p in arcs), 1.0)
            assert all(child[0] == node[0] + 1 for child, _ in arcs)
        else:
            assert node[0] == 12
    # More nodes in the volatile late months than in month 1.
    assert sum(1 for t, _ in nodes if t == 1) <= sum(1 for t, _ in nodes if t == 12)


def test_graph_plot(fitted, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    sim, graph = fitted
    fig = mp.plot_price_process(sim, graph, n=20, filename=str(tmp_path / "graph.png"))
    assert (tmp_path / "graph.png").stat().st_size > 0
    n_edges = sum(len(arcs) for arcs in graph.nodes.values())
    assert len(fig.axes[0].lines) == 20 + n_edges


def test_model_train_and_simulate(fitted, tmp_path):
    sim, graph = fitted
    model = mp.build_milk_producer(graph)
    assert len(model.nodes) == 30
    node = model[(12, max(p for t, p in graph.nodes if t == 12))]
    assert [n.term for n in node.noise_terms] == [(node.index[1], p) for p in mp.OMEGA_PRODUCTION]
    assert "u_forward_sell" in node.subproblem
    mp.train_milk_producer(model, sim, iteration_limit=40, seed=1, **KW)
    assert sddp.termination_status(model) == "iteration_limit"
    log = model.most_recent_training_results.log
    assert len(log) == 40 and all(math.isfinite(entry.bound) for entry in log)
    # Max problem: the (upper) bound decreases monotonically.
    assert all(b.bound <= a.bound + 1e-6 for a, b in zip(log, log[1:]))
    sims = mp.simulate_milk_producer(model, sim, 200, seed=2)
    assert len(sims) == 200 and all(len(s) == 12 for s in sims)
    lattice_12 = sorted(p for t, p in graph.nodes if t == 12)
    stage = sims[0][11]
    t, node_price = stage["node_index"]
    sim_price, production = stage["noise_term"]
    assert t == 12 and node_price in lattice_12
    assert production in mp.OMEGA_PRODUCTION and 3.0 <= sim_price <= 9.0
    # The node is the closest lattice value to the out-of-sample simulator price...
    assert node_price == min(lattice_12, key=lambda p: abs(p - sim_price))
    # ...which is (almost surely) not itself a lattice value.
    assert any(s[11]["noise_term"][0] != s[11]["node_index"][1] for s in sims)
    for s in sims:
        for stage in s:
            tt, node_price = stage["node_index"]
            price, prod = stage["noise_term"]
            assert node_price == min(
                (p for t2, p in graph.nodes if t2 == tt), key=lambda p: abs(p - price)
            )
            # The out-of-sample price is what the stage objective was priced with.
            revenue = (
                price * (stage["u_spot_sell"] - 1.5 * stage["u_spot_buy"])
                + (price * 1.05 - 0.01) * stage["u_forward_sell"]
            )
            assert close(stage["stage_objective"], revenue, atol=1e-6)
            if tt > 8:
                assert stage["u_forward_sell"] == 0.0
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    mp.plot_policy(sims, str(tmp_path / "policy.png"))
    assert (tmp_path / "policy.png").stat().st_size > 0


def _julia_graph(d: dict) -> sddp.Graph:
    root = tuple(d["root"])
    nodes = [tuple(n) for n in d["nodes"]]
    edges = [((tuple(p), tuple(c)), pr) for p, c, pr in d["edges"]]
    return sddp.Graph.from_edges(root, nodes, edges)


def test_parity_on_julia_graph():
    """Train on the graph SDDP.jl fitted (its RNG differs from Python's) and compare converged
    bounds: risk-neutral first (tight), then the page's risk measure."""
    d = load_oracle("tutorials_b")["milk_producer"]
    graph = _julia_graph(d["graph"])
    assert len(graph.nodes) == 31
    sim = mp.make_simulator(random.Random(1))
    model = mp.build_milk_producer(graph)
    sddp.train(
        model,
        iteration_limit=400,
        seed=1,
        sampling_scheme=sddp.SimulatorSamplingScheme(sim),
        **KW,
    )
    bound = sddp.calculate_bound(model)
    assert close(bound, d["expectation_bound_400"], rel=EXPECTATION_RTOL), (
        bound,
        d["expectation_bound_400"],
    )
    model = mp.build_milk_producer(graph)
    mp.train_milk_producer(model, sim, iteration_limit=400, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    assert close(bound, d["bound_400"], rel=RISK_RTOL), (bound, d["bound_400"], d["bound_100"])


# Observed after 400 iterations each: 9.94075 vs 9.94034 (4e-5), 7.88170 vs 7.88125 (6e-5).
EXPECTATION_RTOL = 1e-3
RISK_RTOL = 2e-3
