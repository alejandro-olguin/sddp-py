"""Tutorial pages "Example: deterministic to stochastic" and "Example: capacity expansion models"
(``examples/tutorial_reservoir.py``) vs the Julia oracle ``reference/oracle/tutorials_b.json``."""

from __future__ import annotations

import math
import random
import statistics
import warnings

import pytest
from examples import tutorial_reservoir as tr

import sddp
from sddp.solver.model import HiGHS as HiGHS_LP
from sddp.solver.model import Model, Sense
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}
# Page-faithful training (cut selection on) makes the bound occasionally decrease a little
# in both SDDP.jl and the port (see ``_assert_bound_log``); the tolerance is relative.
CUT_SELECTION_DROP_RTOL = 0.05


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


def _assert_bound_log(model, rtol=CUT_SELECTION_DROP_RTOL, monotone=True):
    """Bounds are finite; with ``monotone`` they are non-decreasing up to small drops caused
    by cut selection. Cyclic graphs trained with cut selection (the page's default) can show
    large early drops in both SDDP.jl and the port (see
    ``test_capex_cyclic_monotone_without_cut_selection`` for the strict property)."""
    log = model.most_recent_training_results.log
    bounds = [entry.bound for entry in log]
    assert all(math.isfinite(b) for b in bounds)
    assert all(b >= 0.0 for b in bounds)  # lower_bound = 0.0, costs are non-negative
    if monotone:
        assert bounds[-1] >= (1 - rtol) * max(bounds)
        for a, b in zip(bounds, bounds[1:]):
            assert b >= a * (1 - rtol) - 1e-6, (a, b)
    return bounds


# ======================================================== deterministic to stochastic
def test_data():
    assert tr.T == 52
    assert tr.DATA["inflow"][:3] == [3.0, 2.0, 3.0]
    assert tr.DATA["demand"][-1] == 7.0 and tr.DATA["cost"][-1] == 17.8
    d = load_oracle("tutorials_b")["data"]
    assert tr.DATA["inflow"] == d["inflow"]
    assert tr.DATA["demand"] == d["demand"]
    assert tr.DATA["cost"] == d["cost"]


def test_deterministic_lp():
    d = load_oracle("tutorials_b")["reservoir_lp"]
    lp = tr.solve_deterministic_lp()
    assert lp["termination_status"].name == "OPTIMAL"
    assert close(lp["objective"], d["objective"])
    assert close(lp["objective"], 682.91, atol=1e-2)  # the number printed on the page
    assert lp["x_storage"][0] == 300.0 and len(lp["x_storage"]) == 53
    # The trace has the same cost as Julia's (the LP may have ties).
    trace_cost = sum(c * u for c, u in zip(tr.DATA["cost"], lp["u_thermal"]))
    assert close(trace_cost, d["objective"])
    assert all(close(a, b, atol=1e-6) for a, b in zip(lp["u_thermal"], d["u_thermal"]))
    assert all(close(a, b, atol=1e-6) for a, b in zip(lp["x_storage"], d["x_storage"]))


def test_deterministic_sddp_matches_lp():
    lp = tr.solve_deterministic_lp()
    r = tr.run_deterministic_sddp(iteration_limit=10, seed=1, **KW)
    assert close(r["bound"], lp["objective"])
    d = load_oracle("tutorials_b")["reservoir_deterministic"]
    assert close(r["bound"], d["bound_10"])
    sims = r["simulations"]
    assert len(sims) == 1 and len(sims[0]) == 52
    assert set(sims[0][9]) >= {"node_index", "noise_term", "stage_objective", "x_storage"}
    assert sims[0][9]["node_index"] == 10
    # The simulated traces solve the LP too: same cost and the same storage trajectory.
    trace_cost = sum(c * u for c, u in zip(tr.DATA["cost"], r["u_thermal"]))
    assert close(trace_cost, lp["objective"])
    assert all(close(a, b, atol=1e-6) for a, b in zip(r["x_storage"], lp["x_storage"][1:]))
    assert all(close(a, b, atol=1e-6) for a, b in zip(r["u_thermal"], lp["u_thermal"]))
    assert all(close(a, b, atol=1e-6) for a, b in zip(r["u_flow"], lp["u_flow"]))
    assert all(close(a, b, atol=1e-6) for a, b in zip(r["x_storage"], d["x_storage_out"]))
    # Balance: water in = water out + thermal makes up the demand.
    for t, s in enumerate(sims[0]):
        assert close(s["u_flow"] + r["u_thermal"][t], tr.DATA["demand"][t])


def test_stochastic_sddp(tmp_path):
    html = tmp_path / "spaghetti_plot.html"
    r = tr.run_stochastic_sddp(iteration_limit=100, seed=1, spaghetti_file=str(html), **KW)
    _assert_bound_log(r["model"])
    assert html.stat().st_size > 0
    assert repr(r["spaghetti"]) == "A spaghetti plot with 100 scenarios and 52 stages."
    assert [p["title"] for p in r["spaghetti"].data] == ["Storage", "Hydro", "Inflow"]
    sims = r["simulations"]
    assert len(sims) == 100 and all(len(s) == 52 for s in sims)
    for sim in sims:
        for t, s in enumerate(sim):
            assert s["noise_term"] in tr.OMEGA
            assert close(s["omega_inflow"], tr.DATA["inflow"][t] + s["noise_term"])
    sddp.plot_graph(r["model"], str(tmp_path / "model_hydro2.html"))
    assert (tmp_path / "model_hydro2.html").stat().st_size > 0


def test_stochastic_sddp_parity():
    d = load_oracle("tutorials_b")["reservoir_stochastic"]
    model = tr.build_stochastic_sddp()
    sddp.train(model, iteration_limit=300, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    # The 52-stage model is not fully converged at 300 iterations (Julia: 280.90 at 300,
    # 283.73 at 600), so compare like with like at 5e-3 and with the longer run at 2e-2.
    assert close(bound, d["bound_300"], rel=STOCHASTIC_RTOL), (bound, d["bound_300"])
    assert close(bound, d["bound_600"], rel=STOCHASTIC_RTOL_600), (bound, d["bound_600"])


STOCHASTIC_RTOL = 5e-3  # observed: 280.46 vs 280.90 (1.6e-3)
STOCHASTIC_RTOL_600 = 2e-2  # observed: 280.46 vs 283.73 (1.2e-2)
CYCLIC_RTOL = 1e-3  # observed: 93390.53 vs 93390.76 (2.5e-6) after 100 iterations each


def test_cyclic_graph_repr():
    assert repr(sddp.UnicyclicGraph(0.7, num_nodes=2)) == (
        "Root\n 0\nNodes\n 1\n 2\nArcs\n 0 => 1 w.p. 1.0\n 1 => 2 w.p. 1.0\n 2 => 1 w.p. 0.7"
    )
    g = sddp.UnicyclicGraph(0.95, num_nodes=tr.T)
    assert len(g.nodes) == 53 and g.nodes[52] == [(1, 0.95)]


def test_cyclic_sddp(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # cut-scaling warnings from the long cycle
        r = tr.run_cyclic_sddp(iteration_limit=20, replications=20, seed=1, **KW)
    _assert_bound_log(r["model"])
    # Free simulations stop only after node 52 (the dummy leaf has probability 0.05 there).
    assert len(r["free_lengths"]) == 3
    assert all(n > 0 and n % 52 == 0 for n in r["free_lengths"])
    assert r["fixed_lengths"] == [5 * tr.T] * 20
    sims = r["simulations"]
    assert [s["node_index"] for s in sims[0][:53]] == list(range(1, 53)) + [1]
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fig = tr.plot_storage_and_hydro(sims, str(tmp_path / "cyclic.png"))
    assert (tmp_path / "cyclic.png").stat().st_size > 0
    assert len(fig.axes) == 2


@pytest.mark.slow
def test_cyclic_sddp_parity():
    d = load_oracle("tutorials_b")["reservoir_cyclic"]
    model = tr.build_cyclic_sddp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sddp.train(model, iteration_limit=100, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    assert close(bound, d["bound_100"], rel=CYCLIC_RTOL), (bound, d["bound_100"], d["bound_300"])


# ================================================================ capacity expansion
FAST_CAPEX = [
    "capex_operational",
    "capex_invest_then_operate",
    "capex_multiple_investments",
    "capex_invest_operate_invest_operate",
    "capex_strategic_uncertainty",
]
# Converged bounds after 300 iterations (Python vs Julia): 1010.63 vs 1010.89 (2.6e-4),
# 1306.78 vs 1306.84 (5e-5), 591.656 vs 591.642 (2.4e-5).
CAPEX_RTOL = {
    "capex_operational": 1e-3,
    "capex_invest_then_operate": 1e-3,
    "capex_multiple_investments": 1e-3,
}


def _check_capex_simulations(name, sims, replications):
    assert len(sims) == replications
    lengths = {len(s) for s in sims}
    if name in ("capex_loop", "capex_strategic_uncertainty"):
        assert lengths == {5 * 52 + 2}
    elif name == "capex_epicycles":
        assert lengths == {3 + 3 * 3 * 52}
    elif name == "capex_operational":
        assert lengths == {52}
    elif name == "capex_invest_operate_invest_operate":
        assert lengths == {2 * 52 + 2}
    else:
        assert lengths == {53}
    for sim in sims:
        for s in sim:
            assert s["x_storage"].out >= -1e-6 and s["u_flow"] >= -1e-6
            if "x_reservoir_max" in s:
                assert s["x_storage"].out <= s["x_reservoir_max"].out + 1e-6
            if "x_flow_max" in s:
                assert s["u_flow"] <= s["x_flow_max"].out + 1e-6


@pytest.mark.parametrize("name", FAST_CAPEX)
def test_capex_page_run(name, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tr.run_capex(name, seed=1, replications=30, **KW)
    log = r["model"].most_recent_training_results.log
    assert len(log) == tr.CAPEX_MODELS[name][1]
    _assert_bound_log(r["model"])
    _check_capex_simulations(name, r["simulations"], 30)
    d = load_oracle("tutorials_b")[name]
    # Same ballpark as Julia after the page's iteration count (different sample paths).
    assert 0.5 * d["bound_100"] <= r["bound"] <= 1.5 * d["bound_100"], (r["bound"], d)
    if name == "capex_invest_then_operate":
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        fig = tr.plot_capex(r["simulations"], str(tmp_path / f"{name}.png"))
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert sum(ax.get_visible() for ax in fig.axes) == 3


@pytest.mark.parametrize("name", list(CAPEX_RTOL))
def test_capex_finite_parity(name):
    d = load_oracle("tutorials_b")[name]
    build, _ = tr.CAPEX_MODELS[name]
    model = build()
    sddp.train(model, iteration_limit=300, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    assert close(bound, d["bound_300"], rel=CAPEX_RTOL[name]), (bound, d["bound_300"])


@pytest.mark.parametrize("name", ["capex_loop", "capex_epicycles"])
def test_capex_cyclic_short_run(name):
    """The two slow (infinite-horizon) models with a short training run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tr.run_capex(name, iteration_limit=15, seed=1, replications=10, **KW)
    _assert_bound_log(r["model"], monotone=False)
    _check_capex_simulations(name, r["simulations"], 10)
    d = load_oracle("tutorials_b")[name]
    key = f"bound_{d['page_iterations']}"
    assert 0.0 < r["bound"] <= 1.2 * d[key]


@pytest.mark.parametrize("name", ["capex_loop", "capex_epicycles"])
def test_capex_cyclic_monotone_without_cut_selection(name):
    """Without cut selection the bound of a minimisation problem never decreases."""
    build, _ = tr.CAPEX_MODELS[name]
    model = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sddp.train(model, iteration_limit=12, seed=1, cut_deletion_minimum=10**9, **KW)
    bounds = _assert_bound_log(model, monotone=False)
    assert all(b >= a - 1e-6 for a, b in zip(bounds, bounds[1:])), bounds
    assert bounds[-1] > bounds[0]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["capex_loop", "capex_epicycles"])
def test_capex_cyclic_page_run(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tr.run_capex(name, seed=1, replications=100, **KW)
    _assert_bound_log(r["model"], monotone=False)
    _check_capex_simulations(name, r["simulations"], 100)
    d = load_oracle("tutorials_b")[name]
    key = f"bound_{d['page_iterations']}"
    assert 0.7 * d[key] <= r["bound"] <= 1.3 * d[key], (r["bound"], d[key])


def test_capex_graphs():
    g = tr.capex_loop_graph()
    assert g.nodes[2 * 52 + 2] == [(52 + 3, 0.95)]
    assert len(g.nodes) == 2 * 52 + 3
    g6 = tr.capex_strategic_graph()
    assert len(g6.nodes) == 1 + 2 + 3 * 52
    assert g6.nodes[("Y1", 52)] == [(("invest_2", 0), 0.9)]
    assert sorted(g6.nodes[("invest_2", 0)]) == [(("Y2_high", 1), 0.5), (("Y2_normal", 1), 0.5)]
    g7 = tr.epicycles_graph()
    assert len(g7.nodes) == 1 + 7 + 52
    p, T = tr.EPICYCLES_P, tr.EPICYCLES_T
    assert dict(g7.nodes[("inv", 0)]) == {
        ("inv_h", 0): p**T / 2,
        ("inv_l", 0): p**T / 2,
        ("op", 1): pytest.approx(T * (1 - p)),
    }
    assert g7.nodes[("op", 52)] == [(("op", 1), p)]
    assert sum(pr for _, pr in g7.nodes[("inv_h", 0)]) == pytest.approx(p**T + T * (1 - p))


def test_epicycles_scenario_and_historical():
    rng = random.Random(3)
    scen = tr.sample_epicycles_scenario(rng)
    assert len(scen) == 3 + 3 * 3 * 52
    assert scen[0] == (("inv", 0), None)
    inv_1 = scen[1 + 3 * 52][0]
    inv_2 = scen[2 + 6 * 52][0]
    assert inv_1[0] in ("inv_l", "inv_h") and inv_2[0] in (inv_1[0] + "l", inv_1[0] + "h")
    ops = [s for s in scen if s[0][0] == "op"]
    assert [s[0][1] for s in ops] == list(range(1, 53)) * 9
    assert all(s[1] in tr.OMEGA for s in ops)
    # Historical with None noise for the investment nodes is accepted by simulate.
    model = tr.build_capex_epicycles()
    sims = sddp.simulate(
        model, 2, ["x_scale"], sampling_scheme=sddp.Historical([scen, scen]), seed=1
    )
    assert [s["node_index"] for s in sims[0]] == [s[0] for s in scen]
    assert [s["noise_term"] for s in sims[0]] == [s[1] for s in scen]


def test_epicycles_parameterize_modifies_incoming_state_coefficient():
    """``set_normalized_coefficient(c_omega, x_scale.in_, ...)`` inside ``parameterize``: the
    inflow must equal ``x_scale.in * (inflow[t] + ω)`` in every operational node."""
    model = tr.build_capex_epicycles()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sddp.train(model, iteration_limit=5, seed=1, **KW)
    sims = sddp.simulate(
        model,
        3,
        ["x_scale", "omega_inflow", "u_flow", "u_thermal"],
        sampling_scheme=sddp.Historical(
            [tr.sample_epicycles_scenario(random.Random(i)) for i in range(3)]
        ),
    )
    scales = set()
    for sim in sims:
        for s in sim:
            name, t = s["node_index"]
            if t > 0:
                assert close(
                    s["omega_inflow"],
                    s["x_scale"].in_ * (tr.DATA["inflow"][t - 1] + s["noise_term"]),
                )
                assert close(
                    s["u_flow"] + s["u_thermal"], s["x_scale"].in_ * tr.DATA["demand"][t - 1]
                )
                scales.add(round(s["x_scale"].in_, 6))
            else:
                factor = 1.5 if name.endswith("h") else 0.8 if name.endswith("l") else 1.0
                assert close(s["x_scale"].out, factor * s["x_scale"].in_)
    assert len(scales) >= 2  # the scale really changes along the scenario tree


def _coefficient_lp(a: float, scales: list[float], probs: list[float], demand: float) -> float:
    """Deterministic equivalent (by hand) of ``_coefficient_model``."""
    m = Model(HiGHS_LP)
    s = m.add_variable("s", lb=0.5, ub=2.0)
    obj = 3.0 * s
    for k, (scale, p) in enumerate(zip(scales, probs)):
        u = m.add_variable(f"u[{k}]", lb=0.0)
        w = m.add_variable(f"w[{k}]")
        m.add_constraint(w - (a + scale) * s == 0.0)
        m.add_constraint(u >= demand - w)
        obj = obj + p * 10.0 * u
    m.set_objective(obj, Sense.MIN)
    m.optimize()
    return m.objective_value()


def test_set_normalized_coefficient_on_incoming_state_builds_valid_cuts():
    """Regression: modifying the coefficient of ``x.in_`` in ``parameterize`` (as the epicycles
    model does) must give cuts that converge to the deterministic-equivalent value."""
    a, scales, probs, demand = 2.0, [-1.0, 0.0, 3.0], [0.3, 0.4, 0.3], 6.0

    def builder(sp, t):
        s = sp.add_state("s", initial_value=1.0)
        if t == 1:
            sp.add_constraint(s.out >= 0.5)
            sp.add_constraint(s.out <= 2.0)
            sp.set_stage_objective(3.0 * s.out)
        else:
            u = sp.add_variable("u", lb=0.0)
            w = sp.add_variable("w")
            sp.add_constraint(s.out == s.in_)
            c = sp.add_constraint(w - a * s.in_ == 0.0, name="c")
            sp.add_constraint(u >= demand - w)
            sp.parameterize(
                lambda o: sp.set_normalized_coefficient(c, s.in_, -(a + o)), scales, probs
            )
            sp.set_stage_objective(10.0 * u)

    model = sddp.LinearPolicyGraph(builder, stages=2, sense="Min", lower_bound=0.0)
    sddp.train(model, iteration_limit=20, seed=1, **KW)
    assert close(sddp.calculate_bound(model), _coefficient_lp(a, scales, probs, demand))
    # The simulated first-stage cost + second-stage cost is consistent with the parameterization.
    for sim in sddp.simulate(model, 5, ["s", "w"], seed=1, skip_undefined_variables=True):
        assert close(sim[1]["w"], sim[1]["s"].in_ * (a + sim[1]["noise_term"]))


def test_bound_is_statistically_below_policy_cost_epicycles():
    """A cheap cut-validity check on the epicycles model: the (min) bound cannot exceed the
    in-sample expected policy cost."""
    model = tr.build_capex_epicycles()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sddp.train(model, iteration_limit=10, seed=1, **KW)
    sims = sddp.simulate(model, 40, seed=7)
    objs = [sum(s["stage_objective"] for s in sim) for sim in sims]
    mean, se = statistics.mean(objs), statistics.stdev(objs) / math.sqrt(len(objs))
    assert sddp.calculate_bound(model) <= mean + 3 * se
