"""Two-stage newsvendor tutorial vs the Julia oracle ``tutorials_c.json`` (key ``newsvendor``)."""

from __future__ import annotations

import math

import pytest
from examples import tutorial_newsvendor as nv

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


@pytest.fixture(scope="module")
def jl():
    return load_oracle("tutorials_c")["newsvendor"]


def test_demand_sample_matches_julia(jl):
    # Same data in both languages (the Julia script embeds the numpy sample verbatim).
    assert len(nv.d) == 100
    assert nv.d == sorted(nv.d)
    assert all(150.0 <= v <= 250.0 for v in nv.d)
    assert nv.d == jl["d"]


def test_kelley(jl):
    r = nv.run_kelley(verbose=False)
    # Concave quadratic: the true maximiser is (1, -2) with f = 1.
    assert r["lower_bound"] <= 1.0 + 1e-9 <= r["upper_bound"] + 1e-9
    assert r["upper_bound"] - r["lower_bound"] < 1e-2
    assert abs(r["x"][0] - 1.0) < 0.05 and abs(r["x"][1] + 2.0) < 0.05
    assert r["iterations"] == jl["kelley"]["iterations"]
    assert r["status"] == jl["kelley"]["status"]
    assert close(r["upper_bound"], jl["kelley"]["upper_bound"], rel=1e-6, atol=1e-6)


def test_finite_difference_gradient():
    g = nv.finite_difference_gradient(nv.kelley_example_function)
    import numpy as np

    x = np.array([0.3, -1.2])
    assert np.allclose(g(x), nv.kelley_example_gradient(x), atol=1e-6)


def test_second_stage(jl):
    r = nv.solve_second_stage(200, 170)
    assert r == {"V": 847.0, "lambda": -0.1, "x": 30.0, "u": 170.0}
    j = jl["second_stage_200_170"]
    assert close(r["V"], j["V"]) and close(r["lambda"], j["lambda"])
    assert close(r["x"], j["x"]) and close(r["u"], j["u"])


def test_l_shaped_sddp_and_deterministic_equivalent_agree(jl):
    ls = nv.l_shaped(verbose=False)
    assert ls["upper_bound"] - ls["lower_bound"] < 1e-6
    assert close(ls["x"], jl["l_shaped"]["x"]) and close(
        ls["objective"], jl["l_shaped"]["objective"]
    )
    assert ls["iterations"] == jl["l_shaped"]["iterations"]
    r2 = nv.solve_second_stage(ls["x"], 170.0)
    assert close(r2["V"], jl["l_shaped"]["second_stage_170"]["V"])

    det = sddp.deterministic_equivalent(nv.build_newsvendor())
    det.optimize()
    assert close(det.objective_value(), jl["deterministic_equivalent"])
    assert close(det.objective_value(), ls["objective"])

    model = nv.build_newsvendor()
    sddp.train(model, log_every_iteration=True, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    # Train to convergence: the default stopping rule may stop with a small gap, so keep
    # going with BoundStalling until the bound is stationary.
    if not close(bound, ls["objective"]):
        sddp.train(
            model,
            add_to_existing_cuts=True,
            stopping_rules=[sddp.BoundStalling(5, atol=1e-9)],
            iteration_limit=200,
            seed=1,
            **KW,
        )
        bound = sddp.calculate_bound(model)
    assert close(bound, ls["objective"])
    assert close(bound, jl["train_default"]["bound"])

    # Decision rules
    s1 = sddp.evaluate(sddp.DecisionRule(model, node=1), incoming_state={"x": 0.0})
    j1 = jl["decision_rule"]["node1"]
    assert close(s1.outgoing_state["x"], j1["x_out"], rel=1e-6, atol=1e-5)
    assert close(s1.stage_objective, j1["stage_objective"], rel=1e-6, atol=1e-5)
    s2 = sddp.evaluate(
        sddp.DecisionRule(model, node=2),
        incoming_state={"x": s1.outgoing_state["x"]},
        noise=170.0,
        controls_to_record=["u_sell"],
    )
    j2 = jl["decision_rule"]["node2"]
    assert close(s2.controls["u_sell"], j2["u_sell"]) and s2.controls["u_sell"] == 170.0
    assert close(s2.outgoing_state["x"], j2["x_out"], rel=1e-6, atol=1e-5)
    assert close(s2.stage_objective, j2["stage_objective"], rel=1e-6, atol=1e-5)

    # Simulation structure
    sims = sddp.simulate(
        model, 10, ["x", "u_sell", "u_make"], skip_undefined_variables=True, seed=1
    )
    assert len(sims) == jl["simulate"]["replications"] == 10
    assert len(sims[0]) == jl["simulate"]["stages"] == 2
    for key in ("node_index", "noise_term", "stage_objective", "bellman_term", "x", "u_sell"):
        assert key in sims[0][0] and key in sims[0][1]
    assert sims[0][0]["node_index"] == 1 and sims[0][1]["node_index"] == 2
    assert sims[0][0]["noise_term"] is None and sims[0][1]["noise_term"] in nv.d
    assert isinstance(sims[0][0]["x"], sddp.StateValue)
    assert math.isnan(sims[0][0]["u_sell"]) and math.isnan(sims[0][1]["u_make"])
    objectives = [sum(data["stage_objective"] for data in sim) for sim in sims]
    mu, t = sddp.confidence_interval(objectives)
    assert t >= 0.0


def test_risk_measures_exact_quantiles(jl):
    # The value function is piecewise linear with breakpoints at the demand sample, so the
    # risk-averse order quantities are order statistics of ``d``: WorstCase -> min(d);
    # CVaR(beta) -> F^{-1}(beta * cu / (cu + co)) with underage cost cu = 5 - 2 = 3 and
    # overage cost co = 2 + 0.1, i.e. the empirical 0.4 * 3 / 5.1 = 0.235-quantile, the 24th
    # smallest demand.
    x_worst = nv.solve_newsvendor(sddp.WorstCase(), seed=1, run_numerical_stability_report=False)
    assert close(x_worst, min(nv.d))
    assert close(x_worst, jl["worst_case"]["x"])
    x_cvar = nv.solve_newsvendor(sddp.CVaR(0.4), seed=1, run_numerical_stability_report=False)
    assert any(close(x_cvar, v) for v in nv.d)
    assert close(x_cvar, nv.d[23])
    assert close(x_cvar, jl["cvar_0.4"]["x"])


def test_entropic_sweep(jl):
    gammas = nv.entropic_gammas()
    assert [round(g, 12) for g in gammas] == [round(g, 12) for g in jl["entropic_gammas"]]
    # Fixed iteration count: the 2-stage backward pass is deterministic (the sampled noise
    # only affects the simulated value), so both languages build the same cuts. The bounds
    # agree to ~1e-10 relative. The first-stage decision itself is less precise: the
    # risk-adjusted value function is smooth and flat around its maximiser, so the cutting
    # planes there are nearly parallel and the LP vertex moves by O(sqrt(gap)); Python and
    # Julia (different HiGHS builds, port at 1e-9 tolerances) differ by up to ~1e-3 in x.
    res_30 = [
        nv.solve_newsvendor_details(
            sddp.Entropic(g), seed=1, iteration_limit=30, run_numerical_stability_report=False
        )
        for g in gammas
    ]
    for g, r, j in zip(gammas, res_30, jl["entropic_iter30"]):
        assert r["iterations"] == j["iterations"] == 30
        assert close(r["bound"], j["bound"], rel=1e-8, atol=1e-6), (g, r, j)
        assert close(r["x"], j["x"], rel=0.0, atol=1e-2), (g, r, j)
    buy_30 = [r["x"] for r in res_30]
    # Monotone: more risk aversion -> fewer pies, between the two extremes.
    assert all(a >= b - 1e-6 for a, b in zip(buy_30, buy_30[1:]))
    assert min(nv.d) - 1e-6 <= min(buy_30) and max(buy_30) <= jl["l_shaped"]["x"] + 1e-6
    # Page-faithful default stopping rule: the number of iterations may differ between the
    # languages, so compare loosely (the converged decision is what the page plots).
    res = [
        nv.solve_newsvendor_details(sddp.Entropic(g), seed=1, run_numerical_stability_report=False)
        for g in gammas
    ]
    for g, r, j in zip(gammas, res, jl["entropic_default"]):
        assert close(r["bound"], j["bound"], rel=1e-6, atol=1e-4), (g, r, j)
        assert close(r["x"], j["x"], rel=0.0, atol=1e-2), (g, r, j)
