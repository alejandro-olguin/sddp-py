"""The "Risk aversion" explanation page (examples/risk_explanation.py).

Cross-validates the page's primal/dual risk measures and the risk-averse subgradient
theorem against the package's ``adjust_probability`` and SDDP.jl's (oracle
``reference/oracle/explanation.json``), and the risk-averse vanilla SDDP against
``sddp.train(..., risk_measure=...)``.
"""

from __future__ import annotations

import math
import random

import pytest
from examples import risk_explanation as rx
from examples.risk_explanation import (
    Entropic,
    Expectation,
    WorstCase,
    build_risk_averse_model,
    dual_risk,
    dual_risk_averse_subgradient,
    dual_risk_inner,
    evaluate_policy,
    lower_bound,
    primal_risk,
    primal_risk_averse_subgradient,
    train,
)

import sddp
from tests.conftest import load_oracle
from tests.problems import build_hydro_thermal

KW = {"print_level": 0, "run_numerical_stability_report": False}
Z = [1.0, 2.0, 3.0, 4.0]
P = [0.1, 0.2, 0.4, 0.3]
GAMMAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SUBGRAD_KW = {"Ω": rx.Ω_EXAMPLE, "p": rx.P_EXAMPLE, "x̃": rx.X_TILDE}


def close(a: float, b: float, rel: float = 1e-9, atol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


# ---------------------------------------------------------------------------
# Primal and dual risk measures
# ---------------------------------------------------------------------------


def test_primal_values():
    assert close(primal_risk(Expectation(), Z, P), 2.9)
    assert primal_risk(WorstCase(), Z, P) == 4.0
    values = [primal_risk(Entropic(g), Z, P) for g in GAMMAS]
    # Entropic interpolates between the expectation (γ → 0) and the worst case (γ → ∞).
    assert all(a < b for a, b in zip(values, values[1:]))
    assert 2.9 < values[0] < 2.901 and 3.998 < values[-1] < 4.0
    # exp(1000 * 4) would overflow: log-sum-exp keeps it finite.
    assert math.isfinite(primal_risk(Entropic(1e6), Z, P))
    with pytest.raises(ValueError):
        Entropic(0.0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_primal_equals_dual(gamma):
    for F in (Expectation(), WorstCase(), Entropic(gamma)):
        assert close(primal_risk(F, Z, P), dual_risk(F, Z, P)), type(F).__name__
    q, alpha = dual_risk_inner(Entropic(gamma), Z, P)
    assert close(sum(q), 1.0) and all(qi >= 0.0 for qi in q) and alpha >= 0.0
    assert dual_risk_inner(Expectation(), Z, P) == (P, 0.0)
    assert dual_risk_inner(WorstCase(), Z, P) == ([0.0, 0.0, 0.0, 1.0], 0.0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_dual_q_matches_package_adjust_probability(gamma):
    """The package returns ``-α`` as its additive offset (Julia convention); q* is the same."""
    q, alpha = dual_risk_inner(Entropic(gamma), Z, P)
    q_pkg = [0.0] * len(P)
    offset = sddp.Entropic(gamma).adjust_probability(q_pkg, P, Z, Z, True)
    assert all(close(a, b, atol=1e-14) for a, b in zip(q, q_pkg)), (q, q_pkg)
    assert close(offset, -alpha, atol=1e-12)
    # E_q[Z] + offset is the primal risk (the package's cut intercept convention).
    assert close(
        sum(qi * zi for qi, zi in zip(q_pkg, Z)) + offset, primal_risk(Entropic(gamma), Z, P)
    )


def test_dual_q_matches_package_for_expectation_and_worst_case():
    q_pkg = [0.0] * 4
    assert sddp.WorstCase().adjust_probability(q_pkg, P, Z, Z, True) == 0.0
    assert q_pkg == dual_risk_inner(WorstCase(), Z, P)[0]
    q_pkg = [0.0] * 4
    assert sddp.Expectation().adjust_probability(q_pkg, P, Z, Z, True) == 0.0
    assert q_pkg == dual_risk_inner(Expectation(), Z, P)[0]


def test_dual_q_matches_julia_oracle():
    o = load_oracle("explanation")["risk"]
    assert o["Z"] == Z and o["p"] == P
    for entry in o["entropic"]:
        g = entry["gamma"]
        q, alpha = dual_risk_inner(Entropic(g), Z, P)
        assert all(close(a, b, atol=1e-12) for a, b in zip(q, entry["q"])), g
        assert close(-alpha, entry["offset"], atol=1e-12), g
        # SDDP.jl's BigFloat primal vs our log-sum-exp primal.
        assert close(primal_risk(Entropic(g), Z, P), entry["primal"], atol=1e-12), g
    assert o["worst_case"]["q"] == [0.0, 0.0, 0.0, 1.0] and o["worst_case"]["offset"] == 0.0
    assert o["expectation"]["q"] == P and o["expectation"]["offset"] == 0.0


# ---------------------------------------------------------------------------
# The risk-averse subgradient theorem
# ---------------------------------------------------------------------------


def test_subgradient_theorem_expectation_and_worst_case():
    # F[V] = 2x² → 12 at x = 3; F[V] = 3x² → 18. Finite differences on quadratics are exact
    # up to rounding (~1e-8 here).
    for F, expected in ((Expectation(), 12.0), (WorstCase(), 18.0)):
        d = dual_risk_averse_subgradient(rx.V_example, F=F, **SUBGRAD_KW)
        p = primal_risk_averse_subgradient(rx.V_example, F=F, **SUBGRAD_KW)
        assert len(d) == 1 and close(d[0], expected, rel=1e-7)
        assert close(p[0], expected, rel=1e-7)
    # An explicit λ (the analytic gradient 2ωx) gives exactly 12 / 18.
    d = dual_risk_averse_subgradient(
        rx.V_example, lambda x, w: [2 * w * x[0]], F=Expectation(), **SUBGRAD_KW
    )
    assert close(d[0], 12.0)


def test_subgradient_theorem_entropic():
    o = {
        e["gamma"]: e["dual_subgradient"]
        for e in load_oracle("explanation")["subgradient"]["entropic"]
    }
    duals = []
    for g in GAMMAS:
        d = dual_risk_averse_subgradient(rx.V_example, F=Entropic(g), **SUBGRAD_KW)[0]
        duals.append(d)
        assert close(d, o[g], rel=1e-8), g
        if g <= 10.0:
            p = primal_risk_averse_subgradient(rx.V_example, F=Entropic(g), **SUBGRAD_KW)[0]
            # primal is a finite difference of a smooth function: 1e-6 absolute agreement.
            assert abs(p - d) < 1e-6, (g, p, d)
        else:
            # On the page this is where the primal version overflows to NaN; our log-sum-exp
            # primal stays finite and still agrees.
            p = primal_risk_averse_subgradient(rx.V_example, F=Entropic(g), **SUBGRAD_KW)[0]
            assert math.isfinite(p) and abs(p - d) < 1e-6
    # Monotone from the expectation's 12 (γ → 0) to the worst case's 18 (γ → ∞).
    assert all(a <= b + 1e-12 for a, b in zip(duals, duals[1:]))
    assert 12.0 < duals[0] < 12.1 and close(duals[-1], 18.0, rel=1e-8)


# ---------------------------------------------------------------------------
# Risk-averse SDDP vs the package
# ---------------------------------------------------------------------------


def _first_stage_objectives(measure_vanilla, measure_pkg, iterations=30):
    model = build_risk_averse_model()
    train(
        model,
        iteration_limit=iterations,
        replications=10,
        risk_measure=measure_vanilla,
        rng=random.Random(1),
        io=None,
    )
    pk = build_hydro_thermal()
    sddp.train(pk, iteration_limit=iterations, seed=1, risk_measure=measure_pkg, **KW)
    return lower_bound(model), sddp.calculate_bound(pk), model, pk


def test_risk_averse_sddp_entropic_matches_package_and_julia():
    # 3 stages x 3 noises: 30 iterations converge both implementations; the "lower bound"
    # (first-stage objective, an expectation over Ω₁ of the risk-adjusted subproblems)
    # agrees to 1e-9 relative. It is 15000 (the worst-case cost) minus the entropic α
    # penalties: with γ = 1 and cost gaps in the thousands q* ≈ worst case and
    # α ≈ log(3)/γ per cut.
    lb, bound, model, pk = _first_stage_objectives(Entropic(1.0), sddp.Entropic(1.0))
    assert close(lb, bound, rel=1e-9, atol=1e-6), (lb, bound)
    assert 14990.0 < lb < 15000.0
    o = load_oracle("explanation")["hydro_thermal_entropic_1"]
    assert o["iteration_limit"] == 30
    assert close(o["bound"], lb, rel=1e-9, atol=1e-6), (o["bound"], lb)
    # The page's query of the risk-averse policy.
    res = evaluate_policy(model, node=1, incoming_state={"volume": 150.0}, random_variable=75)
    assert close(res["volume_out"], 200.0) and close(res["thermal_generation"], 125.0)
    r = sddp.evaluate(
        sddp.DecisionRule(pk, node=1),
        incoming_state={"volume": 150.0},
        noise=75.0,
        controls_to_record=["thermal_generation"],
    )
    assert close(r.controls["thermal_generation"], res["thermal_generation"])
    assert close(r.outgoing_state["volume"], res["volume_out"])


def test_risk_averse_sddp_worst_case_and_expectation_match_package():
    lb, bound, _, _ = _first_stage_objectives(WorstCase(), sddp.WorstCase())
    assert close(lb, 15000.0, rel=1e-9, atol=1e-6) and close(bound, 15000.0, rel=1e-9, atol=1e-6)
    o = load_oracle("explanation")["hydro_thermal_worst_case"]
    assert close(o["bound"], 15000.0, rel=1e-9, atol=1e-6)
    # Expectation reduces to the risk-neutral page: the deterministic equivalent 8333.33.
    lb, bound, _, _ = _first_stage_objectives(Expectation(), sddp.Expectation(), iterations=20)
    det = load_oracle("hydro_thermal")["deterministic_equivalent"]
    assert close(lb, det, rel=1e-9, atol=1e-6) and close(bound, det, rel=1e-9, atol=1e-6)


def test_risk_averse_cuts_are_expectation_cuts_plus_second_order_term():
    """One backward pass at a fixed trajectory: Entropic(γ) ≈ Expectation + γ Var_p[Z] / 2.

    The entropic risk is ``E_p[Z] + (γ/2) Var_p[Z] + O(γ²)``, so for small γ the cut it
    produces at the same point sits that much above the expectation cut (same slope here
    because the subgradients are identical across the three noises).
    """
    from examples.theory_intro import forward_pass, get_node

    cuts = {}
    for F in (Entropic(1e-9), Expectation()):
        model = build_risk_averse_model()
        traj, _ = forward_pass(model, random.Random(5), io=None)
        rx.backward_pass(model, traj, None, risk_measure=F)
        cuts[type(F).__name__] = get_node(model, 2).cuts[0]
        if isinstance(F, Expectation):
            # Re-solve node 3 at the trajectory's outgoing state to get Z = V(x_k, φ).
            node3 = get_node(model, 3)
            node3.subproblem.fix(node3.states["volume"].in_, traj[1][1]["volume"])
            Z_k = []
            for φ in node3.uncertainty.Ω:
                node3.uncertainty.parameterize(φ)
                node3.subproblem.optimize()
                Z_k.append(node3.subproblem.objective_value())
    tiny, expected = cuts["Entropic"], cuts["Expectation"]
    mean = sum(Z_k) / 3
    var = sum((z - mean) ** 2 for z in Z_k) / 3
    assert tiny.intercept >= expected.intercept - 1e-9  # a convex risk measure ≥ E
    assert close(tiny.intercept - expected.intercept, 1e-9 * var / 2, rel=1e-3, atol=1e-9)
    assert close(tiny.coefficients["volume"], expected.coefficients["volume"], rel=1e-6, atol=1e-6)


def test_main_runs_silently():
    rx.main(seed=7, io=None)
