"""The post-training sections of https://sddp.dev/stable/tutorial/first_steps/ (decision rules,
simulation, confidence intervals, custom recorders, value functions) and the Historical
simulation of https://sddp.dev/stable/tutorial/markov_uncertainty/, checked against
``reference/oracle/first_steps.json``.

The policy is trained with the same fixed Historical scenarios in both languages
(``hydro_thermal.json["deterministic_run_20"]``), so every number below is deterministic.
"""

from __future__ import annotations

import math

from examples.hydro_thermal import subproblem_builder

import sddp
from tests.conftest import load_oracle
from tests.problems import OMEGA_MARKOV, build_markov_uncertainty
from tests.test_parity import _scenario, assert_sim_mean_close, simulate_objectives

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _trained_model() -> sddp.PolicyGraph:
    run = load_oracle("hydro_thermal")["deterministic_run_20"]
    model = sddp.LinearPolicyGraph(subproblem_builder, stages=3, sense="Min", lower_bound=0.0)
    sddp.train(
        model,
        iteration_limit=run["iterations"],
        sampling_scheme=sddp.Historical([_scenario(s) for s in run["scenarios"]]),
        **KW,
    )
    return model


def test_decision_rule_simulation_recorders_and_value_function():
    d = load_oracle("first_steps")
    model = _trained_model()
    assert close(sddp.calculate_bound(model), d["bound"])

    # Obtaining the decision rule
    rule = sddp.DecisionRule(model, node=1)
    assert repr(rule) == "A decision rule for node 1"
    sol = sddp.evaluate(
        rule,
        incoming_state={"volume": 150.0},
        noise=50.0,
        controls_to_record=["hydro_generation", "thermal_generation"],
    )
    j = d["decision_rule"]
    assert close(sol.stage_objective, j["stage_objective"])
    assert close(sol.outgoing_state["volume"], j["outgoing_volume"])
    assert close(sol.controls["hydro_generation"], j["hydro_generation"])
    assert close(sol.controls["thermal_generation"], j["thermal_generation"])

    # Simulating the policy along a fixed scenario, with a custom recorder for the price
    h = d["historical"]
    sims = sddp.simulate(
        model,
        1,
        ["volume", "thermal_generation", "hydro_generation", "hydro_spill"],
        sampling_scheme=sddp.Historical(_scenario(h["scenario"])),
        custom_recorders={"price": lambda sp: sp.dual(sp["demand_constraint"])},
    )
    assert len(sims) == 1 and len(sims[0]) == 3
    stage = sims[0][1]
    keys = {"node_index", "noise_term", "stage_objective", "bellman_term", "volume", "price"}
    assert keys <= set(stage)
    for t, s in enumerate(sims[0]):
        assert close(s["volume"].out, h["outgoing_volume"][t])
        assert close(s["thermal_generation"], h["thermal_generation"][t])
        assert close(s["stage_objective"], h["stage_objective"][t])
        assert close(s["price"], h["price"][t])  # dual of the demand constraint == price

    # Obtaining bounds: Monte Carlo confidence interval vs. the Julia sample (99% CI)
    objs = simulate_objectives(model, d["simulation_100"]["replications"])
    assert_sim_mean_close(objs, d["simulation_100"])
    mu, ci = sddp.confidence_interval(objs)
    assert close(mu, sum(objs) / len(objs)) and ci > 0
    assert sddp.calculate_bound(model) <= mu + ci  # lower bound below the upper-bound estimate

    # Extracting the marginal water values
    V = sddp.ValueFunction(model, node=1)
    for key, point in (("value_function_at_10", 10.0), ("value_function_at_150", 150.0)):
        cost, price = sddp.evaluate_value_function(V, {"volume": point})
        assert close(cost, d[key]["cost"])
        assert close(price["volume"], d[key]["price"])
    assert price["volume"] <= 0.0  # minimising: more water never increases the cost-to-go


def test_markov_historical_simulation_visits_requested_nodes():
    d = load_oracle("first_steps")["markov_historical"]
    model = build_markov_uncertainty()
    sddp.train(model, iteration_limit=40, seed=1, **KW)
    sims = sddp.simulate(
        model,
        1,
        sampling_scheme=sddp.Historical(
            [((1, 1), OMEGA_MARKOV[0]), ((2, 2), OMEGA_MARKOV[2]), ((3, 1), OMEGA_MARKOV[1])]
        ),
    )
    assert [list(s["node_index"]) for s in sims[0]] == d["node_index"]
    assert [[s["noise_term"]["inflow"], s["noise_term"]["fuel_multiplier"]] for s in sims[0]] == d[
        "noise_term"
    ]
