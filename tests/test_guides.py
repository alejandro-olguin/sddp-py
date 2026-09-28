"""The SDDP.jl how-to guides (docs/src/guides) vs the Julia oracle ``guides.json``."""

from __future__ import annotations

import json
import math
import threading

import pytest
from examples import guides as g

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


@pytest.fixture(scope="module")
def oracle():
    return load_oracle("guides")


# ------------------------------------------------------------ access_previous_variables
class TestAccessPreviousVariables:
    def test_first_stage_capacity_decision(self, oracle):
        d = oracle["access_previous_variables"]
        model = g.build_first_stage_capacity_model(d["capacity_w"])
        sddp.train(model, iteration_limit=300, seed=1, **KW)
        bound = sddp.calculate_bound(model)
        # 300 iterations with different forward samples: both are close to the optimum.
        assert close(bound, d["capacity_bound_300"], rel=1e-3)
        sims = sddp.simulate(model, 1, ["capacity"], seed=1)
        # Capacity is decided in stage 1 and carried unchanged through all 10 stages.
        caps = [s["capacity"].out for s in sims[0]]
        assert len(caps) == 10 and all(close(c, caps[0]) for c in caps)

    def test_pipeline_lag_is_deterministic(self, oracle):
        d = oracle["access_previous_variables"]
        det = sddp.deterministic_equivalent(g.build_pipeline_lag_model())
        det.optimize()
        # Orders placed at t are sellable at t+5: buy 10 in stages 1..5, sell in 6..10 -> 50.
        assert close(det.objective_value(), 50.0)
        assert close(det.objective_value(), d["pipeline_deterministic_equivalent"])
        model = g.build_pipeline_lag_model()
        sddp.train(model, iteration_limit=100, seed=1, **KW)
        assert close(sddp.calculate_bound(model), 50.0, rel=1e-6)
        assert close(sddp.calculate_bound(model), d["pipeline_bound_100"], rel=1e-6)
        assert all(close(b, 50.0) for b in d["pipeline_bound_log"])

    def test_stochastic_lead_time_as_written(self, oracle):
        d = oracle["access_previous_variables"]
        model = g.build_stochastic_lead_time_model()
        sddp.train(model, iteration_limit=30, seed=1, **KW)
        # The guide sets the *normalized* coefficient to +1, which flips the sign of the order
        # (buying removes stock from the pipeline), so never buying is optimal: value 0.
        assert close(sddp.calculate_bound(model), 0.0)
        assert close(sddp.calculate_bound(model), d["lead_time_bound_30"])
        # The parameterized coefficient really is rewritten (ω=3 puts u_buy in c_pipeline[3]).
        node = model[1]
        sddp.parameterize(node, 3)
        u_buy = node.subproblem["u_buy"]
        coefs = [
            node.model.get_normalized_coefficient(node.subproblem[f"c_pipeline[{i}]"], u_buy)
            for i in range(1, 11)
        ]
        assert coefs == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_stochastic_lead_time_corrected(self, oracle):
        d = oracle["access_previous_variables"]
        assert close(g.stochastic_lead_time_optimal_value(), 145.0)
        model = g.build_stochastic_lead_time_model(corrected=True)
        sddp.train(model, iteration_limit=100, seed=1, **KW)
        assert close(sddp.calculate_bound(model), 145.0, rel=1e-6)
        assert close(sddp.calculate_bound(model), d["lead_time_corrected_bound_100"], rel=1e-6)


# ==================================================================== add_a_custom_cut
class TestAddACustomCut:
    def test_cuts_after_one_iteration_are_exact(self, oracle, tmp_path):
        d = oracle["add_a_custom_cut"]
        for seed in (1, 2, 3):  # the sampled ω does not matter: x.out = 0 on every path
            text = g.write_cuts_after_one_iteration(str(tmp_path / "cuts.json"), seed=seed)
            cuts = {c["node"]: c for c in json.loads(text)}
            assert set(cuts) == {"1", "2", "3"}
            for c in cuts.values():
                assert c["multi_cuts"] == [] and c["risk_set_cuts"] == []
            assert cuts["3"]["single_cuts"] == []
            (c2,) = cuts["2"]["single_cuts"]
            assert c2["coefficients"] == {"x": -200.0} and c2["intercept"] == 30000.0
            assert c2["state"] == {"x": 0.0}
            (c1,) = cuts["1"]["single_cuts"]
            assert c1["coefficients"] == {"x": -200.0} and c1["intercept"] == 57500.0
            assert c1["state"] == {"x": 0.0}
        julia = {c["node"]: c for c in d["cuts_after_one_iteration"]}
        for n in ("1", "2", "3"):
            assert cuts[n]["single_cuts"] == julia[n]["single_cuts"]

    def test_read_custom_cut(self, oracle, tmp_path):
        d = oracle["add_a_custom_cut"]
        before, after = g.bound_with_custom_cut(str(tmp_path / "new_cuts.json"))
        assert before == 10000.0 == d["bound_fresh"]
        assert after == 62500.0 == d["bound_after_custom_cut"]


# ================================================= add_a_multidimensional_state_variable
def test_multidimensional_state_variable(oracle):
    d = oracle["add_a_multidimensional_state_variable"]
    record: list[float] = []
    printed: list[str] = []
    model = g.build_multidimensional_state_model(record, printer=printed.append)
    assert record == [0.0, 1.0, 3.0] == d["lower_bounds"]
    assert printed == [
        "Lower bound of outgoing x is: 0.0",
        "Lower bound of outgoing y[1] is: 1.0",
        "Lower bound of outgoing z[3, :B] is: 3.0",
    ]
    assert len(model.nodes) == 1 == d["num_nodes"]
    node = model[1]
    assert set(node.states) == {"x", "y[1]", "y[2]", "z[3,A]", "z[3,B]", "z[4,A]", "z[4,B]"}
    assert node.model.lower_bound(node.states["z[4,A]"].out) == 4.0
    assert model.initial_root_state["y[2]"] == 2.0
    # A 1-stage model trains and simulates.
    sddp.train(model, iteration_limit=2, seed=1, **KW)
    assert close(sddp.calculate_bound(model), 0.0)


# ================================================================== add_a_risk_measure
class TestAddARiskMeasure:
    def test_three_ways_give_identical_bounds(self, oracle):
        d = oracle["add_a_risk_measure"]
        n = d["iteration_limit"]
        bounds = []
        # The same measure (WorstCase) expressed as a measure, a dict, and a function.
        for rm in (
            sddp.WorstCase(),
            {1: sddp.WorstCase(), 2: sddp.WorstCase(), 3: sddp.WorstCase()},
            lambda node_index: sddp.WorstCase(),
        ):
            model = g.build_risk_measure_model()
            sddp.train(model, risk_measure=rm, iteration_limit=n, seed=1, **KW)
            bounds.append(sddp.calculate_bound(model))
        assert bounds[0] == bounds[1] == bounds[2]
        for key in ("worst_case_measure", "worst_case_dict", "worst_case_function"):
            assert close(bounds[0], d[key], rel=1e-6)

    def test_mixed_measures_dict_and_function_agree(self, oracle):
        d = oracle["add_a_risk_measure"]
        n = d["iteration_limit"]
        m1 = g.build_risk_measure_model()
        g.train_with_risk_measure_dict(m1, iteration_limit=n, seed=1)
        m2 = g.build_risk_measure_model()
        g.train_with_risk_measure_function(m2, iteration_limit=n, seed=1)
        assert sddp.calculate_bound(m1) == sddp.calculate_bound(m2)
        assert close(sddp.calculate_bound(m1), d["mixed_dict"], rel=1e-6)
        assert close(sddp.calculate_bound(m2), d["mixed_function"], rel=1e-6)
        m3 = g.build_risk_measure_model()
        g.train_with_risk_measure(m3, iteration_limit=n, seed=1)
        # Expectation at node 1 is less conservative than WorstCase everywhere.
        assert sddp.calculate_bound(m1) < sddp.calculate_bound(m3)
        assert close(sddp.calculate_bound(m3), d["worst_case_measure"], rel=1e-6)
        # The simulated value is the risk-neutral value of the (risk-averse) policy.
        sims = sddp.simulate(m3, 50, seed=1)
        assert all(len(s) == 3 for s in sims)

    def test_supported_risk_measures_are_constructible(self):
        measures = g.supported_risk_measures()
        names = [type(m).__name__ for m in measures]
        assert names == [
            "Expectation",
            "AVaR",
            "ConvexCombination",
            "AVaR",  # CVaR is an alias of AVaR
            "ConvexCombination",  # EAVaR
            "Entropic",
            "ModifiedChiSquared",
            "Wasserstein",
            "WorstCase",
        ]
        arithmetic = 0.5 * sddp.Expectation() + 0.5 * sddp.AVaR(0.25)
        assert isinstance(arithmetic, sddp.ConvexCombination)


# =================================================================== add_integrality
class TestAddIntegrality:
    def test_integer_model_default_handler(self, oracle):
        d = oracle["add_integrality"]
        det = sddp.deterministic_equivalent(g.build_integrality_model())
        det.optimize()
        assert close(det.objective_value(), 900.0) and close(
            det.objective_value(), d["deterministic_equivalent"]
        )
        model = g.build_integrality_model()
        assert model[1].has_integrality
        sddp.train(model, iteration_limit=20, seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound_default_20"], rel=1e-6)

    def test_bandit_duality_log_every_iteration(self, oracle, capsys):
        d = oracle["add_integrality"]
        model = g.build_integrality_model()
        g.train_with_bandit_duality(model, iteration_limit=20, seed=1)
        out = capsys.readouterr().out
        log = model.most_recent_training_results.log
        assert len(log) == 20
        # Every iteration was logged, with the arm's key (" " or "S") after the iteration.
        assert sum(1 for line in out.splitlines() if line.strip().startswith(("1 ", "1S"))) >= 1
        assert set(l.duality_key for l in log) <= {" ", "S"}
        assert close(sddp.calculate_bound(model), d["bound_bandit_20"], rel=1e-6)

    def test_conic_duality_uses_the_given_optimizer(self, oracle):
        d = oracle["add_integrality"]
        model = g.build_integrality_model()
        seen: list[tuple[str, bool]] = []

        def hook(model_, node, state, noise, path, handler):
            m = node.model
            seen.append((m.raw.get_raw_parameter("presolve"), m.has_integrality()))

        for node in model.nodes.values():
            node.pre_optimize_hook = hook
        g.train_with_conic_duality_optimizer(model, iteration_limit=20, seed=1, **KW)
        # Relaxed (backward-pass) solves used presolve="off"; MIP (forward) solves did not.
        assert ("off", False) in seen
        assert ("choose", True) in seen
        assert ("off", True) not in seen
        # Options are restored after every backward pass.
        assert all(
            n.model.raw.get_raw_parameter("presolve") == "choose" for n in model.nodes.values()
        )
        assert all(n.model.has_integrality() for n in model.nodes.values())
        assert close(sddp.calculate_bound(model), d["bound_conic_with_optimizer_20"], rel=1e-6)

    def test_other_backend_is_rejected(self):
        model = g.build_integrality_model()
        other = sddp.OptimizerFactory("not_highs", lambda: None)
        with pytest.raises(NotImplementedError, match="same backend"):
            sddp.train(
                model,
                duality_handler=sddp.ContinuousConicDuality(other),
                iteration_limit=1,
                seed=1,
                **KW,
            )


# ========================================================== add_multidimensional_noise
class TestAddMultidimensionalNoise:
    def test_simple_cartesian_product(self, oracle):
        d = oracle["add_multidimensional_noise"]
        model = g.build_simple_multidimensional_noise_model()
        omega = [g.Realization(v, c) for v in [1, 2] for c in [3, 4, 5]]
        for node in model.nodes.values():
            assert [n.term for n in node.noise_terms] == omega
            assert close(sum(n.probability for n in node.noise_terms), 1.0)
        for s in sddp.simulate(model, 1, seed=1)[0]:
            assert s["noise_term"] in omega
        # A fresh model: simulating fixes `x.in` and (as in SDDP.jl, which copies the fixing
        # constraint too) the extensive form of a simulated model is infeasible.
        det = sddp.deterministic_equivalent(g.build_simple_multidimensional_noise_model())
        det.optimize()
        assert close(det.objective_value(), 3 * 1.5 * 3.9)  # E[v] E[c] per stage
        assert close(det.objective_value(), d["simple_deterministic_equivalent"])
        sddp.train(model, iteration_limit=2, seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["simple_bound_2"])

    def test_finite_discrete_distributions(self, oracle):
        d = oracle["add_multidimensional_noise"]
        omega, P = g.finite_product_distribution()
        assert len(omega) == 11 * 2 * 7 and close(sum(P), 1.0, rel=1e-12)
        assert omega == d["finite_omega"]
        assert all(close(p, q, rel=1e-12, atol=1e-15) for p, q in zip(P, d["finite_P"]))
        model = g.build_vector_noise_model(omega, P)
        for s in sddp.simulate(model, 1, seed=1)[0]:
            assert s["noise_term"] in omega
        sddp.train(model, iteration_limit=2, seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["finite_bound_2"], rel=1e-9)

    def test_sampled_distribution(self):
        omega, P = g.sampled_product_distribution(N=100, seed=1)
        assert len(omega) == 100 and close(sum(P), 1.0, rel=1e-12)
        assert all(len(w) == 3 and w[1] >= 0 for w in omega)
        model = g.build_vector_noise_model(omega, P)
        for s in sddp.simulate(model, 1, seed=1)[0]:
            assert s["noise_term"] in omega
        sddp.train(model, iteration_limit=2, seed=1, **KW)
        expected = 3 * sum(p * (w[0] * w[1] + w[2]) for w, p in zip(omega, P))
        assert close(sddp.calculate_bound(model), expected, rel=1e-9)


# ================================================== add_noise_in_the_constraint_matrix
def test_noise_in_the_constraint_matrix(oracle):
    d = oracle["add_noise_in_the_constraint_matrix"]
    model = g.build_constraint_matrix_noise_model()
    assert repr(model) == "A policy graph with 3 nodes.\n Node indices: 1, 2, 3\n"
    node = model[1]
    for omega in (0.2, 0.5, 1.0):
        sddp.parameterize(node, omega)
        c = node.subproblem["emissions"]
        assert node.model.get_normalized_coefficient(c, node.states["x"].out) == omega
    det = sddp.deterministic_equivalent(model)
    det.optimize()
    assert close(det.objective_value(), -8.0)  # x.out = 1/ω, 3 stages of E[-1/ω] = -8/3
    assert close(det.objective_value(), d["deterministic_equivalent"])
    sddp.train(model, iteration_limit=10, seed=1, **KW)
    # lower_bound = 0 is not a valid lower bound for this model (its value is negative), so
    # the cuts are truncated by θ ≥ 0 and the bound is the first-stage value -8/3, as in Julia.
    assert close(sddp.calculate_bound(model), -8 / 3)
    assert close(sddp.calculate_bound(model), d["bound_10"])


# ============================================================== choose_a_stopping_rule
class TestChooseAStoppingRule:
    def test_statuses_match_sddp_jl(self, oracle):
        d = oracle["choose_a_stopping_rule"]
        assert g.train_iteration_limit(g.build_risk_measure_model(), seed=1) == "iteration_limit"
        assert d["iteration_limit"] == "iteration_limit"
        assert g.train_time_limit(g.build_risk_measure_model(), 0.0, seed=1) == "time_limit"
        assert d["time_limit"] == "time_limit"
        model = g.build_risk_measure_model()
        assert g.train_bound_stalling(model, seed=1) == "bound_stalling" == d["bound_stalling"]
        assert close(sddp.calculate_bound(model), 8333.3333333333, rel=1e-6)
        assert (
            g.train_time_limit_or_bound_stalling(g.build_risk_measure_model(), seed=1)
            == "bound_stalling"
            == d["time_limit_or_bound_stalling"]
        )
        assert (
            g.train_time_limit_and_bound_stalling(g.build_risk_measure_model(), 0.0, seed=1)
            == "time_limit ∧ bound_stalling"
            == d["time_limit_and_bound_stalling"]
        )

    def test_default_rule_is_simulation_stopping(self, oracle):
        d = oracle["choose_a_stopping_rule"]
        model = g.build_risk_measure_model()
        sddp.train(model, seed=1, **KW)
        assert sddp.termination_status(model) == "simulation_stopping" == d["default"]

    def test_supported_rules_status_names(self, oracle):
        d = oracle["choose_a_stopping_rule"]
        assert [r.stopping_rule_status() for r in g.supported_stopping_rules()] == d["supported"]
        assert d["supported"] == [
            "iteration_limit",
            "time_limit",
            "statistical",
            "bound_stalling",
            "iteration_limit ∧ time_limit",
            "simulation_stopping",
            "first_stage_stopping",
        ]
        assert sddp.termination_status(g.build_risk_measure_model()) == "model_not_solved"


# =============================================================== create_a_belief_state
def test_create_a_belief_state(oracle):
    d = oracle["create_a_belief_state"]
    G = sddp.MarkovianGraph([[0.5, 0.5], [[0.2, 0.8], [0.8, 0.2]]])
    assert repr(G) == d["repr_before"]
    G = g.create_belief_graph()
    assert repr(G) == d["repr"]
    assert G.belief_partition == [[(1, 1), (1, 2)], [(2, 1), (2, 2)]]
    assert G.belief_lipschitz == [[1e5, 1e5], [1e5, 1e5]]
    model = g.build_belief_model(G)
    assert model.belief_partition == [{(1, 1), (1, 2)}, {(2, 1), (2, 2)}]
    assert all(node.belief_state is not None for node in model.nodes.values())
    sddp.train(model, iteration_limit=5, seed=1, **KW)
    assert sddp.termination_status(model) == "iteration_limit"


# ======================================================= create_a_general_policy_graph
class TestCreateAGeneralPolicyGraph:
    def test_graph_reprs_match_julia(self, oracle):
        d = oracle["create_a_general_policy_graph"]
        before, graph = g.linear_graph_example()
        assert before == d["linear_graph"]
        assert repr(graph) == d["linear_graph_cyclic"]
        assert repr(g.unicyclic_graph_example()) == d["unicyclic"]
        assert repr(g.markovian_graph_example()) == d["markovian"]
        before, graph = g.general_graph_example()
        assert before == d["general_empty"]
        assert repr(graph) == d["general"]
        # The strings are those printed on the documentation page.
        assert d["linear_graph"] == (
            "Root\n 0\nNodes\n 1\n 2\n 3\nArcs\n"
            " 0 => 1 w.p. 1.0\n 1 => 2 w.p. 1.0\n 2 => 3 w.p. 1.0"
        )
        assert d["general_empty"] == "Root\n root_node\nNodes\n {}\nArcs\n {}"

    def test_policy_graph_from_graph(self, oracle):
        d = oracle["create_a_general_policy_graph"]
        printed: list[str] = []
        model, called = g.create_policy_graph_from_graph(printer=printed.append)
        assert called == ["decision_node"] == d["called_from"]
        assert printed == ["Called from node: decision_node"]
        assert model.root_node == "root_node" and list(model.nodes) == ["decision_node"]
        assert [(c.term, c.probability) for c in model["decision_node"].children] == [
            ("decision_node", 0.9)
        ]

    def test_simulate_fixed_depth_on_cyclic_graph(self):
        model = g.build_cyclic_model()
        assert sddp.is_cyclic(model)
        sddp.train(model, iteration_limit=5, seed=1, **KW)
        sims = g.simulate_fixed_depth(model, n=3, max_depth=10, seed=1)
        assert [len(s) for s in sims] == [10, 10, 10]
        assert [s["node_index"] for s in sims[0]] == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        # Without max_depth the length is random (the 4 -> 1 arc has probability 0.9).
        with pytest.raises(ValueError):
            sddp.InSampleMonteCarlo(terminate_on_dummy_leaf=False)

    def test_markovian_graph_from_simulator(self):
        model = g.build_simulator_markovian_model(budget=10, scenarios=100, seed=1)
        assert model.root_node == (0, 0.0)
        assert len(model.nodes) == 10
        stages = sorted({t for t, _ in model.nodes})
        assert stages == [1, 2, 3, 4, 5]
        assert all(price == 0.0 for t, price in model.nodes if t == 1)
        sddp.train(model, iteration_limit=10, seed=1, **KW)
        assert sddp.termination_status(model) == "iteration_limit"
        sims = sddp.simulate(model, 2, ["x"], seed=1)
        assert all(len(s) == 5 for s in sims)
        assert all(s["x"].out <= s["x"].in_ + 1e-9 for sim in sims for s in sim)


# ============================================================= deterministic_equivalent
def test_deterministic_equivalent(oracle):
    d = oracle["deterministic_equivalent"]
    model = g.build_deterministic_equivalent_model()
    assert repr(model) == "A policy graph with 2 nodes.\n Node indices: 1, 2\n"
    det, obj = g.solve_deterministic_equivalent(model)
    assert abs(obj - (-5.4725)) < 1e-9
    assert abs(obj - d["objective_value"]) < 1e-9
    # Same extensive form as JuMP's: 24 variables (6 scenario nodes x 4) and 10 equality rows
    # (6 balance + 4 linking); variable bounds are not rows in the port.
    assert det.num_variables() == 24 == d["num_variables"]
    assert det.num_constraints() == 10 == d["num_equality_rows"]
    assert d["num_constraints_total"] == 28


# ================================================================ implement_a_par_model
class TestImplementAPARModel:
    def test_par_recursion(self, oracle):
        d = oracle["implement_a_par_model"]
        model = g.build_par_model()
        assert model[1].states.keys() == {"y[1]", "y[2]", "y[3]"}
        y = g.simulate_par_model(model, seed=1)
        assert len(y) == 12
        y = g.simulate_par_model(model, g.par_historical_scheme())
        expected = g.par_recursion(g.ALPHA, g.Y_HISTORY, [0.1, -0.1, 0.2])
        assert expected == [1.1, 0.95, 1.205] or all(
            close(a, b, rel=1e-12) for a, b in zip(expected, [1.1, 0.95, 1.205])
        )
        assert all(close(a, b, rel=1e-9, atol=1e-9) for a, b in zip(y, expected))
        assert all(close(a, b, rel=1e-9, atol=1e-9) for a, b in zip(y, d["historical_y"]))
        # The shift constraints move y[1] -> y[2] -> y[3] between stages.
        sims = sddp.simulate(
            model, 1, ["y[1]", "y[2]", "y[3]"], sampling_scheme=g.par_historical_scheme()
        )
        rows = [[s["y[1]"].out, s["y[2]"].out, s["y[3]"].out] for s in sims[0]]
        for r, jr in zip(rows, d["historical_y_all"]):
            assert all(close(a, b, rel=1e-9, atol=1e-9) for a, b in zip(r, jr))
        assert all(close(rows[1][1], rows[0][0]) and close(rows[2][2], rows[0][0]) for _ in [0])

    def test_noise_term_unfix_then_fix(self, oracle):
        d = oracle["implement_a_par_model"]
        model = g.build_par_model_with_noise_terms()
        y = g.simulate_par_model(model, seed=1)
        assert len(y) == 12
        y = g.simulate_par_model(model, g.par_noise_term_scheme())
        assert y == [1.0, 1.2, 1.4] == d["noiseterm_y"]
        sims = sddp.simulate(
            model, 1, ["y[1]", "y[2]", "y[3]"], sampling_scheme=g.par_noise_term_scheme()
        )
        rows = [[s["y[1]"].out, s["y[2]"].out, s["y[3]"].out] for s in sims[0]]
        assert rows == d["noiseterm_y_all"] == [[1.0, 1.0, 1.0], [1.2, 1.0, 1.0], [1.4, 1.2, 1.0]]
        # As in Julia, `fix(y[1].out, ...)` in `parameterize` leaves the state fixed afterwards.
        assert (
            [model[t].model.is_fixed(model[t].states["y[1]"].out) for t in (1, 2, 3)]
            == [
                True,
                True,
                True,
            ]
            == d["noiseterm_y1_out_fixed_after"]
        )
        assert not model[4].model.is_fixed(model[4].states["y[1]"].out)

    def test_cuts_stay_valid_after_fixing_state_in_parameterize(self):
        # A variant with an objective on y[1] so the bound is informative: fixing y[1].out inside
        # parameterize changes the outgoing bounds mid-training; the bound must still equal the
        # deterministic equivalent (a converged deterministic-in-noise recursion).
        def build():
            def builder(sp, t):
                y = [sp.add_state(f"y[{p}]", initial_value=1.0) for p in (1, 2, 3)]
                w = sp.add_variable("omega")
                u = sp.add_variable("u", lb=0.0)
                sp.add_constraint(y[0].out == sum(a * yp.in_ for a, yp in zip(g.ALPHA, y)) + w)
                for p in (1, 2):
                    sp.add_constraint(y[p].out == y[p - 1].in_)
                sp.add_constraint(u >= y[0].out)
                sp.set_stage_objective(u)
                omega = [g.NoiseTerm(True, -0.25), g.NoiseTerm(False, 1.5), g.NoiseTerm(True, 0.25)]

                def modify(term):
                    if term.is_noise:
                        sp.unfix(y[0].out)
                        sp.fix(w, term.value)
                    else:
                        sp.unfix(w)
                        sp.fix(y[0].out, term.value)

                sp.parameterize(modify, omega)

            return sddp.LinearPolicyGraph(builder, stages=4, lower_bound=0.0, optimizer=sddp.HiGHS)

        det = sddp.deterministic_equivalent(build())
        det.optimize()
        model = build()
        sddp.train(model, iteration_limit=30, seed=1, **KW)
        assert close(sddp.calculate_bound(model), det.objective_value(), rel=1e-6)


# ==================================================== improve_computational_performance
def test_improve_computational_performance():
    for train in (g.train_single_cut, g.train_multi_cut):
        model = g.build_two_stage_model()
        train(model, seed=1, **KW)
        assert sddp.termination_status(model) == "iteration_limit"
        assert close(sddp.calculate_bound(model), 0.0)
    model = g.build_two_stage_model()
    # Like SDDP.jl, a fresh BellmanFunction defaults to MULTI_CUT; `train` sets `cut_type`.
    assert model[1].bellman_function.cut_type == sddp.MULTI_CUT
    g.train_single_cut(model, seed=1, **KW)
    assert model[1].bellman_function.cut_type == sddp.SINGLE_CUT
    g.train_multi_cut(model, seed=1, add_to_existing_cuts=True, **KW)
    assert model[1].bellman_function.cut_type == sddp.MULTI_CUT
    with pytest.warns(UserWarning, match="fewer nodes"):
        sims = g.train_and_simulate_threaded(g.build_two_stage_model(), seed=1, **KW)
    assert len(sims) == 10 and all(len(s) == 2 for s in sims)


# ============================================ simulate_using_a_different_sampling_scheme
@pytest.fixture(scope="module")
def markov_model(oracle):
    model = g.build_markov_hydro_model()
    det = sddp.deterministic_equivalent(model)
    det.optimize()
    assert close(
        det.objective_value(),
        oracle["simulate_using_a_different_sampling_scheme"]["deterministic_equivalent"],
    )
    sddp.train(model, iteration_limit=10, seed=1, **KW)
    return model


class TestSimulateUsingADifferentSamplingScheme:
    def test_in_sample_monte_carlo(self, markov_model):
        model = markov_model
        sims = g.simulate_in_sample(model, 20, seed=1)
        assert len(sims) == 20 and all(len(s) == 3 for s in sims)
        assert sorted({s["noise_term"] for sim in sims for s in sim}) == g.OMEGA

    def test_out_of_sample_monte_carlo(self, markov_model, oracle):
        model = markov_model
        d = oracle["simulate_using_a_different_sampling_scheme"]
        scheme = g.out_of_sample_scheme(model)
        sims = sddp.simulate(model, 1, sampling_scheme=scheme, seed=1)
        assert sims[0][2]["noise_term"] == (75.0, 1.2)
        assert list(sims[0][2]["noise_term"]) == d["out_of_sample_noise_3"]
        assert [c.term for c in scheme.children[(1, 1)]] == [(2, 1), (2, 2)]
        assert [c.probability for c in scheme.children[(1, 1)]] == [0.5, 0.5]
        assert scheme.children[(3, 1)] == [] and scheme.root_children[0].term == (1, 1)
        nodes = {
            tuple(s["node_index"] for s in sim)
            for sim in sddp.simulate(model, 40, sampling_scheme=scheme, seed=2)
        }
        assert (1, 1) in {n[0] for n in nodes} and len(nodes) > 1

    def test_out_of_sample_insample_transition(self, markov_model, oracle):
        model = markov_model
        d = oracle["simulate_using_a_different_sampling_scheme"]
        scheme = g.out_of_sample_scheme_insample_transition(model)
        sims = sddp.simulate(model, 1, sampling_scheme=scheme, seed=1)
        assert sims[0][2]["noise_term"] == (65.0, 1.1)
        assert list(sims[0][2]["noise_term"]) == d["insample_transition_noise_3"]
        assert [(c.term, c.probability) for c in scheme.children[(2, 1)]] == [
            ((3, 1), 0.75),
            ((3, 2), 0.25),
        ]

    def test_historical(self, markov_model, oracle):
        model = markov_model
        d = oracle["simulate_using_a_different_sampling_scheme"]
        sim = g.simulate_historical(model)
        assert [s["node_index"] for s in sim] == [(1, 1), (2, 2), (3, 1)]
        assert [list(s["node_index"]) for s in sim] == d["historical_nodes"]
        assert [s["noise_term"] for s in sim] == [g.OMEGA[0], g.OMEGA[2], g.OMEGA[1]]
        assert all(
            close(a, b, rel=1e-6)
            for a, b in zip([s["stage_objective"] for s in sim], d["historical_stage_objectives"])
        )

    def test_historical_reprs(self, oracle):
        d = oracle["simulate_using_a_different_sampling_scheme"]
        seq = g.historical_sequential()
        assert repr(seq) == "A Historical sampler with 2 scenarios sampled sequentially."
        assert repr(seq) == d["historical_sequential_repr"]
        prob = g.historical_probabilistic()
        assert repr(prob) == "A Historical sampler with 2 scenarios sampled probabilistically."
        assert repr(prob) == d["historical_probabilistic_repr"]
        model = g.build_markov_hydro_model()
        sddp.train(model, iteration_limit=3, seed=1, **KW)
        sims = sddp.simulate(model, 4, sampling_scheme=seq)
        assert [s[1]["noise_term"] for s in sims] == [
            (10.0, 1.4),
            (100.0, 0.75),
            (10.0, 1.4),
            (100.0, 0.75),
        ]
        sims = sddp.simulate(model, 20, sampling_scheme=prob, seed=1)
        assert {s[1]["noise_term"] for s in sims} <= {(10.0, 1.4), (100.0, 0.75)}


# ================================================================== use_multithreading
def test_use_multithreading(oracle, capsys):
    d = oracle["use_multithreading"]
    model = g.build_twelve_stage_model()
    g.train_threaded(model, 4, iteration_limit=10, seed=1, run_numerical_stability_report=False)
    out = capsys.readouterr().out
    assert "Threaded(4)" in out
    log = model.most_recent_training_results.log
    assert 10 <= len(log) <= 13 and sddp.termination_status(model) == "iteration_limit"
    assert close(sddp.calculate_bound(model), d["deterministic_equivalent"]) and close(
        d["bound_10"], 0.0
    )
    sims = g.simulate_threaded_with_thread_ids(model, 100, num_threads=4)
    assert len(sims) == 100 and all(len(s) == 12 for s in sims)
    ids = {s[0]["thread_id"] for s in sims}
    assert len(ids) > 1 and threading.get_ident() not in ids
    # The 4 replications quoted on the page come from different chunks/threads.
    assert len({sims[i][0]["thread_id"] for i in (0, 25, 50, 75)}) == 4


# ============================================================ write_subproblems_to_file
def test_write_subproblems_to_file(tmp_path):
    model = g.build_deterministic_equivalent_model()
    filename = str(tmp_path / "subproblem.lp")
    for omega in (1.1, 3.3):
        lp = g.parse_lp_file(g.write_parameterized_subproblem(model, omega, filename))
        assert lp["sense"] == "min"
        assert lp["objective"]["x_out"] == omega  # ω x.out
        theta = [v for v in lp["objective"] if v != "x_out"]
        assert len(theta) == 1 and lp["objective"][theta[0]] == 1.0  # + θ
        coefs, sense, rhs = lp["rows"]["balance"]
        assert coefs == {"x_in": 1.0, "x_out": -1.0, "y": -1.0} and sense == "=" and rhs == 0.0
        assert lp["bounds"]["y"] == (omega, omega)  # y fixed to ω
        assert lp["bounds"]["x_in"] == (-math.inf, math.inf)
        assert lp["bounds"]["x_out"] == (-math.inf, math.inf)
    assert not (tmp_path / "subproblem_1.lp").exists()
