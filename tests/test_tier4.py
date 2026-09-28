"""Parity tests for the Tier 4 modules against ``reference/oracle/*.json``."""

from __future__ import annotations

import json
import math
import pathlib
import re

import numpy as np
import pytest

import sddp
from tests.conftest import load_oracle
from tests.problems import build_hydro_thermal, build_markov_uncertainty

REF = pathlib.Path(__file__).resolve().parent.parent / "reference"
KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


def _scen(s):
    return [(tuple(n) if isinstance(n, list) else n, w) for n, w in s]


# ---------------------------------------------------------------- value functions
class TestValueFunctions:
    def test_hydro_thermal_deterministic(self):
        d = load_oracle("value_functions")["hydro_thermal_deterministic_20"]
        model = build_hydro_thermal()
        sddp.train(
            model,
            iteration_limit=20,
            sampling_scheme=sddp.Historical([_scen(s) for s in d["scenarios"]]),
            **KW,
        )
        for node, pts in d["nodes"].items():
            V = sddp.ValueFunction(model, node=int(node))
            for p in pts:
                value, duals = sddp.evaluate_value_function(V, {"volume": p["volume"]})
                assert close(value, p["value"]), (node, p, value)
                assert close(duals["volume"], p["dual"]), (node, p, duals)

    def test_multicut_cvar(self):
        d = load_oracle("value_functions")["multicut_cvar"]

        def builder(sp, t):
            x = sp.add_state("x", lb=0.0, initial_value=1.5)
            sp.add_constraint(x.out == x.in_)
            sp.parameterize(lambda w: sp.set_stage_objective(w * x.out), [1, 2])

        model = sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0)
        sddp.train(
            model, iteration_limit=2, risk_measure=sddp.CVaR(0.25), cut_type=sddp.MULTI_CUT, **KW
        )
        V = sddp.ValueFunction(model[1])
        for p in d:
            value, duals = sddp.evaluate_value_function(V, {"x": p["x"]})
            assert close(value, p["value"]) and close(duals["x"], p["dual"])
        assert repr(V) == "A value function for node 1"

    def test_objective_state(self):
        d = load_oracle("value_functions")["objective_state"]

        def builder(sp, t):
            x = sp.add_state("x", lb=0.0, initial_value=1.5)
            sp.add_objective_state(lambda p, w: p + w, initial_value=0.0, lipschitz=10.0)
            sp.add_constraint(x.out == x.in_)

            def modify(w):
                sp.set_stage_objective(sp.objective_state() * x.out)

            sp.parameterize(modify, [1, 2])

        model = sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0)
        sddp.train(model, iteration_limit=10, **KW)
        V = sddp.ValueFunction(model[1])
        with pytest.raises(AssertionError):
            sddp.evaluate_value_function(V, {"x": 1.0})
        for p in d:
            value, duals = sddp.evaluate_value_function(
                V, {"x": p["x"]}, objective_state=p["objective_state"]
            )
            assert close(value, p["value"]) and close(duals["x"], p["dual"])

    def test_belief_state(self):
        d = load_oracle("value_functions")["belief_state"]
        graph = sddp.MarkovianGraph([[[0.5, 0.5]], [[1.0, 0.0], [0.0, 1.0]]])
        graph.add_ambiguity_set([(1, 1), (1, 2)])
        graph.add_ambiguity_set([(2, 1), (2, 2)])

        def builder(sp, node):
            t, i = node
            x = sp.add_state("x", lb=0.0, initial_value=1.5)
            sp.add_constraint(x.out == x.in_)
            P = [[0.2, 0.8], [0.8, 0.2]]
            sp.parameterize(lambda w: sp.set_stage_objective(w * x.out), [1, 2], P[i - 1])

        model = sddp.PolicyGraph(builder, graph, lower_bound=0.0)
        sddp.train(model, iteration_limit=10, **KW)
        b = {(1, 1): 0.8, (1, 2): 0.2}
        for node in [(1, 1), (1, 2)]:
            V = sddp.ValueFunction(model[node])
            value, duals = sddp.evaluate_value_function(V, {"x": 1.0}, belief_state=b)
            assert close(value, d["value"], rel=1e-6, atol=1e-6) and close(duals["x"], d["dual"])


# ---------------------------------------------------------------- inner approximation
def build_inner(sp, t):
    reservoir = sp.add_state("reservoir", lb=5.0, ub=15.0, initial_value=10.0)
    thermal = sp.add_variable("thermal_generation", lb=0.0)
    hydro = sp.add_variable("hydro_generation", lb=0.0)
    spill = sp.add_variable("spill", lb=0.0)
    inflow = sp.add_variable("inflow")
    demand = sp.add_variable("demand")
    sp.add_constraint(reservoir.out == reservoir.in_ - hydro - spill + inflow)
    sp.add_constraint(hydro + thermal == demand)
    sp.set_stage_objective(10 * spill + thermal)

    def modify(w):
        sp.fix(inflow, w["inflow"])
        sp.fix(demand, w["demand"])

    sp.parameterize(
        modify,
        [
            {"inflow": 0.0, "demand": 7.5},
            {"inflow": 5.0, "demand": 5.0},
            {"inflow": 10.0, "demand": 2.5},
        ],
    )


class TestInner:
    def test_inner_dp(self):
        d = load_oracle("inner")
        model = sddp.LinearPolicyGraph(build_inner, stages=4, lower_bound=0.0)
        assert close(sddp.calculate_bound(model), d["initial_bound"])
        sddp.train(
            model,
            iteration_limit=50,
            sampling_scheme=sddp.Historical([_scen(s) for s in d["scenarios"]]),
            **KW,
        )
        assert close(sddp.calculate_bound(model), d["outer_bound_50"])
        ibf = sddp.InnerBellmanFunction(
            lambda t: 11.0 * (4 - t),
            upper_bound=lambda t: 17.5 * 11.0 * (4 - t),
            vertex_type=sddp.SINGLE_CUT,
        )
        inner, ub, _ = sddp.inner_dp(
            build_inner,
            model,
            stages=4,
            sense="Min",
            optimizer=sddp.HiGHS,
            lower_bound=0.0,
            bellman_function=ibf,
            risk_measure=sddp.Expectation(),
            print_level=0,
        )
        assert close(ub, d["inner_upper_bound"])
        assert close(sddp.calculate_bound(inner), d["inner_bound_recomputed"])
        assert d["outer_bound_50"] <= ub
        for node, verts in d["vertices"].items():
            py = inner[int(node)].bellman_function.global_theta.vertices
            assert len(py) == len(verts)
            for pv, jv in zip(py, verts):
                assert close(pv.value, jv["value"]) and close(
                    pv.state["reservoir"], jv["state"]["reservoir"]
                )
            assert sum(1 for v in py if v.variable_ref is not None) == d["active_vertices"][node]
        empty = sddp.PolicyGraph(
            build_inner, sddp.LinearGraph(4), lower_bound=0.0, bellman_function=ibf
        )
        for node in empty.nodes.values():
            sddp.algorithm.set_objective(node)
        assert close(sddp.calculate_bound(empty), d["empty_inner_bound"])

    def test_vertices_roundtrip(self, tmp_path):
        d = load_oracle("inner")
        model = sddp.LinearPolicyGraph(build_inner, stages=4, lower_bound=0.0)
        sddp.train(
            model,
            iteration_limit=50,
            sampling_scheme=sddp.Historical([_scen(s) for s in d["scenarios"]]),
            **KW,
        )
        ibf = sddp.InnerBellmanFunction(
            lambda t: 11.0 * (4 - t),
            upper_bound=lambda t: 17.5 * 11.0 * (4 - t),
            vertex_type=sddp.SINGLE_CUT,
        )
        inner, ub, _ = sddp.inner_dp(
            build_inner,
            model,
            stages=4,
            optimizer=sddp.HiGHS,
            lower_bound=0.0,
            bellman_function=ibf,
            risk_measure=sddp.Expectation(),
            print_level=0,
        )
        f = tmp_path / "vertices.json"
        sddp.write_vertices_to_file(inner, str(f))
        model2 = sddp.PolicyGraph(
            build_inner, sddp.LinearGraph(4), lower_bound=0.0, bellman_function=ibf
        )
        for node in model2.nodes.values():
            sddp.algorithm.set_objective(node)
        with pytest.raises(ValueError):
            sddp.read_vertices_from_file(model2, str(f), vertex_selection=True)
        sddp.read_vertices_from_file(model2, str(f), vertex_selection=True, optimizer=sddp.HiGHS)
        assert close(sddp.calculate_bound(model2), ub, rel=1e-6)

    def test_inner_policy_graph_20_stages(self):
        d = load_oracle("inner")
        cut_model = sddp.LinearPolicyGraph(build_inner, stages=20, lower_bound=0.0)
        sddp.train(
            cut_model,
            iteration_limit=200,
            sampling_scheme=sddp.Historical([_scen(s) for s in d["scenarios_20"]]),
            **KW,
        )
        assert close(sddp.calculate_bound(cut_model), d["cut_model_20_bound"])
        for optimizer, key in [
            (sddp.HiGHS, "vertex_model_20_bound"),
            (None, "vertex_model_20_bound_no_selection"),
        ]:
            vm = sddp.InnerPolicyGraph(
                build_inner,
                sddp.LinearGraph(20),
                lower_bound=0.0,
                upper_bound=1000,
                lipschitz_constant=10.0,
            )
            sddp.dp_vertices_from_visited_states(vm, cut_model, optimizer=optimizer, print_level=0)
            assert close(sddp.calculate_bound(vm), d[key])
        sims = sddp.simulate(vm, 3, ["vertex_coverage_distance"])
        assert all("vertex_coverage_distance" in s for sim in sims for s in sim)

    def test_inner_policy_graph_errors_and_train(self):
        g = sddp.LinearGraph(4)
        with pytest.raises(ValueError):
            sddp.InnerPolicyGraph(
                build_inner, g, lower_bound=0.0, upper_bound=1000.0
            )  # no Lipschitz
        with pytest.raises(ValueError):
            sddp.InnerPolicyGraph(
                build_inner, g, lower_bound=0.0, lipschitz_constant=10.0
            )  # no upper bound
        with pytest.raises(ValueError):
            sddp.InnerPolicyGraph(
                build_inner,
                sddp.UnicyclicGraph(0.9, num_nodes=4),
                lower_bound=0.0,
                upper_bound=1.0,
                lipschitz_constant=10.0,
            )
        model = sddp.InnerPolicyGraph(
            build_inner,
            sddp.LinearGraph(20),
            lower_bound=0.0,
            upper_bound=1000,
            lipschitz_constant=10.0,
        )
        sddp.train(model, iteration_limit=100, **KW)
        assert sddp.calculate_bound(model) >= 241.38  # an upper bound must stay above the optimum
        with pytest.raises(NotImplementedError):
            sddp.train(
                model, iteration_limit=2, cut_type=sddp.MULTI_CUT, add_to_existing_cuts=True, **KW
            )


# ---------------------------------------------------------------- MSPFormat
class TestMSPFormat:
    @pytest.mark.parametrize("name", ["hydro_thermal", "electric"])
    def test_read_train(self, name):
        d = load_oracle("mspformat")[name]
        model = sddp.read_msp_format(str(REF / "msp" / name))
        assert sorted(model.nodes) == d["nodes"]
        assert {k: len(n.noise_terms) for k, n in model.nodes.items()} == {
            k: v for k, v in d["noise_counts"].items()
        }
        assert {
            k: [(c.term, c.probability) for c in n.children] for k, n in model.nodes.items()
        } == {k: [tuple(c) for c in v] for k, v in d["children"].items()}
        assert model.initial_root_state == d["initial_state"]
        det = sddp.deterministic_equivalent(sddp.read_msp_format(str(REF / "msp" / name)))
        det.optimize()
        assert close(det.objective_value(), d["deterministic_equivalent"])
        sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound"])

    def test_electric_tree(self):
        d = load_oracle("mspformat")["electric_tree"]
        model = sddp.read_msp_format(
            str(REF / "msp" / "electric.problem.json"),
            str(REF / "msp" / "electric-tree.lattice.json"),
        )
        assert sorted(model.nodes) == d["nodes"]
        assert {
            k: [(c.term, c.probability) for c in n.children] for k, n in model.nodes.items()
        } == {k: [tuple(c) for c in v] for k, v in d["children"].items()}
        sddp.train(model, iteration_limit=30, seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound"])

    def test_get_constant(self):
        from sddp.msp_format import _get_constant

        state = {"inflow": 12.0}
        assert (
            _get_constant([1.0]) == 1.0
            and _get_constant(["inf"]) == math.inf
            and _get_constant(["-inf"]) == -math.inf
        )
        assert _get_constant(["inflow"]) == ["inflow"] and _get_constant(["inflow"], state) == 12.0
        terms = [{"ADD": "inflow"}, {"ADD": 200.0}, {"ADD": [1.0]}]
        assert _get_constant(terms) is terms and _get_constant(terms, state) == 213.0
        terms = [{"ADD": "inflow"}, {"MUL": -1.0}, {"MUL": 0.5}]
        assert _get_constant(terms, state) == -6.0
        assert _get_constant("bad_inflow", state) == 0.0


# ---------------------------------------------------------------- StochOptFormat
def build_experimental(minimization: bool):
    def builder(sp, t):
        N, C, S, DEMAND = 2, [0.2, 0.7], [2.33, 2.54], [2, 10]
        x = [sp.add_state(f"x[{i + 1}]", lb=0.0, initial_value=0.0) for i in range(N)]
        s = [sp.add_variable(f"s[{i + 1}]", lb=0.0) for i in range(N)]
        d = sp.add_variable("d")
        for i in range(N):
            sp.add_constraint(s[i] <= x[i].in_)
        c = sp.add_constraint(s[0] + s[1] <= d + 1, name="c")

        def modify(w):
            sp.fix(d, DEMAND[w - 1])
            sp.set_upper_bound(s[0], 0.1 * w)
            sp.set_lower_bound(x[0].out, w)
            sp.set_normalized_rhs(c, w)
            sgn = 1.0 if minimization else -1.0
            sp.set_stage_objective(
                sgn
                * (
                    sum(C[i] * x[i].out for i in range(N))
                    - S[w - 1] * s[w - 1]
                    - s[w - 1] * S[w - 1]
                    + w
                )
            )

        sp.parameterize(modify, [1] if t == 1 else [1, 2])

    return sddp.PolicyGraph(
        builder,
        sddp.LinearGraph(3),
        sense="Min" if minimization else "Max",
        lower_bound=-50.0,
        upper_bound=50.0,
    )


class TestStochOptFormat:
    def test_electric(self):
        d = load_oracle("stochoptformat")["electric"]
        model, validation = sddp.read_from_file(str(REF / "sof" / "electric.sof.json"))
        assert len(model.nodes) == d["n_nodes"] and sorted(model.nodes) == d["nodes"]
        assert validation is not None and validation.sha256 == d["sha256"]
        sddp.train(model, iteration_limit=60, seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound"])
        ev = sddp.evaluate_validation_scenarios(model, validation)
        assert ev["problem_sha256_checksum"] == d["sha256"]
        assert len(ev["scenarios"]) == 3
        for py, jl in zip(ev["scenarios"], d["scenario_objectives"]):
            assert close(sum(s["objective"] for s in py), sum(jl))

    @pytest.mark.parametrize("label,minimization", [("min", True), ("max", False)])
    def test_experimental_julia_file(self, label, minimization):
        d = load_oracle("stochoptformat")[f"experimental_{label}"]
        model, validation = sddp.read_from_file(str(REF / "sof" / f"experimental_{label}.sof.json"))
        assert validation.sha256 == d["sha256"]
        sddp.train(model, iteration_limit=50, seed=3, **KW)
        assert close(sddp.calculate_bound(model), d["roundtrip_bound"])
        assert close(sddp.calculate_bound(model), d["deterministic_equivalent"])
        ev = sddp.evaluate_validation_scenarios(model, validation)
        assert len(ev["scenarios"]) == 10 and len(ev["scenarios"][0]) == 3
        for py, jl in zip(ev["scenarios"], d["scenario_objectives"]):
            for a, b in zip(py, jl):
                assert close(a["objective"], b)
        n11 = ev["scenarios"][0][0]
        assert n11["primal"]["d"] == 2 and close(
            n11["primal"]["x[1]_out"], d["scenario_primals"][0][0]["x[1]_out"]
        )

    @pytest.mark.parametrize("minimization", [True, False])
    def test_python_roundtrip(self, tmp_path, minimization):
        d = load_oracle("stochoptformat")[
            "experimental_min" if minimization else "experimental_max"
        ]
        model = build_experimental(minimization)
        f = tmp_path / "m.sof.json"
        sddp.write_to_file(
            model,
            str(f),
            validation_scenarios=10,
            sampling_scheme=sddp.PSRSamplingScheme(2),
            name="Experimental",
            author="me",
        )
        data = json.load(open(f))
        assert data["author"] == "me" and len(data["validation_scenarios"]) == 10
        new_model, vs = sddp.read_from_file(str(f))
        assert len(vs.scenarios) == 10 and len(vs.sha256) == 64
        sddp.train(new_model, iteration_limit=50, seed=3, **KW)
        assert close(sddp.calculate_bound(new_model), d["deterministic_equivalent"])
        ev = sddp.evaluate_validation_scenarios(new_model, vs)
        demands = [[s["primal"]["d"] for s in sc] for sc in ev["scenarios"]]
        for i in range(2, 10, 2):
            assert demands[0] == demands[i] and demands[1] == demands[i + 1]  # PSR(2) cycles
        sddp.train(model, iteration_limit=1, **KW)
        with pytest.raises(ValueError, match="after a call"):
            sddp.write_to_file(model, str(f))


# ---------------------------------------------------------------- lattice
class TestLattice:
    def test_lattice_approximation(self):
        d = load_oracle("lattice")
        support, probability = sddp.lattice_approximation(None, d["states"], 60, d["simulations"])
        for s, js in zip(support, d["support"]):
            assert np.allclose(s, js, rtol=0, atol=1e-12)
        for p, jp in zip(probability, d["probability"]):
            assert np.allclose(p, np.array(jp), rtol=0, atol=1e-12)
        assert sddp.allocate_support_budget(d["simulations"], 10, 60) == d["budget_alloc_10"]
        assert sddp.allocate_support_budget(d["simulations"], 3, 60) == d["budget_alloc_3"]
        from sddp.modeling_aids import find_min

        assert [list(find_min([1.0, 2.0, 3.0], y)) for y in (2.1, 0.0, 5.0)] == d["find_min"]
        g8 = d["graph8"]
        states = sddp.allocate_support_budget(d["simulations"], 8, 60)
        assert states == g8["states"]
        support, probability = sddp.lattice_approximation(None, states, 60, d["simulations"])
        for s, js in zip(support, g8["support"]):
            assert np.allclose(s, js, rtol=0, atol=1e-12)

    def test_markovian_graph_from_simulator_and_sampling(self):
        import random

        r = random.Random(1)
        g = sddp.markovian_graph_from_simulator(
            lambda: list(np.cumsum([r.random() for _ in range(5)])), budget=10, scenarios=100
        )
        assert g.root_node == (0, 0.0) and len(g.nodes) == 11
        for arcs in g.nodes.values():
            if arcs:
                assert close(sum(p for _, p in arcs), 1.0)

        def builder(sp, node):
            t, ms = node
            x = sp.add_state("x", lb=0.0, initial_value=1.0)
            u = sp.add_variable("u", lb=0.0)
            sp.add_constraint(x.out == x.in_ - u)
            omega = [(ms, {"u": u_max}) for u_max in (0.0, 0.5)]

            def modify(w):
                sp.set_upper_bound(u, w[1]["u"])
                sp.set_stage_objective(w[0] * u)

            sp.parameterize(modify, omega)

        model = sddp.PolicyGraph(builder, g, sense="Max", upper_bound=12.0)
        sddp.train(
            model,
            iteration_limit=10,
            sampling_scheme=sddp.SimulatorSamplingScheme(
                lambda: list(np.cumsum([r.random() for _ in range(5)]))
            ),
            **KW,
        )
        assert model.most_recent_training_results.status == "iteration_limit"


# ---------------------------------------------------------------- biobjective
def test_biobjective():
    d = load_oracle("biobjective")["solutions"]

    def builder(sp, _):
        v = sp.add_state("v", lb=0.0, ub=200.0, initial_value=50.0)
        g = [sp.add_variable(f"g[{i}]", lb=0.0, ub=100.0) for i in (1, 2)]
        u = sp.add_variable("u", lb=0.0, ub=150.0)
        s = sp.add_variable("s", lb=0.0)
        shortage = sp.add_variable("shortage_cost", lb=0.0)
        objective_1 = g[0] + 10 * g[1]
        objective_2 = 1.0 * shortage
        inflow = sp.add_constraint(v.out == v.in_ - u - s)
        sp.add_constraint(g[0] + g[1] + u == 150)
        sp.add_constraint(shortage >= 40 - v.out)
        sp.add_constraint(shortage >= 60 - 2 * v.out)
        sp.add_constraint(shortage >= 80 - 4 * v.out)
        sddp.initialize_biobjective_subproblem(sp)

        def modify(w):
            sp.set_normalized_rhs(inflow, w)
            sddp.set_biobjective_functions(sp, objective_1, objective_2)

        sp.parameterize(modify, [5.0 * i for i in range(11)])

    model = sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0)
    sols = sddp.train_biobjective(
        model, solution_limit=5, iteration_limit=500, print_level=0, seed=7
    )
    assert sorted(sols) == sorted(float(k) for k in d)
    for k, v in d.items():
        assert close(sols[float(k)], v), (k, sols[float(k)], v)
    ordered = sorted(sols.items())
    grads = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(ordered, ordered[1:])]
    assert all(g2 < g1 for g1, g2 in zip(grads, grads[1:]))


# ---------------------------------------------------------------- misc
class TestMisc:
    def test_numerical_stability_report(self, capsys):
        d = load_oracle("misc_tier4")
        assert (
            sddp.numerical_stability_report(build_hydro_thermal(), print=False)
            == d["hydro_thermal_report"]
        )

        def bad(sp, t):
            x = sp.add_state("x", lb=0.0, ub=1e9, initial_value=1.0)
            u = sp.add_variable("u", lb=0.0)
            sp.add_constraint(1e-6 * x.out + u >= 1e-5)
            sp.set_stage_objective(1e8 * u + x.out)

        m = sddp.LinearPolicyGraph(bad, stages=2, lower_bound=0.0)
        assert sddp.numerical_stability_report(m, print=False) == d["bad_report"]
        assert (
            sddp.numerical_stability_report(m, print=False, by_node=True) == d["bad_report_by_node"]
        )
        sddp.train(m, iteration_limit=1, print_level=1)
        assert "WARNING: numerical stability issues detected" in capsys.readouterr().out

    def test_importance_sampling_and_alternative_forward(self):
        d = load_oracle("misc_tier4")
        model = build_hydro_thermal()
        sddp.train(
            model,
            iteration_limit=60,
            seed=1,
            forward_pass=sddp.ImportanceSamplingForwardPass(),
            cut_type=sddp.MULTI_CUT,
            **KW,
        )
        assert close(sddp.calculate_bound(model), d["importance_sampling_bound_60"])
        model, fwd = build_hydro_thermal(), build_hydro_thermal()
        sddp.train(
            model,
            iteration_limit=40,
            seed=1,
            forward_pass=sddp.AlternativeForwardPass(fwd),
            post_iteration_callback=sddp.AlternativePostIterationCallback(fwd),
            **KW,
        )
        assert close(sddp.calculate_bound(model), d["alternative_forward_bound_40"])
        assert close(sddp.calculate_bound(fwd), d["alternative_forward_model_bound_40"])

    def test_logging_forward_pass_and_csv_log(self, tmp_path):
        model = build_hydro_thermal()
        f = tmp_path / "states.csv"
        sddp.train(
            model,
            iteration_limit=3,
            seed=1,
            forward_pass=sddp.LoggingForwardPass(filename=str(f)),
            **KW,
        )
        lines = f.read_text().splitlines()
        assert lines[0] == "iteration,index,volume" and len(lines) == 1 + 3 * 3
        g = tmp_path / "log.csv"
        sddp.write_log_to_csv(model, str(g))
        assert g.read_text().splitlines()[0] == "iteration, simulation, bound, time"

    def test_binary_expansion(self):
        d = load_oracle("misc_tier4")["binexpand"]
        assert (
            sddp.binexpand(5, 5) == d["5_5"]
            and sddp.binexpand(0.56, 0.56, 0.1) == d["0.56_0.56_0.1"]
        )
        assert sddp.binexpand(0.54, 0.54, 0.01) == d["0.54_0.54_0.01"]
        assert sddp.bincontract([1, 0, 1]) == d["bincontract_101"] and close(
            sddp.bincontract([1, 0, 1], 0.1), d["bincontract_eps"]
        )
        assert [sddp.binexpand(i, i) for i in range(1, 8)] == [
            [1],
            [0, 1],
            [1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        with pytest.raises(ValueError):
            sddp.binexpand(8, 7)
        with pytest.raises(ValueError):
            sddp.binexpand(-1, 5)
        with pytest.raises(ValueError):
            sddp.binexpand(5, 0)


# ---------------------------------------------------------------- multiprocess
def test_multiprocess_train_and_simulate():
    d = load_oracle("hydro_thermal")
    model = build_hydro_thermal()
    sddp.train(
        model,
        iteration_limit=40,
        seed=1,
        parallel_scheme=sddp.Multiprocess(build_hydro_thermal, num_workers=2),
        **KW,
    )
    assert close(sddp.calculate_bound(model), d["deterministic_equivalent"])
    assert {log.pid for log in model.most_recent_training_results.log} <= {0, 1, 2}
    sims = sddp.simulate(
        model, 20, ["volume"], parallel_scheme=sddp.Multiprocess(build_hydro_thermal, num_workers=2)
    )
    assert len(sims) == 20
    d = load_oracle("markov_uncertainty")
    model = build_markov_uncertainty()
    sddp.train(
        model,
        iteration_limit=80,
        seed=1,
        parallel_scheme=sddp.Multiprocess(build_markov_uncertainty, num_workers=2),
        **KW,
    )
    assert close(sddp.calculate_bound(model), d["deterministic_equivalent"])


# ---------------------------------------------------------------- visualisation
class TestVisualisation:
    def test_spaghetti_html_matches_julia_control(self, tmp_path):
        simulations = [
            [{"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 5.0}, {"x": 3.0, "y": 6.0}],
            [{"x": 1.5, "y": 4.5}, {"x": 2.5, "y": 5.5}, {"x": 3.5, "y": 6.5}],
        ]
        plt = sddp.SpaghettiPlot(simulations)
        plt.add_spaghetti(lambda data: data["x"], cumulative=True)
        plt.add_spaghetti(lambda data: 2 * data["y"], title="y")
        assert repr(plt) == "A spaghetti plot with 2 scenarios and 3 stages."
        f = plt.plot(str(tmp_path / "test.html"))
        py = open(f, encoding="utf-8").read()
        jl = open(REF / "control.html", encoding="utf-8").read()

        def data_of(html):
            m = re.search(r"main\( (\[.*?\])\);", html, re.S)
            assert m, "plot data not found"
            return json.loads(m.group(1))

        assert data_of(py) == data_of(jl)

    def test_graph_and_value_function_html(self, tmp_path):
        model = build_hydro_thermal()
        f = sddp.plot_graph(model, str(tmp_path / "g.html"))
        assert "id: '1'" in open(f).read() and "has_noise: true" in open(f).read()
        f2 = sddp.plot_graph(sddp.LinearGraph(2), str(tmp_path / "g2.html"))
        assert "id: '2'" in open(f2).read()
        sddp.train(model, iteration_limit=5, **KW)
        V = sddp.ValueFunction(model, node=1)
        f3 = sddp.plot_value_function_html(V, str(tmp_path / "v.html"), volume=[0.0, 100.0, 200.0])
        assert "<!--X-->" not in open(f3).read()
        import matplotlib

        matplotlib.use("Agg")
        ax = sddp.plot_value_function(V, volume=np.linspace(0, 200, 5))
        assert ax is not None

    def test_dashboard(self):
        import urllib.request

        cb = sddp.launch_dashboard(port=8765, open=False)
        model = build_hydro_thermal()
        sddp.train(model, iteration_limit=2, **KW)
        for log in model.most_recent_training_results.log:
            cb(log, False)
        with urllib.request.urlopen("http://127.0.0.1:8765/", timeout=5) as r:
            line = r.readline().decode()
            assert line.startswith("event: iteration")
        cb(None, True)
