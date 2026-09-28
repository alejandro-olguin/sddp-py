"""sddp.dev tutorial pages (warnings, arma, decision_hazard, production_planning, batteries,
inventory) vs the Julia oracle ``reference/oracle/tutorials_a.json``."""

from __future__ import annotations

import math
import random
import statistics

import numpy as np
import pytest
from examples import tutorial_modelling as tm

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


@pytest.fixture(scope="module")
def oracle_a():
    return load_oracle("tutorials_a")


def _det(build):
    de = sddp.deterministic_equivalent(build())
    de.optimize()
    return de.objective_value()


# =============================================================================== warnings
class TestWarnings:
    def test_relatively_complete_recourse_raises(self, oracle_a, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = tm.build_relatively_complete_recourse()
        with pytest.raises(RuntimeError) as excinfo:
            sddp.train(model, iteration_limit=1, print_level=0)
        msg = str(excinfo.value)
        assert "relatively complete recourse" in msg
        assert "Unable to retrieve solution from node 2" in msg
        assert "subproblem_2.lp" in msg
        assert (tmp_path / "subproblem_2.lp").exists()
        assert (tmp_path / "subproblem_2.lp").stat().st_size > 0
        assert "INFEASIBLE" in msg
        # SDDP.jl raises the same diagnosis for the same node.
        julia = oracle_a["warnings"]["recourse_error"]
        assert "relatively complete recourse" in julia
        assert "node 2" in julia

    def test_numerical_stability_report_matches_julia(self, oracle_a, capsys):
        d = oracle_a["warnings"]
        model = tm.build_badly_scaled()
        assert sddp.numerical_stability_report(model, print=False) == d["report"]
        assert (
            sddp.numerical_stability_report(model, print=False, by_node=True)
            == (d["report_by_node"])
        )
        text = sddp.numerical_stability_report(model)
        assert capsys.readouterr().out == text
        assert "WARNING: numerical stability issues detected" in text

    @pytest.mark.parametrize("lb,expected", [(0.0, 3.5), (10.0, 11.0)])
    def test_choosing_an_initial_bound(self, oracle_a, lb, expected):
        model = tm.build_initial_bound_model(lb)
        sddp.train(model, iteration_limit=5, **KW)
        assert sddp.calculate_bound(model) == expected
        julia = oracle_a["warnings"][f"initial_bound_lb_{int(lb)}"]
        assert sddp.calculate_bound(model) == julia["bound"]
        # The model is deterministic, so the whole bound trajectory is reproducible.
        log = model.most_recent_training_results.log
        assert [x.bound for x in log] == [x["bound"] for x in julia["log"]]
        assert [x.simulation_value for x in log] == [x["simulation_value"] for x in julia["log"]]
        assert _det(lambda: tm.build_initial_bound_model(lb)) == pytest.approx(3.5)

    def test_run_warnings(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        out = tm.run_warnings(print_level=0)
        assert "relatively complete recourse" in out["error"]
        assert out["bound_lb_0"] == 3.5 and out["bound_lb_10"] == 11.0
        assert capsys.readouterr().out == ""


# =================================================================================== arma
class TestARMA:
    def test_state_space_expansion(self, oracle_a):
        d = oracle_a["arma"]["state_space"]
        assert close(_det(tm.build_state_space_expansion), d["deterministic_equivalent"])
        model = tm.build_state_space_expansion()
        sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound"])
        assert close(sddp.calculate_bound(model), d["deterministic_equivalent"])
        sims = sddp.simulate(model, 2, ["inflow", "ε"], seed=1)
        for sim in sims:
            inflow = 50.0
            for s in sim:
                assert s["ε"] in tm.ARMA_OMEGA
                assert close(s["inflow"].in_, inflow)
                inflow += s["ε"]
                assert close(s["inflow"].out, inflow)

    def test_var_model(self, oracle_a):
        d = oracle_a["arma"]["var"]
        assert close(_det(tm.build_var_model), d["deterministic_equivalent"])
        model = tm.build_var_model()
        assert len(model[1].noise_terms) == 9
        sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW)
        assert close(sddp.calculate_bound(model), d["bound"])
        assert close(sddp.calculate_bound(model), d["deterministic_equivalent"])

    def test_simulator(self):
        rng = random.Random(0)
        sim = tm.make_simulator(rng)
        path = sim()
        assert len(path) == 3
        assert close(path[0] - 50.0, min(tm.ARMA_OMEGA, key=lambda w: abs(w - (path[0] - 50))))
        assert len(tm.simulator()) == 3

    def test_markov_chain_graph_shape_and_training(self, oracle_a):
        d = oracle_a["arma"]["markov_chain"]
        julia_shapes = {tuple(g["per_stage"]) for g in d["graphs"].values()}
        # `budget` is an upper bound: coincident support points collapse into one node (Julia's
        # seed 4 has 7 nodes), so 7 or 8 nodes can occur; stage counts still sum to the budget.
        for g in d["graphs"].values():
            assert 7 <= g["num_nodes"] <= 8 and g["num_nodes_including_root"] == g["num_nodes"] + 1
            assert sum(g["per_stage"]) == g["num_nodes"] and min(g["per_stage"]) >= 1
        for seed in (1, 2, 3):
            rng = random.Random(seed)
            graph = tm.build_markov_chain_graph(tm.make_simulator(rng), budget=8, scenarios=30)
            nodes = [n for n in graph.nodes if n != graph.root_node]
            assert graph.root_node == (0, 0.0)
            assert 7 <= len(nodes) <= 8 and len(graph.nodes) == len(nodes) + 1  # root key
            per_stage = [sum(1 for n in nodes if n[0] == t) for t in (1, 2, 3)]
            assert sum(per_stage) == len(nodes) and min(per_stage) >= 1
            assert {n[0] for n in nodes} == {1, 2, 3}
            # Julia allocates the budget by per-stage variance; the fitted split is one of
            # the splits SDDP.jl produced for its own seeds.
            assert tuple(per_stage) in julia_shapes, (per_stage, julia_shapes)
            for parent, children in graph.nodes.items():
                for child, p in children:
                    assert child[0] == parent[0] + 1 and 0.0 <= p <= 1.0
                if parent[0] < 3:
                    assert close(sum(p for _, p in children), 1.0)
                else:
                    assert children == []
            for n in nodes:  # node values are inflows on the random walk's range
                assert 50 - 30 <= n[1] <= 50 + 30
            model = tm.build_markov_chain_model(graph)
            sddp.train(model, iteration_limit=20, seed=seed, **KW)
            assert model.most_recent_training_results.status == "iteration_limit"
            bound = sddp.calculate_bound(model)
            assert 0.0 < bound < 150 * 300  # finite, sensible cost
            sims = sddp.simulate(model, 5, ["x"], seed=seed)
            for sim in sims:
                assert [s["node_index"][0] for s in sim] == [1, 2, 3]
                assert all(s["node_index"] in graph.nodes for s in sim)
        # Julia's bound on its own fitted graphs is the same order of magnitude.
        julia_bounds = [g["bound_20"] for g in d["graphs"].values()]
        assert min(julia_bounds) < bound * 1.5 and bound < max(julia_bounds) * 1.5

    def test_plot_graph(self, tmp_path):
        rng = random.Random(1)
        graph = tm.build_markov_chain_graph(tm.make_simulator(rng))
        model = tm.build_markov_chain_model(graph)
        f = tmp_path / "model_arma.html"
        sddp.plot_graph(model, str(f))
        assert f.stat().st_size > 0


# ========================================================================= decision_hazard
class TestDecisionHazard:
    @pytest.mark.parametrize(
        "name,build,expected",
        [
            ("hazard_decision", tm.build_hazard_decision, 400.0),
            ("decision_hazard", tm.build_decision_hazard, 650.0),
            ("decision_hazard_2", tm.build_decision_hazard_2, 410.0),
        ],
    )
    def test_cost(self, oracle_a, name, build, expected, capsys):
        d = oracle_a["decision_hazard"][name]
        assert close(_det(build), d["deterministic_equivalent"])
        assert close(d["deterministic_equivalent"], expected)
        model = build()
        cost = tm.train_and_compute_cost(model, seed=1)
        assert capsys.readouterr().out == f"Cost = ${cost}\n"
        assert close(cost, expected) and close(cost, d["bound"])
        assert model.most_recent_training_results.status == "simulation_stopping"
        assert d["status"] == "simulation_stopping"

    def test_page_claims(self):
        models = {
            "hd": tm.build_hazard_decision(),
            "dh": tm.build_decision_hazard(),
            "dh2": tm.build_decision_hazard_2(),
        }
        costs = {}
        for k, m in models.items():
            sddp.train(m, seed=1, **KW)
            costs[k] = sddp.calculate_bound(m)
        assert close(costs["dh"] - costs["hd"], 250.0)  # "adds a cost of $250"
        assert close(costs["dh2"] - costs["hd"], 10.0)  # "a much more reasonable cost of $10"
        # the here-and-now thermal decision is a state; the first stage of the fixed model
        # chooses it and leaves storage unchanged
        sims = sddp.simulate(models["dh2"], 4, ["u_thermal", "x_storage"], seed=1)
        for sim in sims:
            assert len(sim) == 5
            assert close(sim[0]["stage_objective"], 0.0)
            assert close(sim[0]["x_storage"].out, 6.0)
            assert sim[0]["u_thermal"].in_ == 0.0 and sim[0]["u_thermal"].out >= 0.0
            for a, b in zip(sim, sim[1:]):
                assert close(a["u_thermal"].out, b["u_thermal"].in_)
                assert b["noise_term"] in (2, 3)

    def test_graph_plots(self, tmp_path):
        for name, build in (
            ("model_hazard_decision", tm.build_hazard_decision),
            ("model_decision_hazard", tm.build_decision_hazard),
            ("model_decision_hazard2", tm.build_decision_hazard_2),
        ):
            f = tmp_path / f"{name}.html"
            sddp.plot_graph(build(), str(f))
            assert f.stat().st_size > 0


# ===================================================================== production_planning
class TestProductionPlanning:
    def test_structure(self):
        model = tm.build_production_planning()
        assert len(model.nodes) == 10
        assert len(model[1].noise_terms) == 1 and model[1].noise_terms[0].term is None
        assert len(model[2].noise_terms) == 5
        assert model[2].noise_terms[0].term == tm.DEMAND_SCENARIOS[0]
        sp = model[2].subproblem
        assert sp.model.is_fixed(sp["w[Chicago]"])
        assert sorted(model[1].states) == ["x[San-Diego]", "x[Seattle]"]

    def test_parity_and_plots(self, oracle_a):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        d = oracle_a["production_planning"]
        model = tm.build_production_planning()
        sddp.train(
            model,
            iteration_limit=d["iteration_limit"],
            risk_measure=sddp.Expectation(),
            seed=1,
            **KW,
        )
        bound = sddp.calculate_bound(model)
        assert close(bound, d["bound"], rel=1e-3), (bound, d["bound"])
        assert bound <= d["bound_1000"] * (1 + 1e-3)
        variables = [f"x[{p}]" for p in tm.PLANTS] + [f"u_prod[{p}]" for p in tm.PLANTS]
        sims = sddp.simulate(model, d["simulation"]["replications"], variables, seed=42)
        objs = [sum(s["stage_objective"] for s in sim) for sim in sims]
        assert len(sims) == 50 and all(len(sim) == 10 for sim in sims)
        mu, sd, n = statistics.mean(objs), statistics.stdev(objs), len(objs)
        js = d["simulation"]
        half = 2.576 * math.sqrt(sd**2 / n + js["std"] ** 2 / js["replications"])
        assert abs(mu - js["mean"]) <= half, (mu, js["mean"], half)
        # first stage is deterministic: no demand, so production is only stockpiling
        for sim in sims:
            assert sim[0]["noise_term"] is None
            for p in tm.PLANTS:
                assert 0.0 <= sim[0][f"u_prod[{p}]"] <= tm.PLANT_CAPACITY[p] + 1e-9
                assert close(sim[0][f"x[{p}]"].out, sim[0][f"u_prod[{p}]"])
        fig = tm.plot_production_planning(sims)
        assert len(fig.axes) == 4
        assert fig.axes[0].get_title() == "San-Diego" and fig.axes[0].get_ylabel() == "u_prod"
        assert fig.axes[2].get_xlabel() == "Week"
        ax = sddp.publication_plot(sims, lambda data: data["u_prod[San-Diego]"], title="San-Diego")
        assert ax.get_title() == "San-Diego"
        matplotlib.pyplot.close("all")


# ============================================================================== batteries
class TestBatteries:
    def test_structure(self):
        model = tm.build_batteries()
        assert len(model.nodes) == 24
        node = model[7]
        assert [n.term for n in node.noise_terms] == tm.BATTERY_OMEGA
        assert "λ" in node.subproblem
        sddp.parameterize(node, 2.0)
        assert node.model.fix_value(node.subproblem["w_load"]) == tm.BATTERY_LOAD[6] + 2.0

    def test_parity_and_duals(self, oracle_a):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        d = oracle_a["batteries"]
        model, results = tm.run_batteries(
            iteration_limit=d["iteration_limit"],
            replications=100,
            seed=1,
            parallel_scheme=sddp.Serial(),
            print_level=0,
        )
        bound = sddp.calculate_bound(model)
        assert close(bound, d["bound"], rel=1e-3), (bound, d["bound"])
        assert bound <= d["bound_1500"] * (1 + 1e-3)
        lam = np.array([[s["λ"] for s in sim] for sim in results])
        assert lam.shape == (100, 24)
        # The energy price is the thermal fuel price ($70) in the early hours ...
        frac_70 = [(np.isclose(lam[:, h], 70.0)).mean() for h in range(6)]
        assert all(f >= 0.9 for f in frac_70), frac_70
        assert all(f >= 0.9 for f in d["lambda_frac_70_hours_1_6"])
        # ... it is a minimisation sensitivity (cost per unit of load), never negative, and
        # spikes to at most the value of lost load ($500) in the evening peak.
        assert lam.min() >= -1e-6 and lam.max() <= 500.0 + 1e-6
        assert d["lambda_min"] >= -1e-6 and d["lambda_max"] <= 500.0 + 1e-6
        assert lam[:, 19:21].max() > 70.0  # evening peak above the fuel price
        assert lam[:, 13:17].mean() < 70.0  # afternoon price below the fuel price
        mean_by_hour = lam.mean(axis=0)
        assert np.allclose(mean_by_hour[:6], d["lambda_mean_by_hour"][:6], atol=1e-6)
        assert np.abs(mean_by_hour - np.array(d["lambda_mean_by_hour"])).max() < 25.0
        # battery ends the day empty (finite horizon, no terminal value)
        assert all(sim[-1]["x_soc"].out <= 1e-6 for sim in results)
        assert d["x_soc_out_hour_24_max"] <= 1e-6
        objs = [sum(s["stage_objective"] for s in sim) for sim in results]
        mu, sd, n = statistics.mean(objs), statistics.stdev(objs), len(objs)
        js = d["simulation"]
        half = 2.576 * math.sqrt(sd**2 / n + js["std"] ** 2 / js["replications"])
        assert abs(mu - js["mean"]) <= half, (mu, js["mean"], half)
        fig = tm.plot_batteries(results)
        assert [ax.get_ylabel() for ax in fig.axes] == [
            "Net load",
            "Thermal generation",
            "State of charge",
            "Dual λ",
            "Lost load",
        ]
        matplotlib.pyplot.close("all")


# ============================================================================== inventory
class TestInventory:
    def test_structure(self):
        model = tm.build_inventory_finite()
        assert len(model.nodes) == 11
        assert model[1].noise_terms[0].term is None
        assert [n.term for n in model[2].noise_terms] == tm.INV_OMEGA
        assert close(tm.INV_OMEGA[-1], 800.0) and len(tm.INV_OMEGA) == 20
        sp1, sp11 = model[1].subproblem, model[11].subproblem
        assert sp1.model.is_fixed(sp1["u_sell"]) and sp1.model.fix_value(sp1["u_sell"]) == 0.0
        assert sp11.model.is_fixed(sp11["u_buy"].out)
        assert not sp1.model.is_fixed(sp1["u_buy"].out)
        g = tm.build_inventory_graph()
        assert repr(g) == (
            "Root\n 0\nNodes\n 1\n 2\nArcs\n 0 => 1 w.p. 1.0\n 1 => 2 w.p. 1.0\n 2 => 2 w.p. 0.95"
        )

    def test_finite_T3_deterministic_equivalent(self, oracle_a):
        d = oracle_a["inventory"]["finite_T3"]
        det = _det(lambda: tm.build_inventory_finite(T=3))
        assert close(det, d["deterministic_equivalent"])
        model = tm.build_inventory_finite(T=3)
        sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW)
        bound = sddp.calculate_bound(model)
        assert close(bound, d["bound_lb_0"], rel=1e-6), (bound, d["bound_lb_0"])
        # The page's `lower_bound = 0` is not a valid bound on the terminal value function
        # (recovering inventory at cost c makes it negative), so SDDP.jl and the port both
        # converge ABOVE the deterministic equivalent. Both agree with each other.
        assert bound > det + 1.0
        assert close(d["bound_lb_neg1e6"], det, rel=1e-6)

    def test_finite_horizon(self, oracle_a, capsys):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        d = oracle_a["inventory"]["finite"]
        model, sims = tm.run_inventory_finite(
            seed=1, print_level=1, run_numerical_stability_report=False
        )
        out = capsys.readouterr().out
        assert "Confidence interval: " in out and "Lower bound: " in out
        assert model.most_recent_training_results.status == "simulation_stopping"
        bound = sddp.calculate_bound(model)
        # both stop with the stochastic default rule; the converged values agree to ~1%
        assert close(bound, d["bound"], rel=1e-2), (bound, d["bound"])
        assert len(sims) == 200 and all(len(s) == 11 for s in sims)
        levels = np.array([[tm.order_up_to_level(s) for s in sim] for sim in sims])
        med = np.median(levels, axis=0)
        # order-up-to policy in the early stages, end-of-horizon effects at the end
        assert np.allclose(med[:5], np.array(d["order_up_to_median_by_stage"][:5]), rtol=0.05)
        assert med[0] > 500 and med[-1] < med[0]
        ax = tm.plot_inventory(sims)
        assert ax.get_ylim() == (0.0, 1000.0)
        matplotlib.pyplot.close("all")

    def test_infinite_horizon_short(self):
        model, sims = tm.run_inventory_infinite(iteration_limit=40, seed=1, print_level=0)
        assert model.most_recent_training_results.status == "iteration_limit"
        assert len(sims) == 200 and all(len(s) == 50 for s in sims)
        assert all(s["node_index"] == 2 for sim in sims for s in sim[1:])
        assert sddp.calculate_bound(model) > 0.0

    @pytest.mark.slow
    def test_infinite_horizon_parity(self, oracle_a):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        d = oracle_a["inventory"]["infinite"]
        model, sims = tm.run_inventory_infinite(
            iteration_limit=d["iteration_limit"], seed=1, print_level=0
        )
        bound = sddp.calculate_bound(model)
        assert close(bound, d["bound"], rel=1e-2), (bound, d["bound"])
        assert bound <= d["bound_1000"] * (1 + 1e-3)
        assert sorted({len(s) for s in sims}) == d["simulation_lengths"] == [50]
        levels = np.array([tm.order_up_to_level(s) for sim in sims for s in sim[5:]])
        med = float(np.median(levels))
        # "We again recover an order-up-to policy. The analytic solution is to order-up-to
        # 662 units" (the SAA with 20 points does not hit it exactly).
        assert abs(med - 662) / 662 < 0.1, med
        assert abs(med - d["order_up_to_later_stages"]["median"]) / 662 < 0.05
        ax = tm.plot_inventory(sims, analytic=True)
        assert ax.get_legend() is not None
        matplotlib.pyplot.close("all")
