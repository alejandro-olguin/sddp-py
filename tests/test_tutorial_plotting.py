"""Plotting tutorial: Markov model, spaghetti/publication plots, value function, graph plot."""

from __future__ import annotations

import math
import os

import pytest
from examples import tutorial_plotting as tp

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


@pytest.fixture(scope="module")
def trained():
    return tp.train_and_simulate(seed=1)


def test_train_and_simulate_structure(trained):
    model, sims = trained
    jl = load_oracle("tutorials_c")["plotting"]["random_20"]
    assert len(sims) == jl["replications"] == 100
    assert all(len(s) == jl["stages"] == 3 for s in sims)
    assert len(model.most_recent_training_results.log) == 20
    assert sims[0][0]["node_index"] == (1, 1)
    assert all(s[t]["node_index"][0] == t + 1 for s in sims for t in range(3))
    for s in sims:
        for stage in s:
            assert isinstance(stage["noise_term"], tp.Realization)
            assert 0.0 <= stage["volume"].out <= 200.0
            assert stage["thermal_generation"] + stage["hydro_generation"] == pytest.approx(150.0)
    assert sddp.calculate_bound(model) > 0


def test_spaghetti_plot(trained, tmp_path):
    _, sims = trained
    f = str(tmp_path / "spaghetti_plot.html")
    plt = tp.spaghetti(sims, f)
    assert repr(plt) == "A spaghetti plot with 100 scenarios and 3 stages."
    assert len(plt.data) == 2
    assert plt.data[0]["title"] == "Reservoir volume"
    assert plt.data[1]["title"] == "Fuel cost" and plt.data[1]["ymin"] == 0
    assert plt.data[1]["ymax"] == 250
    assert len(plt.data[0]["data"]) == 100 and len(plt.data[0]["data"][0]) == 3
    # fuel cost is multiplier * fuel_cost[t] whenever thermal generation is positive (the
    # page's ``> 0`` test also lets through solver noise like 1e-13, whose ratio is garbage)
    for sim, series in zip(sims, plt.data[1]["data"]):
        for t, (stage, y) in enumerate(zip(sim, series)):
            if stage["thermal_generation"] > 1e-6:
                expected = stage["noise_term"].fuel_multiplier * [50.0, 100.0, 150.0][t]
                assert y == pytest.approx(expected)
            elif stage["thermal_generation"] <= 0:
                assert y == 0.0
    assert os.path.getsize(f) > 0
    with open(f, encoding="utf-8") as io:
        assert "Reservoir volume" in io.read()


def test_publication_plots(trained, tmp_path):
    pytest.importorskip("matplotlib")
    _, sims = trained
    f = str(tmp_path / "publication.png")
    fig = tp.publication_plots(sims, f)
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Outgoing volume", "Thermal generation"]
    assert all(ax.get_ylim() == (0.0, 200.0) for ax in fig.axes)
    assert os.path.getsize(f) > 0


def test_value_function_and_graph_plot(trained, tmp_path):
    model, _ = trained
    height, subgradient = tp.value_function(model, str(tmp_path / "value_function.html"))
    assert height >= 0.0 and set(subgradient) == {"volume"}
    assert subgradient["volume"] <= 0.0  # more water never costs more
    assert os.path.getsize(tmp_path / "value_function.html") > 0
    f = sddp.plot_graph(model, str(tmp_path / "model_plotting.html"))
    assert os.path.getsize(f) > 0


def test_value_function_matches_julia_with_historical_training():
    # Julia's 20-iteration training is random, so the page's numbers cannot be reproduced;
    # replaying the same 20 Historical scenarios in both languages makes them deterministic.
    jl = load_oracle("tutorials_c")["plotting"]["historical_20"]
    scenarios = [[((t, m), tp.Omega[k - 1]) for t, m, k in s] for s in jl["scenarios"]]
    model = tp.build_model()
    sddp.train(model, iteration_limit=20, sampling_scheme=sddp.Historical(scenarios), **KW)
    assert close(sddp.calculate_bound(model), jl["bound"])
    log = model.most_recent_training_results.log
    assert len(log) == len(jl["log"]) == 20
    theta = model[(1, 1)].bellman_function.global_theta
    py_states = [s.state["volume"] for s in theta.sampled_states]
    # In iteration 1 the future cost is still 0, so the first-stage LP is degenerate between
    # storing and spilling water; which vertex HiGHS returns decides the first sampled state
    # and hence the one cut that shapes V below volume 50. Everything after that (bounds per
    # iteration, cuts at the visited states, V on [50, 200]) must agree exactly.
    jl_states = jl["sampled_states"]
    same_first_state = close(py_states[0], jl_states[0])
    same = [close(a, b) for a, b in zip(py_states, jl_states)]
    assert len(same) == 20 and all(same[2:]), (py_states, jl_states)  # tie resolved by iter 3
    for i, (l, j) in enumerate(zip(log, jl["log"])):
        if same[i]:
            assert close(l.bound, j["bound"]), (i, l, j)
    assert len(theta.cuts) == len(jl["cuts"]) == 20
    for i, (c, j) in enumerate(zip(theta.cuts, jl["cuts"])):
        if same[i]:
            assert close(c.intercept, j["intercept"]) and close(
                c.coefficients["volume"], j["coefficient"]
            )
    V = sddp.ValueFunction(model, node=(1, 1))
    for p in jl["values"]:
        if p["volume"] < 50 and not same_first_state:
            continue
        h, g = sddp.evaluate_value_function(V, {"volume": p["volume"]})
        assert close(h, p["height"]), p
        assert close(g["volume"], p["subgradient"]), p
    height, subgradient = tp.value_function(model)
    if same_first_state:
        assert close(height, jl["value_at_1"]["height"])
        assert close(subgradient["volume"], jl["value_at_1"]["subgradient"])
    else:
        # Both are valid outer approximations built from the same cuts except the first.
        assert height > 0 and subgradient["volume"] <= 0
