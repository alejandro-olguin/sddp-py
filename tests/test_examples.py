"""Documented SDDP.jl examples (docs/src/examples) vs the Julia oracle ``examples.json``."""

from __future__ import annotations

import math
import statistics

import pytest
from examples.asset_management_simple import build_asset_management_simple
from examples.doc_examples import (
    booking_management_model,
    build_all_blacks,
    build_generation_expansion,
    build_mccardle_farm,
    build_multistock,
    build_prob52,
    build_sldp_two,
    create_air_conditioning_model,
    hydro_valley_model,
)

import sddp
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


def _check(name, build, **train_kwargs):
    d = load_oracle("examples")[name]
    if "deterministic_equivalent" in d:
        det = sddp.deterministic_equivalent(build())
        det.optimize()
        assert close(det.objective_value(), d["deterministic_equivalent"]), (name, "det")
    model = build()
    sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW, **train_kwargs)
    assert close(sddp.calculate_bound(model), d["bound"]), (
        name,
        sddp.calculate_bound(model),
        d["bound"],
    )
    return model, d


def test_asset_management_simple_documented_value():
    model = build_asset_management_simple()
    sddp.train(model, seed=1, **KW)  # default stopping rule, as on the website
    assert close(sddp.calculate_bound(model), 1.514, atol=1e-4)  # value quoted on sddp.dev


def test_asset_management_simple_parity():
    model, d = _check("asset_management_simple", build_asset_management_simple)
    sims = sddp.simulate(model, d["simulation"]["replications"], seed=42)
    objs = [sum(s["stage_objective"] for s in sim) for sim in sims]
    mu, sd, n = statistics.mean(objs), statistics.stdev(objs), len(objs)
    half = 2.576 * math.sqrt(
        sd**2 / n + d["simulation"]["std"] ** 2 / d["simulation"]["replications"]
    )
    assert abs(mu - d["simulation"]["mean"]) <= half


def test_mccardle_farm():
    _check("agriculture_mccardle_farm", build_mccardle_farm)


def test_generation_expansion():
    _check("generation_expansion", build_generation_expansion)


@pytest.mark.parametrize(
    "name,kwargs,train",
    [
        ("hydro_valley_deterministic", {"hasmarkovprice": False, "hasstagewiseinflows": False}, {}),
        ("hydro_valley_stagewise", {"hasmarkovprice": False}, {}),
        ("hydro_valley_markov", {"hasstagewiseinflows": False}, {}),
        ("hydro_valley_markov_stagewise", {}, {}),
        ("hydro_valley_riskaverse", {}, {"risk_measure": sddp.EAVaR(lambda_=0.5, beta=0.66)}),
        (
            "hydro_valley_worst_case_min",
            {"sense": "Min"},
            {"risk_measure": sddp.EAVaR(lambda_=0.5, beta=0.0)},
        ),
        (
            "hydro_valley_dro_1_6",
            {"hasmarkovprice": False},
            {"risk_measure": sddp.ModifiedChiSquared(1 / 6)},
        ),
        (
            "hydro_valley_dro_worst",
            {"hasmarkovprice": False},
            {"risk_measure": sddp.ModifiedChiSquared(math.sqrt(2 / 3) - 1e-6)},
        ),
    ],
)
def test_hydro_valley(name, kwargs, train):
    _check(name, lambda: hydro_valley_model(**kwargs), **train)


@pytest.mark.parametrize("name,args", [("booking_1_2_5", (1, 2, 5)), ("booking_2_2_3", (2, 2, 3))])
def test_booking_management(name, args):
    # Binary states trained with ContinuousConicDuality: the cuts come from LP relaxations,
    # so the converged "bound" depends on the sampled trajectories (Julia: 9.003 / 6.824).
    # SDDP.jl's own example therefore only checks validity against the MIP optimum, and so
    # does this test; the deterministic equivalent itself is compared exactly.
    d = load_oracle("examples")[name]
    det = sddp.deterministic_equivalent(booking_management_model(*args))
    det.optimize()
    assert close(det.objective_value(), d["deterministic_equivalent"])
    model = booking_management_model(*args)
    sddp.train(model, iteration_limit=d["iteration_limit"], seed=1, **KW)
    assert sddp.calculate_bound(model) >= d["deterministic_equivalent"] - 1e-4


@pytest.mark.parametrize("name,stages", [("prob52_2stages", 2), ("prob52_3stages", 3)])
def test_prob52(name, stages):
    _check(name, lambda: build_prob52(stages))


def test_multistock():
    model, d = _check("multistock", build_multistock, cut_type=sddp.SINGLE_CUT)
    sims = sddp.simulate(model, d["simulation"]["replications"], seed=42)
    objs = [sum(s["stage_objective"] for s in sim) for sim in sims]
    mu, sd, n = statistics.mean(objs), statistics.stdev(objs), len(objs)
    half = 2.576 * math.sqrt(
        sd**2 / n + d["simulation"]["std"] ** 2 / d["simulation"]["replications"]
    )
    assert abs(mu - d["simulation"]["mean"]) <= half


def test_all_blacks():
    _check("all_blacks_lagrangian", build_all_blacks, duality_handler=sddp.LagrangianDuality())
    _check("all_blacks_conic", build_all_blacks)


def test_air_conditioning_forward():
    d = load_oracle("examples")["air_conditioning_forward"]
    convex, non_convex = create_air_conditioning_model(True), create_air_conditioning_model(False)
    sddp.train(
        convex,
        forward_pass=sddp.AlternativeForwardPass(non_convex),
        post_iteration_callback=sddp.AlternativePostIterationCallback(non_convex),
        iteration_limit=d["iteration_limit"],
        seed=1,
        **KW,
    )
    assert close(sddp.calculate_bound(convex), d["bound_convex"])
    assert close(sddp.calculate_bound(non_convex), d["bound_non_convex"])


@pytest.mark.parametrize("N", [2, 3, 6])
def test_sldp_example_two(N):
    _check(f"sldp_two_N{N}", lambda: build_sldp_two(N))
