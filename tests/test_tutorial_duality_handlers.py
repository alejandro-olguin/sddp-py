"""Duality handlers tutorial: every (model, handler) pair vs the Julia oracle ``tutorials_c.json``.

The four models are deterministic, so Julia's printed lower/upper bounds must be reproduced
exactly (1e-6) by every duality handler.
"""

from __future__ import annotations

import math

import pytest
from examples import tutorial_duality_handlers as dh

import sddp
from tests.conftest import load_oracle

MODELS = ["model_1", "model_2", "model_3", "model_4"]
HANDLERS = ["continuous", "strengthened", "lagrangian", "bandit", "fixed_discrete"]


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


@pytest.fixture(scope="module")
def jl():
    return load_oracle("tutorials_c")["duality_handlers"]


def test_page_combinations_are_all_covered():
    assert {(m, h) for m, h in dh.PAGE_COMBINATIONS} <= {(m, h) for m in MODELS for h in HANDLERS}


def test_presolve_off_option():
    assert dh.Optimizer.options["presolve"] == "off"
    assert dh.Optimizer.name == "HiGHS"


@pytest.mark.parametrize("handler", HANDLERS)
@pytest.mark.parametrize("model", MODELS)
def test_bounds_match_julia(model, handler, jl):
    # HiGHS's MIP solver (both languages, mip_feasibility_tolerance = 1e-6) may return points
    # that violate a cut/integrality by up to 1e-6, so e.g. Python reports 1.499999 where
    # Julia happens to print 1.5 for model_1/continuous; the cuts themselves are identical
    # (see test_model_1_cut_is_exact). Hence atol = 1e-5.
    lb, ub = dh.train_and_evaluate_bounds(dh.MODELS[model], dh.handler_by_name(handler), seed=1)
    j = jl[f"{model}/{handler}"]
    assert close(lb, j["lower_bound"], atol=1e-5), (model, handler, lb, j)
    assert close(ub, j["upper_bound"], atol=1e-5), (model, handler, ub, j)
    assert lb <= ub + 1e-5


def test_model_1_cut_is_exact(tmp_path):
    model = dh.model_1()
    sddp.train(
        model,
        print_level=0,
        duality_handler=sddp.ContinuousConicDuality(),
        seed=1,
        run_numerical_stability_report=False,
    )
    import json

    f = tmp_path / "cuts.json"
    sddp.write_cuts_to_file(model, str(f))
    with open(f) as io:
        cuts = json.load(io)
    node1 = next(c for c in cuts if c["node"] == "1")
    assert node1["single_cuts"]
    for cut in node1["single_cuts"]:
        assert cut["intercept"] == pytest.approx(0.5) and cut["coefficients"] == {"x": 1.0}
    assert model.most_recent_training_results.log[-1].bound == pytest.approx(1.5)


def test_page_narrative():
    # The relationships the page describes (independent of the oracle).
    lb_c, ub_c = dh.train_and_evaluate_bounds(dh.model_1, sddp.ContinuousConicDuality())
    lb_s, ub_s = dh.train_and_evaluate_bounds(dh.model_1, sddp.StrengthenedConicDuality())
    assert lb_c < lb_s and close(lb_s, 2.0) and close(ub_c, 2.0) and close(ub_s, 2.0)
    lb_l, _ = dh.train_and_evaluate_bounds(dh.model_2, sddp.LagrangianDuality())
    lb_f, _ = dh.train_and_evaluate_bounds(dh.model_2, sddp.FixedDiscreteDuality())
    lb_s2, _ = dh.train_and_evaluate_bounds(dh.model_2, sddp.StrengthenedConicDuality())
    assert close(lb_l, 1.1) and close(lb_f, 1.1) and lb_s2 < lb_f
    _, ub4c = dh.train_and_evaluate_bounds(dh.model_4, sddp.ContinuousConicDuality())
    _, ub4l = dh.train_and_evaluate_bounds(dh.model_4, sddp.LagrangianDuality())
    assert close(ub4c, 0.45) and ub4l < ub4c
