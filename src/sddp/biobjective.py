# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Biobjective SDDP (experimental in SDDP.jl too). Ported from ``src/biobjective.jl``."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from sddp.algorithm import calculate_bound, train
from sddp.policy_graph import PolicyGraph, Subproblem


def set_biobjective_functions(sp: Subproblem, objective_1: Any, objective_2: Any) -> None:
    """Set the scalarised stage objective ``λ f₁ + (1 − λ) f₂``. Call inside ``parameterize``."""
    lam = sp.objective_state()
    sp.set_stage_objective(lam * objective_1 + (1 - lam) * objective_2)


def initialize_biobjective_subproblem(sp: Subproblem) -> None:
    """Add the trade-off weight as an objective state. Call outside ``parameterize``."""
    sp.add_objective_state(
        lambda y, _: y, initial_value=0.0, lower_bound=0.0, upper_bound=1.0, lipschitz=1e6
    )


def set_trade_off_weight(model: PolicyGraph, weight: float) -> None:
    assert 0 <= weight <= 1
    for node in model.nodes.values():
        assert node.objective_state is not None
        node.objective_state.initial_value = (weight,)
        node.objective_state.state = (weight,)


def train_biobjective(
    model: PolicyGraph,
    solution_limit: int,
    include_timing: bool = False,
    log_file_prefix: str | None = None,
    stopping_rules: Callable[[float], list[Any]] = lambda w: [],
    **kwargs: Any,
) -> dict[float, Any]:
    """Non-inferior set estimation over trade-off weights; returns ``{weight: bound}``
    (or ``{weight: (bound, elapsed)}`` with ``include_timing``)."""
    start_time = time.time()
    solutions: dict[float, Any] = {}

    def value(bound: float) -> Any:
        return (bound, time.time() - start_time) if include_timing else bound

    def _train(w: float) -> None:
        train(
            model,
            add_to_existing_cuts=True,
            run_numerical_stability_report=False,
            log_file=f"{log_file_prefix}_{w}.log" if log_file_prefix else None,
            stopping_rules=stopping_rules(w),
            **kwargs,
        )

    for weight in (0.0, 1.0):
        set_trade_off_weight(model, weight)
        _train(weight)
        solutions[weight] = value(calculate_bound(model))
    queue: list[tuple[float, float]] = [(0.0, 1.0)]
    while queue and len(solutions) < solution_limit:
        a, b = queue.pop(0)
        w = 0.5 * (a + b)
        set_trade_off_weight(model, w)
        _train(w)
        bound = calculate_bound(model)
        solutions[w] = value(bound)
        sa = solutions[a][0] if include_timing else solutions[a]
        sb = solutions[b][0] if include_timing else solutions[b]
        best_bound = 0.5 * (sa + sb)
        if not math.isclose(best_bound, bound, rel_tol=1e-4):
            queue.append((a, w))
            queue.append((w, b))
    return solutions
