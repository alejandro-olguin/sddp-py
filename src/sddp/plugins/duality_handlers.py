# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Duality handlers. Ported from ``src/plugins/duality_handlers.jl``.

Tier 1 provides :class:`ContinuousConicDuality`. Lagrangian and strengthened
variants are Tier 3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sddp.plugins.base import DualityHandler
from sddp.policy_graph import Node

if TYPE_CHECKING:
    from sddp.algorithm import Options


def get_dual_solution_none(node: Node) -> tuple[float, dict[str, float]]:
    """``get_dual_solution(node, nothing)``: objective only, no duals."""
    return node.model.objective_value(), {}


class ContinuousConicDuality(DualityHandler):
    """Duals from the LP relaxation (integrality relaxed on the backward pass).

    The dual of the fishing constraint ``x_in == x̄`` is the reduced cost of the fixed
    variable. Our solver layer returns it as ``d objective / d x̄`` in the model's own
    sense, so no sign flip is needed (SDDP.jl needs one because of JuMP's convention).
    """

    def __init__(self, optimizer: object | None = None):
        self.optimizer = optimizer

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        from sddp.algorithm import attempt_numerical_recovery

        m = node.model
        if not m.has_dual_solution():
            attempt_numerical_recovery(node.policy_graph, node, require_dual=True)
        lam = {name: m.reduced_cost(state.in_) for name, state in node.states.items()}
        return m.objective_value(), lam

    def prepare_backward_pass(self, node: Node, options: Options) -> Callable[[], None]:
        return _relax_integrality(node)

    def duality_log_key(self) -> str:
        return " "


def _relax_integrality(node: Node) -> Callable[[], None]:
    if not node.has_integrality:
        return lambda: None
    return node.model.relax_integrality()
