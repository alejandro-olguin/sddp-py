# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Parallel schemes. Only ``Serial`` is ported (``src/plugins/parallel_schemes.jl``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sddp.plugins.base import ParallelScheme
from sddp.policy_graph import PolicyGraph

if TYPE_CHECKING:
    from sddp.algorithm import Options


class Serial(ParallelScheme):
    """Run SDDP in serial mode."""

    def __repr__(self) -> str:
        return "serial mode"

    def master_loop(self, model: PolicyGraph, options: Options) -> str:
        from sddp.algorithm import iteration, log_iteration

        while True:
            result = iteration(model, options)
            options.post_iteration_callback(result)
            log_iteration(options)
            if result.has_converged:
                return result.status

    def simulate(
        self, model: PolicyGraph, number_replications: int, variables: Sequence[str], **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        from sddp.algorithm import _simulate_one

        return [_simulate_one(model, variables, **kwargs) for _ in range(number_replications)]
