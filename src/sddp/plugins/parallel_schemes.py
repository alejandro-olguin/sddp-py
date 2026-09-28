# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Parallel schemes: ``Serial`` and ``Threaded`` (``src/plugins/parallel_schemes.jl``).

``Threaded`` runs whole iterations concurrently in Python threads with one lock per node,
exactly as SDDP.jl does. It pays off because the solver releases the GIL during a solve;
Python-side work remains serialised, so speed-ups are bounded by the solver share of the
run time. Results are valid but not reproducible run to run (cut order depends on timing).
"""

from __future__ import annotations

import os
import threading
import warnings
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


class Threaded(ParallelScheme):
    """Run iterations (and simulations) concurrently in ``num_threads`` threads."""

    def __init__(self, num_threads: int | None = None):
        self.num_threads = num_threads if num_threads is not None else (os.cpu_count() or 1)

    def __repr__(self) -> str:
        return f"Threaded({self.num_threads})"

    def _threads(self, model: PolicyGraph) -> int:
        n = self.num_threads
        if len(model.nodes) < n:
            warnings.warn(
                f"There are fewer nodes in the graph ({len(model.nodes)}) than there are "
                f"threads requested ({n}). Limiting the number of threads to {len(model.nodes)}.",
                stacklevel=3,
            )
            n = len(model.nodes)
        return max(1, n)

    def master_loop(self, model: PolicyGraph, options: Options) -> str:
        from sddp.algorithm import iteration, log_iteration

        state: dict[str, Any] = {"keep_iterating": True, "status": "not_solved", "error": None}

        def worker() -> None:
            try:
                while state["keep_iterating"]:
                    result = iteration(model, options)
                    with options.lock:
                        options.post_iteration_callback(result)
                        log_iteration(options)
                        if result.has_converged:
                            state["keep_iterating"] = False
                            state["status"] = result.status
            except BaseException as e:  # propagate the first error to the caller
                with options.lock:
                    state["keep_iterating"] = False
                    if state["error"] is None:
                        state["error"] = e

        threads = [
            threading.Thread(target=worker, daemon=True) for _ in range(self._threads(model))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if state["error"] is not None:
            raise state["error"]
        return state["status"]

    def simulate(
        self, model: PolicyGraph, number_replications: int, variables: Sequence[str], **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        from sddp.algorithm import _simulate_one

        n = min(self._threads(model), max(1, number_replications))
        results: list[Any] = [None] * number_replications
        errors: list[BaseException] = []

        def worker(indices: range) -> None:
            try:
                for j in indices:
                    results[j] = _simulate_one(model, variables, **kwargs)
            except BaseException as e:
                errors.append(e)

        w = number_replications // n
        chunks = [range(i * w, (i + 1) * w if i < n - 1 else number_replications) for i in range(n)]
        threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results
