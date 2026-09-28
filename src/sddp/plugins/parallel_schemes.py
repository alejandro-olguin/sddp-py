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


def slave_update(model: PolicyGraph, result: Any) -> None:
    """Add the cuts found in ``result`` (an ``IterationResult``) to ``model``."""
    from sddp.plugins.bellman_functions import _add_cut

    for node_index, cuts in result.cuts.items():
        for cut in cuts:
            if cut is None:
                raise ValueError(
                    "This model uses features that are not supported in async mode. Use "
                    "`parallel_scheme = Serial()` instead."
                )
            _add_cut(
                model[node_index].bellman_function.global_theta,
                cut["theta"],
                dict(cut["pi"]),
                dict(cut["x"]),
                cut["obj_y"],
                cut["belief_y"],
                cut_selection=True,
            )


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


# ---------------------------------------------------------------------------
# Multiprocess (the analogue of SDDP.jl's Asynchronous / Distributed scheme)
# ---------------------------------------------------------------------------
def _mp_worker(
    model_factory: Any,
    train_kwargs: dict[str, Any],
    updates: Any,
    results: Any,
    worker_id: int,
    seed: int | None,
) -> None:  # pragma: no cover - runs in a child process
    import queue
    import random

    from sddp.algorithm import Options, iteration

    model = model_factory()
    options = Options.create(
        model,
        model.initial_root_state,
        rng=random.Random(None if seed is None else seed + worker_id),
        **train_kwargs,
    )
    for node in model.nodes.values():
        node.bellman_function.cut_type = train_kwargs.get(
            "cut_type", node.bellman_function.cut_type
        )
    pending: list[Any] = []
    try:
        while True:
            result = iteration(model, options)
            payload = {
                "pid": worker_id,
                "bound": result.bound,
                "cumulative_value": result.cumulative_value,
                "cuts": result.cuts,
                "numerical_issue": result.numerical_issue,
            }
            try:
                results.put(payload)
            except (EOFError, BrokenPipeError, OSError):
                return
            while True:
                try:
                    msg = updates.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    return
                pending.append(msg)
            for msg in pending:
                slave_update(model, msg)
            pending.clear()
    except (EOFError, BrokenPipeError, OSError):
        return


class Multiprocess(ParallelScheme):
    """Asynchronous SDDP over worker processes (cf. SDDP.jl's ``Asynchronous``).

    Each worker rebuilds the model with ``model_factory`` (a module-level, picklable
    function), runs iterations on its own copy, and ships cuts to the master, which merges
    them, recomputes the bound and tests convergence, while forwarding cuts to the other
    workers. The master also iterates while no worker result is pending (``use_master``).
    Only picklable training options can be used (no lambdas in risk measures etc.).
    """

    def __init__(self, model_factory: Any, num_workers: int | None = None, use_master: bool = True):
        self.model_factory = model_factory
        self.num_workers = (
            num_workers if num_workers is not None else max(1, (os.cpu_count() or 2) - 1)
        )
        self.use_master = use_master
        self.train_kwargs: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Multiprocess({self.num_workers} workers)"

    def master_loop(self, model: PolicyGraph, options: Options) -> str:
        import multiprocessing as mp
        import queue
        import time

        from sddp.algorithm import Log, calculate_bound, convergence_test, iteration, log_iteration

        ctx = mp.get_context("spawn")
        results: Any = ctx.Queue()
        updates = [ctx.Queue() for _ in range(self.num_workers)]
        seed = options.rng.randrange(2**31)
        procs = [
            ctx.Process(
                target=_mp_worker,
                args=(self.model_factory, self.train_kwargs, updates[i], results, i + 1, seed),
                daemon=True,
            )
            for i in range(self.num_workers)
        ]
        for p in procs:
            p.start()

        def shutdown() -> None:
            for q in updates:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

        try:
            while True:
                while self.use_master and results.empty():
                    result = iteration(model, options)
                    options.post_iteration_callback(result)
                    payload = {
                        "pid": 0,
                        "bound": result.bound,
                        "cumulative_value": result.cumulative_value,
                        "cuts": result.cuts,
                        "numerical_issue": result.numerical_issue,
                    }
                    for q in updates:
                        q.put(payload)
                    log_iteration(options)
                    if result.has_converged:
                        return result.status
                try:
                    msg = results.get(timeout=1.0)
                except queue.Empty:
                    continue
                for i, q in enumerate(updates, start=1):
                    if i != msg["pid"]:
                        q.put(msg)
                slave_update(model, msg)
                bound = calculate_bound(model, risk_measure=options.root_node_risk_measure)
                options.log.append(
                    Log(
                        len(options.log) + 1,
                        bound,
                        msg["cumulative_value"],
                        time.time() - options.start_time,
                        msg["pid"],
                        model.ext.get("total_solves", 0),
                        options.duality_handler.duality_log_key(),
                        msg["numerical_issue"],
                    )
                )
                log_iteration(options)
                has_converged, status = convergence_test(model, options.log, options.stopping_rules)
                if has_converged:
                    return status
        finally:
            shutdown()

    def simulate(
        self, model: PolicyGraph, number_replications: int, variables: Sequence[str], **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        # Simulation replications are independent; a thread pool gives the same speed-up
        # without shipping models across processes.
        return Threaded(self.num_workers).simulate(model, number_replications, variables, **kwargs)
