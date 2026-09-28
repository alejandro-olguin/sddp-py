# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Stopping rules. Ported from ``src/plugins/stopping_rules.jl``."""

from __future__ import annotations

import math
import statistics
import warnings
from typing import TYPE_CHECKING, Any

from sddp.plugins.base import SamplingScheme, StoppingRule
from sddp.policy_graph import Log, PolicyGraph

if TYPE_CHECKING:
    pass


def _isapprox(x: float, y: float, atol: float = 0.0, rtol: float = 1.4901161193847656e-08) -> bool:
    return math.isclose(x, y, rel_tol=rtol, abs_tol=atol)


class IterationLimit(StoppingRule):
    """Terminate after ``limit`` iterations."""

    def __init__(self, limit: int):
        self.limit = limit

    def stopping_rule_status(self) -> str:
        return "iteration_limit"

    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool:
        return log[-1].iteration >= self.limit


class TimeLimit(StoppingRule):
    """Terminate after ``limit`` seconds."""

    def __init__(self, limit: float):
        self.limit = float(limit)

    def stopping_rule_status(self) -> str:
        return "time_limit"

    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool:
        return log[-1].time >= self.limit


class Statistical(StoppingRule):
    """Terminate when the bound falls in the confidence interval of an in-sample simulation.

    See the SDDP.jl docstring for why this rule is not recommended.
    """

    def __init__(
        self,
        num_replications: int,
        iteration_period: int = 1,
        z_score: float = 1.96,
        verbose: bool = True,
        disable_warning: bool = False,
    ):
        if not disable_warning:
            warnings.warn(
                "Are you really sure you want to use this stopping rule? Read why we don't "
                "recommend it in the SDDP.jl docstring for `SDDP.Statistical`. Disable this "
                "warning with `Statistical(..., disable_warning=True)`",
                stacklevel=2,
            )
        self.num_replications = num_replications
        self.iteration_period = iteration_period
        self.z_score = z_score
        self.verbose = verbose

    def stopping_rule_status(self) -> str:
        return "statistical"

    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool:
        from sddp.algorithm import simulate

        if len(log) % self.iteration_period != 0:
            return False
        results = simulate(graph, self.num_replications)
        objectives = [sum(s["stage_objective"] for s in sim) for sim in results]
        sample_mean = statistics.mean(objectives)
        sample_ci = self.z_score * statistics.stdev(objectives) / math.sqrt(self.num_replications)
        if self.verbose:
            print(
                f"Simulated policy value: [{sample_mean - sample_ci:1.6e}, "
                f"{sample_mean + sample_ci:1.6e}]"
            )
        current_bound = log[-1].bound
        if graph.is_minimization:
            return sample_mean - sample_ci <= current_bound
        return current_bound <= sample_mean + sample_ci


class BoundStalling(StoppingRule):
    """Terminate once the bound fails to improve for ``num_previous_iterations`` iterations."""

    def __init__(self, num_previous_iterations: int, atol: float = 0.0, rtol: float = 0.0):
        self.num_previous_iterations = num_previous_iterations
        self.atol = atol
        self.rtol = rtol

    def _isapprox(self, x: float, y: float) -> bool:
        return math.isclose(x, y, rel_tol=self.rtol, abs_tol=self.atol)

    def stopping_rule_status(self) -> str:
        return "bound_stalling"

    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool:
        if len(log) < self.num_previous_iterations + 1:
            return False
        for i in range(1, self.num_previous_iterations + 1):
            if not self._isapprox(log[-1 - i].bound, log[-i].bound):
                return False
        if self._isapprox(log[0].bound, log[-1].bound):
            return all(self._isapprox(l.bound, l.simulation_value) for l in log)
        return True


class StoppingChain(StoppingRule):
    """Terminate once all of the rules are satisfied (short-circuits)."""

    def __init__(self, *rules: StoppingRule):
        self.rules = list(rules)

    def stopping_rule_status(self) -> str:
        return " ∧ ".join(r.stopping_rule_status() for r in self.rules)

    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool:
        return all(rule.convergence_test(graph, log) for rule in self.rules)


def _compute_distance(x: Any, y: Any) -> float:
    if isinstance(x, (int, float)):
        if math.isclose(x, y, rel_tol=1.4901161193847656e-08):
            return 0.0
        return abs(x - y) / max(1.0, abs(x), abs(y))
    return math.sqrt(sum(_compute_distance(a, b) ** 2 for a, b in zip(x, y)))


def _period(period: int, iterations: int) -> int:
    if period != -1:
        return period
    elif iterations <= 100:
        return 20
    elif iterations <= 1_000:
        return 100
    return 500


class SimulationStoppingRule(StoppingRule):
    """The default stopping rule: bound stalls and successive simulations agree."""

    def __init__(
        self,
        sampling_scheme: SamplingScheme | None = None,
        replications: int = -1,
        period: int = -1,
        distance_tol: float = 1e-2,
        bound_tol: float = 1e-4,
    ):
        from sddp.plugins.sampling_schemes import InSampleMonteCarlo, PSRSamplingScheme

        self.cached_sampling_scheme = PSRSamplingScheme(
            replications, sampling_scheme=sampling_scheme or InSampleMonteCarlo()
        )
        self.replications = replications
        self.period = period
        self.data: list[Any] = []
        self.last_iteration = 0
        self.distance_tol = distance_tol
        self.bound_tol = bound_tol

    def simulator(self, model: PolicyGraph, N: int) -> list[Any]:
        from sddp.algorithm import simulate

        self.cached_sampling_scheme.N = max(N, self.cached_sampling_scheme.N)
        scenarios = simulate(model, N, sampling_scheme=self.cached_sampling_scheme)
        keys = ["stage_objective", "bellman_term"]
        return [[[s[k] for s in scenario] for k in keys] for scenario in scenarios]

    def stopping_rule_status(self) -> str:
        return "simulation_stopping"

    def convergence_test(self, model: PolicyGraph, log: list[Log]) -> bool:
        from sddp.print import _unique_paths

        if self.replications == -1:
            self.replications = int(min(100, _unique_paths(model)))
        if not self.data:
            self.data = self.simulator(model, self.replications)
            self.last_iteration = 0
            return False
        if len(log) <= 5:
            return False
        if not _isapprox(log[-1].bound, log[-6].bound, atol=self.bound_tol, rtol=self.bound_tol):
            return False
        if len(log) - self.last_iteration < _period(self.period, len(log)):
            return False
        new_data = self.simulator(model, self.replications)
        distance = _compute_distance(new_data, self.data)
        self.data = new_data
        self.last_iteration = len(log)
        return distance < self.distance_tol


class FirstStageStoppingRule(StoppingRule):
    """Terminate when the first-stage outgoing state stops changing."""

    def __init__(self, atol: float = 1e-3, iterations: int = 50):
        self.data: list[dict[str, float]] = []
        self.atol = atol
        self.iterations = iterations

    def stopping_rule_status(self) -> str:
        return "first_stage_stopping"

    def convergence_test(self, model: PolicyGraph, log: list[Log]) -> bool:
        from sddp.algorithm import get_outgoing_state, parameterize, set_incoming_state

        if len(model.root_children) != 1:
            raise ValueError(
                "FirstStageStoppingRule cannot be applied because first-stage is not deterministic"
            )
        node = model[model.root_children[0].term]
        if len(node.noise_terms) > 1:
            raise ValueError(
                "FirstStageStoppingRule cannot be applied because first-stage is not deterministic"
            )
        set_incoming_state(node, model.initial_root_state)
        parameterize(node, node.noise_terms[0].term)
        node.model.optimize()
        state = get_outgoing_state(node)
        self.data.append(state)
        if len(self.data) < self.iterations:
            return False
        for i in range(1, self.iterations):
            for k, v in state.items():
                if not math.isclose(self.data[-1 - i][k], v, abs_tol=self.atol, rel_tol=0.0):
                    return False
        return True
