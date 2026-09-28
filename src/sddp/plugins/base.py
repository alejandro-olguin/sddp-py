# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Abstract plugin interfaces. Ported from ``src/plugins/headers.jl``.

SDDP.jl uses multiple dispatch on abstract types; here each interface is an
abstract base class with the corresponding method.
"""

from __future__ import annotations

import abc
import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sddp.algorithm import Log, Options
    from sddp.policy_graph import Node, Noise, PolicyGraph


class RiskMeasure(abc.ABC):
    """Interface: :meth:`adjust_probability`."""

    @abc.abstractmethod
    def adjust_probability(
        self,
        risk_adjusted_probability: list[float],
        original_probability: Sequence[float],
        noise_support: Sequence[Any],
        objective_realizations: Sequence[float],
        is_minimization: bool,
    ) -> float:
        """Fill ``risk_adjusted_probability`` in place and return the offset ``α``."""

    def __add__(self, other: Any) -> Any:
        from sddp.plugins.risk_measures import ConvexCombination

        if isinstance(other, ConvexCombination):
            return ConvexCombination((1.0, self), *other.measures)
        if isinstance(other, RiskMeasure):
            return ConvexCombination((1.0, self), (1.0, other))
        return NotImplemented

    def __rmul__(self, weight: float) -> Any:
        from sddp.plugins.risk_measures import ConvexCombination

        return ConvexCombination((float(weight), self))


class SamplingScheme(abc.ABC):
    """Interface: :meth:`sample_scenario`."""

    @abc.abstractmethod
    def sample_scenario(
        self, graph: PolicyGraph, rng: random.Random
    ) -> tuple[list[tuple[Any, Any]], bool]:
        """Return ``(scenario_path, terminated_due_to_cycle)``.

        ``scenario_path`` is a list of ``(node_index, noise_term)`` tuples.
        """


class StoppingRule(abc.ABC):
    """Interface: :meth:`stopping_rule_status` and :meth:`convergence_test`."""

    @abc.abstractmethod
    def stopping_rule_status(self) -> str: ...

    @abc.abstractmethod
    def convergence_test(self, graph: PolicyGraph, log: list[Log]) -> bool: ...


def convergence_test(graph: PolicyGraph, log: list[Log], rules: Sequence[StoppingRule]) -> tuple[bool, str]:
    for rule in rules:
        if rule.convergence_test(graph, log):
            return True, rule.stopping_rule_status()
    return False, "not_solved"


class BackwardSamplingScheme(abc.ABC):
    """Interface: :meth:`sample_backward_noise_terms`."""

    @abc.abstractmethod
    def sample_backward_noise_terms(self, node: Node, rng: random.Random) -> list[Noise]: ...

    def sample_backward_noise_terms_with_state(
        self, node: Node, state: dict[str, float], rng: random.Random
    ) -> list[Noise]:
        return self.sample_backward_noise_terms(node, rng)


class DualityHandler(abc.ABC):
    """Interface: :meth:`get_dual_solution`, :meth:`prepare_backward_pass`, :meth:`duality_log_key`."""

    @abc.abstractmethod
    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        """Return ``(objective, {state_name: dual})`` for the fishing constraints."""

    def prepare_backward_pass(self, node: Node, options: Options) -> Callable[[], None]:
        return lambda: None

    def duality_log_key(self) -> str:
        return " "


class ForwardPass(abc.ABC):
    """Interface: :meth:`forward_pass` returning a :class:`ForwardPassResult`."""

    @abc.abstractmethod
    def forward_pass(self, model: PolicyGraph, options: Options) -> Any: ...


class ParallelScheme(abc.ABC):
    """Interface: :meth:`master_loop` and :meth:`simulate`."""

    @abc.abstractmethod
    def master_loop(self, model: PolicyGraph, options: Options) -> str: ...

    @abc.abstractmethod
    def simulate(
        self, model: PolicyGraph, number_replications: int, variables: Sequence[str], **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...

    def interrupt(self) -> None:
        return None
