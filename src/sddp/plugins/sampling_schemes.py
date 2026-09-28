# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Forward-pass sampling schemes. Ported from ``src/plugins/sampling_schemes.jl``."""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Sequence
from typing import Any

from sddp.plugins.base import SamplingScheme
from sddp.policy_graph import Node, Noise, PolicyGraph


def sample_noise(noise_terms: Sequence[Noise], rng: random.Random) -> Any:
    """Sample a noise term proportionally to probability (``sample_noise`` in SDDP.jl)."""
    if len(noise_terms) == 0:
        return None
    rnd = rng.random() * sum(n.probability for n in noise_terms)
    for noise in noise_terms:
        rnd -= noise.probability
        if rnd <= 0.0:
            return noise.term
    raise RuntimeError(f"Internal SDDP error: unable to sample noise from {noise_terms}")


class _Rollout:
    def __init__(self, rollout_limit: Callable[[int], int]):
        self.i = 0
        self.f = rollout_limit

    def __call__(self) -> int:
        self.i += 1
        return self.f(self.i)


class InSampleMonteCarlo(SamplingScheme):
    """Monte Carlo sampling using the in-sample noise and transition probabilities."""

    def __init__(
        self,
        max_depth: int = 0,
        terminate_on_cycle: bool = False,
        terminate_on_dummy_leaf: bool = True,
        rollout_limit: Callable[[int], int] = lambda i: sys.maxsize,
        initial_node: Any = None,
    ):
        if not terminate_on_cycle and not terminate_on_dummy_leaf and max_depth == 0:
            raise ValueError(
                "terminate_on_cycle and terminate_on_dummy_leaf cannot both be false when max_depth=0."
            )
        self.max_depth = max_depth
        self.terminate_on_cycle = terminate_on_cycle
        self.terminate_on_dummy_leaf = terminate_on_dummy_leaf
        self.rollout_limit = _Rollout(rollout_limit)
        self.initial_node = initial_node

    def get_noise_terms(self, node: Node, node_index: Any) -> Sequence[Noise]:
        return node.noise_terms

    def get_children(self, node: Node, node_index: Any) -> Sequence[Noise]:
        return node.children

    def get_root_children(self, graph: PolicyGraph) -> Sequence[Noise]:
        return graph.root_children

    def sample_scenario(self, graph: PolicyGraph, rng: random.Random) -> tuple[list[tuple[Any, Any]], bool]:
        return _sample_scenario_monte_carlo(self, graph, rng)


class OutOfSampleMonteCarlo(SamplingScheme):
    """Monte Carlo sampling with user-supplied out-of-sample noise / transitions.

    ``f(node)`` returns ``(children: list[Noise], noise_terms: list[Noise])`` — or only the
    noise terms if ``use_insample_transition``. ``f(root_node)`` returns the root children.
    """

    def __init__(
        self,
        f: Callable[[Any], Any],
        graph: PolicyGraph,
        use_insample_transition: bool = False,
        max_depth: int = 0,
        terminate_on_cycle: bool = False,
        terminate_on_dummy_leaf: bool = True,
        rollout_limit: Callable[[int], int] = lambda i: sys.maxsize,
        initial_node: Any = None,
    ):
        if not terminate_on_cycle and not terminate_on_dummy_leaf and max_depth == 0:
            raise ValueError(
                "terminate_on_cycle and terminate_on_dummy_leaf cannot both be false when max_depth=0."
            )
        self.noise_terms: dict[Any, list[Noise]] = {}
        self.children: dict[Any, list[Noise]] = {}
        if use_insample_transition:
            self.root_children = list(graph.root_children)
        else:
            self.root_children = list(f(graph.root_node))
        for key in graph.nodes:
            if use_insample_transition:
                child = list(graph.nodes[key].children)
                noise = f(key)
            else:
                child, noise = f(key)
            self.noise_terms[key] = list(noise)
            self.children[key] = list(child)
        self.terminate_on_cycle = terminate_on_cycle
        self.terminate_on_dummy_leaf = terminate_on_dummy_leaf
        self.max_depth = max_depth
        self.rollout_limit = _Rollout(rollout_limit)
        self.initial_node = initial_node

    def get_noise_terms(self, node: Node, node_index: Any) -> Sequence[Noise]:
        return self.noise_terms[node_index]

    def get_children(self, node: Node, node_index: Any) -> Sequence[Noise]:
        return self.children[node_index]

    def get_root_children(self, graph: PolicyGraph) -> Sequence[Noise]:
        return self.root_children

    def sample_scenario(self, graph: PolicyGraph, rng: random.Random) -> tuple[list[tuple[Any, Any]], bool]:
        return _sample_scenario_monte_carlo(self, graph, rng)


def _sample_scenario_monte_carlo(
    scheme: InSampleMonteCarlo | OutOfSampleMonteCarlo, graph: PolicyGraph, rng: random.Random
) -> tuple[list[tuple[Any, Any]], bool]:
    max_depth = min(scheme.max_depth, scheme.rollout_limit())
    scenario_path: list[tuple[Any, Any]] = []
    visited_nodes: set = set()
    node_index = (
        scheme.initial_node
        if scheme.initial_node is not None
        else sample_noise(scheme.get_root_children(graph), rng)
    )
    while True:
        node = graph[node_index]
        noise_terms = scheme.get_noise_terms(node, node_index)
        children = scheme.get_children(node, node_index)
        noise = sample_noise(noise_terms, rng)
        scenario_path.append((node_index, noise))
        if len(children) == 0:
            return scenario_path, False
        elif scheme.terminate_on_cycle and node_index in visited_nodes:
            return scenario_path, True
        elif 0 < max_depth <= len(scenario_path):
            return scenario_path, False
        elif scheme.terminate_on_dummy_leaf and rng.random() < 1 - sum(c.probability for c in children):
            return scenario_path, False
        if scheme.terminate_on_cycle:
            visited_nodes.add(node_index)
        node_index = sample_noise(children, rng)


class Historical(SamplingScheme):
    """Sample from a fixed list of scenarios.

    * ``Historical(scenarios, probability)`` samples a scenario with the given probability.
    * ``Historical(scenarios)`` iterates the list sequentially (cycling).
    * ``Historical(scenario)`` (a single list of ``(node, noise)`` tuples) always returns it.
    """

    def __init__(
        self,
        scenarios: Sequence[Any],
        probability: Sequence[float] | None = None,
        terminate_on_cycle: bool = False,
    ):
        scenarios = list(scenarios)
        if scenarios and isinstance(scenarios[0], tuple):
            scenarios = [scenarios]  # single scenario
        scenarios = [[(n, w) for n, w in s] for s in scenarios]
        if probability is not None:
            if not math.isclose(sum(probability), 1.0, rel_tol=1.4901161193847656e-08):
                raise ValueError(
                    f"Probability of historical scenarios must sum to 1. Currently: {sum(probability)}."
                )
            self.scenarios = [Noise(s, float(p)) for s, p in zip(scenarios, probability)]
            self.sequential = False
        else:
            self.scenarios = [Noise(s, math.nan) for s in scenarios]
            self.sequential = True
        self.counter = 0
        self.terminate_on_cycle = terminate_on_cycle

    def __repr__(self) -> str:
        how = "sequentially" if self.sequential else "probabilistically"
        return f"A Historical sampler with {len(self.scenarios)} scenarios sampled {how}."

    def sample_scenario(self, graph: PolicyGraph, rng: random.Random) -> tuple[list[tuple[Any, Any]], bool]:
        ret = self.terminate_on_cycle
        if self.sequential:
            self.counter += 1
            if self.counter > len(self.scenarios):
                self.counter = 1
            return list(self.scenarios[self.counter - 1].term), ret
        return list(sample_noise(self.scenarios, rng)), ret


class PSRSamplingScheme(SamplingScheme):
    """Cache ``N`` scenarios from ``sampling_scheme`` and cycle through them (like PSR does)."""

    def __init__(self, N: int, sampling_scheme: SamplingScheme | None = None):
        self.N = N
        self.sampling_scheme = sampling_scheme if sampling_scheme is not None else InSampleMonteCarlo()
        self.scenarios: list[tuple[list[tuple[Any, Any]], bool]] = []
        self.counter = 0

    def __repr__(self) -> str:
        return f"A sampler with {len(self.scenarios)} scenarios like PSR does."

    def sample_scenario(self, graph: PolicyGraph, rng: random.Random) -> tuple[list[tuple[Any, Any]], bool]:
        self.counter += 1
        if self.counter > self.N:
            self.counter = 1
        if self.counter > len(self.scenarios):
            self.scenarios.append(self.sampling_scheme.sample_scenario(graph, rng))
        path, flag = self.scenarios[self.counter - 1]
        return list(path), flag
