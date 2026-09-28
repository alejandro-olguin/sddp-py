# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2019: Kipngeno Kirui (lattice approximation, MIT, via ScenTrees.jl).
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Fitting Markov chains to sample paths. Ported from ``src/modeling_aids.jl`` and the
``SimulatorSamplingScheme`` in ``src/plugins/sampling_schemes.jl``."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sddp.graph import Graph
from sddp.plugins.base import SamplingScheme
from sddp.plugins.sampling_schemes import sample_noise
from sddp.policy_graph import PolicyGraph


def find_min(x: Sequence[float], y: float) -> tuple[float, int]:
    """``(min |x_i - y|, argmin)`` with a 1-based index like SDDP.jl."""
    best_i, best_z = 0, math.inf
    for i, xi in enumerate(x, start=1):
        z = abs(xi - y)
        if z < best_z:
            best_i, best_z = i, z
    return best_z, best_i


def _quantiles(x: Sequence[float], N: int) -> list[float]:
    if N == 1:
        return [float(np.mean(x))]
    return [float(v) for v in np.quantile(np.asarray(x, dtype=float), np.linspace(0.01, 0.99, N))]


def lattice_approximation(
    f: Callable[[], Sequence[float]] | None,
    states: Sequence[int],
    scenarios: int,
    simulations: Sequence[Sequence[float]] | None = None,
) -> tuple[list[list[float]], list[np.ndarray]]:
    """Fit a lattice with ``states[t]`` nodes per stage to ``scenarios`` sample paths.

    Returns ``(support, probability)``: ``support[t]`` lists the node values of stage ``t``
    and ``probability[t]`` is the ``(states[t-1], states[t])`` transition matrix
    (``(1, states[0])`` for the first stage).
    """
    if simulations is None:
        assert f is not None
        simulations = [list(f()) for _ in range(scenarios)]
    sims = [list(map(float, s)) for s in simulations]
    T = len(states)
    matrix = np.array(sims, dtype=float)  # scenarios × T
    support = [_quantiles(list(matrix[:, t]), states[t]) for t in range(T)]
    probability = [np.zeros((1, states[0]))] + [
        np.zeros((states[t - 1], states[t])) for t in range(1, T)
    ]
    distance = 0.0
    for n, path in enumerate(sims, start=1):
        dist, last_index = 0.0, 1
        for t in range(T):
            # SDDP.jl writes `for i in 1:length(states[t])`, and `length(::Int) == 1`, so only
            # the first node of each stage is ever re-seeded. Replicated for parity.
            for i in range(1):
                if probability[t][:, i].sum() < 1.3 * math.sqrt(n) / states[t]:
                    support[t][i] = path[t]
            min_dist, best_idx = find_min(support[t], path[t])
            dist += min_dist**2
            probability[t][last_index - 1, best_idx - 1] += 1.0
            support[t][best_idx - 1] -= (
                min_dist * (support[t][best_idx - 1] - path[t]) / (3000 + n) ** 0.75
            )
            last_index = best_idx
        distance = (distance * (n - 1) + dist) / n
    for p in probability:
        with np.errstate(invalid="ignore", divide="ignore"):
            p /= p.sum(axis=1, keepdims=True)
        p[np.isnan(p).any(axis=1), :] = 0.0
    return support, probability


def allocate_support_budget(
    f_or_simulations: Callable[[], Sequence[float]] | Sequence[Sequence[float]],
    budget: int | Sequence[int],
    scenarios: int,
) -> list[int]:
    """Split ``budget`` nodes over the stages in proportion to the stage variance."""
    if not isinstance(budget, int):
        return list(budget)
    if callable(f_or_simulations):
        simulations = [list(f_or_simulations()) for _ in range(scenarios)]
    else:
        simulations = [list(s) for s in f_or_simulations]
    matrix = np.array(simulations, dtype=float)
    stage_var = matrix.var(axis=0, ddof=1)
    states = [1] * len(stage_var)
    if budget < len(stage_var):
        warnings.warn(
            "Budget for nodes is less than the number of stages. Using one node per stage.",
            stacklevel=2,
        )
        return states
    s = float(stage_var.sum())
    if math.isclose(s, 0.0, rel_tol=1.4901161193847656e-08):
        return states
    for i in range(len(states)):
        states[i] = max(1, round(budget * float(stage_var[i]) / s))
    while sum(states) != budget:
        if sum(states) > budget:
            states[int(np.argmax(states))] -= 1
        else:
            states[int(np.argmin(states))] += 1
    return states


def markovian_graph_from_simulator(
    simulator: Callable[[], Sequence[float]], budget: int | Sequence[int], scenarios: int = 1000
) -> Graph:
    """``MarkovianGraph(simulator; budget, scenarios)``: nodes are ``(t, value)`` tuples with
    root ``(0, 0.0)``."""
    scenarios = max(scenarios, 10)
    simulations = [list(map(float, simulator())) for _ in range(scenarios)]
    states = allocate_support_budget(simulations, budget, scenarios)
    support, probability = lattice_approximation(None, states, scenarios, simulations)
    g: Graph = Graph((0, 0.0))
    for i, si in enumerate(support[0]):
        g._add_node_if_missing((1, si))
        g._add_to_or_create_edge((0, 0.0), (1, si), float(probability[0][0, i]))
    for t in range(1, len(support)):
        for j, sj in enumerate(support[t]):
            g._add_node_if_missing((t + 1, sj))
            for i, si in enumerate(support[t - 1]):
                g._add_to_or_create_edge((t, si), (t + 1, sj), float(probability[t][i, j]))
    return g


def _closest_index(graph: PolicyGraph, t: int, value: float) -> tuple[int, float]:
    min_value, min_dist = value, math.inf
    for t_, value_ in graph.nodes:
        if t_ == t and abs(value - value_) < min_dist:
            min_value, min_dist = value_, abs(value - value_)
    return (t, min_value)


class SimulatorSamplingScheme(SamplingScheme):
    """Sample paths from ``simulator()`` mapped onto the closest nodes of a simulator-fitted
    Markovian graph. Noise terms of the nodes must be tuples whose first element is the Markov
    state; the sampled noise is ``(value,)`` or ``(value, second_element)``."""

    def __init__(self, simulator: Callable[[], Sequence[float]]):
        self.simulator = simulator

    def __repr__(self) -> str:
        return "SimulatorSamplingScheme"

    def sample_scenario(
        self, graph: PolicyGraph, rng: random.Random
    ) -> tuple[list[tuple[Any, Any]], bool]:
        scenario_path: list[tuple[Any, Any]] = []
        for t, value in enumerate(self.simulator(), start=1):
            node_index = _closest_index(graph, t, float(value))
            node = graph[node_index]
            noise = sample_noise(node.noise_terms, rng)
            assert noise[0] == node_index[1]
            omega = (float(value),) if len(noise) == 1 else (float(value), noise[1])
            scenario_path.append((node_index, omega))
        return scenario_path, False
