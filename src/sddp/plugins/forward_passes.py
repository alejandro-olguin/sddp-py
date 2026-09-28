# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Forward passes. Ported from ``src/plugins/forward_passes.jl``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sddp.plugins.base import ForwardPass, RiskMeasure
from sddp.policy_graph import PolicyGraph, initialize_objective_state, update_objective_state

if TYPE_CHECKING:
    from sddp.algorithm import Options


@dataclass
class ForwardPassResult:
    scenario_path: list[tuple[Any, Any]]
    sampled_states: list[dict[str, float]]
    objective_states: list[tuple[float, ...]]
    belief_states: list[tuple[int, dict[Any, float]]]
    cumulative_value: float
    extra: dict[str, Any] = field(default_factory=dict)


class DefaultForwardPass(ForwardPass):
    """The default forward pass. ``include_last_node=False`` drops the node that closed a cycle."""

    def __init__(self, include_last_node: bool = True):
        self.include_last_node = include_last_node

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        from sddp.algorithm import distance, initialize_belief, solve_subproblem

        scenario_path, terminated_due_to_cycle = options.sampling_scheme.sample_scenario(
            model, options.rng
        )
        final_node = scenario_path[-1]
        if terminated_due_to_cycle and not self.include_last_node:
            scenario_path.pop()
        sampled_states: list[dict[str, float]] = []
        belief_states: list[tuple[int, dict[Any, float]]] = []
        current_belief = initialize_belief(model)
        incoming_state_value = dict(options.initial_state)
        cumulative_value = 0.0
        objective_state_vector, N = initialize_objective_state(model[scenario_path[0][0]])
        objective_states: list[tuple[float, ...]] = []
        for depth, (node_index, noise) in enumerate(scenario_path, start=1):
            node = model[node_index]
            node.lock.acquire()
            objective_state_vector = update_objective_state(
                node.objective_state, objective_state_vector, noise
            )
            if objective_state_vector is not None:
                objective_states.append(objective_state_vector)
            if node.belief_state is not None:
                belief = node.belief_state
                partition_index = belief.partition_index
                current_belief = belief.updater(
                    belief.belief, current_belief, partition_index, noise
                )
                belief_states.append((partition_index, dict(current_belief)))
            # ===== starting state for infinite horizon =====
            starting_states = options.starting_states[node_index]
            if len(starting_states) > 0:
                if (
                    distance(starting_states, incoming_state_value)
                    > options.cycle_discretization_delta
                ):
                    starting_states.append(incoming_state_value)
                idx = options.rng.randrange(len(starting_states))
                incoming_state_value = starting_states.pop(idx)
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path[:depth],
                duality_handler=None,
            )
            node.lock.release()
            cumulative_value += subproblem_results.stage_objective
            incoming_state_value = dict(subproblem_results.state)
            sampled_states.append(incoming_state_value)
        if terminated_due_to_cycle:
            starting_states = options.starting_states[final_node[0]]
            incoming_state_value = (
                sampled_states[-2] if self.include_last_node else sampled_states[-1]
            )
            if distance(starting_states, incoming_state_value) > options.cycle_discretization_delta:
                starting_states.append(incoming_state_value)
        return ForwardPassResult(
            scenario_path, sampled_states, objective_states, belief_states, cumulative_value
        )


class RevisitingForwardPass(ForwardPass):
    """Generate ``period`` new passes, then revisit all previous ones, repeatedly."""

    def __init__(self, period: int = 500, sub_pass: ForwardPass | None = None):
        assert period > 0
        self.period = period
        self.sub_pass = sub_pass if sub_pass is not None else DefaultForwardPass()
        self.archive: list[ForwardPassResult] = []
        self.last_index = 0
        self.counter = 0

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        self.counter += 1
        if self.counter - self.period > self.last_index:
            self.counter = 1
            self.last_index = len(self.archive)
        if self.counter <= len(self.archive):
            return self.archive[self.counter - 1]
        result = self.sub_pass.forward_pass(model, options)
        self.archive.append(result)
        return result


class RiskAdjustedForwardPass(ForwardPass):
    """Resample a previous pass with probability ``resampling_probability`` (risk-adjusted)."""

    def __init__(
        self,
        forward_pass: ForwardPass,
        risk_measure: RiskMeasure,
        resampling_probability: float,
        rejection_count: int = 5,
    ):
        if not (0 < resampling_probability < 1):
            raise ValueError("Resampling probability must be in `(0, 1)`")
        self.forward_pass_ = forward_pass
        self.risk_measure = risk_measure
        self.resampling_probability = resampling_probability
        self.rejection_count = rejection_count
        self.objectives: list[float] = []
        self.nominal_probability: list[float] = []
        self.adjusted_probability: list[float] = []
        self.archive: list[ForwardPassResult] = []
        self.resample_count: list[int] = []

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        rng = options.rng
        if len(self.archive) > 0 and rng.random() < self.resampling_probability:
            r = rng.random()
            for i in range(len(self.adjusted_probability)):
                r -= self.adjusted_probability[i]
                if r > 1e-8:
                    continue
                result = self.archive[i]
                if self.resample_count[i] >= self.rejection_count:
                    del self.objectives[i]
                    del self.nominal_probability[i]
                    del self.adjusted_probability[i]
                    del self.archive[i]
                    del self.resample_count[i]
                else:
                    self.resample_count[i] += 1
                return result
        result = self.forward_pass_.forward_pass(model, options)
        self.objectives.append(result.cumulative_value)
        self.nominal_probability.append(0.0)
        n = len(self.nominal_probability)
        self.nominal_probability[:] = [1 / n] * n
        self.adjusted_probability.append(0.0)
        self.archive.append(result)
        self.resample_count.append(1)
        self.risk_measure.adjust_probability(
            self.adjusted_probability,
            self.nominal_probability,
            self.objectives,
            self.objectives,
            model.is_minimization,
        )
        return result


class RegularizedForwardPass(ForwardPass):
    """Trust-region regularisation of the first-stage outgoing state (investment problems)."""

    def __init__(self, rho: float = 0.05, forward_pass: ForwardPass | None = None):
        self.forward_pass_ = forward_pass if forward_pass is not None else DefaultForwardPass()
        self.trial_centre: dict[str, float] = {}
        self.old_bounds: dict[str, tuple[float, float]] = {}
        self.rho = rho

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        if len(model.root_children) != 1:
            raise ValueError(
                "RegularizedForwardPass cannot be applied because first-stage is not deterministic"
            )
        node = model[model.root_children[0].term]
        if len(node.noise_terms) > 1:
            raise ValueError(
                "RegularizedForwardPass cannot be applied because first-stage is not deterministic"
            )
        m = node.model
        for k, v in node.states.items():
            if not (m.has_lower_bound(v.out) and m.has_upper_bound(v.out)):
                continue
            if k not in self.old_bounds:
                self.old_bounds[k] = (m.lower_bound(v.out), m.upper_bound(v.out))
            l, u = self.old_bounds[k]
            x = self.trial_centre.get(k, model.initial_root_state[k])
            m.set_lower_bound(v.out, max(l, x - self.rho * (u - l)))
            m.set_upper_bound(v.out, min(u, x + self.rho * (u - l)))
        result = self.forward_pass_.forward_pass(model, options)
        for k, (l, u) in self.old_bounds.items():
            self.trial_centre[k] = result.sampled_states[0][k]
            m.set_lower_bound(node.states[k].out, l)
            m.set_upper_bound(node.states[k].out, u)
        return result


__all__ = [
    "DefaultForwardPass",
    "ForwardPassResult",
    "RegularizedForwardPass",
    "RevisitingForwardPass",
    "RiskAdjustedForwardPass",
    "math",
]


class ImportanceSamplingForwardPass(ForwardPass):
    """Risk-adjusted exploration of Dias Garcia et al. (2023): the next node/noise is sampled
    from the risk-adjusted probabilities of the multi-cut cost-to-go variables."""

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        from sddp.algorithm import solve_subproblem
        from sddp.plugins.sampling_schemes import sample_noise
        from sddp.policy_graph import Noise

        assert not model.belief_partition
        rng = options.rng
        scenario_path: list[tuple[Any, Any]] = []
        sampled_states: list[dict[str, float]] = []
        cumulative_value = 0.0
        incoming_state_value = dict(options.initial_state)
        node_index = sample_noise(model.root_children, rng)
        node = model[node_index]
        noise = sample_noise(node.noise_terms, rng)
        while node_index is not None:
            node = model[node_index]
            with node.lock:
                scenario_path.append((node_index, noise))
                results = solve_subproblem(
                    model, node, incoming_state_value, noise, scenario_path, duality_handler=None
                )
                cumulative_value += results.stage_objective
                incoming_state_value = dict(results.state)
                sampled_states.append(incoming_state_value)
                if not node.bellman_function.local_thetas:
                    node_index = sample_noise(node.children, rng)
                    if node_index is not None:
                        noise = sample_noise(model[node_index].noise_terms, rng)
                else:
                    objectives = [
                        node.model.value(t.theta) for t in node.bellman_function.local_thetas
                    ]
                    nominal: list[float] = []
                    support: list[Any] = []
                    for child in node.children:
                        for w in model[child.term].noise_terms:
                            nominal.append(child.probability * w.probability)
                            support.append((child.term, w.term))
                    assert len(nominal) == len(objectives)
                    adjusted = [0.0] * len(objectives)
                    options.risk_measures[node_index].adjust_probability(
                        adjusted, nominal, support, objectives, model.is_minimization
                    )
                    terms: list[Noise] = [Noise(s, p) for s, p in zip(support, adjusted)]
                    picked = sample_noise(terms, rng)
                    node_index, noise = picked
        return ForwardPassResult(scenario_path, sampled_states, [], [], cumulative_value)


class LoggingForwardPass(ForwardPass):
    """Wrap a forward pass and append the sampled states to a CSV file."""

    def __init__(self, inner: ForwardPass | None = None, *, filename: str):
        self.inner = inner if inner is not None else DefaultForwardPass()
        self.filename = filename
        self.iteration = 0

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        ret = self.inner.forward_pass(model, options)
        with options.lock:
            self.iteration += 1
            with open(self.filename, "a") as io:
                for index, state in enumerate(ret.sampled_states, start=1):
                    keys = sorted(state)
                    if self.iteration == 1 and index == 1:
                        io.write("iteration,index" + "".join(f",{k}" for k in keys) + "\n")
                    io.write(
                        f"{self.iteration},{index}" + "".join(f",{state[k]}" for k in keys) + "\n"
                    )
        return ret


class AlternativeForwardPass(ForwardPass):
    """Simulate the forward pass on ``forward_model`` (e.g. a non-convex model) while the
    backward pass uses the trained model. Pair with :class:`AlternativePostIterationCallback`."""

    def __init__(self, forward_model: PolicyGraph, forward_pass: ForwardPass | None = None):
        self.model = forward_model
        self.forward_pass_ = forward_pass if forward_pass is not None else DefaultForwardPass()

    def forward_pass(self, model: PolicyGraph, options: Options) -> ForwardPassResult:
        return self.forward_pass_.forward_pass(self.model, options)


class AlternativePostIterationCallback:
    """Copy the cuts of each iteration into ``forward_model``."""

    def __init__(self, forward_model: PolicyGraph):
        self.model = forward_model

    def __call__(self, result: Any) -> None:
        from sddp.plugins.parallel_schemes import slave_update

        with self.model.lock:
            slave_update(self.model, result)
