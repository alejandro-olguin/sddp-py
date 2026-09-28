# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""The core SDDP algorithm. Ported from ``src/algorithm.jl``.

Forward pass, backward pass, bound calculation, ``train``, ``simulate``, and
decision-rule evaluation.
"""

from __future__ import annotations

import math
import random
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

import pyoptinterface as _poi

from sddp.plugins.backward_sampling_schemes import CompleteSampler
from sddp.plugins.base import (
    BackwardSamplingScheme,
    DualityHandler,
    ForwardPass,
    ParallelScheme,
    RiskMeasure,
    SamplingScheme,
    StoppingRule,
    convergence_test,
)
from sddp.plugins.bellman_functions import CutType, bellman_term, refine_bellman_function
from sddp.plugins.duality_handlers import ContinuousConicDuality, get_dual_solution_none
from sddp.plugins.forward_passes import DefaultForwardPass, ForwardPassResult
from sddp.plugins.parallel_schemes import Serial
from sddp.plugins.risk_measures import Expectation
from sddp.plugins.sampling_schemes import InSampleMonteCarlo
from sddp.plugins.stopping_rules import IterationLimit, SimulationStoppingRule, TimeLimit
from sddp.policy_graph import (
    Log,
    Node,
    Noise,
    PolicyGraph,
    StateValue,
    TrainingResults,
    build_phi,
    get_belief_state_component,
    get_objective_state_component,
    initialize_objective_state,
    update_objective_state,
)
from sddp.print import (
    print_banner,
    print_footer,
    print_iteration,
    print_iteration_header,
    print_problem_statistics,
)
from sddp.solver.model import Model, Sense, to_expression

_INTEGER = _poi.VariableDomain.Integer
_BINARY = _poi.VariableDomain.Binary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_nodal_form(model: PolicyGraph, element: Any) -> dict[Any, Any]:
    """Expand ``element`` (scalar / dict / callable) into a per-node dict."""
    if isinstance(element, dict):
        for key in model.nodes:
            if key not in element:
                raise KeyError(f"Missing key: {key}.")
        return element
    if callable(element) and not isinstance(element, RiskMeasure):
        return {k: element(k) for k in model.nodes}
    return {k: element for k in model.nodes}


def get_same_children(model: PolicyGraph) -> dict[Any, list[Any]]:
    tmp: dict[frozenset, set] = {}
    for key, node in model.nodes.items():
        children = frozenset(child.term for child in node.children)
        if len(children) == 0:
            continue
        tmp.setdefault(children, set()).add(key)
    same_children: dict[Any, list[Any]] = {key: [] for key in model.nodes}
    for s in tmp.values():
        for v in s:
            same_children[v] = [x for x in s if x != v]
    return same_children


@dataclass
class Options:
    """Internal storage for training options and cached data."""

    initial_state: dict[str, float]
    sampling_scheme: SamplingScheme
    backward_sampling_scheme: BackwardSamplingScheme
    starting_states: dict[Any, list[dict[str, float]]]
    risk_measures: dict[Any, RiskMeasure]
    cycle_discretization_delta: float
    refine_at_similar_nodes: bool
    phi: dict[tuple[Any, Any], float]
    similar_children: dict[Any, list[Any]]
    stopping_rules: list[StoppingRule]
    print_level: int
    start_time: float
    log: list[Log]
    log_file_handle: TextIO | None
    log_frequency: Callable[[list[Log]], bool]
    forward_pass: ForwardPass
    duality_handler: DualityHandler
    forward_pass_callback: Callable[[ForwardPassResult], Any]
    post_iteration_callback: Callable[[IterationResult], Any]
    root_node_risk_measure: RiskMeasure
    rng: random.Random
    last_log_iteration: int = 0
    dashboard_callback: Callable[..., Any] = lambda a, b: None
    lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def create(
        cls,
        model: PolicyGraph,
        initial_state: dict[str, float],
        *,
        sampling_scheme: SamplingScheme | None = None,
        backward_sampling_scheme: BackwardSamplingScheme | None = None,
        risk_measures: Any = None,
        cycle_discretization_delta: float = 0.0,
        refine_at_similar_nodes: bool = True,
        stopping_rules: Sequence[StoppingRule] = (),
        print_level: int = 0,
        start_time: float = 0.0,
        log: list[Log] | None = None,
        log_file_handle: TextIO | None = None,
        log_frequency: Callable[[list[Log]], bool] | int = 1,
        forward_pass: ForwardPass | None = None,
        duality_handler: DualityHandler | None = None,
        forward_pass_callback: Callable[[ForwardPassResult], Any] = lambda x: None,
        post_iteration_callback: Callable[[Any], Any] = lambda r: None,
        root_node_risk_measure: RiskMeasure | None = None,
        rng: random.Random | None = None,
    ) -> Options:
        log_frequency_fn: Callable[[list[Log]], bool]
        if isinstance(log_frequency, int):
            n = log_frequency

            def _every_n(lg: list[Log]) -> bool:
                return len(lg) % n == 0

            log_frequency_fn = _every_n
        else:
            log_frequency_fn = log_frequency
        return cls(
            initial_state=initial_state,
            sampling_scheme=sampling_scheme or InSampleMonteCarlo(),
            backward_sampling_scheme=backward_sampling_scheme or CompleteSampler(),
            starting_states={k: [] for k in model.nodes},
            risk_measures=to_nodal_form(
                model, risk_measures if risk_measures is not None else Expectation()
            ),
            cycle_discretization_delta=cycle_discretization_delta,
            refine_at_similar_nodes=refine_at_similar_nodes,
            phi=build_phi(model),
            similar_children=get_same_children(model),
            stopping_rules=list(stopping_rules),
            print_level=print_level,
            start_time=start_time,
            log=log if log is not None else [],
            log_file_handle=log_file_handle,
            log_frequency=log_frequency_fn,
            forward_pass=forward_pass or DefaultForwardPass(),
            duality_handler=duality_handler or ContinuousConicDuality(),
            forward_pass_callback=forward_pass_callback,
            post_iteration_callback=post_iteration_callback,
            root_node_risk_measure=root_node_risk_measure or Expectation(),
            rng=rng if rng is not None else random.Random(),
        )


# ---------------------------------------------------------------------------
# Subproblem solves
# ---------------------------------------------------------------------------
def set_incoming_state(node: Node, state: dict[str, float]) -> None:
    for state_name, value in state.items():
        node.model.fix(node.states[state_name].in_, value)


def _outgoing_info(node: Node) -> list[tuple[str, Any, float, float, bool]]:
    """``(name, out_var, lb, ub, is_discrete)`` per state, cached until a state's bounds change."""
    m = node.model
    cached = node.ext.get("_outgoing_info")
    if cached is not None and cached[0] == m.watch_version:
        return cached[1]
    info = []
    for name, state in node.states.items():
        d = m.domain(state.out)
        info.append(
            (
                name,
                state.out,
                m.lower_bound(state.out),
                m.upper_bound(state.out),
                d in (_INTEGER, _BINARY),
            )
        )
    node.ext["_outgoing_info"] = (m.watch_version, info)
    return info


def get_outgoing_state(node: Node) -> dict[str, float]:
    """Outgoing state values, projected onto their bounds and rounded if discrete."""
    m = node.model
    values: dict[str, float] = {}
    for name, out, lb, ub, discrete in _outgoing_info(node):
        outgoing_value = m.value(out)
        if ub < outgoing_value:
            outgoing_value = ub
        if lb > outgoing_value:
            outgoing_value = lb
        if discrete:
            outgoing_value = float(round(outgoing_value))
        values[name] = outgoing_value
    return values


def set_objective(node: Node) -> None:
    """Set the subproblem objective to ``stage_objective + <y,μ> + <b,μ> + θ``."""
    obj_component = get_objective_state_component(node)
    belief_component = get_belief_state_component(node)
    if obj_component or belief_component:
        node.stage_objective_set = False
    if not node.stage_objective_set:
        m = node.model
        expr = Model.expression(
            obj_component + belief_component + [(bellman_term(node.bellman_function), 1.0)]
        )
        total = to_expression(node.stage_objective) + expr
        m.set_objective(total, m.objective_sense)
    node.stage_objective_set = True


def stage_objective_value(node: Node) -> float:
    m = node.model
    if node.objective_state is not None or node.belief_state is not None:
        return m.value(node.stage_objective)
    return m.objective_value() - m.value(bellman_term(node.bellman_function))


def parameterize(node: Node, noise: Any) -> None:
    """Parameterize ``node`` with ``noise`` and (re)set the objective."""
    node.parameterize_fn(noise)
    set_objective(node)


def write_subproblem_to_file(node: Node, filename: str, throw_error: bool = False) -> None:
    node.model.write(filename)
    if throw_error:
        m = node.model
        raise RuntimeError(
            f"Unable to retrieve solution from node {node.index}.\n\n"
            f"  Termination status : {m.termination_status()}\n"
            f"  Primal status      : {m.primal_status()}\n"
            f"  Dual status        : {m.dual_status()}.\n\n"
            f"The current subproblem was written to `{filename}`.\n\n"
            "There are two common causes of this error:\n"
            "  1) you have a mistake in your formulation, or you violated\n"
            "     the assumption of relatively complete recourse\n"
            "  2) the solver encountered numerical issues\n\n"
            "See https://odow.github.io/SDDP.jl/stable/tutorial/warnings/ for more information."
        )


def set_numerical_difficulty_callback(model: PolicyGraph, callback: Callable[..., Any]) -> None:
    model.ext["numerical_difficulty_callback"] = callback


def default_numerical_difficulty_callback(
    model: PolicyGraph, node: Node, require_dual: bool = False
) -> None:
    """Recovery ladder: cold restart (like SDDP.jl's ``reset_optimizer``), then presolve off,
    then an interior-point solve, stopping at the first attempt that yields a solution."""
    m = node.model

    def ok() -> bool:
        return m.has_primal_solution() and (not require_dual or m.has_dual_solution())

    if m.reset_optimizer():
        m.optimize()
        if ok():
            return
    if m.optimizer.name == "HiGHS":
        m.solve_with_options(presolve="off")
        if ok():
            return
        m.reset_optimizer()
        m.solve_with_options(solver="ipm", presolve="off", output_flag=False)
        if ok():
            return
    m.optimize()


def attempt_numerical_recovery(model: PolicyGraph, node: Node, require_dual: bool = False) -> None:
    from sddp.plugins.bellman_functions import write_cuts_to_file

    with model.lock:
        model.ext["numerical_issue"] = True
    callback = model.ext.get("numerical_difficulty_callback", default_numerical_difficulty_callback)
    callback(model, node, require_dual=require_dual)
    missing_dual = require_dual and not node.model.has_dual_solution()
    if not node.model.has_primal_solution() or missing_dual:
        filename = f"model_infeasible_node_{node.index}.cuts.json"
        try:
            write_cuts_to_file(model, filename)
            print(f"Writing cuts to the file `{filename}`", file=sys.stderr)
        except ValueError:
            pass
        write_subproblem_to_file(node, f"subproblem_{node.index}.lp", throw_error=True)


@dataclass
class SubproblemResult:
    state: dict[str, float]
    duals: dict[str, float]
    objective: float
    stage_objective: float


def solve_subproblem(
    model: PolicyGraph,
    node: Node,
    state: dict[str, float],
    noise: Any,
    scenario_path: list[tuple[Any, Any]],
    duality_handler: DualityHandler | None,
) -> SubproblemResult:
    """Solve ``node`` at incoming ``state`` with ``noise`` and extract the results."""
    set_incoming_state(node, state)
    parameterize(node, noise)
    pre_optimize_ret = None
    if node.pre_optimize_hook is not None:
        pre_optimize_ret = node.pre_optimize_hook(
            model, node, state, noise, scenario_path, duality_handler
        )
    node.model.optimize()
    with model.lock:
        model.ext["total_solves"] = model.ext.get("total_solves", 0) + 1
    if not node.model.has_primal_solution():
        attempt_numerical_recovery(model, node)
    out_state = get_outgoing_state(node)
    stage_obj = stage_objective_value(node)
    if duality_handler is None:
        objective, dual_values = get_dual_solution_none(node)
    else:
        objective, dual_values = duality_handler.get_dual_solution(node)
    if node.post_optimize_hook is not None:
        node.post_optimize_hook(pre_optimize_ret)
    return SubproblemResult(out_state, dual_values, objective, stage_obj)


def initialize_belief(model: PolicyGraph) -> dict[Any, float]:
    current_belief = {k: 0.0 for k in model.nodes}
    current_belief[model.root_node] = 1.0
    return current_belief


def inf_norm(x: dict[str, float], y: dict[str, float]) -> float:
    norm = 0.0
    for key, value in y.items():
        if abs(x[key] - value) > norm:
            norm = abs(x[key] - value) / (1 + abs(value))
    return norm


def distance(starting_states: list[dict[str, float]], state: dict[str, float]) -> float:
    if len(starting_states) == 0:
        return math.inf
    return min(inf_norm(s, state) for s in starting_states)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------
@dataclass
class BackwardPassItems:
    cached_solutions: dict[tuple[Any, int], int] = field(default_factory=dict)
    duals: list[dict[str, float]] = field(default_factory=list)
    supports: list[Noise] = field(default_factory=list)
    nodes: list[Any] = field(default_factory=list)
    probability: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    belief: list[float] = field(default_factory=list)


def backward_pass(
    model: PolicyGraph,
    options: Options,
    scenario_path: list[tuple[Any, Any]],
    sampled_states: list[dict[str, float]],
    objective_states: list[tuple[float, ...]],
    belief_states: list[tuple[int, dict[Any, float]]],
) -> dict[Any, list[Any]]:
    cuts: dict[Any, list[Any]] = {index: [] for index in model.nodes}
    for index in range(len(scenario_path) - 1, -1, -1):
        outgoing_state = sampled_states[index]
        objective_state = objective_states[index] if index < len(objective_states) else None
        partition_index, belief_state = (
            belief_states[index] if index < len(belief_states) else (0, None)
        )
        items = BackwardPassItems()
        if belief_state is not None:
            for node_index, belief in belief_state.items():
                if belief == 0.0:
                    continue
                solve_all_children(
                    model,
                    model[node_index],
                    items,
                    belief,
                    belief_state,
                    objective_state,
                    outgoing_state,
                    options.backward_sampling_scheme,
                    scenario_path[: index + 1],
                    options.duality_handler,
                    options,
                )
            for node_index in model.belief_partition[partition_index]:
                node = model[node_index]
                with node.lock:
                    current_belief = node.belief_state
                    assert current_belief is not None
                    for idx, belief in belief_state.items():
                        current_belief.belief[idx] = belief
                    new_cuts = refine_bellman_function(
                        model,
                        node,
                        node.bellman_function,
                        options.risk_measures[node_index],
                        outgoing_state,
                        items.duals,
                        items.supports,
                        [p * b for p, b in zip(items.probability, items.belief)],
                        items.objectives,
                    )
                cuts[node_index].append(new_cuts)
        else:
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if len(node.children) == 0:
                continue
            solve_all_children(
                model,
                node,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[: index + 1],
                options.duality_handler,
                options,
            )
            new_cuts = refine_bellman_function(
                model,
                node,
                node.bellman_function,
                options.risk_measures[node_index],
                outgoing_state,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
            )
            cuts[node_index].append(new_cuts)
            if options.refine_at_similar_nodes:
                for other_index in options.similar_children[node_index]:
                    other_node = model[other_index]
                    other_children = {c.term for c in other_node.children}
                    assert not (other_children - set(items.nodes))
                    copied_probability = [
                        options.phi.get((other_index, child_index), 0.0)
                        * items.supports[idx].probability
                        for idx, child_index in enumerate(items.nodes)
                    ]
                    new_cuts = refine_bellman_function(
                        model,
                        other_node,
                        other_node.bellman_function,
                        options.risk_measures[other_index],
                        outgoing_state,
                        items.duals,
                        items.supports,
                        copied_probability,
                        items.objectives,
                    )
                    cuts[other_index].append(new_cuts)
    return cuts


def solve_all_children(
    model: PolicyGraph,
    node: Node,
    items: BackwardPassItems,
    belief: float,
    belief_state: dict[Any, float] | None,
    objective_state: tuple[float, ...] | None,
    outgoing_state: dict[str, float],
    backward_sampling_scheme: BackwardSamplingScheme,
    scenario_path: list[tuple[Any, Any]],
    duality_handler: DualityHandler | None,
    options: Options,
) -> None:
    length_scenario_path = len(scenario_path)
    for child in node.children:
        if belief_state is not None and math.isclose(child.probability, 0.0, abs_tol=1e-6):
            continue
        child_node = model[child.term]
        child_node.lock.acquire()
        try:
            _solve_child(
                model,
                node,
                child,
                child_node,
                items,
                belief,
                belief_state,
                objective_state,
                outgoing_state,
                backward_sampling_scheme,
                scenario_path,
                length_scenario_path,
                duality_handler,
                options,
            )
        finally:
            child_node.lock.release()
    if len(scenario_path) != length_scenario_path:
        scenario_path.pop()


def _solve_child(
    model: PolicyGraph,
    node: Node,
    child: Noise,
    child_node: Node,
    items: BackwardPassItems,
    belief: float,
    belief_state: dict[Any, float] | None,
    objective_state: tuple[float, ...] | None,
    outgoing_state: dict[str, float],
    backward_sampling_scheme: BackwardSamplingScheme,
    scenario_path: list[tuple[Any, Any]],
    length_scenario_path: int,
    duality_handler: DualityHandler | None,
    options: Options,
) -> None:
    if True:
        restore_duality = options.duality_handler.prepare_backward_pass(child_node, options)
        noise_terms = backward_sampling_scheme.sample_backward_noise_terms_with_state(
            child_node, outgoing_state, options.rng
        )
        for noise_idx, noise in enumerate(noise_terms):
            if len(scenario_path) == length_scenario_path:
                scenario_path.append((child.term, noise.term))
            else:
                scenario_path[-1] = (child.term, noise.term)
            key = (child.term, noise_idx)
            if key in items.cached_solutions:
                sol_index = items.cached_solutions[key]
                items.duals.append(items.duals[sol_index])
                items.supports.append(items.supports[sol_index])
                items.nodes.append(child_node.index)
                items.probability.append(items.probability[sol_index])
                items.objectives.append(items.objectives[sol_index])
                items.belief.append(belief)
            else:
                if belief_state is not None:
                    current_belief = child_node.belief_state
                    assert current_belief is not None
                    current_belief.updater(
                        current_belief.belief,
                        belief_state,
                        current_belief.partition_index,
                        noise.term,
                    )
                if objective_state is not None:
                    update_objective_state(child_node.objective_state, objective_state, noise.term)
                subproblem_results = solve_subproblem(
                    model,
                    child_node,
                    outgoing_state,
                    noise.term,
                    scenario_path,
                    duality_handler=duality_handler,
                )
                items.duals.append(subproblem_results.duals)
                items.supports.append(noise)
                items.nodes.append(child_node.index)
                items.probability.append(child.probability * noise.probability)
                items.objectives.append(subproblem_results.objective)
                items.belief.append(belief)
                items.cached_solutions[key] = len(items.duals) - 1
        restore_duality()


# ---------------------------------------------------------------------------
# Bound
# ---------------------------------------------------------------------------
def calculate_bound(
    model: PolicyGraph,
    root_state: dict[str, float] | None = None,
    risk_measure: RiskMeasure | None = None,
) -> float:
    """The deterministic bound (lower if minimising, upper if maximising) at the root."""
    if root_state is None:
        root_state = model.initial_root_state
    if risk_measure is None:
        risk_measure = Expectation()
    noise_supports: list[Any] = []
    probabilities: list[float] = []
    objectives: list[float] = []
    current_belief = initialize_belief(model)
    for child in model.root_children:
        if math.isclose(child.probability, 0.0, abs_tol=1e-6):
            continue
        node = model[child.term]
        with node.lock:
            for noise in node.noise_terms:
                if node.objective_state is not None:
                    update_objective_state(
                        node.objective_state, node.objective_state.initial_value, noise.term
                    )
                if node.belief_state is not None:
                    belief = node.belief_state
                    belief.updater(
                        belief.belief, current_belief, belief.partition_index, noise.term
                    )
                subproblem_results = solve_subproblem(
                    model,
                    node,
                    root_state,
                    noise.term,
                    [(child.term, noise.term)],
                    duality_handler=None,
                )
                objectives.append(subproblem_results.objective)
                probabilities.append(child.probability * noise.probability)
                noise_supports.append(noise.term)
    risk_adjusted_probability = [0.0] * len(probabilities)
    offset = risk_measure.adjust_probability(
        risk_adjusted_probability, probabilities, noise_supports, objectives, model.is_minimization
    )
    return sum(obj * prob for obj, prob in zip(objectives, risk_adjusted_probability)) + offset


# ---------------------------------------------------------------------------
# Iteration and training loop
# ---------------------------------------------------------------------------
@dataclass
class IterationResult:
    pid: int
    bound: float
    cumulative_value: float
    has_converged: bool
    status: str
    cuts: dict[Any, list[Any]]
    numerical_issue: bool


_THREAD_IDS: dict[int, int] = {}


def _pid() -> int:
    ident = threading.get_ident()
    if ident not in _THREAD_IDS:
        _THREAD_IDS[ident] = len(_THREAD_IDS) + 1
    return _THREAD_IDS[ident]


def iteration(model: PolicyGraph, options: Options) -> IterationResult:
    with model.lock:
        model.ext["numerical_issue"] = False
    forward_trajectory = options.forward_pass.forward_pass(model, options)
    options.forward_pass_callback(forward_trajectory)
    cuts = backward_pass(
        model,
        options,
        forward_trajectory.scenario_path,
        forward_trajectory.sampled_states,
        forward_trajectory.objective_states,
        forward_trajectory.belief_states,
    )
    bound = calculate_bound(model, risk_measure=options.root_node_risk_measure)
    with options.lock:
        numerical_issue = bool(model.ext["numerical_issue"])
        options.log.append(
            Log(
                len(options.log) + 1,
                bound,
                forward_trajectory.cumulative_value,
                time.time() - options.start_time,
                _pid(),
                model.ext.get("total_solves", 0),
                options.duality_handler.duality_log_key(),
                numerical_issue,
            )
        )
        has_converged, status = convergence_test(model, options.log, options.stopping_rules)
        return IterationResult(
            _pid(),
            bound,
            forward_trajectory.cumulative_value,
            has_converged,
            status,
            cuts,
            numerical_issue,
        )


def termination_status(model: PolicyGraph) -> str:
    if model.most_recent_training_results is None:
        return "model_not_solved"
    return model.most_recent_training_results.status


def _should_log(options: Options) -> bool:
    return options.print_level > 0 and options.log_frequency(options.log)


def log_iteration(options: Options, force_if_needed: bool = False) -> None:
    options.dashboard_callback(options.log[-1], False)
    force_if_needed = force_if_needed and options.last_log_iteration != len(options.log)
    if force_if_needed or _should_log(options):
        print_iteration(sys.stdout, options.log[-1])
        if options.log_file_handle is not None:
            print_iteration(options.log_file_handle, options.log[-1])
            options.log_file_handle.flush()
        options.last_log_iteration = len(options.log)


def _print_helper(f: Callable[..., Any], io: TextIO | None, *args: Any) -> None:
    f(sys.stdout, *args)
    if io is not None:
        f(io, *args)


def train(
    model: PolicyGraph,
    *,
    iteration_limit: int | None = None,
    time_limit: float | None = None,
    print_level: int = 1,
    log_file: str | None = None,
    log_frequency: int | None = None,
    log_every_seconds: float | None = None,
    log_every_iteration: bool = False,
    run_numerical_stability_report: bool = True,
    stopping_rules: Sequence[StoppingRule] = (),
    risk_measure: Any = None,
    root_node_risk_measure: RiskMeasure | None = None,
    sampling_scheme: SamplingScheme | None = None,
    cut_type: CutType = CutType.SINGLE_CUT,
    cycle_discretization_delta: float = 0.0,
    refine_at_similar_nodes: bool = True,
    cut_deletion_minimum: int = 1,
    backward_sampling_scheme: BackwardSamplingScheme | None = None,
    dashboard: bool = False,
    parallel_scheme: ParallelScheme | None = None,
    forward_pass: ForwardPass | None = None,
    forward_pass_resampling_probability: float | None = None,
    add_to_existing_cuts: bool = False,
    duality_handler: DualityHandler | None = None,
    forward_pass_callback: Callable[[ForwardPassResult], Any] = lambda x: None,
    post_iteration_callback: Callable[[IterationResult], Any] = lambda r: None,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> None:
    """Train the policy for ``model`` (``SDDP.train``).

    Keyword arguments mirror SDDP.jl. Additional Python-only arguments: ``seed`` /
    ``rng`` control the random number generator used for sampling.
    """
    from sddp.plugins.forward_passes import RiskAdjustedForwardPass

    if rng is None:
        rng = random.Random(seed)
    if risk_measure is None:
        risk_measure = Expectation()
    if root_node_risk_measure is None:
        root_node_risk_measure = Expectation()
    if sampling_scheme is None:
        sampling_scheme = InSampleMonteCarlo()
    if backward_sampling_scheme is None:
        backward_sampling_scheme = CompleteSampler()
    if parallel_scheme is None:
        parallel_scheme = Serial()
    elif not isinstance(parallel_scheme, Serial) and any(
        n.objective_state is not None for n in model.nodes.values()
    ):
        warnings.warn(
            "Threaded training is not supported with objective states; using Serial().",
            stacklevel=2,
        )
        parallel_scheme = Serial()
    if forward_pass is None:
        forward_pass = DefaultForwardPass()
    if duality_handler is None:
        duality_handler = ContinuousConicDuality()
    if log_every_seconds is None:
        log_every_seconds = -1.0 if log_frequency is None else 0.0
    log_frequency = 1 if log_frequency is None else log_frequency
    if log_frequency <= 0:
        raise ValueError(f"`log_frequency` must be at least `1`. Got {log_frequency}.")
    if log_every_iteration:
        log_frequency = 1
        log_every_seconds = 0.0

    def log_frequency_f(log: list[Log]) -> bool:
        if len(log) % log_frequency != 0:
            return False
        last = options.last_log_iteration
        if last == 0:
            return True
        elif last == len(log):
            return False
        seconds = log_every_seconds
        if log_every_seconds < 0.0:
            if log[-1].time <= 10:
                seconds = 1.0
            elif log[-1].time <= 120:
                seconds = 5.0
            else:
                seconds = 30.0
        return log[-1].time - log[last - 1].time >= seconds

    if not add_to_existing_cuts and model.most_recent_training_results is not None:
        warnings.warn(
            "Re-training a model with existing cuts! If you meant to refine a previously trained "
            "policy, pass `add_to_existing_cuts=True`.",
            stacklevel=2,
        )
    if forward_pass_resampling_probability is not None:
        forward_pass = RiskAdjustedForwardPass(
            forward_pass=forward_pass,
            risk_measure=risk_measure,
            resampling_probability=forward_pass_resampling_probability,
        )
    log_file_handle = open(log_file, "a") if log_file else None
    log: list[Log] = []
    if print_level > 0:
        _print_helper(print_banner, log_file_handle)
        _print_helper(
            print_problem_statistics,
            log_file_handle,
            model,
            model.most_recent_training_results is not None,
            parallel_scheme,
            risk_measure,
            sampling_scheme,
        )
        _print_helper(print_iteration_header, log_file_handle)
    rules: list[StoppingRule] = list(stopping_rules)
    if iteration_limit is not None:
        rules.append(IterationLimit(iteration_limit))
    if time_limit is not None:
        rules.append(TimeLimit(time_limit))
    if not rules:
        rules.append(SimulationStoppingRule())
    if cut_deletion_minimum < 0:
        cut_deletion_minimum = sys.maxsize
    for node in model.nodes.values():
        node.bellman_function.cut_type = cut_type
        node.bellman_function.global_theta.deletion_minimum = cut_deletion_minimum
        for oracle in node.bellman_function.local_thetas:
            oracle.deletion_minimum = cut_deletion_minimum
    options = Options.create(
        model,
        model.initial_root_state,
        sampling_scheme=sampling_scheme,
        backward_sampling_scheme=backward_sampling_scheme,
        risk_measures=risk_measure,
        cycle_discretization_delta=cycle_discretization_delta,
        refine_at_similar_nodes=refine_at_similar_nodes,
        stopping_rules=rules,
        print_level=print_level,
        start_time=time.time(),
        log=log,
        log_file_handle=log_file_handle,
        log_frequency=log_frequency_f,
        forward_pass=forward_pass,
        duality_handler=duality_handler,
        forward_pass_callback=forward_pass_callback,
        post_iteration_callback=post_iteration_callback,
        root_node_risk_measure=root_node_risk_measure,
        rng=rng,
    )
    status = "not_solved"
    try:
        status = parallel_scheme.master_loop(model, options)
    except KeyboardInterrupt:
        status = "interrupted"
        parallel_scheme.interrupt()
    finally:
        options.dashboard_callback(None, True)
    training_results = TrainingResults(status, log)
    model.most_recent_training_results = training_results
    if print_level > 0 and log:
        log_iteration(options, force_if_needed=True)
        _print_helper(print_footer, log_file_handle, training_results)
    if log_file_handle is not None:
        log_file_handle.close()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _simulate_one(
    model: PolicyGraph,
    variables: Sequence[str],
    *,
    sampling_scheme: SamplingScheme,
    custom_recorders: dict[str, Callable[[Any], Any]],
    duality_handler: DualityHandler | None,
    skip_undefined_variables: bool,
    incoming_state: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    scenario_path, _ = sampling_scheme.sample_scenario(model, rng)
    simulation: list[dict[str, Any]] = []
    current_belief = initialize_belief(model)
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
            current_belief = belief.updater(
                belief.belief, current_belief, belief.partition_index, noise
            )
        else:
            current_belief = {node_index: 1.0}
        subproblem_results = solve_subproblem(
            model,
            node,
            incoming_state,
            noise,
            scenario_path[:depth],
            duality_handler=duality_handler,
        )
        cumulative_value += subproblem_results.stage_objective
        store: dict[str, Any] = {
            "node_index": node_index,
            "noise_term": noise,
            "stage_objective": subproblem_results.stage_objective,
            "bellman_term": subproblem_results.objective - subproblem_results.stage_objective,
            "objective_state": objective_state_vector,
            "belief": dict(current_belief),
        }
        if objective_state_vector is not None and N == 1:
            store["objective_state"] = objective_state_vector[0]
        sp = node.subproblem
        for variable in variables:
            if variable in sp:
                store[variable] = sp.value(sp[variable])
            elif skip_undefined_variables:
                store[variable] = math.nan
            else:
                raise KeyError(
                    f"No variable named {variable} exists in the subproblem. If you want to "
                    "simulate the value of a variable, make sure it is defined in _all_ "
                    "subproblems, or pass "
                    "`skip_undefined_variables=True` to `simulate`."
                )
        for sym, recorder in custom_recorders.items():
            store[sym] = recorder(sp)
        node.lock.release()
        simulation.append(store)
        incoming_state = dict(subproblem_results.state)
    return simulation


def simulate(
    model: PolicyGraph,
    number_replications: int = 1,
    variables: Sequence[str] = (),
    *,
    sampling_scheme: SamplingScheme | None = None,
    custom_recorders: dict[str, Callable[[Any], Any]] | None = None,
    duality_handler: DualityHandler | None = None,
    skip_undefined_variables: bool = False,
    parallel_scheme: ParallelScheme | None = None,
    incoming_state: dict[str, float] | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> list[list[dict[str, Any]]]:
    """Simulate the policy (``SDDP.simulate``).

    Returns one list per replication; each entry is a list of per-node dicts with the
    keys ``node_index``, ``noise_term``, ``stage_objective``, ``bellman_term``,
    ``objective_state``, ``belief``, plus one key per requested variable (state
    variables give a :class:`StateValue`).
    """
    if rng is None:
        rng = random.Random(seed)
    if parallel_scheme is None:
        parallel_scheme = Serial()
    return parallel_scheme.simulate(
        model,
        number_replications,
        list(variables),
        sampling_scheme=sampling_scheme or InSampleMonteCarlo(),
        custom_recorders=custom_recorders or {},
        duality_handler=duality_handler,
        skip_undefined_variables=skip_undefined_variables,
        incoming_state=dict(incoming_state)
        if incoming_state is not None
        else dict(model.initial_root_state),
        rng=rng,
    )


def confidence_interval(x: Sequence[float], z_score: float = 1.96) -> tuple[float, float]:
    """Return ``(mean, half-width)`` of a confidence interval for the mean of ``x``."""
    import statistics

    mu = statistics.mean(x)
    if len(x) < 2:
        return mu, math.nan
    return mu, z_score * statistics.stdev(x) / math.sqrt(len(x))


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
class DecisionRule:
    """A decision rule for ``node`` in ``model``; query it with :func:`evaluate`."""

    def __init__(self, model: PolicyGraph, node: Any):
        self.model = model
        self.node = model[node]

    def __repr__(self) -> str:
        return f"A decision rule for node {self.node.index}"


@dataclass
class EvaluateResult:
    stage_objective: float
    outgoing_state: dict[str, float]
    controls: dict[str, Any]


def evaluate(
    rule: DecisionRule,
    incoming_state: dict[str, float],
    noise: Any = None,
    controls_to_record: Sequence[str] = (),
) -> EvaluateResult:
    """Evaluate the decision rule at ``incoming_state`` and ``noise``."""
    x = {str(k): float(v) for k, v in incoming_state.items()}
    ret = solve_subproblem(rule.model, rule.node, x, noise, [], duality_handler=None)
    sp = rule.node.subproblem
    controls = {c: sp.value(sp[c]) for c in controls_to_record}
    return EvaluateResult(ret.stage_objective, ret.state, controls)


__all__ = [
    "DecisionRule",
    "EvaluateResult",
    "IterationResult",
    "Options",
    "Sense",
    "StateValue",
    "SubproblemResult",
    "attempt_numerical_recovery",
    "backward_pass",
    "calculate_bound",
    "confidence_interval",
    "evaluate",
    "get_outgoing_state",
    "iteration",
    "parameterize",
    "set_incoming_state",
    "set_numerical_difficulty_callback",
    "simulate",
    "solve_subproblem",
    "termination_status",
    "train",
    "write_subproblem_to_file",
]
