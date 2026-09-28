# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Bellman functions, cuts, and cut selection. Ported from ``src/plugins/bellman_functions.jl``."""

from __future__ import annotations

import enum
import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sddp.solver.model import Constraint, Model, Sense, Variable

if TYPE_CHECKING:
    from sddp.plugins.base import RiskMeasure
    from sddp.policy_graph import Node, PolicyGraph


@dataclass
class Cut:
    intercept: float
    coefficients: dict[str, float]
    obj_y: tuple[float, ...] | None
    belief_y: dict[Any, float] | None
    non_dominated_count: int
    constraint_ref: Constraint | None


@dataclass
class SampledState:
    state: dict[str, float]
    obj_y: tuple[float, ...] | None
    belief_y: dict[Any, float] | None
    dominating_cut: Cut
    best_objective: float


class ConvexApproximation:
    """A variable ``theta`` plus the cuts that bound it (``ConvexApproximation`` in SDDP.jl)."""

    def __init__(
        self,
        theta: Variable,
        states: dict[str, Variable],
        objective_states: tuple[Variable, ...] | None,
        belief_states: dict[Any, Variable] | None,
        deletion_minimum: int,
        model: Model,
    ):
        self.theta = theta
        self.states = states
        self.objective_states = objective_states
        self.belief_states = belief_states
        self.cuts: list[Cut] = []
        self.sampled_states: list[SampledState] = []
        self.cuts_to_be_deleted: list[Cut] = []
        self.deletion_minimum = deletion_minimum
        self.model = model


def _magnitude(x: float) -> float:
    return math.log10(abs(x)) if abs(x) > 0 else 0.0


_warned_dynamic_range = False


def _dynamic_range_warning(intercept: float, coefficients: dict[str, float]) -> None:
    global _warned_dynamic_range
    if _warned_dynamic_range:
        return
    lo = hi = _magnitude(intercept)
    lo_v = hi_v = intercept
    for v in coefficients.values():
        i = _magnitude(v)
        if v < lo_v:
            lo, lo_v = i, v
        elif v > hi_v:
            hi, hi_v = i, v
    if hi - lo > 10:
        _warned_dynamic_range = True
        warnings.warn(
            "Found a cut with a mix of small and large coefficients. The order of magnitude "
            f"difference is {hi - lo}. The smallest coefficient is {lo_v}. The largest "
            f"coefficient is {hi_v}. Consider rescaling your model.",
            stacklevel=3,
        )


def _add_cut(
    V: ConvexApproximation,
    theta_k: float,
    pi_k: dict[str, float],
    x_k: dict[str, float],
    obj_y: tuple[float, ...] | None,
    belief_y: dict[Any, float] | None,
    cut_selection: bool = True,
) -> None:
    for key, x in x_k.items():
        theta_k -= pi_k[key] * x
    _dynamic_range_warning(theta_k, pi_k)
    cut = Cut(theta_k, pi_k, obj_y, belief_y, 1, None)
    _add_cut_constraint_to_model(V, cut)
    if cut_selection:
        _cut_selection_update(V, cut, x_k)


def _add_cut_constraint_to_model(V: ConvexApproximation, cut: Cut) -> None:
    model = V.model
    terms: list[tuple[Variable, float]] = [(V.theta, 1.0)]
    if V.objective_states is not None and cut.obj_y is not None:
        for y, mu in zip(cut.obj_y, V.objective_states):
            terms.append((mu, y))
    if V.belief_states is not None and cut.belief_y is not None:
        for k, mu in V.belief_states.items():
            terms.append((mu, cut.belief_y[k]))
    for name, x in V.states.items():
        terms.append((x, -cut.coefficients[name]))
    expr = Model.expression(terms)
    if model.objective_sense is Sense.MIN:
        cut.constraint_ref = model.add_constraint_normalized(expr, ">=", cut.intercept)
    else:
        cut.constraint_ref = model.add_constraint_normalized(expr, "<=", cut.intercept)


def _eval_height(cut: Cut, sampled_state: SampledState) -> float:
    height = cut.intercept
    for key, value in cut.coefficients.items():
        height += value * sampled_state.state[key]
    return height


def _dominates(candidate: float, incumbent: float, minimization: bool) -> bool:
    return candidate >= incumbent if minimization else candidate <= incumbent


def _cut_selection_update(V: ConvexApproximation, cut: Cut, state: dict[str, float]) -> None:
    model = V.model
    is_minimization = model.objective_sense is Sense.MIN
    sampled_state = SampledState(state, cut.obj_y, cut.belief_y, cut, math.nan)
    sampled_state.best_objective = _eval_height(cut, sampled_state)
    for old_state in V.sampled_states:
        if old_state.obj_y != cut.obj_y or old_state.belief_y != cut.belief_y:
            continue
        height = _eval_height(cut, old_state)
        if _dominates(height, old_state.best_objective, is_minimization):
            old_state.dominating_cut.non_dominated_count -= 1
            cut.non_dominated_count += 1
            old_state.dominating_cut = cut
            old_state.best_objective = height
    V.sampled_states.append(sampled_state)
    for old_cut in V.cuts:
        if old_cut.constraint_ref is not None:
            continue
        elif old_cut.obj_y != sampled_state.obj_y:
            continue
        elif old_cut.belief_y != sampled_state.belief_y:
            continue
        height = _eval_height(old_cut, sampled_state)
        if _dominates(height, sampled_state.best_objective, is_minimization):
            sampled_state.dominating_cut.non_dominated_count -= 1
            old_cut.non_dominated_count += 1
            sampled_state.dominating_cut = old_cut
            sampled_state.best_objective = height
            _add_cut_constraint_to_model(V, old_cut)
    V.cuts.append(cut)
    for c in V.cuts:
        if c.non_dominated_count < 1 and c.constraint_ref is not None:
            V.cuts_to_be_deleted.append(c)
    if len(V.cuts_to_be_deleted) >= V.deletion_minimum:
        for c in V.cuts_to_be_deleted:
            model.delete_constraint(c.constraint_ref)  # type: ignore[arg-type]
            c.constraint_ref = None
            c.non_dominated_count = 0
    V.cuts_to_be_deleted.clear()


class CutType(enum.Enum):
    SINGLE_CUT = "SINGLE_CUT"
    MULTI_CUT = "MULTI_CUT"


SINGLE_CUT = CutType.SINGLE_CUT
MULTI_CUT = CutType.MULTI_CUT


@dataclass
class BellmanFunctionFactory:
    """Cached keyword arguments for :class:`BellmanFunction` (``InstanceFactory`` in SDDP.jl)."""

    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    deletion_minimum: int = 1
    cut_type: CutType = CutType.MULTI_CUT


def BellmanFunction(
    lower_bound: float = -math.inf,
    upper_bound: float = math.inf,
    deletion_minimum: int = 1,
    cut_type: CutType = CutType.MULTI_CUT,
) -> BellmanFunctionFactory:
    return BellmanFunctionFactory(lower_bound, upper_bound, deletion_minimum, cut_type)


@dataclass
class BellmanFunctionInstance:
    """``V(x, b, y) = min μᵀb + νᵀy + θ`` subject to single, multi, and risk-set cuts."""

    cut_type: CutType
    global_theta: ConvexApproximation
    local_thetas: list[ConvexApproximation] = field(default_factory=list)
    risk_set_cuts: set[tuple[float, ...]] = field(default_factory=set)


def bellman_term(bf: BellmanFunctionInstance) -> Variable:
    return bf.global_theta.theta


def initialize_bellman_function(
    factory: BellmanFunctionFactory, model: PolicyGraph, node: Node
) -> BellmanFunctionInstance:
    lower_bound, upper_bound = factory.lower_bound, factory.upper_bound
    deletion_minimum, cut_type = factory.deletion_minimum, factory.cut_type
    if lower_bound == -math.inf and upper_bound == math.inf:
        raise ValueError("You must specify a finite bound on the cost-to-go term.")
    if len(node.children) == 0:
        lower_bound = upper_bound = 0.0
    m = node.model
    theta = m.add_variable("__theta__")
    if lower_bound > -math.inf:
        m.set_lower_bound(theta, lower_bound)
    if upper_bound < math.inf:
        m.set_upper_bound(theta, upper_bound)
    _add_initial_bounds(node.objective_state, theta, m)
    x_out = {key: s.out for key, s in node.states.items()}
    obj_mu = node.objective_state.mu if node.objective_state is not None else None
    belief_mu = node.belief_state.mu if node.belief_state is not None else None
    return BellmanFunctionInstance(
        cut_type,
        ConvexApproximation(theta, x_out, obj_mu, belief_mu, deletion_minimum, m),
        [],
        set(),
    )


def _add_objective_state_constraint(
    theta: Variable, y: tuple[float, ...], mu: tuple[Variable, ...], m: Model
) -> None:
    N = len(y)
    is_finite = [-math.inf < y[i] < math.inf for i in range(N)]
    lower_bound = m.lower_bound(theta) if m.has_lower_bound(theta) else -math.inf
    upper_bound = m.upper_bound(theta) if m.has_upper_bound(theta) else math.inf
    if math.isclose(lower_bound, 0.0, abs_tol=0.0) and math.isclose(upper_bound, 0.0, abs_tol=0.0):
        for i in range(N):
            m.add_constraint_normalized(1.0 * mu[i], "==", 0.0)
        return
    terms = [(mu[i], y[i]) for i in range(N) if is_finite[i]] + [(theta, 1.0)]
    expr = Model.expression(terms)
    if lower_bound > -math.inf:
        m.add_constraint_normalized(expr, ">=", lower_bound)
    if upper_bound < math.inf:
        m.add_constraint_normalized(expr, "<=", upper_bound)


def _add_initial_bounds(obj_state: Any, theta: Variable, m: Model) -> None:
    if obj_state is None:
        return
    import itertools

    if len(obj_state.mu) < 5:
        for y in itertools.product(*zip(obj_state.lower_bound, obj_state.upper_bound)):
            _add_objective_state_constraint(theta, tuple(y), obj_state.mu, m)
    else:
        _add_objective_state_constraint(theta, obj_state.lower_bound, obj_state.mu, m)
        _add_objective_state_constraint(theta, obj_state.upper_bound, obj_state.mu, m)


def refine_bellman_function(
    model: PolicyGraph,
    node: Node,
    bellman_function: BellmanFunctionInstance,
    risk_measure: RiskMeasure,
    outgoing_state: dict[str, float],
    dual_variables: Sequence[dict[str, float]],
    noise_supports: Sequence[Any],
    nominal_probability: Sequence[float],
    objective_realizations: Sequence[float],
) -> Any:
    assert (
        len(dual_variables) == len(noise_supports) == len(nominal_probability) == len(objective_realizations)
    )
    risk_adjusted_probability = [0.0] * len(nominal_probability)
    offset = risk_measure.adjust_probability(
        risk_adjusted_probability,
        list(nominal_probability),
        list(noise_supports),
        list(objective_realizations),
        model.is_minimization,
    )
    if bellman_function.cut_type is CutType.SINGLE_CUT:
        return _add_average_cut(
            node, outgoing_state, risk_adjusted_probability, objective_realizations, dual_variables, offset
        )
    assert bellman_function.cut_type is CutType.MULTI_CUT
    _add_locals_if_necessary(node, bellman_function, len(dual_variables))
    return _add_multi_cut(
        node, outgoing_state, risk_adjusted_probability, objective_realizations, dual_variables, offset
    )


def _copy_value(state: Any) -> Any:
    if state is None:
        return None
    if hasattr(state, "belief"):
        return dict(state.belief)
    return state.state


def _add_average_cut(
    node: Node,
    outgoing_state: dict[str, float],
    risk_adjusted_probability: Sequence[float],
    objective_realizations: Sequence[float],
    dual_variables: Sequence[dict[str, float]],
    offset: float,
) -> dict[str, Any]:
    N = len(risk_adjusted_probability)
    assert N == len(objective_realizations) == len(dual_variables)
    pi_k = {key: 0.0 for key in outgoing_state}
    theta_k = offset
    for i in range(N):
        p = risk_adjusted_probability[i]
        theta_k += p * objective_realizations[i]
        for key, dual in dual_variables[i].items():
            pi_k[key] += p * dual
    obj_y = _copy_value(node.objective_state)
    belief_y = _copy_value(node.belief_state)
    _add_cut(node.bellman_function.global_theta, theta_k, pi_k, outgoing_state, obj_y, belief_y)
    return {"theta": theta_k, "pi": pi_k, "x": outgoing_state, "obj_y": obj_y, "belief_y": belief_y}


def _add_multi_cut(
    node: Node,
    outgoing_state: dict[str, float],
    risk_adjusted_probability: Sequence[float],
    objective_realizations: Sequence[float],
    dual_variables: Sequence[dict[str, float]],
    offset: float,
) -> None:
    from sddp.policy_graph import get_belief_state_component, get_objective_state_component

    N = len(risk_adjusted_probability)
    assert N == len(objective_realizations) == len(dual_variables)
    bf = node.bellman_function
    mu_y = get_objective_state_component(node) + get_belief_state_component(node)
    for i in range(N):
        _add_cut(
            bf.local_thetas[i],
            objective_realizations[i],
            dual_variables[i],
            outgoing_state,
            _copy_value(node.objective_state),
            _copy_value(node.belief_state),
        )
    m = node.model
    # theta >= sum(p_i * theta_i) - (1 - sum(p)) * μᵀy + offset
    terms: list[tuple[Variable, float]] = [(bf.global_theta.theta, 1.0)]
    for i in range(N):
        terms.append((bf.local_thetas[i].theta, -risk_adjusted_probability[i]))
    one_minus = 1.0 - sum(risk_adjusted_probability)
    for mu, y in mu_y:
        terms.append((mu, one_minus * y))
    xi = tuple(risk_adjusted_probability)
    if xi not in bf.risk_set_cuts or mu_y:
        bf.risk_set_cuts.add(xi)
        expr = Model.expression(terms)
        if m.objective_sense is Sense.MIN:
            m.add_constraint_normalized(expr, ">=", offset)
        else:
            m.add_constraint_normalized(expr, "<=", offset)
    return None


def _add_locals_if_necessary(node: Node, bf: BellmanFunctionInstance, N: int) -> None:
    num_local = len(bf.local_thetas)
    if num_local == N:
        return
    if num_local > 0:
        raise ValueError(f"Expected {N} local θ variables but there were {num_local}.")
    g = bf.global_theta
    m = node.model
    lb = m.lower_bound(g.theta) if m.has_lower_bound(g.theta) else -math.inf
    ub = m.upper_bound(g.theta) if m.has_upper_bound(g.theta) else math.inf
    for i in range(N):
        local = m.add_variable(f"__theta_{i}__", lb=lb, ub=ub)
        bf.local_thetas.append(
            ConvexApproximation(
                local,
                g.states,
                node.objective_state.mu if node.objective_state is not None else None,
                node.belief_state.mu if node.belief_state is not None else None,
                g.deletion_minimum,
                m,
            )
        )


# ---------------------------------------------------------------------------
# Cut serialisation (compatible with SDDP.jl's write_cuts_to_file JSON)
# ---------------------------------------------------------------------------
def cuts_to_list(
    model: PolicyGraph, node_name_parser: Any = str, write_only_selected_cuts: bool = False
) -> list[dict[str, Any]]:
    cuts: list[dict[str, Any]] = []
    for node_name, node in model.nodes.items():
        if node.objective_state is not None or node.belief_state is not None:
            raise ValueError(
                "Unable to write cuts to file because model contains objective states or belief states."
            )
        node_cuts: dict[str, Any] = {
            "node": node_name_parser(node_name),
            "single_cuts": [],
            "multi_cuts": [],
            "risk_set_cuts": [],
        }
        oracle = node.bellman_function.global_theta
        for cut, state in zip(oracle.cuts, oracle.sampled_states):
            if write_only_selected_cuts and cut.constraint_ref is None:
                continue
            intercept = cut.intercept
            for key, pi in cut.coefficients.items():
                intercept += pi * state.state[key]
            node_cuts["single_cuts"].append(
                {"intercept": intercept, "coefficients": dict(cut.coefficients), "state": dict(state.state)}
            )
        for i, theta in enumerate(node.bellman_function.local_thetas, start=1):
            for cut, state in zip(theta.cuts, theta.sampled_states):
                if write_only_selected_cuts and cut.constraint_ref is None:
                    continue
                intercept = cut.intercept
                for key, pi in cut.coefficients.items():
                    intercept += pi * state.state[key]
                node_cuts["multi_cuts"].append(
                    {
                        "realization": i,
                        "intercept": intercept,
                        "coefficients": dict(cut.coefficients),
                        "state": dict(state.state),
                    }
                )
        for p in node.bellman_function.risk_set_cuts:
            node_cuts["risk_set_cuts"].append(list(p))
        cuts.append(node_cuts)
    return cuts


def write_cuts_to_file(
    model: PolicyGraph, filename: str, node_name_parser: Any = str, write_only_selected_cuts: bool = False
) -> None:
    """Write the cuts to ``filename`` in SDDP.jl's JSON format."""
    with open(filename, "w") as io:
        json.dump(cuts_to_list(model, node_name_parser, write_only_selected_cuts), io)


def _default_node_name_parser(model: PolicyGraph, name: str) -> Any:
    sample = next(iter(model.nodes))
    if isinstance(sample, int):
        return int(name)
    if isinstance(sample, str):
        return name
    if isinstance(sample, tuple):
        keys = [int(s.strip()) for s in name.strip("()").split(",") if s.strip()]
        if len(keys) != len(sample):
            raise ValueError(f"Unable to parse node called {name}. Expected {len(sample)} elements.")
        return tuple(keys)
    raise ValueError(
        f"Unable to read name {name}. Provide a custom parser to `read_cuts_from_file` using "
        "the `node_name_parser` keyword."
    )


def read_cuts_from_file(
    model: PolicyGraph, filename: str, node_name_parser: Any = None, cut_selection: bool = True
) -> None:
    """Read cuts written by :func:`write_cuts_to_file` (or SDDP.jl) into ``model``."""
    with open(filename) as io:
        cuts = json.load(io)
    for node_cuts in cuts:
        if node_name_parser is None:
            node_name = _default_node_name_parser(model, node_cuts["node"])
        else:
            node_name = node_name_parser(node_cuts["node"])
        if node_name is None:
            continue
        node = model[node_name]
        bf = node.bellman_function
        for json_cut in node_cuts["single_cuts"]:
            has_state = "state" in json_cut
            state = (
                {k: float(v) for k, v in json_cut["state"].items()}
                if has_state
                else {k: 0.0 for k in json_cut["coefficients"]}
            )
            _add_cut(
                bf.global_theta,
                float(json_cut["intercept"]),
                {k: float(v) for k, v in json_cut["coefficients"].items()},
                state,
                None,
                None,
                cut_selection=(cut_selection and has_state),
            )
        if node_cuts["risk_set_cuts"]:
            _add_locals_if_necessary(node, bf, len(node_cuts["risk_set_cuts"][0]))
        for json_cut in node_cuts["multi_cuts"]:
            has_state = "state" in json_cut
            state = (
                {k: float(v) for k, v in json_cut["state"].items()}
                if has_state
                else {k: 0.0 for k in json_cut["coefficients"]}
            )
            _add_cut(
                bf.local_thetas[json_cut["realization"] - 1],
                float(json_cut["intercept"]),
                {k: float(v) for k, v in json_cut["coefficients"].items()},
                state,
                None,
                None,
                cut_selection=(cut_selection and has_state),
            )
        for json_cut in node_cuts["risk_set_cuts"]:
            terms = [(bf.global_theta.theta, 1.0)] + [
                (V.theta, -float(p)) for p, V in zip(json_cut, bf.local_thetas)
            ]
            expr = Model.expression(terms)
            if node.model.objective_sense is Sense.MIN:
                node.model.add_constraint_normalized(expr, ">=", 0.0)
            else:
                node.model.add_constraint_normalized(expr, "<=", 0.0)


def add_all_cuts(model: PolicyGraph) -> None:
    """Add back any cuts that cut selection removed from the subproblems."""
    for node in model.nodes.values():
        g = node.bellman_function.global_theta
        for cut in g.cuts:
            if cut.constraint_ref is None:
                _add_cut_constraint_to_model(g, cut)
        for approx in node.bellman_function.local_thetas:
            for cut in approx.cuts:
                if cut.constraint_ref is None:
                    _add_cut_constraint_to_model(approx, cut)
