# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Inner (vertex-based) approximations of the value function: deterministic upper bounds.

Ported from ``src/Inner.jl``. Only linear graphs, single-cut vertices and resource states
are supported, as in SDDP.jl.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sddp.plugins.base import RiskMeasure
from sddp.plugins.bellman_functions import CutType, SampledState, _add_initial_bounds
from sddp.plugins.risk_measures import Expectation
from sddp.policy_graph import Node, PolicyGraph, Subproblem, _get_incoming_domain
from sddp.solver.model import Constraint, Model, OptimizerFactory, Sense, Variable


@dataclass
class Vertex:
    value: float
    state: dict[str, float]
    obj_y: tuple[float, ...] | None
    belief_y: dict[Any, float] | None
    non_dominated_count: int
    variable_ref: Variable | None


class InnerConvexApproximation:
    def __init__(
        self,
        theta: Variable,
        states: dict[str, Variable],
        objective_states: Any,
        belief_states: Any,
        deletion_minimum: int,
        lipschitz_constant: float,
        model: Model,
        cc: dict[str, Any],
    ):
        self.theta = theta
        self.states = states
        self.objective_states = objective_states
        self.belief_states = belief_states
        self.vertices: list[Vertex] = []
        self.sampled_states: list[SampledState] = []
        self.vertices_to_be_deleted: list[Vertex] = []
        self.deletion_minimum = deletion_minimum
        self.lipschitz_constant = lipschitz_constant
        self.model = model
        self.cc = cc  # {"sigma_cc": Constraint, "x_cc": {name: Constraint}, "theta_cc": Constraint}


def _add_vertex(
    V: InnerConvexApproximation,
    theta_k: float,
    x_k: dict[str, float],
    obj_y: tuple[float, ...] | None,
    belief_y: dict[Any, float] | None,
) -> None:
    vertex = Vertex(theta_k, dict(x_k), obj_y, belief_y, 1, None)
    _add_vertex_var_to_model(V, vertex)
    V.vertices.append(vertex)


def _add_vertex_var_to_model(V: InnerConvexApproximation, vertex: Vertex) -> None:
    if V.objective_states is not None:
        raise NotImplementedError("Objective states not yet implemented.")
    m = V.model
    sigma_k = m.add_variable(lb=0.0, ub=1.0)
    m.set_normalized_coefficient(V.cc["sigma_cc"], sigma_k, 1.0)
    for key, xk in vertex.state.items():
        m.set_normalized_coefficient(V.cc["x_cc"][key], sigma_k, xk)
    m.set_normalized_coefficient(V.cc["theta_cc"], sigma_k, vertex.value)
    vertex.variable_ref = sigma_k


def _vertex_selection(
    V: InnerConvexApproximation, optimizer: OptimizerFactory, tol: float = 1e-6
) -> Model:
    """Exact vertex selection: drop vertices covered by convex combinations of the others."""
    n = len(V.vertices)
    keys = list(V.states)
    m = Model(optimizer)
    sigma = [m.add_variable(lb=0.0, ub=1.0) for _ in range(n)]
    t = m.add_variable()
    m.set_objective(1.0 * t, Sense.MAX)
    cc_x: dict[str, Constraint] = {
        k: m.add_constraint_normalized(
            Model.expression([(sigma[i], V.vertices[i].state[k]) for i in range(n)]), "==", 0.0
        )
        for k in keys
    }
    cc_t = m.add_constraint_normalized(
        Model.expression([(sigma[i], V.vertices[i].value) for i in range(n)] + [(t, 1.0)]),
        "==",
        0.0,
    )
    m.add_constraint_normalized(Model.expression([(sigma[i], 1.0) for i in range(n)]), "==", 1.0)
    remove: list[Vertex] = []
    for j in range(n):
        v = V.vertices[j]
        for k in keys:
            m.set_normalized_rhs(cc_x[k], v.state[k])
        m.set_normalized_rhs(cc_t, v.value)
        m.optimize()
        margin = m.objective_value()
        if margin > tol:
            remove.append(v)
            m.set_bounds(sigma[j], 0.0, 0.0)  # JuMP deletes σ_j; fixing it to 0 is equivalent
    for v in remove:
        if v.variable_ref is not None:
            V.model.delete_variable(v.variable_ref)
            v.variable_ref = None
    if remove:
        print(f"Selection removed {len(remove)} vertices")
    return m


@dataclass
class InnerBellmanFunctionInstance:
    cut_type: CutType
    global_theta: InnerConvexApproximation
    local_thetas: list[Any] = field(default_factory=list)
    risk_set_cuts: set[tuple[float, ...]] = field(default_factory=set)
    lipschitz_constant: float = 0.0

    def refine(
        self,
        model: PolicyGraph,
        node: Node,
        risk_measure: RiskMeasure,
        outgoing_state: dict[str, float],
        dual_variables: Sequence[dict[str, float]],
        noise_supports: Sequence[Any],
        nominal_probability: Sequence[float],
        objective_realizations: Sequence[float],
    ) -> Any:
        assert (
            len(dual_variables)
            == len(noise_supports)
            == len(nominal_probability)
            == len(objective_realizations)
        )
        q = [0.0] * len(nominal_probability)
        offset = risk_measure.adjust_probability(
            q,
            list(nominal_probability),
            list(noise_supports),
            list(objective_realizations),
            model.is_minimization,
        )
        if self.cut_type is not CutType.SINGLE_CUT:
            raise NotImplementedError("Multi-vertices not yet implemented.")
        theta_k = offset + sum(p * v for p, v in zip(q, objective_realizations))
        obj_y = None if node.objective_state is None else node.objective_state.state
        belief_y = None if node.belief_state is None else dict(node.belief_state.belief)
        _add_vertex(self.global_theta, theta_k, outgoing_state, obj_y, belief_y)
        return {"theta": theta_k, "x": outgoing_state, "obj_y": obj_y, "belief_y": belief_y}


def _to_node(node: Node, element: Any) -> Any:
    if isinstance(element, dict):
        return element[node.index]
    if callable(element):
        return element(node.index)
    return element


@dataclass
class InnerBellmanFunctionFactory:
    lipschitz_constant: Any
    lower_bound: Any = -math.inf
    upper_bound: Any = math.inf
    deletion_minimum: int = 1
    vertex_type: CutType = CutType.MULTI_CUT

    def initialize(self, model: PolicyGraph, node: Node) -> InnerBellmanFunctionInstance:
        lower_bound = float(_to_node(node, self.lower_bound))
        upper_bound = float(_to_node(node, self.upper_bound))
        lipschitz = float(_to_node(node, self.lipschitz_constant))
        deletion_minimum = int(_to_node(node, self.deletion_minimum))
        vertex_type = _to_node(node, self.vertex_type)
        if len(node.children) == 0:
            lower_bound = upper_bound = 0.0
        m = node.model
        theta = m.add_variable("__theta__")
        _add_initial_bounds(node.objective_state, theta, m)
        x_out = {key: var.out for key, var in node.states.items()}
        obj_mu = node.objective_state.mu if node.objective_state is not None else None
        belief_mu = node.belief_state.mu if node.belief_state is not None else None
        cc: dict[str, Any] = {}
        if node.children:
            delta = {k: m.add_variable(f"__delta_{k}__") for k in x_out}
            delta_abs = {k: m.add_variable(f"__delta_abs_{k}__", lb=0.0) for k in x_out}
            for k in x_out:
                m.add_constraint_normalized(1.0 * delta_abs[k] - 1.0 * delta[k], ">=", 0.0)
                m.add_constraint_normalized(1.0 * delta_abs[k] + 1.0 * delta[k], ">=", 0.0)
            sigma0 = m.add_variable("__sigma0__", lb=0.0, ub=1.0)
            abs_terms = [(delta_abs[k], lipschitz) for k in x_out]
            if m.objective_sense is Sense.MIN:
                cc["theta_cc"] = m.add_constraint_normalized(
                    Model.expression(abs_terms + [(sigma0, upper_bound), (theta, -1.0)]), "<=", 0.0
                )
            else:
                cc["theta_cc"] = m.add_constraint_normalized(
                    Model.expression(abs_terms + [(sigma0, lower_bound), (theta, -1.0)]), ">=", 0.0
                )
            cc["x_cc"] = {
                k: m.add_constraint_normalized(1.0 * delta[k] - 1.0 * x_out[k], "==", 0.0)
                for k in x_out
            }
            cc["sigma_cc"] = m.add_constraint_normalized(1.0 * sigma0, "==", 1.0)
            m.names["vertex_coverage_distance"] = Model.expression(
                [(delta_abs[k], 1.0) for k in x_out]
            )
        else:
            m.fix(theta, 0.0)
            m.names["vertex_coverage_distance"] = Model.expression([], 0.0)
        return InnerBellmanFunctionInstance(
            vertex_type,
            InnerConvexApproximation(
                theta, x_out, obj_mu, belief_mu, deletion_minimum, lipschitz, m, cc
            ),
            [],
            set(),
            lipschitz,
        )


def InnerBellmanFunction(
    lipschitz_constant: Any,
    lower_bound: Any = -math.inf,
    upper_bound: Any = math.inf,
    deletion_minimum: int = 1,
    vertex_type: CutType = CutType.MULTI_CUT,
) -> InnerBellmanFunctionFactory:
    """Factory for inner Bellman functions; scalar, dict or ``f(node)`` arguments per node."""
    return InnerBellmanFunctionFactory(
        lipschitz_constant, lower_bound, upper_bound, deletion_minimum, vertex_type
    )


def _validate_linear_graph(graph: Any) -> None:
    has_leaf = False
    for node, children in graph.nodes.items():
        if len(children) == 0:
            has_leaf = True
        elif len(children) != 1:
            raise ValueError(
                f"The graph is not linear since node {node} has {len(children)} children "
                "and should have 1."
            )
    if not has_leaf:
        raise ValueError(
            f"The graph is not linear since no leaf node was found among {len(graph.nodes)} nodes."
        )


def InnerPolicyGraph(
    builder: Callable[[Subproblem, Any], Any],
    graph: Any,
    *,
    sense: str = "Min",
    lower_bound: float = -math.inf,
    upper_bound: float = math.inf,
    optimizer: OptimizerFactory | None = None,
    lipschitz_constant: float | None = None,
    bellman_function: Any = None,
) -> PolicyGraph:
    """A policy graph whose nodes hold inner (vertex) approximations; linear graphs only."""
    _validate_linear_graph(graph)
    s = Sense.parse(sense)
    if bellman_function is None:
        if s is Sense.MIN and lower_bound == -math.inf:
            raise ValueError(
                "You must specify a finite lower bound on the objective value using the "
                "`lower_bound = value` keyword argument."
            )
        if s is Sense.MAX and upper_bound == math.inf:
            raise ValueError(
                "You must specify a finite upper bound on the objective value using the "
                "`upper_bound = value` keyword argument."
            )
        if lipschitz_constant is None:
            raise ValueError(
                "With inner approximations, you must specify an estimate for the Lipschitz "
                "constant to be used in the InnerBellmanFunction"
            )
        if s is Sense.MIN and upper_bound == math.inf:
            raise ValueError(
                "With inner approximations, you must specify a finite upper bound on the "
                "objective value using the `upper_bound = value` keyword argument even when "
                "sense = :Min."
            )
        if s is Sense.MAX and lower_bound == -math.inf:
            raise ValueError(
                "With inner approximations, you must specify a finite lower bound on the "
                "objective value using the `lower_bound = value` keyword argument even when "
                "sense = :Max."
            )
        bellman_function = InnerBellmanFunction(
            lipschitz_constant,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            vertex_type=CutType.SINGLE_CUT,
        )
    model: PolicyGraph = PolicyGraph(
        builder,
        graph,
        sense=sense,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        optimizer=optimizer,
        bellman_function=bellman_function,
    )
    domain = _get_incoming_domain(model)
    for node_name, node in model.nodes.items():
        for k, v in domain[node_name].items():
            node.incoming_state_bounds[k] = v if v is not None else (-math.inf, math.inf, False)
    return model


def _dp_pass(
    inner_model: PolicyGraph,
    outer_model: PolicyGraph,
    opts: Any,
    optimizer: OptimizerFactory | None,
    print_level: int,
    tol: float,
) -> float:
    from sddp.algorithm import BackwardPassItems, solve_all_children
    from sddp.plugins.bellman_functions import refine_bellman_function

    total = 0.0
    keys = sorted(outer_model.nodes, reverse=True)[1:]
    for node_index in keys:
        t0 = time.time()
        node = inner_model[node_index]
        for sampled_state in outer_model[node_index].bellman_function.global_theta.sampled_states:
            outgoing_state = sampled_state.state
            items = BackwardPassItems()
            solve_all_children(
                inner_model,
                node,
                items,
                1.0,
                None,
                None,
                outgoing_state,
                opts.backward_sampling_scheme,
                [],
                opts.duality_handler,
                opts,
            )
            refine_bellman_function(
                inner_model,
                node,
                node.bellman_function,
                opts.risk_measures[node_index],
                outgoing_state,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
            )
        dt = time.time() - t0
        dt_vs = 0.0
        if optimizer is not None:
            t1 = time.time()
            _vertex_selection(node.bellman_function.global_theta, optimizer, tol)
            dt_vs = time.time() - t1
        if print_level > 0:
            print(
                f"Node: {node_index} - elapsed time: {dt:.2f} plus {dt_vs:.2f} "
                "for vertex selection."
            )
        total += dt + dt_vs
    return total


def dp_vertices_from_visited_states(
    inner_model: PolicyGraph,
    outer_model: PolicyGraph,
    optimizer: OptimizerFactory | None = None,
    risk_measures: Any = None,
    print_level: int = 0,
    vertex_selection_tol: float = 1e-6,
) -> None:
    """One backward DP pass seeding ``inner_model`` with vertices at the states visited by
    ``outer_model`` (a trained cut-based model)."""
    from sddp.algorithm import Options

    opts = Options.create(
        inner_model,
        outer_model.initial_root_state,
        risk_measures=risk_measures if risk_measures is not None else Expectation(),
    )
    _dp_pass(inner_model, outer_model, opts, optimizer, print_level, vertex_selection_tol)


def inner_dp(
    build: Callable[[Subproblem, Any], Any],
    pb: PolicyGraph,
    *,
    stages: int,
    sense: str = "Min",
    optimizer: OptimizerFactory | None,
    lower_bound: float = -math.inf,
    upper_bound: float = math.inf,
    bellman_function: InnerBellmanFunctionFactory,
    risk_measure: RiskMeasure,
    print_level: int = 1,
    vertex_selection_tol: float = 1e-6,
) -> tuple[PolicyGraph, float, float]:
    """Build an inner model with ``build`` and run one DP pass over the states sampled by the
    trained ``pb``. Returns ``(inner_model, upper_bound, elapsed)``."""
    from sddp.algorithm import Options, calculate_bound, set_objective
    from sddp.policy_graph import LinearPolicyGraph

    pb_inner = LinearPolicyGraph(
        build,
        stages=stages,
        sense=sense,
        optimizer=optimizer,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bellman_function=bellman_function,
    )
    for node in pb_inner.nodes.values():
        set_objective(node)
    opts = Options.create(pb_inner, pb.initial_root_state, risk_measures=risk_measure)
    total = _dp_pass(
        pb_inner,
        pb,
        opts,
        optimizer if optimizer is not None else pb_inner.optimizer,
        print_level,
        vertex_selection_tol,
    )
    ub = calculate_bound(pb_inner, risk_measure=risk_measure)
    if print_level > 0:
        print("First-stage upper bound: ", ub)
        print("Total time for upper bound: ", total)
    return pb_inner, ub, total


def write_vertices_to_file(model: PolicyGraph, filename: str, node_name_parser: Any = str) -> None:
    vertices = []
    for node_name, node in model.nodes.items():
        vertices.append(
            {
                "node": node_name_parser(node_name),
                "vertices": [
                    {"value": v.value, "state": dict(v.state)}
                    for v in node.bellman_function.global_theta.vertices
                ],
                "multi_vertices": [],
                "risk_set_cuts": [],
            }
        )
    with open(filename, "w") as io:
        json.dump(vertices, io)


def read_vertices_from_file(
    model: PolicyGraph,
    filename: str,
    node_name_parser: Any = None,
    vertex_selection: bool = False,
    vertex_selection_tol: float = 1e-6,
    optimizer: OptimizerFactory | None = None,
) -> None:
    from sddp.plugins.bellman_functions import _default_node_name_parser

    if vertex_selection and optimizer is None:
        raise ValueError("You must select an optimizer for performing vertex selection.")
    with open(filename) as io:
        vertices = json.load(io)
    for node_info in vertices:
        node_name = (
            _default_node_name_parser(model, node_info["node"])
            if node_name_parser is None
            else node_name_parser(node_info["node"])
        )
        if node_name is None:
            continue
        node = model[node_name]
        bf = node.bellman_function
        for jv in node_info["vertices"]:
            _add_vertex(
                bf.global_theta,
                float(jv["value"]),
                {k: float(v) for k, v in jv["state"].items()},
                None,
                None,
            )
        if vertex_selection:
            assert optimizer is not None
            _vertex_selection(bf.global_theta, optimizer, vertex_selection_tol)
