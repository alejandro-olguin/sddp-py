# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Deterministic equivalent (extensive form). Ported from ``src/deterministic_equivalent.jl``."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from sddp.algorithm import parameterize
from sddp.graph import is_cyclic
from sddp.policy_graph import Node, PolicyGraph, State
from sddp.solver.model import Model, OptimizerFactory, Variable, expr_constant, to_expression


class DeterministicEquivalentError(RuntimeError):
    pass


def _err(msg: str) -> None:
    raise DeterministicEquivalentError("Unable to formulate deterministic equivalent: " + msg)


@dataclass
class ScenarioTreeNode:
    node: Node
    noise: Any
    probability: float
    children: list[ScenarioTreeNode] = field(default_factory=list)
    states: dict[str, State] = field(default_factory=dict)


def _add_node_to_scenario_tree(
    parent: list[ScenarioTreeNode], pg: PolicyGraph, node: Node, probability: float, check_time_limit: Any
) -> None:
    if node.objective_state is not None:
        _err("Objective states detected!")
    elif node.belief_state is not None:
        _err("Belief states detected!")
    elif len(node.bellman_function.global_theta.cuts) > 0:
        _err("Model has been used for training. Can only form deterministic equivalent on a fresh model.")
    else:
        check_time_limit()
    for noise in node.noise_terms:
        scenario_node = ScenarioTreeNode(node, noise.term, probability * noise.probability)
        for child in node.children:
            _add_node_to_scenario_tree(
                scenario_node.children, pg, pg[child.term],
                probability * noise.probability * child.probability, check_time_limit,
            )
        parent.append(scenario_node)


def _copy_expression(src: Any, var_map: dict[int, Variable]) -> Any:
    f = to_expression(src)
    terms = [(var_map[int(i)], float(c)) for i, c in zip(f.variables, f.coefficients)]
    return Model.expression(terms, expr_constant(f))


def _add_scenario_to_ef(model: Model, child: ScenarioTreeNode, check_time_limit: Any) -> None:
    check_time_limit()
    node = child.node
    parameterize(node, child.noise)
    src = node.model
    var_map: dict[int, Variable] = {}
    for v in src.variables():
        name = src.variable_name(v)
        dest = model.add_variable(
            f"{name or '_[' + str(v.index) + ']'}#{node.index}#{id(child)}",
            lb=src.lower_bound(v),
            ub=src.upper_bound(v),
            integer=src.is_integer(v),
            binary=src.is_binary(v),
        )
        var_map[v.index] = dest
    for c in src.constraints():
        terms, sense, rhs = src.constraint_data(c)
        expr = Model.expression([(var_map[v.index], coef) for v, coef in terms])
        model.add_constraint_normalized(expr, sense, rhs)
    current = model.objective_function()
    sub_obj = _copy_expression(node.stage_objective, var_map)
    model.set_objective(current + child.probability * sub_obj)
    for key, state in node.states.items():
        child.states[key] = State(var_map[state.in_.index], var_map[state.out.index])
    for child_2 in child.children:
        _add_scenario_to_ef(model, child_2, check_time_limit)


def _add_linking_constraints(model: Model, node: ScenarioTreeNode, check_time_limit: Any) -> None:
    check_time_limit()
    for child in node.children:
        for key in node.states:
            model.add_constraint_normalized(
                Model.expression([(node.states[key].out, 1.0), (child.states[key].in_, -1.0)]), "==", 0.0
            )
        _add_linking_constraints(model, child, check_time_limit)


def deterministic_equivalent(
    pg: PolicyGraph, optimizer: OptimizerFactory | None = None, time_limit: float | None = 60.0
) -> Model:
    """Form the extensive-form model of ``pg``. Solve it with ``.optimize()``."""
    start_time = time.time()
    limit = math.inf if time_limit is None else float(time_limit)

    def check_time_limit() -> None:
        if time.time() - start_time > limit:
            _err("Time limit exceeded!")

    if is_cyclic(pg):
        _err("Cyclic policy graph detected!")
    tree: list[ScenarioTreeNode] = []
    for child in pg.root_children:
        _add_node_to_scenario_tree(tree, pg, pg[child.term], child.probability, check_time_limit)
    model = Model(optimizer if optimizer is not None else pg.optimizer)
    model.set_objective_sense(pg.objective_sense)
    model.set_objective(0.0)
    for child in tree:
        _add_scenario_to_ef(model, child, check_time_limit)
    for child in tree:
        _add_linking_constraints(model, child, check_time_limit)
        for key, value in pg.initial_root_state.items():
            model.fix(child.states[key].in_, value)
    return model
