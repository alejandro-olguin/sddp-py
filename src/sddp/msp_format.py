# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Reader for MSPFormat problem/lattice files. Ported from ``src/MSPFormat.jl``."""

from __future__ import annotations

import gzip
import json
import math
import os
from typing import Any

from sddp.graph import Graph
from sddp.policy_graph import PolicyGraph, State, Subproblem
from sddp.solver.model import OptimizerFactory, to_expression


def _load_json(filename: str) -> Any:
    if filename.endswith(".gz"):
        with gzip.open(filename, "rt") as io:
            return json.load(io)
    with open(filename) as io:
        return json.load(io)


def _parse_lattice(filename: str) -> tuple[Graph, dict[str, Any]]:
    data = _load_json(filename)
    graph: Graph = Graph("root")
    max_stage = max(int(v["stage"]) for v in data.values())
    stage_node_combinations: dict[int, list[str]] = {s: [] for s in range(max_stage + 1)}
    for key, value in data.items():
        graph.add_node(key)
        stage_node_combinations[int(value["stage"])].append(key)
    for key, value in data.items():
        if int(value["stage"]) == max_stage:
            continue
        for child in sorted(stage_node_combinations[int(value["stage"]) + 1]):
            graph.add_edge(key, child, float(value["successors"].get(child, 0.0)))
    stage_zero = [key for key, value in data.items() if int(value["stage"]) == 0]
    for key in stage_zero:
        graph.add_edge("root", key, 1 / len(stage_zero))
    return _reduce_lattice(graph, data)


def _reduce_lattice(graph: Graph, data: dict[str, Any]) -> tuple[Graph, dict[str, Any]]:
    """Collapse a Markovian lattice to a stagewise-independent linear graph if possible."""
    arcs_to_node: dict[tuple, list[str]] = {}
    for node, arcs in graph.nodes.items():
        arcs_to_node.setdefault(tuple(arcs), []).append(node)
    for v in arcs_to_node.values():
        v.sort()
    nodes_by_stage: dict[int, list[str]] = {}
    for node, d in data.items():
        nodes_by_stage.setdefault(int(d["stage"]), []).append(node)
    for v in nodes_by_stage.values():
        v.sort()
    groups = list(arcs_to_node.values())
    if not all(n in groups for n in nodes_by_stage.values()):
        graph_data = {
            k: {"stage": int(v["stage"]), "sample_space": [v["state"]], "probability": [1.0]}
            for k, v in data.items()
        }
        return graph, graph_data
    new_graph: Graph = Graph("root")
    for t in nodes_by_stage:
        new_graph.add_node(str(t))
    for t in nodes_by_stage:
        if t == 0:
            new_graph.add_edge("root", "0", 1.0)
        else:
            new_graph.add_edge(str(t - 1), str(t), 1.0)
    sample_space: dict[str, list[Any]] = {str(t): [] for t in nodes_by_stage}
    probability: dict[str, list[float]] = {str(t): [] for t in nodes_by_stage}
    parent = "root"
    while graph.nodes[parent]:
        for node, prob in sorted(graph.nodes[parent]):
            key = str(int(data[node]["stage"]))
            sample_space[key].append(data[node]["state"])
            probability[key].append(prob)
            parent = node
    graph_data = {
        str(t): {
            "stage": t,
            "sample_space": sample_space[str(t)],
            "probability": probability[str(t)],
        }
        for t in nodes_by_stage
    }
    return new_graph, graph_data


def _get_constant(terms: Any, state: dict[str, Any] | None = None) -> Any:
    """Evaluate an MSPFormat constant expression; returns the term list itself if it depends
    on random data and no ``state`` is given."""
    if isinstance(terms, str):
        return state.get(terms, 0.0) if state is not None else None
    if isinstance(terms, (int, float)) and not isinstance(terms, bool):
        return terms
    if len(terms) == 1:
        t0 = terms[0]
        if isinstance(t0, (int, float)):
            return t0
        elif t0 == "inf":
            return math.inf
        elif t0 == "-inf":
            return -math.inf
        elif isinstance(t0, str):
            value = _get_constant(t0, state)
            return terms if value is None else value
    result = None
    for term in terms:
        assert isinstance(term, dict)
        if "ADD" in term:
            value = _get_constant(term["ADD"], state)
            if value is None:
                return terms
            result = (0.0 if result is None else result) + value
        else:
            assert "MUL" in term
            value = _get_constant(term["MUL"], state)
            if value is None:
                return terms
            result = (1.0 if result is None else result) * value
    return result


def _set_type(rhs: Any, type_: str) -> tuple[str, float]:
    value = float(rhs) if isinstance(rhs, (int, float)) else 0.0
    return {"EQ": "==", "LEQ": "<=", "GEQ": ">="}[type_], value


def _build_lhs(
    stage: int, sp: Subproblem, terms: list[dict[str, Any]]
) -> tuple[Any, dict[Any, Any] | None]:
    if max(int(t["stage"]) for t in terms) != stage:
        return None, None
    lhs = to_expression(0.0)
    lhs_data: dict[Any, Any] = {}
    for term in terms:
        x = sp[term["name"]]
        if isinstance(x, State):
            if int(term["stage"]) == stage:
                x = x.out
            elif int(term["stage"]) == stage - 1:
                x = x.in_
            else:
                raise ValueError(
                    "SDDP.jl does not support this MSPFormat file because it contains state "
                    "variables from stages other than `t` or `t-1`. Got "
                    f"`t-{stage - int(term['stage'])}`"
                )
        else:
            assert int(term["stage"]) == stage
        coef = _get_constant(term["coefficient"])
        if isinstance(coef, list):
            lhs_data[x] = coef
            lhs = lhs + 1.0 * x
        else:
            lhs = lhs + float(coef) * x
    return lhs, lhs_data


def _state_variables(problem: dict[str, Any]) -> list[str]:
    states: set[str] = set()
    for constraint in problem["constraints"]:
        terms = constraint["lhs"]
        stage = max(int(t["stage"]) for t in terms)
        for term in terms:
            if int(term["stage"]) != stage:
                states.add(term["name"])
    return sorted(states)


def read_from_file(
    problem_filename: str,
    lattice_filename: str | None = None,
    bound: float = 1e6,
    optimizer: OptimizerFactory | None = None,
) -> PolicyGraph:
    """Build a :class:`PolicyGraph` from MSPFormat files.

    ``read_from_file("name")`` reads ``name.problem.json`` and ``name.lattice.json`` (or
    ``.lattice.json.gz``). ``bound`` is the absolute value of the cost-to-go bound.
    """
    if lattice_filename is None:
        lattice_filename = problem_filename + ".lattice.json"
        if not os.path.isfile(lattice_filename):
            lattice_filename += ".gz"
        problem_filename = problem_filename + ".problem.json"
    graph, graph_data = _parse_lattice(lattice_filename)
    problem = _load_json(problem_filename)
    state_variables = _state_variables(problem)
    initial_values: dict[str, float] = {}

    def builder(sp: Subproblem, node: str) -> None:
        w_lower: dict[Any, Any] = {}
        w_upper: dict[Any, Any] = {}
        w_objective: dict[Any, Any] = {}
        w_lhs: dict[Any, Any] = {}
        w_rhs: dict[Any, Any] = {}
        stage = int(graph_data[node]["stage"])
        stage_objective = to_expression(0.0)
        for variable in problem["variables"]:
            if int(variable["stage"]) != stage:
                continue
            lower = _get_constant(variable["lb"])
            upper = _get_constant(variable["ub"])
            objective = _get_constant(variable["obj"])
            name = variable["name"]
            initial_values.setdefault(name, 0.0)
            if isinstance(lower, (int, float)) and math.isfinite(lower):
                initial_values[name] = max(initial_values[name], lower)
            if isinstance(upper, (int, float)) and math.isfinite(upper):
                initial_values[name] = min(initial_values[name], upper)
            if name in state_variables:
                x = sp.add_state(name, initial_value=initial_values[name]).out
            else:
                x = sp.add_variable(name)
            if variable["type"] == "BINARY":
                sp.model.set_binary(x)
            elif variable["type"] == "INTEGER":
                sp.model.set_integer(x)
            else:
                assert variable["type"] == "CONTINUOUS"
            if isinstance(lower, (int, float)) and math.isfinite(lower):
                sp.set_lower_bound(x, lower)
            elif isinstance(lower, list):
                w_lower[x] = lower
            if isinstance(upper, (int, float)) and math.isfinite(upper):
                sp.set_upper_bound(x, upper)
            elif isinstance(upper, list):
                w_upper[x] = upper
            if isinstance(objective, (int, float)):
                stage_objective = stage_objective + float(objective) * x
            elif isinstance(objective, list):
                w_objective[x] = objective
        for name in state_variables:
            if name not in sp:
                sp.add_state(name, initial_value=0.0)
        for constraint in problem["constraints"]:
            lhs, lhs_data = _build_lhs(stage, sp, constraint["lhs"])
            if lhs is None:
                continue
            rhs = _get_constant(constraint["rhs"])
            sense, value = _set_type(rhs, constraint["type"])
            con = sp.model.add_constraint_normalized(lhs, sense, value)
            if isinstance(rhs, list):
                w_rhs[con] = rhs
            if lhs_data:
                w_lhs[con] = lhs_data
        omega = graph_data[node]["sample_space"]
        probability = graph_data[node]["probability"]

        def modify(w: dict[str, Any]) -> None:
            obj = stage_objective
            for x, terms in w_objective.items():
                obj = obj + float(_get_constant(terms, w)) * x
            sp.set_stage_objective(obj)
            for x, terms in w_lower.items():
                sp.set_lower_bound(x, float(_get_constant(terms, w)))
            for x, terms in w_upper.items():
                sp.set_upper_bound(x, float(_get_constant(terms, w)))
            for con, lhs_data in w_lhs.items():
                for x, terms in lhs_data.items():
                    sp.set_normalized_coefficient(con, x, float(_get_constant(terms, w)))
            for con, terms in w_rhs.items():
                sp.set_normalized_rhs(con, float(_get_constant(terms, w)))

        sp.parameterize(modify, omega, probability)

    maximize = bool(problem["maximize"])
    return PolicyGraph(
        builder,
        graph,
        sense="Max" if maximize else "Min",
        lower_bound=-math.inf if maximize else -bound,
        upper_bound=bound if maximize else math.inf,
        optimizer=optimizer,
    )
