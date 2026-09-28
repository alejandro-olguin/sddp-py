# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Printing utilities. Ported from ``src/print.jl``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from sddp.algorithm import Log, TrainingResults
    from sddp.policy_graph import PolicyGraph

_RULE = "-------------------------------------------------------------------"


def print_value(x: float | int) -> str:
    if isinstance(x, int):
        return f"{x:9d}"
    return f"{x:1.6e}".rjust(13)


def print_banner(io: TextIO) -> None:
    io.write(_RULE + "\n")
    io.write("     sddp-py: a port of SDDP.jl (c) Oscar Dowson and contributors\n")
    io.write(_RULE + "\n")


def _unique_paths(model: PolicyGraph) -> float:
    from sddp.graph import is_cyclic

    if is_cyclic(model):
        return math.inf
    parents: dict[Any, set] = {t: set() for t in model.nodes}
    children: dict[Any, set] = {t: set() for t in model.nodes}
    for t, node in model.nodes.items():
        for child in node.children:
            if child.probability > 0:
                parents[child.term].add(t)
                children[t].add(child.term)
    ordered: list = []
    in_order = {t: False for t in model.nodes}
    stack: list[tuple[Any, bool]] = []
    for root_child in model.root_children:
        if root_child.probability == 0 or in_order[root_child.term]:
            continue
        stack.append((root_child.term, True))
        while stack:
            node, needs_checking = stack.pop()
            if not needs_checking:
                ordered.append(node)
                in_order[node] = True
                continue
            elif in_order[node]:
                continue
            stack.append((node, False))
            for child in children[node]:
                if not in_order[child]:
                    stack.append((child, True))
    total_scenarios = 0.0
    incoming: dict[Any, float] = {t: 0.0 for t in model.nodes}
    for node in reversed(ordered):
        N = len(model[node].noise_terms)
        if not parents[node]:
            incoming[node] = N
        else:
            incoming[node] = N * sum(incoming[p] for p in parents[node])
        if not children[node]:
            total_scenarios += incoming[node]
    return total_scenarios


def print_problem_statistics(
    io: TextIO,
    model: PolicyGraph,
    existing_cuts: bool,
    parallel_scheme: Any,
    risk_measure: Any,
    sampling_scheme: Any,
) -> None:
    n_vars = [n.subproblem.model.num_variables() for n in model.nodes.values()]
    n_cons = [n.subproblem.model.num_constraints() for n in model.nodes.values()]
    io.write("problem\n")
    io.write(f"  nodes           : {len(model.nodes)}\n")
    io.write(f"  state variables : {len(model.initial_root_state)}\n")
    io.write(f"  scenarios       : {_unique_paths(model):1.5e}\n")
    io.write(f"  existing cuts   : {existing_cuts}\n")
    io.write("options\n")
    io.write(f"  solver          : {parallel_scheme}\n")
    io.write(f"  risk measure    : {risk_measure}\n")
    io.write(f"  sampling scheme : {type(sampling_scheme).__name__}\n")
    io.write("subproblem structure\n")
    io.write(f"  variables       : [{min(n_vars)}, {max(n_vars)}]\n")
    io.write(f"  constraints     : [{min(n_cons)}, {max(n_cons)}]\n")


def print_iteration_header(io: TextIO) -> None:
    io.write(_RULE + "\n")
    io.write(" iteration    simulation      bound        time (s)     solves  pid\n")
    io.write(_RULE + "\n")


def print_iteration(io: TextIO, log: Log) -> None:
    io.write("†" if log.serious_numerical_issue else " ")
    io.write(print_value(log.iteration))
    io.write(log.duality_key)
    io.write(" " + print_value(log.simulation_value))
    io.write(" " + print_value(log.bound))
    io.write(" " + print_value(log.time))
    io.write(" " + print_value(log.total_solves))
    io.write(" " + f"{log.pid:3d}")
    io.write("\n")


def print_footer(io: TextIO, results: TrainingResults) -> None:
    io.write(_RULE + "\n")
    io.write(f"status         : {results.status}\n")
    io.write(f"total time (s) :{print_value(results.log[-1].time)}\n")
    io.write(f"total solves   : {results.log[-1].total_solves}\n")
    io.write(f"best bound     : {print_value(results.log[-1].bound)}\n")
    num_issues = sum(1 for l in results.log if l.serious_numerical_issue)
    io.write(f"numeric issues : {num_issues}\n")
    io.write(_RULE + "\n")
