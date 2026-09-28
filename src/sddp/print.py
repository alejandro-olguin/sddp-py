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


# ---------------------------------------------------------------------------
# Numerical stability report (``numerical_stability_report`` in print.jl)
# ---------------------------------------------------------------------------
class CoefficientRanges:
    def __init__(self) -> None:
        self.matrix = [math.inf, -math.inf]
        self.objective = [math.inf, -math.inf]
        self.bounds = [math.inf, -math.inf]
        self.rhs = [math.inf, -math.inf]

    def merge(self, other: CoefficientRanges) -> None:
        for a, b in (
            (self.matrix, other.matrix),
            (self.objective, other.objective),
            (self.bounds, other.bounds),
            (self.rhs, other.rhs),
        ):
            a[0] = min(a[0], b[0])
            a[1] = max(a[1], b[1])


def _update_range(rng: list[float], value: float) -> None:
    if value != 0.0 and not math.isnan(value) and not math.isinf(value):
        rng[0] = min(rng[0], abs(value))
        rng[1] = max(rng[1], abs(value))


def _print_value(x: float) -> str:
    return f"{x:1.0e}"


def _stringify_bounds(bounds: list[float]) -> str:
    lower = _print_value(bounds[0]) if bounds[0] < math.inf else "0e+00"
    upper = _print_value(bounds[1]) if bounds[1] > -math.inf else "0e+00"
    return f"[{lower}, {upper}]"


def _coefficient_ranges(model: Any) -> CoefficientRanges:
    from sddp.solver.model import expr_constant, to_expression

    ranges = CoefficientRanges()
    f = to_expression(model.objective_function())
    for c in f.coefficients:
        _update_range(ranges.objective, float(c))
    _ = expr_constant(f)
    for v in model.variables():
        if model.is_fixed(v):
            continue  # JuMP: a fixed variable has no lower/upper bound
        if model.has_lower_bound(v):
            _update_range(ranges.bounds, model.lower_bound(v))
        if model.has_upper_bound(v):
            _update_range(ranges.bounds, model.upper_bound(v))
    for c in model.constraints():
        terms, _, rhs = model.constraint_data(c)
        for _, coef in terms:
            _update_range(ranges.matrix, coef)
        _update_range(ranges.rhs, rhs)
    return ranges


def _print_numerical_stability_report(
    io: TextIO, ranges: CoefficientRanges, print_: bool, warn: bool
) -> None:
    warnings_: list[tuple[str, str]] = []
    for name, rng in (
        ("matrix", ranges.matrix),
        ("objective", ranges.objective),
        ("bounds", ranges.bounds),
        ("rhs", ranges.rhs),
    ):
        if print_:
            io.write("  " + f"{name} range".ljust(17) + _stringify_bounds(rng) + "\n")
        if rng[0] < 1e-4:
            warnings_.append((name, "small"))
        if rng[1] > 1e7:
            warnings_.append((name, "large"))
    if warn and warnings_:
        io.write("WARNING: numerical stability issues detected\n")
        for name, sense in warnings_:
            io.write(f"  - {name} range contains {sense} coefficients\n")
        io.write(
            "Very large or small absolute values of coefficients\n"
            "can cause numerical stability issues. Consider\n"
            "reformulating the model.\n"
        )


def numerical_stability_report(
    model: PolicyGraph,
    io: TextIO | None = None,
    by_node: bool = False,
    print: bool = True,
    warn: bool = True,
) -> str:
    """Report the coefficient ranges of every subproblem (``SDDP.numerical_stability_report``).

    Returns the report text and writes it to ``io`` (default: stdout) if ``print``.
    """
    import io as _io
    import sys

    from sddp.algorithm import parameterize

    buf = _io.StringIO()
    graph_ranges = CoefficientRanges()
    node_keys = list(model.nodes)
    try:
        node_keys = sorted(node_keys)
    except TypeError:
        pass
    for key in node_keys:
        node = model[key]
        node_ranges = CoefficientRanges()
        for noise in node.noise_terms:
            parameterize(node, noise.term)
            node_ranges.merge(_coefficient_ranges(node.model))
        if by_node:
            buf.write(f"numerical stability report for node: {key}\n")
            _print_numerical_stability_report(buf, node_ranges, True, warn)
        graph_ranges.merge(node_ranges)
    if not by_node:
        buf.write("numerical stability report\n")
        _print_numerical_stability_report(buf, graph_ranges, True, warn)
    text = buf.getvalue()
    if print:
        (io if io is not None else sys.stdout).write(text)
    return text


def write_log_to_csv(model: PolicyGraph, filename: str) -> None:
    """Write the most recent training log to ``filename`` (``SDDP.write_log_to_csv``)."""
    if model.most_recent_training_results is None:
        raise ValueError("Unable to write the log to file because the model has not been trained.")
    with open(filename, "w") as io:
        io.write("iteration, simulation, bound, time\n")
        for log in model.most_recent_training_results.log:
            io.write(f"{log.iteration}, {log.simulation_value}, {log.bound}, {log.time}\n")
