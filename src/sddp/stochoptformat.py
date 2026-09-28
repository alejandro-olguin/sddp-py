# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""StochOptFormat reader/writer and validation scenarios. Ported from ``src/Experimental.jl``.

Subproblems are stored in MathOptFormat (MOF); the subset written and read here covers what
SDDP subproblems contain: named variables, variable bounds/integrality as
``Variable``-in-set constraints, ``ScalarAffineFunction`` constraints, and an affine (or
quadratic, for random objective coefficients) objective.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pyoptinterface as poi

from sddp.graph import Graph
from sddp.plugins.base import SamplingScheme
from sddp.plugins.sampling_schemes import InSampleMonteCarlo
from sddp.policy_graph import Node, PolicyGraph, Subproblem
from sddp.solver.model import Model, OptimizerFactory, Sense, Variable, expr_constant, to_expression


@dataclass
class ValidationScenario:
    scenario: list[tuple[Any, Any]]


class ValidationScenarios(SamplingScheme):
    """A sampling scheme that cycles through fixed validation scenarios."""

    def __init__(self, scenarios: Sequence[ValidationScenario], sha256: str = ""):
        self.scenarios = list(scenarios)
        self.last = 0
        self.sha256 = sha256

    def sample_scenario(
        self, model: PolicyGraph, rng: random.Random
    ) -> tuple[list[tuple[Any, Any]], bool]:
        self.last += 1
        if self.last > len(self.scenarios):
            self.last = 1
        return list(self.scenarios[self.last - 1].scenario), False


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _throw_if_unsupported(model: PolicyGraph) -> None:
    if model.belief_partition:
        raise ValueError("StochOptFormat does not support belief states.")
    for node in model.nodes.values():
        if node.objective_state is not None:
            raise ValueError("StochOptFormat does not support objective states.")
    for node in model.nodes.values():
        if node.bellman_function.global_theta.cuts:
            raise ValueError(
                "StochOptFormat does not support writing after a call to `SDDP.train`."
            )


def _var_name(m: Model, v: Variable) -> str:
    name = m.variable_name(v)
    if not name or name.startswith("__"):  # internal variables (θ, inner-approximation δ, ...)
        return f"_v{v.index}"
    return name


def _saf_dict(m: Model, expr: Any) -> dict[str, Any]:
    f = to_expression(expr)
    terms = [
        {"coefficient": float(c), "variable": _var_name(m, m._var_by_index(int(i)))}
        for i, c in zip(f.variables, f.coefficients)
    ]
    return {"type": "ScalarAffineFunction", "terms": terms, "constant": expr_constant(f)}


def _model_to_mof(m: Model, objective: dict[str, Any]) -> dict[str, Any]:
    variables = [{"name": _var_name(m, v)} for v in m.variables()]
    constraints: list[dict[str, Any]] = []
    for v in m.variables():
        fn: dict[str, Any] = {"type": "Variable", "name": _var_name(m, v)}
        if m.is_fixed(v):
            constraints.append(
                {"function": fn, "set": {"type": "EqualTo", "value": m.fix_value(v)}}
            )
        else:
            if m.has_lower_bound(v):
                constraints.append(
                    {"function": fn, "set": {"type": "GreaterThan", "lower": m.lower_bound(v)}}
                )
            if m.has_upper_bound(v):
                constraints.append(
                    {"function": fn, "set": {"type": "LessThan", "upper": m.upper_bound(v)}}
                )
        if m.is_binary(v):
            constraints.append({"function": fn, "set": {"type": "ZeroOne"}})
        elif m.is_integer(v):
            constraints.append({"function": fn, "set": {"type": "Integer"}})
    for c in m.constraints():
        terms, sense, rhs = m.constraint_data(c)
        fn = {
            "type": "ScalarAffineFunction",
            "terms": [{"coefficient": coef, "variable": _var_name(m, v)} for v, coef in terms],
            "constant": 0.0,
        }
        st = {
            "==": {"type": "EqualTo", "value": rhs},
            "<=": {"type": "LessThan", "upper": rhs},
            ">=": {"type": "GreaterThan", "lower": rhs},
        }[sense]
        entry: dict[str, Any] = {"function": fn, "set": st}
        if m.constraint_name(c):
            entry["name"] = m.constraint_name(c)
        constraints.append(entry)
    return {
        "name": "MathOptFormat Model",
        "version": {"major": 1, "minor": 7},
        "variables": variables,
        "objective": objective,
        "constraints": constraints,
    }


def _reformulate_uncertainty(
    node: Node, realizations: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str], Callable[[], None]]:
    """Detect what ``parameterize`` changes across the noise terms and express it with random
    variables (fixed variables, bound changes, RHS changes, objective constant/coefficients).

    Returns ``(objective_dict, random_variable_names, undo)``.
    """
    from sddp.algorithm import set_objective

    m = node.model
    bound_storage: list[dict[int, tuple[float, float, float]]] = []
    objective_storage: list[Any] = []
    rhs_storage: list[dict[int, float]] = []
    changing_lower: set[int] = set()
    changing_upper: set[int] = set()
    changing_fixed: set[int] = set()
    changing_obj_constant = False
    changing_obj_coef: set[int] = set()
    changing_rhs: set[int] = set()
    var_by_index = {v.index: v for v in m.variables()}
    con_by_index = {c.index: c for c in m.constraints()}
    for noise in node.noise_terms:
        node.parameterize_fn(noise.term)
        bounds: dict[int, tuple[float, float, float]] = {}
        for v in m.variables():
            l = m.lower_bound(v) if m.has_lower_bound(v) else -math.inf
            u = m.upper_bound(v) if m.has_upper_bound(v) else math.inf
            f = m.fix_value(v) if m.is_fixed(v) else 0.0
            if m.is_fixed(v):
                l, u = -math.inf, math.inf
            if bound_storage:
                if bound_storage[0][v.index][0] != l:
                    changing_lower.add(v.index)
                if bound_storage[0][v.index][1] != u:
                    changing_upper.add(v.index)
                if bound_storage[0][v.index][2] != f:
                    changing_fixed.add(v.index)
            bounds[v.index] = (l, u, f)
        bound_storage.append(bounds)
        f_obj = to_expression(node.stage_objective)
        obj_terms: dict[int, float] = {}
        for i, c in zip(f_obj.variables, f_obj.coefficients):
            obj_terms[int(i)] = obj_terms.get(int(i), 0.0) + float(c)
        objective_storage.append((expr_constant(f_obj), obj_terms))
        if len(objective_storage) > 1:
            if objective_storage[-1][0] != objective_storage[0][0]:
                changing_obj_constant = True
            a, b = objective_storage[0][1], objective_storage[-1][1]
            for k in set(a) | set(b):
                if a.get(k) != b.get(k):
                    changing_obj_coef.add(k)
        rhs: dict[int, float] = {}
        for c in m.constraints():
            rhs[c.index] = m.get_normalized_rhs(c)
            if rhs_storage and rhs_storage[0][c.index] != rhs[c.index]:
                changing_rhs.add(c.index)
        rhs_storage.append(rhs)
    added_variables: list[Variable] = []
    added_constraints: list[Any] = []
    random_variables: list[str] = []

    def new_random_variable(name: str) -> Variable:
        y = m.add_variable(name)
        added_variables.append(y)
        random_variables.append(name)
        return y

    # Objective: affine part plus quadratic terms (random coefficient × variable).
    obj_const, obj_terms = objective_storage[0]
    affine_terms = {k: v for k, v in obj_terms.items()}
    quad_terms: list[dict[str, Any]] = []
    if changing_obj_constant:
        name = "_SDDPjl_random_objective_constant_"
        y = new_random_variable(name)
        for r, (c, _) in zip(realizations, objective_storage):
            r["support"][name] = c
        obj_const = 0.0
        affine_terms[y.index] = 1.0
    for k in sorted(changing_obj_coef):
        xname = _var_name(m, var_by_index[k])
        name = f"_SDDPjl_random_objective_{xname}_"
        y = new_random_variable(name)
        for r, (_, terms) in zip(realizations, objective_storage):
            r["support"][name] = terms.get(k, 0.0)
        affine_terms.pop(k, None)
        quad_terms.append({"coefficient": 1.0, "variable_1": name, "variable_2": xname})
    aff = {
        "type": "ScalarAffineFunction",
        "terms": [
            {
                "coefficient": c,
                "variable": _var_name(
                    m,
                    var_by_index[k]
                    if k in var_by_index
                    else next(v for v in added_variables if v.index == k),
                ),
            }
            for k, c in affine_terms.items()
        ],
        "constant": obj_const,
    }
    if quad_terms:
        objective_fn: dict[str, Any] = {
            "type": "ScalarQuadraticFunction",
            "affine_terms": aff["terms"],
            "quadratic_terms": quad_terms,
            "constant": obj_const,
        }
    else:
        objective_fn = aff
    objective = {
        "sense": "min" if m.objective_sense is Sense.MIN else "max",
        "function": objective_fn,
    }
    # Fixed variables: recorded as random variables, nothing else to do (they are unfixed).
    saved_bounds: list[tuple[Variable, float, float]] = []
    for k in sorted(changing_fixed):
        x = var_by_index[k]
        for r, b in zip(realizations, bound_storage):
            r["support"][_var_name(m, x)] = b[k][2]
        random_variables.append(_var_name(m, x))
        saved_bounds.append((x, m.lower_bound(x), m.upper_bound(x)))
        m.unfix(x)
    for k in sorted(changing_lower):
        x = var_by_index[k]
        name = f"_SDDPjl_lower_bound_{_var_name(m, x)}_"
        y = new_random_variable(name)
        added_constraints.append(m.add_constraint_normalized(1.0 * x - 1.0 * y, ">=", 0.0))
        saved_bounds.append((x, m.lower_bound(x), m.upper_bound(x)))
        m.delete_lower_bound(x)
        for r, b in zip(realizations, bound_storage):
            r["support"][name] = b[k][0]
    for k in sorted(changing_upper):
        x = var_by_index[k]
        name = f"_SDDPjl_upper_bound_{_var_name(m, x)}_"
        y = new_random_variable(name)
        added_constraints.append(m.add_constraint_normalized(1.0 * x - 1.0 * y, "<=", 0.0))
        saved_bounds.append((x, m.lower_bound(x), m.upper_bound(x)))
        m.delete_upper_bound(x)
        for r, b in zip(realizations, bound_storage):
            r["support"][name] = b[k][1]
    saved_rhs: list[tuple[Any, float, Variable]] = []
    for k in sorted(changing_rhs):
        ci = con_by_index[k]
        name = f"_SDDPjl_rhs_{m.constraint_name(ci) or k}_"
        y = new_random_variable(name)
        m.set_normalized_coefficient(ci, y, -1.0)
        saved_rhs.append((ci, m.get_normalized_rhs(ci), y))
        m.set_normalized_rhs(ci, 0.0)
        for r, rr in zip(realizations, rhs_storage):
            r["support"][name] = rr[k]

    def undo() -> None:
        for ci, rhs_value, y in saved_rhs:
            m.set_normalized_coefficient(ci, y, 0.0)
            m.set_normalized_rhs(ci, rhs_value)
        for c in added_constraints:
            m.delete_constraint(c)
        for x, l, u in reversed(saved_bounds):
            m.set_bounds(x, l, u)
        for v in added_variables:
            m.delete_variable(v)
        node.stage_objective_set = False
        set_objective(node)

    return objective, random_variables, undo


def to_dict(
    model: PolicyGraph,
    validation_scenarios: int | ValidationScenarios | None = None,
    sampling_scheme: SamplingScheme | None = None,
    rng: random.Random | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Serialise ``model`` to a StochOptFormat dictionary."""
    _throw_if_unsupported(model)
    nodes: dict[str, Any] = {}
    subproblems: dict[str, Any] = {}
    scenario_map: dict[Any, dict[Any, Any]] = {}
    for node_name, node in model.nodes.items():
        s_name = str(node_name)
        realizations: list[dict[str, Any]] = [
            {"probability": noise.probability, "support": {}} for noise in node.noise_terms
        ]
        objective, random_variables, undo = _reformulate_uncertainty(node, realizations)
        try:
            nodes[s_name] = {"subproblem": s_name}
            if realizations:
                nodes[s_name]["realizations"] = realizations
            if node.children:
                nodes[s_name]["successors"] = {str(c.term): c.probability for c in node.children}
            m = node.model
            subproblems[s_name] = {
                "state_variables": {
                    name: {"in": _var_name(m, s.in_), "out": _var_name(m, s.out)}
                    for name, s in node.states.items()
                },
                "subproblem": _model_to_mof(m, objective),
            }
            if random_variables:
                subproblems[s_name]["random_variables"] = random_variables
        finally:
            undo()
        scenario_map[node_name] = {
            i: realizations[i]["support"] for i in range(len(node.noise_terms))
        }
        node.ext["_sof_noise_index"] = {id(n.term): i for i, n in enumerate(node.noise_terms)}
    sof: dict[str, Any] = {
        "version": {"major": 1, "minor": 0},
        "root": {
            "name": str(model.root_node),
            "state_variables": {k: v for k, v in model.initial_root_state.items()},
            "successors": {str(c.term): c.probability for c in model.root_children},
        },
        "nodes": nodes,
        "subproblems": subproblems,
    }
    if validation_scenarios is not None:
        rng = rng if rng is not None else random.Random()
        scheme = sampling_scheme if sampling_scheme is not None else InSampleMonteCarlo()
        if isinstance(validation_scenarios, int):
            scenarios = [
                ValidationScenario(scheme.sample_scenario(model, rng)[0])
                for _ in range(validation_scenarios)
            ]
        else:
            scenarios = validation_scenarios.scenarios
        out = []
        for sc in scenarios:
            items = []
            for node_index, w in sc.scenario:
                node = model[node_index]
                idx = _noise_index(node, w)
                items.append({"node": str(node_index), "support": scenario_map[node_index][idx]})
            out.append(items)
        sof["validation_scenarios"] = out
    for k, v in kwargs.items():
        sof[k] = v
    return sof


def _noise_index(node: Node, w: Any) -> int:
    for i, n in enumerate(node.noise_terms):
        if n.term is w or n.term == w:
            return i
    raise ValueError(f"Noise {w!r} is not a noise term of node {node.index}")


def write_to_file(model: PolicyGraph, filename: str, **kwargs: Any) -> None:
    """Write ``model`` to ``filename`` in StochOptFormat (``.gz`` compresses)."""
    data = to_dict(model, **kwargs)
    text = json.dumps(data)
    if filename.endswith(".gz"):
        with gzip.open(filename, "wt") as io:
            io.write(text)
    else:
        with open(filename, "w") as io:
            io.write(text)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _load_mof(sp: Subproblem, mof: dict[str, Any]) -> tuple[dict[str, Variable], Any]:
    m = sp.model
    variables: dict[str, Variable] = {}
    for v in mof["variables"]:
        variables[v["name"]] = m.add_variable(v["name"])
    for c in mof["constraints"]:
        fn, st = c["function"], c["set"]
        if fn["type"] == "Variable":
            x = variables[fn["name"]]
            t = st["type"]
            if t == "GreaterThan":
                m.set_lower_bound(x, float(st["lower"]))
            elif t == "LessThan":
                m.set_upper_bound(x, float(st["upper"]))
            elif t == "EqualTo":
                m.fix(x, float(st["value"]))
            elif t == "Interval":
                m.set_bounds(x, float(st["lower"]), float(st["upper"]))
            elif t == "Integer":
                m.set_integer(x)
            elif t == "ZeroOne":
                m.set_binary(x)
            else:
                raise ValueError(f"Unsupported variable set {t}")
        elif fn["type"] == "ScalarAffineFunction":
            expr = Model.expression(
                [(variables[t["variable"]], float(t["coefficient"])) for t in fn["terms"]],
                float(fn.get("constant", 0.0)),
            )
            t = st["type"]
            if t == "GreaterThan":
                m.add_constraint_normalized(expr, ">=", float(st["lower"]), name=c.get("name"))
            elif t == "LessThan":
                m.add_constraint_normalized(expr, "<=", float(st["upper"]), name=c.get("name"))
            elif t == "EqualTo":
                m.add_constraint_normalized(expr, "==", float(st["value"]), name=c.get("name"))
            elif t == "Interval":
                m.add_constraint_normalized(expr, ">=", float(st["lower"]))
                m.add_constraint_normalized(expr, "<=", float(st["upper"]))
            else:
                raise ValueError(f"Unsupported constraint set {t}")
        else:
            raise ValueError(f"Unsupported constraint function {fn['type']}")
    return variables, mof["objective"]


def _read_model(
    data: dict[str, Any], bound: float, optimizer: OptimizerFactory | None
) -> PolicyGraph:
    graph: Graph = Graph("__root__")
    for from_node in data["nodes"]:
        graph.add_node(from_node)
    for to_node, probability in data["root"].get("successors", {}).items():
        graph.add_edge("__root__", to_node, float(probability))
    for from_node, node in data["nodes"].items():
        for to_node, probability in node.get("successors", {}).items():
            graph.add_edge(from_node, to_node, float(probability))
    n_min = sum(
        1 for sp in data["subproblems"].values() if sp["subproblem"]["objective"]["sense"] == "min"
    )
    model_sense = Sense.MIN if n_min / len(data["subproblems"]) >= 0.5 else Sense.MAX

    def builder(sp: Subproblem, node_name: str) -> None:
        sub_name = data["nodes"][node_name]["subproblem"]
        sub = data["subproblems"][sub_name]
        variables, objective = _load_mof(sp, sub["subproblem"])
        for s, state in sub["state_variables"].items():
            sp.register_state(s, variables[state["in"]], variables[state["out"]])
        omega: list[Any] = []
        prob: list[float] = []
        for realization in data["nodes"][node_name].get("realizations", []):
            prob.append(float(realization["probability"]))
            omega.append(realization.get("support", {}))
        sp_sense = Sense.MIN if objective["sense"] == "min" else Sense.MAX
        if sp_sense is not model_sense:
            import warnings

            warnings.warn(
                f"Flipping the objective sense of node {node_name} so that it matches the "
                "majority of the subproblems.",
                stacklevel=2,
            )
        obj_sgn = 1.0 if sp_sense is model_sense else -1.0
        rvs = set(sub.get("random_variables", []))
        fn = objective["function"]
        aff_terms: dict[str, float] = {}
        constant = float(fn.get("constant", 0.0))
        random_coefficients: dict[str, tuple[str, float]] = {}  # random var -> (variable, coef)
        if fn["type"] == "ScalarAffineFunction":
            for t in fn["terms"]:
                aff_terms[t["variable"]] = aff_terms.get(t["variable"], 0.0) + float(
                    t["coefficient"]
                )
        elif fn["type"] == "ScalarQuadraticFunction":
            for t in fn["affine_terms"]:
                aff_terms[t["variable"]] = aff_terms.get(t["variable"], 0.0) + float(
                    t["coefficient"]
                )
            for t in fn["quadratic_terms"]:
                a, b, coef = t["variable_1"], t["variable_2"], float(t["coefficient"])
                if a in rvs:
                    random_coefficients[a] = (b, coef)
                elif b in rvs:
                    random_coefficients[b] = (a, coef)
                else:
                    raise ValueError(
                        "Quadratic objective terms without a random variable are not supported."
                    )
        else:
            raise ValueError(f"Unsupported objective function {fn['type']}")
        base_aff = {k: aff_terms.get(k, 0.0) for k in aff_terms}

        def modify(w: dict[str, float] | None) -> None:
            terms = dict(base_aff)
            for k, v in (w or {}).items():
                if k in random_coefficients:
                    x, coef = random_coefficients[k]
                    terms[x] = base_aff.get(x, 0.0) + v * coef
                sp.fix(variables[k], float(v))
            expr = Model.expression([(variables[k], c) for k, c in terms.items()], constant)
            sp.set_stage_objective(obj_sgn * expr)

        sp.parameterize(modify, omega, prob)

    model: PolicyGraph
    if model_sense is Sense.MIN:
        model = PolicyGraph(
            builder, graph, sense="Min", lower_bound=-abs(bound), optimizer=optimizer
        )
    else:
        model = PolicyGraph(
            builder, graph, sense="Max", upper_bound=abs(bound), optimizer=optimizer
        )
    for k, v in data["root"]["state_variables"].items():
        model.initial_root_state[k] = float(v)
    return model


def _validation_from_data(data: dict[str, Any], sha256: str) -> ValidationScenarios | None:
    if "validation_scenarios" not in data:
        return None
    scenarios = []
    for scenario in data["validation_scenarios"]:
        items = []
        for item in scenario:
            support = item.get("support", {})
            items.append((item["node"], None if not support else support))
        scenarios.append(ValidationScenario(items))
    return ValidationScenarios(scenarios, sha256=sha256)


def read_from_file(
    filename: str, bound: float = 1e6, optimizer: OptimizerFactory | None = None
) -> tuple[PolicyGraph, ValidationScenarios | None]:
    """Read a StochOptFormat file; returns ``(model, validation_scenarios)``."""
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rb") as io:
        raw = io.read()
    data = json.loads(raw)
    model = _read_model(data, bound, optimizer)
    return model, _validation_from_data(data, hashlib.sha256(raw).hexdigest())


def evaluate(model: PolicyGraph, validation_scenarios: ValidationScenarios) -> dict[str, Any]:
    """Simulate the trained policy on the validation scenarios (``SDDP.evaluate``)."""
    from sddp.algorithm import simulate

    validation_scenarios.last = 0

    def primal(sp: Subproblem) -> dict[str, float]:
        m = sp.model
        return {m.variable_name(x): m.value(x) for x in m.variables() if m.variable_name(x)}

    sims = simulate(
        model,
        len(validation_scenarios.scenarios),
        sampling_scheme=validation_scenarios,
        custom_recorders={"primal": primal},
    )
    return {
        "problem_sha256_checksum": validation_scenarios.sha256,
        "scenarios": [
            [{"objective": s["stage_objective"], "primal": s["primal"]} for s in sim]
            for sim in sims
        ],
    }


__all__ = [
    "ValidationScenario",
    "ValidationScenarios",
    "evaluate",
    "read_from_file",
    "to_dict",
    "write_to_file",
    "poi",
]
