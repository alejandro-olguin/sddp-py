# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Thin wrapper around a pyoptinterface model.

This module plays the role that JuMP + MathOptInterface play for SDDP.jl: a
persistent model with incremental constraint addition/deletion, bound and RHS
modification, and primal/dual queries.

Dual sign contract (see PORTING_NOTES.md §4): :meth:`Model.reduced_cost` and
:meth:`Model.dual` return *sensitivities of the objective value in the model's
own sense*, i.e. ``d objective / d (fixed value)`` and ``d objective / d rhs``,
for both minimisation and maximisation. This differs from JuMP's convention for
maximisation problems (JuMP negates), which is why SDDP.jl applies a
``dual_sign``; the Python port does not.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyoptinterface as poi

Variable = poi.VariableIndex
Constraint = poi.ConstraintIndex
Expression = poi.ScalarAffineFunction
TerminationStatus = poi.TerminationStatusCode
ResultStatus = poi.ResultStatusCode

INF = math.inf


class Sense(enum.Enum):
    """Optimisation sense (``MOI.MIN_SENSE`` / ``MOI.MAX_SENSE``)."""

    MIN = "Min"
    MAX = "Max"

    @property
    def poi(self) -> poi.ObjectiveSense:
        return poi.ObjectiveSense.Minimize if self is Sense.MIN else poi.ObjectiveSense.Maximize

    @staticmethod
    def parse(value: str | Sense) -> Sense:
        if isinstance(value, Sense):
            return value
        v = str(value).lower()
        if v in ("min", "minimize", "minimise"):
            return Sense.MIN
        if v in ("max", "maximize", "maximise"):
            return Sense.MAX
        raise ValueError(f"The optimization sense must be 'Min' or 'Max'. It is {value!r}.")


class OptimizerFactory:
    """A factory that creates a fresh pyoptinterface model.

    ``sddp.HiGHS`` is the default. Other pyoptinterface backends can be wrapped
    with :func:`pyoptinterface_optimizer`.
    """

    def __init__(self, name: str, constructor: Callable[[], Any], options: dict | None = None):
        self.name = name
        self._constructor = constructor
        self.options = dict(options or {})

    def with_options(self, **options: Any) -> OptimizerFactory:
        """Return a copy of the factory with extra raw solver options."""
        return OptimizerFactory(self.name, self._constructor, {**self.options, **options})

    def create(self) -> Any:
        m = self._constructor()
        for k, v in self.options.items():
            m.set_raw_parameter(k, v)
        return m

    def __repr__(self) -> str:
        return f"OptimizerFactory({self.name!r})"


def _highs_constructor() -> Any:
    from pyoptinterface import highs

    return highs.Model()


def pyoptinterface_optimizer(module_name: str, **options: Any) -> OptimizerFactory:
    """Wrap any pyoptinterface backend, e.g. ``pyoptinterface_optimizer("gurobi")``."""
    import importlib

    mod = importlib.import_module(f"pyoptinterface.{module_name}")
    return OptimizerFactory(module_name, mod.Model, options)


HiGHS = OptimizerFactory("HiGHS", _highs_constructor)


@dataclass
class _VarInfo:
    name: str
    lb: float
    ub: float
    integer: bool
    binary: bool


@dataclass
class _ConInfo:
    name: str
    sense: poi.ConstraintSense
    rhs: float
    variables: dict[int, Variable] = field(default_factory=dict)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def expr_constant(f: Expression) -> float:
    """The constant term of an affine expression (``None`` in pyoptinterface means 0)."""
    c = f.constant
    return 0.0 if c is None else float(c)


def to_expression(x: Any) -> Expression:
    """Convert a number, variable, or affine expression into a ScalarAffineFunction."""
    if isinstance(x, Expression):
        return x
    if isinstance(x, Variable):
        return 1.0 * x
    if _is_number(x):
        e = poi.ScalarAffineFunction()
        e.add_constant(float(x))
        return e
    if isinstance(x, poi.ExprBuilder):
        if x.degree() > 1:
            raise TypeError("Quadratic expressions are not supported in this port.")
        return poi.ScalarAffineFunction(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to an affine expression.")


class Model:
    """A persistent optimisation model with the operations SDDP needs."""

    def __init__(self, optimizer: OptimizerFactory | None = None, silent: bool = True):
        self.optimizer: OptimizerFactory = optimizer if optimizer is not None else HiGHS
        self._m = self.optimizer.create()
        if silent:
            self._m.set_model_attribute(poi.ModelAttribute.Silent, True)
        self._vars: list[Variable] = []
        self._var_info: dict[int, _VarInfo] = {}
        self._cons: list[Constraint] = []
        self._con_info: dict[int, _ConInfo] = {}
        self._deleted: set[int] = set()
        self._fixed: dict[int, float] = {}
        self.names: dict[str, Any] = {}
        self._objective: Expression | None = None
        self._sense: Sense = Sense.MIN
        self.solve_count = 0

    # ------------------------------------------------------------------ raw
    @property
    def raw(self) -> Any:
        """The underlying pyoptinterface model."""
        return self._m

    # ------------------------------------------------------------ variables
    def add_variable(
        self,
        name: str | None = None,
        lb: float = -INF,
        ub: float = INF,
        integer: bool = False,
        binary: bool = False,
    ) -> Variable:
        if binary:
            domain = poi.VariableDomain.Binary
            lb, ub = max(lb, 0.0), min(ub, 1.0)
        elif integer:
            domain = poi.VariableDomain.Integer
        else:
            domain = poi.VariableDomain.Continuous
        v = self._m.add_variable(domain=domain, lb=lb, ub=ub, name=name or "")
        self._vars.append(v)
        self._var_info[v.index] = _VarInfo(name or "", lb, ub, integer, binary)
        if name:
            if name in self.names:
                raise ValueError(f"An object named {name!r} is already registered.")
            self.names[name] = v
        return v

    def variables(self) -> list[Variable]:
        return list(self._vars)

    def variable_name(self, v: Variable) -> str:
        return self._var_info[v.index].name

    def variable_by_index(self, index: int) -> Variable:
        return self._vars[index] if self._vars[index].index == index else next(
            x for x in self._vars if x.index == index
        )

    def lower_bound(self, v: Variable) -> float:
        return float(self._m.get_variable_attribute(v, poi.VariableAttribute.LowerBound))

    def upper_bound(self, v: Variable) -> float:
        return float(self._m.get_variable_attribute(v, poi.VariableAttribute.UpperBound))

    def has_lower_bound(self, v: Variable) -> bool:
        return self.lower_bound(v) > -INF

    def has_upper_bound(self, v: Variable) -> bool:
        return self.upper_bound(v) < INF

    def set_lower_bound(self, v: Variable, value: float) -> None:
        self._m.set_variable_attribute(v, poi.VariableAttribute.LowerBound, float(value))

    def set_upper_bound(self, v: Variable, value: float) -> None:
        self._m.set_variable_attribute(v, poi.VariableAttribute.UpperBound, float(value))

    def delete_lower_bound(self, v: Variable) -> None:
        self.set_lower_bound(v, -INF)

    def delete_upper_bound(self, v: Variable) -> None:
        self.set_upper_bound(v, INF)

    def fix(self, v: Variable, value: float) -> None:
        """Fix ``v`` to ``value`` (JuMP.fix): sets lb = ub = value."""
        value = float(value)
        self._m.set_variable_bounds(v, value, value)
        self._fixed[v.index] = value

    def unfix(self, v: Variable) -> None:
        """Remove a fix, restoring the bounds the variable was created with."""
        info = self._var_info[v.index]
        self._m.set_variable_bounds(v, info.lb, info.ub)
        self._fixed.pop(v.index, None)

    def is_fixed(self, v: Variable) -> bool:
        return v.index in self._fixed

    def fix_value(self, v: Variable) -> float:
        return self._fixed[v.index]

    def domain(self, v: Variable) -> poi.VariableDomain:
        return self._m.get_variable_attribute(v, poi.VariableAttribute.Domain)

    def is_integer(self, v: Variable) -> bool:
        return self.domain(v) == poi.VariableDomain.Integer

    def is_binary(self, v: Variable) -> bool:
        return self.domain(v) == poi.VariableDomain.Binary

    def set_integer(self, v: Variable) -> None:
        self._m.set_variable_attribute(v, poi.VariableAttribute.Domain, poi.VariableDomain.Integer)

    def set_binary(self, v: Variable) -> None:
        self._m.set_variable_attribute(v, poi.VariableAttribute.Domain, poi.VariableDomain.Binary)

    def set_continuous(self, v: Variable) -> None:
        self._m.set_variable_attribute(
            v, poi.VariableAttribute.Domain, poi.VariableDomain.Continuous
        )

    def relax_integrality(self) -> Callable[[], None]:
        """Relax all integer/binary variables; returns a function that undoes it."""
        relaxed: list[tuple[Variable, poi.VariableDomain, float, float]] = []
        for v in self._vars:
            d = self.domain(v)
            if d in (poi.VariableDomain.Integer, poi.VariableDomain.Binary):
                lb, ub = self.lower_bound(v), self.upper_bound(v)
                relaxed.append((v, d, lb, ub))
                self.set_continuous(v)
                if d == poi.VariableDomain.Binary:
                    # Binary variables keep their [0, 1] box when relaxed.
                    self._m.set_variable_bounds(v, max(lb, 0.0), min(ub, 1.0))

        def undo() -> None:
            for v, d, lb, ub in relaxed:
                self._m.set_variable_attribute(v, poi.VariableAttribute.Domain, d)
                self._m.set_variable_bounds(v, lb, ub)

        return undo

    def has_integrality(self) -> bool:
        return any(
            self.domain(v) in (poi.VariableDomain.Integer, poi.VariableDomain.Binary)
            for v in self._vars
        )

    # ---------------------------------------------------------- constraints
    def add_constraint(self, con: Any, name: str | None = None) -> Constraint:
        """Add a linear constraint.

        ``con`` is a comparison built with operators, e.g. ``x + y <= 1`` or
        ``x.out == x.in_ + u``. Both sides may be expressions; the constraint is
        normalised to ``expr (sense) rhs`` with all variables on the left.
        """
        if hasattr(con, "lhs") and hasattr(con, "rhs") and hasattr(con, "sense"):
            lhs, rhs, sense = con.lhs, con.rhs, con.sense
        elif isinstance(con, tuple) and len(con) == 3:
            lhs, sense, rhs = con
        else:
            raise TypeError(
                "add_constraint expects a comparison like `expr <= rhs`, `expr == rhs`."
            )
        f = to_expression(lhs) - to_expression(rhs)
        f = to_expression(f)
        rhs_value = -expr_constant(f)
        g = poi.ScalarAffineFunction()
        terms: dict[int, float] = {}
        for i, c in zip(f.variables, f.coefficients):
            terms[int(i)] = terms.get(int(i), 0.0) + float(c)
        var_map: dict[int, Variable] = {}
        for i, c in terms.items():
            v = self._var_by_index(i)
            var_map[i] = v
            g.add_term(v, c)
        return self._add_normalized(g, sense, rhs_value, name, var_map)

    def add_constraint_normalized(
        self, expr: Any, sense: str, rhs: float, name: str | None = None
    ) -> Constraint:
        """Add ``expr (sense) rhs`` where sense is one of ``"==", "<=", ">="``."""
        s = {
            "==": poi.ConstraintSense.Equal,
            "<=": poi.ConstraintSense.LessEqual,
            ">=": poi.ConstraintSense.GreaterEqual,
        }[sense]
        f = to_expression(expr)
        g = poi.ScalarAffineFunction()
        var_map: dict[int, Variable] = {}
        for i, c in zip(f.variables, f.coefficients):
            v = self._var_by_index(int(i))
            var_map[int(i)] = v
            g.add_term(v, float(c))
        return self._add_normalized(g, s, float(rhs) - expr_constant(f), name, var_map)

    def _add_normalized(
        self,
        g: Expression,
        sense: poi.ConstraintSense,
        rhs: float,
        name: str | None,
        var_map: dict[int, Variable],
    ) -> Constraint:
        c = self._m.add_linear_constraint(g, sense, rhs, name=name or "")
        self._cons.append(c)
        self._con_info[c.index] = _ConInfo(name or "", sense, float(rhs), var_map)
        if name:
            if name in self.names:
                raise ValueError(f"An object named {name!r} is already registered.")
            self.names[name] = c
        return c

    def delete_constraint(self, c: Constraint) -> None:
        self._m.delete_constraint(c)
        self._deleted.add(c.index)

    def constraints(self) -> list[Constraint]:
        return [c for c in self._cons if c.index not in self._deleted]

    def constraint_name(self, c: Constraint) -> str:
        return self._con_info[c.index].name

    def constraint_data(self, c: Constraint) -> tuple[list[tuple[Variable, float]], str, float]:
        """Return the *current* ``(terms, sense, rhs)`` of a constraint."""
        info = self._con_info[c.index]
        terms = [
            (v, float(self._m.get_normalized_coefficient(c, v))) for v in info.variables.values()
        ]
        sense = {
            poi.ConstraintSense.Equal: "==",
            poi.ConstraintSense.LessEqual: "<=",
            poi.ConstraintSense.GreaterEqual: ">=",
        }[info.sense]
        return terms, sense, info.rhs

    def set_normalized_rhs(self, c: Constraint, value: float) -> None:
        # The RHS is tracked here rather than read back from the solver: pyoptinterface's
        # HiGHS `get_normalized_rhs` returns the row *lower* bound for `<=` rows.
        self._m.set_normalized_rhs(c, float(value))
        self._con_info[c.index].rhs = float(value)

    def get_normalized_rhs(self, c: Constraint) -> float:
        return self._con_info[c.index].rhs

    def set_normalized_coefficient(self, c: Constraint, v: Variable, value: float) -> None:
        self._m.set_normalized_coefficient(c, v, float(value))
        self._con_info[c.index].variables.setdefault(v.index, v)

    def get_normalized_coefficient(self, c: Constraint, v: Variable) -> float:
        return float(self._m.get_normalized_coefficient(c, v))

    # ------------------------------------------------------------ objective
    def set_objective(self, expr: Any, sense: Sense | None = None) -> None:
        if sense is not None:
            self._sense = sense
        self._objective = to_expression(expr)
        self._m.set_objective(self._objective, self._sense.poi)

    def set_objective_sense(self, sense: Sense) -> None:
        self._sense = sense
        self._m.set_obj_sense(sense.poi)

    @property
    def objective_sense(self) -> Sense:
        return self._sense

    def objective_function(self) -> Expression:
        if self._objective is None:
            return to_expression(0.0)
        return self._objective

    # ---------------------------------------------------------------- solve
    def optimize(self) -> None:
        self._m.optimize()
        self.solve_count += 1

    def termination_status(self) -> TerminationStatus:
        return self._m.get_model_attribute(poi.ModelAttribute.TerminationStatus)

    def primal_status(self) -> ResultStatus:
        return self._m.get_model_attribute(poi.ModelAttribute.PrimalStatus)

    def dual_status(self) -> ResultStatus:
        return self._m.get_model_attribute(poi.ModelAttribute.DualStatus)

    def has_primal_solution(self) -> bool:
        return self.primal_status() in (
            ResultStatus.FEASIBLE_POINT,
            ResultStatus.NEARLY_FEASIBLE_POINT,
        )

    def has_dual_solution(self) -> bool:
        return self.dual_status() in (
            ResultStatus.FEASIBLE_POINT,
            ResultStatus.NEARLY_FEASIBLE_POINT,
        )

    def objective_value(self) -> float:
        return float(self._m.get_obj_value())

    def value(self, x: Any) -> float:
        """Primal value of a variable, expression, or number."""
        if isinstance(x, Variable):
            return float(self._m.get_value(x)) + 0.0
        if _is_number(x):
            return float(x)
        f = to_expression(x)
        total = expr_constant(f)
        for i, c in zip(f.variables, f.coefficients):
            total += float(c) * float(self._m.get_value(self._var_by_index(int(i))))
        return total

    def reduced_cost(self, v: Variable) -> float:
        """d objective / d (bound) for a variable at a bound (sensitivity, model sense)."""
        return float(self._m.get_variable_attribute(v, poi.VariableAttribute.ReducedCost))

    def dual(self, c: Constraint) -> float:
        """d objective / d rhs of a constraint (sensitivity, model sense)."""
        return float(self._m.get_constraint_dual(c))

    def constraint_primal(self, c: Constraint) -> float:
        return float(self._m.get_constraint_primal(c))

    def reset_optimizer(self) -> None:
        """Best-effort analogue of ``MOI.Utilities.reset_optimizer``: drop the warm start."""
        try:
            self._m.set_raw_parameter("presolve", "on")
        except Exception:  # pragma: no cover - solver specific
            pass

    def set_raw_parameter(self, name: str, value: Any) -> None:
        self._m.set_raw_parameter(name, value)

    def write(self, filename: str) -> None:
        self._m.write(filename)

    # -------------------------------------------------------------- helpers
    def _var_by_index(self, index: int) -> Variable:
        v = self._vars[index] if index < len(self._vars) else None
        if v is not None and v.index == index:
            return v
        for x in self._vars:
            if x.index == index:
                return x
        raise KeyError(f"No variable with index {index}")

    def __getitem__(self, name: str) -> Any:
        return self.names[name]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._vars)

    def num_variables(self) -> int:
        return len(self._vars)

    def num_constraints(self) -> int:
        return len(self.constraints())

    @staticmethod
    def expression(terms: Sequence[tuple[Variable, float]], constant: float = 0.0) -> Expression:
        e = poi.ScalarAffineFunction()
        for v, c in terms:
            e.add_term(v, float(c))
        if constant:
            e.add_constant(float(constant))
        return e
