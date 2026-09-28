# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Duality handlers. Ported from ``src/plugins/duality_handlers.jl``.

* :class:`ContinuousConicDuality` (Tier 1): duals of the LP relaxation.
* :class:`LagrangianDuality`, :class:`StrengthenedConicDuality`,
  :class:`FixedDiscreteDuality`, :class:`BanditDuality` (Tier 3): integer states.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from sddp.plugins.base import DualityHandler
from sddp.plugins.local_improvement_search import BFGS, AbstractSearchMethod
from sddp.policy_graph import Log, Node, PolicyGraph, _get_incoming_domain
from sddp.solver.model import Model, OptimizerFactory, Sense, TerminationStatus

if TYPE_CHECKING:
    from sddp.algorithm import Options


def get_dual_solution_none(node: Node) -> tuple[float, dict[str, float]]:
    """``get_dual_solution(node, nothing)``: objective only, no duals."""
    return node.model.objective_value(), {}


class ContinuousConicDuality(DualityHandler):
    """Duals from the LP relaxation (integrality relaxed on the backward pass).

    The dual of the fishing constraint ``x_in == x̄`` is the reduced cost of the fixed
    variable. Our solver layer returns it as ``d objective / d x̄`` in the model's own
    sense, so no sign flip is needed (SDDP.jl needs one because of JuMP's convention).
    """

    def __init__(self, optimizer: object | None = None):
        self.optimizer = optimizer

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        from sddp.algorithm import attempt_numerical_recovery

        m = node.model
        if not m.has_dual_solution():
            attempt_numerical_recovery(node.policy_graph, node, require_dual=True)
        lam = {name: m.reduced_cost(state.in_) for name, state in node.states.items()}
        return m.objective_value(), lam

    def prepare_backward_pass(self, node: Node, options: Options) -> Callable[[], None]:
        return _relax_integrality(node, self.optimizer)

    def duality_log_key(self) -> str:
        return " "


def _relax_integrality(node: Node, optimizer: object | None = None) -> Callable[[], None]:
    """Relax integrality for the backward pass (``_relax_integrality(node, optimizer)``).

    SDDP.jl calls ``JuMP.set_optimizer(subproblem, optimizer)`` for the relaxed solves and
    switches back afterwards. The port's models are bound to one backend, so ``optimizer``
    must be an :class:`OptimizerFactory` of the *same* backend (typically
    ``sddp.HiGHS.with_options(...)``): its raw options are applied while integrality is
    relaxed and the previous values are restored by the returned undo function.
    """
    if not node.has_integrality:
        return lambda: None
    undo_relax = node.model.relax_integrality()
    if optimizer is None:
        return undo_relax
    undo_options = _apply_temporary_optimizer(node.model, optimizer)

    def undo() -> None:
        undo_options()
        undo_relax()

    return undo


def _apply_temporary_optimizer(m: Model, optimizer: object) -> Callable[[], None]:
    if not isinstance(optimizer, OptimizerFactory) or optimizer.name != m.optimizer.name:
        raise NotImplementedError(
            "The `optimizer` argument of a duality handler must be an OptimizerFactory of the "
            f"same backend as the model ({m.optimizer.name!r}), e.g. "
            "`sddp.HiGHS.with_options(presolve='off')`; the port cannot swap solver backends "
            f"for the backward pass. Got {optimizer!r}."
        )
    saved = {k: m.raw.get_raw_parameter(k) for k in optimizer.options}
    for k, v in optimizer.options.items():
        m.set_raw_parameter(k, v)

    def undo() -> None:
        for k, v in saved.items():
            m.set_raw_parameter(k, v)

    return undo


def _sparsify(x: float) -> float:
    return 0.0 if abs(x) < 1e-15 else x


def _initialize_incoming_state_bounds(policy_graph: PolicyGraph) -> None:
    domain = _get_incoming_domain(policy_graph)
    for node_name, node in policy_graph.nodes.items():
        for k, v in domain[node_name].items():
            node.incoming_state_bounds[k] = v if v is not None else (-math.inf, math.inf, False)


def _solve_primal_problem(
    model: Model, lam: np.ndarray, h_expr: list[Any], h_k: np.ndarray
) -> float | None:
    """Solve ``min primal_obj - λ'h(x̄)``; fill ``h_k = -h(x̄*)``; restore the objective."""
    primal_obj = model.objective_function()
    obj = primal_obj
    for i, h in enumerate(h_expr):
        obj = obj - float(lam[i]) * h
    model.set_objective(obj)
    model.optimize()
    if model.termination_status() != TerminationStatus.OPTIMAL:
        model.set_objective(primal_obj)
        return None
    for i, h in enumerate(h_expr):
        h_k[i] = -model.value(h)
    L = model.objective_value()
    model.set_objective(primal_obj)
    return L


class LagrangianDuality(DualityHandler):
    """Duals from Lagrangian duality: maximise ``L(λ) = min C(x̄,u,w) + θ − λ'(x̄ − x)``."""

    def __init__(self, optimizer: object | None = None, method: AbstractSearchMethod | None = None):
        self.optimizer = optimizer
        self.method = method if method is not None else BFGS(100)

    def prepare_backward_pass(self, node: Node, options: Options) -> Callable[[], None]:
        if not node.incoming_state_bounds:
            _initialize_incoming_state_bounds(node.policy_graph)
        return lambda: None

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        m = node.model
        undo_relax = _relax_integrality(node, self.optimizer)
        m.optimize()
        conic_obj, conic_dual = ContinuousConicDuality().get_dual_solution(node)
        undo_relax()
        s = -1.0 if m.objective_sense is Sense.MIN else 1.0
        keys = list(node.states)
        N = len(keys)
        x_in_value = np.zeros(N)
        lam_star = np.zeros(N)
        h_expr: list[Any] = []
        h_k = np.zeros(N)
        was_integer: list[bool] = []
        for i, key in enumerate(keys):
            state = node.states[key]
            x_in_value[i] = m.fix_value(state.in_)
            h_expr.append(1.0 * state.in_ - x_in_value[i])
            m.unfix(state.in_)
            l, u, is_integer = node.incoming_state_bounds[key]
            if l > -math.inf:
                m.set_lower_bound(state.in_, l)
            if u < math.inf:
                m.set_upper_bound(state.in_, u)
            if is_integer:
                m.set_integer(state.in_)
            was_integer.append(is_integer)
            lam_star[i] = conic_dual[key]

        def restore() -> None:
            for i, key in enumerate(keys):
                state = node.states[key]
                if was_integer[i]:
                    m.set_continuous(state.in_)
                m.fix(state.in_, x_in_value[i])

        L_k = _solve_primal_problem(m, lam_star, h_expr, h_k)
        if L_k is None:
            restore()
            return conic_obj, conic_dual

        def f(x: np.ndarray) -> tuple[float, np.ndarray] | None:
            L = _solve_primal_problem(m, x, h_expr, h_k)
            if L is None:
                return None
            return s * L, s * h_k.copy()

        L_star, lam_star = self.method.minimize(f, lam_star, conic_obj)
        restore()
        lam_solution = {k: _sparsify(float(lam_star[i])) for i, k in enumerate(keys)}
        return s * L_star, lam_solution

    def duality_log_key(self) -> str:
        return "L"


class StrengthenedConicDuality(DualityHandler):
    """Strengthened Benders: conic duals with the intercept from the Lagrangian evaluation."""

    def __init__(self, optimizer: object | None = None):
        self.optimizer = optimizer

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        m = node.model
        undo_relax = _relax_integrality(node, self.optimizer)
        m.optimize()
        conic_obj, conic_dual = ContinuousConicDuality().get_dual_solution(node)
        undo_relax()
        if not node.has_integrality:
            return conic_obj, conic_dual
        keys = list(node.states)
        N = len(keys)
        lam_k, h_k, x = np.zeros(N), np.zeros(N), np.zeros(N)
        h_expr: list[Any] = []
        for i, key in enumerate(keys):
            state = node.states[key]
            x[i] = m.fix_value(state.in_)
            h_expr.append(1.0 * state.in_ - x[i])
            m.unfix(state.in_)
            lam_k[i] = conic_dual[key]
        lagrangian_obj = _solve_primal_problem(m, lam_k, h_expr, h_k)
        for i, key in enumerate(keys):
            m.fix(node.states[key].in_, x[i])
        return (lagrangian_obj if lagrangian_obj is not None else conic_obj), conic_dual

    def duality_log_key(self) -> str:
        return "S"


class FixedDiscreteDuality(DualityHandler):
    """Solve the MIP, fix the discrete variables, take LP duals, and re-evaluate the intercept."""

    def __init__(self, optimizer: object | None = None):
        self.optimizer = optimizer

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        m = node.model
        if not node.has_integrality:
            return ContinuousConicDuality().get_dual_solution(node)
        undo_fix = _fix_discrete_variables(m)
        m.optimize()
        _, conic_dual = ContinuousConicDuality().get_dual_solution(node)
        undo_fix()
        keys = list(node.states)
        N = len(keys)
        lam_k, h_k, x = np.zeros(N), np.zeros(N), np.zeros(N)
        h_expr: list[Any] = []
        for i, key in enumerate(keys):
            state = node.states[key]
            x[i] = m.fix_value(state.in_)
            h_expr.append(1.0 * state.in_ - x[i])
            m.unfix(state.in_)
            lam_k[i] = conic_dual[key]
        lagrangian_obj = _solve_primal_problem(m, lam_k, h_expr, h_k)
        for i, key in enumerate(keys):
            m.fix(node.states[key].in_, x[i])
        if lagrangian_obj is not None:
            return lagrangian_obj, conic_dual
        undo_relax = _relax_integrality(node, self.optimizer)
        m.optimize()
        ret = ContinuousConicDuality().get_dual_solution(node)
        undo_relax()
        return ret

    def duality_log_key(self) -> str:
        return "F"


def _fix_discrete_variables(m: Model) -> Callable[[], None]:
    """``JuMP.fix_discrete_variables``: fix integer/binary variables at their current values."""
    saved = []
    for v in m.variables():
        if m.is_integer(v) or m.is_binary(v):
            saved.append((v, m.domain(v), m.lower_bound(v), m.upper_bound(v), m.value(v)))
    for v, _, _, _, val in saved:
        m.set_continuous(v)
        m.raw.set_variable_bounds(v, float(round(val)), float(round(val)))

    def undo() -> None:
        for v, d, lb, ub, _ in saved:
            m.raw.set_variable_bounds(v, lb, ub)
            m.raw.set_variable_attribute(
                v, __import__("pyoptinterface").VariableAttribute.Domain, d
            )

    return undo


class _BanditArm:
    def __init__(self, handler: DualityHandler):
        self.handler = handler
        self.rewards: list[float] = []


class BanditDuality(DualityHandler):
    """Choose between duality handlers with a simple multi-armed-bandit heuristic."""

    def __init__(self, *args: DualityHandler, optimizer: object | None = None):
        if not args:
            args = (ContinuousConicDuality(optimizer), StrengthenedConicDuality(optimizer))
        self.arms = [_BanditArm(a) for a in args]
        self.last_arm_index = 0
        self.logs_seen = 1

    def __repr__(self) -> str:
        return "BanditDuality with arms:" + "".join(f"\n * {a.handler}" for a in self.arms)

    def _reset(self) -> None:
        for arm in self.arms:
            arm.rewards.clear()
        self.last_arm_index = 0
        self.logs_seen = 1

    def _update_arm(self, rng: Any) -> None:
        scores: list[float] = []
        for arm in self.arms:
            if not arm.rewards:
                scores.append(math.nan)
                continue
            mu = statistics.mean(arm.rewards)
            sigma = statistics.stdev(arm.rewards) if len(arm.rewards) > 1 else math.nan
            scores.append(mu if math.isnan(sigma) else mu + sigma)
        if any(math.isnan(s) for s in scores):
            self.last_arm_index = rng.choice([i for i, s in enumerate(scores) if math.isnan(s)])
            return
        mx = max(scores)
        z = [math.exp(s - mx) for s in scores]
        total = sum(z)
        z = [v / total for v in z]
        r = rng.random()
        index = len(z) - 1
        for i, zi in enumerate(z):
            r -= zi
            if r <= 0:
                index = i
                break
        self.last_arm_index = index

    def _update_rewards(self, log: list[Log]) -> None:
        t, t2 = log[-1], log[-2]
        reward = abs(t.bound - t2.bound) / max(t.time - t2.time, 0.1)
        self.arms[self.last_arm_index].rewards.append(reward)

    def prepare_backward_pass(self, node: Node, options: Options) -> Callable[[], None]:
        log = options.log
        if not log:
            self._reset()
        if len(log) > self.logs_seen:
            self._update_rewards(log)
            self.logs_seen = len(log)
            if len(log) >= 2 and math.isclose(log[-1].bound, log[-2].bound, abs_tol=1e-6):
                self.last_arm_index = (self.last_arm_index + 1) % len(self.arms)
            else:
                self._update_arm(options.rng)
        return self.arms[self.last_arm_index].handler.prepare_backward_pass(node, options)

    def get_dual_solution(self, node: Node) -> tuple[float, dict[str, float]]:
        return self.arms[self.last_arm_index].handler.get_dual_solution(node)

    def duality_log_key(self) -> str:
        return self.arms[self.last_arm_index].handler.duality_log_key()
