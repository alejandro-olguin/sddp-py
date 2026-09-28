# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""First-order minimisation of convex piecewise-linear functions.

Ported from ``src/plugins/local_improvement_search.jl``. Used by
:class:`sddp.plugins.duality_handlers.LagrangianDuality` to maximise the Lagrangian dual.

``f(x)`` returns ``None`` if ``x`` is infeasible, otherwise ``(value, gradient)``.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence

import numpy as np

SearchFunction = Callable[[np.ndarray], "tuple[float, np.ndarray] | None"]


def _norm(x: np.ndarray) -> float:
    return float(math.sqrt(float(np.dot(x, x))))


class AbstractSearchMethod(abc.ABC):
    @abc.abstractmethod
    def minimize(
        self, f: SearchFunction, x0: Sequence[float] | np.ndarray, lower_bound: float = -math.inf
    ) -> tuple[float, np.ndarray]: ...


class BFGS(AbstractSearchMethod):
    """A modified BFGS with a back-tracking inexact line search (``BFGS(evaluation_limit)``)."""

    def __init__(self, evaluation_limit: int = 100):
        self.evaluation_limit = evaluation_limit

    def minimize(
        self, f: SearchFunction, x0: Sequence[float] | np.ndarray, lower_bound: float = -math.inf
    ) -> tuple[float, np.ndarray]:
        x_k = np.asarray(x0, dtype=float).copy()
        n = len(x_k)
        B = np.eye(n)
        ret = f(x_k)
        assert ret is not None, "the initial iterate must be feasible"
        f_k, g_k = ret
        alpha_k = 1.0
        evals = [self.evaluation_limit]
        while True:
            p_k = np.linalg.solve(B, -g_k) if n > 0 else np.zeros(0)
            alpha_k, f_k1, g_k1 = _line_search(f, f_k, g_k, x_k, p_k, alpha_k, evals)
            if _norm(alpha_k * p_k) / max(1.0, _norm(x_k)) < 1e-3:
                return f_k, x_k
            elif _norm(g_k1) < 1e-6:
                return f_k1, x_k + alpha_k * p_k
            elif evals[0] <= 0:
                return f_k1, x_k + alpha_k * p_k
            s_k = alpha_k * p_k
            y_k = g_k1 - g_k
            if _norm(y_k) > 1e-12:
                Bs = B @ s_k
                B = B + np.outer(y_k, y_k) / float(y_k @ s_k) - np.outer(Bs, Bs) / float(s_k @ Bs)
            f_k, g_k, x_k = f_k1, g_k1, x_k + s_k


def _line_search(
    f: SearchFunction,
    f_k: float,
    g_k: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    alpha: float,
    evals: list[int],
) -> tuple[float, float, np.ndarray]:
    while _norm(alpha * p) > 1e-3 * max(1.0, _norm(x)):
        x_k = x + alpha * p
        ret = f(x_k)
        evals[0] -= 1
        if ret is None:
            alpha /= 2
            continue
        f_k1, g_k1 = ret
        if float(p @ g_k1) < 1e-6:
            return alpha, f_k1, g_k1
        elif math.isclose(f_k + alpha * float(p @ g_k), f_k1, abs_tol=1e-8, rel_tol=0.0):
            return alpha, f_k1, g_k1
        alpha = (f_k1 - f_k - float(p @ g_k1) * alpha) / (float(p @ g_k) - float(p @ g_k1))
    return 0.0, f_k, g_k


class OuterApproximation(AbstractSearchMethod):
    """Kelley's outer approximation using an LP solver (``OuterApproximation(optimizer)``)."""

    def __init__(self, optimizer: object):
        self.optimizer = optimizer

    def minimize(
        self, f: SearchFunction, x0: Sequence[float] | np.ndarray, lower_bound: float = -math.inf
    ) -> tuple[float, np.ndarray]:
        from sddp.solver.model import Model, Sense

        model = Model(self.optimizer)  # type: ignore[arg-type]
        n = len(x0)
        x = [model.add_variable() for _ in range(n)]
        theta = model.add_variable(lb=lower_bound)
        model.set_objective(1.0 * theta, Sense.MIN)
        x_k = np.asarray(x0, dtype=float)
        ret = f(x_k)
        assert ret is not None
        f_k, g_k = ret

        def add_cut(fv: float, gv: np.ndarray, xv: np.ndarray) -> None:
            terms = [(theta, 1.0)] + [(x[i], -float(gv[i])) for i in range(n)]
            model.add_constraint_normalized(Model.expression(terms), ">=", fv - float(gv @ xv))

        add_cut(f_k, g_k, x_k)
        evals = 0
        d_step = math.inf
        while d_step > 1e-8 and evals < 20:
            model.optimize()
            x_k1 = np.array([model.value(v) for v in x])
            ret = f(x_k1)
            while ret is None:
                x_k1 = 0.5 * (x_k + x_k1)
                ret = f(x_k1)
            f_k1, g_k1 = ret
            evals += 1
            add_cut(f_k1, g_k1, x_k1)
            d = x_k1 - x_k
            d_step = _norm(d)
            if np.sign(float(g_k @ d)) != np.sign(float(g_k1 @ d)):
                x_k2 = 0.5 * (x_k + x_k1)
                ret2 = f(x_k2)
                assert ret2 is not None
                f_k2, g_k2 = ret2
                evals += 1
                add_cut(f_k2, g_k2, x_k2)
                f_k, x_k = f_k2, x_k2
            else:
                f_k, x_k = f_k1, x_k1
        return f_k, x_k
