# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Risk measures. Ported from ``src/plugins/risk_measures.jl``.

Every measure implements ``adjust_probability(q, p, supports, V, is_min) -> offset``:
it fills the risk-adjusted probabilities ``q`` in place and returns an additive
offset (non-zero only for :class:`Entropic`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sddp.plugins.base import RiskMeasure
from sddp.solver.model import Model, OptimizerFactory, Sense


def _isapprox(a: float, b: float, atol: float = 0.0) -> bool:
    return math.isclose(a, b, rel_tol=1.4901161193847656e-08, abs_tol=atol)


class Expectation(RiskMeasure):
    """The expectation with respect to the nominal distribution."""

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        q[:] = list(p)
        return 0.0

    def __repr__(self) -> str:
        return "Expectation()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expectation)

    def __hash__(self) -> int:
        return hash("Expectation")


class WorstCase(RiskMeasure):
    """All probability weight on the worst outcome (among outcomes with ``p > 0``)."""

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        q[:] = [0.0] * len(p)
        worst_index = 0
        worst_observation = -math.inf if is_min else math.inf
        for index, (probability, observation) in enumerate(zip(p, V)):
            if probability > 0.0:
                if (is_min and observation > worst_observation) or (
                    not is_min and observation < worst_observation
                ):
                    worst_index = index
                    worst_observation = observation
        q[worst_index] = 1.0
        return 0.0

    def __repr__(self) -> str:
        return "WorstCase()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorstCase)

    def __hash__(self) -> int:
        return hash("WorstCase")


class AVaR(RiskMeasure):
    """Average value at risk: expectation of the ``β`` fraction of worst outcomes."""

    def __init__(self, beta: float):
        if not (0 <= beta <= 1):
            raise ValueError(f"Risk-quantile β must be in [0, 1]. Currently it is {beta}.")
        self.beta = float(beta)

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        if _isapprox(self.beta, 0.0):
            return WorstCase().adjust_probability(q, p, supports, V, is_min)
        elif _isapprox(self.beta, 1.0):
            return Expectation().adjust_probability(q, p, supports, V, is_min)
        q[:] = [0.0] * len(p)
        quantile_collected = 0.0
        # Julia: sortperm(V; rev = is_min) -- stable, so ties keep their original order.
        order = sorted(range(len(V)), key=lambda i: V[i], reverse=is_min)
        for i in order:
            if quantile_collected >= self.beta:
                break
            avar_prob = min(p[i], self.beta - quantile_collected) / self.beta
            q[i] = avar_prob
            quantile_collected += avar_prob * self.beta
        return 0.0

    def __repr__(self) -> str:
        return f"AVaR({self.beta})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AVaR) and other.beta == self.beta

    def __hash__(self) -> int:
        return hash(("AVaR", self.beta))


CVaR = AVaR


class ConvexCombination(RiskMeasure):
    """A weighted combination ``Σ wᵢ · measureᵢ`` of risk measures."""

    def __init__(self, *measures: tuple[float, RiskMeasure]):
        self.measures: tuple[tuple[float, RiskMeasure], ...] = tuple((float(w), m) for w, m in measures)

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        q[:] = [0.0] * len(p)
        alpha = 0.0
        for weight, measure in self.measures:
            partial = [0.0] * len(p)
            a = measure.adjust_probability(partial, p, supports, V, is_min)
            alpha += weight * a
            for i in range(len(p)):
                q[i] += weight * partial[i]
        return alpha

    def __add__(self, other: Any) -> ConvexCombination:
        if isinstance(other, ConvexCombination):
            return ConvexCombination(*self.measures, *other.measures)
        if isinstance(other, RiskMeasure):
            return ConvexCombination(*self.measures, (1.0, other))
        return NotImplemented

    def __repr__(self) -> str:
        return "A convex combination of " + " + ".join(f"{w} * {m!r}" for w, m in self.measures)


def EAVaR(lambda_: float = 1.0, beta: float = 1.0, **kwargs: Any) -> ConvexCombination:
    """``λ E[x] + (1 − λ) AV@R(β)[x]``. Accepts ``lambda_=`` (or ``lambda=`` via kwargs)."""
    if "lambda" in kwargs:
        lambda_ = kwargs.pop("lambda")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError(
            "Lambda must be in the range [0, 1]. Increasing values of lambda are less risk "
            "averse. lambda=1 is identical to taking the expectation."
        )
    if not (0.0 <= beta <= 1.0):
        raise ValueError(
            "Beta must be in the range [0, 1]. Increasing values of beta are less risk averse."
        )
    return ConvexCombination((lambda_, Expectation()), (1 - lambda_, AVaR(beta)))


class ModifiedChiSquared(RiskMeasure):
    """Distributionally robust risk measure of Philpott, de Matos & Kapelevich (2018)."""

    def __init__(self, radius: float, minimum_std: float = 1e-5):
        if abs(radius) < 1e-9:
            import warnings

            warnings.warn("Radius is very small. You should probably use `Expectation()` instead.", stacklevel=2)
        self.radius = float(radius)
        self.minimum_std = float(minimum_std)

    def __repr__(self) -> str:
        return f"ModifiedChiSquared with radius={self.radius}"

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        if float(np.std(np.asarray(V, dtype=float))) < self.minimum_std:  # uncorrected std
            return Expectation().adjust_probability(q, p, supports, V, is_min)
        m = len(V)
        if all(_isapprox(x, 1 / m) for x in p):
            _uniform_dro(self, q, p, V, is_min)
        else:
            _non_uniform_dro(self, q, p, V, is_min)
        return 0.0


def _non_uniform_dro(measure: ModifiedChiSquared, p: list[float], q: Sequence[float], z_in: Sequence[float], is_min: bool) -> float:
    """Algorithm (1) of Philpott et al.: nominal distribution is not uniform."""
    m = len(z_in)
    z = list(z_in)
    if float(np.std(np.asarray(z), ddof=1)) < 1e-6:
        p[:] = list(q)
        return 0.0
    if not is_min:
        z = [-v for v in z]
    K = list(range(m))
    # Julia mutates K while iterating (splice! inside `for i in K`); replicate its effect.
    for i in list(K):
        if _isapprox(q[i], 0.0, atol=1e-10):
            p[i] = 0.0
            if i in K:
                K.remove(i)
    m = len(K)
    not_in_K: list[int] = []
    while len(K) > 1:
        z_bar = sum(z[i] for i in K) / len(K)
        s = math.sqrt(sum(z[i] ** 2 - z_bar**2 for i in K) / len(K))
        if _isapprox(s, 0.0, atol=1e-10):
            raise ValueError("s is too small")
        if len(K) == m:
            for i in K:
                p[i] = q[i] + (z[i] - z_bar) / (math.sqrt(m) * s) * measure.radius
        else:
            for i in not_in_K:
                p[i] = 0.0
            sum_qj = sum(q[i] for i in not_in_K)
            sum_qj_squared = sum(q[i] ** 2 for i in not_in_K)
            len_k = len(K)
            n = math.sqrt(len_k * (measure.radius**2 - sum_qj_squared) - sum_qj**2)
            for i in K:
                p[i] = q[i] + 1 / len_k * (sum_qj + n * (z[i] - z_bar) / s)
        if all(pi >= 0.0 for pi in p):
            return 0.0
        negative_p = [i for i in K if p[i] < 0]
        sum_qj = 0.0
        sum_qj_squared = 0.0
        if not_in_K:
            sum_qj = sum(q[i] for i in not_in_K)
            sum_qj_squared = sum(q[i] ** 2 for i in not_in_K)
        len_k = len(K)
        computed_r = [
            (((-q[i] * len_k - sum_qj) / (z[i] - z_bar) / s) ** 2 + sum_qj_squared**2) / len_k + sum_qj_squared
            for i in negative_p
        ]
        i_K = negative_p[int(np.argmin(computed_r))]
        not_in_K.append(i_K)
        K = [e for e in K if e != i_K]
    for i in not_in_K:
        p[i] = 0.0
    p[K[0]] = 1.0
    return 0.0


def _uniform_dro(measure: ModifiedChiSquared, q_out: list[float], p: Sequence[float], V: Sequence[float], is_min: bool) -> float:
    """Algorithm (2) of Philpott et al.: nominal distribution is uniform."""
    m = len(V)
    perm = sorted(range(m), key=lambda i: V[i], reverse=not is_min)
    z = [V[i] for i in perm]
    pv = [0.0] * m  # permuted view

    def flush() -> None:
        for j, i in enumerate(perm):
            q_out[i] = pv[j]

    for k in range(0, m - 1):
        z_bar = sum(z[i] for i in range(k, m)) / (m - k)
        s2 = sum(z[i] ** 2 - z_bar**2 for i in range(k, m)) / (m - k)
        if s2 < -1e-8:
            raise ValueError(f"Something unexpected happened with s² term: `{s2} < 0.0`.")
        elif s2 <= 0.0:
            raise ValueError("`s²<0`: choose a larger threshold for `minimum_std`.")
        term_1 = 1 / (m - k)
        term_2 = math.sqrt((m - k) * measure.radius**2 - k / m) / ((m - k) * math.sqrt(s2))
        if k > 0:
            pv[k - 1] = 0.0
        if is_min:
            for i in range(k, m):
                pv[i] = term_1 + term_2 * (z[i] - z_bar)
        else:
            for i in range(k, m):
                pv[i] = term_1 + term_2 * (z_bar - z[i])
        if pv[k] >= 0.0:
            flush()
            return 0.0
    pv[m - 1] = 1.0
    flush()
    return 0.0


class Wasserstein(RiskMeasure):
    """Distributionally robust measure based on the Wasserstein distance (solves a small LP)."""

    def __init__(self, norm: Callable[[Any, Any], float], optimizer: OptimizerFactory, alpha: float):
        if alpha < 0.0:
            raise ValueError(f"alpha cannot be {alpha} as it must be in the range [0, ∞).")
        self.alpha = float(alpha)
        self.optimizer = optimizer
        self.norm = norm

    def __repr__(self) -> str:
        return "Wasserstein"

    def adjust_probability(self, q: list[float], p: Sequence[float], supports: Sequence[Any], V: Sequence[float], is_min: bool) -> float:
        N = len(V)
        m = Model(self.optimizer)
        z = [[m.add_variable(lb=0.0) for _ in range(N)] for _ in range(N)]
        pv = [m.add_variable(lb=0.0) for _ in range(N)]
        for i in range(N):
            m.add_constraint_normalized(Model.expression([(z[k][i], 1.0) for k in range(N)]), "==", p[i])
            m.add_constraint_normalized(
                Model.expression([(z[i][k], 1.0) for k in range(N)] + [(pv[i], -1.0)]), "==", 0.0
            )
        m.add_constraint_normalized(
            Model.expression(
                [(z[i][j], float(self.norm(supports[i], supports[j]))) for i in range(N) for j in range(N)]
            ),
            "<=",
            self.alpha,
        )
        m.set_objective(
            Model.expression([(pv[i], float(V[i])) for i in range(N)]),
            Sense.MAX if is_min else Sense.MIN,
        )
        m.optimize()
        assert m.has_primal_solution()
        q[:] = [m.value(v) for v in pv]
        return 0.0


class Entropic(RiskMeasure):
    """The entropic risk measure ``F[X] = (1/γ) log E[exp(γX)]`` (Dowson, Morton & Pagnoncelli)."""

    def __init__(self, gamma: float):
        self.gamma = float(gamma)

    def __repr__(self) -> str:
        return f"Entropic risk measure with γ = {self.gamma}"

    def adjust_probability(self, Q: list[float], p: Sequence[float], supports: Sequence[Any], X: Sequence[float], is_min: bool) -> float:
        if self.gamma == 0.0:
            Q[:] = list(p)
            return 0.0
        gamma = (1.0 if is_min else -1.0) * self.gamma
        # Julia uses BigFloat to avoid overflow; subtracting the max is equivalent after
        # normalisation.
        gx = [gamma * x for x in X]
        shift = max(gx)
        y = [pi * math.exp(g - shift) for pi, g in zip(p, gx)]
        total = sum(y)
        Q[:] = [yi / total for yi in y]
        alpha = sum(qi * math.log(qi / pi) for pi, qi in zip(p, Q) if pi > 1e-14 and qi > 1e-14)
        return -alpha / gamma
