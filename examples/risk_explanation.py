"""Python port of the SDDP.jl "Risk aversion" explanation page.

Source: ``docs/src/explanation/risk.jl`` (https://sddp.dev/stable/explanation/risk/).

The page extends the vanilla SDDP implementation of the "Introductory theory" page with
risk aversion. This module imports the structures and the forward pass from
:mod:`examples.theory_intro` and, like the page, redefines ``backward_pass`` and
``train`` to take a ``risk_measure``. Nothing here uses ``sddp.train``; the package is
only used by the tests to cross-validate the numbers.

Differences from the Julia page (all deliberate):

* ``ForwardDiff`` is replaced by central finite differences.
* ``BigFloat`` (used on the page to avoid ``exp`` overflow) is replaced by the
  log-sum-exp trick: ``exp(γ z)`` is only ever evaluated as ``exp(γ z - max γ z)``, which
  is in ``(0, 1]``, and the ``max`` is added back outside the ``log``. Sums use
  ``math.fsum``. Consequently the "Uh oh!" moment of the page (the primal subgradient
  returning ``NaN`` for ``γ = 100``) does not happen here; see
  :func:`primal_risk_averse_subgradient`.
* The Ipopt version of ``dual_risk_inner`` for :class:`Entropic` is not ported (no NLP
  solver in this project); only the closed form of Dowson, Morton & Pagnoncelli is.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from examples.theory_intro import (
    FINITE_GRAPH,
    PolicyGraph,
    State,
    Trajectory,
    Uncertainty,
    _println,
    add_cut,
    build_policy_graph,
    evaluate_policy,
    finite_difference_gradient,
    forward_pass,
    get_node,
    lower_bound,
    upper_bound,
)
from sddp.solver.model import HiGHS, Model, OptimizerFactory, Sense

# Re-exported so that this module, like the Julia page, is a complete description.
__all__ = [
    "AbstractRiskMeasure",
    "Entropic",
    "Expectation",
    "WorstCase",
    "backward_pass",
    "build_risk_averse_model",
    "dual_risk",
    "dual_risk_averse_subgradient",
    "dual_risk_inner",
    "evaluate_policy",
    "primal_risk",
    "primal_risk_averse_subgradient",
    "train",
]

# ---------------------------------------------------------------------------
# Primal risk measures
# ---------------------------------------------------------------------------

Z = [1.0, 2.0, 3.0, 4.0]
p = [0.1, 0.2, 0.4, 0.3]


class AbstractRiskMeasure:
    pass


class Expectation(AbstractRiskMeasure):
    """``F[Z] = Σ p_ω z_ω``."""


class WorstCase(AbstractRiskMeasure):
    """``F[Z] = max_ω z_ω`` (ignores ``p``)."""


class Entropic(AbstractRiskMeasure):
    """``F_γ[Z] = (1/γ) log(Σ p_ω exp(γ z_ω))`` with ``γ > 0``."""

    def __init__(self, γ: float):
        if not γ > 0:
            raise ValueError(f"Entropic risk measure must have γ > 0. Got {γ}.")
        self.γ = float(γ)


def primal_risk(F: AbstractRiskMeasure, Z: Sequence[float], p: Sequence[float]) -> float:
    """Risk under ``F`` of the random variable with costs ``Z`` and probabilities ``p``."""
    if isinstance(F, Expectation):
        return math.fsum(p[i] * Z[i] for i in range(len(p)))
    if isinstance(F, WorstCase):
        return max(Z)
    if isinstance(F, Entropic):
        # `exp(x)` overflows for x > 709, so the page switches to BigFloat. Here we use
        # log-sum-exp instead: with m = max γz, (1/γ) log Σ p e^{γz} = (1/γ)(m + log Σ p e^{γz-m})
        # and every exponent is ≤ 0. This is exact in real arithmetic and never overflows.
        γz = [F.γ * z for z in Z]
        m = max(γz)
        return (m + math.log(math.fsum(p[i] * math.exp(γz[i] - m) for i in range(len(p))))) / F.γ
    raise TypeError(f"Unknown risk measure {F!r}")


# ---------------------------------------------------------------------------
# Dual risk measures
# ---------------------------------------------------------------------------


def dual_risk_inner(
    F: AbstractRiskMeasure, Z: Sequence[float], p: Sequence[float]
) -> tuple[list[float], float]:
    """Return the worst-case probability vector ``q`` and the penalty ``α(p, q)``."""
    if isinstance(F, Expectation):
        # M(p) = {p}, α = 0.
        return list(p), 0.0
    if isinstance(F, WorstCase):
        # M(p) = P, α = 0: all the weight on the maximum outcome (first one on ties).
        q = [0.0] * len(Z)
        index = max(range(len(Z)), key=lambda i: Z[i])
        q[index] = 1.0
        return q, 0.0
    if isinstance(F, Entropic):
        # Closed form of Dowson, Morton & Pagnoncelli (2020):
        #   q_ω = p_ω e^{γ z_ω} / Σ_φ p_φ e^{γ z_φ},  α(p, q) = (1/γ) Σ q_ω log(q_ω / p_ω).
        # Same overflow protection as in `primal_risk`: shift the exponents by their max
        # (the shift cancels in the ratio).
        γz = [F.γ * z for z in Z]
        m = max(γz)
        peγz = [p[i] * math.exp(γz[i] - m) for i in range(len(p))]
        sum_peγz = math.fsum(peγz)
        q = [v / sum_peγz for v in peγz]
        # q log(q / p) -> 0 as q -> 0, so entries that underflowed to exactly 0 are skipped.
        α = math.fsum(q[i] * math.log(q[i] / p[i]) for i in range(len(q)) if q[i] > 0.0)
        return q, α / F.γ
    raise TypeError(f"Unknown risk measure {F!r}")


def dual_risk(F: AbstractRiskMeasure, Z: Sequence[float], p: Sequence[float]) -> float:
    """``F[Z] = sup_{q ∈ M(p)} E_q[Z] - α(p, q)`` at the ``q`` returned by ``dual_risk_inner``."""
    q, α = dual_risk_inner(F, Z, p)
    return math.fsum(q[i] * Z[i] for i in range(len(q))) - α


# ---------------------------------------------------------------------------
# Risk-averse subgradients
# ---------------------------------------------------------------------------


def dual_risk_averse_subgradient(
    V: Callable[[list[float], Any], float],
    λ: Callable[[list[float], Any], Sequence[float]] | None = None,
    *,
    F: AbstractRiskMeasure,
    Ω: Sequence[Any],
    p: Sequence[float],
    x̃: Sequence[float],
) -> list[float]:
    """``Σ_ω q*_ω λ(x̃, ω)``: the risk-averse subgradient theorem.

    ``λ(x, ω)`` is a subgradient of ``V(·, ω)``; by default it is computed by central
    finite differences (the page uses automatic differentiation).
    """
    x0 = [float(v) for v in x̃]
    if λ is None:

        def λ(x: list[float], ω: Any) -> Sequence[float]:
            return finite_difference_gradient(lambda y: V(y, ω), x)

    # Evaluate the function at x = x̃ for all ω ∈ Ω.
    V_ω = [V(x0, ω) for ω in Ω]
    # Solve the dual problem to obtain an optimal q*.
    q, _ = dual_risk_inner(F, V_ω, p)
    # The risk-averse subgradient is the expectation of the subgradients w.r.t. q*.
    dVdx = [0.0] * len(x0)
    for i, ω in enumerate(Ω):
        g = λ(x0, ω)
        for d in range(len(x0)):
            dVdx[d] += q[i] * float(g[d])
    return dVdx


def primal_risk_averse_subgradient(
    V: Callable[[list[float], Any], float],
    *,
    F: AbstractRiskMeasure,
    Ω: Sequence[Any],
    p: Sequence[float],
    x̃: Sequence[float],
) -> list[float]:
    """Finite-difference gradient of ``x ↦ primal_risk(F, [V(x, ω) for ω in Ω], p)``.

    On the page this returns ``NaN`` for ``Entropic(100)`` because ``exp(γ V)`` overflows
    inside ForwardDiff; our :func:`primal_risk` uses log-sum-exp so it does not. The
    lesson of the page stands: the dual form needs neither big floats nor a robust primal.
    """

    def inner(x: list[float]) -> float:
        return primal_risk(F, [V(x, ω) for ω in Ω], p)

    return finite_difference_gradient(inner, [float(v) for v in x̃])


def V_example(x: Sequence[float], ω: float) -> float:
    """The page's example ``V(x, ω) = ω x[1]^2``."""
    return ω * x[0] ** 2


Ω_EXAMPLE = [1.0, 2.0, 3.0]
P_EXAMPLE = [0.3, 0.4, 0.3]
X_TILDE = [3.0]

# ---------------------------------------------------------------------------
# Risk-averse decision rules, Part II: the implementation
# ---------------------------------------------------------------------------


def backward_pass(
    model: PolicyGraph,
    trajectory: Trajectory,
    io: TextIO | None = sys.stdout,
    *,
    risk_measure: AbstractRiskMeasure,
) -> None:
    _println(io, "| Backward pass")
    for index, outgoing_states in reversed(trajectory):
        node = get_node(model, index)
        _println(io, f"| | Visiting node {index}")
        if len(model.arcs[index - 1]) == 0:
            continue
        # New! Vectors storing each cut expression (as intercept + coefficients),
        # V(x, ω) and p:
        cut_expressions: list[tuple[float, dict[str, float]]] = []
        V_ω: list[float] = []
        p: list[float] = []
        for j, P_ij in model.arcs[index - 1].items():
            next_node = get_node(model, j)
            for k, v in outgoing_states.items():
                next_node.subproblem.fix(next_node.states[k].in_, v)
            for pφ, φ in zip(next_node.uncertainty.P, next_node.uncertainty.Ω):
                next_node.uncertainty.parameterize(φ)
                next_node.subproblem.optimize()
                V = next_node.subproblem.objective_value()
                dVdx = {
                    k: next_node.subproblem.reduced_cost(v.in_) for k, v in next_node.states.items()
                }
                # New! `V_j^K(x_k, φ) + dVdx_j^K(x'_k, φ)ᵀ(x - x_k)` as intercept + coefficients:
                cut_expressions.append((V - sum(dVdx[k] * outgoing_states[k] for k in dVdx), dVdx))
                # Add the objective value to Z and the probability to p:
                V_ω.append(V)
                p.append(P_ij * pφ)
        # New! Using the solutions in V_ω, compute q and α:
        q, α = dual_risk_inner(risk_measure, V_ω, p)
        _println(io, "| | | Z = ", V_ω)
        _println(io, "| | | p = ", p)
        _println(io, "| | | q = ", q)
        _println(io, "| | | α = ", α)
        # Then add the cut  θ >= Σ_i q_i cut_expressions[i] - α:
        intercept = math.fsum(q[i] * cut_expressions[i][0] for i in range(len(q))) - α
        coefficients = {
            k: math.fsum(q[i] * cut_expressions[i][1][k] for i in range(len(q)))
            for k in node.states
        }
        add_cut(node, intercept, coefficients, io)


def train(
    model: PolicyGraph,
    *,
    iteration_limit: int,
    replications: int,
    # New! A risk_measure argument.
    risk_measure: AbstractRiskMeasure,
    rng: random.Random | None = None,
    io: TextIO | None = sys.stdout,
) -> tuple[float, float]:
    rng = rng if rng is not None else random.Random()
    for i in range(1, iteration_limit + 1):
        _println(io, f"Starting iteration {i}")
        outgoing_states, _ = forward_pass(model, rng, io)
        # New! Pass the risk measure to the backward pass.
        backward_pass(model, outgoing_states, io, risk_measure=risk_measure)
        _println(io, "| Finished iteration")
        # With a risk measure other than Expectation this "lower bound" is just the
        # first-stage objective; it converges but is not a bound on the optimal policy.
        _println(io, "| | lower_bound = ", lower_bound(model))
    μ, tσ = upper_bound(model, replications=replications, rng=rng)
    _println(io, f"Upper bound = {μ} ± {tσ}")
    return μ, tσ


# ---------------------------------------------------------------------------
# Example: risk-averse hydro-thermal scheduling
# ---------------------------------------------------------------------------


def _hydro_thermal_builder(subproblem: Model, t: int) -> tuple[dict[str, State], Uncertainty]:
    volume_in = subproblem.add_variable("volume_in")
    subproblem.fix(volume_in, 200.0)
    volume_out = subproblem.add_variable("volume_out", lb=0.0, ub=200.0)
    states = {"volume": State(volume_in, volume_out)}
    thermal_generation = subproblem.add_variable("thermal_generation", lb=0.0)
    hydro_generation = subproblem.add_variable("hydro_generation", lb=0.0)
    hydro_spill = subproblem.add_variable("hydro_spill", lb=0.0)
    inflow = subproblem.add_variable("inflow")
    subproblem.add_constraint(
        volume_out == volume_in + inflow - hydro_generation - hydro_spill, name="balance"
    )
    subproblem.add_constraint(
        thermal_generation + hydro_generation == 150.0, name="demand_constraint"
    )
    fuel_cost = [50.0, 100.0, 150.0]
    subproblem.set_objective(fuel_cost[t - 1] * thermal_generation, Sense.MIN)
    uncertainty = Uncertainty(lambda ω: subproblem.fix(inflow, ω), [0.0, 50.0, 100.0], [1 / 3] * 3)
    return states, uncertainty


def build_risk_averse_model(optimizer: OptimizerFactory = HiGHS) -> PolicyGraph:
    """The same hydro-thermal problem as the "Introductory theory" page."""
    return build_policy_graph(
        _hydro_thermal_builder, graph=FINITE_GRAPH, lower_bound=0.0, optimizer=optimizer
    )


# ---------------------------------------------------------------------------
# main: reproduce the page
# ---------------------------------------------------------------------------


def main(seed: int = 1234, io: TextIO | None = sys.stdout) -> None:
    _println(io, "primal_risk(Expectation) = ", primal_risk(Expectation(), Z, p))
    _println(io, "primal_risk(WorstCase)   = ", primal_risk(WorstCase(), Z, p))
    for γ in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0]:
        _println(io, f"γ = {γ}, F[Z] = ", primal_risk(Entropic(γ), Z, p))
    _println(io, dual_risk(Expectation(), Z, p) == primal_risk(Expectation(), Z, p))
    _println(io, dual_risk(WorstCase(), Z, p) == primal_risk(WorstCase(), Z, p))
    for γ in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        primal = primal_risk(Entropic(γ), Z, p)
        dual = dual_risk(Entropic(γ), Z, p)
        success = "✓" if math.isclose(primal, dual, rel_tol=1e-8) else "×"
        _println(io, f"{success} γ = {γ}, primal = {primal}, dual = {dual}")
    kw: dict[str, Any] = {"Ω": Ω_EXAMPLE, "p": P_EXAMPLE, "x̃": X_TILDE}
    _println(io, dual_risk_averse_subgradient(V_example, F=Expectation(), **kw))
    _println(io, primal_risk_averse_subgradient(V_example, F=Expectation(), **kw))
    _println(io, dual_risk_averse_subgradient(V_example, F=WorstCase(), **kw))
    _println(io, primal_risk_averse_subgradient(V_example, F=WorstCase(), **kw))
    for γ in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        dual = dual_risk_averse_subgradient(V_example, F=Entropic(γ), **kw)
        primal = primal_risk_averse_subgradient(V_example, F=Entropic(γ), **kw)
        success = "✓" if math.isclose(primal[0], dual[0], rel_tol=1e-4) else "×"
        _println(io, f"{success} γ = {γ}, primal = {primal}, dual = {dual}")
    model = build_risk_averse_model()
    train(
        model,
        iteration_limit=3,
        replications=100,
        risk_measure=Entropic(1.0),
        rng=random.Random(seed),
        io=io,
    )
    _println(
        io,
        evaluate_policy(model, node=1, incoming_state={"volume": 150.0}, random_variable=75),
    )


if __name__ == "__main__":
    main()
