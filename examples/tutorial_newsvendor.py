"""Example: two-stage newsvendor (https://sddp.dev/stable/tutorial/example_newsvendor/).

A faithful port of ``docs/src/tutorial/example_newsvendor.jl``: Kelley's cutting-plane
algorithm on a concave function, a hand-written L-shaped method, the same problem as an
SDDP.jl policy graph, and the risk-averse variants.

The demand sample ``d`` is drawn from ``Triangular(150, 250, 200)`` with a fixed numpy seed;
``reference/generate_tutorials_c.jl`` embeds the very same vector so both languages use
identical data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

import sddp
from sddp.solver.model import Model, Sense

# ----------------------------------------------------------------------------- data
N = 100
SEED = 20240611


def demand_sample(n: int = N, seed: int = SEED) -> list[float]:
    """``sort!(rand(TriangularDist(150, 250, 200), N))`` with a fixed numpy seed."""
    rng = np.random.default_rng(seed)
    return sorted(float(v) for v in rng.triangular(150.0, 200.0, 250.0, n))


d = demand_sample()
Omega = list(range(N))
P = [1.0 / N] * N


# --------------------------------------------------------- Kelley's cutting plane
def finite_difference_gradient(
    f: Callable[[np.ndarray], float], h: float = 1e-6
) -> Callable[[np.ndarray], np.ndarray]:
    """Central finite differences (the Julia page uses ForwardDiff)."""

    def grad(x: np.ndarray) -> np.ndarray:
        g = np.zeros(len(x))
        for i in range(len(x)):
            e = np.zeros(len(x))
            e[i] = h
            g[i] = (f(x + e) - f(x - e)) / (2 * h)
        return g

    return grad


def kelleys_cutting_plane(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    input_dimension: int,
    upper_bound: float,
    iteration_limit: int,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Kelley's cutting-plane algorithm for maximising the concave function ``f``."""
    if grad_f is None:
        grad_f = finite_difference_gradient(f)
    K = 1
    model = Model(sddp.HiGHS)
    theta = model.add_variable("theta", ub=upper_bound)
    x = [model.add_variable(f"x[{i + 1}]") for i in range(input_dimension)]
    model.set_objective(1.0 * theta, Sense.MAX)
    x_k = np.full(input_dimension, math.nan)
    lower_bound, upper_bound = -math.inf, math.inf
    status = ""
    while True:
        model.optimize()
        x_k = np.array([model.value(xi) for xi in x])
        upper_bound = model.objective_value()
        lower_bound = min(upper_bound, f(x_k))
        if verbose:
            print(f"K = {K} : {lower_bound} <= f(x*) <= {upper_bound}")
        g = grad_f(x_k)
        model.add_constraint(
            theta <= f(x_k) + sum(float(g[i]) * (x[i] - float(x_k[i])) for i in range(len(x)))
        )
        K += 1
        if K > iteration_limit:
            status = "iteration limit"
            break
        elif abs(upper_bound - lower_bound) < tolerance:
            status = "converged"
            break
    if verbose:
        print(f"-- Termination status: {status} --")
        print("Found solution: x_K = ", list(x_k))
    return {
        "x": [float(v) for v in x_k],
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "iterations": K - 1,
        "status": status,
    }


def kelley_example_function(x: np.ndarray) -> float:
    return -((x[0] - 1) ** 2) + -((x[1] + 2) ** 2) + 1.0


def kelley_example_gradient(x: np.ndarray) -> np.ndarray:
    return np.array([-2 * (x[0] - 1), -2 * (x[1] + 2)])


def run_kelley(verbose: bool = True) -> dict:
    return kelleys_cutting_plane(
        kelley_example_function,
        kelley_example_gradient,
        input_dimension=2,
        upper_bound=10.0,
        iteration_limit=20,
        verbose=verbose,
    )


# -------------------------------------------------------------- L-shaped method
def solve_second_stage(x_bar: float, d_omega: float) -> dict:
    """``V_2(x̄, d_ω)`` and the reduced cost of the fixed incoming variable."""
    model = Model(sddp.HiGHS)
    x_in = model.add_variable("x_in")
    x_out = model.add_variable("x_out", lb=0.0)
    model.fix(x_in, x_bar)
    u_sell = model.add_variable("u_sell", lb=0.0, ub=d_omega)
    model.add_constraint(x_out == x_in - u_sell)
    model.add_constraint(u_sell <= x_in)
    model.set_objective(5 * u_sell - 0.1 * x_out, Sense.MAX)
    model.optimize()
    return {
        "V": model.objective_value(),
        "lambda": model.reduced_cost(x_in),
        "x": model.value(x_out),
        "u": model.value(u_sell),
    }


def l_shaped(
    d: Sequence[float] = d,
    P: Sequence[float] = P,
    iteration_limit: int = 100,
    verbose: bool = True,
) -> dict:
    """The hand-written L-shaped method of the page; returns the first-stage solution."""
    model = Model(sddp.HiGHS)
    x_in = model.add_variable("x_in")
    model.fix(x_in, 0.0)
    x_out = model.add_variable("x_out", lb=0.0)
    u_make = model.add_variable("u_make", lb=0.0)
    model.add_constraint(x_out == x_in + u_make)
    M = 5 * max(d)
    theta = model.add_variable("theta", ub=M)
    model.set_objective(-2 * u_make + theta, Sense.MAX)
    iterations = 0
    ub = lb = math.nan
    for k in range(1, iteration_limit + 1):
        iterations = k
        if verbose:
            print(f"Solving iteration k = {k}")
        model.optimize()
        x_k = model.value(x_out)
        if verbose:
            print(f"  xᵏ = {x_k}")
        ub = model.objective_value()
        if verbose:
            print(f"  V̅ = {ub}")
        ret = [solve_second_stage(x_k, d[w]) for w in range(len(d))]
        lb = model.value(-2 * u_make) + sum(p * r["V"] for p, r in zip(P, ret))
        if verbose:
            print(f"  V̲ = {lb}")
        if ub - lb < 1e-6:
            if verbose:
                print("Terminating with near-optimal solution")
            break
        model.add_constraint(
            theta <= sum(p * (r["V"] + r["lambda"] * (x_out - x_k)) for p, r in zip(P, ret))
        )
    model.optimize()
    x_k = model.value(x_out)
    return {
        "x": x_k,
        "objective": model.objective_value(),
        "lower_bound": lb,
        "upper_bound": ub,
        "iterations": iterations,
    }


# --------------------------------------------------------------- Policy graph
def build_newsvendor(d: Sequence[float] = d, P: Sequence[float] = P) -> sddp.PolicyGraph:
    """The two-stage newsvendor as an SDDP.jl policy graph (with ``u_make``)."""

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=0.0)
        if stage == 1:
            u_make = sp.add_variable("u_make", lb=0.0)
            sp.add_constraint(x.out == x.in_ + u_make)
            sp.set_stage_objective(-2 * u_make)
        else:
            u_sell = sp.add_variable("u_sell", lb=0.0)
            sp.add_constraint(u_sell <= x.in_)
            sp.add_constraint(x.out == x.in_ - u_sell)
            sp.parameterize(lambda w: sp.set_upper_bound(u_sell, w), list(d), list(P))
            sp.set_stage_objective(5 * u_sell - 0.1 * x.out)

    return sddp.LinearPolicyGraph(
        builder, stages=2, sense="Max", upper_bound=5 * max(d), optimizer=sddp.HiGHS
    )


def build_newsvendor_risk(d: Sequence[float] = d, P: Sequence[float] = P) -> sddp.PolicyGraph:
    """The variant used by ``solve_newsvendor`` (stage objective ``-2 * x.out``)."""

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=0.0)
        if node == 1:
            sp.set_stage_objective(-2 * x.out)
        else:
            u_sell = sp.add_variable("u_sell", lb=0.0)
            sp.add_constraint(u_sell <= x.in_)
            sp.add_constraint(x.out == x.in_ - u_sell)
            sp.parameterize(lambda w: sp.set_upper_bound(u_sell, w), list(d), list(P))
            sp.set_stage_objective(5 * u_sell - 0.1 * x.out)

    return sddp.LinearPolicyGraph(
        builder, stages=2, sense="Max", upper_bound=5 * max(d), optimizer=sddp.HiGHS
    )


def solve_newsvendor_details(
    risk_measure: sddp.plugins.risk_measures.RiskMeasure,
    seed: int | None = None,
    **train_kwargs: object,
) -> dict:
    """Like :func:`solve_newsvendor` but also returns the bound and the iteration count."""
    model = build_newsvendor_risk()
    sddp.train(model, risk_measure=risk_measure, print_level=0, seed=seed, **train_kwargs)
    first_stage_rule = sddp.DecisionRule(model, node=1)
    solution = sddp.evaluate(first_stage_rule, incoming_state={"x": 0.0})
    assert model.most_recent_training_results is not None
    return {
        "x": solution.outgoing_state["x"],
        "bound": sddp.calculate_bound(model),
        "iterations": len(model.most_recent_training_results.log),
    }


def solve_newsvendor(
    risk_measure: sddp.plugins.risk_measures.RiskMeasure,
    seed: int | None = None,
    **train_kwargs: object,
) -> float:
    """Train with ``risk_measure`` and return the first-stage order quantity ``x``."""
    return solve_newsvendor_details(risk_measure, seed=seed, **train_kwargs)["x"]


def entropic_gammas() -> list[float]:
    """``Γ = [10^i for i in -4:0.5:1]``."""
    return [10.0**i for i in np.arange(-4.0, 1.0 + 1e-9, 0.5)]


def main() -> None:
    print("Kelley's cutting plane:")
    run_kelley()
    print("\nsolve_second_stage(200, 170) =", solve_second_stage(200, 170))
    res = l_shaped()
    print("L-shaped first-stage solution:", res["x"])
    print("second stage at 170:", solve_second_stage(res["x"], 170.0))

    model = build_newsvendor()
    sddp.plot_graph(model, "model_newsvendor.html")
    sddp.train(model, log_every_iteration=True, seed=1)
    first_stage_rule = sddp.DecisionRule(model, node=1)
    solution_1 = sddp.evaluate(first_stage_rule, incoming_state={"x": 0.0})
    print(solution_1)
    second_stage_rule = sddp.DecisionRule(model, node=2)
    solution = sddp.evaluate(
        second_stage_rule,
        incoming_state={"x": solution_1.outgoing_state["x"]},
        noise=170.0,
        controls_to_record=["u_sell"],
    )
    print(solution)
    simulations = sddp.simulate(
        model, 10, ["x", "u_sell", "u_make"], skip_undefined_variables=True, seed=1
    )
    print(len(simulations), len(simulations[0]))
    print(simulations[0][0])
    print(simulations[0][1])
    objectives = [sum(data["stage_objective"] for data in sim) for sim in simulations]
    mu, t = sddp.confidence_interval(objectives)
    print(f"Simulation ci : {mu} ± {t}")

    print(0.5 * sddp.Expectation() + 0.5 * sddp.WorstCase())
    print("CVaR(0.4):", solve_newsvendor(sddp.CVaR(0.4), seed=1))
    print("WorstCase:", solve_newsvendor(sddp.WorstCase(), seed=1))
    gammas = entropic_gammas()
    buy = [solve_newsvendor(sddp.Entropic(g), seed=1) for g in gammas]
    for g, b in zip(gammas, buy):
        print(f"  γ = {g:10.5g}  buy = {b}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.semilogx(gammas, buy)
        plt.xlabel("Risk aversion parameter γ")
        plt.ylabel("Number of pies to make")
        plt.savefig("newsvendor_entropic.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
