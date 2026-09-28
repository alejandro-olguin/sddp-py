"""Duality handlers (https://sddp.dev/stable/tutorial/duality_handlers/).

A faithful port of ``docs/src/tutorial/duality_handlers.jl``: four trivial two-stage
mixed-integer models that expose the strengths and weaknesses of every duality handler.
As on the page, HiGHS presolve is turned off.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sddp

Optimizer = sddp.HiGHS.with_options(presolve="off")


def train_and_evaluate_bounds(
    model_fn: Callable[[], sddp.PolicyGraph],
    duality_handler: Any,
    print_level: int = 0,
    seed: int | None = 1,
    **kwargs: Any,
) -> tuple[float, float]:
    """Train a fresh model with ``duality_handler``; return ``(lower_bound, upper_bound)``.

    The models are deterministic, so a single simulation evaluates the upper bound.
    """
    model = model_fn()
    sddp.train(
        model,
        print_level=print_level,
        duality_handler=duality_handler,
        seed=seed,
        run_numerical_stability_report=False,
        **kwargs,
    )
    simulations = sddp.simulate(model, 1, seed=seed)
    lower_bound = sddp.calculate_bound(model)
    print(f"lower_bound: {lower_bound}")
    (simulation,) = simulations
    upper_bound = sum(data["stage_objective"] for data in simulation)
    print(f"upper_bound: {upper_bound}")
    return lower_bound, upper_bound


def model_1() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", binary=True, initial_value=1.0)
        y = sp.add_variable("y", binary=True)
        sp.add_constraint(x.out == x.in_)
        if t == 1:
            sp.set_stage_objective(x.out)
        else:
            sp.set_stage_objective(y)
            sp.add_constraint(y >= x.in_ - 0.5)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0, optimizer=Optimizer)


def model_2() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=0.1)
        y = sp.add_variable("y", integer=True)
        sp.add_constraint(x.out == x.in_)
        if t == 1:
            sp.set_stage_objective(x.out)
        else:
            sp.set_stage_objective(y)
            sp.add_constraint(y >= x.in_ + 0.1)
            sp.add_constraint(y >= -x.in_ + 0.1)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0, optimizer=Optimizer)


def model_3() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=-1.0, ub=0.5, initial_value=0.0)
        y = sp.add_variable("y")
        sp.set_stage_objective(y)
        if t == 1:
            sp.add_constraint(y >= x.out)
            sp.add_constraint(y >= -x.out)
        else:
            z = sp.add_variable("z", binary=True)
            sp.add_constraint(y >= 1 - x.in_ - 3 * z)
            sp.add_constraint(y >= 1 + x.in_ - 3 * (1 - z))

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=-1.0, optimizer=Optimizer)


def model_4() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=-1.0, ub=0.5, initial_value=0.0)
        y = sp.add_variable("y")
        z = sp.add_variable("z", binary=True)
        if t == 1:
            sp.set_stage_objective(-0.1 * x.out)
        else:
            sp.set_stage_objective(y)
            sp.add_constraint(y >= 1 - x.in_ - 3 * z)
            sp.add_constraint(y >= 1 + x.in_ - 3 * (1 - z))

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=-1.0, optimizer=Optimizer)


MODELS: dict[str, Callable[[], sddp.PolicyGraph]] = {
    "model_1": model_1,
    "model_2": model_2,
    "model_3": model_3,
    "model_4": model_4,
}


def bandit_duality() -> sddp.BanditDuality:
    return sddp.BanditDuality(
        sddp.ContinuousConicDuality(),
        sddp.StrengthenedConicDuality(),
        sddp.LagrangianDuality(),
    )


def handler_by_name(name: str) -> Any:
    """The duality handler for a short key (used by the tests and the oracle)."""
    return {
        "continuous": sddp.ContinuousConicDuality,
        "strengthened": sddp.StrengthenedConicDuality,
        "lagrangian": sddp.LagrangianDuality,
        "bandit": bandit_duality,
        "fixed_discrete": sddp.FixedDiscreteDuality,
    }[name]()


# Every (model, handler) combination shown on the page, in page order.
PAGE_COMBINATIONS: list[tuple[str, str]] = [
    ("model_1", "continuous"),
    ("model_1", "strengthened"),
    ("model_2", "continuous"),
    ("model_2", "strengthened"),
    ("model_1", "lagrangian"),
    ("model_2", "lagrangian"),
    ("model_3", "lagrangian"),
    ("model_3", "strengthened"),
    ("model_4", "continuous"),
    ("model_4", "lagrangian"),
    ("model_1", "bandit"),
    ("model_2", "bandit"),
    ("model_3", "bandit"),
    ("model_2", "fixed_discrete"),
    ("model_2", "strengthened"),
    ("model_1", "fixed_discrete"),
    ("model_1", "strengthened"),
]


def main() -> None:
    for model_name, handler_name in PAGE_COMBINATIONS:
        print(f"-- {model_name} with {handler_name}")
        train_and_evaluate_bounds(MODELS[model_name], handler_by_name(handler_name))


if __name__ == "__main__":
    main()
