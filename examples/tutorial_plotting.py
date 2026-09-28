"""Plotting tools (https://sddp.dev/stable/tutorial/plotting/).

A faithful port of ``docs/src/tutorial/plotting.jl``: the Markovian hydro-thermal model,
20 training iterations, 100 simulations, the interactive spaghetti plot, matplotlib
publication plots, the value function of node ``(1, 1)`` and the graph plot.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from typing import Any

import sddp

Realization = namedtuple("Realization", ["inflow", "fuel_multiplier"])

Omega = [
    Realization(inflow=0.0, fuel_multiplier=1.5),
    Realization(inflow=50.0, fuel_multiplier=1.0),
    Realization(inflow=100.0, fuel_multiplier=0.75),
]

TRANSITION_MATRICES = [[[1.0]], [[0.75, 0.25]], [[0.75, 0.25], [0.25, 0.75]]]


def build_model() -> sddp.PolicyGraph:
    """The model of the Markovian policy graphs tutorial."""

    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, markov_state = node
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal_generation = sp.add_variable("thermal_generation", lb=0.0)
        hydro_generation = sp.add_variable("hydro_generation", lb=0.0)
        hydro_spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.add_constraint(volume.out == volume.in_ + inflow - hydro_generation - hydro_spill)
        sp.add_constraint(thermal_generation + hydro_generation == 150.0)
        probability = [1 / 6, 1 / 3, 1 / 2] if markov_state == 1 else [1 / 2, 1 / 3, 1 / 6]
        fuel_cost = [50.0, 100.0, 150.0]

        @sp.parameterize(Omega, probability)
        def _(w: Realization) -> None:
            sp.fix(inflow, w.inflow)
            sp.set_stage_objective(w.fuel_multiplier * fuel_cost[t - 1] * thermal_generation)

    return sddp.MarkovianPolicyGraph(
        builder,
        transition_matrices=TRANSITION_MATRICES,
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


SIM_VARIABLES = ["volume", "thermal_generation", "hydro_generation", "hydro_spill"]


def train_and_simulate(
    seed: int | None = 1, iteration_limit: int = 20, replications: int = 100, **kwargs: Any
) -> tuple[sddp.PolicyGraph, list[list[dict[str, Any]]]]:
    model = build_model()
    sddp.train(
        model,
        iteration_limit=iteration_limit,
        run_numerical_stability_report=False,
        seed=seed,
        **kwargs,
    )
    simulations = sddp.simulate(model, replications, SIM_VARIABLES, seed=seed)
    return model, simulations


def fuel_cost(data: dict[str, Any]) -> float:
    """The second spaghetti of the page: stage objective per unit of thermal generation."""
    if data["thermal_generation"] > 0:
        return data["stage_objective"] / data["thermal_generation"]
    return 0.0  # No thermal generation, so return 0.0.


def spaghetti(simulations: Sequence[Sequence[dict[str, Any]]], filename: str) -> sddp.SpaghettiPlot:
    plt = sddp.SpaghettiPlot(simulations)
    plt.add_spaghetti(lambda data: data["volume"].out, title="Reservoir volume")
    plt.add_spaghetti(fuel_cost, title="Fuel cost", ymin=0, ymax=250)
    plt.plot(filename)
    return plt


def publication_plots(simulations: Sequence[Sequence[dict[str, Any]]], filename: str) -> Any:
    """Outgoing volume and thermal generation side by side (``layout = (1, 2)``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sddp.publication_plot(
        simulations, lambda data: data["volume"].out, ax=axes[0], title="Outgoing volume"
    )
    sddp.publication_plot(
        simulations,
        lambda data: data["thermal_generation"],
        ax=axes[1],
        title="Thermal generation",
    )
    for ax in axes:
        ax.set_xlabel("Stage")
        ax.set_ylim(0, 200)
    fig.savefig(filename)
    plt.close(fig)
    return fig


def value_function(model: sddp.PolicyGraph, filename: str | None = None) -> tuple[float, dict]:
    V = sddp.ValueFunction(model, node=(1, 1))
    height, subgradient = sddp.evaluate_value_function(V, {"volume": 1})
    if filename is not None:
        sddp.plot_value_function_html(V, filename, volume=list(range(0, 201)))
    return height, subgradient


def main() -> None:
    model, simulations = train_and_simulate(seed=1)
    print(f"Completed {len(simulations)} simulations.")
    sddp.plot_graph(model, "model_plotting.html")
    plt = spaghetti(simulations, "spaghetti_plot.html")
    print(plt)
    try:
        publication_plots(simulations, "publication_plot.png")
    except ImportError:
        pass
    height, subgradient = value_function(model, "value_function.html")
    print("V(volume = 1) =", height, subgradient)


if __name__ == "__main__":
    main()
