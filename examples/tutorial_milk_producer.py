"""Python translation of the sddp.dev tutorial "Example: the milk producer"
(docs/src/tutorial/example_milk_producer.jl): fitting a Markovian policy graph to a
univariate price process with ``markovian_graph_from_simulator`` and training / simulating
with ``SimulatorSamplingScheme``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import sddp

# ------------------------------------------------------------- price process
RESIDUALS = [0.0987, 0.199, 0.303, 0.412, 0.530, 0.661, 0.814, 1.010, 1.290]
RESIDUALS = [0.1 * r for r in [-x for x in RESIDUALS] + [0.0] + RESIDUALS]


def make_simulator(rng: random.Random | None = None) -> Callable[[], list[float]]:
    """A multiplicative AR(1) spot-price simulator with empirical residuals (12 months,
    mean-reverting to 6 $/kg, clamped to [3, 9])."""
    rng = rng if rng is not None else random.Random()

    def simulator() -> list[float]:
        scenario = [0.0] * 12
        y, mu, alpha = 4.5, 6.0, 0.05
        for t in range(12):
            y = math.exp((1 - alpha) * math.log(y) + alpha * math.log(mu) + rng.choice(RESIDUALS))
            scenario[t] = min(max(y, 3.0), 9.0)
        return scenario

    return simulator


simulator = make_simulator()


def build_graph(
    simulator: Callable[[], Sequence[float]] = simulator, budget: int = 30, scenarios: int = 10_000
) -> sddp.Graph:
    """``SDDP.MarkovianGraph(simulator; budget = 30, scenarios = 10_000)``."""
    return sddp.markovian_graph_from_simulator(simulator, budget=budget, scenarios=scenarios)


def plot_price_process(
    simulator: Callable[[], Sequence[float]] = simulator,
    graph: sddp.Graph | None = None,
    n: int = 500,
    filename: str | None = None,
) -> Any:
    """500 grey simulations of the price process, with the fitted graph's arcs drawn in red
    (line width proportional to the transition probability) when ``graph`` is given."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    months = list(range(1, 13))
    for _ in range(n):
        ax.plot(months, list(simulator()), color="gray", alpha=0.2)
    if graph is not None:
        for (t, price), edges in graph.nodes.items():
            for (t2, price2), probability in edges:
                ax.plot([t, t2], [price, price2], color="red", linewidth=3 * probability)
    ax.set_xlabel("Month")
    ax.set_ylabel("Price [$/kg]")
    ax.set_xlim(1, 12)
    ax.set_ylim(3, 9)
    if filename is not None:
        fig.savefig(filename)
    return fig


# --------------------------------------------------------------------- model
OMEGA_PRODUCTION = [0.1 + 0.025 * i for i in range(5)]  # range(0.1, 0.2; length = 5)


def build_milk_producer(graph: sddp.Graph) -> sddp.PolicyGraph:
    """The milk-producer model on a simulator-fitted Markovian graph. Noise terms are
    ``(price, production)`` tuples (the first element must be the Markov state)."""

    def builder(sp: sddp.Subproblem, node: tuple[int, float]) -> None:
        # Decompose the node into the month (int) and spot price (float)
        t, price = node
        # Transactions on the futures market cost 0.01
        c_transaction = 0.01
        # It costs the company +50% to buy milk on the spot market and deliver to customers
        c_buy_premium = 1.5
        # Buyer is willing to pay +5% for certainty
        c_contango = 1.05
        # Distribution of production
        c_max_production = 12 * max(OMEGA_PRODUCTION)
        # x_stock: quantity of milk in stock pile
        x_stock = sp.add_state("x_stock", lb=0.0, initial_value=0.0)
        # x_forward[i]: quantity of milk for delivery in i months
        x_forward = [
            sp.add_state(f"x_forward[{i}]", lb=0.0, initial_value=0.0) for i in range(1, 5)
        ]
        # u_spot_sell: quantity of milk to sell on spot market
        u_spot_sell = sp.add_variable("u_spot_sell", lb=0.0, ub=c_max_production)
        # u_spot_buy: quantity of milk to buy on spot market
        u_spot_buy = sp.add_variable("u_spot_buy", lb=0.0, ub=c_max_production)
        # u_forward_sell: quantity of milk to sell on futures market
        c_max_futures = c_max_production if t <= 8 else 0.0
        u_forward_sell = sp.add_variable("u_forward_sell", lb=0.0, ub=c_max_futures)
        # omega_production: production random variable
        omega_production = sp.add_variable("omega_production")
        # Forward contracting constraints:
        for i in range(3):
            sp.add_constraint(x_forward[i].out == x_forward[i + 1].in_)
        sp.add_constraint(x_forward[3].out == u_forward_sell)
        # Stockpile balance constraint
        sp.add_constraint(
            x_stock.out
            == x_stock.in_ + omega_production + u_spot_buy - x_forward[0].in_ - u_spot_sell
        )
        # The random variables. `price` comes from the Markov node. The elements of Ω MUST be
        # tuples with 1 or 2 values: (price,) or (price, production).
        Omega = [(price, p) for p in OMEGA_PRODUCTION]

        @sp.parameterize(Omega)
        def _(w: tuple[float, float]) -> None:
            # Fix the omega_production variable
            sp.fix(omega_production, w[1])
            sp.set_stage_objective(
                # Sales on spot market
                w[0] * (u_spot_sell - c_buy_premium * u_spot_buy)
                # Sales on futures market
                + (w[0] * c_contango - c_transaction) * u_forward_sell
            )

    return sddp.PolicyGraph(builder, graph, sense="Max", upper_bound=1e2, optimizer=sddp.HiGHS)


RISK_MEASURE = 0.5 * sddp.Expectation() + 0.5 * sddp.AVaR(0.25)


def train_milk_producer(
    model: sddp.PolicyGraph,
    simulator: Callable[[], Sequence[float]] = simulator,
    *,
    iteration_limit: int | None = None,
    time_limit: float | None = 20.0,
    **kwargs: Any,
) -> None:
    """Train with the page's risk measure and ``SimulatorSamplingScheme``. The page uses
    ``time_limit = 20``; pass ``iteration_limit`` for a reproducible run."""
    sddp.train(
        model,
        iteration_limit=iteration_limit,
        time_limit=None if iteration_limit is not None else time_limit,
        risk_measure=RISK_MEASURE,
        sampling_scheme=sddp.SimulatorSamplingScheme(simulator),
        **kwargs,
    )


def simulate_milk_producer(
    model: sddp.PolicyGraph,
    simulator: Callable[[], Sequence[float]] = simulator,
    replications: int = 200,
    **kwargs: Any,
) -> list[list[dict[str, Any]]]:
    return sddp.simulate(
        model,
        replications,
        ["x_stock", "u_forward_sell", "u_spot_sell", "u_spot_buy"],
        sampling_scheme=sddp.SimulatorSamplingScheme(simulator),
        **kwargs,
    )


def plot_policy(
    simulations: Sequence[Sequence[dict[str, Any]]], filename: str | None = None
) -> Any:
    """The ``(2, 2)`` publication plot of the page."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2)
    panels = [
        ("x_stock.out", lambda d: d["x_stock"].out),
        ("u_forward_sell", lambda d: d["u_forward_sell"]),
        ("u_spot_buy", lambda d: d["u_spot_buy"]),
        ("u_spot_sell", lambda d: d["u_spot_sell"]),
    ]
    for ax, (title, fn) in zip(axes.flat, panels):
        sddp.publication_plot(simulations, fn, ax=ax, title=title)
    if filename is not None:
        fig.savefig(filename)
    return fig


def main() -> None:
    print(simulator())
    graph = build_graph()
    print(f"graph with {len(graph.nodes) - 1} nodes")
    model = build_milk_producer(graph)
    train_milk_producer(model)
    simulations = simulate_milk_producer(model)
    print("node_index at stage 12:", simulations[0][11]["node_index"])
    print("noise_term at stage 12:", simulations[0][11]["noise_term"])


if __name__ == "__main__":
    main()
