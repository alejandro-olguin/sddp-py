"""Python translations of six sddp.dev tutorial pages (docs/src/tutorial):

* ``warnings.jl``            -- Words of warning
* ``arma.jl``                -- Auto-regressive stochastic processes
* ``decision_hazard.jl``     -- Here-and-now and hazard-decision
* ``production_planning.jl`` -- Example: production planning
* ``batteries.jl``           -- Example: batteries
* ``inventory.jl``           -- Example: inventory management

Each model lives in its own ``build_*`` function (a fresh model per call); ``run_*`` functions
reproduce what the page does with the model, and :func:`main` reproduces the pages' printed
numbers. ``tests/test_tutorial_modelling.py`` checks the models against SDDP.jl
(``reference/oracle/tutorials_a.json``).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import sddp

# =============================================================================
# Words of warning (tutorial/warnings.jl)
# =============================================================================


def build_relatively_complete_recourse() -> sddp.PolicyGraph:
    """A model that violates relatively complete recourse: ``x.out = 0`` is feasible in stage 1
    but stage 2 requires ``x.in >= 1``. Training raises an infeasibility error."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=1.0)
        if t == 2:
            sp.add_constraint(x.in_ >= 1)
        sp.set_stage_objective(x.out)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0, optimizer=sddp.HiGHS)


def build_badly_scaled() -> sddp.PolicyGraph:
    """The badly scaled model used to demonstrate ``numerical_stability_report``."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=-1e7, initial_value=1e-5)
        sp.add_constraint(1e9 * x.out >= 1e-6 * x.in_ + 1e-8)
        sp.set_stage_objective(1e9 * x.out)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=-1e10)


def build_initial_bound_model(lower_bound: float) -> sddp.PolicyGraph:
    """The deterministic 3-stage model of the "Choosing an initial bound" section.

    With ``lower_bound=0.0`` the bound converges to 3.5 (the true optimum); with
    ``lower_bound=10.0`` it converges to 11.0 because the bound cuts off the feasible region.
    """

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=2.0)
        u = sp.add_variable("u", lb=0.0)
        v = sp.add_variable("v", lb=0.0)
        sp.add_constraint(x.out == x.in_ - u)
        sp.add_constraint(u + v == 1.5)
        sp.set_stage_objective(t * v)

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=lower_bound, optimizer=sddp.HiGHS
    )


def run_warnings(print_level: int = 1) -> dict[str, Any]:
    """Reproduce the page: the infeasibility error, the stability report and the two bounds."""
    out: dict[str, Any] = {}
    model = build_relatively_complete_recourse()
    try:
        sddp.train(model, iteration_limit=1, print_level=0)
    except RuntimeError as err:
        out["error"] = str(err)
        if print_level > 0:
            print(err)
    out["report"] = sddp.numerical_stability_report(build_badly_scaled(), print=print_level > 0)
    for lb in (0.0, 10.0):
        model = build_initial_bound_model(lb)
        sddp.train(
            model,
            iteration_limit=5,
            run_numerical_stability_report=False,
            print_level=print_level,
        )
        out[f"bound_lb_{lb:g}"] = sddp.calculate_bound(model)
    return out


# =============================================================================
# Auto-regressive stochastic processes (tutorial/arma.jl)
# =============================================================================
ARMA_OMEGA = [-10.0, 0.1, 9.6]
ARMA_COST = [50, 100, 150]


def build_state_space_expansion() -> sddp.PolicyGraph:
    """``inflow_t = inflow_{t-1} + eps`` modelled with ``inflow`` as a state variable."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=200.0, initial_value=200.0)
        g_t = sp.add_variable("g_t", lb=0.0)
        g_h = sp.add_variable("g_h", lb=0.0)
        s = sp.add_variable("s", lb=0.0)
        sp.add_constraint(g_h + g_t == 150)
        sp.set_stage_objective(ARMA_COST[t - 1] * g_t)
        # Add inflow as a state
        inflow = sp.add_state("inflow", initial_value=50.0)
        # Add the random variable as a control variable
        eps = sp.add_variable("ε")
        # The equation describing our statistical model
        sp.add_constraint(inflow.out == inflow.in_ + eps)
        # The new water balance constraint using the state variable
        sp.add_constraint(x.out == x.in_ - g_h - s + inflow.out)
        # Assume we have some empirical residuals:
        sp.parameterize(lambda w: sp.fix(eps, w), ARMA_OMEGA)

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def make_simulator(rng: random.Random | None = None) -> Callable[[], list[float]]:
    """The page's ``simulator()``: a 3-stage random walk starting at 50 with steps from
    ``ARMA_OMEGA``. ``rng`` defaults to the module-level ``random`` generator, like Julia's
    global RNG."""
    r: Any = rng if rng is not None else random

    def simulator() -> list[float]:
        inflow = [0.0] * 3
        current = 50.0
        for t in range(3):
            current += r.choice(ARMA_OMEGA)
            inflow[t] = current
        return inflow

    return simulator


simulator = make_simulator()


def build_markov_chain_graph(
    simulator: Callable[[], Sequence[float]] = simulator, budget: int = 8, scenarios: int = 30
) -> sddp.Graph:
    """``SDDP.MarkovianGraph(simulator; budget = 8, scenarios = 30)``."""
    return sddp.markovian_graph_from_simulator(simulator, budget=budget, scenarios=scenarios)


def build_markov_chain_model(graph: sddp.Graph) -> sddp.PolicyGraph:
    """The hydro-thermal model on a simulator-fitted Markov chain; nodes are ``(t, inflow)``."""

    def builder(sp: sddp.Subproblem, node: tuple[int, float]) -> None:
        t, inflow = node
        x = sp.add_state("x", lb=0.0, ub=200.0, initial_value=200.0)
        g_t = sp.add_variable("g_t", lb=0.0)
        g_h = sp.add_variable("g_h", lb=0.0)
        s = sp.add_variable("s", lb=0.0)
        sp.add_constraint(g_h + g_t == 150)
        sp.set_stage_objective(ARMA_COST[t - 1] * g_t)
        # The new water balance constraint using the node:
        sp.add_constraint(x.out == x.in_ - g_h - s + inflow)

    return sddp.PolicyGraph(builder, graph, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS)


def build_var_model() -> sddp.PolicyGraph:
    """A 2-dimensional vector auto-regressive inflow model ``inflow_t = A inflow_{t-1} + eps``."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=200.0, initial_value=200.0)
        g_t = sp.add_variable("g_t", lb=0.0)
        g_h = sp.add_variable("g_h", lb=0.0)
        s = sp.add_variable("s", lb=0.0)
        sp.add_constraint(g_h + g_t == 150)
        sp.set_stage_objective(ARMA_COST[t - 1] * g_t)
        # Add inflow as a (vector) state
        inflow = [sp.add_state(f"inflow[{i}]", initial_value=50.0) for i in (1, 2)]
        # Add the random variable as a control variable
        eps = [sp.add_variable(f"ε[{i}]") for i in (1, 2)]
        # The equation describing our statistical model
        A = [[0.8, 0.2], [0.2, 0.8]]
        for i in range(2):
            sp.add_constraint(
                inflow[i].out == sum(A[i][j] * inflow[j].in_ for j in range(2)) + eps[i]
            )
        # The new water balance constraint using the state variable
        sp.add_constraint(x.out == x.in_ - g_h - s + inflow[0].out + inflow[1].out)
        # Assume we have some empirical residuals:
        omega = [(w1, w2) for w1 in ARMA_OMEGA for w2 in ARMA_OMEGA]

        def modify(w: tuple[float, float]) -> None:
            sp.fix(eps[0], w[0])
            sp.fix(eps[1], w[1])

        sp.parameterize(modify, omega)

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


# =============================================================================
# Here-and-now and hazard-decision (tutorial/decision_hazard.jl)
# =============================================================================


def build_hazard_decision() -> sddp.PolicyGraph:
    """Wait-and-see: ``u_thermal`` is decided after observing the inflow."""

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x_storage = sp.add_state("x_storage", lb=0.0, ub=8.0, initial_value=6.0)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_hydro = sp.add_variable("u_hydro", lb=0.0)
        u_unmet_demand = sp.add_variable("u_unmet_demand", lb=0.0)
        sp.add_constraint(u_thermal + u_hydro == 9 - u_unmet_demand)
        c_balance = sp.add_constraint(
            x_storage.out == x_storage.in_ - u_hydro + 0, name="c_balance"
        )
        sp.parameterize(lambda w: sp.set_normalized_rhs(c_balance, w), [2, 3])
        sp.set_stage_objective(500 * u_unmet_demand + 20 * u_thermal)

    return sddp.LinearPolicyGraph(
        builder, stages=4, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def build_decision_hazard() -> sddp.PolicyGraph:
    """Here-and-now: ``u_thermal`` becomes a state decided in the previous stage. Its first-stage
    incoming value is fixed to 0, which is the "mistake" fixed by :func:`build_decision_hazard_2`.
    """

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x_storage = sp.add_state("x_storage", lb=0.0, ub=8.0, initial_value=6.0)
        u_thermal = sp.add_state("u_thermal", lb=0.0, initial_value=0.0)  # <-- changed
        u_hydro = sp.add_variable("u_hydro", lb=0.0)
        u_unmet_demand = sp.add_variable("u_unmet_demand", lb=0.0)
        sp.add_constraint(u_thermal.in_ + u_hydro == 9 - u_unmet_demand)  # <-- changed
        c_balance = sp.add_constraint(
            x_storage.out == x_storage.in_ - u_hydro + 0, name="c_balance"
        )
        sp.parameterize(lambda w: sp.set_normalized_rhs(c_balance, w), [2, 3])
        sp.set_stage_objective(500 * u_unmet_demand + 20 * u_thermal.in_)  # <-- changed

    return sddp.LinearPolicyGraph(
        builder, stages=4, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def build_decision_hazard_2() -> sddp.PolicyGraph:
    """The decision-hazard model with an extra deterministic first stage that chooses the
    first ``u_thermal``."""

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x_storage = sp.add_state("x_storage", lb=0.0, ub=8.0, initial_value=6.0)
        u_thermal = sp.add_state("u_thermal", lb=0.0, initial_value=0.0)
        u_hydro = sp.add_variable("u_hydro", lb=0.0)
        u_unmet_demand = sp.add_variable("u_unmet_demand", lb=0.0)
        if node == 1:  # <-- new
            sp.add_constraint(x_storage.out == x_storage.in_)
            sp.set_stage_objective(0)
        else:
            sp.add_constraint(u_thermal.in_ + u_hydro == 9 - u_unmet_demand)
            c_balance = sp.add_constraint(
                x_storage.out == x_storage.in_ - u_hydro + 0, name="c_balance"
            )
            sp.parameterize(lambda w: sp.set_normalized_rhs(c_balance, w), [2, 3])
            sp.set_stage_objective(500 * u_unmet_demand + 20 * u_thermal.in_)

    return sddp.LinearPolicyGraph(
        builder, stages=5, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def train_and_compute_cost(model: sddp.PolicyGraph, seed: int | None = None) -> float:
    """The page's helper: train with the default stopping rule and print the bound."""
    sddp.train(model, print_level=0, seed=seed)
    cost = sddp.calculate_bound(model)
    print("Cost = $", cost, sep="")
    return cost


# =============================================================================
# Example: production planning (tutorial/production_planning.jl)
# =============================================================================
PLANTS = ["Seattle", "San-Diego"]
MARKETS = ["New-York", "Chicago", "Topeka"]
DEMAND_SCENARIOS = [
    {"New-York": 300, "Chicago": 300, "Topeka": 300},
    {"New-York": 350, "Chicago": 320, "Topeka": 310},
    {"New-York": 320, "Chicago": 390, "Topeka": 350},
    {"New-York": 250, "Chicago": 220, "Topeka": 330},
    {"New-York": 290, "Chicago": 200, "Topeka": 290},
]
PLANT_CAPACITY = {"Seattle": 350, "San-Diego": 600}
SHIP_COST = {
    ("Seattle", "New-York"): 2.6,
    ("Seattle", "Chicago"): 1.7,
    ("Seattle", "Topeka"): 1.8,
    ("San-Diego", "New-York"): 2.5,
    ("San-Diego", "Chicago"): 1.8,
    ("San-Diego", "Topeka"): 1.4,
}


def build_production_planning(stages: int = 10) -> sddp.PolicyGraph:
    """Two plants ship to three markets with uncertain demand; the first stage is
    deterministic."""
    P, M, Omega = PLANTS, MARKETS, DEMAND_SCENARIOS

    def builder(sp: sddp.Subproblem, t: int) -> None:
        # State variable: inventory levels
        x = {p: sp.add_state(f"x[{p}]", lb=0.0, initial_value=0.0) for p in P}
        # Control variable: quantity to produce at each plant
        u_prod = {p: sp.add_variable(f"u_prod[{p}]", lb=0.0, ub=PLANT_CAPACITY[p]) for p in P}
        # Control variable: quantity to ship from plant p to market m
        u_ship = {(p, m): sp.add_variable(f"u_ship[{p},{m}]", lb=0.0) for p in P for m in M}
        # Control variable: unmet demand
        u_unmet = {m: sp.add_variable(f"u_unmet[{m}]", lb=0.0) for m in M}
        # Random variable: demand in each market (a fixed variable, `w[m] == 0`)
        w = {m: sp.add_variable(f"w[{m}]") for m in M}
        for m in M:
            sp.fix(w[m], 0.0)
        if t > 1:
            # In the first-stage there is no uncertainty

            def modify(omega: dict[str, float]) -> None:
                for m in M:
                    sp.fix(w[m], omega[m])

            sp.parameterize(modify, Omega)
        # Constraint: can ship only from existing inventory
        for p in P:
            sp.add_constraint(sum(u_ship[p, m] for m in M) <= x[p].in_)
        # Constraint: balance inventory at each plant
        for p in P:
            sp.add_constraint(x[p].out == x[p].in_ + u_prod[p] - sum(u_ship[p, m] for m in M))
        # Constraint: balance demand in each market
        for m in M:
            sp.add_constraint(sum(u_ship[p, m] for p in P) == w[m] - u_unmet[m])
        sp.set_stage_objective(
            # Cost of production and holding inventory
            sum(1.0 * u_prod[p] + 0.01 * x[p].out for p in P)
            # Cost of unmet demand
            + sum(10 * u_unmet[m] for m in M)
            # Cost of shipping
            + sum(SHIP_COST[p, m] * u_ship[p, m] for p in P for m in M)
        )

    return sddp.LinearPolicyGraph(
        builder, stages=stages, sense="Min", optimizer=sddp.HiGHS, lower_bound=0.0
    )


def plot_production_planning(simulations: Sequence[Sequence[dict[str, Any]]]) -> Any:
    """The page's 2x2 panel of publication plots (returns the matplotlib Figure)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    sddp.publication_plot(
        simulations,
        lambda data: data["u_prod[San-Diego]"],
        ax=axes[0][0],
        title="San-Diego",
        ylabel="u_prod",
    )
    sddp.publication_plot(
        simulations, lambda data: data["u_prod[Seattle]"], ax=axes[0][1], title="Seattle"
    )
    sddp.publication_plot(
        simulations,
        lambda data: data["x[San-Diego]"].out,
        ax=axes[1][0],
        ylabel="x",
        xlabel="Week",
    )
    sddp.publication_plot(
        simulations, lambda data: data["x[Seattle]"].out, ax=axes[1][1], xlabel="Week"
    )
    fig.tight_layout()
    return fig


def run_production_planning(
    seed: int | None = None, print_level: int = 1, **train_kwargs: Any
) -> tuple[sddp.PolicyGraph, list[list[dict[str, Any]]]]:
    model = build_production_planning()
    sddp.train(
        model,
        risk_measure=sddp.Expectation(),
        print_level=print_level,
        seed=seed,
        **train_kwargs,
    )
    variables = [f"x[{p}]" for p in PLANTS] + [f"u_prod[{p}]" for p in PLANTS]
    simulations = sddp.simulate(model, 50, variables, seed=seed)
    return model, simulations


# =============================================================================
# Example: batteries (tutorial/batteries.jl)
# =============================================================================
BATTERY_LOAD = [40, 41, 42, 43, 35, 40, 40, 25, 10, 8, 6, 5] + [  # Hours 01-12
    5,
    6,
    8,
    10,
    20,
    30,
    55,
    72,
    75,
    70,
    64,
    60,  # Hours 13-24
]
BATTERY_OMEGA = [-4.0, -2.0, 0.0, 2.0, 4.0]


def build_batteries() -> sddp.PolicyGraph:
    """A day in the life of a power system with one thermal generator and one battery. The
    energy-balance constraint is registered under the name ``"λ"``."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        # State variables
        x_soc = sp.add_state("x_soc", lb=0.0, ub=30.0, initial_value=4.0)
        x_thermal = sp.add_state("x_thermal", lb=0.0, ub=70.0, initial_value=35.0)
        # Control variables
        u_charge = sp.add_variable("u_charge", lb=0.0, ub=15.0)
        u_discharge = sp.add_variable("u_discharge", lb=0.0, ub=15.0)
        u_slack = sp.add_variable("u_slack", lb=0.0)
        u_surplus = sp.add_variable("u_surplus", lb=0.0)
        # Random variables
        w_load = sp.add_variable("w_load")
        d = BATTERY_LOAD
        sp.parameterize(lambda w: sp.fix(w_load, d[t - 1] + w), BATTERY_OMEGA)
        # Objective function
        sp.set_stage_objective(70 * x_thermal.out + 500 * u_slack)
        # Constraints
        sp.add_constraint(x_soc.out == x_soc.in_ + 0.8 * u_charge - u_discharge)
        sp.add_constraint(x_thermal.out - x_thermal.in_ <= 10)
        sp.add_constraint(
            x_thermal.out + u_discharge - u_charge + u_slack == w_load + u_surplus, name="λ"
        )

    return sddp.LinearPolicyGraph(
        builder, stages=24, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def run_batteries(
    iteration_limit: int = 500,
    replications: int = 100,
    seed: int | None = None,
    parallel_scheme: Any = None,
    print_level: int = 1,
) -> tuple[sddp.PolicyGraph, list[list[dict[str, Any]]]]:
    """Train (the page uses ``SDDP.Threaded()``) and simulate, recording the dual ``λ`` of the
    energy-balance constraint with ``custom_recorders``."""
    model = build_batteries()
    sddp.train(
        model,
        iteration_limit=iteration_limit,
        parallel_scheme=parallel_scheme,
        print_level=print_level,
        seed=seed,
    )
    results = sddp.simulate(
        model,
        replications,
        ["x_soc", "x_thermal", "u_slack", "w_load"],
        custom_recorders={"λ": lambda sp: sp.dual(sp["λ"])},
        seed=seed,
    )
    return model, results


def plot_batteries(results: Sequence[Sequence[dict[str, Any]]]) -> Any:
    """The page's five publication plots (returns the matplotlib Figure)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(6, 15))
    panels: list[tuple[Callable[[dict[str, Any]], float], str]] = [
        (lambda d: d["w_load"], "Net load"),
        (lambda d: d["x_thermal"].out, "Thermal generation"),
        (lambda d: d["x_soc"].out, "State of charge"),
        (lambda d: d["λ"], "Dual λ"),
        (lambda d: d["u_slack"], "Lost load"),
    ]
    for ax, (fn, ylabel) in zip(axes, panels):
        sddp.publication_plot(results, fn, ax=ax, xlabel="Hour", ylabel=ylabel)
    fig.tight_layout()
    return fig


# =============================================================================
# Example: inventory management (tutorial/inventory.jl)
# =============================================================================
INV_X0 = 10  # initial inventory
INV_C = 35  # unit inventory cost
INV_H = 1  # unit inventory holding cost
INV_P = 15  # unit order cost
INV_OMEGA = [800.0 * i / 19 for i in range(20)]  # range(0, 800; length = 20)


def build_inventory_finite(T: int = 10) -> sddp.PolicyGraph:
    """Finite horizon: ``T + 1`` stages; in the last stage leftover inventory is recovered and
    any backlog bought out at the unit cost ``c``."""
    x_0, c, h, p, Omega = INV_X0, INV_C, INV_H, INV_P, INV_OMEGA

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x_inventory = sp.add_state("x_inventory", lb=0.0, initial_value=x_0)
        x_demand = sp.add_state("x_demand", lb=0.0, initial_value=0.0)
        # u_buy is a Decision-Hazard control variable. We decide u.out for use in the next stage
        u_buy = sp.add_state("u_buy", lb=0.0, initial_value=0.0)
        u_sell = sp.add_variable("u_sell", lb=0.0)
        w_demand = sp.add_variable("w_demand")
        sp.fix(w_demand, 0.0)
        sp.add_constraint(x_inventory.out == x_inventory.in_ + u_buy.in_ - u_sell)
        sp.add_constraint(x_demand.out == x_demand.in_ + w_demand - u_sell)
        if t == 1:
            sp.fix(u_sell, 0.0)
            sp.set_stage_objective(c * u_buy.out)
        elif t == T + 1:
            sp.fix(u_buy.out, 0.0)
            sp.set_stage_objective(-c * x_inventory.out + c * x_demand.out)
            sp.parameterize(lambda w: sp.fix(w_demand, w), Omega)
        else:
            sp.set_stage_objective(c * u_buy.out + h * x_inventory.out + p * x_demand.out)
            sp.parameterize(lambda w: sp.fix(w_demand, w), Omega)

    return sddp.LinearPolicyGraph(
        builder, stages=T + 1, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def build_inventory_graph(alpha: float = 0.95) -> sddp.Graph:
    """``LinearGraph(2)`` with a self-loop ``2 -> 2`` of probability ``alpha``."""
    graph = sddp.LinearGraph(2)
    graph.add_edge(2, 2, alpha)
    return graph


def build_inventory_infinite(alpha: float = 0.95) -> sddp.PolicyGraph:
    """Infinite horizon with discount factor ``alpha``."""
    x_0, c, h, p, Omega = INV_X0, INV_C, INV_H, INV_P, INV_OMEGA

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x_inventory = sp.add_state("x_inventory", lb=0.0, initial_value=x_0)
        x_demand = sp.add_state("x_demand", lb=0.0, initial_value=0.0)
        u_buy = sp.add_state("u_buy", lb=0.0, initial_value=0.0)
        u_sell = sp.add_variable("u_sell", lb=0.0)
        w_demand = sp.add_variable("w_demand")
        sp.fix(w_demand, 0.0)
        sp.add_constraint(x_inventory.out == x_inventory.in_ + u_buy.in_ - u_sell)
        sp.add_constraint(x_demand.out == x_demand.in_ + w_demand - u_sell)
        if t == 1:
            sp.fix(u_sell, 0.0)
            sp.set_stage_objective(c * u_buy.out)
        else:
            sp.set_stage_objective(c * u_buy.out + h * x_inventory.out + p * x_demand.out)
            sp.parameterize(lambda w: sp.fix(w_demand, w), Omega)

    return sddp.PolicyGraph(
        builder, build_inventory_graph(alpha), sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def order_up_to_level(data: dict[str, Any]) -> float:
    """``x_inventory.out + u_buy.out``: the level the policy orders up to."""
    return data["x_inventory"].out + data["u_buy"].out


def run_inventory_finite(
    seed: int | None = None, print_level: int = 1, **train_kwargs: Any
) -> tuple[sddp.PolicyGraph, list[list[dict[str, Any]]]]:
    model = build_inventory_finite()
    sddp.train(model, print_level=print_level, seed=seed, **train_kwargs)
    simulations = sddp.simulate(model, 200, ["x_inventory", "u_buy"], seed=seed)
    objective_values = [sum(t["stage_objective"] for t in s) for s in simulations]
    mu, ci = (round(v, 2) for v in sddp.confidence_interval(objective_values, 1.96))
    lower_bound = round(sddp.calculate_bound(model), 2)
    if print_level > 0:
        print("Confidence interval: ", mu, " ± ", ci, sep="")
        print("Lower bound: ", lower_bound, sep="")
    return model, simulations


def run_inventory_infinite(
    iteration_limit: int = 400, seed: int | None = None, print_level: int = 1
) -> tuple[sddp.PolicyGraph, list[list[dict[str, Any]]]]:
    model = build_inventory_infinite()
    sddp.train(model, iteration_limit=iteration_limit, print_level=print_level, seed=seed)
    simulations = sddp.simulate(
        model,
        200,
        ["x_inventory", "u_buy"],
        sampling_scheme=sddp.InSampleMonteCarlo(max_depth=50, terminate_on_dummy_leaf=False),
        seed=seed,
    )
    return model, simulations


def plot_inventory(simulations: Sequence[Sequence[dict[str, Any]]], analytic: bool = False) -> Any:
    """Publication plot of the order-up-to level; ``analytic`` adds the 662 line of the
    infinite-horizon section."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(6, 3))
    sddp.publication_plot(
        simulations,
        order_up_to_level,
        ax=ax,
        title="x_inventory.out + u_buy.out",
        xlabel="Stage",
        ylabel="Quantity",
    )
    ax.set_ylim(0, 1_000)
    if analytic:
        ax.axhline(662, color="C1", label="Analytic solution")
        ax.legend()
    return ax


# =============================================================================
def main() -> None:
    print("== Words of warning")
    run_warnings()
    print("== Auto-regressive stochastic processes")
    model = build_state_space_expansion()
    sddp.train(model, iteration_limit=20, print_level=0)
    print("state-space expansion bound:", sddp.calculate_bound(model))
    print("simulator():", simulator())
    graph = build_markov_chain_graph()
    print(graph)
    model = build_markov_chain_model(graph)
    sddp.plot_graph(model, "model_arma.html")
    sddp.train(model, iteration_limit=20, print_level=0)
    print("Markov chain bound:", sddp.calculate_bound(model))
    model = build_var_model()
    sddp.train(model, iteration_limit=100, print_level=0)
    print("VAR bound:", sddp.calculate_bound(model))
    print("== Here-and-now and hazard-decision")
    train_and_compute_cost(build_hazard_decision())
    train_and_compute_cost(build_decision_hazard())
    train_and_compute_cost(build_decision_hazard_2())
    print("== Example: production planning")
    _, sims = run_production_planning()
    plot_production_planning(sims)
    print("== Example: batteries")
    _, results = run_batteries()
    plot_batteries(results)
    print("== Example: inventory management")
    _, sims = run_inventory_finite()
    plot_inventory(sims)
    _, sims = run_inventory_infinite()
    plot_inventory(sims, analytic=True)


if __name__ == "__main__":
    main()
