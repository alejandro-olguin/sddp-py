"""Python translations of two sddp.dev tutorial pages that share the same 52-week data set:

* "Example: deterministic to stochastic" (docs/src/tutorial/example_reservoir.jl), and
* "Example: capacity expansion models" (docs/src/tutorial/capacity_expansion.jl).

Every model on the two pages is a ``build_*`` function returning a fresh model; the
``run_*`` functions reproduce the page sections (training, simulation, plots) and ``main()``
walks through both pages. Plots are optional (matplotlib is imported inside the functions).
"""

from __future__ import annotations

import csv
import io
import random
from collections.abc import Sequence
from typing import Any

import sddp
from sddp.solver.model import HiGHS as HiGHS_LP
from sddp.solver.model import Model, Sense

# ------------------------------------------------------------------------------- data
# The CSV data of the page (written to a temporary file in Julia and read with CSV.jl).
CSV_DATA = """week,inflow,demand,cost
1,3,7,10.2\n2,2,7.1,10.4\n3,3,7.2,10.6\n4,2,7.3,10.9\n5,3,7.4,11.2\n
6,2,7.6,11.5\n7,3,7.8,11.9\n8,2,8.1,12.3\n9,3,8.3,12.7\n10,2,8.6,13.1\n
11,3,8.9,13.6\n12,2,9.2,14\n13,3,9.5,14.5\n14,2,9.8,14.9\n15,3,10.1,15.3\n
16,2,10.4,15.8\n17,3,10.7,16.2\n18,2,10.9,16.6\n19,3,11.2,17\n20,3,11.4,17.4\n
21,3,11.6,17.7\n22,2,11.7,18\n23,3,11.8,18.3\n24,2,11.9,18.5\n25,3,12,18.7\n
26,2,12,18.9\n27,3,12,19\n28,2,11.9,19.1\n29,3,11.8,19.2\n30,2,11.7,19.2\n
31,3,11.6,19.2\n32,2,11.4,19.2\n33,3,11.2,19.1\n34,2,10.9,19\n35,3,10.7,18.9\n
36,2,10.4,18.8\n37,3,10.1,18.6\n38,2,9.8,18.5\n39,3,9.5,18.4\n40,3,9.2,18.2\n
41,2,8.9,18.1\n42,3,8.6,17.9\n43,2,8.3,17.8\n44,3,8.1,17.7\n45,2,7.8,17.6\n
46,3,7.6,17.5\n47,2,7.4,17.5\n48,3,7.3,17.5\n49,2,7.2,17.5\n50,3,7.1,17.6\n
51,3,7,17.7\n52,3,7,17.8\n
"""


def read_data(text: str = CSV_DATA) -> dict[str, list[float]]:
    """Parse the CSV text into ``{"week": [...], "inflow": [...], "demand": [...], "cost": [...]}``
    (blank lines are skipped, as CSV.jl does)."""
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["week"] not in (None, "")]
    return {k: [float(r[k]) for r in rows] for k in ("week", "inflow", "demand", "cost")}


DATA = read_data()
T = len(DATA["week"])  # 52 weeks

# Constants of "Example: deterministic to stochastic".
RESERVOIR_MAX = 320.0
RESERVOIR_INITIAL = 300.0
FLOW_MAX = 12.0

# Constants of "Example: capacity expansion models".
CAPEX_RESERVOIR_MAX = 350.0
CAPEX_RESERVOIR_INITIAL = 300.0
CAPEX_FLOW_MAX = 9.0

OMEGA, PROB = [-2.0, 0.0, 5.0], [0.3, 0.4, 0.3]


# ======================================================================================
# Example: deterministic to stochastic
# ======================================================================================


# ---------------------------------------------------------------- deterministic LP
def solve_deterministic_lp(data: dict[str, list[float]] = DATA) -> dict[str, Any]:
    """The pure-JuMP model of the page as a single LP (``sddp.solver.model.Model``).

    Returns the objective and the ``x_storage`` (T+1 values), ``u_flow``, ``u_spill`` and
    ``u_thermal`` traces.
    """
    T = len(data["week"])
    model = Model(HiGHS_LP)
    x_storage = [
        model.add_variable(f"x_storage[{t}]", lb=0.0, ub=RESERVOIR_MAX) for t in range(T + 1)
    ]
    model.fix(x_storage[0], RESERVOIR_INITIAL)
    u_flow = [model.add_variable(f"u_flow[{t}]", lb=0.0, ub=FLOW_MAX) for t in range(T)]
    u_spill = [model.add_variable(f"u_spill[{t}]", lb=0.0) for t in range(T)]
    u_thermal = [model.add_variable(f"u_thermal[{t}]", lb=0.0) for t in range(T)]
    omega_inflow = [model.add_variable(f"omega_inflow[{t}]") for t in range(T)]
    for t in range(T):
        model.fix(omega_inflow[t], data["inflow"][t])
    for t in range(T):
        model.add_constraint(
            x_storage[t + 1] == x_storage[t] - u_flow[t] - u_spill[t] + omega_inflow[t]
        )
    for t in range(T):
        model.add_constraint(u_flow[t] + u_thermal[t] == data["demand"][t])
    model.set_objective(sum(data["cost"][t] * u_thermal[t] for t in range(T)), Sense.MIN)
    model.optimize()
    return {
        "termination_status": model.termination_status(),
        "objective": model.objective_value(),
        "x_storage": [model.value(v) for v in x_storage],
        "u_flow": [model.value(v) for v in u_flow],
        "u_spill": [model.value(v) for v in u_spill],
        "u_thermal": [model.value(v) for v in u_thermal],
    }


# ------------------------------------------------------------ deterministic SDDP
def _reservoir_subproblem(
    sp: sddp.Subproblem, t: int, data: dict[str, list[float]], stochastic: bool
) -> None:
    x_storage = sp.add_state("x_storage", lb=0.0, ub=RESERVOIR_MAX, initial_value=RESERVOIR_INITIAL)
    u_flow = sp.add_variable("u_flow", lb=0.0, ub=FLOW_MAX)
    u_thermal = sp.add_variable("u_thermal", lb=0.0)
    u_spill = sp.add_variable("u_spill", lb=0.0)
    omega_inflow = sp.add_variable("omega_inflow")
    if stochastic:
        # <--- This bit is new
        @sp.parameterize(OMEGA, PROB)
        def _(omega: float) -> None:
            sp.fix(omega_inflow, data["inflow"][t - 1] + omega)

        # --->
    else:
        sp.fix(omega_inflow, data["inflow"][t - 1])
    sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
    sp.add_constraint(u_flow + u_thermal == data["demand"][t - 1])
    sp.set_stage_objective(data["cost"][t - 1] * u_thermal)


def build_deterministic_sddp(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """The JuMP model decomposed into a ``LinearPolicyGraph`` with fixed inflows."""
    return sddp.LinearPolicyGraph(
        lambda sp, t: _reservoir_subproblem(sp, t, data, stochastic=False),
        stages=len(data["week"]),
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def build_stochastic_sddp(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Inflow is ``data + ω`` with ``Ω = [-2, 0, 5]`` and ``P = [0.3, 0.4, 0.3]``."""
    return sddp.LinearPolicyGraph(
        lambda sp, t: _reservoir_subproblem(sp, t, data, stochastic=True),
        stages=len(data["week"]),
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def build_cyclic_sddp(
    data: dict[str, list[float]] = DATA, discount: float = 0.95
) -> sddp.PolicyGraph:
    """Same subproblems on ``UnicyclicGraph(0.95, num_nodes=T)``."""
    graph = sddp.UnicyclicGraph(discount, num_nodes=len(data["week"]))
    return sddp.PolicyGraph(
        lambda sp, t: _reservoir_subproblem(sp, t, data, stochastic=True),
        graph,
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def run_deterministic_sddp(iteration_limit: int = 10, **kwargs: Any) -> dict[str, Any]:
    """Train the deterministic model, compare its bound with the LP and simulate once."""
    model = build_deterministic_sddp()
    sddp.train(model, iteration_limit=iteration_limit, **kwargs)
    simulations = sddp.simulate(model, 1, ["x_storage", "u_flow", "u_thermal"])
    r_sim = [sim["u_thermal"] for sim in simulations[0]]
    u_sim = [sim["u_flow"] for sim in simulations[0]]
    x_sim = [sim["x_storage"].out for sim in simulations[0]]
    return {
        "model": model,
        "bound": sddp.calculate_bound(model),
        "simulations": simulations,
        "u_thermal": r_sim,
        "u_flow": u_sim,
        "x_storage": x_sim,
    }


def run_stochastic_sddp(
    iteration_limit: int = 100,
    replications: int = 100,
    spaghetti_file: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train the stochastic model, simulate and (optionally) write the spaghetti plot HTML."""
    model = build_stochastic_sddp()
    sddp.train(model, iteration_limit=iteration_limit, **kwargs)
    simulations = sddp.simulate(
        model, replications, ["x_storage", "u_flow", "u_thermal", "omega_inflow"]
    )
    plot = sddp.SpaghettiPlot(simulations)
    plot.add_spaghetti(lambda sim: sim["x_storage"].out, title="Storage")
    plot.add_spaghetti(lambda sim: sim["u_flow"], title="Hydro")
    plot.add_spaghetti(lambda sim: sim["omega_inflow"], title="Inflow")
    html = plot.plot(spaghetti_file) if spaghetti_file is not None else None
    return {
        "model": model,
        "bound": sddp.calculate_bound(model),
        "simulations": simulations,
        "spaghetti": plot,
        "spaghetti_file": html,
    }


def run_cyclic_sddp(
    iteration_limit: int = 100, replications: int = 100, **kwargs: Any
) -> dict[str, Any]:
    """Train the cyclic model; simulate 3 free-length paths and ``replications`` paths of 5T."""
    model = build_cyclic_sddp()
    sddp.train(model, iteration_limit=iteration_limit, **kwargs)
    free = sddp.simulate(model, 3)
    fixed = sddp.simulate(
        model,
        replications,
        ["x_storage", "u_flow"],
        sampling_scheme=sddp.InSampleMonteCarlo(max_depth=5 * T, terminate_on_dummy_leaf=False),
    )
    return {
        "model": model,
        "bound": sddp.calculate_bound(model),
        "free_lengths": [len(s) for s in free],
        "fixed_lengths": [len(s) for s in fixed],
        "simulations": fixed,
    }


def plot_data(data: dict[str, list[float]] = DATA, filename: str | None = None) -> Any:
    """The 3-panel plot of inflow / demand / cost."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True)
    for ax, key in zip(axes, ("inflow", "demand", "cost")):
        ax.plot(data["week"], data[key])
        ax.set_ylabel(key.capitalize())
    axes[-1].set_xlabel("Week")
    if filename is not None:
        fig.savefig(filename)
    return fig


def plot_storage_and_hydro(
    simulations: Sequence[Sequence[dict[str, Any]]], filename: str | None = None
) -> Any:
    """``publication_plot`` of storage and hydro, ``layout = (2, 1)``."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1)
    sddp.publication_plot(
        simulations, lambda sim: sim["x_storage"].out, ax=axes[0], ylabel="Storage"
    )
    sddp.publication_plot(simulations, lambda sim: sim["u_flow"], ax=axes[1], ylabel="Hydro")
    if filename is not None:
        fig.savefig(filename)
    return fig


# ======================================================================================
# Example: capacity expansion models
# ======================================================================================


def build_capex_operational(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 1: the operational problem (``reservoir_max = 350``, ``flow_max = 9``)."""

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x_storage = sp.add_state(
            "x_storage", lb=0.0, ub=CAPEX_RESERVOIR_MAX, initial_value=CAPEX_RESERVOIR_INITIAL
        )
        u_flow = sp.add_variable("u_flow", lb=0.0, ub=CAPEX_FLOW_MAX)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_spill = sp.add_variable("u_spill", lb=0.0)
        omega_inflow = sp.add_variable("omega_inflow")
        sp.parameterize(lambda w: sp.fix(omega_inflow, data["inflow"][t - 1] + w), OMEGA, PROB)
        sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
        sp.add_constraint(u_flow + u_thermal == data["demand"][t - 1])
        sp.set_stage_objective(data["cost"][t - 1] * u_thermal)

    return sddp.LinearPolicyGraph(
        builder, stages=len(data["week"]), sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def build_capex_invest_then_operate(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 2: ``reservoir_max`` becomes an investment made in node 1."""
    T = len(data["week"])

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x_reservoir_max = sp.add_state("x_reservoir_max", lb=0.0, initial_value=0.0)
        x_storage = sp.add_state("x_storage", lb=0.0, initial_value=0.0)
        sp.add_constraint(x_storage.out <= x_reservoir_max.out)
        u_flow = sp.add_variable("u_flow", lb=0.0, ub=CAPEX_FLOW_MAX)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_spill = sp.add_variable("u_spill", lb=0.0)
        omega_inflow = sp.add_variable("omega_inflow")
        if node == 1:  # Investment node
            sp.set_stage_objective(x_reservoir_max.out)
            sp.add_constraint(x_storage.out <= CAPEX_RESERVOIR_INITIAL)
        else:  # Operational node
            t = (node - 1) % (T + 1)
            sp.add_constraint(x_reservoir_max.out == x_reservoir_max.in_)
            sp.parameterize(lambda w: sp.fix(omega_inflow, data["inflow"][t - 1] + w), OMEGA, PROB)
            sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
            sp.add_constraint(u_flow + u_thermal == data["demand"][t - 1])
            sp.set_stage_objective(data["cost"][t - 1] * u_thermal)

    return sddp.LinearPolicyGraph(
        builder, stages=T + 1, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def build_capex_multiple_investments(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 3: both ``reservoir_max`` and ``flow_max`` are investments."""
    T = len(data["week"])

    def builder(sp: sddp.Subproblem, node: int) -> None:
        x_reservoir_max = sp.add_state("x_reservoir_max", lb=0.0, initial_value=0.0)
        x_flow_max = sp.add_state("x_flow_max", lb=0.0, initial_value=0.0)
        x_storage = sp.add_state("x_storage", lb=0.0, initial_value=0.0)
        sp.add_constraint(x_storage.out <= x_reservoir_max.out)
        u_flow = sp.add_variable("u_flow", lb=0.0)
        sp.add_constraint(u_flow <= x_flow_max.out)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_spill = sp.add_variable("u_spill", lb=0.0)
        omega_inflow = sp.add_variable("omega_inflow")
        if node == 1:  # Investment node
            sp.set_stage_objective(x_reservoir_max.out + x_flow_max.out)
            sp.add_constraint(x_storage.out <= CAPEX_RESERVOIR_INITIAL)
        else:  # Operational node
            t = (node - 1) % (T + 1)
            sp.add_constraint(x_reservoir_max.out == x_reservoir_max.in_)
            sp.add_constraint(x_flow_max.out == x_flow_max.in_)
            sp.parameterize(lambda w: sp.fix(omega_inflow, data["inflow"][t - 1] + w), OMEGA, PROB)
            sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
            sp.add_constraint(u_flow + u_thermal == data["demand"][t - 1])
            sp.set_stage_objective(data["cost"][t - 1] * u_thermal)

    return sddp.LinearPolicyGraph(
        builder, stages=T + 1, sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def _invest_operate_twice_builder(
    sp: sddp.Subproblem, node: int, data: dict[str, list[float]]
) -> None:
    """Shared subproblem of models 4 (finite) and 5 (loop): nodes 1 and T+2 invest."""
    T = len(data["week"])
    x_reservoir_max = sp.add_state("x_reservoir_max", lb=0.0, initial_value=0.0)
    x_flow_max = sp.add_state("x_flow_max", lb=0.0, initial_value=0.0)
    x_storage = sp.add_state("x_storage", lb=0.0, initial_value=0.0)
    sp.add_constraint(x_storage.out <= x_reservoir_max.out)
    u_flow = sp.add_variable("u_flow", lb=0.0)
    sp.add_constraint(u_flow <= x_flow_max.out)
    u_thermal = sp.add_variable("u_thermal", lb=0.0)
    u_spill = sp.add_variable("u_spill", lb=0.0)
    omega_inflow = sp.add_variable("omega_inflow")
    if node == 1:  # First investment node
        sp.set_stage_objective(x_reservoir_max.out + x_flow_max.out)
        sp.add_constraint(x_storage.out <= CAPEX_RESERVOIR_INITIAL)
    elif node == T + 2:  # Second investment node
        sp.set_stage_objective(
            (x_reservoir_max.out - x_reservoir_max.in_) + (x_flow_max.out - x_flow_max.in_)
        )
        sp.add_constraint(x_storage.out == x_storage.in_)
    else:  # Operational node
        t = (node - 1) % (T + 1)
        sp.add_constraint(x_reservoir_max.out == x_reservoir_max.in_)
        sp.add_constraint(x_flow_max.out == x_flow_max.in_)
        sp.parameterize(lambda w: sp.fix(omega_inflow, data["inflow"][t - 1] + w), OMEGA, PROB)
        sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
        sp.add_constraint(u_flow + u_thermal == data["demand"][t - 1])
        sp.set_stage_objective(data["cost"][t - 1] * u_thermal)


def build_capex_invest_operate_invest_operate(
    data: dict[str, list[float]] = DATA,
) -> sddp.PolicyGraph:
    """Model 4: two operational years, each preceded by an investment node (2T+2 stages)."""
    T = len(data["week"])
    return sddp.LinearPolicyGraph(
        lambda sp, node: _invest_operate_twice_builder(sp, node, data),
        stages=2 * T + 2,
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def capex_loop_graph(T: int = T, discount: float = 0.95) -> sddp.Graph:
    """``LinearGraph(2T+2)`` plus the arc ``2T+2 => T+3`` with probability 0.95."""
    graph = sddp.LinearGraph(2 * T + 2)
    graph.add_edge(2 * T + 2, T + 3, discount)
    return graph


def build_capex_loop(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 5: model 4 whose second operational year loops with probability 0.95."""
    return sddp.PolicyGraph(
        lambda sp, node: _invest_operate_twice_builder(sp, node, data),
        capex_loop_graph(len(data["week"])),
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )


def capex_strategic_graph() -> sddp.Graph:
    """The general graph of model 6: nodes are ``(name, t)`` tuples (Julia uses Symbols)."""
    graph: sddp.Graph = sddp.Graph(("root", 0))
    graph.add_node(("invest_1", 0))  # First investment
    graph.add_node(("invest_2", 0))  # Second investment
    for t in range(1, 53):
        graph.add_node(("Y1", t))
        graph.add_node(("Y2_normal", t))
        graph.add_node(("Y2_high", t))
    for t in range(2, 53):
        graph.add_edge(("Y1", t - 1), ("Y1", t), 1.0)
        graph.add_edge(("Y2_normal", t - 1), ("Y2_normal", t), 1.0)
        graph.add_edge(("Y2_high", t - 1), ("Y2_high", t), 1.0)
    graph.add_edge(("root", 0), ("invest_1", 0), 1.0)
    graph.add_edge(("invest_1", 0), ("Y1", 1), 1.0)
    graph.add_edge(("Y1", 52), ("invest_2", 0), 0.9)
    graph.add_edge(("invest_2", 0), ("Y2_normal", 1), 0.5)
    graph.add_edge(("invest_2", 0), ("Y2_high", 1), 0.5)
    graph.add_edge(("Y2_normal", 52), ("Y2_normal", 1), 0.9)
    graph.add_edge(("Y2_high", 52), ("Y2_high", 1), 0.9)
    return graph


def build_capex_strategic_uncertainty(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 6: two possible second-year cycles (normal, or 50% higher inflow and demand)."""

    def builder(sp: sddp.Subproblem, index: tuple[str, int]) -> None:
        node, t = index
        x_reservoir_max = sp.add_state("x_reservoir_max", lb=0.0, initial_value=0.0)
        x_flow_max = sp.add_state("x_flow_max", lb=0.0, ub=20.0, initial_value=0.0)
        x_storage = sp.add_state("x_storage", lb=0.0, initial_value=0.0)
        sp.add_constraint(x_storage.out <= x_reservoir_max.out)
        u_flow = sp.add_variable("u_flow", lb=0.0)
        sp.add_constraint(u_flow <= x_flow_max.out)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_spill = sp.add_variable("u_spill", lb=0.0)
        omega_inflow = sp.add_variable("omega_inflow")
        if node == "invest_1":  # First investment node
            sp.set_stage_objective(x_reservoir_max.out + x_flow_max.out)
            sp.add_constraint(x_storage.out <= CAPEX_RESERVOIR_INITIAL)
        elif node == "invest_2":  # Second investment node
            sp.set_stage_objective(
                (x_reservoir_max.out - x_reservoir_max.in_) + (x_flow_max.out - x_flow_max.in_)
            )
            sp.add_constraint(x_storage.out == x_storage.in_)
        else:  # Operational node
            sp.add_constraint(x_reservoir_max.out == x_reservoir_max.in_)
            sp.add_constraint(x_flow_max.out == x_flow_max.in_)
            scale = 1.5 if node == "Y2_high" else 1.0
            sp.parameterize(
                lambda w: sp.fix(omega_inflow, scale * data["inflow"][t - 1] + w), OMEGA, PROB
            )
            sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
            sp.add_constraint(u_flow + u_thermal == scale * data["demand"][t - 1])
            sp.set_stage_objective(data["cost"][t - 1] * u_thermal)

    return sddp.PolicyGraph(
        builder, capex_strategic_graph(), sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


EPICYCLES_T, EPICYCLES_P = 3, 0.9


def epicycles_strategic_graph(T: int = EPICYCLES_T, p: float = EPICYCLES_P) -> sddp.Graph:
    """The strategic scenario tree of the epicycles model (before the operational cycle)."""
    graph: sddp.Graph = sddp.Graph(("root", 0))
    for name in ("inv", "inv_h", "inv_l", "inv_hh", "inv_hl", "inv_lh", "inv_ll"):
        graph.add_node((name, 0))
    graph.add_edge(("root", 0), ("inv", 0), 1.0)
    graph.add_edge(("inv", 0), ("inv_h", 0), p**T / 2)
    graph.add_edge(("inv", 0), ("inv_l", 0), p**T / 2)
    graph.add_edge(("inv_h", 0), ("inv_hh", 0), p**T / 2)
    graph.add_edge(("inv_h", 0), ("inv_hl", 0), p**T / 2)
    graph.add_edge(("inv_l", 0), ("inv_lh", 0), p**T / 2)
    graph.add_edge(("inv_l", 0), ("inv_ll", 0), p**T / 2)
    return graph


def epicycles_graph(T: int = EPICYCLES_T, p: float = EPICYCLES_P) -> sddp.Graph:
    """The strategic tree plus the 52-node operational cycle (``(:op, t)`` nodes)."""
    graph = epicycles_strategic_graph(T, p)
    graph.add_node(("op", 1))
    for t in range(2, 53):
        graph.add_node(("op", t))
        graph.add_edge(("op", t - 1), ("op", t), 1.0)
    graph.add_edge(("op", 52), ("op", 1), p)
    graph.add_edge(("inv", 0), ("op", 1), T * (1 - p))
    graph.add_edge(("inv_h", 0), ("op", 1), T * (1 - p))
    graph.add_edge(("inv_l", 0), ("op", 1), T * (1 - p))
    graph.add_edge(("inv_hh", 0), ("op", 1), 1.0)
    graph.add_edge(("inv_hl", 0), ("op", 1), 1.0)
    graph.add_edge(("inv_lh", 0), ("op", 1), 1.0)
    graph.add_edge(("inv_ll", 0), ("op", 1), 1.0)
    return graph


def build_capex_epicycles(data: dict[str, list[float]] = DATA) -> sddp.PolicyGraph:
    """Model 7: strategic scenario tree with an infinite-horizon operational cycle. The demand
    scale ``x_scale`` is a state; the inflow noise modifies the coefficient of ``x_scale.in_``."""

    def builder(sp: sddp.Subproblem, index: tuple[str, int]) -> None:
        node, t = index
        # Investment state variables
        x_reservoir_max = sp.add_state("x_reservoir_max", lb=0.0, initial_value=0.0)
        x_flow_max = sp.add_state("x_flow_max", lb=0.0, ub=20.0, initial_value=0.0)
        # Reservoir state variables
        x_storage = sp.add_state("x_storage", lb=0.0, initial_value=0.0)
        # Demand state variables
        x_scale = sp.add_state("x_scale", initial_value=1.0)
        # Control variables
        u_flow = sp.add_variable("u_flow", lb=0.0)
        u_thermal = sp.add_variable("u_thermal", lb=0.0)
        u_spill = sp.add_variable("u_spill", lb=0.0)
        # Random variables
        omega_inflow = sp.add_variable("omega_inflow")
        if t > 0:  # Operational node
            sp.set_stage_objective(data["cost"][t - 1] * u_thermal)
            # Investment and demand states are fixed
            sp.add_constraint(x_reservoir_max.out == x_reservoir_max.in_)
            sp.add_constraint(x_flow_max.out == x_flow_max.in_)
            sp.add_constraint(x_scale.out == x_scale.in_)
            # Investments impact current decisions
            sp.add_constraint(x_storage.out <= x_reservoir_max.out)
            sp.add_constraint(u_flow <= x_flow_max.out)
            sp.add_constraint(x_storage.out == x_storage.in_ - u_flow - u_spill + omega_inflow)
            sp.add_constraint(u_flow + u_thermal == x_scale.in_ * data["demand"][t - 1])
            # Random variable
            c_omega = sp.add_constraint(
                omega_inflow - x_scale.in_ * data["inflow"][t - 1] == 0, name="c_omega"
            )
            sp.parameterize(
                lambda w: sp.set_normalized_coefficient(
                    c_omega, x_scale.in_, -(data["inflow"][t - 1] + w)
                ),
                OMEGA,
                PROB,
            )
        else:  # Investment node
            sp.set_stage_objective(
                12 * (x_reservoir_max.out - x_reservoir_max.in_) + (x_flow_max.out - x_flow_max.in_)
            )
            # Assume reservoir starts out at 80% full
            sp.add_constraint(x_storage.out == 0.8 * x_reservoir_max.out)
            # Update scale factors based on scenario tree
            if node.endswith("h"):
                sp.add_constraint(x_scale.out == 1.5 * x_scale.in_)
            elif node.endswith("l"):
                sp.add_constraint(x_scale.out == 0.8 * x_scale.in_)
            else:
                sp.add_constraint(x_scale.out == x_scale.in_)

    return sddp.PolicyGraph(
        builder, epicycles_graph(), sense="Min", lower_bound=0.0, optimizer=sddp.HiGHS
    )


def sample_epicycles_scenario(rng: random.Random, T: int = EPICYCLES_T) -> list[tuple[Any, Any]]:
    """The page's ``sample_scenario()``: three investment nodes (noise ``None``) each followed
    by ``T`` full years of operational nodes with sampled inflow noise."""
    D = [sddp.Noise(w, p) for w, p in zip(OMEGA, PROB)]
    inv_1 = "inv_" + rng.choice(["l", "h"])
    inv_2 = inv_1 + rng.choice(["l", "h"])
    scenario: list[tuple[Any, Any]] = [(("inv", 0), None)]
    scenario += [(("op", t), sddp.sample_noise(D, rng)) for _ in range(T) for t in range(1, 53)]
    scenario.append(((inv_1, 0), None))
    scenario += [(("op", t), sddp.sample_noise(D, rng)) for _ in range(T) for t in range(1, 53)]
    scenario.append(((inv_2, 0), None))
    scenario += [(("op", t), sddp.sample_noise(D, rng)) for _ in range(T) for t in range(1, 53)]
    return scenario


CAPEX_MODELS = {
    "capex_operational": (build_capex_operational, 100),
    "capex_invest_then_operate": (build_capex_invest_then_operate, 100),
    "capex_multiple_investments": (build_capex_multiple_investments, 100),
    "capex_invest_operate_invest_operate": (build_capex_invest_operate_invest_operate, 100),
    "capex_loop": (build_capex_loop, 100),
    "capex_strategic_uncertainty": (build_capex_strategic_uncertainty, 100),
    "capex_epicycles": (build_capex_epicycles, 200),
}


def capex_sampling_scheme(
    name: str, rng: random.Random | None = None, replications: int = 100
) -> Any:
    """The simulation sampling scheme each capacity-expansion section uses (``None`` = default)."""
    if name in ("capex_loop", "capex_strategic_uncertainty"):
        return sddp.InSampleMonteCarlo(max_depth=5 * 52 + 2, terminate_on_dummy_leaf=False)
    if name == "capex_epicycles":
        rng = rng if rng is not None else random.Random()
        return sddp.Historical([sample_epicycles_scenario(rng) for _ in range(replications)])
    return None


def run_capex(
    name: str,
    iteration_limit: int | None = None,
    replications: int = 100,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train and simulate one capacity-expansion model as on the page."""
    build, iters = CAPEX_MODELS[name]
    model = build()
    sddp.train(model, iteration_limit=iteration_limit or iters, seed=seed, **kwargs)
    variables = ["x_storage", "u_flow", "x_reservoir_max", "x_flow_max"]
    if name == "capex_operational":
        variables = ["x_storage", "u_flow"]
    elif name == "capex_invest_then_operate":
        variables = ["x_storage", "u_flow", "x_reservoir_max"]
    scheme = capex_sampling_scheme(name, random.Random(seed), replications)
    simulations = sddp.simulate(model, replications, variables, sampling_scheme=scheme, seed=seed)
    return {"model": model, "bound": sddp.calculate_bound(model), "simulations": simulations}


def plot_capex(simulations: Sequence[Sequence[dict[str, Any]]], filename: str | None = None) -> Any:
    """The ``(2, 2)`` publication plot of storage / hydro / reservoir max / flow max (panels
    whose variables were not recorded are skipped)."""
    import matplotlib.pyplot as plt

    panels = [
        ("Storage", lambda sim: sim["x_storage"].out),
        ("Hydro", lambda sim: sim["u_flow"]),
        ("Reservoir Max", lambda sim: sim["x_reservoir_max"].out),
        ("Flow Max", lambda sim: sim["x_flow_max"].out),
    ]
    fig, axes = plt.subplots(2, 2)
    for ax, (ylabel, fn) in zip(axes.flat, panels):
        try:
            fn(simulations[0][0])
        except KeyError:
            ax.set_visible(False)
            continue
        sddp.publication_plot(simulations, fn, ax=ax, ylabel=ylabel)
    if filename is not None:
        fig.savefig(filename)
    return fig


def main() -> None:
    lp = solve_deterministic_lp()
    print("LP objective:", lp["objective"])
    det = run_deterministic_sddp(seed=1)
    print("deterministic SDDP bound after 10 iterations:", det["bound"])
    print("stage 10 of the first replication:", det["simulations"][0][9])
    sto = run_stochastic_sddp(seed=1, spaghetti_file="spaghetti_plot.html")
    print("stochastic SDDP bound after 100 iterations:", sto["bound"])
    print(repr(sddp.UnicyclicGraph(0.7, num_nodes=2)))
    cyc = run_cyclic_sddp(seed=1)
    print("cyclic bound:", cyc["bound"], "lengths:", cyc["free_lengths"], set(cyc["fixed_lengths"]))
    for name in CAPEX_MODELS:
        r = run_capex(name, seed=1)
        print(f"{name}: bound {r['bound']:.4f}, {len(r['simulations'])} simulations")


if __name__ == "__main__":
    main()
