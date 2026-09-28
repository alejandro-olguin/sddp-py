"""Python translations of the reference problems in ``reference/generate.jl``.

Each builder returns a fresh :class:`sddp.PolicyGraph`. Keep these in sync with the
Julia definitions; the oracle JSON in ``reference/oracle`` is the ground truth.
"""

from __future__ import annotations

import math
from typing import Any

import sddp

# Optimizer used by every builder. Tests may swap it (e.g. for Julia-equivalent solver
# tolerances) via `set_optimizer`.
_OPTIMIZER = sddp.HiGHS


def set_optimizer(factory: sddp.OptimizerFactory) -> None:
    global _OPTIMIZER
    _OPTIMIZER = factory


def _optimizer() -> sddp.OptimizerFactory:
    return _OPTIMIZER


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def build_hydro_thermal() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.parameterize(lambda w: sp.fix(inflow, w), [0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3])
        sp.add_constraint(volume.out == volume.in_ - hydro - spill + inflow)
        sp.add_constraint(hydro + thermal == 150.0, name="demand_constraint")
        fuel_cost = [50.0, 100.0, 150.0]
        sp.set_stage_objective(fuel_cost[t - 1] * thermal)

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=0.0, optimizer=_optimizer()
    )


def build_fast_quickstart() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, initial_value=0.0)
        if t == 1:
            sp.set_stage_objective(x.out)
        else:
            s = sp.add_variable("s", lb=0.0)
            sp.add_constraint(s <= x.in_)
            sp.parameterize(lambda w: sp.set_upper_bound(s, w), [2, 3])
            sp.set_stage_objective(-2 * s)

    return sddp.PolicyGraph(builder, sddp.LinearGraph(2), lower_bound=-5, optimizer=_optimizer())


def build_fast_hydro_thermal() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=8.0, initial_value=0.0)
        y = sp.add_variable("y", lb=0.0)
        p = sp.add_variable("p", lb=0.0)
        xi = sp.add_variable("ξ")
        sp.add_constraint(p + y >= 6)
        sp.add_constraint(x.out <= x.in_ - y + xi)
        rainfall = [6] if t == 1 else [2, 10]
        sp.parameterize(lambda w: sp.fix(xi, w), rainfall)
        sp.set_stage_objective(-5 * p)

    return sddp.LinearPolicyGraph(
        builder, stages=2, upper_bound=0.0, sense="Max", optimizer=_optimizer()
    )


def build_fast_production_management() -> sddp.PolicyGraph:
    DEMAND = [2, 10]
    H, N = 3, 2
    C = [0.2, 0.7]
    S = [2 + 0.33, 2 + 0.54]

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = [sp.add_state(f"x[{i + 1}]", lb=0.0, initial_value=0.0) for i in range(N)]
        s = [sp.add_variable(f"s[{i + 1}]", lb=0.0) for i in range(N)]
        d = sp.add_variable("d")
        for i in range(N):
            sp.add_constraint(s[i] <= x[i].in_)
        sp.add_constraint(sum(s) <= d)
        sp.parameterize(lambda w: sp.fix(d, w), [0] if t == 1 else DEMAND)
        sp.set_stage_objective(
            sum(C[i] * x[i].out for i in range(N)) - sum(S[i] * s[i] for i in range(N))
        )

    return sddp.LinearPolicyGraph(builder, stages=H, lower_bound=-50.0, optimizer=_optimizer())


def build_stock_example() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, stage: int) -> None:
        state = sp.add_state("state", lb=0.0, ub=1.0, initial_value=0.5)
        control = sp.add_variable("control", lb=0.0, ub=0.5)
        xi = sp.add_variable("ξ")
        sp.add_constraint(state.out == state.in_ - control + xi)
        sp.parameterize(lambda w: sp.fix(xi, w), [i * (1 / 30) for i in range(10)])
        sp.set_stage_objective((math.sin(3 * stage) - 1) * control)

    return sddp.PolicyGraph(builder, sddp.LinearGraph(5), lower_bound=-2, optimizer=_optimizer())


def build_farmers() -> sddp.PolicyGraph:
    MAX_AREA = 500.0
    CROPS = ["wheat", "corn", "sugar_beet"]
    PLANTING_COST = {"wheat": 150.0, "corn": 230.0, "sugar_beet": 260.0}
    MIN_QUANTITIES = {"wheat": 200.0, "corn": 240.0, "sugar_beet": 0.0}
    QUOTA_MAX = {"wheat": math.inf, "corn": math.inf, "sugar_beet": 6_000.0}
    SELL_IN_QUOTA = {"wheat": 170.0, "corn": 150.0, "sugar_beet": 36.0}
    SELL_NO_QUOTA = {"wheat": 0.0, "corn": 0.0, "sugar_beet": 10.0}
    BUY_PRICE = {"wheat": 238.0, "corn": 210.0, "sugar_beet": 1_000.0}
    MEAN_YIELD = {"wheat": 2.5, "corn": 3.0, "sugar_beet": 20.0}
    YIELD_MULTIPLIER = {"good": 1.2, "fair": 1.0, "bad": 0.8}

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        area = {c: sp.add_state(f"area[{c}]", lb=0.0, initial_value=0.0) for c in CROPS}
        if stage == 1:
            sp.add_constraint(sum(area[c].out for c in CROPS) <= MAX_AREA)
            sp.set_stage_objective(-sum(PLANTING_COST[c] * area[c].out for c in CROPS))
        else:
            yield_ = {c: sp.add_variable(f"yield[{c}]", lb=0.0) for c in CROPS}
            buy = {c: sp.add_variable(f"buy[{c}]", lb=0.0) for c in CROPS}
            sell_in = {
                c: sp.add_variable(f"sell_in_quota[{c}]", lb=0.0, ub=QUOTA_MAX[c]) for c in CROPS
            }
            sell_no = {c: sp.add_variable(f"sell_no_quota[{c}]", lb=0.0) for c in CROPS}
            for c in CROPS:
                sp.add_constraint(yield_[c] + buy[c] - sell_in[c] - sell_no[c] >= MIN_QUANTITIES[c])
            uncertainty = {
                c: sp.add_constraint(1.0 * area[c].in_ - yield_[c] == 0.0) for c in CROPS
            }

            def modify(w: str) -> None:
                for c in CROPS:
                    sp.set_normalized_coefficient(
                        uncertainty[c], area[c].in_, MEAN_YIELD[c] * YIELD_MULTIPLIER[w]
                    )

            sp.parameterize(modify, ["good", "fair", "bad"])
            sp.set_stage_objective(
                sum(
                    SELL_IN_QUOTA[c] * sell_in[c]
                    + SELL_NO_QUOTA[c] * sell_no[c]
                    - BUY_PRICE[c] * buy[c]
                    for c in CROPS
                )
            )

    return sddp.LinearPolicyGraph(
        builder, stages=2, sense="Max", upper_bound=500_000.0, optimizer=_optimizer()
    )


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
OMEGA_MARKOV = [
    {"inflow": 0.0, "fuel_multiplier": 1.5},
    {"inflow": 50.0, "fuel_multiplier": 1.0},
    {"inflow": 100.0, "fuel_multiplier": 0.75},
]


def _hydro_markov_body(sp: sddp.Subproblem, t: int, probability: list[float]) -> None:
    volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
    thermal = sp.add_variable("thermal_generation", lb=0.0)
    hydro = sp.add_variable("hydro_generation", lb=0.0)
    spill = sp.add_variable("hydro_spill", lb=0.0)
    inflow = sp.add_variable("inflow")
    sp.add_constraint(volume.out == volume.in_ + inflow - hydro - spill)
    sp.add_constraint(thermal + hydro == 150.0)
    fuel_cost = [50.0, 100.0, 150.0]

    def modify(w: dict[str, float]) -> None:
        sp.fix(inflow, w["inflow"])
        sp.set_stage_objective(w["fuel_multiplier"] * fuel_cost[t - 1] * thermal)

    sp.parameterize(modify, OMEGA_MARKOV, probability)


def build_markov_uncertainty() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, markov_state = node
        probability = [1 / 6, 1 / 3, 1 / 2] if markov_state == 1 else [1 / 2, 1 / 3, 1 / 6]
        _hydro_markov_body(sp, t, probability)

    return sddp.MarkovianPolicyGraph(
        builder,
        transition_matrices=[[[1.0]], [[0.75, 0.25]], [[0.75, 0.25], [0.25, 0.75]]],
        sense="Min",
        lower_bound=0.0,
        optimizer=_optimizer(),
    )


def build_objective_uncertainty() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        _hydro_markov_body(sp, t, [1 / 3, 1 / 3, 1 / 3])

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=0.0, optimizer=_optimizer()
    )


def build_infinite_trivial() -> sddp.PolicyGraph:
    graph = sddp.Graph.from_edges(
        "root_node", ["week"], [(("root_node", "week"), 1.0), (("week", "week"), 0.9)]
    )

    def builder(sp: sddp.Subproblem, node: str) -> None:
        state = sp.add_state("state", initial_value=0.0)
        sp.add_constraint(state.in_ == state.out)
        sp.set_stage_objective(2.0)

    return sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=_optimizer())


def build_no_strong_duality() -> sddp.PolicyGraph:
    graph = sddp.Graph.from_edges(
        "root", ["node"], [(("root", "node"), 1.0), (("node", "node"), 0.5)]
    )

    def builder(sp: sddp.Subproblem, node: str) -> None:
        x = sp.add_state("x", initial_value=1.0)
        sp.set_stage_objective(x.out)
        sp.add_constraint(x.in_ == x.out)

    return sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=_optimizer())


def build_infinite_hydro_thermal() -> sddp.PolicyGraph:
    omega = [
        {"inflow": 0.0, "demand": 7.5},
        {"inflow": 5.0, "demand": 5.0},
        {"inflow": 10.0, "demand": 2.5},
    ]
    graph = sddp.Graph.from_edges(
        "root_node", ["week"], [(("root_node", "week"), 1.0), (("week", "week"), 0.9)]
    )

    def builder(sp: sddp.Subproblem, node: str) -> None:
        reservoir = sp.add_state("reservoir", lb=5.0, ub=15.0, initial_value=10.0)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        demand = sp.add_variable("demand")
        sp.add_constraint(reservoir.out == reservoir.in_ - hydro - spill + inflow)
        sp.add_constraint(hydro + thermal == demand)
        sp.set_stage_objective(10 * spill + thermal)

        def modify(w: dict[str, float]) -> None:
            sp.fix(inflow, w["inflow"])
            sp.fix(demand, w["demand"])

        sp.parameterize(modify, omega)

    return sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=_optimizer())


def build_asset_management_stagewise() -> sddp.PolicyGraph:
    w_s = [1.25, 1.06]
    w_b = [1.14, 1.12]
    Phi = [-1, 5]
    Psi = [0.02, 0.0]

    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, i = node
        xs = sp.add_state("xs", lb=0.0, initial_value=0.0)
        xb = sp.add_state("xb", lb=0.0, initial_value=0.0)
        if t == 1:
            sp.add_constraint(xs.out + xb.out == 55 + xs.in_ + xb.in_)
            sp.set_stage_objective(0.0)
        elif t in (2, 3):
            phi = sp.add_variable("phi")
            sp.add_constraint(w_s[i - 1] * xs.in_ + w_b[i - 1] * xb.in_ + phi == xs.out + xb.out)

            def modify(w: int) -> None:
                sp.fix(phi, Phi[w - 1])
                sp.set_stage_objective(Psi[w - 1] * xs.out)

            sp.parameterize(modify, [1, 2], [0.6, 0.4])
        else:
            u = sp.add_variable("u", lb=0.0)
            v = sp.add_variable("v", lb=0.0)
            sp.add_constraint(w_s[i - 1] * xs.in_ + w_b[i - 1] * xb.in_ + u - v == 80)
            sp.set_stage_objective(-4 * u + v)

    return sddp.MarkovianPolicyGraph(
        builder,
        sense="Max",
        transition_matrices=[
            [[1.0]],
            [[0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        upper_bound=1000.0,
        optimizer=_optimizer(),
    )


def asset_risk(node: tuple[int, int]) -> sddp.Expectation | sddp.ConvexCombination:
    return sddp.Expectation() if node[0] != 3 else sddp.EAVaR(lambda_=0.5, beta=0.5)


def build_objective_states() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.add_constraint(volume.out == volume.in_ + inflow - hydro - spill)
        sp.add_constraint(thermal + hydro == 150.0, name="demand_constraint")
        sp.add_objective_state(
            lambda fuel_cost, w: w["fuel"] * fuel_cost,
            initial_value=50.0,
            lipschitz=10_000.0,
            lower_bound=50.0,
            upper_bound=150.0,
        )
        omega = [
            {"fuel": f, "inflow": w} for f in [0.75, 0.9, 1.1, 1.25] for w in [0.0, 50.0, 100.0]
        ]

        def modify(w: dict[str, float]) -> None:
            fuel_cost = sp.objective_state()
            sp.set_stage_objective(fuel_cost * thermal)
            sp.fix(inflow, w["inflow"])

        sp.parameterize(modify, omega)

    return sddp.LinearPolicyGraph(
        builder, stages=3, sense="Min", lower_bound=0.0, optimizer=_optimizer()
    )


def build_belief() -> sddp.PolicyGraph:
    demand_values = [1.0, 2.0]
    demand_prob = {"Ah": [0.2, 0.8], "Bh": [0.8, 0.2]}
    graph = sddp.Graph.from_edges(
        "root_node",
        ["Ad", "Ah", "Bd", "Bh"],
        [
            (("root_node", "Ad"), 0.5),
            (("root_node", "Bd"), 0.5),
            (("Ad", "Ah"), 1.0),
            (("Ah", "Ad"), 0.8),
            (("Ah", "Bd"), 0.1),
            (("Bd", "Bh"), 1.0),
            (("Bh", "Bd"), 0.8),
            (("Bh", "Ad"), 0.1),
        ],
    )
    graph.add_ambiguity_set(["Ad", "Bd"], 1e2)
    graph.add_ambiguity_set(["Ah", "Bh"], 1e2)

    def builder(sp: sddp.Subproblem, node: str) -> None:
        inventory = sp.add_state("inventory", lb=0.0, ub=2.0, initial_value=0.0)
        buy = sp.add_variable("buy", lb=0.0)
        demand = sp.add_variable("demand")
        sp.add_constraint(demand == inventory.in_ - inventory.out + buy)
        if node in ("Ad", "Bd"):
            sp.fix(demand, 0.0)
            sp.set_stage_objective(buy)
        else:
            sp.parameterize(lambda w: sp.fix(demand, w), demand_values, demand_prob[node])
            sp.set_stage_objective(2 * buy + inventory.out)

    return sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=_optimizer())


# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------
def build_air_conditioning() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, stage: int) -> None:
        stored = sp.add_state(
            "stored_production", lb=0.0, ub=100.0, integer=True, initial_value=0.0
        )
        production = sp.add_variable("production", lb=0.0, ub=200.0, integer=True)
        overtime = sp.add_variable("overtime", lb=0.0, integer=True)
        demand = sp.add_variable("demand")
        DEMAND = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        sp.parameterize(lambda w: sp.fix(demand, w), DEMAND[stage - 1])
        sp.add_constraint(stored.out == stored.in_ + production + overtime - demand)
        sp.set_stage_objective(100 * production + 300 * overtime + 50 * stored.out)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=_optimizer())


def build_stochastic_all_blacks() -> sddp.PolicyGraph:
    T, N = 3, 2
    R = [[3, 3, 6], [3, 3, 6]]
    offers = [[[1, 1], [0, 0], [1, 1]], [[1, 0], [0, 0], [0, 0]], [[0, 1], [1, 0], [1, 1]]]

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        x = [
            sp.add_state(f"x[{i + 1}]", lb=0.0, ub=1.0, binary=True, initial_value=1.0)
            for i in range(N)
        ]
        accept = [sp.add_variable(f"accept_offer[{i + 1}]", binary=True) for i in range(N)]
        offers_made = [sp.add_variable(f"offers_made[{i + 1}]") for i in range(N)]
        for i in range(N):
            sp.add_constraint(x[i].in_ - x[i].out == accept[i])
        sp.set_stage_objective(sum(R[i][stage - 1] * accept[i] for i in range(N)))

        def modify(o: list[int]) -> None:
            for i in range(N):
                sp.fix(offers_made[i], o[i])

        sp.parameterize(modify, offers[stage - 1])
        for i in range(N):
            sp.add_constraint(accept[i] <= offers_made[i])

    return sddp.LinearPolicyGraph(
        builder, stages=T, sense="Max", upper_bound=100.0, optimizer=_optimizer()
    )


def build_sldp_example_one() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", initial_value=2.0)
        xp = sp.add_variable("x⁺", lb=0.0)
        xm = sp.add_variable("x⁻", lb=0.0)
        u = sp.add_variable("u", lb=0.0, ub=1.0, binary=True)
        w = sp.add_variable("ω")
        sp.set_stage_objective(0.9 ** (t - 1) * (xp + xm))
        sp.add_constraint(x.out == x.in_ + 2 * u - 1 + w)
        sp.add_constraint(xp >= x.out)
        sp.add_constraint(xm >= -1.0 * x.out)
        points = [
            -0.3089653673606697,
            -0.2718277412744214,
            -0.09611178608243474,
            0.24645863921577763,
            0.5204224537256875,
        ]
        sp.parameterize(lambda phi: sp.fix(w, phi), points + [-p for p in points])

    return sddp.LinearPolicyGraph(builder, stages=8, lower_bound=0.0, optimizer=_optimizer())


BUILDERS: dict[str, Any] = {
    "hydro_thermal": build_hydro_thermal,
    "fast_quickstart": build_fast_quickstart,
    "fast_hydro_thermal": build_fast_hydro_thermal,
    "fast_production_management": build_fast_production_management,
    "stock_example": build_stock_example,
    "farmers": build_farmers,
    "markov_uncertainty": build_markov_uncertainty,
    "objective_uncertainty": build_objective_uncertainty,
    "infinite_trivial": build_infinite_trivial,
    "no_strong_duality": build_no_strong_duality,
    "infinite_hydro_thermal": build_infinite_hydro_thermal,
    "asset_management_stagewise": build_asset_management_stagewise,
    "objective_states": build_objective_states,
    "belief": build_belief,
    "air_conditioning": build_air_conditioning,
    "stochastic_all_blacks": build_stochastic_all_blacks,
    "sldp_example_one": build_sldp_example_one,
}
