"""Python translations of further SDDP.jl documentation examples (docs/src/examples).

Each ``build_*`` returns a fresh model; ``tests/test_examples.py`` checks them against the
values SDDP.jl produces (``reference/oracle/examples.json``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

import sddp


# --------------------------------------------------------------- agriculture_mccardle_farm
def build_mccardle_farm() -> sddp.PolicyGraph:
    S = [[0, 1, 2], [0, 0, 1], [0, 0, 0]]  # S[cutting][stage]
    t = [60, 60, 245]
    D = [210, 210, 858]
    q = [4.5, 5.5, 6.5]  # per cutting (constant over stage/weather)
    b = [[30, 75, 37.5], [15, 37.5, 18.25], [7.5, 18.75, 9.325]]  # b[stage][weather]
    w = 3000
    C = 50.0
    r = [5, 6, 7]  # per cutting
    M, H, V, L = 60.0, 0.0, [0.05, 0.05, 0.05], 3000.0
    graph = sddp.MarkovianGraph(
        [
            [[1.0]],
            [[0.14, 0.69, 0.17]],
            [[0.14, 0.69, 0.17]] * 3,
            [[0.14, 0.69, 0.17]] * 3,
        ]
    )

    def builder(sp: sddp.Subproblem, index: tuple[int, int]) -> None:
        stage, weather = index
        acres = sp.add_state("acres", lb=0.0, ub=M, initial_value=M)
        bales = [
            sp.add_state(f"bales[{i + 1}]", lb=0.0, initial_value=H if i == 0 else 0.0)
            for i in range(3)
        ]
        buy = [sp.add_variable(f"buy[{i + 1}]", lb=0.0) for i in range(3)]
        sell = [sp.add_variable(f"sell[{i + 1}]", lb=0.0) for i in range(3)]
        eat = [sp.add_variable(f"eat[{i + 1}]", lb=0.0) for i in range(3)]
        pen_p = [sp.add_variable(f"pen_p[{i + 1}]", lb=0.0) for i in range(3)]
        pen_n = [sp.add_variable(f"pen_n[{i + 1}]", lb=0.0) for i in range(3)]
        if stage == 1:
            sp.add_constraint(acres.out <= acres.in_)
            for i in range(3):
                sp.add_constraint(bales[i].in_ == bales[i].out)
            sp.set_stage_objective(0.0)
        else:
            s = stage - 2  # 0-based previous stage
            cut_ex = [
                bales[c].in_ + buy[c] - eat[c] - sell[c] + pen_p[c] - pen_n[c] for c in range(3)
            ]
            sp.add_constraint(acres.out <= acres.in_)
            sp.add_constraint(sum(eat) >= D[s])
            sp.add_constraint(bales[s].out == cut_ex[s] + acres.in_ * b[s][weather - 1])
            for c in range(3):
                if c != s:
                    sp.add_constraint(bales[c].out == cut_ex[c])
            sp.add_constraint(sum(bl.out for bl in bales) <= w)
            for c in range(3):
                sp.add_constraint(sell[c] <= bales[c].in_)
            sp.add_constraint(sum(sell) <= L)
            sp.set_stage_objective(
                1000 * (sum(pen_p) + sum(pen_n))
                + C * acres.in_
                + sum(
                    V[s] * bales[c].in_ * t[s] + r[c] * buy[c] + S[c][s] * eat[c] - q[c] * sell[c]
                    for c in range(3)
                )
            )

    return sddp.PolicyGraph(builder, graph, lower_bound=0.0, optimizer=sddp.HiGHS)


# ------------------------------------------------------------------- generation_expansion
def build_generation_expansion() -> sddp.PolicyGraph:
    build_cost, use_cost, num_units = 1e4, 4, 5
    capacities = [1.0] * num_units
    demand_vals = 0.5 * np.array(
        [
            [5, 5, 5, 5, 5, 5, 5, 5],
            [4, 3, 1, 3, 0, 9, 8, 17],
            [0, 9, 4, 2, 19, 19, 13, 7],
            [25, 11, 4, 14, 4, 6, 15, 12],
            [6, 7, 5, 3, 8, 4, 17, 13],
        ],
        dtype=float,
    )
    penalty, rho = 5e5, 0.99

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        invested = [
            sp.add_state(f"invested[{i + 1}]", lb=0.0, ub=1.0, integer=True, initial_value=0.0)
            for i in range(num_units)
        ]
        generation = sp.add_variable("generation", lb=0.0)
        unmet = sp.add_variable("unmet", lb=0.0)
        demand = sp.add_variable("demand")
        for i in range(num_units):
            sp.add_constraint(invested[i].out >= invested[i].in_)
        sp.add_constraint(
            sum(capacities[i] * invested[i].out for i in range(num_units)) >= generation
        )
        sp.add_constraint(unmet >= demand - generation)
        for j in range(num_units - 1):
            sp.add_constraint(invested[j].out <= invested[j + 1].out)
        sp.parameterize(lambda w: sp.fix(demand, w), list(demand_vals[stage - 1, :]))
        investment_cost = build_cost * sum(
            invested[i].out - invested[i].in_ for i in range(num_units)
        )
        sp.set_stage_objective(
            (investment_cost + generation * use_cost) * rho ** (stage - 1) + penalty * unmet
        )

    return sddp.LinearPolicyGraph(builder, stages=5, lower_bound=0.0, optimizer=sddp.HiGHS)


# ------------------------------------------------------------------------- hydro_valley
@dataclass
class Turbine:
    flowknots: list[float]
    powerknots: list[float]


@dataclass
class Reservoir:
    min: float
    max: float
    initial: float
    turbine: Turbine
    spill_cost: float
    inflows: list[float]


def hydro_valley_model(
    hasstagewiseinflows: bool = True, hasmarkovprice: bool = True, sense: str = "Max"
) -> sddp.PolicyGraph:
    valley_chain = [
        Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 20, 50]),
        Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 0, 20]),
    ]
    prices = [[1, 2, 0], [2, 1, 0], [3, 4, 0]]
    if hasmarkovprice:
        transition = [[[1.0]], [[0.6, 0.4]], [[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]]]
    else:
        transition = [[[1.0]] for _ in range(3)]
    flipobj = 1.0 if sense == "Max" else -1.0
    lower = -math.inf if sense == "Max" else -1e6
    upper = 1e6 if sense == "Max" else math.inf
    N = len(valley_chain)

    def builder(sp: sddp.Subproblem, node: tuple[int, int]) -> None:
        t, markov_state = node
        reservoir = [
            sp.add_state(
                f"reservoir[{r + 1}]",
                lb=valley_chain[r].min,
                ub=valley_chain[r].max,
                initial_value=valley_chain[r].initial,
            )
            for r in range(N)
        ]
        outflow = [sp.add_variable(f"outflow[{r + 1}]", lb=0.0) for r in range(N)]
        spill = [sp.add_variable(f"spill[{r + 1}]", lb=0.0) for r in range(N)]
        inflow = [sp.add_variable(f"inflow[{r + 1}]", lb=0.0) for r in range(N)]
        generation_quantity = sp.add_variable("generation_quantity", lb=0.0)
        dispatch = [
            [
                sp.add_variable(f"dispatch[{r + 1},{lv + 1}]", lb=0.0, ub=1.0)
                for lv in range(len(valley_chain[r].turbine.flowknots))
            ]
            for r in range(N)
        ]
        rainfall = [sp.add_variable(f"rainfall[{i + 1}]") for i in range(N)]
        sp.add_constraint(reservoir[0].out == reservoir[0].in_ + inflow[0] - outflow[0] - spill[0])
        for i in range(1, N):
            sp.add_constraint(
                reservoir[i].out
                == reservoir[i].in_
                + inflow[i]
                - outflow[i]
                - spill[i]
                + outflow[i - 1]
                + spill[i - 1]
            )
        sp.add_constraint(
            generation_quantity
            == sum(
                valley_chain[r].turbine.powerknots[lv] * dispatch[r][lv]
                for r in range(N)
                for lv in range(len(valley_chain[r].turbine.powerknots))
            )
        )
        for r in range(N):
            sp.add_constraint(
                outflow[r]
                == sum(
                    valley_chain[r].turbine.flowknots[lv] * dispatch[r][lv]
                    for lv in range(len(valley_chain[r].turbine.flowknots))
                )
            )
            sp.add_constraint(sum(dispatch[r]) <= 1)
        if hasstagewiseinflows and t > 1:
            for i in range(N):
                sp.add_constraint(inflow[i] <= rainfall[i])

            def modify(w: tuple[float, float]) -> None:
                for i in range(N):
                    sp.fix(rainfall[i], w[i])

            sp.parameterize(
                modify,
                [
                    (valley_chain[0].inflows[i], valley_chain[1].inflows[i])
                    for i in range(len(transition))
                ],
            )
        else:
            for i in range(N):
                sp.add_constraint(inflow[i] <= valley_chain[i].inflows[0])
        price = prices[t - 1][markov_state - 1] if hasmarkovprice else prices[t - 1][0]
        sp.set_stage_objective(
            flipobj
            * (
                price * generation_quantity
                - sum(valley_chain[i].spill_cost * spill[i] for i in range(N))
            )
        )

    return sddp.MarkovianPolicyGraph(
        builder,
        sense=sense,
        lower_bound=lower,
        upper_bound=upper,
        transition_matrices=transition,
        optimizer=sddp.HiGHS,
    )


# --------------------------------------------------------------------- booking_management
def booking_management_model(num_days: int, num_rooms: int, num_requests: int) -> sddp.PolicyGraph:
    max_revenue = (num_rooms + num_requests) * num_days * num_rooms
    booking_requests = []
    for room in range(num_rooms):
        for day in range(num_days):
            for length_of_stay in range(num_days - day):
                req = np.zeros((num_rooms, num_days), dtype=int)
                req[room, day : day + length_of_stay + 1] = 1
                booking_requests.append(req)

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        vacancy = [
            [
                sp.add_state(
                    f"vacancy[{r + 1},{d + 1}]", lb=0.0, ub=1.0, binary=True, initial_value=1.0
                )
                for d in range(num_days)
            ]
            for r in range(num_rooms)
        ]
        accept_request = sp.add_variable("accept_request", lb=0.0, ub=1.0, binary=True)
        rra = [
            [
                sp.add_variable(
                    f"room_request_accepted[{r + 1},{d + 1}]", lb=0.0, ub=1.0, binary=True
                )
                for d in range(num_days)
            ]
            for r in range(num_rooms)
        ]
        req = [
            [sp.add_variable(f"req[{r + 1},{d + 1}]") for d in range(num_days)]
            for r in range(num_rooms)
        ]
        for r in range(num_rooms):
            for d in range(num_days):
                sp.add_constraint(vacancy[r][d].out == vacancy[r][d].in_ - rra[r][d])
                sp.add_constraint(rra[r][d] <= vacancy[r][d].in_)
                sp.add_constraint(rra[r][d] <= accept_request)
                sp.add_constraint(rra[r][d] <= req[r][d])
                sp.add_constraint(rra[r][d] + (1 - accept_request) >= req[r][d])

        def modify(request: np.ndarray) -> None:
            for r in range(num_rooms):
                for d in range(num_days):
                    sp.fix(req[r][d], float(request[r, d]))

        sp.parameterize(modify, booking_requests)
        sp.set_stage_objective(
            sum((r + 1 + stage - 1) * rra[r][d] for r in range(num_rooms) for d in range(num_days))
        )

    return sddp.LinearPolicyGraph(
        builder, stages=num_requests, upper_bound=max_revenue, sense="Max", optimizer=sddp.HiGHS
    )


# ------------------------------------------------------------ StructDualDynProg prob5.2
def build_prob52(stages: int) -> sddp.PolicyGraph:
    n, m = 4, 3
    i_c = [16, 5, 32, 2]
    C = [25, 80, 6.5, 160]
    T = [8760 / 8760, 7000 / 8760, 1500 / 8760]
    D2 = np.column_stack([np.diff([0, 3919, 7329, 10315]), np.diff([0, 7086, 9004, 11169])])
    p2 = [0.9, 0.1]
    pen = 1e6 if stages == 2 else 1e5

    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = [sp.add_state(f"x[{i + 1}]", lb=0.0, initial_value=0.0) for i in range(n)]
        y = [[sp.add_variable(f"y[{i + 1},{j + 1}]", lb=0.0) for j in range(m)] for i in range(n)]
        v = [sp.add_variable(f"v[{i + 1}]", lb=0.0) for i in range(n)]
        penalty = sp.add_variable("penalty", lb=0.0)
        xi = [sp.add_variable(f"ξ[{j + 1}]") for j in range(m)]
        for i in range(n):
            sp.add_constraint(x[i].out == x[i].in_ + v[i])
            sp.add_constraint(sum(y[i]) <= x[i].in_)
        for j in range(m):
            sp.add_constraint(sum(y[i][j] for i in range(n)) + penalty >= xi[j])
        sp.set_stage_objective(
            sum(i_c[i] * v[i] for i in range(n))
            + sum(C[i] * y[i][j] * T[j] for i in range(n) for j in range(m))
            + pen * penalty
        )
        if t != 1:

            def modify(w: int) -> None:
                for j in range(m):
                    sp.fix(xi[j], float(D2[j, w - 1]))

            sp.parameterize(modify, [1, 2], p2)
        if t == stages:
            sp.add_constraint(sum(v) == 0)

    return sddp.LinearPolicyGraph(builder, stages=stages, lower_bound=0.0, optimizer=sddp.HiGHS)


# ------------------------------------------------------------------------- multistock
def build_multistock() -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, stage: int) -> None:
        stock = [
            sp.add_state(f"stock[{i + 1}]", lb=0.0, ub=1.0, initial_value=0.5) for i in range(3)
        ]
        control = [sp.add_variable(f"control[{i + 1}]", lb=0.0, ub=0.5) for i in range(3)]
        xi = [sp.add_variable(f"ξ[{i + 1}]") for i in range(3)]
        sp.add_constraint(sum(control) - 0.5 * 3 <= 0)
        for i in range(3):
            sp.add_constraint(stock[i].out == stock[i].in_ + control[i] - xi[i])
        omega = [tuple(w) for w in itertools.product((0.0, 0.15, 0.3), repeat=3)]

        def modify(w: tuple[float, float, float]) -> None:
            for i in range(3):
                sp.fix(xi[i], w[i])

        sp.parameterize(modify, omega)
        sp.set_stage_objective((math.sin(3 * stage) - 1) * sum(control))

    return sddp.LinearPolicyGraph(builder, stages=5, lower_bound=-5.0, optimizer=sddp.HiGHS)


# --------------------------------------------------------------------------- all_blacks
def build_all_blacks() -> sddp.PolicyGraph:
    T, N = 3, 2
    R = [[3, 3, 6], [3, 3, 6]]
    offer = [[1, 1, 0], [1, 0, 1]]

    def builder(sp: sddp.Subproblem, stage: int) -> None:
        x = [
            sp.add_state(f"x[{i + 1}]", lb=0.0, ub=1.0, binary=True, initial_value=1.0)
            for i in range(N)
        ]
        accept = sp.add_variable("accept_offer", binary=True)
        for i in range(N):
            sp.add_constraint(x[i].out == x[i].in_ - offer[i][stage - 1] * accept)
        sp.set_stage_objective(
            sum(R[i][stage - 1] * offer[i][stage - 1] * accept for i in range(N))
        )

    return sddp.LinearPolicyGraph(
        builder, stages=T, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS
    )


# --------------------------------------------------------------- air_conditioning_forward
def create_air_conditioning_model(convex: bool) -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = sp.add_state("x", lb=0.0, ub=100.0, initial_value=0.0, integer=not convex)
        u_production = sp.add_variable("u_production", lb=0.0, ub=200.0, integer=not convex)
        u_overtime = sp.add_variable("u_overtime", lb=0.0, integer=not convex)
        demand = sp.add_constraint(x.in_ - x.out + u_production + u_overtime == 0, name="demand")
        omega = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        sp.parameterize(lambda w: sp.set_normalized_rhs(demand, w), omega[t - 1])
        sp.set_stage_objective(100 * u_production + 300 * u_overtime + 50 * x.out)

    return sddp.LinearPolicyGraph(builder, stages=3, lower_bound=0.0, optimizer=sddp.HiGHS)


# ---------------------------------------------------------------------- sldp_example_two
def build_sldp_two(N: int) -> sddp.PolicyGraph:
    def builder(sp: sddp.Subproblem, t: int) -> None:
        x = [sp.add_state(f"x[{i + 1}]", lb=0.0, ub=5.0, initial_value=0.0) for i in range(2)]
        if t == 1:
            u = [sp.add_variable(f"u[{i + 1}]", lb=0.0, ub=5.0, integer=True) for i in range(2)]
            for i in range(2):
                sp.add_constraint(u[i] == x[i].out)
            sp.set_stage_objective(-1.5 * x[0].out - 4 * x[1].out)
        else:
            y = [sp.add_variable(f"y[{i + 1}]", lb=0.0, ub=1.0, binary=True) for i in range(4)]
            w = [sp.add_variable(f"ω[{i + 1}]") for i in range(2)]
            sp.set_stage_objective(-16 * y[0] - 19 * y[1] - 23 * y[2] - 28 * y[3])
            sp.add_constraint(2 * y[0] + 3 * y[1] + 4 * y[2] + 5 * y[3] <= w[0] - x[0].in_)
            sp.add_constraint(6 * y[0] + 1 * y[1] + 3 * y[2] + 2 * y[3] <= w[1] - x[1].in_)
            steps = list(np.linspace(5, 15, N))

            def modify(phi: list[float]) -> None:
                sp.fix(w[0], phi[0])
                sp.fix(w[1], phi[1])

            sp.parameterize(modify, [[i, j] for i in steps for j in steps])

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=-100.0, optimizer=sddp.HiGHS)
