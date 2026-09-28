"""Asset management (simple), from https://sddp.dev/stable/examples/asset_management_simple/

A four-stage Markovian policy graph: stage 1 splits 55 between stocks and bonds, stages 2–3
apply Markov-state-dependent returns, stage 4 penalises shortfall against a target of 80.
"""

import sddp


def build_asset_management_simple() -> sddp.PolicyGraph:
    graph = sddp.MarkovianGraph(
        [[[1.0]], [[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]
    )

    def builder(sp: sddp.Subproblem, index: tuple[int, int]) -> None:
        stage, markov_state = index
        r_stock = [1.25, 1.06]
        r_bonds = [1.14, 1.12]
        stocks = sp.add_state("stocks", lb=0.0, initial_value=0.0)
        bonds = sp.add_state("bonds", lb=0.0, initial_value=0.0)
        if stage == 1:
            sp.add_constraint(stocks.out + bonds.out == 55)
            sp.set_stage_objective(0.0)
        elif 1 < stage < 4:
            sp.add_constraint(
                r_stock[markov_state - 1] * stocks.in_ + r_bonds[markov_state - 1] * bonds.in_
                == stocks.out + bonds.out
            )
            sp.set_stage_objective(0.0)
        else:
            over = sp.add_variable("over", lb=0.0)
            short = sp.add_variable("short", lb=0.0)
            sp.add_constraint(
                r_stock[markov_state - 1] * stocks.in_
                + r_bonds[markov_state - 1] * bonds.in_
                - over
                + short
                == 80
            )
            sp.set_stage_objective(-over + 4 * short)

    return sddp.PolicyGraph(builder, graph, lower_bound=-1_000.0, optimizer=sddp.HiGHS)


if __name__ == "__main__":
    model = build_asset_management_simple()
    sddp.train(model, log_frequency=5)
    print("bound:", sddp.calculate_bound(model))  # SDDP.jl documents ≈ 1.514
