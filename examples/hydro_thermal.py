"""Python translation of SDDP.jl's "first steps" hydro-thermal tutorial.

This file was written *before* the library, to fix the API design. See
docs/src/tutorial/first_steps.jl in SDDP.jl for the original.
"""
import sddp


def subproblem_builder(sp: sddp.Subproblem, node: int) -> None:
    # State variables: `volume` is a State with `.in_` (incoming) and `.out` (outgoing).
    volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
    # Control variables
    thermal_generation = sp.add_variable("thermal_generation", lb=0.0)
    hydro_generation = sp.add_variable("hydro_generation", lb=0.0)
    hydro_spill = sp.add_variable("hydro_spill", lb=0.0)
    # Random variables
    inflow = sp.add_variable("inflow")

    @sp.parameterize([0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3])
    def _(omega: float) -> None:
        sp.fix(inflow, omega)

    # Transition function and constraints
    sp.add_constraint(volume.out == volume.in_ - hydro_generation - hydro_spill + inflow)
    sp.add_constraint(hydro_generation + thermal_generation == 150.0, name="demand_constraint")
    # Stage objective
    fuel_cost = [50.0, 100.0, 150.0]
    sp.set_stage_objective(fuel_cost[node - 1] * thermal_generation)


def main() -> None:
    model = sddp.LinearPolicyGraph(
        subproblem_builder,
        stages=3,
        sense="Min",
        lower_bound=0.0,
        optimizer=sddp.HiGHS,
    )
    sddp.train(model, iteration_limit=10, seed=1234)

    rule = sddp.DecisionRule(model, node=1)
    solution = sddp.evaluate(
        rule,
        incoming_state={"volume": 150.0},
        noise=50.0,
        controls_to_record=["hydro_generation", "thermal_generation"],
    )
    print(solution)

    simulations = sddp.simulate(
        model,
        100,
        ["volume", "thermal_generation", "hydro_generation", "hydro_spill"],
        seed=42,
    )
    replication, stage = 0, 1
    print(simulations[replication][stage])
    outgoing_volume = [node["volume"].out for node in simulations[0]]
    print("outgoing volume:", outgoing_volume)
    objectives = [sum(stage["stage_objective"] for stage in sim) for sim in simulations]
    mu, ci = sddp.confidence_interval(objectives)
    print(f"Confidence interval: {mu} ± {ci}")
    print("Lower bound:", sddp.calculate_bound(model))

    # Custom recorders: record the dual of a named constraint.
    simulations = sddp.simulate(
        model,
        1,
        custom_recorders={"price": lambda sp: sp.dual(sp["demand_constraint"])},
        seed=1,
    )
    print("prices:", [node["price"] for node in simulations[0]])


if __name__ == "__main__":
    main()
