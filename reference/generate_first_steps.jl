# Oracle for the post-training sections of https://sddp.dev/stable/tutorial/first_steps/
# (decision rules, simulation, confidence interval, custom recorders, value function) and the
# Historical simulation of https://sddp.dev/stable/tutorial/markov_uncertainty/.
# Usage: julia --project=. generate_first_steps.jl
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics

results = Dict{String,Any}("versions" => versions())

# Train with the same fixed Historical scenarios as `deterministic_run_20` so that the trained
# policy is identical in Julia and Python (cuts match to 1e-6, see tests/test_parity.py).
run = JSON.parsefile(joinpath(OUT, "hydro_thermal.json"))["deterministic_run_20"]
scenarios = [[(Int(n), Float64(w)) for (n, w) in s] for s in run["scenarios"]]
model = build_hydro_thermal()
SDDP.train(model; iteration_limit = run["iterations"], print_level = 0, log_file = LOGFILE,
           sampling_scheme = SDDP.Historical(scenarios), run_numerical_stability_report = false)
results["bound"] = SDDP.calculate_bound(model)

# Obtaining the decision rule
rule = SDDP.DecisionRule(model; node = 1)
solution = SDDP.evaluate(rule; incoming_state = Dict(:volume => 150.0), noise = 50.0,
                         controls_to_record = [:hydro_generation, :thermal_generation])
results["decision_rule"] = Dict(
    "stage_objective" => solution.stage_objective,
    "outgoing_volume" => solution.outgoing_state[:volume],
    "hydro_generation" => solution.controls[:hydro_generation],
    "thermal_generation" => solution.controls[:thermal_generation],
)

# Simulating the policy (a fixed scenario so the values are reproducible) + custom recorders
scenario = [(1, 0.0), (2, 50.0), (3, 100.0)]
sims = SDDP.simulate(model, 1, [:volume, :thermal_generation, :hydro_generation, :hydro_spill];
                     sampling_scheme = SDDP.Historical(scenario),
                     custom_recorders = Dict{Symbol,Function}(:price => (sp) -> dual(sp[:demand_constraint])))
results["historical"] = Dict(
    "scenario" => scenario,
    "outgoing_volume" => [s[:volume].out for s in sims[1]],
    "thermal_generation" => [s[:thermal_generation] for s in sims[1]],
    "stage_objective" => [s[:stage_objective] for s in sims[1]],
    "price" => [s[:price] for s in sims[1]],
)

# Obtaining bounds: Monte Carlo confidence interval
results["simulation_100"] = simulate_stats(model, 100; seed = 42)

# Extracting the marginal water values
V = SDDP.ValueFunction(model; node = 1)
cost, price = SDDP.evaluate(V, Dict("volume" => 10))
results["value_function_at_10"] = Dict("cost" => cost, "price" => price[:volume])
cost, price = SDDP.evaluate(V, Dict("volume" => 150))
results["value_function_at_150"] = Dict("cost" => cost, "price" => price[:volume])

# markov_uncertainty.jl: Historical simulation visits the requested nodes.
Ω = Ω_MARKOV
mm = build_markov_uncertainty()
SDDP.train(mm; iteration_limit = 40, print_level = 0, log_file = LOGFILE, run_numerical_stability_report = false)
msims = SDDP.simulate(mm; sampling_scheme = SDDP.Historical([((1, 1), Ω[1]), ((2, 2), Ω[3]), ((3, 1), Ω[2])]))
results["markov_historical"] = Dict(
    "node_index" => [collect(s[:node_index]) for s in msims[1]],
    "noise_term" => [collect(s[:noise_term]) for s in msims[1]],
)

open(joinpath(OUT, "first_steps.json"), "w") do io
    JSON.print(io, results, 2)
end
println("wrote first_steps.json: bound $(results["bound"])")
