# Oracle for the "Explanation" pages (theory_intro.jl, risk.jl).
# Usage: julia --project=. generate_explanation.jl
# Records: SDDP.jl's Entropic/WorstCase/Expectation adjust_probability on the page's data,
# Kelley's cutting plane run on the page's example (JuMP + HiGHS, analytic gradient),
# and SDDP.jl trained bounds for the models the Python vanilla implementation is compared with.
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
const JuMP = SDDP.JuMP
const OUTX = joinpath(@__DIR__, "oracle")
results = Dict{String,Any}("versions" => versions())

# ---- risk.jl: dual risk measures via SDDP.adjust_probability(measure, Q, p, supports, X, is_min)
Z = [1.0, 2.0, 3.0, 4.0]
p = [0.1, 0.2, 0.4, 0.3]
gammas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
entropic = Any[]
for γ in gammas
    local q = zeros(length(p))
    offset = SDDP.adjust_probability(SDDP.Entropic(γ), q, p, Z, Z, true)
    # primal value (1/γ) log Σ p e^{γZ} with BigFloat, as on the page
    primal = Float64(1 / γ * log(sum(p[i] * exp(γ * big(Z[i])) for i in 1:length(p))))
    push!(entropic, Dict("gamma" => γ, "q" => q, "offset" => offset, "primal" => primal))
end
q = zeros(length(p)); off = SDDP.adjust_probability(SDDP.WorstCase(), q, p, Z, Z, true)
worst_case = Dict("q" => copy(q), "offset" => off)
q = zeros(length(p)); off = SDDP.adjust_probability(SDDP.Expectation(), q, p, Z, Z, true)
expectation = Dict("q" => copy(q), "offset" => off)
results["risk"] = Dict("Z" => Z, "p" => p, "entropic" => entropic, "worst_case" => worst_case,
                       "expectation" => expectation)

# ---- risk.jl: subgradient example V(x, ω) = ω x², Ω = [1,2,3], p = [.3,.4,.3], x̃ = [3]
Ω = [1.0, 2.0, 3.0]; pΩ = [0.3, 0.4, 0.3]; x̃ = 3.0
subgrad = Any[]
for γ in gammas
    Vω = [ω * x̃^2 for ω in Ω]
    local qγ = zeros(length(pΩ))
    SDDP.adjust_probability(SDDP.Entropic(γ), qγ, pΩ, Ω, Vω, true)
    push!(subgrad, Dict("gamma" => γ, "dual_subgradient" => sum(qγ[i] * 2 * Ω[i] * x̃ for i in 1:3)))
end
results["subgradient"] = Dict("entropic" => subgrad, "expectation" => 12.0, "worst_case" => 18.0)

# ---- theory_intro.jl: Kelley's cutting plane on (x1 - 1)^2 + (x2 + 2)^2 + 1
function kelleys(f, dfdx; input_dimension, lower_bound, iteration_limit, tolerance = 1e-6)
    K = 0
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, θ >= lower_bound)
    JuMP.@variable(model, x[1:input_dimension])
    JuMP.@objective(model, Min, θ)
    x_k = fill(NaN, input_dimension)
    lb, ub = -Inf, Inf
    status = "iteration limit"
    while true
        JuMP.optimize!(model)
        x_k .= JuMP.value.(x)
        lb = JuMP.objective_value(model)
        ub = min(ub, f(x_k))
        JuMP.@constraint(model, θ >= f(x_k) + dfdx(x_k)' * (x .- x_k))
        K += 1
        if K == iteration_limit
            break
        elseif abs(ub - lb) < tolerance
            status = "converged"
            break
        end
    end
    return Dict("x" => x_k, "lower_bound" => lb, "upper_bound" => ub, "iterations" => K, "status" => status)
end
f(x) = (x[1] - 1)^2 + (x[2] + 2)^2 + 1.0
dfdx(x) = [2 * (x[1] - 1), 2 * (x[2] + 2)]
results["kelley"] = Dict(
    "limit_20" => kelleys(f, dfdx; input_dimension = 2, lower_bound = 0.0, iteration_limit = 20),
    "limit_200" => kelleys(f, dfdx; input_dimension = 2, lower_bound = 0.0, iteration_limit = 200),
)

# ---- hydro-thermal (the page's example) with SDDP.jl: risk-averse and cyclic variants
function hydro_thermal(graph)
    return SDDP.PolicyGraph(graph; sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variable(sp, thermal_generation >= 0)
        @variable(sp, hydro_generation >= 0)
        @variable(sp, hydro_spill >= 0)
        @variable(sp, inflow)
        SDDP.parameterize(ω -> JuMP.fix(inflow, ω), sp, [0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3])
        @constraint(sp, volume.out == volume.in - hydro_generation - hydro_spill + inflow)
        @constraint(sp, demand_constraint, hydro_generation + thermal_generation == 150.0)
        fuel_cost = [50.0, 100.0, 150.0]
        @stageobjective(sp, fuel_cost[t] * thermal_generation)
    end
end
function trained_bound(graph; iters, kwargs...)
    Random.seed!(1)
    m = hydro_thermal(graph)
    SDDP.train(m; iteration_limit = iters, print_level = 0, log_file = LOGFILE, kwargs...)
    return Dict("iteration_limit" => iters, "bound" => SDDP.calculate_bound(m))
end
linear = SDDP.LinearGraph(3)
cyclic = SDDP.Graph(0)
for i in 1:3; SDDP.add_node(cyclic, i); end
SDDP.add_edge(cyclic, 0 => 1, 1.0); SDDP.add_edge(cyclic, 1 => 2, 1.0)
SDDP.add_edge(cyclic, 2 => 3, 1.0); SDDP.add_edge(cyclic, 3 => 2, 0.5)
results["hydro_thermal_expectation"] = trained_bound(linear; iters = 20)
results["hydro_thermal_entropic_1"] = trained_bound(linear; iters = 30, risk_measure = SDDP.Entropic(1.0))
results["hydro_thermal_worst_case"] = trained_bound(linear; iters = 30, risk_measure = SDDP.WorstCase())
results["infinite_cyclic"] = trained_bound(cyclic; iters = 120)
for (k, v) in results
    k == "versions" && continue
    println("=== $k: ", v)
end
open(joinpath(OUTX, "explanation.json"), "w") do io
    JSON.print(io, results, 2)
end
