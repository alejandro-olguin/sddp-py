# Oracle for six sddp.dev tutorial pages (docs/src/tutorial): warnings, arma, decision_hazard,
# production_planning, batteries, inventory. Usage: julia --project=. generate_tutorials_a.jl
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
const OUTT = joinpath(@__DIR__, "oracle")
results = Dict{String,Any}()

function det_objective(m)
    de = SDDP.deterministic_equivalent(m, HiGHS.Optimizer)
    set_silent(de)
    optimize!(de)
    return objective_value(de)
end

# ============================================================================ warnings.jl
let d = Dict{String,Any}()
    model = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 1)
        if t == 2
            @constraint(sp, x.in >= 1)
        end
        @stageobjective(sp, x.out)
    end
    d["recourse_error"] = try
        SDDP.train(model; iteration_limit = 1, print_level = 0, log_file = LOGFILE)
        "NO ERROR"
    catch err
        sprint(showerror, err)
    end
    # the .mps file SDDP.jl writes next to the script; do not leave it behind
    for f in readdir(pwd())
        if startswith(f, "subproblem_") || startswith(f, "model_infeasible_node_")
            rm(joinpath(pwd(), f); force = true)
        end
    end
    model = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = -1e10) do subproblem, t
        @variable(subproblem, x >= -1e7, SDDP.State, initial_value = 1e-5)
        @constraint(subproblem, 1e9 * x.out >= 1e-6 * x.in + 1e-8)
        @stageobjective(subproblem, 1e9 * x.out)
    end
    d["report"] = sprint(io -> SDDP.numerical_stability_report(io, model))
    d["report_by_node"] = sprint(io -> SDDP.numerical_stability_report(io, model; by_node = true))
    function build_initial_bound(lb)
        return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = lb, optimizer = HiGHS.Optimizer) do subproblem, t
            @variable(subproblem, x >= 0, SDDP.State, initial_value = 2)
            @variable(subproblem, u >= 0)
            @variable(subproblem, v >= 0)
            @constraint(subproblem, x.out == x.in - u)
            @constraint(subproblem, u + v == 1.5)
            @stageobjective(subproblem, t * v)
        end
    end
    for lb in (0.0, 10.0)
        m = build_initial_bound(lb)
        SDDP.train(m; iteration_limit = 5, run_numerical_stability_report = false, print_level = 0, log_file = LOGFILE)
        d["initial_bound_lb_$(Int(lb))"] = Dict("bound" => SDDP.calculate_bound(m), "log" => log_to_list(m))
    end
    results["warnings"] = d
    println("=== warnings: bounds ", d["initial_bound_lb_0"]["bound"], " ", d["initial_bound_lb_10"]["bound"])
end

# ============================================================================ arma.jl
const ARMA_Ω = [-10.0, 0.1, 9.6]
function build_state_space_expansion()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 200, SDDP.State, initial_value = 200)
        @variable(sp, g_t >= 0)
        @variable(sp, g_h >= 0)
        @variable(sp, s >= 0)
        @constraint(sp, g_h + g_t == 150)
        c = [50, 100, 150]
        @stageobjective(sp, c[t] * g_t)
        @variable(sp, inflow, SDDP.State, initial_value = 50.0)
        @variable(sp, ε)
        @constraint(sp, inflow.out == inflow.in + ε)
        @constraint(sp, x.out == x.in - g_h - s + inflow.out)
        SDDP.parameterize(sp, ARMA_Ω) do ω
            fix(ε, ω)
            return
        end
    end
end

function arma_simulator()
    inflow = zeros(3)
    current = 50.0
    for t in 1:3
        current += rand(ARMA_Ω)
        inflow[t] = current
    end
    return inflow
end

function build_markov_chain_model(graph)
    return SDDP.PolicyGraph(graph; sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        t, inflow = node
        @variable(sp, 0 <= x <= 200, SDDP.State, initial_value = 200)
        @variable(sp, g_t >= 0)
        @variable(sp, g_h >= 0)
        @variable(sp, s >= 0)
        @constraint(sp, g_h + g_t == 150)
        c = [50, 100, 150]
        @stageobjective(sp, c[t] * g_t)
        @constraint(sp, x.out == x.in - g_h - s + inflow)
    end
end

function build_var_model()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 200, SDDP.State, initial_value = 200)
        @variable(sp, g_t >= 0)
        @variable(sp, g_h >= 0)
        @variable(sp, s >= 0)
        @constraint(sp, g_h + g_t == 150)
        c = [50, 100, 150]
        @stageobjective(sp, c[t] * g_t)
        @variable(sp, inflow[1:2], SDDP.State, initial_value = 50.0)
        @variable(sp, ε[1:2])
        A = [0.8 0.2; 0.2 0.8]
        @constraint(sp, [i = 1:2], inflow[i].out == sum(A[i, j] * inflow[j].in for j in 1:2) + ε[i])
        @constraint(sp, x.out == x.in - g_h - s + inflow[1].out + inflow[2].out)
        Ω = [(ω₁, ω₂) for ω₁ in ARMA_Ω for ω₂ in ARMA_Ω]
        SDDP.parameterize(sp, Ω) do ω
            fix(ε[1], ω[1])
            fix(ε[2], ω[2])
            return
        end
    end
end

let d = Dict{String,Any}()
    for (name, build, iters) in (("state_space", build_state_space_expansion, 30), ("var", build_var_model, 100))
        e = Dict{String,Any}("iteration_limit" => iters, "deterministic_equivalent" => det_objective(build()))
        Random.seed!(1)
        m = build()
        SDDP.train(m; iteration_limit = iters, print_level = 0, log_file = LOGFILE)
        e["bound"] = SDDP.calculate_bound(m)
        d[name] = e
        println("=== arma/$name: det ", e["deterministic_equivalent"], " bound ", e["bound"])
    end
    # Simulator-fitted Markov chain: the fit is random, so record the shape for several seeds.
    graphs = Dict{String,Any}()
    for seed in 1:5
        Random.seed!(seed)
        graph = SDDP.MarkovianGraph(arma_simulator; budget = 8, scenarios = 30)
        nodes = [k for k in keys(graph.nodes) if k != graph.root_node]
        per_stage = [count(n -> n[1] == t, nodes) for t in 1:3]
        arcs = Dict{String,Any}()
        for (k, children) in graph.nodes
            arcs[string(k)] = sum(p for (_, p) in children; init = 0.0)
        end
        m = build_markov_chain_model(graph)
        Random.seed!(seed)
        SDDP.train(m; iteration_limit = 20, print_level = 0, log_file = LOGFILE)
        graphs["seed_$seed"] = Dict(
            "num_nodes_including_root" => length(graph.nodes),
            "num_nodes" => length(nodes),
            "per_stage" => per_stage,
            "nodes" => [[n[1], n[2]] for n in sort(nodes)],
            "outgoing_probability" => arcs,
            "bound_20" => SDDP.calculate_bound(m),
        )
    end
    d["markov_chain"] = Dict("budget" => 8, "scenarios" => 30, "graphs" => graphs)
    println("=== arma/markov: per-stage ", [graphs["seed_$s"]["per_stage"] for s in 1:5])
    results["arma"] = d
end

# ============================================================================ decision_hazard.jl
function build_hazard_decision()
    return SDDP.LinearPolicyGraph(; stages = 4, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variables(sp, begin
            0 <= x_storage <= 8, (SDDP.State, initial_value = 6)
            u_thermal >= 0
            u_hydro >= 0
            u_unmet_demand >= 0
        end)
        @constraint(sp, u_thermal + u_hydro == 9 - u_unmet_demand)
        @constraint(sp, c_balance, x_storage.out == x_storage.in - u_hydro + 0)
        SDDP.parameterize(sp, [2, 3]) do ω_inflow
            return set_normalized_rhs(c_balance, ω_inflow)
        end
        @stageobjective(sp, 500 * u_unmet_demand + 20 * u_thermal)
    end
end

function build_decision_hazard()
    return SDDP.LinearPolicyGraph(; stages = 4, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variables(sp, begin
            0 <= x_storage <= 8, (SDDP.State, initial_value = 6)
            u_thermal >= 0, (SDDP.State, initial_value = 0)
            u_hydro >= 0
            u_unmet_demand >= 0
        end)
        @constraint(sp, u_thermal.in + u_hydro == 9 - u_unmet_demand)
        @constraint(sp, c_balance, x_storage.out == x_storage.in - u_hydro + 0)
        SDDP.parameterize(sp, [2, 3]) do ω
            return set_normalized_rhs(c_balance, ω)
        end
        @stageobjective(sp, 500 * u_unmet_demand + 20 * u_thermal.in)
    end
end

function build_decision_hazard_2()
    return SDDP.LinearPolicyGraph(; stages = 5, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variables(sp, begin
            0 <= x_storage <= 8, (SDDP.State, initial_value = 6)
            u_thermal >= 0, (SDDP.State, initial_value = 0)
            u_hydro >= 0
            u_unmet_demand >= 0
        end)
        if node == 1
            @constraint(sp, x_storage.out == x_storage.in)
            @stageobjective(sp, 0)
        else
            @constraint(sp, u_thermal.in + u_hydro == 9 - u_unmet_demand)
            @constraint(sp, c_balance, x_storage.out == x_storage.in - u_hydro + 0)
            SDDP.parameterize(sp, [2, 3]) do ω
                return set_normalized_rhs(c_balance, ω)
            end
            @stageobjective(sp, 500 * u_unmet_demand + 20 * u_thermal.in)
        end
    end
end

let d = Dict{String,Any}()
    for (name, build) in (("hazard_decision", build_hazard_decision), ("decision_hazard", build_decision_hazard), ("decision_hazard_2", build_decision_hazard_2))
        e = Dict{String,Any}("deterministic_equivalent" => det_objective(build()))
        Random.seed!(1)
        m = build()
        SDDP.train(m; print_level = 0, log_file = LOGFILE)  # default stopping rule, as on the page
        e["bound"] = SDDP.calculate_bound(m)
        e["iterations"] = length(m.most_recent_training_results.log)
        e["status"] = string(m.most_recent_training_results.status)
        d[name] = e
        println("=== decision_hazard/$name: det ", e["deterministic_equivalent"], " Cost = \$", e["bound"], " (", e["iterations"], " its)")
    end
    results["decision_hazard"] = d
end

# ============================================================================ production_planning.jl
function build_production_planning()
    P, M = ["Seattle", "San-Diego"], ["New-York", "Chicago", "Topeka"]
    Ω = [
        Dict("New-York" => 300, "Chicago" => 300, "Topeka" => 300),
        Dict("New-York" => 350, "Chicago" => 320, "Topeka" => 310),
        Dict("New-York" => 320, "Chicago" => 390, "Topeka" => 350),
        Dict("New-York" => 250, "Chicago" => 220, "Topeka" => 330),
        Dict("New-York" => 290, "Chicago" => 200, "Topeka" => 290),
    ]
    c_capacity = Dict("Seattle" => 350, "San-Diego" => 600)
    c_ship = Containers.DenseAxisArray([2.6 1.7 1.8; 2.5 1.8 1.4], P, M)
    return SDDP.LinearPolicyGraph(; stages = 10, sense = :Min, optimizer = HiGHS.Optimizer, lower_bound = 0.0) do sp, t
        @variable(sp, x[p in P] >= 0, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= u_prod[p in P] <= c_capacity[p])
        @variable(sp, u_ship[p in P, m in M] >= 0)
        @variable(sp, u_unmet[m in M] >= 0)
        @variable(sp, w[m in M] == 0)
        if t > 1
            SDDP.parameterize(sp, Ω) do ω
                for m in M
                    fix(w[m], ω[m])
                end
                return
            end
        end
        @constraint(sp, [p in P], sum(u_ship[p, :]) <= x[p].in)
        @constraint(sp, [p in P], x[p].out == x[p].in + u_prod[p] - sum(u_ship[p, :]))
        @constraint(sp, [m in M], sum(u_ship[:, m]) == w[m] - u_unmet[m])
        @stageobjective(
            sp,
            sum(1.0 * u_prod[p] + 0.01 * x[p].out for p in P) +
            sum(10 * u_unmet[m] for m in M) +
            sum(c_ship[p, m] * u_ship[p, m] for p in P, m in M),
        )
    end
end

let d = Dict{String,Any}("iteration_limit" => 300)
    Random.seed!(1)
    m = build_production_planning()
    SDDP.train(m; iteration_limit = 300, risk_measure = SDDP.Expectation(), print_level = 0, log_file = LOGFILE)
    d["bound"] = SDDP.calculate_bound(m)
    d["log"] = log_to_list(m)
    d["simulation"] = simulate_stats(m, 50; seed = 42)
    # first-stage decision is deterministic: record it
    Random.seed!(3)
    sims = SDDP.simulate(m, 1, [:x, :u_prod])
    d["first_stage"] = Dict(
        "u_prod" => Dict(p => sims[1][1][:u_prod][p] for p in ["Seattle", "San-Diego"]),
        "x_out" => Dict(p => sims[1][1][:x][p].out for p in ["Seattle", "San-Diego"]),
    )
    Random.seed!(1)
    m = build_production_planning()
    SDDP.train(m; iteration_limit = 1000, risk_measure = SDDP.Expectation(), print_level = 0, log_file = LOGFILE)
    d["bound_1000"] = SDDP.calculate_bound(m)
    results["production_planning"] = d
    println("=== production_planning: bound(300) ", d["bound"], " bound(1000) ", d["bound_1000"], " sim mean ", d["simulation"]["mean"])
end

# ============================================================================ batteries.jl
function build_batteries()
    return SDDP.LinearPolicyGraph(; stages = 24, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x_soc <= 30, SDDP.State, initial_value = 4)
        @variable(sp, 0 <= x_thermal <= 70, SDDP.State, initial_value = 35)
        @variable(sp, 0 <= u_charge <= 15)
        @variable(sp, 0 <= u_discharge <= 15)
        @variable(sp, u_slack >= 0)
        @variable(sp, u_surplus >= 0)
        @variable(sp, w_load)
        Ω = [-4.0, -2.0, 0.0, 2.0, 4.0]
        d = vcat([40, 41, 42, 43, 35, 40, 40, 25, 10, 8, 6, 5], [5, 6, 8, 10, 20, 30, 55, 72, 75, 70, 64, 60])
        SDDP.parameterize(ω -> fix(w_load, d[t] + ω), sp, Ω)
        @stageobjective(sp, 70 * x_thermal.out + 500 * u_slack)
        @constraint(sp, x_soc.out == x_soc.in + 0.8 * u_charge - u_discharge)
        @constraint(sp, x_thermal.out - x_thermal.in <= 10)
        @constraint(sp, λ, x_thermal.out + u_discharge - u_charge + u_slack == w_load + u_surplus)
        return
    end
end

let d = Dict{String,Any}("iteration_limit" => 500)
    Random.seed!(1)
    m = build_batteries()
    SDDP.train(m; iteration_limit = 500, print_level = 0, log_file = LOGFILE)
    d["bound"] = SDDP.calculate_bound(m)
    Random.seed!(42)
    results_sim = SDDP.simulate(m, 100, [:x_soc, :x_thermal, :u_slack, :w_load]; custom_recorders = Dict{Symbol,Function}(:λ => sp -> dual(sp[:λ])))
    λ = [[s[:λ] for s in sim] for sim in results_sim]
    d["lambda_mean_by_hour"] = [mean(l[h] for l in λ) for h in 1:24]
    d["lambda_frac_70_hours_1_6"] = [mean(isapprox(l[h], 70.0; atol = 1e-6) for l in λ) for h in 1:6]
    d["lambda_min"] = minimum(minimum(l) for l in λ)
    d["lambda_max"] = maximum(maximum(l) for l in λ)
    d["x_soc_out_hour_24_max"] = maximum(sim[24][:x_soc].out for sim in results_sim)
    objs = [sum(s[:stage_objective] for s in sim) for sim in results_sim]
    d["simulation"] = Dict("replications" => 100, "seed" => 42, "mean" => mean(objs), "std" => std(objs))
    # a second, longer run to show how converged 500 iterations is
    Random.seed!(1)
    m = build_batteries()
    SDDP.train(m; iteration_limit = 1500, print_level = 0, log_file = LOGFILE)
    d["bound_1500"] = SDDP.calculate_bound(m)
    results["batteries"] = d
    println("=== batteries: bound(500) ", d["bound"], " bound(1500) ", d["bound_1500"], " λ frac 70 ", d["lambda_frac_70_hours_1_6"])
end

# ============================================================================ inventory.jl
const INV_x_0 = 10
const INV_c = 35
const INV_h = 1
const INV_p = 15
const INV_Ω = range(0, 800; length = 20)

function build_inventory_finite(T; lower_bound = 0.0)
    x_0, c, h, p, Ω = INV_x_0, INV_c, INV_h, INV_p, INV_Ω
    return SDDP.LinearPolicyGraph(; stages = T + 1, sense = :Min, lower_bound = lower_bound, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x_inventory >= 0, SDDP.State, initial_value = x_0)
        @variable(sp, x_demand >= 0, SDDP.State, initial_value = 0)
        @variable(sp, u_buy >= 0, SDDP.State, initial_value = 0)
        @variable(sp, u_sell >= 0)
        @variable(sp, w_demand == 0)
        @constraint(sp, x_inventory.out == x_inventory.in + u_buy.in - u_sell)
        @constraint(sp, x_demand.out == x_demand.in + w_demand - u_sell)
        if t == 1
            fix(u_sell, 0; force = true)
            @stageobjective(sp, c * u_buy.out)
        elseif t == T + 1
            fix(u_buy.out, 0; force = true)
            @stageobjective(sp, -c * x_inventory.out + c * x_demand.out)
            SDDP.parameterize(ω -> fix(w_demand, ω), sp, Ω)
        else
            @stageobjective(sp, c * u_buy.out + h * x_inventory.out + p * x_demand.out)
            SDDP.parameterize(ω -> fix(w_demand, ω), sp, Ω)
        end
        return
    end
end

function build_inventory_infinite(α = 0.95)
    x_0, c, h, p, Ω = INV_x_0, INV_c, INV_h, INV_p, INV_Ω
    graph = SDDP.LinearGraph(2)
    SDDP.add_edge(graph, 2 => 2, α)
    return SDDP.PolicyGraph(graph; sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x_inventory >= 0, SDDP.State, initial_value = x_0)
        @variable(sp, x_demand >= 0, SDDP.State, initial_value = 0)
        @variable(sp, u_buy >= 0, SDDP.State, initial_value = 0)
        @variable(sp, u_sell >= 0)
        @variable(sp, w_demand == 0)
        @constraint(sp, x_inventory.out == x_inventory.in + u_buy.in - u_sell)
        @constraint(sp, x_demand.out == x_demand.in + w_demand - u_sell)
        if t == 1
            fix(u_sell, 0; force = true)
            @stageobjective(sp, c * u_buy.out)
        else
            @stageobjective(sp, c * u_buy.out + h * x_inventory.out + p * x_demand.out)
            SDDP.parameterize(ω -> fix(w_demand, ω), sp, Ω)
        end
        return
    end
end

let d = Dict{String,Any}()
    # Finite horizon, T = 10, default stopping rule (as on the page)
    Random.seed!(1)
    m = build_inventory_finite(10)
    SDDP.train(m; print_level = 0, log_file = LOGFILE)
    f = Dict{String,Any}("T" => 10, "bound" => SDDP.calculate_bound(m), "iterations" => length(m.most_recent_training_results.log), "status" => string(m.most_recent_training_results.status))
    Random.seed!(2)
    sims = SDDP.simulate(m, 200, [:x_inventory, :u_buy])
    objs = [sum(t[:stage_objective] for t in s) for s in sims]
    μ, ci = SDDP.confidence_interval(objs, 1.96)
    f["ci_mean"] = μ
    f["ci_half"] = ci
    f["order_up_to_median_by_stage"] = [median(s[t][:x_inventory].out + s[t][:u_buy].out for s in sims) for t in 1:11]
    d["finite"] = f
    # Finite horizon, T = 3: small enough for the deterministic equivalent. The page's
    # lower_bound = 0 cuts off the (negative) terminal value function, so the converged bound
    # sits ABOVE the deterministic equivalent; with a valid bound they agree.
    g = Dict{String,Any}("T" => 3, "iteration_limit" => 100, "deterministic_equivalent" => det_objective(build_inventory_finite(3)))
    for (key, lb) in (("bound_lb_0", 0.0), ("bound_lb_neg1e6", -1e6))
        Random.seed!(1)
        m = build_inventory_finite(3; lower_bound = lb)
        SDDP.train(m; iteration_limit = 100, print_level = 0, log_file = LOGFILE)
        g[key] = SDDP.calculate_bound(m)
    end
    d["finite_T3"] = g
    # Infinite horizon: 400 iterations as on the page
    Random.seed!(1)
    m = build_inventory_infinite()
    SDDP.train(m; iteration_limit = 400, print_level = 0, log_file = LOGFILE)
    inf = Dict{String,Any}("iteration_limit" => 400, "bound" => SDDP.calculate_bound(m))
    Random.seed!(2)
    sims = SDDP.simulate(m, 200, [:x_inventory, :u_buy]; sampling_scheme = SDDP.InSampleMonteCarlo(; max_depth = 50, terminate_on_dummy_leaf = false))
    inf["simulation_lengths"] = sort(unique(length(s) for s in sims))
    levels = [s[t][:x_inventory].out + s[t][:u_buy].out for s in sims for t in 6:length(s)]
    inf["order_up_to_later_stages"] = Dict("median" => median(levels), "mean" => mean(levels), "min" => minimum(levels), "max" => maximum(levels))
    Random.seed!(1)
    m = build_inventory_infinite()
    SDDP.train(m; iteration_limit = 1000, print_level = 0, log_file = LOGFILE)
    inf["bound_1000"] = SDDP.calculate_bound(m)
    d["infinite"] = inf
    results["inventory"] = d
    println("=== inventory: finite bound ", f["bound"], " (", f["iterations"], " its); T3 det ", g["deterministic_equivalent"], " lb0 ", g["bound_lb_0"], " lbneg ", g["bound_lb_neg1e6"], "; infinite bound(400) ", inf["bound"], " bound(1000) ", inf["bound_1000"], " median level ", inf["order_up_to_later_stages"]["median"])
end

results["versions"] = versions()
open(joinpath(OUTT, "tutorials_a.json"), "w") do io
    JSON.print(io, results, 2)
end
println("ALL DONE")
