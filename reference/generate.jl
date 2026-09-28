# Oracle generator: runs SDDP.jl on the reference problems and writes JSON ground truth.
# Usage: julia --project=. generate.jl [tier1|tier2|tier3|all] [problem_name]
# NEVER edit the produced JSON files by hand.
using SDDP, HiGHS, JSON, Random, Statistics, Pkg

const OUT = joinpath(@__DIR__, "oracle")
mkpath(OUT)
const LOGFILE = tempname()

function versions()
    deps = Pkg.dependencies()
    v(name) = string(first(d for (_, d) in deps if d.name == name).version)
    return Dict("julia" => string(VERSION), "SDDP" => v("SDDP"), "HiGHS" => v("HiGHS"), "JuMP" => v("JuMP"))
end

function log_to_list(model)
    return [Dict("iteration" => l.iteration, "bound" => l.bound, "simulation_value" => l.simulation_value)
            for l in model.most_recent_training_results.log]
end

function simulate_stats(model, n; seed, kwargs...)
    Random.seed!(seed)
    sims = SDDP.simulate(model, n; kwargs...)
    objs = [sum(s[:stage_objective] for s in sim) for sim in sims]
    return Dict("replications" => n, "seed" => seed, "mean" => mean(objs), "std" => std(objs),
                "min" => minimum(objs), "max" => maximum(objs))
end

function det_equiv(builder; time_limit = 120.0)
    model = builder()
    det = try
        SDDP.deterministic_equivalent(model, HiGHS.Optimizer; time_limit = time_limit)
    catch e
        return nothing
    end
    set_silent(det)
    optimize!(det)
    return objective_value(det)
end

function cuts_json(model)
    f = tempname() * ".json"
    SDDP.write_cuts_to_file(model, f)
    return JSON.parsefile(f)
end

function train_bound(builder; seed, iteration_limit, kwargs...)
    Random.seed!(seed)
    model = builder()
    SDDP.train(model; iteration_limit = iteration_limit, print_level = 0, log_file = LOGFILE, kwargs...)
    return model, SDDP.calculate_bound(model), log_to_list(model)
end

# Deterministic run: fixed Historical scenarios so trajectories are identical in Python.
function deterministic_run(builder, scenarios; iterations, kwargs...)
    model = builder()
    SDDP.train(model; iteration_limit = iterations, print_level = 0, log_file = LOGFILE,
               sampling_scheme = SDDP.Historical(scenarios), run_numerical_stability_report = false, kwargs...)
    return Dict("scenarios" => scenarios, "iterations" => iterations, "log" => log_to_list(model),
                "bound" => SDDP.calculate_bound(model), "cuts" => cuts_json(model))
end

# ---------------------------------------------------------------------------
# Tier 1 problems
# ---------------------------------------------------------------------------
function build_hydro_thermal()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
        end)
        @variable(sp, inflow)
        SDDP.parameterize(sp, [0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3]) do ω
            fix(inflow, ω)
        end
        @constraints(sp, begin
            volume.out == volume.in - hydro_generation - hydro_spill + inflow
            demand_constraint, hydro_generation + thermal_generation == 150
        end)
        fuel_cost = [50, 100, 150]
        @stageobjective(sp, fuel_cost[t] * thermal_generation)
    end
end

function build_fast_quickstart()
    return SDDP.PolicyGraph(SDDP.LinearGraph(2); lower_bound = -5, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        if t == 1
            @stageobjective(sp, x.out)
        else
            @variable(sp, s >= 0)
            @constraint(sp, s <= x.in)
            SDDP.parameterize(sp, [2, 3]) do ω
                set_upper_bound(s, ω)
            end
            @stageobjective(sp, -2s)
        end
    end
end

function build_fast_hydro_thermal()
    return SDDP.LinearPolicyGraph(; stages = 2, upper_bound = 0.0, sense = :Max, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 8, SDDP.State, initial_value = 0.0)
        @variables(sp, begin
            y >= 0
            p >= 0
            ξ
        end)
        @constraints(sp, begin
            p + y >= 6
            x.out <= x.in - y + ξ
        end)
        RAINFALL = (t == 1 ? [6] : [2, 10])
        SDDP.parameterize(sp, RAINFALL) do ω
            fix(ξ, ω)
        end
        @stageobjective(sp, -5 * p)
    end
end

function build_fast_production_management()
    DEMAND = [2, 10]; H = 3; N = 2; C = [0.2, 0.7]; S = 2 .+ [0.33, 0.54]
    return SDDP.LinearPolicyGraph(; stages = H, lower_bound = -50.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x[1:N] >= 0, SDDP.State, initial_value = 0.0)
        @variables(sp, begin
            s[i = 1:N] >= 0
            d
        end)
        @constraints(sp, begin
            [i = 1:N], s[i] <= x[i].in
            sum(s) <= d
        end)
        SDDP.parameterize(sp, t == 1 ? [0] : DEMAND) do ω
            fix(d, ω)
        end
        @stageobjective(sp, sum(C[i] * x[i].out for i in 1:N) - S's)
    end
end

function build_stock_example()
    return SDDP.PolicyGraph(SDDP.LinearGraph(5); lower_bound = -2, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= state <= 1, SDDP.State, initial_value = 0.5)
        @variable(sp, 0 <= control <= 0.5)
        @variable(sp, ξ)
        @constraint(sp, state.out == state.in - control + ξ)
        SDDP.parameterize(sp, 0.0:(1 / 30):0.3) do ω
            fix(ξ, ω)
        end
        @stageobjective(sp, (sin(3 * stage) - 1) * control)
    end
end

function build_farmers()
    MAX_AREA = 500.0
    CROPS = [:wheat, :corn, :sugar_beet]
    PLANTING_COST = Dict(:wheat => 150.0, :corn => 230.0, :sugar_beet => 260.0)
    MIN_QUANTITIES = Dict(:wheat => 200.0, :corn => 240.0, :sugar_beet => 0.0)
    QUOTA_MAX = Dict(:wheat => Inf, :corn => Inf, :sugar_beet => 6_000.0)
    SELL_IN_QUOTA = Dict(:wheat => 170.0, :corn => 150.0, :sugar_beet => 36.0)
    SELL_NO_QUOTA = Dict(:wheat => 0.0, :corn => 0.0, :sugar_beet => 10.0)
    BUY_PRICE = Dict(:wheat => 238.0, :corn => 210.0, :sugar_beet => 1_000.0)
    MEAN_YIELD = Dict(:wheat => 2.5, :corn => 3.0, :sugar_beet => 20.0)
    YIELD_MULTIPLIER = Dict(:good => 1.2, :fair => 1.0, :bad => 0.8)
    return SDDP.LinearPolicyGraph(; stages = 2, sense = :Max, upper_bound = 500_000.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, area[c = CROPS] >= 0, SDDP.State, initial_value = 0)
        if stage == 1
            @constraint(sp, sum(area[c].out for c in CROPS) <= MAX_AREA)
            @stageobjective(sp, -sum(PLANTING_COST[c] * area[c].out for c in CROPS))
        else
            @variables(sp, begin
                0 <= yield[c = CROPS]
                0 <= buy[c = CROPS]
                0 <= sell_in_quota[c = CROPS] <= QUOTA_MAX[c]
                0 <= sell_no_quota[c = CROPS]
            end)
            @constraint(sp, [c = CROPS], yield[c] + buy[c] - sell_in_quota[c] - sell_no_quota[c] >= MIN_QUANTITIES[c])
            @constraint(sp, uncertainty[c = CROPS], 1.0 * area[c].in == yield[c])
            SDDP.parameterize(sp, [:good, :fair, :bad]) do ω
                for c in CROPS
                    set_normalized_coefficient(uncertainty[c], area[c].in, MEAN_YIELD[c] * YIELD_MULTIPLIER[ω])
                end
            end
            @stageobjective(sp, sum(SELL_IN_QUOTA[c] * sell_in_quota[c] + SELL_NO_QUOTA[c] * sell_no_quota[c] - BUY_PRICE[c] * buy[c] for c in CROPS))
        end
    end
end

# ---------------------------------------------------------------------------
# Tier 2 problems
# ---------------------------------------------------------------------------
const Ω_MARKOV = [(inflow = 0.0, fuel_multiplier = 1.5), (inflow = 50.0, fuel_multiplier = 1.0), (inflow = 100.0, fuel_multiplier = 0.75)]

function build_markov_uncertainty()
    return SDDP.MarkovianPolicyGraph(; transition_matrices = Array{Float64,2}[[1.0]', [0.75 0.25], [0.75 0.25; 0.25 0.75]],
                                       sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        t, markov_state = node
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
            inflow
        end)
        @constraints(sp, begin
            volume.out == volume.in + inflow - hydro_generation - hydro_spill
            thermal_generation + hydro_generation == 150.0
        end)
        probability = markov_state == 1 ? [1 / 6, 1 / 3, 1 / 2] : [1 / 2, 1 / 3, 1 / 6]
        fuel_cost = [50.0, 100.0, 150.0]
        SDDP.parameterize(sp, Ω_MARKOV, probability) do ω
            fix(inflow, ω.inflow)
            @stageobjective(sp, ω.fuel_multiplier * fuel_cost[t] * thermal_generation)
        end
    end
end

function build_objective_uncertainty()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
            inflow
        end)
        @constraints(sp, begin
            volume.out == volume.in + inflow - hydro_generation - hydro_spill
            thermal_generation + hydro_generation == 150.0
        end)
        fuel_cost = [50.0, 100.0, 150.0]
        SDDP.parameterize(sp, Ω_MARKOV, [1 / 3, 1 / 3, 1 / 3]) do ω
            fix(inflow, ω.inflow)
            @stageobjective(sp, ω.fuel_multiplier * fuel_cost[t] * thermal_generation)
        end
    end
end

function build_infinite_trivial()
    graph = SDDP.Graph(:root_node, [:week], [(:root_node => :week, 1.0), (:week => :week, 0.9)])
    return SDDP.PolicyGraph(graph; lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variable(sp, state, SDDP.State, initial_value = 0)
        @constraint(sp, state.in == state.out)
        @stageobjective(sp, 2.0)
    end
end

function build_no_strong_duality()
    return SDDP.PolicyGraph(SDDP.Graph(:root, [:node], [(:root => :node, 1.0), (:node => :node, 0.5)]);
                            optimizer = HiGHS.Optimizer, lower_bound = 0.0) do sp, t
        @variable(sp, x, SDDP.State, initial_value = 1.0)
        @stageobjective(sp, x.out)
        @constraint(sp, x.in == x.out)
    end
end

function build_infinite_hydro_thermal()
    Ω = [(inflow = 0.0, demand = 7.5), (inflow = 5.0, demand = 5), (inflow = 10.0, demand = 2.5)]
    graph = SDDP.Graph(:root_node, [:week], [(:root_node => :week, 1.0), (:week => :week, 0.9)])
    return SDDP.PolicyGraph(graph; lower_bound = 0, optimizer = HiGHS.Optimizer) do sp, node
        @variable(sp, 5.0 <= reservoir <= 15.0, SDDP.State, initial_value = 10.0)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            spill >= 0
            inflow
            demand
        end)
        @constraints(sp, begin
            reservoir.out == reservoir.in - hydro_generation - spill + inflow
            hydro_generation + thermal_generation == demand
        end)
        @stageobjective(sp, 10 * spill + thermal_generation)
        SDDP.parameterize(sp, Ω) do ω
            fix(inflow, ω.inflow)
            fix(demand, ω.demand)
        end
    end
end

function build_asset_management_stagewise()
    w_s = [1.25, 1.06]; w_b = [1.14, 1.12]; Phi = [-1, 5]; Psi = [0.02, 0.0]
    return SDDP.MarkovianPolicyGraph(; sense = :Max,
        transition_matrices = Array{Float64,2}[[1.0]', [0.5 0.5], [0.5 0.5; 0.5 0.5], [0.5 0.5; 0.5 0.5]],
        upper_bound = 1000.0, optimizer = HiGHS.Optimizer) do sp, node
        t, i = node
        @variable(sp, xs >= 0, SDDP.State, initial_value = 0)
        @variable(sp, xb >= 0, SDDP.State, initial_value = 0)
        if t == 1
            @constraint(sp, xs.out + xb.out == 55 + xs.in + xb.in)
            @stageobjective(sp, 0)
        elseif t == 2 || t == 3
            @variable(sp, phi)
            @constraint(sp, w_s[i] * xs.in + w_b[i] * xb.in + phi == xs.out + xb.out)
            SDDP.parameterize(sp, [1, 2], [0.6, 0.4]) do ω
                fix(phi, Phi[ω])
                @stageobjective(sp, Psi[ω] * xs.out)
            end
        else
            @variable(sp, u >= 0)
            @variable(sp, v >= 0)
            @constraint(sp, w_s[i] * xs.in + w_b[i] * xb.in + u - v == 80)
            @stageobjective(sp, -4u + v)
        end
    end
end
asset_risk(node) = node[1] != 3 ? SDDP.Expectation() : SDDP.EAVaR(; lambda = 0.5, beta = 0.5)

function build_objective_states()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
            inflow
        end)
        @constraints(sp, begin
            volume.out == volume.in + inflow - hydro_generation - hydro_spill
            demand_constraint, thermal_generation + hydro_generation == 150.0
        end)
        SDDP.add_objective_state(sp; initial_value = 50.0, lipschitz = 10_000.0, lower_bound = 50.0, upper_bound = 150.0) do fuel_cost, ω
            return ω.fuel * fuel_cost
        end
        Ω = [(fuel = f, inflow = w) for f in [0.75, 0.9, 1.1, 1.25] for w in [0.0, 50.0, 100.0]]
        SDDP.parameterize(sp, Ω) do ω
            fuel_cost = SDDP.objective_state(sp)
            @stageobjective(sp, fuel_cost * thermal_generation)
            fix(inflow, ω.inflow)
        end
    end
end

function build_belief()
    demand_values = [1.0, 2.0]
    demand_prob = Dict(:Ah => [0.2, 0.8], :Bh => [0.8, 0.2])
    graph = SDDP.Graph(:root_node, [:Ad, :Ah, :Bd, :Bh],
        [(:root_node => :Ad, 0.5), (:root_node => :Bd, 0.5), (:Ad => :Ah, 1.0), (:Ah => :Ad, 0.8),
         (:Ah => :Bd, 0.1), (:Bd => :Bh, 1.0), (:Bh => :Bd, 0.8), (:Bh => :Ad, 0.1)])
    SDDP.add_ambiguity_set(graph, [:Ad, :Bd], 1e2)
    SDDP.add_ambiguity_set(graph, [:Ah, :Bh], 1e2)
    return SDDP.PolicyGraph(graph; lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        @variables(sp, begin
            0 <= inventory <= 2, (SDDP.State, initial_value = 0.0)
            buy >= 0
            demand
        end)
        @constraint(sp, demand == inventory.in - inventory.out + buy)
        if node == :Ad || node == :Bd
            fix(demand, 0)
            @stageobjective(sp, buy)
        else
            SDDP.parameterize(sp, demand_values, demand_prob[node]) do ω
                fix(demand, ω)
            end
            @stageobjective(sp, 2 * buy + inventory.out)
        end
    end
end

# ---------------------------------------------------------------------------
# Tier 3 problems
# ---------------------------------------------------------------------------
function build_air_conditioning()
    return SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= stored_production <= 100, Int, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= production <= 200, Int)
        @variable(sp, overtime >= 0, Int)
        @variable(sp, demand)
        DEMAND = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        SDDP.parameterize(ω -> fix(demand, ω), sp, DEMAND[stage])
        @constraint(sp, stored_production.out == stored_production.in + production + overtime - demand)
        @stageobjective(sp, 100 * production + 300 * overtime + 50 * stored_production.out)
    end
end

function build_stochastic_all_blacks()
    T = 3; N = 2; R = [3 3 6; 3 3 6]
    offers = [[[1, 1], [0, 0], [1, 1]], [[1, 0], [0, 0], [0, 0]], [[0, 1], [1, 0], [1, 1]]]
    return SDDP.LinearPolicyGraph(; stages = T, sense = :Max, upper_bound = 100.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= x[1:N] <= 1, SDDP.State, Bin, initial_value = 1)
        @variable(sp, accept_offer[1:N], Bin)
        @variable(sp, offers_made[1:N])
        @constraint(sp, balance[i in 1:N], x[i].in - x[i].out == accept_offer[i])
        @stageobjective(sp, sum(R[i, stage] * accept_offer[i] for i in 1:N))
        SDDP.parameterize(sp, offers[stage]) do o
            fix.(offers_made, o)
        end
        @constraint(sp, accept_offer .<= offers_made)
    end
end

function build_sldp_example_one()
    return SDDP.LinearPolicyGraph(; stages = 8, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x, SDDP.State, initial_value = 2.0)
        @variables(sp, begin
            x⁺ >= 0
            x⁻ >= 0
            0 <= u <= 1, Bin
            ω
        end)
        @stageobjective(sp, 0.9^(t - 1) * (x⁺ + x⁻))
        @constraints(sp, begin
            x.out == x.in + 2 * u - 1 + ω
            x⁺ >= x.out
            x⁻ >= -x.out
        end)
        points = [-0.3089653673606697, -0.2718277412744214, -0.09611178608243474, 0.24645863921577763, 0.5204224537256875]
        SDDP.parameterize(φ -> fix(ω, φ), sp, [points; -points])
    end
end

# ---------------------------------------------------------------------------
# Problem registry: name => (tier, generator function returning Dict)
# ---------------------------------------------------------------------------
wnorm(x::SDDP.Noise, y::SDDP.Noise) = abs(x.term - y.term)

const PROBLEMS = Dict{String,Tuple{Int,Function}}(
    "hydro_thermal" => (1, () -> begin
        d = Dict{String,Any}("stages" => 3)
        d["deterministic_equivalent"] = det_equiv(build_hydro_thermal)
        m, b, l = train_bound(build_hydro_thermal; seed = 1234, iteration_limit = 10)
        d["train_10"] = Dict("seed" => 1234, "iteration_limit" => 10, "bound" => b, "log" => l)
        d["simulation_10"] = simulate_stats(m, 500; seed = 42)
        m, b, l = train_bound(build_hydro_thermal; seed = 1234, iteration_limit = 40)
        d["train_40"] = Dict("seed" => 1234, "iteration_limit" => 40, "bound" => b, "log" => l)
        d["simulation_40"] = simulate_stats(m, 1000; seed = 42)
        scen = [[(1, 0.0), (2, 50.0), (3, 100.0)], [(1, 100.0), (2, 0.0), (3, 50.0)], [(1, 50.0), (2, 50.0), (3, 0.0)],
                [(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 100.0), (2, 100.0), (3, 100.0)]]
        d["deterministic_run"] = deterministic_run(build_hydro_thermal, scen; iterations = 5)
        d["deterministic_run_20"] = deterministic_run(build_hydro_thermal, scen; iterations = 20)
        # Historical simulation of a single fixed scenario: all recorded values are deterministic.
        Random.seed!(1234); m = build_hydro_thermal()
        SDDP.train(m; iteration_limit = 40, print_level = 0, log_file = LOGFILE)
        sim = SDDP.simulate(m, 1, [:volume, :thermal_generation, :hydro_generation, :hydro_spill];
                            sampling_scheme = SDDP.Historical([(1, 50.0), (2, 0.0), (3, 100.0)]))[1]
        d["historical_simulation"] = Dict("scenario" => [(1, 50.0), (2, 0.0), (3, 100.0)],
            "stages" => [Dict("node_index" => s[:node_index], "noise_term" => s[:noise_term], "stage_objective" => s[:stage_objective],
                              "bellman_term" => s[:bellman_term], "volume_in" => s[:volume].in, "volume_out" => s[:volume].out,
                              "thermal_generation" => s[:thermal_generation], "hydro_generation" => s[:hydro_generation],
                              "hydro_spill" => s[:hydro_spill]) for s in sim])
        # Risk-measure variants (Tier 2) on the same model: converged bounds after 30 iterations
        d["risk_measures"] = Dict{String,Any}()
        for (name, rm) in [("WorstCase", SDDP.WorstCase()), ("AVaR_0.5", SDDP.AVaR(0.5)), ("EAVaR_0.5_0.25", SDDP.EAVaR(; lambda = 0.5, beta = 0.25)),
                           ("Entropic_0.1", SDDP.Entropic(0.1)), ("ModifiedChiSquared_0.5", SDDP.ModifiedChiSquared(0.5)),
                           ("Wasserstein_10", SDDP.Wasserstein(wnorm, HiGHS.Optimizer; alpha = 10.0))]
            _, b, l = train_bound(build_hydro_thermal; seed = 1234, iteration_limit = 30, risk_measure = rm)
            d["risk_measures"][name] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        end
        # Multi-cut variant
        _, b, l = train_bound(build_hydro_thermal; seed = 1234, iteration_limit = 30, cut_type = SDDP.MULTI_CUT)
        d["multi_cut_30"] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        d["deterministic_run_multi_cut"] = deterministic_run(build_hydro_thermal, scen; iterations = 5, cut_type = SDDP.MULTI_CUT)
        d["deterministic_run_avar"] = deterministic_run(build_hydro_thermal, scen; iterations = 5, risk_measure = SDDP.AVaR(0.5))
        d
    end),
    "fast_quickstart" => (1, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_fast_quickstart)
        _, b, l = train_bound(build_fast_quickstart; seed = 1234, iteration_limit = 20)
        d["train_20"] = Dict("seed" => 1234, "iteration_limit" => 20, "bound" => b, "log" => l)
        scen = [[(1, nothing), (2, 2)], [(1, nothing), (2, 3)]]
        d["deterministic_run"] = deterministic_run(build_fast_quickstart, scen; iterations = 4)
        d
    end),
    "fast_hydro_thermal" => (1, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_fast_hydro_thermal)
        m, b, l = train_bound(build_fast_hydro_thermal; seed = 1234, iteration_limit = 20)
        d["train_20"] = Dict("seed" => 1234, "iteration_limit" => 20, "bound" => b, "log" => l)
        d["simulation_20"] = simulate_stats(m, 500; seed = 42)
        scen = [[(1, 6), (2, 2)], [(1, 6), (2, 10)]]
        d["deterministic_run"] = deterministic_run(build_fast_hydro_thermal, scen; iterations = 4)
        d
    end),
    "fast_production_management" => (1, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_fast_production_management)
        m, b, l = train_bound(build_fast_production_management; seed = 1234, iteration_limit = 50)
        d["train_50"] = Dict("seed" => 1234, "iteration_limit" => 50, "bound" => b, "log" => l)
        d["simulation_50"] = simulate_stats(m, 500; seed = 42)
        _, b, l = train_bound(build_fast_production_management; seed = 1234, iteration_limit = 50, cut_type = SDDP.MULTI_CUT)
        d["train_50_multi_cut"] = Dict("seed" => 1234, "iteration_limit" => 50, "bound" => b, "log" => l)
        d
    end),
    "stock_example" => (1, () -> begin
        d = Dict{String,Any}()
        m, b, l = train_bound(build_stock_example; seed = 1234, iteration_limit = 60)
        d["train_60"] = Dict("seed" => 1234, "iteration_limit" => 60, "bound" => b, "log" => l)
        d["simulation_60"] = simulate_stats(m, 1000; seed = 42)
        d
    end),
    "farmers" => (1, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_farmers)
        m, b, l = train_bound(build_farmers; seed = 1234, iteration_limit = 40)
        d["train_40"] = Dict("seed" => 1234, "iteration_limit" => 40, "bound" => b, "log" => l)
        d["simulation_40"] = simulate_stats(m, 500; seed = 42)
        d
    end),
    "markov_uncertainty" => (2, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_markov_uncertainty)
        m, b, l = train_bound(build_markov_uncertainty; seed = 1234, iteration_limit = 40)
        d["train_40"] = Dict("seed" => 1234, "iteration_limit" => 40, "bound" => b, "log" => l)
        d["simulation_40"] = simulate_stats(m, 500; seed = 42)
        scen = [[((1, 1), Ω_MARKOV[1]), ((2, 2), Ω_MARKOV[3]), ((3, 1), Ω_MARKOV[2])],
                [((1, 1), Ω_MARKOV[3]), ((2, 1), Ω_MARKOV[1]), ((3, 2), Ω_MARKOV[1])]]
        r = deterministic_run(build_markov_uncertainty, scen; iterations = 4)
        r["scenarios"] = [[(collect(n), Dict(pairs(ω))) for (n, ω) in s] for s in scen]
        d["deterministic_run"] = r
        d
    end),
    "objective_uncertainty" => (2, () -> begin
        d = Dict{String,Any}()
        d["deterministic_equivalent"] = det_equiv(build_objective_uncertainty)
        m, b, l = train_bound(build_objective_uncertainty; seed = 1234, iteration_limit = 40)
        d["train_40"] = Dict("seed" => 1234, "iteration_limit" => 40, "bound" => b, "log" => l)
        d["simulation_40"] = simulate_stats(m, 500; seed = 42)
        d
    end),
    "infinite_trivial" => (2, () -> begin
        d = Dict{String,Any}()
        _, b, l = train_bound(build_infinite_trivial; seed = 1234, iteration_limit = 30)
        d["train_30"] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        d
    end),
    "no_strong_duality" => (2, () -> begin
        d = Dict{String,Any}()
        _, b, l = train_bound(build_no_strong_duality; seed = 1234, iteration_limit = 30)
        d["train_30"] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        d
    end),
    "infinite_hydro_thermal" => (2, () -> begin
        d = Dict{String,Any}()
        for (ct, name) in [(SDDP.SINGLE_CUT, "single"), (SDDP.MULTI_CUT, "multi")]
            m, b, l = train_bound(build_infinite_hydro_thermal; seed = 1234, iteration_limit = 300, cut_type = ct,
                                  sampling_scheme = SDDP.InSampleMonteCarlo(; terminate_on_cycle = true), cycle_discretization_delta = 0.1)
            d["train_300_" * name] = Dict("seed" => 1234, "iteration_limit" => 300, "bound" => b, "log" => l)
            d["simulation_300_" * name] = simulate_stats(m, 500; seed = 42)
        end
        d
    end),
    "asset_management_stagewise" => (2, () -> begin
        d = Dict{String,Any}()
        for (ct, name) in [(SDDP.SINGLE_CUT, "single"), (SDDP.MULTI_CUT, "multi")]
            m, b, l = train_bound(build_asset_management_stagewise; seed = 1234, iteration_limit = 100, cut_type = ct, risk_measure = asset_risk)
            d["train_100_" * name] = Dict("seed" => 1234, "iteration_limit" => 100, "bound" => b, "log" => l)
        end
        d
    end),
    "objective_states" => (2, () -> begin
        d = Dict{String,Any}()
        m, b, l = train_bound(build_objective_states; seed = 1234, iteration_limit = 60, run_numerical_stability_report = false)
        d["train_60"] = Dict("seed" => 1234, "iteration_limit" => 60, "bound" => b, "log" => l)
        d["simulation_60"] = simulate_stats(m, 500; seed = 42)
        d
    end),
    "belief" => (2, () -> begin
        d = Dict{String,Any}()
        m, b, l = train_bound(build_belief; seed = 123, iteration_limit = 100, cut_type = SDDP.SINGLE_CUT)
        d["train_100"] = Dict("seed" => 123, "iteration_limit" => 100, "bound" => b, "log" => l)
        d["simulation_100"] = simulate_stats(m, 500; seed = 42)
        # The cyclic graph converges slowly: 100 iterations is not converged (18.69 vs 18.8168).
        m, b, l = train_bound(build_belief; seed = 123, iteration_limit = 1500, cut_type = SDDP.SINGLE_CUT)
        d["train_1500"] = Dict("seed" => 123, "iteration_limit" => 1500, "bound" => b, "log" => l)
        d["simulation_1500"] = simulate_stats(m, 500; seed = 42)
        d
    end),
    "air_conditioning" => (3, () -> begin
        d = Dict{String,Any}()
        for (dh, name) in [(SDDP.ContinuousConicDuality(), "conic"), (SDDP.LagrangianDuality(), "lagrangian"), (SDDP.StrengthenedConicDuality(), "strengthened")]
            m, b, l = train_bound(build_air_conditioning; seed = 1234, iteration_limit = 30, duality_handler = dh)
            d["train_30_" * name] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        end
        d["deterministic_equivalent"] = det_equiv(build_air_conditioning)
        d
    end),
    "stochastic_all_blacks" => (3, () -> begin
        d = Dict{String,Any}()
        for (dh, name) in [(SDDP.ContinuousConicDuality(), "conic"), (SDDP.LagrangianDuality(), "lagrangian")]
            m, b, l = train_bound(build_stochastic_all_blacks; seed = 1234, iteration_limit = 30, duality_handler = dh)
            d["train_30_" * name] = Dict("seed" => 1234, "iteration_limit" => 30, "bound" => b, "log" => l)
        end
        d["deterministic_equivalent"] = det_equiv(build_stochastic_all_blacks)
        d
    end),
    "sldp_example_one" => (3, () -> begin
        d = Dict{String,Any}()
        m, b, l = train_bound(build_sldp_example_one; seed = 1234, iteration_limit = 50)
        d["train_50"] = Dict("seed" => 1234, "iteration_limit" => 50, "bound" => b, "log" => l)
        d
    end),
)

function main(args)
    which = isempty(args) ? "all" : args[1]
    only = length(args) >= 2 ? args[2] : nothing
    tiers = which == "all" ? [1, 2, 3] : [parse(Int, replace(which, "tier" => ""))]
    vers = versions()
    for name in sort(collect(keys(PROBLEMS)))
        tier, gen = PROBLEMS[name]
        if !(tier in tiers) || (only !== nothing && only != name)
            continue
        end
        print("=== $name (tier $tier) ... "); flush(stdout)
        t0 = time()
        try
            data = gen()
            data["name"] = name
            data["tier"] = tier
            data["versions"] = vers
            data["generated_seconds"] = time() - t0
            open(joinpath(OUT, name * ".json"), "w") do io
                JSON.print(io, data, 2)
            end
            println("ok ($(round(time() - t0; digits = 1))s)")
        catch e
            println("FAILED: ", sprint(showerror, e))
            showerror(stdout, e, catch_backtrace()); println()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
