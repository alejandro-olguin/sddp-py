# Oracle for the sddp.dev tutorial pages "Example: deterministic to stochastic",
# "Example: capacity expansion models" and "Example: the milk producer".
# Usage: cd reference && julia --project=. generate_tutorials_b.jl
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics

const OUTB = joinpath(@__DIR__, "oracle")
results = Dict{String,Any}()

# ------------------------------------------------------------------------------ data
csv_data = """week,inflow,demand,cost
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
# Parse the CSV by hand (CSV.jl / DataFrames.jl are not in this environment).
rows = [split(l, ",") for l in split(csv_data, "\n") if !isempty(strip(l)) && !startswith(l, "week")]
data = (
    inflow = [parse(Float64, r[2]) for r in rows],
    demand = [parse(Float64, r[3]) for r in rows],
    cost = [parse(Float64, r[4]) for r in rows],
)
T = length(data.inflow)
@assert T == 52
results["data"] = Dict("T" => T, "inflow" => data.inflow, "demand" => data.demand, "cost" => data.cost)

function bounds_at(build; seed, checkpoints, kwargs...)
    # Train once for max(checkpoints) iterations, recording the bound at each checkpoint.
    Random.seed!(seed)
    m = build()
    d = Dict{String,Any}("seed" => seed)
    done = 0
    for n in sort(collect(checkpoints))
        SDDP.train(m; iteration_limit = n - done, add_to_existing_cuts = done > 0, print_level = 0,
                   log_file = LOGFILE, run_numerical_stability_report = false, kwargs...)
        done = n
        d["bound_$n"] = SDDP.calculate_bound(m)
        println("    $n iterations: bound $(d["bound_$n"])")
    end
    d["log"] = log_to_list(m)
    return m, d
end

# ============================================================== example_reservoir
reservoir_max, reservoir_initial, flow_max = 320.0, 300, 12

let
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x_storage[1:T+1] <= reservoir_max)
    fix(x_storage[1], reservoir_initial; force = true)
    @variable(model, 0 <= u_flow[1:T] <= flow_max)
    @variable(model, 0 <= u_spill[1:T])
    @variable(model, 0 <= u_thermal[1:T])
    @variable(model, ω_inflow[1:T])
    for t in 1:T
        fix(ω_inflow[t], data.inflow[t])
    end
    @constraint(model, [t in 1:T], x_storage[t+1] == x_storage[t] - u_flow[t] - u_spill[t] + ω_inflow[t])
    @constraint(model, [t in 1:T], u_flow[t] + u_thermal[t] == data.demand[t])
    @objective(model, Min, sum(data.cost[t] * u_thermal[t] for t in 1:T))
    optimize!(model)
    results["reservoir_lp"] = Dict(
        "objective" => objective_value(model),
        "x_storage" => value.(x_storage),
        "u_flow" => value.(u_flow),
        "u_thermal" => value.(u_thermal),
    )
    println("=== reservoir_lp: ", objective_value(model))
end

function reservoir_subproblem(sp, t; stochastic)
    @variable(sp, 0 <= x_storage <= reservoir_max, SDDP.State, initial_value = reservoir_initial)
    @variable(sp, 0 <= u_flow <= flow_max)
    @variable(sp, 0 <= u_thermal)
    @variable(sp, 0 <= u_spill)
    @variable(sp, ω_inflow)
    if stochastic
        Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
        SDDP.parameterize(sp, Ω, P) do ω
            fix(ω_inflow, data.inflow[t] + ω)
            return
        end
    else
        fix(ω_inflow, data.inflow[t])
    end
    @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
    @constraint(sp, u_flow + u_thermal == data.demand[t])
    @stageobjective(sp, data.cost[t] * u_thermal)
    return
end

build_reservoir_det() = SDDP.LinearPolicyGraph(; stages = T, sense = :Min, lower_bound = 0.0,
    optimizer = HiGHS.Optimizer) do sp, t
    reservoir_subproblem(sp, t; stochastic = false)
end
build_reservoir_sto() = SDDP.LinearPolicyGraph(; stages = T, sense = :Min, lower_bound = 0.0,
    optimizer = HiGHS.Optimizer) do sp, t
    reservoir_subproblem(sp, t; stochastic = true)
end
build_reservoir_cyclic() = SDDP.PolicyGraph(SDDP.UnicyclicGraph(0.95; num_nodes = T); sense = :Min,
    lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
    reservoir_subproblem(sp, t; stochastic = true)
end

println("=== reservoir_deterministic")
let (m, d) = bounds_at(build_reservoir_det; seed = 1, checkpoints = [10])
    sims = SDDP.simulate(m, 1, [:x_storage, :u_flow, :u_thermal])
    d["u_thermal"] = [s[:u_thermal] for s in sims[1]]
    d["u_flow"] = [s[:u_flow] for s in sims[1]]
    d["x_storage_out"] = [s[:x_storage].out for s in sims[1]]
    results["reservoir_deterministic"] = d
end
println("=== reservoir_stochastic")
results["reservoir_stochastic"] = bounds_at(build_reservoir_sto; seed = 1, checkpoints = [100, 300, 600])[2]
println("=== reservoir_cyclic")
let (m, d) = bounds_at(build_reservoir_cyclic; seed = 1, checkpoints = [100, 300])
    Random.seed!(1)
    d["free_lengths"] = length.(SDDP.simulate(m, 3))
    sims = SDDP.simulate(m, 100, [:x_storage, :u_flow];
        sampling_scheme = SDDP.InSampleMonteCarlo(; max_depth = 5 * T, terminate_on_dummy_leaf = false))
    d["fixed_lengths"] = unique(length.(sims))
    results["reservoir_cyclic"] = d
end

# ============================================================= capacity_expansion
capex_reservoir_max, capex_reservoir_initial, capex_flow_max = 350.0, 300, 9

build_capex_1() = SDDP.LinearPolicyGraph(; stages = T, sense = :Min, lower_bound = 0.0,
    optimizer = HiGHS.Optimizer) do sp, t
    @variable(sp, 0 <= x_storage <= capex_reservoir_max, SDDP.State, initial_value = capex_reservoir_initial)
    @variable(sp, 0 <= u_flow <= capex_flow_max)
    @variable(sp, 0 <= u_thermal)
    @variable(sp, 0 <= u_spill)
    @variable(sp, ω_inflow)
    Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
    SDDP.parameterize(sp, Ω, P) do ω
        fix(ω_inflow, data.inflow[t] + ω)
        return
    end
    @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
    @constraint(sp, u_flow + u_thermal == data.demand[t])
    @stageobjective(sp, data.cost[t] * u_thermal)
    return
end

build_capex_2() = SDDP.LinearPolicyGraph(; stages = T + 1, sense = :Min, lower_bound = 0.0,
    optimizer = HiGHS.Optimizer) do sp, node
    @variable(sp, x_reservoir_max >= 0, SDDP.State, initial_value = 0)
    @variable(sp, x_storage >= 0, SDDP.State, initial_value = 0)
    @constraint(sp, x_storage.out <= x_reservoir_max.out)
    @variable(sp, 0 <= u_flow <= capex_flow_max)
    @variable(sp, 0 <= u_thermal)
    @variable(sp, 0 <= u_spill)
    @variable(sp, ω_inflow)
    if node == 1
        @stageobjective(sp, x_reservoir_max.out)
        @constraint(sp, x_storage.out <= capex_reservoir_initial)
    else
        t = mod(node - 1, T + 1)
        @constraint(sp, x_reservoir_max.out == x_reservoir_max.in)
        Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
        SDDP.parameterize(sp, Ω, P) do ω
            fix(ω_inflow, data.inflow[t] + ω)
            return
        end
        @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
        @constraint(sp, u_flow + u_thermal == data.demand[t])
        @stageobjective(sp, data.cost[t] * u_thermal)
    end
    return
end

build_capex_3() = SDDP.LinearPolicyGraph(; stages = T + 1, sense = :Min, lower_bound = 0.0,
    optimizer = HiGHS.Optimizer) do sp, node
    @variable(sp, x_reservoir_max >= 0, SDDP.State, initial_value = 0)
    @variable(sp, x_flow_max >= 0, SDDP.State, initial_value = 0)
    @variable(sp, x_storage >= 0, SDDP.State, initial_value = 0)
    @constraint(sp, x_storage.out <= x_reservoir_max.out)
    @variable(sp, 0 <= u_flow)
    @constraint(sp, u_flow <= x_flow_max.out)
    @variable(sp, 0 <= u_thermal)
    @variable(sp, 0 <= u_spill)
    @variable(sp, ω_inflow)
    if node == 1
        @stageobjective(sp, x_reservoir_max.out + x_flow_max.out)
        @constraint(sp, x_storage.out <= capex_reservoir_initial)
    else
        t = mod(node - 1, T + 1)
        @constraint(sp, x_reservoir_max.out == x_reservoir_max.in)
        @constraint(sp, x_flow_max.out == x_flow_max.in)
        Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
        SDDP.parameterize(sp, Ω, P) do ω
            fix(ω_inflow, data.inflow[t] + ω)
            return
        end
        @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
        @constraint(sp, u_flow + u_thermal == data.demand[t])
        @stageobjective(sp, data.cost[t] * u_thermal)
    end
    return
end

function invest_operate_twice(sp, node)
    @variable(sp, x_reservoir_max >= 0, SDDP.State, initial_value = 0)
    @variable(sp, x_flow_max >= 0, SDDP.State, initial_value = 0)
    @variable(sp, x_storage >= 0, SDDP.State, initial_value = 0)
    @constraint(sp, x_storage.out <= x_reservoir_max.out)
    @variable(sp, 0 <= u_flow)
    @constraint(sp, u_flow <= x_flow_max.out)
    @variable(sp, 0 <= u_thermal)
    @variable(sp, 0 <= u_spill)
    @variable(sp, ω_inflow)
    if node == 1
        @stageobjective(sp, x_reservoir_max.out + x_flow_max.out)
        @constraint(sp, x_storage.out <= capex_reservoir_initial)
    elseif node == T + 2
        @stageobjective(sp, (x_reservoir_max.out - x_reservoir_max.in) + (x_flow_max.out - x_flow_max.in))
        @constraint(sp, x_storage.out == x_storage.in)
    else
        t = mod(node - 1, T + 1)
        @constraint(sp, x_reservoir_max.out == x_reservoir_max.in)
        @constraint(sp, x_flow_max.out == x_flow_max.in)
        Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
        SDDP.parameterize(sp, Ω, P) do ω
            fix(ω_inflow, data.inflow[t] + ω)
            return
        end
        @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
        @constraint(sp, u_flow + u_thermal == data.demand[t])
        @stageobjective(sp, data.cost[t] * u_thermal)
    end
    return
end

build_capex_4() = SDDP.LinearPolicyGraph(invest_operate_twice; stages = 2 * T + 2, sense = :Min,
    lower_bound = 0.0, optimizer = HiGHS.Optimizer)

function build_capex_5()
    graph = SDDP.LinearGraph(2 * T + 2)
    SDDP.add_edge(graph, 2 * T + 2 => T + 3, 0.95)
    return SDDP.PolicyGraph(invest_operate_twice, graph; sense = :Min, lower_bound = 0.0,
        optimizer = HiGHS.Optimizer)
end

function build_capex_6()
    graph = SDDP.Graph((:root, 0))
    SDDP.add_node(graph, (:invest_1, 0))
    SDDP.add_node(graph, (:invest_2, 0))
    for t in 1:52
        SDDP.add_node(graph, (:Y1, t))
        SDDP.add_node(graph, (:Y2_normal, t))
        SDDP.add_node(graph, (:Y2_high, t))
    end
    for t in 2:52
        SDDP.add_edge(graph, (:Y1, t - 1) => (:Y1, t), 1.0)
        SDDP.add_edge(graph, (:Y2_normal, t - 1) => (:Y2_normal, t), 1.0)
        SDDP.add_edge(graph, (:Y2_high, t - 1) => (:Y2_high, t), 1.0)
    end
    SDDP.add_edge(graph, (:root, 0) => (:invest_1, 0), 1.0)
    SDDP.add_edge(graph, (:invest_1, 0) => (:Y1, 1), 1.0)
    SDDP.add_edge(graph, (:Y1, 52) => (:invest_2, 0), 0.9)
    SDDP.add_edge(graph, (:invest_2, 0) => (:Y2_normal, 1), 0.5)
    SDDP.add_edge(graph, (:invest_2, 0) => (:Y2_high, 1), 0.5)
    SDDP.add_edge(graph, (:Y2_normal, 52) => (:Y2_normal, 1), 0.9)
    SDDP.add_edge(graph, (:Y2_high, 52) => (:Y2_high, 1), 0.9)
    return SDDP.PolicyGraph(graph; sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, (node, t)
        @variable(sp, x_reservoir_max >= 0, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= x_flow_max <= 20, SDDP.State, initial_value = 0)
        @variable(sp, x_storage >= 0, SDDP.State, initial_value = 0)
        @constraint(sp, x_storage.out <= x_reservoir_max.out)
        @variable(sp, 0 <= u_flow)
        @constraint(sp, u_flow <= x_flow_max.out)
        @variable(sp, 0 <= u_thermal)
        @variable(sp, 0 <= u_spill)
        @variable(sp, ω_inflow)
        if node == :invest_1
            @stageobjective(sp, x_reservoir_max.out + x_flow_max.out)
            @constraint(sp, x_storage.out <= capex_reservoir_initial)
        elseif node == :invest_2
            @stageobjective(sp, (x_reservoir_max.out - x_reservoir_max.in) + (x_flow_max.out - x_flow_max.in))
            @constraint(sp, x_storage.out == x_storage.in)
        else
            @constraint(sp, x_reservoir_max.out == x_reservoir_max.in)
            @constraint(sp, x_flow_max.out == x_flow_max.in)
            Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
            scale = node == :Y2_high ? 1.5 : 1.0
            SDDP.parameterize(sp, Ω, P) do ω
                fix(ω_inflow, scale * data.inflow[t] + ω)
                return
            end
            @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
            @constraint(sp, u_flow + u_thermal == scale * data.demand[t])
            @stageobjective(sp, data.cost[t] * u_thermal)
        end
        return
    end
end

const EPI_T, EPI_P = 3, 0.9
function epicycles_graph()
    Tn, p = EPI_T, EPI_P
    graph = SDDP.Graph((:root, 0))
    SDDP.add_node(graph, (:inv, 0))
    SDDP.add_node(graph, (:inv_h, 0))
    SDDP.add_node(graph, (:inv_l, 0))
    SDDP.add_node(graph, (:inv_hh, 0))
    SDDP.add_node(graph, (:inv_hl, 0))
    SDDP.add_node(graph, (:inv_lh, 0))
    SDDP.add_node(graph, (:inv_ll, 0))
    SDDP.add_edge(graph, (:root, 0) => (:inv, 0), 1.0)
    SDDP.add_edge(graph, (:inv, 0) => (:inv_h, 0), p^Tn / 2)
    SDDP.add_edge(graph, (:inv, 0) => (:inv_l, 0), p^Tn / 2)
    SDDP.add_edge(graph, (:inv_h, 0) => (:inv_hh, 0), p^Tn / 2)
    SDDP.add_edge(graph, (:inv_h, 0) => (:inv_hl, 0), p^Tn / 2)
    SDDP.add_edge(graph, (:inv_l, 0) => (:inv_lh, 0), p^Tn / 2)
    SDDP.add_edge(graph, (:inv_l, 0) => (:inv_ll, 0), p^Tn / 2)
    SDDP.add_node(graph, (:op, 1))
    for t in 2:52
        SDDP.add_node(graph, (:op, t))
        SDDP.add_edge(graph, (:op, t - 1) => (:op, t), 1.0)
    end
    SDDP.add_edge(graph, (:op, 52) => (:op, 1), p)
    SDDP.add_edge(graph, (:inv, 0) => (:op, 1), Tn * (1 - p))
    SDDP.add_edge(graph, (:inv_h, 0) => (:op, 1), Tn * (1 - p))
    SDDP.add_edge(graph, (:inv_l, 0) => (:op, 1), Tn * (1 - p))
    SDDP.add_edge(graph, (:inv_hh, 0) => (:op, 1), 1.0)
    SDDP.add_edge(graph, (:inv_hl, 0) => (:op, 1), 1.0)
    SDDP.add_edge(graph, (:inv_lh, 0) => (:op, 1), 1.0)
    SDDP.add_edge(graph, (:inv_ll, 0) => (:op, 1), 1.0)
    return graph
end

function build_capex_7()
    return SDDP.PolicyGraph(epicycles_graph(); sense = :Min, lower_bound = 0.0,
        optimizer = HiGHS.Optimizer) do sp, (node, t)
        @variable(sp, x_reservoir_max >= 0, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= x_flow_max <= 20, SDDP.State, initial_value = 0)
        @variable(sp, x_storage >= 0, SDDP.State, initial_value = 0)
        @variable(sp, x_scale, SDDP.State, initial_value = 1)
        @variable(sp, u_flow >= 0)
        @variable(sp, u_thermal >= 0)
        @variable(sp, u_spill >= 0)
        @variable(sp, ω_inflow)
        if t > 0
            @stageobjective(sp, data.cost[t] * u_thermal)
            @constraint(sp, x_reservoir_max.out == x_reservoir_max.in)
            @constraint(sp, x_flow_max.out == x_flow_max.in)
            @constraint(sp, x_scale.out == x_scale.in)
            @constraint(sp, x_storage.out <= x_reservoir_max.out)
            @constraint(sp, u_flow <= x_flow_max.out)
            @constraint(sp, x_storage.out == x_storage.in - u_flow - u_spill + ω_inflow)
            @constraint(sp, u_flow + u_thermal == x_scale.in * data.demand[t])
            @constraint(sp, c_ω, ω_inflow - x_scale.in * data.inflow[t] == 0)
            Ω, P = [-2, 0, 5], [0.3, 0.4, 0.3]
            SDDP.parameterize(sp, Ω, P) do ω
                set_normalized_coefficient(c_ω, x_scale.in, -(data.inflow[t] + ω))
                return
            end
        else
            @stageobjective(sp, 12 * (x_reservoir_max.out - x_reservoir_max.in) + (x_flow_max.out - x_flow_max.in))
            @constraint(sp, x_storage.out == 0.8 * x_reservoir_max.out)
            if endswith("$node", "h")
                @constraint(sp, x_scale.out == 1.5 * x_scale.in)
            elseif endswith("$node", "l")
                @constraint(sp, x_scale.out == 0.8 * x_scale.in)
            else
                @constraint(sp, x_scale.out == x_scale.in)
            end
        end
        return
    end
end

function epicycles_sample_scenario()
    D = SDDP.Noise.([-2, 0, 5], [0.3, 0.4, 0.3])
    inv_1 = Symbol("inv_$(rand((:l, :h)))")
    inv_2 = Symbol("$(inv_1)$(rand((:l, :h)))")
    return vcat(
        ((:inv, 0), nothing),
        [((:op, t), SDDP.sample_noise(D)) for t in 1:52 for year in 1:EPI_T],
        ((inv_1, 0), nothing),
        [((:op, t), SDDP.sample_noise(D)) for t in 1:52 for year in 1:EPI_T],
        ((inv_2, 0), nothing),
        [((:op, t), SDDP.sample_noise(D)) for t in 1:52 for year in 1:EPI_T],
    )
end

for (name, build, checkpoints) in [
    ("capex_operational", build_capex_1, [100, 300]),
    ("capex_invest_then_operate", build_capex_2, [100, 300]),
    ("capex_multiple_investments", build_capex_3, [100, 300]),
    ("capex_invest_operate_invest_operate", build_capex_4, [100, 300]),
    ("capex_loop", build_capex_5, [100, 300]),
    ("capex_strategic_uncertainty", build_capex_6, [100, 300]),
    ("capex_epicycles", build_capex_7, [200, 300]),
]
    println("=== $name")
    m, d = bounds_at(build; seed = 1, checkpoints = checkpoints)
    d["page_iterations"] = checkpoints[1]
    Random.seed!(1)
    if name in ("capex_loop", "capex_strategic_uncertainty")
        sims = SDDP.simulate(m, 100, [:x_storage, :u_flow, :x_reservoir_max, :x_flow_max];
            sampling_scheme = SDDP.InSampleMonteCarlo(; max_depth = 5 * 52 + 2, terminate_on_dummy_leaf = false))
        d["simulation_lengths"] = unique(length.(sims))
    elseif name == "capex_epicycles"
        sims = SDDP.simulate(m, 100, [:x_storage, :u_flow, :x_reservoir_max, :x_flow_max];
            sampling_scheme = SDDP.Historical([epicycles_sample_scenario() for _ in 1:100]))
        d["simulation_lengths"] = unique(length.(sims))
        d["simulation_objective_mean"] = mean(sum(s[:stage_objective] for s in sim) for sim in sims)
    else
        sims = SDDP.simulate(m, 100, [:x_storage, :u_flow])
        d["simulation_lengths"] = unique(length.(sims))
        d["simulation_objective_mean"] = mean(sum(s[:stage_objective] for s in sim) for sim in sims)
    end
    results[name] = d
end

# ================================================================= milk producer
function milk_simulator()
    residuals = [0.0987, 0.199, 0.303, 0.412, 0.530, 0.661, 0.814, 1.010, 1.290]
    residuals = 0.1 * vcat(-residuals, 0.0, residuals)
    scenario = zeros(12)
    y, μ, α = 4.5, 6.0, 0.05
    for t in 1:12
        y = exp((1 - α) * log(y) + α * log(μ) + rand(residuals))
        scenario[t] = clamp(y, 3.0, 9.0)
    end
    return scenario
end

function build_milk_producer(graph)
    return SDDP.PolicyGraph(graph; sense = :Max, upper_bound = 1e2, optimizer = HiGHS.Optimizer) do sp, node
        t, price = node::Tuple{Int,Float64}
        c_transaction = 0.01
        c_buy_premium = 1.5
        c_contango = 1.05
        Ω_production = range(0.1, 0.2; length = 5)
        c_max_production = 12 * maximum(Ω_production)
        @variable(sp, 0 <= x_stock, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= x_forward[1:4], SDDP.State, initial_value = 0)
        @variable(sp, 0 <= u_spot_sell <= c_max_production)
        @variable(sp, 0 <= u_spot_buy <= c_max_production)
        c_max_futures = t <= 8 ? c_max_production : 0.0
        @variable(sp, 0 <= u_forward_sell <= c_max_futures)
        @variable(sp, ω_production)
        @constraint(sp, [i in 1:3], x_forward[i].out == x_forward[i+1].in)
        @constraint(sp, x_forward[4].out == u_forward_sell)
        @constraint(sp, x_stock.out == x_stock.in + ω_production + u_spot_buy - x_forward[1].in - u_spot_sell)
        Ω = [(price, p) for p in Ω_production]
        SDDP.parameterize(sp, Ω) do ω
            fix(ω_production, ω[2])
            @stageobjective(sp,
                ω[1] * (u_spot_sell - c_buy_premium * u_spot_buy) +
                (ω[1] * c_contango - c_transaction) * u_forward_sell)
            return
        end
        return
    end
end

println("=== milk_producer")
let d = Dict{String,Any}()
    Random.seed!(1)
    graph = SDDP.MarkovianGraph(milk_simulator; budget = 30, scenarios = 10_000)
    # Serialise the fitted graph so Python can build the identical model.
    d["graph"] = Dict(
        "root" => collect(graph.root_node),
        "nodes" => [collect(n) for n in keys(graph.nodes) if n != graph.root_node],
        "edges" => [[collect(parent), collect(child), p] for (parent, edges) in graph.nodes for (child, p) in edges],
    )
    risk = 0.5 * SDDP.Expectation() + 0.5 * SDDP.AVaR(0.25)
    m, dd = bounds_at(() -> build_milk_producer(graph); seed = 1, checkpoints = [100, 400],
        risk_measure = risk, sampling_scheme = SDDP.SimulatorSamplingScheme(milk_simulator))
    merge!(d, dd)
    # Expectation-only training on the same graph: its converged bound is a sharper parity target.
    _, de = bounds_at(() -> build_milk_producer(graph); seed = 1, checkpoints = [400],
        sampling_scheme = SDDP.SimulatorSamplingScheme(milk_simulator))
    d["expectation_bound_400"] = de["bound_400"]
    Random.seed!(2)
    sims = SDDP.simulate(m, 200, [:x_stock, :u_forward_sell, :u_spot_sell, :u_spot_buy];
        sampling_scheme = SDDP.SimulatorSamplingScheme(milk_simulator))
    d["stage12_node_index"] = collect(sims[1][12][:node_index])
    d["stage12_noise_term"] = collect(sims[1][12][:noise_term])
    d["simulation_objective_mean"] = mean(sum(s[:stage_objective] for s in sim) for sim in sims)
    results["milk_producer"] = d
end

results["versions"] = versions()
open(joinpath(OUTB, "tutorials_b.json"), "w") do io
    JSON.print(io, results, 2)
end
println("ALL DONE")
