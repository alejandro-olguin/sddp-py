# Oracle generator for the remaining modules (value functions, inner approximation, MSPFormat,
# StochOptFormat, lattice fitting, biobjective, stability report, extra forward passes).
# Usage: julia --project=. generate_tier4.jl
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
import SDDP: Inner, MSPFormat

const OUT4 = joinpath(@__DIR__, "oracle")
vers = versions()
const ONLY = Set(ARGS)
should(name) = isempty(ONLY) || name in ONLY

function save(name, d)
    d["name"] = name; d["versions"] = vers
    open(joinpath(OUT4, name * ".json"), "w") do io
        JSON.print(io, d, 2)
    end
    println("=== $name ok")
end

# ---------------------------------------------------------------- value functions
if should("value_functions")
let d = Dict{String,Any}()
    scen = [[(1, 0.0), (2, 50.0), (3, 100.0)], [(1, 100.0), (2, 0.0), (3, 50.0)], [(1, 50.0), (2, 50.0), (3, 0.0)],
            [(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 100.0), (2, 100.0), (3, 100.0)]]
    m = build_hydro_thermal()
    SDDP.train(m; iteration_limit = 20, print_level = 0, log_file = LOGFILE, sampling_scheme = SDDP.Historical(scen), run_numerical_stability_report = false)
    d["hydro_thermal_deterministic_20"] = Dict("scenarios" => scen, "nodes" => Dict())
    for node in 1:3
        V = SDDP.ValueFunction(m; node = node)
        pts = [0.0, 25.0, 50.0, 100.0, 150.0, 200.0]
        d["hydro_thermal_deterministic_20"]["nodes"]["$node"] = [Dict("volume" => x, "value" => (r = SDDP.evaluate(V, Dict(:volume => x)); r[1]), "dual" => SDDP.evaluate(V, Dict(:volume => x))[2][:volume]) for x in pts]
    end
    # multi-cut + CVaR (from SDDP.jl's own test, but recorded from Julia)
    m2 = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        SDDP.parameterize(sp, [1, 2]) do w
            @stageobjective(sp, w * x.out)
        end
    end
    SDDP.train(m2; iteration_limit = 2, print_level = 0, log_file = LOGFILE, risk_measure = SDDP.CVaR(0.25), cut_type = SDDP.MULTI_CUT)
    V = SDDP.ValueFunction(m2[1])
    d["multicut_cvar"] = [Dict("x" => x, "value" => SDDP.evaluate(V, Dict(:x => x))[1], "dual" => SDDP.evaluate(V, Dict(:x => x))[2][:x]) for x in [0.0, 1.0, 2.0]]
    # objective state
    m3 = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 1.5)
        SDDP.add_objective_state(sp; initial_value = 0.0, lipschitz = 10.0) do p, ω
            return p + ω
        end
        @constraint(sp, x.out == x.in)
        SDDP.parameterize(sp, [1, 2]) do ω
            price = SDDP.objective_state(sp)
            @stageobjective(sp, price * x.out)
        end
    end
    SDDP.train(m3; iteration_limit = 10, print_level = 0, log_file = LOGFILE)
    V = SDDP.ValueFunction(m3[1])
    d["objective_state"] = [Dict("x" => x, "objective_state" => y, "value" => SDDP.evaluate(V, Dict(:x => x); objective_state = y)[1], "dual" => SDDP.evaluate(V, Dict(:x => x); objective_state = y)[2][:x]) for (x, y) in [(1.0, 1), (0.0, 2), (2.0, 1)]]
    # belief state
    graph = SDDP.MarkovianGraph(Matrix{Float64}[[0.5 0.5], [1.0 0.0; 0.0 1.0]])
    SDDP.add_ambiguity_set(graph, [(1, 1), (1, 2)])
    SDDP.add_ambiguity_set(graph, [(2, 1), (2, 2)])
    m4 = SDDP.PolicyGraph(graph; lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, node
        (t, i) = node
        @variable(sp, x >= 0, SDDP.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        P = [[0.2, 0.8], [0.8, 0.2]]
        SDDP.parameterize(sp, [1, 2], P[i]) do ω
            @stageobjective(sp, ω * x.out)
        end
    end
    SDDP.train(m4; iteration_limit = 10, print_level = 0, log_file = LOGFILE)
    V11 = SDDP.ValueFunction(m4[(1, 1)])
    b = Dict((1, 1) => 0.8, (1, 2) => 0.2)
    y, duals = SDDP.evaluate(V11, Dict(:x => 1.0); belief_state = b)
    d["belief_state"] = Dict("x" => 1.0, "belief" => Dict("(1, 1)" => 0.8, "(1, 2)" => 0.2), "value" => y, "dual" => duals[:x])
    save("value_functions", d)
end
end

# ---------------------------------------------------------------- inner approximation
if should("inner")
function build_inner(sp, t)
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
    SDDP.parameterize(sp, [(inflow = 0.0, demand = 7.5), (inflow = 5.0, demand = 5.0), (inflow = 10.0, demand = 2.5)]) do ω
        JuMP.fix(inflow, ω.inflow)
        JuMP.fix(demand, ω.demand)
    end
end
let d = Dict{String,Any}()
    nstages = 4
    Ω = [(inflow = 0.0, demand = 7.5), (inflow = 5.0, demand = 5.0), (inflow = 10.0, demand = 2.5)]
    Random.seed!(11)
    scen = [[(t, Ω[rand(1:3)]) for t in 1:nstages] for _ in 1:12]
    d["scenarios"] = [[(t, Dict(pairs(ω))) for (t, ω) in s] for s in scen]
    model = SDDP.LinearPolicyGraph(build_inner; stages = nstages, lower_bound = 0.0, optimizer = HiGHS.Optimizer)
    d["initial_bound"] = SDDP.calculate_bound(model)
    SDDP.train(model; iteration_limit = 50, print_level = 0, log_file = LOGFILE, sampling_scheme = SDDP.Historical(scen), run_numerical_stability_report = false)
    d["outer_bound_50"] = SDDP.calculate_bound(model)
    base_Lip = 11.0; base_ub = 17.5 * base_Lip
    ibf = Inner.InnerBellmanFunction(t -> base_Lip * (nstages - t); upper_bound = t -> base_ub * (nstages - t), vertex_type = SDDP.SINGLE_CUT)
    model_inner, ub, _ = Inner.inner_dp(build_inner, model; stages = nstages, sense = :Min, optimizer = HiGHS.Optimizer, lower_bound = 0.0, bellman_function = ibf, risk_measure = SDDP.Expectation(), print_level = 0)
    d["inner_upper_bound"] = ub
    d["inner_bound_recomputed"] = SDDP.calculate_bound(model_inner)
    d["lipschitz"] = [base_Lip * (nstages - t) for t in 1:nstages]
    d["upper_bounds"] = [base_ub * (nstages - t) for t in 1:nstages]
    model_2 = SDDP.PolicyGraph(build_inner, SDDP.LinearGraph(nstages); lower_bound = 0.0, optimizer = HiGHS.Optimizer, bellman_function = ibf)
    for (k, node) in model_2.nodes
        SDDP.set_objective(node)
    end
    d["empty_inner_bound"] = SDDP.calculate_bound(model_2)
    # vertices per node (values & states) for exact comparison
    d["vertices"] = Dict("$k" => [Dict("value" => v.value, "state" => v.state) for v in node.bellman_function.global_theta.vertices] for (k, node) in model_inner.nodes)
    # Number of vertices kept after selection (Julia deletes variables of removed vertices)
    d["active_vertices"] = Dict("$k" => count(v -> v.variable_ref !== nothing && JuMP.is_valid(node.subproblem, v.variable_ref), node.bellman_function.global_theta.vertices) for (k, node) in model_inner.nodes)
    # dp_vertices_from_visited_states with InnerPolicyGraph (20 stages, deterministic outer)
    Random.seed!(12)
    scen20 = [[(t, Ω[rand(1:3)]) for t in 1:20] for _ in 1:30]
    d["scenarios_20"] = [[(t, Dict(pairs(ω))) for (t, ω) in s] for s in scen20]
    cut_model = SDDP.LinearPolicyGraph(build_inner; stages = 20, lower_bound = 0.0, optimizer = HiGHS.Optimizer)
    SDDP.train(cut_model; iteration_limit = 200, print_level = 0, log_file = LOGFILE, sampling_scheme = SDDP.Historical(scen20), run_numerical_stability_report = false)
    d["cut_model_20_bound"] = SDDP.calculate_bound(cut_model)
    vertex_model = Inner.InnerPolicyGraph(build_inner, SDDP.LinearGraph(20); lower_bound = 0.0, upper_bound = 1000, optimizer = HiGHS.Optimizer, lipschitz_constant = 10.0)
    Inner.dp_vertices_from_visited_states(vertex_model, cut_model; optimizer = HiGHS.Optimizer, print_level = 0)
    d["vertex_model_20_bound"] = SDDP.calculate_bound(vertex_model)
    vertex_model2 = Inner.InnerPolicyGraph(build_inner, SDDP.LinearGraph(20); lower_bound = 0.0, upper_bound = 1000, optimizer = HiGHS.Optimizer, lipschitz_constant = 10.0)
    Inner.dp_vertices_from_visited_states(vertex_model2, cut_model; print_level = 0)
    d["vertex_model_20_bound_no_selection"] = SDDP.calculate_bound(vertex_model2)
    save("inner", d)
end
end

# ---------------------------------------------------------------- MSPFormat
if should("mspformat")
let d = Dict{String,Any}()
    for (name, its) in [("hydro_thermal", 40), ("electric", 60)]
        m = MSPFormat.read_from_file(joinpath(@__DIR__, "msp", name))
        JuMP.set_optimizer(m, HiGHS.Optimizer)
        Random.seed!(1)
        SDDP.train(m; iteration_limit = its, print_level = 0, log_file = LOGFILE)
        d[name] = Dict("iteration_limit" => its, "bound" => SDDP.calculate_bound(m), "nodes" => sort(collect(keys(m.nodes))),
                       "noise_counts" => Dict(k => length(n.noise_terms) for (k, n) in m.nodes),
                       "children" => Dict(k => [(c.term, c.probability) for c in n.children] for (k, n) in m.nodes),
                       "initial_state" => Dict(String(k) => v for (k, v) in m.initial_root_state))
        fresh = MSPFormat.read_from_file(joinpath(@__DIR__, "msp", name))
        det = SDDP.deterministic_equivalent(fresh, HiGHS.Optimizer); set_silent(det); optimize!(det)
        d[name]["deterministic_equivalent"] = objective_value(det)
    end
    m = MSPFormat.read_from_file(joinpath(@__DIR__, "msp", "electric.problem.json"), joinpath(@__DIR__, "msp", "electric-tree.lattice.json"))
    d["electric_tree"] = Dict("nodes" => sort(collect(keys(m.nodes))), "children" => Dict(k => [(c.term, c.probability) for c in n.children] for (k, n) in m.nodes),
                              "noise_counts" => Dict(k => length(n.noise_terms) for (k, n) in m.nodes))
    JuMP.set_optimizer(m, HiGHS.Optimizer); SDDP.train(m; iteration_limit = 30, print_level = 0, log_file = LOGFILE)
    d["electric_tree"]["bound"] = SDDP.calculate_bound(m)
    save("mspformat", d)
end
end

# ---------------------------------------------------------------- StochOptFormat
if should("stochoptformat")
function build_experimental(minimization::Bool)
    return SDDP.PolicyGraph(SDDP.LinearGraph(3); sense = minimization ? :Min : :Max, lower_bound = -50.0, upper_bound = 50.0) do sp, t
        N = 2; C = [0.2, 0.7]; S = 2 .+ [0.33, 0.54]; DEMAND = [2, 10]
        @variable(sp, x[1:N] >= 0, SDDP.State, initial_value = 0.0)
        @variables(sp, begin
            s[i = 1:N] >= 0
            d
        end)
        @constraints(sp, begin
            [i = 1:N], s[i] <= x[i].in
            c, sum(s) <= d + 1
        end)
        SDDP.parameterize(sp, t == 1 ? [1] : 1:length(DEMAND)) do ω
            JuMP.fix(d, DEMAND[ω])
            set_upper_bound(s[1], 0.1 * ω)
            set_lower_bound(x[1].out, ω)
            set_normalized_rhs(c, ω)
            sgn = minimization ? 1.0 : -1.0
            @stageobjective(sp, sgn * (sum(C[i] * x[i].out for i in 1:N) - S[ω] * s[ω] - s[ω] * S[ω] + ω))
        end
    end
end
let d = Dict{String,Any}()
    m, validation = SDDP.read_from_file(joinpath(@__DIR__, "sof", "electric.sof.json"))
    set_optimizer(m, HiGHS.Optimizer)
    SDDP.train(m; iteration_limit = 60, print_level = 0, log_file = LOGFILE)
    ev = SDDP.evaluate(m, validation)
    d["electric"] = Dict("bound" => SDDP.calculate_bound(m), "sha256" => ev["problem_sha256_checksum"],
                         "scenario_objectives" => [[s["objective"] for s in sc] for sc in ev["scenarios"]],
                         "scenario_primals" => [[s["primal"] for s in sc] for sc in ev["scenarios"]],
                         "n_nodes" => length(m.nodes), "nodes" => sort(collect(keys(m.nodes))))
    for (mn, label) in [(true, "min"), (false, "max")]
        base = build_experimental(mn); set_optimizer(base, HiGHS.Optimizer); Random.seed!(3)
        SDDP.train(base; iteration_limit = 50, print_level = 0, log_file = LOGFILE)
        model = build_experimental(mn)
        Random.seed!(5)
        file = joinpath(@__DIR__, "sof", "experimental_$label.sof.json")
        SDDP.write_to_file(model, file; validation_scenarios = 10, sampling_scheme = SDDP.PSRSamplingScheme(2))
        new_model, vs = SDDP.read_from_file(file)
        set_optimizer(new_model, HiGHS.Optimizer); Random.seed!(3)
        SDDP.train(new_model; iteration_limit = 50, print_level = 0, log_file = LOGFILE)
        ev = SDDP.evaluate(new_model, vs)
        det = SDDP.deterministic_equivalent(build_experimental(mn), HiGHS.Optimizer); set_silent(det); optimize!(det)
        d["experimental_$label"] = Dict("base_bound" => SDDP.calculate_bound(base), "roundtrip_bound" => SDDP.calculate_bound(new_model),
            "deterministic_equivalent" => objective_value(det), "sha256" => ev["problem_sha256_checksum"],
            "scenario_objectives" => [[s["objective"] for s in sc] for sc in ev["scenarios"]],
            "scenario_primals" => [[s["primal"] for s in sc] for sc in ev["scenarios"]])
    end
    save("stochoptformat", d)
end
end

# ---------------------------------------------------------------- lattice approximation
if should("lattice")
let d = Dict{String,Any}()
    Random.seed!(21)
    sims = [cumsum(rand(5)) for _ in 1:60]
    d["simulations"] = sims
    states = [1, 2, 3, 2, 1]
    support, probability = SDDP._lattice_approximation(() -> nothing, states, 60, sims)
    d["states"] = states
    d["support"] = support
    d["probability"] = [collect(eachrow(p)) for p in probability]
    d["budget_alloc_10"] = SDDP._allocate_support_budget(sims, 10, 60)
    d["budget_alloc_3"] = SDDP._allocate_support_budget(sims, 3, 60)
    d["find_min"] = [SDDP.find_min([1.0, 2.0, 3.0], 2.1), SDDP.find_min([1.0, 2.0, 3.0], 0.0), SDDP.find_min([1.0, 2.0, 3.0], 5.0)]
    # MarkovianGraph from these simulations (budget = 8): use the internal path with fixed sims
    st = SDDP._allocate_support_budget(sims, 8, 60)
    sup2, prob2 = SDDP._lattice_approximation(() -> nothing, st, 60, sims)
    d["graph8"] = Dict("states" => st, "support" => sup2, "probability" => [collect(eachrow(p)) for p in prob2])
    save("lattice", d)
end
end

# ---------------------------------------------------------------- biobjective
if should("biobjective")
let d = Dict{String,Any}()
    m = SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, _
        @variable(sp, 0 <= v <= 200, SDDP.State, initial_value = 50)
        @variables(sp, begin
            0 <= g[i = 1:2] <= 100
            0 <= u <= 150
            s >= 0
            shortage_cost >= 0
        end)
        @expressions(sp, begin
            objective_1, g[1] + 10 * g[2]
            objective_2, shortage_cost
        end)
        @constraints(sp, begin
            inflow_constraint, v.out == v.in - u - s
            g[1] + g[2] + u == 150
            shortage_cost >= 40 - v.out
            shortage_cost >= 60 - 2 * v.out
            shortage_cost >= 80 - 4 * v.out
        end)
        SDDP.initialize_biobjective_subproblem(sp)
        SDDP.parameterize(sp, 0.0:5:50.0) do ω
            set_normalized_rhs(inflow_constraint, ω)
            SDDP.set_biobjective_functions(sp, objective_1, objective_2)
        end
    end
    Random.seed!(7)
    sols = SDDP.train_biobjective(m; solution_limit = 5, iteration_limit = 500, print_level = 0, log_file_prefix = tempname())
    d["solutions"] = Dict(string(k) => v for (k, v) in sols)
    save("biobjective", d)
end
end

# ---------------------------------------------------------------- numerical stability report, extra forward passes, binary expansion
if should("misc_tier4")
let d = Dict{String,Any}()
    m = build_hydro_thermal()
    d["hydro_thermal_report"] = sprint(io -> SDDP.numerical_stability_report(io, m))
    m_bad = SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 1e9, SDDP.State, initial_value = 1.0)
        @variable(sp, u >= 0)
        @constraint(sp, 1e-6 * x.out + u >= 1e-5)
        @stageobjective(sp, 1e8 * u + x.out)
    end
    d["bad_report"] = sprint(io -> SDDP.numerical_stability_report(io, m_bad))
    d["bad_report_by_node"] = sprint(io -> SDDP.numerical_stability_report(io, m_bad; by_node = true))
    Random.seed!(1)
    m = build_hydro_thermal()
    SDDP.train(m; iteration_limit = 60, print_level = 0, log_file = LOGFILE, forward_pass = SDDP.ImportanceSamplingForwardPass(), cut_type = SDDP.MULTI_CUT)
    d["importance_sampling_bound_60"] = SDDP.calculate_bound(m)
    m = build_hydro_thermal(); fwd = build_hydro_thermal()
    Random.seed!(1)
    SDDP.train(m; iteration_limit = 40, print_level = 0, log_file = LOGFILE, forward_pass = SDDP.AlternativeForwardPass(fwd), post_iteration_callback = SDDP.AlternativePostIterationCallback(fwd))
    d["alternative_forward_bound_40"] = SDDP.calculate_bound(m)
    d["alternative_forward_model_bound_40"] = SDDP.calculate_bound(fwd)
    d["binexpand"] = Dict("5_5" => SDDP.binexpand(5, 5), "0.56_0.56_0.1" => SDDP.binexpand(0.56, 0.56, 0.1), "0.54_0.54_0.01" => SDDP.binexpand(0.54, 0.54, 0.01),
                          "bincontract_101" => SDDP.bincontract([1, 0, 1]), "bincontract_eps" => SDDP.bincontract([1, 0, 1], 0.1))
    save("misc_tier4", d)
end
end
println("ALL DONE")
