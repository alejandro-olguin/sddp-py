# Oracle for the tutorial pages example_newsvendor, duality_handlers, mdps, plotting and the
# vehicle_location example. Usage: cd reference && julia --project=. generate_tutorials_c.jl
# NEVER edit the produced JSON by hand.
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
using SDDP.JuMP

const OUTC = joinpath(@__DIR__, "oracle")
results = Dict{String,Any}()
results["versions"] = versions()

# ============================================================== example_newsvendor
# The demand sample is generated in Python (numpy Triangular(150, 250, 200), seed 20240611,
# sorted) and pasted here so both languages use identical data.
const d = [158.62449426300836, 161.01275378388056, 163.10039232377892, 165.14965956340902, 166.5208706927016, 167.18585178006475, 168.97547367627067, 171.46236603390668, 172.50565351012756, 173.38616043859764, 173.71510463846678, 174.91045093654805, 175.308560644122, 177.14276527778074, 178.00657123863135, 178.37514909512672, 180.37931168716779, 180.79292915621573, 181.28261243312176, 183.02189913209975, 183.23272679728302, 183.59006341713786, 185.00763030132373, 186.4494301172686, 187.30917674634165, 187.83793748620522, 188.15187288809602, 188.40836537189395, 188.619394547317, 189.21664844095778, 189.71875568711687, 190.1165326986805, 190.16409934023702, 190.6299296660103, 191.57240687506163, 191.89906338539134, 192.67417822961366, 193.25541305076263, 193.6279326566252, 195.30362341406345, 195.90446219831654, 196.16743062002277, 196.5108587046871, 196.86042900278704, 196.8898857687959, 198.33894922470427, 198.411975535299, 199.21387978709336, 199.58449019739743, 200.06080109826706, 200.0844016854271, 200.12278298373823, 200.41760035878298, 201.00725123552573, 201.41094239955586, 201.58315332395392, 202.300668673002, 202.3735987912538, 202.86486574345037, 203.19463591041534, 203.25008162223952, 204.53907715530974, 204.9675658106605, 207.33178996727085, 207.8388181457992, 207.86809517511608, 207.9482580091102, 208.9415617830614, 209.61691415752531, 209.83396117963036, 210.69924708869664, 210.85150177383187, 211.07187454716714, 211.22016941486135, 212.32414175654614, 213.00231023880104, 215.09820953066557, 216.48361851826039, 216.65339190851813, 216.86349349360913, 217.00069412328793, 217.2910709848719, 217.7166597261308, 218.19131069721644, 219.30970648553586, 220.68588038701913, 221.28593741950206, 222.34053022900994, 225.08298765505987, 226.0713329722414, 227.41209115024498, 230.27572260002887, 230.39764535303, 231.8063904834116, 232.14840513182028, 232.9750351537016, 233.88767713303127, 240.69886218620888, 243.1572851440932, 249.19790459173652]
const N = length(d)
const Ω = 1:N
const P = fill(1 / N, N)

nv = Dict{String,Any}("d" => d)

# Kelley's cutting plane (analytic gradient instead of ForwardDiff: not in this env)
function kelleys_cutting_plane(f, ∇f; input_dimension, upper_bound, iteration_limit, tolerance = 1e-6)
    K = 1
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, θ <= upper_bound)
    @variable(model, x[1:input_dimension])
    @objective(model, Max, θ)
    x_k = fill(NaN, input_dimension)
    lower_bound, upper_bound = -Inf, Inf
    status = ""
    while true
        optimize!(model)
        x_k .= value.(x)
        upper_bound = objective_value(model)
        lower_bound = min(upper_bound, f(x_k))
        @constraint(model, θ <= f(x_k) + ∇f(x_k)' * (x .- x_k))
        K = K + 1
        if K > iteration_limit
            status = "iteration limit"
            break
        elseif abs(upper_bound - lower_bound) < tolerance
            status = "converged"
            break
        end
    end
    return Dict("x" => x_k, "lower_bound" => lower_bound, "upper_bound" => upper_bound,
                "iterations" => K - 1, "status" => status)
end
nv["kelley"] = kelleys_cutting_plane(
    x -> -(x[1] - 1)^2 + -(x[2] + 2)^2 + 1.0,
    x -> [-2 * (x[1] - 1), -2 * (x[2] + 2)];
    input_dimension = 2, upper_bound = 10.0, iteration_limit = 20,
)

function solve_second_stage(x̅, d_ω)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x_in)
    @variable(model, x_out >= 0)
    fix(x_in, x̅)
    @variable(model, 0 <= u_sell <= d_ω)
    @constraint(model, x_out == x_in - u_sell)
    @constraint(model, u_sell <= x_in)
    @objective(model, Max, 5 * u_sell - 0.1 * x_out)
    optimize!(model)
    return (V = objective_value(model), λ = reduced_cost(x_in), x = value(x_out), u = value(u_sell))
end
let r = solve_second_stage(200, 170)
    nv["second_stage_200_170"] = Dict("V" => r.V, "lambda" => r.λ, "x" => r.x, "u" => r.u)
end

function l_shaped()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x_in == 0)
    @variable(model, x_out >= 0)
    @variable(model, u_make >= 0)
    @constraint(model, x_out == x_in + u_make)
    M = 5 * maximum(d)
    @variable(model, θ <= M)
    @objective(model, Max, -2 * u_make + θ)
    iterations = 0
    lb = ub = NaN
    for k in 1:100
        iterations = k
        optimize!(model)
        xᵏ = value(x_out)
        ub = objective_value(model)
        ret = [solve_second_stage(xᵏ, d[ω]) for ω in Ω]
        lb = value(-2 * u_make) + sum(p * r.V for (p, r) in zip(P, ret))
        if ub - lb < 1e-6
            break
        end
        @constraint(model, θ <= sum(p * (r.V + r.λ * (x_out - xᵏ)) for (p, r) in zip(P, ret)))
    end
    optimize!(model)
    x = value(x_out)
    r = solve_second_stage(x, 170.0)
    return Dict("x" => x, "objective" => objective_value(model), "lower_bound" => lb,
                "upper_bound" => ub, "iterations" => iterations,
                "second_stage_170" => Dict("V" => r.V, "lambda" => r.λ, "x" => r.x, "u" => r.u))
end
nv["l_shaped"] = l_shaped()

function build_newsvendor()
    return SDDP.LinearPolicyGraph(; stages = 2, sense = :Max, upper_bound = 5 * maximum(d),
                                  optimizer = HiGHS.Optimizer) do subproblem, stage
        @variable(subproblem, x >= 0, SDDP.State, initial_value = 0)
        if stage == 1
            @variable(subproblem, u_make >= 0)
            @constraint(subproblem, x.out == x.in + u_make)
            @stageobjective(subproblem, -2 * u_make)
        else
            @variable(subproblem, u_sell >= 0)
            @constraint(subproblem, u_sell <= x.in)
            @constraint(subproblem, x.out == x.in - u_sell)
            SDDP.parameterize(subproblem, d, P) do ω
                set_upper_bound(u_sell, ω)
                return
            end
            @stageobjective(subproblem, 5 * u_sell - 0.1 * x.out)
        end
        return
    end
end
let m = build_newsvendor()
    de = SDDP.deterministic_equivalent(m, HiGHS.Optimizer); set_silent(de); optimize!(de)
    nv["deterministic_equivalent"] = objective_value(de)
end
let m = build_newsvendor()
    Random.seed!(1)
    SDDP.train(m; log_every_iteration = true, print_level = 0, log_file = LOGFILE)
    nv["train_default"] = Dict("bound" => SDDP.calculate_bound(m),
                               "iterations" => length(m.most_recent_training_results.log))
    s1 = SDDP.evaluate(SDDP.DecisionRule(m; node = 1); incoming_state = Dict(:x => 0.0))
    s2 = SDDP.evaluate(SDDP.DecisionRule(m; node = 2); incoming_state = Dict(:x => s1.outgoing_state[:x]),
                       noise = 170.0, controls_to_record = [:u_sell])
    nv["decision_rule"] = Dict(
        "node1" => Dict("stage_objective" => s1.stage_objective, "x_out" => s1.outgoing_state[:x]),
        "node2" => Dict("stage_objective" => s2.stage_objective, "x_out" => s2.outgoing_state[:x],
                        "u_sell" => s2.controls[:u_sell]),
    )
    Random.seed!(1)
    sims = SDDP.simulate(m, 10, [:x, :u_sell, :u_make]; skip_undefined_variables = true)
    nv["simulate"] = Dict("replications" => length(sims), "stages" => length(sims[1]),
                          "keys_stage1" => sort(string.(collect(keys(sims[1][1])))),
                          "keys_stage2" => sort(string.(collect(keys(sims[1][2])))))
end

function solve_newsvendor(risk_measure; kwargs...)
    model = SDDP.LinearPolicyGraph(; stages = 2, sense = :Max, upper_bound = 5 * maximum(d),
                                   optimizer = HiGHS.Optimizer) do subproblem, node
        @variable(subproblem, x >= 0, SDDP.State, initial_value = 0)
        if node == 1
            @stageobjective(subproblem, -2 * x.out)
        else
            @variable(subproblem, u_sell >= 0)
            @constraint(subproblem, u_sell <= x.in)
            @constraint(subproblem, x.out == x.in - u_sell)
            SDDP.parameterize(subproblem, d, P) do ω
                set_upper_bound(u_sell, ω)
                return
            end
            @stageobjective(subproblem, 5 * u_sell - 0.1 * x.out)
        end
        return
    end
    Random.seed!(1)
    SDDP.train(model; risk_measure = risk_measure, print_level = 0, log_file = LOGFILE, kwargs...)
    first_stage_rule = SDDP.DecisionRule(model; node = 1)
    solution = SDDP.evaluate(first_stage_rule; incoming_state = Dict(:x => 0.0))
    return Dict("x" => solution.outgoing_state[:x], "bound" => SDDP.calculate_bound(model),
                "iterations" => length(model.most_recent_training_results.log))
end
nv["cvar_0.4"] = solve_newsvendor(SDDP.CVaR(0.4))
nv["worst_case"] = solve_newsvendor(SDDP.WorstCase())
const Γ = [10^i for i in -4:0.5:1]
nv["entropic_gammas"] = Γ
# page-faithful (default stopping rule) and fully deterministic (fixed iteration count) sweeps
nv["entropic_default"] = [solve_newsvendor(SDDP.Entropic(γ)) for γ in Γ]
nv["entropic_iter30"] = [solve_newsvendor(SDDP.Entropic(γ); iteration_limit = 30) for γ in Γ]
nv["cvar_0.4_iter30"] = solve_newsvendor(SDDP.CVaR(0.4); iteration_limit = 30)
nv["worst_case_iter30"] = solve_newsvendor(SDDP.WorstCase(); iteration_limit = 30)
results["newsvendor"] = nv
println("=== newsvendor: L-shaped x = $(nv["l_shaped"]["x"]), bound = $(nv["train_default"]["bound"])")

# ============================================================== duality_handlers
const Optimizer = optimizer_with_attributes(HiGHS.Optimizer, "presolve" => "off")

function train_and_evaluate_bounds(model_fn, duality_handler; kwargs...)
    model = model_fn()
    Random.seed!(1)
    SDDP.train(model; print_level = 0, duality_handler, log_file = LOGFILE, kwargs...)
    simulations = SDDP.simulate(model, 1)
    lower_bound = SDDP.calculate_bound(model)
    upper_bound = sum(data[:stage_objective] for data in only(simulations))
    return Dict("lower_bound" => lower_bound, "upper_bound" => upper_bound,
                "iterations" => length(model.most_recent_training_results.log))
end

function model_1()
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = Optimizer) do sp, t
        @variable(sp, x, Bin, SDDP.State, initial_value = 1.0)
        @variable(sp, y, Bin)
        @constraint(sp, x.out == x.in)
        if t == 1
            @stageobjective(sp, x.out)
        else
            @stageobjective(sp, y)
            @constraint(sp, y >= x.in - 0.5)
        end
    end
end
function model_2()
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = Optimizer) do sp, t
        @variable(sp, x, SDDP.State, initial_value = 0.1)
        @variable(sp, y, Int)
        @constraint(sp, x.out == x.in)
        if t == 1
            @stageobjective(sp, x.out)
        else
            @stageobjective(sp, y)
            @constraint(sp, y >= x.in + 0.1)
            @constraint(sp, y >= -x.in + 0.1)
        end
    end
end
function model_3()
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = -1.0, optimizer = Optimizer) do sp, t
        @variable(sp, -1 <= x <= 0.5, SDDP.State, initial_value = 0.0)
        @variable(sp, y)
        @stageobjective(sp, y)
        if t == 1
            @constraint(sp, y >= x.out)
            @constraint(sp, y >= -x.out)
        else
            @variable(sp, z, Bin)
            @constraint(sp, y >= 1 - x.in - 3 * z)
            @constraint(sp, y >= 1 + x.in - 3 * (1 - z))
        end
    end
end
function model_4()
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = -1.0, optimizer = Optimizer) do sp, t
        @variable(sp, -1 <= x <= 0.5, SDDP.State, initial_value = 0.0)
        @variable(sp, y)
        @variable(sp, z, Bin)
        if t == 1
            @stageobjective(sp, -0.1 * x.out)
        else
            @stageobjective(sp, y)
            @constraint(sp, y >= 1 - x.in - 3 * z)
            @constraint(sp, y >= 1 + x.in - 3 * (1 - z))
        end
    end
end
const MODELS = Dict("model_1" => model_1, "model_2" => model_2, "model_3" => model_3, "model_4" => model_4)
handler(name) = Dict(
    "continuous" => () -> SDDP.ContinuousConicDuality(),
    "strengthened" => () -> SDDP.StrengthenedConicDuality(),
    "lagrangian" => () -> SDDP.LagrangianDuality(),
    "bandit" => () -> SDDP.BanditDuality(SDDP.ContinuousConicDuality(), SDDP.StrengthenedConicDuality(), SDDP.LagrangianDuality()),
    "fixed_discrete" => () -> SDDP.FixedDiscreteDuality(),
)[name]()
dh = Dict{String,Any}()
for m in ["model_1", "model_2", "model_3", "model_4"], h in ["continuous", "strengthened", "lagrangian", "bandit", "fixed_discrete"]
    r = train_and_evaluate_bounds(MODELS[m], handler(h))
    dh["$(m)/$(h)"] = r
    println("=== duality $m $h: lb = $(r["lower_bound"]) ub = $(r["upper_bound"])")
end
results["duality_handlers"] = dh

# ============================================================== mdps (maze; sum-of-squares needs Ipopt)
let
    M, N = 3, 4
    initial_square = (1, 1)
    reward, illegal_squares, penalties = (3, 4), [(2, 2)], [(3, 1), (2, 4)]
    path = fill("⋅", M, N)
    path[initial_square...] = "1"
    for (k, v) in (illegal_squares => "▩", penalties => "†", [reward] => "*")
        for (i, j) in k
            path[i, j] = v
        end
    end
    discount_factor = 0.9
    graph = SDDP.UnicyclicGraph(discount_factor)
    model = SDDP.PolicyGraph(graph; sense = :Max, upper_bound = 1 / (1 - discount_factor),
                             optimizer = HiGHS.Optimizer) do sp, _
        @variable(sp, x[i = 1:M, j = 1:N], Bin, SDDP.State, initial_value = (i, j) == initial_square)
        @constraint(sp, sum(x[i, j].out for i in 1:M, j in 1:N) == 1)
        @stageobjective(sp, x[reward...].out - sum(x[i, j].out for (i, j) in penalties))
        @constraint(sp, [(i, j) in illegal_squares], x[i, j].out <= 0)
        for i in 1:M, j in 1:N
            moves = [(i - 1, j), (i + 1, j), (i, j), (i, j + 1), (i, j - 1)]
            filter!(v -> 1 <= v[1] <= M && 1 <= v[2] <= N, moves)
            @constraint(sp, x[i, j].out <= sum(x[a, b].in for (a, b) in moves))
        end
        return
    end
    Random.seed!(1)
    SDDP.train(model; print_level = 0, log_file = LOGFILE)
    Random.seed!(1)
    simulations = SDDP.simulate(model, 1, [:x];
        sampling_scheme = SDDP.InSampleMonteCarlo(; max_depth = 5, terminate_on_dummy_leaf = false))
    for (t, data) in enumerate(simulations[1]), i in 1:M, j in 1:N
        if data[:x][i, j].in > 0.5
            path[i, j] = "$t"
        end
    end
    results["mdps"] = Dict(
        "maze_bound" => SDDP.calculate_bound(model),
        "maze_iterations" => length(model.most_recent_training_results.log),
        "maze_path" => [join(path[i, :], ' ') for i in 1:M],
        "maze_stages_simulated" => length(simulations[1]),
        "sum_of_squares_optimum" => 25 / 3,
    )
    println("=== maze bound $(results["mdps"]["maze_bound"])")
    println(join(results["mdps"]["maze_path"], '\n'))
end

# ============================================================== plotting
function build_plotting_model()
    Ωp = [(inflow = 0.0, fuel_multiplier = 1.5), (inflow = 50.0, fuel_multiplier = 1.0),
          (inflow = 100.0, fuel_multiplier = 0.75)]
    return SDDP.MarkovianPolicyGraph(;
        transition_matrices = Array{Float64,2}[[1.0]', [0.75 0.25], [0.75 0.25; 0.25 0.75]],
        sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do subproblem, node
        t, markov_state = node
        @variable(subproblem, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(subproblem, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
            inflow
        end)
        @constraints(subproblem, begin
            volume.out == volume.in + inflow - hydro_generation - hydro_spill
            thermal_generation + hydro_generation == 150.0
        end)
        probability = markov_state == 1 ? [1 / 6, 1 / 3, 1 / 2] : [1 / 2, 1 / 3, 1 / 6]
        fuel_cost = [50.0, 100.0, 150.0]
        SDDP.parameterize(subproblem, Ωp, probability) do ω
            fix(inflow, ω.inflow)
            @stageobjective(subproblem, ω.fuel_multiplier * fuel_cost[t] * thermal_generation)
        end
    end
end
let pl = Dict{String,Any}()
    Ωp = [(inflow = 0.0, fuel_multiplier = 1.5), (inflow = 50.0, fuel_multiplier = 1.0),
          (inflow = 100.0, fuel_multiplier = 0.75)]
    # Page-faithful random run (only the structure is comparable).
    model = build_plotting_model()
    Random.seed!(1)
    SDDP.train(model; iteration_limit = 20, run_numerical_stability_report = false, print_level = 0, log_file = LOGFILE)
    Random.seed!(1)
    simulations = SDDP.simulate(model, 100, [:volume, :thermal_generation, :hydro_generation, :hydro_spill])
    pl["random_20"] = Dict{String,Any}("bound" => SDDP.calculate_bound(model), "replications" => length(simulations),
                           "stages" => length(simulations[1]))
    V = SDDP.ValueFunction(model[(1, 1)])
    y, duals = SDDP.evaluate(V; volume = 1)
    pl["random_20"]["value_at_1"] = Dict("height" => y, "subgradient" => duals[:volume])
    # Deterministic run: 20 fixed Historical scenarios (node, index into Ω) so Python can replay.
    scen_idx = [[(1, 1, 3), (2, 1, 1), (3, 1, 2)], [(1, 1, 1), (2, 2, 2), (3, 2, 3)],
                [(1, 1, 2), (2, 1, 1), (3, 1, 1)], [(1, 1, 3), (2, 2, 3), (3, 2, 1)],
                [(1, 1, 1), (2, 1, 2), (3, 2, 2)], [(1, 1, 2), (2, 2, 1), (3, 1, 3)],
                [(1, 1, 3), (2, 1, 3), (3, 1, 3)], [(1, 1, 1), (2, 1, 1), (3, 1, 1)],
                [(1, 1, 2), (2, 2, 2), (3, 2, 2)], [(1, 1, 3), (2, 2, 1), (3, 2, 3)],
                [(1, 1, 1), (2, 2, 3), (3, 1, 2)], [(1, 1, 2), (2, 1, 3), (3, 2, 1)],
                [(1, 1, 3), (2, 1, 2), (3, 1, 1)], [(1, 1, 1), (2, 1, 3), (3, 1, 3)],
                [(1, 1, 2), (2, 2, 1), (3, 2, 2)], [(1, 1, 3), (2, 2, 2), (3, 1, 1)],
                [(1, 1, 1), (2, 1, 1), (3, 2, 3)], [(1, 1, 2), (2, 1, 2), (3, 1, 2)],
                [(1, 1, 3), (2, 2, 3), (3, 2, 3)], [(1, 1, 1), (2, 2, 2), (3, 1, 1)]]
    scenarios = [[((t, m), Ωp[k]) for (t, m, k) in s] for s in scen_idx]
    model = build_plotting_model()
    SDDP.train(model; iteration_limit = 20, run_numerical_stability_report = false, print_level = 0,
               log_file = LOGFILE, sampling_scheme = SDDP.Historical(scenarios))
    V = SDDP.ValueFunction(model[(1, 1)])
    y, duals = SDDP.evaluate(V; volume = 1)
    pl["historical_20"] = Dict(
        "scenarios" => [[[t, m, k] for (t, m, k) in s] for s in scen_idx],
        "bound" => SDDP.calculate_bound(model),
        "log" => log_to_list(model),
        # outgoing volume of node (1, 1) in the first iteration: with θ = 0 the first-stage LP is
        # degenerate (store vs spill), so this tie decides the cut that shapes V below 50.
        "sampled_states" => [s.state[:volume] for s in model[(1, 1)].bellman_function.global_theta.sampled_states],
        "cuts" => [Dict("intercept" => c.intercept, "coefficient" => c.coefficients[:volume],
                        "active" => c.constraint_ref !== nothing) for c in model[(1, 1)].bellman_function.global_theta.cuts],
        "value_at_1" => Dict("height" => y, "subgradient" => duals[:volume]),
        "values" => [Dict("volume" => v, "height" => SDDP.evaluate(V; volume = v)[1],
                          "subgradient" => SDDP.evaluate(V; volume = v)[2][:volume]) for v in [0, 50, 100, 150, 200]],
    )
    results["plotting"] = pl
    println("=== plotting historical bound $(pl["historical_20"]["bound"]) V(1) = $y")
end

# ============================================================== vehicle_location
function vehicle_location_model()
    hospital_location = 0
    bases = vcat(hospital_location, [20, 40, 60, 80, 100])
    vehicles = [1, 2, 3]
    requests = 0:10:100
    shift_cost(src, dest) = abs(src - dest)
    dispatch_cost(base, request) = 2 * (abs(request - hospital_location) + abs(request - base))
    initial_state(b, v) = b == hospital_location ? 1.0 : 0.0
    return SDDP.LinearPolicyGraph(; stages = 10, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= location[b = bases, v = vehicles] <= 1, SDDP.State, initial_value = initial_state(b, v))
        @variables(sp, begin
            0 <= dispatch[bases, vehicles] <= 1, Bin
            0 <= shift[bases, bases, vehicles] <= 1, Bin
        end)
        @expression(sp, base_balance[b in bases, v in vehicles],
            location[b, v].in - dispatch[b, v] - sum(shift[b, :, v]) + sum(shift[:, b, v]))
        @constraints(sp, begin
            sum(dispatch) == 1
            [b in bases, v in vehicles], dispatch[b, v] <= location[b, v].in
            [b in bases, v in vehicles], sum(shift[b, :, v]) <= location[b, v].in
            [b in bases, v in vehicles], sum(shift[b, :, v]) + dispatch[b, v] <= 1
            [b in bases, v in vehicles], shift[b, b, v] == 0
            [b in bases[2:end], v in vehicles], location[b, v].out == base_balance[b, v]
            [v in vehicles], location[hospital_location, v].out == base_balance[hospital_location, v] + sum(dispatch[:, v])
        end)
        SDDP.parameterize(sp, requests) do request
            @stageobjective(sp, sum(dispatch[b, v] * dispatch_cost(b, request) +
                sum(shift_cost(b, dest) * shift[b, dest, v] for dest in bases) for b in bases, v in vehicles))
        end
    end
end
let vl = Dict{String,Any}()
    try
        model = vehicle_location_model()
        Random.seed!(1)
        t0 = time()
        SDDP.train(model; iteration_limit = 20, log_frequency = 10, cut_deletion_minimum = 100,
                   duality_handler = SDDP.ContinuousConicDuality(), print_level = 0, log_file = LOGFILE)
        vl["status"] = "ok"
        vl["bound"] = SDDP.calculate_bound(model)
        vl["time"] = time() - t0
        vl["iterations"] = length(model.most_recent_training_results.log)
        vl["bound_ge_1000"] = vl["bound"] >= 1000
        Random.seed!(1)
        sims = SDDP.simulate(model, 1, [:dispatch])
        vl["simulated_stages"] = length(sims[1])
        vl["dispatch_per_stage"] = [sum(s[:dispatch]) for s in sims[1]]
    catch err
        vl["status"] = "error"
        vl["error"] = sprint(showerror, err)
    end
    results["vehicle_location"] = vl
    println("=== vehicle_location: $(vl)")
end

open(joinpath(OUTC, "tutorials_c.json"), "w") do io
    JSON.print(io, results, 2)
end
println("=== wrote tutorials_c.json")
