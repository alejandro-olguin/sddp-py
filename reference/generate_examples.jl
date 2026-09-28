# Oracle for additional documented examples (docs/src/examples). Usage: julia --project=. generate_examples.jl
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
const OUTE = joinpath(@__DIR__, "oracle")
vers = versions()
results = Dict{String,Any}()

function record!(name, build; iters, det = true, seed = 1, kwargs...)
    d = Dict{String,Any}("iteration_limit" => iters)
    if det
        m = build()
        de = SDDP.deterministic_equivalent(m, HiGHS.Optimizer); set_silent(de); optimize!(de)
        d["deterministic_equivalent"] = objective_value(de)
    end
    Random.seed!(seed)
    m = build()
    SDDP.train(m; iteration_limit = iters, print_level = 0, log_file = LOGFILE, kwargs...)
    d["bound"] = SDDP.calculate_bound(m)
    results[name] = d
    println("=== $name: bound $(d["bound"])" * (det ? ", det $(d["deterministic_equivalent"])" : ""))
    return m
end

# ---- asset_management_simple (https://sddp.dev/stable/examples/asset_management_simple/)
function build_asset_management_simple()
    return SDDP.PolicyGraph(SDDP.MarkovianGraph(Array{Float64,2}[[1.0]', [0.5 0.5], [0.5 0.5; 0.5 0.5], [0.5 0.5; 0.5 0.5]]);
                            lower_bound = -1_000.0, optimizer = HiGHS.Optimizer) do sp, index
        (stage, markov_state) = index
        r_stock = [1.25, 1.06]; r_bonds = [1.14, 1.12]
        @variable(sp, stocks >= 0, SDDP.State, initial_value = 0.0)
        @variable(sp, bonds >= 0, SDDP.State, initial_value = 0.0)
        if stage == 1
            @constraint(sp, stocks.out + bonds.out == 55)
            @stageobjective(sp, 0)
        elseif 1 < stage < 4
            @constraint(sp, r_stock[markov_state] * stocks.in + r_bonds[markov_state] * bonds.in == stocks.out + bonds.out)
            @stageobjective(sp, 0)
        else
            @variable(sp, over >= 0)
            @variable(sp, short >= 0)
            @constraint(sp, r_stock[markov_state] * stocks.in + r_bonds[markov_state] * bonds.in - over + short == 80)
            @stageobjective(sp, -over + 4 * short)
        end
    end
end
m = record!("asset_management_simple", build_asset_management_simple; iters = 60)
results["asset_management_simple"]["simulation"] = simulate_stats(m, 1000; seed = 42)

# ---- agriculture_mccardle_farm
function build_mccardle()
    S = [0 1 2; 0 0 1; 0 0 0]; t = [60, 60, 245]; D = [210, 210, 858]
    q = [[4.5 4.5 4.5; 4.5 4.5 4.5; 4.5 4.5 4.5], [5.5 5.5 5.5; 5.5 5.5 5.5; 5.5 5.5 5.5], [6.5 6.5 6.5; 6.5 6.5 6.5; 6.5 6.5 6.5]]
    b = [30 75 37.5; 15 37.5 18.25; 7.5 18.75 9.325]
    w = 3000; C = [50 50 50; 50 50 50; 50 50 50]
    r = [[5 5 5; 5 5 5; 5 5 5], [6 6 6; 6 6 6; 6 6 6], [7 7 7; 7 7 7; 7 7 7]]
    M = 60.0; H = 0.0; V = [0.05, 0.05, 0.05]; L = 3000.0
    graph = SDDP.MarkovianGraph([ones(Float64, 1, 1), [0.14 0.69 0.17], [0.14 0.69 0.17; 0.14 0.69 0.17; 0.14 0.69 0.17], [0.14 0.69 0.17; 0.14 0.69 0.17; 0.14 0.69 0.17]])
    return SDDP.PolicyGraph(graph; lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, index
        stage, weather = index
        @variable(sp, 0 <= acres <= M, SDDP.State, initial_value = M)
        @variable(sp, bales[i = 1:3] >= 0, SDDP.State, initial_value = (i == 1 ? H : 0))
        @variables(sp, begin
            buy[1:3] >= 0
            sell[1:3] >= 0
            eat[1:3] >= 0
            pen_p[1:3] >= 0
            pen_n[1:3] >= 0
        end)
        if stage == 1
            @constraint(sp, acres.out <= acres.in)
            @constraint(sp, [i = 1:3], bales[i].in == bales[i].out)
        else
            @expression(sp, cut_ex[c = 1:3], bales[c].in + buy[c] - eat[c] - sell[c] + pen_p[c] - pen_n[c])
            @constraints(sp, begin
                acres.out <= acres.in
                sum(eat) >= D[stage-1]
                bales[stage-1].out == cut_ex[stage-1] + acres.in * b[stage-1, weather]
                [c = 1:3; c != stage - 1], bales[c].out == cut_ex[c]
                sum(bales[i].out for i in 1:3) <= w
                [c = 1:3], sell[c] <= bales[c].in
                sum(sell) <= L
            end)
        end
        if stage == 1
            @stageobjective(sp, 0.0)
        else
            @stageobjective(sp, 1000 * (sum(pen_p) + sum(pen_n)) + C[stage-1, weather] * acres.in +
                sum(V[stage-1] * bales[cutting].in * t[stage-1] + r[cutting][stage-1, weather] * buy[cutting] +
                    S[cutting, stage-1] * eat[cutting] - q[cutting][stage-1, weather] * sell[cutting] for cutting in 1:3))
        end
    end
end
record!("agriculture_mccardle_farm", build_mccardle; iters = 60)

# ---- generation_expansion (integer states, conic duality)
function build_generation_expansion()
    build_cost = 1e4; use_cost = 4; num_units = 5; capacities = ones(num_units)
    demand_vals = 0.5 * [5 5 5 5 5 5 5 5; 4 3 1 3 0 9 8 17; 0 9 4 2 19 19 13 7; 25 11 4 14 4 6 15 12; 6 7 5 3 8 4 17 13]
    penalty = 5e5; rho = 0.99
    return SDDP.LinearPolicyGraph(; stages = 5, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= invested[1:num_units] <= 1, SDDP.State, Int, initial_value = 0)
        @variables(sp, begin
            generation >= 0
            unmet >= 0
            demand
        end)
        @constraints(sp, begin
            investment[i in 1:num_units], invested[i].out >= invested[i].in
            sum(capacities[i] * invested[i].out for i in 1:num_units) >= generation
            unmet >= demand - sum(generation)
            [j in 1:(num_units-1)], invested[j].out <= invested[j+1].out
        end)
        SDDP.parameterize(ω -> fix(demand, ω), sp, demand_vals[stage, :])
        @expression(sp, investment_cost, build_cost * sum(invested[i].out - invested[i].in for i in 1:num_units))
        @stageobjective(sp, (investment_cost + generation * use_cost) * rho^(stage - 1) + penalty * unmet)
    end
end
record!("generation_expansion", build_generation_expansion; iters = 100, det = false)

# ---- hydro_valley
struct Turbine; flowknots::Vector{Float64}; powerknots::Vector{Float64}; end
struct Reservoir; min::Float64; max::Float64; initial::Float64; turbine::Turbine; spill_cost::Float64; inflows::Vector{Float64}; end
function hydro_valley_model(; hasstagewiseinflows::Bool = true, hasmarkovprice::Bool = true, sense::Symbol = :Max)
    valley_chain = [Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 20, 50]),
                    Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 0, 20])]
    turbine(i) = valley_chain[i].turbine
    prices = [1 2 0; 2 1 0; 3 4 0]
    transition = hasmarkovprice ? Array{Float64,2}[[1.0]', [0.6 0.4], [0.6 0.4 0.0; 0.3 0.7 0.0]] : [ones(Float64, (1, 1)) for t in 1:3]
    flipobj = (sense == :Max) ? 1.0 : -1.0
    lower = (sense == :Max) ? -Inf : -1e6; upper = (sense == :Max) ? 1e6 : Inf
    N = length(valley_chain)
    return SDDP.MarkovianPolicyGraph(; sense = sense, lower_bound = lower, upper_bound = upper, transition_matrices = transition, optimizer = HiGHS.Optimizer) do sp, node
        t, markov_state = node
        @variable(sp, valley_chain[r].min <= reservoir[r = 1:N] <= valley_chain[r].max, SDDP.State, initial_value = valley_chain[r].initial)
        @variables(sp, begin
            outflow[r = 1:N] >= 0
            spill[r = 1:N] >= 0
            inflow[r = 1:N] >= 0
            generation_quantity >= 0
            0 <= dispatch[r = 1:N, level = 1:length(turbine(r).flowknots)] <= 1
            rainfall[i = 1:N]
        end)
        @constraints(sp, begin
            reservoir[1].out == reservoir[1].in + inflow[1] - outflow[1] - spill[1]
            flow[i = 2:N], reservoir[i].out == reservoir[i].in + inflow[i] - outflow[i] - spill[i] + outflow[i-1] + spill[i-1]
            generation_quantity == sum(turbine(r).powerknots[level] * dispatch[r, level] for r in 1:N for level in 1:length(turbine(r).powerknots))
            turbineflow[r = 1:N], outflow[r] == sum(turbine(r).flowknots[level] * dispatch[r, level] for level in 1:length(turbine(r).flowknots))
            dispatched[r = 1:N], sum(dispatch[r, level] for level in 1:length(turbine(r).flowknots)) <= 1
        end)
        if hasstagewiseinflows && t > 1
            @constraint(sp, inflow_noise[i = 1:N], inflow[i] <= rainfall[i])
            SDDP.parameterize(sp, [(valley_chain[1].inflows[i], valley_chain[2].inflows[i]) for i in 1:length(transition)]) do ω
                for i in 1:N
                    fix(rainfall[i], ω[i])
                end
            end
        else
            @constraint(sp, initial_inflow_noise[i = 1:N], inflow[i] <= valley_chain[i].inflows[1])
        end
        if hasmarkovprice
            @stageobjective(sp, flipobj * (prices[t, markov_state] * generation_quantity - sum(valley_chain[i].spill_cost * spill[i] for i in 1:N)))
        else
            @stageobjective(sp, flipobj * (prices[t, 1] * generation_quantity - sum(valley_chain[i].spill_cost * spill[i] for i in 1:N)))
        end
    end
end
record!("hydro_valley_deterministic", () -> hydro_valley_model(hasmarkovprice = false, hasstagewiseinflows = false); iters = 20)
record!("hydro_valley_stagewise", () -> hydro_valley_model(hasmarkovprice = false); iters = 60)
record!("hydro_valley_markov", () -> hydro_valley_model(hasstagewiseinflows = false); iters = 30)
record!("hydro_valley_markov_stagewise", () -> hydro_valley_model(); iters = 80)
record!("hydro_valley_riskaverse", () -> hydro_valley_model(); iters = 80, det = false, risk_measure = SDDP.EAVaR(; lambda = 0.5, beta = 0.66))
record!("hydro_valley_worst_case_min", () -> hydro_valley_model(sense = :Min); iters = 80, det = false, risk_measure = SDDP.EAVaR(; lambda = 0.5, beta = 0.0))
record!("hydro_valley_dro_1_6", () -> hydro_valley_model(hasmarkovprice = false); iters = 80, det = false, risk_measure = SDDP.ModifiedChiSquared(1 / 6))
record!("hydro_valley_dro_worst", () -> hydro_valley_model(hasmarkovprice = false); iters = 60, det = false, risk_measure = SDDP.ModifiedChiSquared(sqrt(2 / 3) - 1e-6))

# ---- booking_management (conic duality)
function booking_management_model(num_days, num_rooms, num_requests)
    max_revenue = (num_rooms + num_requests) * num_days * num_rooms
    booking_requests = Array{Int,2}[]
    for room in 1:num_rooms, day in 1:num_days, length_of_stay in 0:(num_days-day)
        req = zeros(Int, (num_rooms, num_days))
        req[room:room, day .+ (0:length_of_stay)] .= 1
        push!(booking_requests, req)
    end
    return SDDP.LinearPolicyGraph(; stages = num_requests, upper_bound = max_revenue, sense = :Max, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= vacancy[room = 1:num_rooms, day = 1:num_days] <= 1, SDDP.State, Bin, initial_value = 1)
        @variables(sp, begin
            0 <= accept_request <= 1, Bin
            0 <= room_request_accepted[1:num_rooms, 1:num_days] <= 1, Bin
            req[1:num_rooms, 1:num_days]
        end)
        for room in 1:num_rooms, day in 1:num_days
            @constraints(sp, begin
                vacancy[room, day].out == vacancy[room, day].in - room_request_accepted[room, day]
                room_request_accepted[room, day] <= vacancy[room, day].in
                room_request_accepted[room, day] <= accept_request
                room_request_accepted[room, day] <= req[room, day]
                room_request_accepted[room, day] + (1 - accept_request) >= req[room, day]
            end)
        end
        SDDP.parameterize(sp, booking_requests) do request
            fix.(req, request)
        end
        @stageobjective(sp, sum((room + stage - 1) * room_request_accepted[room, day] for room in 1:num_rooms for day in 1:num_days))
    end
end
record!("booking_1_2_5", () -> booking_management_model(1, 2, 5); iters = 60)
record!("booking_2_2_3", () -> booking_management_model(2, 2, 3); iters = 60)

# ---- StructDualDynProg prob5.2
function build_prob52(stages)
    return SDDP.LinearPolicyGraph(; stages = stages, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        n = 4; m = 3; i_c = [16, 5, 32, 2]; C = [25, 80, 6.5, 160]; T = [8760, 7000, 1500] / 8760
        D2 = [diff([0, 3919, 7329, 10315]) diff([0, 7086, 9004, 11169])]; p2 = [0.9, 0.1]
        @variable(sp, x[i = 1:n] >= 0, SDDP.State, initial_value = 0.0)
        @variables(sp, begin
            y[1:n, 1:m] >= 0
            v[1:n] >= 0
            penalty >= 0
            ξ[j = 1:m]
        end)
        @constraints(sp, begin
            [i = 1:n], x[i].out == x[i].in + v[i]
            [i = 1:n], sum(y[i, :]) <= x[i].in
            [j = 1:m], sum(y[:, j]) + penalty >= ξ[j]
        end)
        pen = stages == 2 ? 1e6 : 1e5
        @stageobjective(sp, i_c' * v + C' * y * T + pen * penalty)
        if t != 1
            SDDP.parameterize(sp, 1:size(D2, 2), p2) do ω
                for j in 1:m
                    fix(ξ[j], D2[j, ω])
                end
            end
        end
        if t == stages
            @constraint(sp, sum(v) == 0)
        end
    end
end
record!("prob52_2stages", () -> build_prob52(2); iters = 60)
record!("prob52_3stages", () -> build_prob52(3); iters = 100)

# ---- multistock
function build_multistock()
    return SDDP.LinearPolicyGraph(; stages = 5, lower_bound = -5.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= stock[i = 1:3] <= 1, SDDP.State, initial_value = 0.5)
        @variables(sp, begin
            0 <= control[i = 1:3] <= 0.5
            ξ[i = 1:3]
        end)
        @constraints(sp, begin
            sum(control) - 0.5 * 3 <= 0
            [i = 1:3], stock[i].out == stock[i].in + control[i] - ξ[i]
        end)
        Ξ = collect(Base.product((0.0, 0.15, 0.3), (0.0, 0.15, 0.3), (0.0, 0.15, 0.3)))[:]
        SDDP.parameterize(sp, Ξ) do ω
            fix.(ξ, ω)
        end
        @stageobjective(sp, (sin(3 * stage) - 1) * sum(control))
    end
end
m = record!("multistock", build_multistock; iters = 1500, det = false, cut_type = SDDP.SINGLE_CUT)
results["multistock"]["simulation"] = simulate_stats(m, 2000; seed = 42)

# ---- all_blacks (Lagrangian)
function build_all_blacks()
    (T, N, R, offer) = (3, 2, [3 3 6; 3 3 6], [1 1 0; 1 0 1])
    return SDDP.LinearPolicyGraph(; stages = T, sense = :Max, upper_bound = 100.0, optimizer = HiGHS.Optimizer) do sp, stage
        @variable(sp, 0 <= x[1:N] <= 1, SDDP.State, Bin, initial_value = 1)
        @variable(sp, accept_offer, Bin)
        @constraint(sp, [i in 1:N], x[i].out == x[i].in - offer[i, stage] * accept_offer)
        @stageobjective(sp, sum(R[i, stage] * offer[i, stage] * accept_offer for i in 1:N))
    end
end
record!("all_blacks_lagrangian", build_all_blacks; iters = 30, duality_handler = SDDP.LagrangianDuality())
record!("all_blacks_conic", build_all_blacks; iters = 30, det = false)

# ---- air_conditioning_forward (alternative forward pass)
function create_air_conditioning_model(; convex::Bool)
    return SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 100, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= u_production <= 200)
        @variable(sp, u_overtime >= 0)
        if !convex
            set_integer(x.out); set_integer(u_production); set_integer(u_overtime)
        end
        @constraint(sp, demand, x.in - x.out + u_production + u_overtime == 0)
        Ω = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        SDDP.parameterize(ω -> set_normalized_rhs(demand, ω), sp, Ω[t])
        @stageobjective(sp, 100 * u_production + 300 * u_overtime + 50 * x.out)
    end
end
let
    convex = create_air_conditioning_model(convex = true); non_convex = create_air_conditioning_model(convex = false)
    Random.seed!(1)
    SDDP.train(convex; forward_pass = SDDP.AlternativeForwardPass(non_convex), post_iteration_callback = SDDP.AlternativePostIterationCallback(non_convex), iteration_limit = 20, print_level = 0, log_file = LOGFILE)
    results["air_conditioning_forward"] = Dict("iteration_limit" => 20, "bound_convex" => SDDP.calculate_bound(convex), "bound_non_convex" => SDDP.calculate_bound(non_convex))
    println("=== air_conditioning_forward: ", results["air_conditioning_forward"])
end

# ---- sldp_example_two
function build_sldp_two(N)
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = -100.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x[1:2] <= 5, SDDP.State, initial_value = 0.0)
        if t == 1
            @variable(sp, 0 <= u[1:2] <= 5, Int)
            @constraint(sp, [i = 1:2], u[i] == x[i].out)
            @stageobjective(sp, -1.5 * x[1].out - 4 * x[2].out)
        else
            @variable(sp, 0 <= y[1:4] <= 1, Bin)
            @variable(sp, ω[1:2])
            @stageobjective(sp, -16 * y[1] - 19 * y[2] - 23 * y[3] - 28 * y[4])
            @constraint(sp, 2 * y[1] + 3 * y[2] + 4 * y[3] + 5 * y[4] <= ω[1] - x[1].in)
            @constraint(sp, 6 * y[1] + 1 * y[2] + 3 * y[3] + 2 * y[4] <= ω[2] - x[2].in)
            steps = range(5; stop = 15, length = N)
            SDDP.parameterize(sp, [[i, j] for i in steps for j in steps]) do φ
                fix.(ω, φ)
            end
        end
    end
end
for N in (2, 3, 6)
    record!("sldp_two_N$N", () -> build_sldp_two(N); iters = 60)
end

results["versions"] = vers
open(joinpath(OUTE, "examples.json"), "w") do io
    JSON.print(io, results, 2)
end
println("ALL DONE")
