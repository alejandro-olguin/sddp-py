# Oracle for the how-to guides (docs/src/guides). Usage: cd reference && julia --project=. generate_guides.jl
# NEVER edit the produced JSON file by hand.
include(joinpath(@__DIR__, "generate.jl"))
using SDDP, HiGHS, JSON, Random, Statistics
const OUTG = joinpath(@__DIR__, "oracle")
results = Dict{String,Any}()
results["versions"] = versions()
say(x...) = (println(x...); flush(stdout))

function det_equiv_value(model)
    de = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
    set_silent(de)
    optimize!(de)
    return objective_value(de)
end

function train_silent(model; kwargs...)
    SDDP.train(model; print_level = 0, log_file = LOGFILE, kwargs...)
    return SDDP.calculate_bound(model)
end

# ------------------------------------------------------------ access_previous_variables
function build_capacity_model(w)
    return SDDP.LinearPolicyGraph(; stages = 10, sense = :Max, upper_bound = 100.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, capacity >= 0, SDDP.State, initial_value = 0)
        @variable(sp, reservoir >= 0, SDDP.State, initial_value = 0)
        @variable(sp, generation >= 0)
        if t == 1
            @constraint(sp, reservoir.out == reservoir.in)
            @stageobjective(sp, -capacity.out)
        else
            @constraint(sp, balance, reservoir.out - reservoir.in + generation == 0)
            @constraint(sp, generation <= capacity.in)
            @constraint(sp, capacity.out == capacity.in)
            @stageobjective(sp, generation)
            SDDP.parameterize(sp, w) do ω
                set_normalized_rhs(balance, ω)
            end
        end
    end
end

function build_pipeline_model()
    return SDDP.LinearPolicyGraph(; stages = 10, sense = :Max, upper_bound = 100, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, inventory >= 0, SDDP.State, initial_value = 0)
        @variable(sp, pipeline[1:5], SDDP.State, initial_value = 0)
        @variable(sp, 0 <= buy <= 10)
        @variable(sp, sell >= 0)
        @constraint(sp, pipeline[1].out == buy)
        @constraint(sp, [i = 2:5], pipeline[i].out == pipeline[i-1].in)
        @constraint(sp, inventory.out == inventory.in - sell + pipeline[5].in)
        @stageobjective(sp, sell)
    end
end

# `corrected = true` puts `u_buy` on the right-hand side (coefficient -1 in normalized form),
# which is what the prose of the guide describes; the guide's code sets +1.
function build_lead_time_model(; corrected::Bool = false)
    T = 10
    return SDDP.LinearPolicyGraph(; stages = 20, sense = :Max, upper_bound = 1000, optimizer = HiGHS.Optimizer) do sp, t
        @variables(sp, begin
            x_inventory >= 0, SDDP.State, (initial_value = 0)
            x_pipeline[1:T+1], SDDP.State, (initial_value = 0)
            0 <= u_buy <= 10
            u_sell >= 0
        end)
        fix(x_pipeline[T+1].out, 0)
        @stageobjective(sp, u_sell)
        @constraints(sp, begin
            c_pipeline[i = 1:T], x_pipeline[i].out == x_pipeline[i+1].in + 1 * u_buy
            x_inventory.out == x_inventory.in - u_sell + x_pipeline[1].in
        end)
        SDDP.parameterize(sp, 1:T) do ω
            for i in 1:T
                v = ω == i ? 1 : 0
                set_normalized_coefficient(c_pipeline[i], u_buy, corrected ? -v : v)
            end
        end
    end
end

let d = Dict{String,Any}()
    w = [0.2, 0.5, 0.7, 0.9]  # the guide uses rand(4); fixed here so Python can reproduce it
    d["capacity_w"] = w
    Random.seed!(1)
    d["capacity_bound_300"] = train_silent(build_capacity_model(w); iteration_limit = 300)
    d["pipeline_deterministic_equivalent"] = det_equiv_value(build_pipeline_model())
    Random.seed!(1)
    m = build_pipeline_model()
    d["pipeline_bound_100"] = train_silent(m; iteration_limit = 100)
    d["pipeline_bound_log"] = [l.bound for l in m.most_recent_training_results.log]
    Random.seed!(1)
    d["lead_time_bound_30"] = train_silent(build_lead_time_model(); iteration_limit = 30)
    Random.seed!(1)
    d["lead_time_corrected_bound_100"] = train_silent(build_lead_time_model(corrected = true); iteration_limit = 100)
    results["access_previous_variables"] = d
    say("access_previous_variables: ", d)
end

# ------------------------------------------------------------------ add_a_custom_cut
function create_custom_cut_model()
    return SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 100, Int, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= u_p <= 200, Int)
        @variable(sp, u_o >= 0, Int)
        @variable(sp, w)
        @constraint(sp, x.out == x.in + u_p + u_o - w)
        @stageobjective(sp, 100 * u_p + 300 * u_o + 50 * x.out)
        Ω = [[100.0], [100.0, 300.0], [100.0, 300.0]]
        SDDP.parameterize(ω -> fix(w, ω), sp, Ω[t])
    end
end

let d = Dict{String,Any}()
    Random.seed!(1)
    model = create_custom_cut_model()
    SDDP.train(model; iteration_limit = 1, print_level = 0, log_file = LOGFILE)
    d["cuts_after_one_iteration"] = cuts_json(model)
    model = create_custom_cut_model()
    d["bound_fresh"] = SDDP.calculate_bound(model)
    f = tempname() * ".json"
    write(f, """
    [{
        "node": "1",
        "single_cuts": [{
            "state": {"x": 10.0},
            "coefficients": {"x": -200.0},
            "intercept": 55500.0
        }],
        "risk_set_cuts": [],
        "multi_cuts": []
    }]
    """)
    SDDP.read_cuts_from_file(model, f)
    d["bound_after_custom_cut"] = SDDP.calculate_bound(model)
    results["add_a_custom_cut"] = d
    say("add_a_custom_cut: ", d)
end

# ------------------------------------------------ add_a_multidimensional_state_variable
let d = Dict{String,Any}()
    lbs = Float64[]
    model = SDDP.LinearPolicyGraph(; stages = 1, lower_bound = 0, optimizer = HiGHS.Optimizer) do subproblem, t
        @variable(subproblem, x >= 0, SDDP.State, initial_value = 0)
        push!(lbs, lower_bound(x.out))
        @variable(subproblem, y[i = 1:2] >= i, SDDP.State, initial_value = i)
        push!(lbs, lower_bound(y[1].out))
        @variable(subproblem, z[i = 3:4, j = [:A, :B]] >= i, SDDP.State, initial_value = i)
        push!(lbs, lower_bound(z[3, :B].out))
    end
    d["lower_bounds"] = lbs
    d["num_nodes"] = length(model.nodes)
    results["add_a_multidimensional_state_variable"] = d
end

# ----------------------------------------------------------------- add_a_risk_measure
function build_first_steps_hydro_thermal()
    return SDDP.LinearPolicyGraph(; stages = 3, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variable(sp, thermal_generation >= 0)
        @variable(sp, hydro_generation >= 0)
        @variable(sp, hydro_spill >= 0)
        @variable(sp, inflow)
        SDDP.parameterize(sp, [0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3]) do ω
            fix(inflow, ω)
        end
        @constraint(sp, volume.out == volume.in - hydro_generation - hydro_spill + inflow)
        @constraint(sp, hydro_generation + thermal_generation == 150.0)
        fuel_cost = [50.0, 100.0, 150.0]
        @stageobjective(sp, fuel_cost[t] * thermal_generation)
    end
end

let d = Dict{String,Any}()
    d["expectation_deterministic_equivalent"] = det_equiv_value(build_first_steps_hydro_thermal())
    Random.seed!(1)
    d["worst_case_measure"] = train_silent(build_first_steps_hydro_thermal(); risk_measure = SDDP.WorstCase(), iteration_limit = 50)
    Random.seed!(1)
    d["worst_case_dict"] = train_silent(build_first_steps_hydro_thermal(); risk_measure = Dict(1 => SDDP.WorstCase(), 2 => SDDP.WorstCase(), 3 => SDDP.WorstCase()), iteration_limit = 50)
    Random.seed!(1)
    d["worst_case_function"] = train_silent(build_first_steps_hydro_thermal(); risk_measure = (node_index) -> SDDP.WorstCase(), iteration_limit = 50)
    Random.seed!(1)
    d["mixed_dict"] = train_silent(build_first_steps_hydro_thermal(); risk_measure = Dict(1 => SDDP.Expectation(), 2 => SDDP.WorstCase(), 3 => SDDP.WorstCase()), iteration_limit = 50)
    Random.seed!(1)
    d["mixed_function"] = train_silent(build_first_steps_hydro_thermal(); risk_measure = (node_index) -> node_index == 1 ? SDDP.Expectation() : SDDP.WorstCase(), iteration_limit = 50)
    d["iteration_limit"] = 50
    results["add_a_risk_measure"] = d
    say("add_a_risk_measure: ", d)
end

# ------------------------------------------------------------------ add_integrality
function build_integrality_model()
    return SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= x <= 100, Int, SDDP.State, initial_value = 0)
        @variable(sp, 0 <= u <= 200, integer = true)
        @variable(sp, v >= 0)
        @constraint(sp, x.out == x.in + u + v - 150)
        @stageobjective(sp, 2u + 6v + x.out)
    end
end

let d = Dict{String,Any}()
    d["deterministic_equivalent"] = det_equiv_value(build_integrality_model())
    Random.seed!(1)
    d["bound_default_20"] = train_silent(build_integrality_model(); iteration_limit = 20)
    Random.seed!(1)
    d["bound_bandit_20"] = train_silent(build_integrality_model(); iteration_limit = 20, duality_handler = SDDP.BanditDuality())
    Random.seed!(1)
    d["bound_conic_with_optimizer_20"] = train_silent(
        build_integrality_model();
        iteration_limit = 20,
        duality_handler = SDDP.ContinuousConicDuality(optimizer_with_attributes(HiGHS.Optimizer, "presolve" => "off")),
    )
    results["add_integrality"] = d
    say("add_integrality: ", d)
end

# ----------------------------------------------------------- add_multidimensional_noise
binomial_pmf(n, p, k) = binomial(n, k) * p^k * (1 - p)^(n - k)
poisson_pmf(λ, k) = exp(-λ) * λ^k / factorial(k)

let d = Dict{String,Any}()
    model = SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0, optimizer = HiGHS.Optimizer) do subproblem, t
        @variable(subproblem, x, SDDP.State, initial_value = 0.0)
        Ω = [(value = v, coefficient = c) for v in [1, 2] for c in [3, 4, 5]]
        P = [v * c for v in [0.5, 0.5] for c in [0.3, 0.5, 0.2]]
        SDDP.parameterize(subproblem, Ω, P) do ω
            fix(x.out, ω.value)
            @stageobjective(subproblem, ω.coefficient * x.out)
        end
    end
    d["simple_deterministic_equivalent"] = det_equiv_value(model)
    Random.seed!(1)
    d["simple_bound_2"] = train_silent(model; iteration_limit = 2)
    # Finite discrete distributions, enumerated by hand (Distributions.jl is not in this env).
    # Order mirrors `Base.product(supports...)`: the first factor varies fastest.
    supports = [collect(0:10), [0, 1], collect(2:8)]
    pmfs = [
        Dict(k => binomial_pmf(10, 0.5, k) for k in supports[1]),
        Dict(0 => 0.5, 1 => 0.5),
        let raw = Dict(k => poisson_pmf(5.0, k) for k in supports[3])
            s = sum(values(raw))
            Dict(k => v / s for (k, v) in raw)
        end,
    ]
    Ω = vec([collect(ω) for ω in Base.product(supports...)])
    P = [prod(pmfs[i][ω[i]] for i in 1:3) for ω in Ω]
    d["finite_omega"] = Ω
    d["finite_P"] = P
    model = SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0, optimizer = HiGHS.Optimizer) do subproblem, t
        @variable(subproblem, x, SDDP.State, initial_value = 0.0)
        SDDP.parameterize(subproblem, Ω, P) do ω
            fix(x.out, ω[1])
            @stageobjective(subproblem, ω[2] * x.out + ω[3])
        end
    end
    Random.seed!(1)
    d["finite_bound_2"] = train_silent(model; iteration_limit = 2)
    results["add_multidimensional_noise"] = d
    say("add_multidimensional_noise: ", d["simple_deterministic_equivalent"], " ", d["finite_bound_2"])
end

# --------------------------------------------------- add_noise_in_the_constraint_matrix
function build_constraint_matrix_model()
    return SDDP.LinearPolicyGraph(; stages = 3, lower_bound = 0, optimizer = HiGHS.Optimizer) do subproblem, t
        @variable(subproblem, x, SDDP.State, initial_value = 0.0)
        @constraint(subproblem, emissions, 1 * x.out <= 1)
        SDDP.parameterize(subproblem, [0.2, 0.5, 1.0]) do ω
            set_normalized_coefficient(emissions, x.out, ω)
            return
        end
        @stageobjective(subproblem, -x.out)
    end
end

let d = Dict{String,Any}()
    d["deterministic_equivalent"] = det_equiv_value(build_constraint_matrix_model())
    Random.seed!(1)
    d["bound_10"] = train_silent(build_constraint_matrix_model(); iteration_limit = 10)
    results["add_noise_in_the_constraint_matrix"] = d
    say("add_noise_in_the_constraint_matrix: ", d)
end

# ------------------------------------------------------------- choose_a_stopping_rule
let d = Dict{String,Any}()
    status(kwargs...) = begin
        Random.seed!(1)
        m = build_first_steps_hydro_thermal()
        SDDP.train(m; print_level = 0, log_file = LOGFILE, kwargs...)
        string(SDDP.termination_status(m))
    end
    d["iteration_limit"] = status(:iteration_limit => 10)
    d["time_limit"] = status(:time_limit => 0.0)
    d["bound_stalling"] = status(:stopping_rules => [SDDP.BoundStalling(10; rtol = 1e-4)])
    d["time_limit_or_bound_stalling"] = status(:stopping_rules => [SDDP.TimeLimit(100.0), SDDP.BoundStalling(10; rtol = 1e-4)])
    d["time_limit_and_bound_stalling"] = status(:stopping_rules => [SDDP.StoppingChain(SDDP.TimeLimit(0.0), SDDP.BoundStalling(10; rtol = 1e-4))])
    d["default"] = status()
    d["supported"] = [string(SDDP.stopping_rule_status(r)) for r in (
        SDDP.IterationLimit(1), SDDP.TimeLimit(1.0), SDDP.Statistical(; num_replications = 1, disable_warning = true),
        SDDP.BoundStalling(1), SDDP.StoppingChain(SDDP.IterationLimit(1), SDDP.TimeLimit(1.0)),
        SDDP.SimulationStoppingRule(), SDDP.FirstStageStoppingRule(),
    )]
    results["choose_a_stopping_rule"] = d
    say("choose_a_stopping_rule: ", d)
end

# --------------------------------------------------------------- create_a_belief_state
let d = Dict{String,Any}()
    G = SDDP.MarkovianGraph([[0.5 0.5], [0.2 0.8; 0.8 0.2]])
    d["repr_before"] = sprint(show, G)
    for t in 1:2
        SDDP.add_ambiguity_set(G, [(t, 1), (t, 2)])
    end
    d["repr"] = sprint(show, G)
    results["create_a_belief_state"] = d
end

# ------------------------------------------------------- create_a_general_policy_graph
let d = Dict{String,Any}()
    graph = SDDP.LinearGraph(3)
    d["linear_graph"] = sprint(show, graph)
    SDDP.add_node(graph, 4)
    SDDP.add_edge(graph, 3 => 4, 1.0)
    SDDP.add_edge(graph, 4 => 1, 0.9)
    d["linear_graph_cyclic"] = sprint(show, graph)
    d["unicyclic"] = sprint(show, SDDP.UnicyclicGraph(0.95; num_nodes = 2))
    d["markovian"] = sprint(show, SDDP.MarkovianGraph(Matrix{Float64}[[1.0]', [0.4 0.6]]))
    graph = SDDP.Graph(:root_node)
    d["general_empty"] = sprint(show, graph)
    SDDP.add_node(graph, :decision_node)
    SDDP.add_edge(graph, :root_node => :decision_node, 1.0)
    SDDP.add_edge(graph, :decision_node => :decision_node, 0.9)
    d["general"] = sprint(show, graph)
    graph = SDDP.Graph(:root_node, [:decision_node], [(:root_node => :decision_node, 1.0), (:decision_node => :decision_node, 0.9)])
    called = String[]
    SDDP.PolicyGraph(graph; lower_bound = 0, optimizer = HiGHS.Optimizer) do subproblem, node
        push!(called, string(node))
    end
    d["called_from"] = called
    results["create_a_general_policy_graph"] = d
end

# ------------------------------------------------------------- deterministic_equivalent
function build_det_equiv_model()
    return SDDP.LinearPolicyGraph(; stages = 2, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do subproblem, t
        @variable(subproblem, x, SDDP.State, initial_value = 1)
        @variable(subproblem, y)
        @constraint(subproblem, balance, x.in == x.out + y)
        SDDP.parameterize(subproblem, [1.1, 2.2]) do ω
            @stageobjective(subproblem, ω * x.out)
            fix(y, ω)
        end
    end
end

let d = Dict{String,Any}()
    det_equiv = SDDP.deterministic_equivalent(build_det_equiv_model(), HiGHS.Optimizer)
    d["num_variables"] = num_variables(det_equiv)
    d["num_equality_rows"] = num_constraints(det_equiv, AffExpr, MOI.EqualTo{Float64})
    d["num_constraints_total"] = num_constraints(det_equiv; count_variable_in_set_constraints = true)
    set_silent(det_equiv)
    optimize!(det_equiv)
    d["objective_value"] = objective_value(det_equiv)
    results["deterministic_equivalent"] = d
    say("deterministic_equivalent: ", d)
end

# ---------------------------------------------------------------- implement_a_par_model
struct NoiseTerm
    is_noise::Bool
    value::Float64
end

let d = Dict{String,Any}()
    alpha = [0.5, 0.3, 0.2]
    Ω = [-0.25, 0.0, 0.25]
    model = SDDP.LinearPolicyGraph(; stages = 12, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        P = length(alpha)
        @variable(sp, y[1:P], SDDP.State, initial_value = 1.0)
        @variable(sp, omega)
        @constraint(sp, y[1].out == sum(alpha[p] * y[p].in for p in 1:P) + omega)
        @constraint(sp, [p in 2:P], y[p].out == y[p-1].in)
        SDDP.parameterize(sp, Ω) do ω
            fix(omega, ω; force = true)
        end
    end
    sampling_scheme = SDDP.Historical(tuple.(1:3, [0.1, -0.1, 0.2]))
    simulations = SDDP.simulate(model, 1, [:y]; sampling_scheme)
    d["historical_y"] = [stage[:y][1].out for stage in simulations[1]]
    d["historical_y_all"] = [[stage[:y][p].out for p in 1:3] for stage in simulations[1]]
    Ω2 = NoiseTerm.(true, [-0.25, 0.0, 0.25])
    model = SDDP.LinearPolicyGraph(; stages = 12, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        P = length(alpha)
        @variable(sp, y[1:P], SDDP.State, initial_value = 1.0)
        @variable(sp, omega)
        @constraint(sp, y[1].out == sum(alpha[p] * y[p].in for p in 1:P) + omega)
        @constraint(sp, [p in 2:P], y[p].out == y[p-1].in)
        SDDP.parameterize(sp, Ω2) do ω
            if ω.is_noise
                fix(omega, ω.value; force = true)
            else
                unfix(omega)
                fix(y[1].out, ω.value; force = true)
            end
        end
    end
    # Named `omega_hist` (the guide uses `omega` at top level): inside a `let` block the
    # builder's `@variable(sp, omega)` would otherwise assign to this outer local.
    # The guide simulates in-sample first; that `fix`es `omega`, without which JuMP's `unfix`
    # in the historical simulation throws "Variable omega does not have fixed bounds".
    Random.seed!(1)
    SDDP.simulate(model, 1, [:y])
    omega_hist = NoiseTerm.(false, [1.0, 1.2, 1.4])
    sampling_scheme = SDDP.Historical(tuple.(1:3, omega_hist))
    simulations = SDDP.simulate(model, 1, [:y]; sampling_scheme)
    d["noiseterm_y"] = [stage[:y][1].out for stage in simulations[1]]
    d["noiseterm_y_all"] = [[stage[:y][p].out for p in 1:3] for stage in simulations[1]]
    # After the historical simulation the fixed `y[1].out` stays fixed (same in the port).
    d["noiseterm_y1_out_fixed_after"] = [is_fixed(model[t].subproblem[:y][1].out) for t in 1:3]
    results["implement_a_par_model"] = d
    say("implement_a_par_model: ", d)
end

# ------------------------------------------- simulate_using_a_different_sampling_scheme
let d = Dict{String,Any}()
    Ω = [(inflow = 0.0, fuel_multiplier = 1.5), (inflow = 50.0, fuel_multiplier = 1.0), (inflow = 100.0, fuel_multiplier = 0.75)]
    model = SDDP.MarkovianPolicyGraph(;
        transition_matrices = Array{Float64,2}[[1.0]', [0.75 0.25], [0.75 0.25; 0.25 0.75]],
        sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer,
    ) do subproblem, node
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
        SDDP.parameterize(subproblem, Ω, probability) do ω
            fix(inflow, ω.inflow)
            @stageobjective(subproblem, ω.fuel_multiplier * fuel_cost[t] * thermal_generation)
            return
        end
        return
    end
    d["deterministic_equivalent"] = det_equiv_value(model)
    Random.seed!(1)
    SDDP.train(model; iteration_limit = 10, print_level = 0, log_file = LOGFILE)
    sampling_scheme = SDDP.OutOfSampleMonteCarlo(model) do node
        stage, markov_state = node
        if stage == 0
            return [SDDP.Noise((1, 1), 1.0)]
        elseif stage == 3
            children = SDDP.Noise[]
            noise_terms = [SDDP.Noise((inflow = 75.0, fuel_multiplier = 1.2), 1.0)]
            return children, noise_terms
        else
            probability = markov_state == 1 ? [1 / 6, 1 / 3, 1 / 2] : [1 / 2, 1 / 3, 1 / 6]
            noise_terms = [SDDP.Noise(ω, p) for (ω, p) in zip(Ω, probability)]
            children = [SDDP.Noise((stage + 1, 1), 0.5), SDDP.Noise((stage + 1, 2), 0.5)]
            return children, noise_terms
        end
    end
    simulations = SDDP.simulate(model, 1; sampling_scheme = sampling_scheme)
    d["out_of_sample_noise_3"] = collect(simulations[1][3][:noise_term])
    sampling_scheme = SDDP.OutOfSampleMonteCarlo(model; use_insample_transition = true) do node
        stage, markov_state = node
        if stage == 3
            return [SDDP.Noise((inflow = 65.0, fuel_multiplier = 1.1), 1.0)]
        else
            probability = markov_state == 1 ? [1 / 6, 1 / 3, 1 / 2] : [1 / 2, 1 / 3, 1 / 6]
            return [SDDP.Noise(ω, p) for (ω, p) in zip(Ω, probability)]
        end
    end
    simulations = SDDP.simulate(model, 1; sampling_scheme = sampling_scheme)
    d["insample_transition_noise_3"] = collect(simulations[1][3][:noise_term])
    simulations = SDDP.simulate(model; sampling_scheme = SDDP.Historical([((1, 1), Ω[1]), ((2, 2), Ω[3]), ((3, 1), Ω[2])]))
    d["historical_nodes"] = [collect(stage[:node_index]) for stage in simulations[1]]
    d["historical_stage_objectives"] = [stage[:stage_objective] for stage in simulations[1]]
    scenarios = [
        [((1, 1), (inflow = 65.0, fuel_multiplier = 1.1)), ((2, 2), (inflow = 10.0, fuel_multiplier = 1.4)), ((3, 1), (inflow = 65.0, fuel_multiplier = 1.1))],
        [((1, 1), (inflow = 65.0, fuel_multiplier = 1.1)), ((2, 2), (inflow = 100.0, fuel_multiplier = 0.75)), ((3, 1), (inflow = 0.0, fuel_multiplier = 1.5))],
    ]
    d["historical_sequential_repr"] = sprint(show, SDDP.Historical(scenarios))
    d["historical_probabilistic_repr"] = sprint(show, SDDP.Historical(scenarios, [0.3, 0.7]))
    results["simulate_using_a_different_sampling_scheme"] = d
    say("simulate_using_a_different_sampling_scheme: ", d)
end

# ------------------------------------------------------------------ use_multithreading
let d = Dict{String,Any}()
    model = SDDP.LinearPolicyGraph(; stages = 12, lower_bound = 0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, x >= 0, SDDP.State, initial_value = 1)
        @stageobjective(sp, x.out)
    end
    d["deterministic_equivalent"] = det_equiv_value(model)
    Random.seed!(1)
    d["bound_10"] = train_silent(model; iteration_limit = 10, parallel_scheme = SDDP.Threaded())
    results["use_multithreading"] = d
end

open(joinpath(OUTG, "guides.json"), "w") do io
    JSON.print(io, results, 2)
end
say("ALL DONE")
