# Timing benchmark: hydro-thermal problem, SDDP.jl. Usage: julia --project=. bench_hydro_thermal.jl
using SDDP, HiGHS, Random
function build(; stages, noises)
    Ω = collect(range(0.0, 100.0; length = noises))
    return SDDP.LinearPolicyGraph(; stages = stages, sense = :Min, lower_bound = 0.0, optimizer = HiGHS.Optimizer) do sp, t
        @variable(sp, 0 <= volume <= 200, SDDP.State, initial_value = 200)
        @variables(sp, begin
            thermal_generation >= 0
            hydro_generation >= 0
            hydro_spill >= 0
        end)
        @variable(sp, inflow)
        SDDP.parameterize(sp, Ω) do ω
            fix(inflow, ω)
        end
        @constraints(sp, begin
            volume.out == volume.in - hydro_generation - hydro_spill + inflow
            hydro_generation + thermal_generation == 150
        end)
        @stageobjective(sp, (50 + 10 * t) * thermal_generation)
    end
end
for (stages, noises, iters, sims) in [(3, 3, 100, 1000), (12, 10, 100, 1000), (24, 20, 200, 1000)]
    Random.seed!(1)
    m = build(; stages, noises)
    SDDP.train(m; iteration_limit = 2, print_level = 0, run_numerical_stability_report = false, log_file = tempname())  # warm-up (compile)
    Random.seed!(1)
    m = build(; stages, noises)
    t0 = time()
    SDDP.train(m; iteration_limit = iters, print_level = 0, run_numerical_stability_report = false, log_file = tempname())
    t_train = time() - t0
    t0 = time()
    SDDP.simulate(m, sims, [:volume])
    t_sim = time() - t0
    println("stages=$stages noises=$noises iters=$iters: train $(round(t_train; digits=3))s, simulate($sims) $(round(t_sim; digits=3))s, bound $(SDDP.calculate_bound(m))")
end
