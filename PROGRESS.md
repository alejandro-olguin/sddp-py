# PROGRESS.md

Updated: 2026-09-28 (session 1, late)

## Current tier: Tier 3 implemented and verified; one Tier 2 problem (belief) under investigation

## Verified (tests green)
- Unit: solver dual-sign contract, risk measures vs hand values, cut formation vs a hand-solved
  two-stage problem (min and max), graph utilities, sampling schemes, stopping rules,
  cut JSON round trip.
- Tier 1 parity: hydro_thermal (det-equiv, bounds, exact cut sets over 5 and 20 deterministic
  iterations, Historical simulation values, simulation CI), fast_quickstart,
  fast_hydro_thermal, fast_production_management, farmers, stock_example. All 1e-6 relative.
- Tier 2 parity: markov_uncertainty (det-equiv, bound, simulation CI, deterministic run on
  unique quantities — see PORTING_NOTES §7.1), objective_uncertainty, infinite_trivial,
  no_strong_duality, infinite_hydro_thermal (single & multi cut, bound + simulation CI),
  asset_management_stagewise (EAVaR, single & multi cut), objective_states (converged value,
  §7.2), hydro_thermal multi-cut (bound + exact deterministic cut sets), hydro_thermal with
  AVaR (exact deterministic cut sets), risk-measure bounds for WorstCase, AVaR, EAVaR,
  Entropic, ModifiedChiSquared, Wasserstein (30 iterations, 1e-6).
- Tier 3 parity: air_conditioning with ContinuousConic / Lagrangian / Strengthened duality,
  stochastic_all_blacks with Conic (8.333) and Lagrangian (8.0 = det-equiv),
  sldp_example_one (<= 1.1675 and within 2e-3 of Julia's 50-iteration value).
- ruff + mypy clean. Plotting (publication_plot, spaghetti_plot via matplotlib) written,
  untested against Julia (no numeric oracle; publication_data uses the same quantile
  definition as Statistics.quantile).

## Failing / open
- `test_belief`: Julia bound after 100 iterations (seed 123) is 18.6909; Python after 100
  iterations gives 18.77–18.79 depending on seed. The graph is cyclic (0.9 continuation) and
  100 iterations is unlikely to be converged on either side. A convergence run (500 and 1500
  iterations, two seeds) is in progress to decide whether both sides agree at convergence;
  if so the oracle will be regenerated with more iterations via generate.jl (never by hand).

## Known gaps / deviations from SDDP.jl
- Parallel schemes: Serial only (no Threaded/Asynchronous).
- `numerical_stability_report` not implemented (printing only).
- No `SimulatorSamplingScheme`, `MarkovianGraph(simulator; budget)`, MSPFormat, Inner
  approximation, biobjective, alternative_forward, dashboard, value-function plots.
- Cut slopes at dual-degenerate points may differ from SDDP.jl (valid subgradients either way).

## Next step
1. Resolve belief (see above). 2. Full-suite run. 3. Final report.
