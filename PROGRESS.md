# PROGRESS.md

Updated: 2026-09-28 (session 2: performance work)

## Current tier: Tiers 1–3 implemented and verified (see FINAL_REPORT.md); performance steps 1–5 done

Performance (PORTING_NOTES §9): objective coefficient diffs, cached outgoing-state info,
cached cut expressions, `Threaded` scheme, Cython rejected; HiGHS tolerances tightened to 1e-9
after warm-started solves produced invalid cuts on the belief model; cold-restart recovery
ladder. Suite: 89 passed.

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
- `test_belief`: resolved by convergence analysis (PORTING_NOTES §7.4): Julia and Python both
  converge to 18.8168 at 1500 iterations (3.4e-7 relative); the oracle gained a 1500-iteration
  entry via generate.jl and the test compares that. The seeded test run (~20 min in Python)
  was the last thing in flight in session 1.

## Known gaps / deviations from SDDP.jl
- Parallel schemes: Serial only (no Threaded/Asynchronous).
- `numerical_stability_report` not implemented (printing only).
- No `SimulatorSamplingScheme`, `MarkovianGraph(simulator; budget)`, MSPFormat, Inner
  approximation, biobjective, alternative_forward, dashboard, value-function plots.
- Cut slopes at dual-degenerate points may differ from SDDP.jl (valid subgradients either way).

## Next step
None required. Possible follow-ups: Threaded parallel scheme, numerical stability report,
speeding up belief-state models (objective rebuild per solve).
