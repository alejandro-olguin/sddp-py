# PROGRESS.md

Updated: 2026-09-28 (session 1)

## Current tier: Tier 1 complete and verified; Tier 2 under verification

## Verified (tests green)
- Solver layer dual-sign contract (`tests/test_solver.py`).
- Risk measures vs hand-computed values from SDDP.jl's own tests + docstrings
  (`tests/test_risk_measures.py`). Note: the SDDP.jl `Entropic` docstring's offset for
  γ=1.0 is stale; the value pinned is what SDDP.jl v1.15.0 actually computes (run in Julia).
- Cut formation vs a hand-solved two-stage problem, min and max (`tests/test_cuts.py`).
- Tier 1 parity (`tests/test_parity.py -m tier1`):
  - hydro_thermal: det-equiv, bound@10/40 (1e-6 rel), **full cut sets after 5 and 20
    deterministic iterations match Julia** (intercept, coefficient, state, 1e-6),
    per-iteration bound and forward value match, Historical simulation values match,
    1000-replication simulation mean within 99% CI.
  - fast_quickstart, fast_hydro_thermal (max), fast_production_management (2 states),
    farmers (max, coefficient parameterisation): det-equiv + converged bound (1e-6 rel);
    deterministic cut sets for the first two; simulation means in CI.
  - stock_example: bound after 60 iterations (1e-6 rel) and 1000-simulation mean in CI.

## In progress
- Tier 2 parity run (`pytest -m tier2`), see scratch log. Implemented: Markovian and
  general/cyclic graphs, all risk measures, multi-cut, all sampling schemes, all stopping
  rules, objective states, belief states. Results pending.

## Known gaps / deviations
- Parallel schemes: Serial only.
- `numerical_stability_report` is accepted but not implemented (printing only in Julia).
- `Statistical` stopping rule uses `simulate` with a fresh unseeded RNG (Julia uses the
  global RNG) — non-deterministic by design, as in Julia.
- Tier 3 (LagrangianDuality, StrengthenedConicDuality, cut serialisation is done, plotting)
  not started.

## Next step
1. Read Tier 2 results; debug any mismatch by comparing intermediate values.
2. ruff/mypy clean-up.
3. Tier 3.
