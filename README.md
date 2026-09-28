# sddp-py

A Python port of [SDDP.jl](https://github.com/odow/SDDP.jl) (MPL-2.0), a package for
solving multistage stochastic programs with Stochastic Dual Dynamic Programming.
The port is verified against SDDP.jl v1.15.0 numerically: see `PORTING_NOTES.md`,
`PROGRESS.md`, and `reference/` (the Julia oracle and the JSON ground truth).

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The default solver is HiGHS, provided through `pyoptinterface` + `highsbox`.

## Example (the hydro-thermal tutorial)

```python
import sddp

def builder(sp: sddp.Subproblem, t: int) -> None:
    volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
    thermal = sp.add_variable("thermal_generation", lb=0.0)
    hydro = sp.add_variable("hydro_generation", lb=0.0)
    spill = sp.add_variable("hydro_spill", lb=0.0)
    inflow = sp.add_variable("inflow")

    @sp.parameterize([0.0, 50.0, 100.0], [1 / 3, 1 / 3, 1 / 3])
    def _(omega):
        sp.fix(inflow, omega)

    sp.add_constraint(volume.out == volume.in_ - hydro - spill + inflow)
    sp.add_constraint(hydro + thermal == 150.0, name="demand_constraint")
    sp.set_stage_objective([50.0, 100.0, 150.0][t - 1] * thermal)

model = sddp.LinearPolicyGraph(builder, stages=3, sense="Min", lower_bound=0.0)
sddp.train(model, iteration_limit=10, seed=1234)
print(sddp.calculate_bound(model))
sims = sddp.simulate(model, 100, ["volume", "thermal_generation"], seed=42)
```

See `examples/hydro_thermal.py` and `tests/problems.py` for translations of the SDDP.jl
tutorials and examples (Markovian graphs, cyclic graphs, risk measures, objective states,
belief states, integer states).

## The sddp.dev documentation, in Python

Every page of https://sddp.dev/stable/ (SDDP.jl v1.15.0) has a Python equivalent under
`examples/`, and every equivalent is tested against numbers produced by SDDP.jl
(`reference/oracle/*.json`) or against exact quantities (deterministic equivalents, closed
forms, structural facts):

| sddp.dev section | Python file(s) | Tests |
|---|---|---|
| Tutorials: first steps, objective/Markov uncertainty, objective states | `examples/hydro_thermal.py`, `tests/problems.py` | `tests/test_parity.py`, `tests/test_tutorial_first_steps.py` |
| Tutorials: warnings, ARMA, decision-hazard, production planning, batteries, inventory | `examples/tutorial_modelling.py` | `tests/test_tutorial_modelling.py` |
| Tutorials: deterministic to stochastic, capacity expansion | `examples/tutorial_reservoir.py` | `tests/test_tutorial_reservoir.py` |
| Tutorial: the milk producer | `examples/tutorial_milk_producer.py` | `tests/test_tutorial_milk_producer.py` |
| Tutorial: two-stage newsvendor (Kelley, L-shaped, policy graph, risk sweep) | `examples/tutorial_newsvendor.py` | `tests/test_tutorial_newsvendor.py` |
| Tutorial: duality handlers | `examples/tutorial_duality_handlers.py` | `tests/test_tutorial_duality_handlers.py` |
| Tutorial: Markov decision processes (quadratic objective, maze) | `examples/tutorial_mdps.py` | `tests/test_tutorial_mdps.py` |
| Tutorial: plotting tools | `examples/tutorial_plotting.py` | `tests/test_tutorial_plotting.py` |
| Tutorial: alternative forward models (pglib_opf) | not ported: needs PowerModels + Ipopt (AC power flow); `AlternativeForwardPass` itself is covered by `examples/doc_examples.py` | `tests/test_examples.py`, `tests/test_tier4.py` |
| All 16 how-to guides | `examples/guides.py` | `tests/test_guides.py` |
| Explanation: introductory theory, risk aversion (vanilla SDDP from scratch) | `examples/theory_intro.py`, `examples/risk_explanation.py` | `tests/test_theory_intro.py`, `tests/test_risk_explanation.py` |
| Examples (29 pages) | `examples/asset_management_simple.py`, `examples/doc_examples.py`, `examples/vehicle_location.py`, `tests/problems.py` | `tests/test_examples.py`, `tests/test_parity.py`, `tests/test_tier4.py`, `tests/test_vehicle_location.py` |
| API reference | every entry mapped in `tests/test_api_reference.py` | same |

Each `examples/tutorial_*.py` and `examples/guides.py` has a `main()` that reproduces the
page's printed output.

## API mapping from SDDP.jl

| SDDP.jl | sddp-py |
|---|---|
| `@variable(sp, 0 <= x <= 1, SDDP.State, initial_value = 0)` | `x = sp.add_state("x", lb=0, ub=1, initial_value=0)`; use `x.in_`, `x.out` |
| `@variable(sp, u >= 0)` | `u = sp.add_variable("u", lb=0)` |
| `@constraint(sp, c, x.out == x.in + u)` | `c = sp.add_constraint(x.out == x.in_ + u, name="c")` |
| `@stageobjective(sp, 2u)` | `sp.set_stage_objective(2 * u)` |
| `SDDP.parameterize(sp, Ω, P) do ω ... end` | `sp.parameterize(fn, Ω, P)` or `@sp.parameterize(Ω, P)` |
| `fix`, `set_upper_bound`, `set_normalized_rhs`, `set_normalized_coefficient` | same names on `sp` |
| `SDDP.train(model; iteration_limit = 10)` | `sddp.train(model, iteration_limit=10, seed=1234)` |
| `SDDP.simulate(model, 100, [:x])` | `sddp.simulate(model, 100, ["x"], seed=42)` |
| `SDDP.calculate_bound`, `deterministic_equivalent`, `DecisionRule`, `evaluate` | same names |
| `SDDP.Expectation()`, `AVaR`, `EAVaR`, `WorstCase`, `Entropic`, `ModifiedChiSquared`, `Wasserstein` | same names (`EAVaR(lambda_=, beta=)`) |
| `SDDP.write_cuts_to_file` / `read_cuts_from_file` | same names; JSON format compatible with SDDP.jl |
| `SDDP.publication_plot` | `sddp.publication_plot(sims, fn)` (matplotlib) |
| `SDDP.SpaghettiPlot`, `add_spaghetti`, `plot` | `sddp.SpaghettiPlot(sims)`, `.add_spaghetti(fn, ...)`, `.plot(filename)` (same d3 HTML) |
| `SDDP.plot(model)` (graph structure) | `sddp.plot_graph(model, filename)` |
| `SDDP.ValueFunction(model; node)`, `SDDP.evaluate(V, point)` | `sddp.ValueFunction(model, node=...)`, `sddp.evaluate_value_function(V, point)` |
| `SDDP.plot(V; x = ...)` | `sddp.plot_value_function(V, x=[...])` (matplotlib) / `sddp.plot_value_function_html` |
| `SDDP.Inner.InnerPolicyGraph`, `inner_dp`, `dp_vertices_from_visited_states` | same names (`sddp.InnerPolicyGraph`, `sddp.inner_dp`, ...) |
| `SDDP.MSPFormat.read_from_file` | `sddp.read_msp_format(path)` |
| `SDDP.write_to_file` / `read_from_file` (StochOptFormat), `SDDP.evaluate(model, validation)` | `sddp.write_to_file`, `sddp.read_from_file`, `sddp.evaluate_validation_scenarios` |
| `SDDP.MarkovianGraph(simulator; budget)`, `SimulatorSamplingScheme` | `sddp.markovian_graph_from_simulator(simulator, budget)`, `sddp.SimulatorSamplingScheme` |
| `SDDP.train_biobjective` and helpers | same names |
| `SDDP.numerical_stability_report`, `write_log_to_csv` | same names |
| `SDDP.ImportanceSamplingForwardPass`, `AlternativeForwardPass`, `LoggingForwardPass` | same names |
| `SDDP.Threaded()`, `SDDP.Asynchronous()` | `sddp.Threaded(n)`, `sddp.Multiprocess(model_factory, n)` |
| `SDDP.binexpand`, `bincontract` | same names |
| `SDDP.parameterize(model[1], ω)`, `SDDP.write_subproblem_to_file(model[1], "f.lp")` | `sddp.parameterize(model[1], ω)`, `sddp.write_subproblem_to_file(model[1], "f.lp")` (HiGHS LP format) |
| `SDDP.sample_noise(D)` | `sddp.sample_noise(noise_terms, rng)` |
| `SDDP.ContinuousConicDuality(Ipopt.Optimizer)` (other optimizer for relaxed solves) | `sddp.ContinuousConicDuality(optimizer=sddp.HiGHS.with_options(presolve="off"))` (same backend, different options) |
| `@stageobjective(sp, x^2)` (convex QP via Ipopt) | `sp.set_stage_objective(x * x)` (HiGHS QP; objectives only, no quadratic constraints) |
| `lower_bound(x.out)` / `upper_bound(x.out)` inside the builder | `sp.lower_bound(x.out)` / `sp.upper_bound(x.out)` |
| `dashboard = true` | `sddp.train(model, dashboard=True)` (SSE server + `assets/dashboard.html`) |

Duals: `sp.dual(c)` and the cut duals are *sensitivities in the model's own sense*
(`d objective / d rhs`), for both minimisation and maximisation (see PORTING_NOTES §4).

## Tests

```bash
.venv/bin/python -m pytest tests -m "not slow"   # everything but the long cyclic models (~10 minutes)
.venv/bin/python -m pytest tests -m slow         # the five long-running page reproductions (~8 minutes)
.venv/bin/python -m pytest tests -m tier1        # Tier 1 parity only
.venv/bin/ruff check src tests && .venv/bin/mypy
```

The oracle can be regenerated (Julia 1.12, SDDP.jl, HiGHS) with:

```bash
cd reference && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. generate.jl all
```

## Reference

This project is a native Python port of the original SDDP.jl implementation by Oscar Dowson
and contributors:

- Dowson, O., & contributors. SDDP.jl: A Julia package for solving multistage stochastic
  optimization problems using stochastic dual dynamic programming.
  https://github.com/odow/SDDP.jl

> "A package for solving multistage stochastic optimization problems using stochastic dual
> dynamic programming."
