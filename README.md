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

Duals: `sp.dual(c)` and the cut duals are *sensitivities in the model's own sense*
(`d objective / d rhs`), for both minimisation and maximisation (see PORTING_NOTES §4).

## Tests

```bash
.venv/bin/python -m pytest tests            # everything (~3 minutes)
.venv/bin/python -m pytest tests -m tier1   # Tier 1 parity only
.venv/bin/ruff check src tests && .venv/bin/mypy
```

The oracle can be regenerated (Julia 1.12, SDDP.jl, HiGHS) with:

```bash
cd reference && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. generate.jl all
```
