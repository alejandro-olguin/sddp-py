# PORTING_NOTES.md — SDDP.jl → sddp-py

Source: SDDP.jl v1.15.0, commit a969cbadbdd46871046c4b7dbb7025bbab156aec (2026-09-11),
MPL-2.0. The port is a derivative work; `LICENSE` is the MPL-2.0 and every ported
module carries a file-level notice.

This file is my memory across context resets. Keep it current.

## 1. Module map (Julia file → Python module)

| Julia                                   | Python (`src/sddp/`)         | Tier | Notes |
|-----------------------------------------|------------------------------|------|-------|
| `user_interface.jl` Graph/LinearGraph/MarkovianGraph/UnicyclicGraph | `graph.py` | 1/2 | pure data, node labels are user objects (hashable) |
| `user_interface.jl` Noise/State/Node/PolicyGraph/parameterize/set_stage_objective | `policy_graph.py` | 1 | builder receives `(sp: Subproblem, node)` |
| `user_interface.jl` add_objective_state / objective_state | `objective_state.py` | 2 | |
| `user_interface.jl` belief partition / construct_belief_update | `belief.py` | 2 | |
| `JuMP.jl` (State variable extension) + JuMP itself | `subproblem.py` (`Subproblem` wrapping a solver `Model`) | 1 | replaces `@variable(sp, x, SDDP.State, initial_value=..)` with `sp.add_state("x", initial_value=..., lb, ub)` |
| `algorithm.jl` Options/solve_subproblem/backward_pass/calculate_bound/iteration/train/simulate/evaluate | `algorithm.py` | 1 | |
| `plugins/headers.jl`                    | `plugins/base.py`            | 1 | abstract base classes |
| `plugins/bellman_functions.jl`          | `plugins/bellman_functions.py` | 1/2 | single cut Tier 1, multi-cut Tier 2, cut selection Tier 1 |
| `plugins/risk_measures.jl`              | `plugins/risk_measures.py`   | 2 | Expectation Tier 1 |
| `plugins/sampling_schemes.jl`           | `plugins/sampling_schemes.py`| 1/2 | InSampleMonteCarlo Tier 1 |
| `plugins/backward_sampling_schemes.jl`  | `plugins/backward_sampling_schemes.py` | 1 | CompleteSampler |
| `plugins/stopping_rules.jl`             | `plugins/stopping_rules.py`  | 1/2 | IterationLimit/TimeLimit Tier 1 |
| `plugins/forward_passes.jl`             | `plugins/forward_passes.py`  | 1 | DefaultForwardPass Tier 1 |
| `plugins/duality_handlers.jl`           | `plugins/duality_handlers.py`| 1/3 | ContinuousConicDuality Tier 1 |
| `plugins/parallel_schemes.jl`           | `plugins/parallel_schemes.py`| 1 | Serial only (master_loop) |
| `plugins/local_improvement_search.jl`   | `plugins/local_improvement_search.py` | 3 | |
| `deterministic_equivalent.jl`           | `deterministic_equivalent.py`| 1 | |
| `cyclic.jl`                             | `graph.py::is_cyclic`        | 2 | Tarjan |
| `print.jl`                              | `print.py`                   | 1 | log table; numerical stability report optional |
| `visualization/value_functions.jl` | `value_function.py` | 4 | |
| `Inner.jl` | `inner.py` | 4 | |
| `MSPFormat.jl` | `msp_format.py` | 4 | |
| `Experimental.jl` (StochOptFormat) | `stochoptformat.py` | 4 | MOF subset |
| `modeling_aids.jl` + `SimulatorSamplingScheme` | `modeling_aids.py` | 4 | |
| `biobjective.jl` | `biobjective.py` | 4 | |
| `alternative_forward.jl`, `ImportanceSamplingForwardPass`, `LoggingForwardPass` | `plugins/forward_passes.py` | 4 | |
| `binary_expansion.jl` | `binary_expansion.py` | 4 | |
| `print.jl` numerical stability report, `write_log_to_csv` | `print.py` | 4 | |
| `parallel_schemes.jl` Threaded / Asynchronous | `plugins/parallel_schemes.py` (`Threaded`, `Multiprocess`) | 2/4 | |
| `visualization/*` (spaghetti, graph, dashboard) | `visualization.py` + `assets/` | 4 | |
| solver abstraction (JuMP/MOI)           | `solver/` (`base.py`, `pyoptinterface_backend.py`) | 1 | see §4 |

## 2. Key algorithmic invariants (what must hold for parity)

### 2.1 Subproblem structure
Each node `i` owns one persistent LP/MIP:

    min/max  stage_objective(x_in, x_out, u, ω) + <y, μ_obj> + <b, μ_belief> + θ
    s.t.     user constraints (possibly modified by parameterize(ω))
             x_in == x̄   (implemented by fixing the variable bounds, JuMP.fix)
             cuts on θ

* `θ` (`global_theta.theta`) is a single variable bounded by `[lower_bound, +∞)` for
  minimisation, `(-∞, upper_bound]` for maximisation. **Leaf nodes (no children) get
  θ fixed to 0** (`lower_bound = upper_bound = 0.0`).
* `parameterize(node, ω)` = call the user's `modify(ω)` then `set_objective(node)`.
  The objective is (re)set only if `stage_objective_set == False`; `set_stage_objective`
  resets that flag, so calling it from inside `parameterize` is how objective noise works.
* Order in `solve_subproblem`: `set_incoming_state` → `parameterize` → pre-hook →
  optimize → check primal status → `get_outgoing_state` → `stage_objective_value` →
  `get_dual_solution(duality_handler)` → post-hook.
* `stage_objective_value = objective_value - value(θ)` (unless objective/belief states,
  where it is `value(stage_objective)` directly).
* `get_outgoing_state` projects `x_out` onto its bounds and rounds integer/binary states.

### 2.2 Cuts (single cut, `_add_average_cut` + `_add_cut`)
Backward pass at node `i` with sampled outgoing state `x̂`. For every child `j` and every
noise `ω` of `j`: solve child with `x_in = x̂` and get `(V_jω, λ_jω)` where
`λ_jω[k] = dual_sign * dual(FixRef(x_in[k]))`, `dual_sign = +1` (min) / `-1` (max).
Nominal probability `p_jω = Φ(i→j) * P_j(ω)` (these do **not** sum to 1 if arcs leave
the graph, e.g. discount factor 0.9 → sum = 0.9; that is the discounting mechanism).
Risk-adjust: `(q, offset) = adjust_probability(measure, p, supports, V, is_min)`.
Then

    θ̄ = offset + Σ q_jω V_jω          π = Σ q_jω λ_jω           (dict over state names)
    intercept = θ̄ − π·x̂
    min: θ + <y,μ_obj> + <b,μ_belief> − π·x_out ≥ intercept
    max: θ + <y,μ_obj> + <b,μ_belief> − π·x_out ≤ intercept

i.e. `θ ≥ θ̄ + π·(x_out − x̂)`. Note `V_jω` is the **full** child objective (stage cost +
child's θ), so cuts nest.
The dual `λ` must equal ∂V_j/∂x_in in the *original* sense (d objective / d fixed value).
That is the whole reason for `dual_sign`. Verified for the Python backend in §4.

### 2.3 Cut selection (`_cut_selection_update`, "Level 1" dominance)
Every cut stores `non_dominated_count`; every sampled state stores its dominating cut and
best height. On adding a cut: evaluate it at all previous sampled states (same obj_y /
belief_y only), update dominance; evaluate all *deactivated* old cuts at the new state and
re-add if dominating; cuts with `non_dominated_count < 1` are queued; when queue length
≥ `deletion_minimum` (default 1 → immediately) they are deleted from the solver.
Deleted cuts remain in `V.cuts` (with `constraint_ref = None`) so they can be re-added.
**Important for parity**: with `cut_deletion_minimum = 1` (default) the LP contains only
non-dominated cuts; this changes the LP but not the lower bound value in exact arithmetic
only if the deleted cuts truly are dominated at every sampled point — the bound is still
valid. To compare cut sets with Julia exactly I compare the *full* `cuts` list (intercept,
coefficients, state) as written by `write_cuts_to_file`, not just active ones.

### 2.4 Multi-cut (Tier 2)
One local θ_k per (child, noise) index k (created lazily, N = number of backward items,
bounds copied from global θ). Each local θ_k gets a plain cut `θ_k ≥ V_k + λ_k(x − x̂)`.
Then a "risk-set" constraint `θ ≥ Σ q_k θ_k − (1 − Σ q_k) μᵀy + offset` is added once per
distinct probability vector `q` (`risk_set_cuts::Set{Vector{Float64}}`) — unless there
are objective/belief states, in which case always.

### 2.5 Lower bound (`calculate_bound`)
For each root child `j` (skip prob≈0) and each noise `ω` of `j`: solve `j` at the root
state with `duality_handler = nothing`, collect `V`, `p = Φ(root→j) P_j(ω)`. Apply the
`root_node_risk_measure` (default Expectation) and return `Σ q V + offset`. The bound is
computed **after** the backward pass, every iteration, and logged. Objective/belief states
are updated from their initial values before each solve.

### 2.6 Forward pass (`DefaultForwardPass`)
`sample_scenario` (InSampleMonteCarlo): start at a root child sampled by probability; at
each node sample ω; stop if no children, or `terminate_on_cycle` and node revisited, or
`max_depth` reached, or with probability `1 − Σ child prob` ("dummy leaf" — this is how
discounted infinite-horizon problems terminate). Solve along the path with
`duality_handler = nothing`; record outgoing states, `cumulative_value += stage_objective`.
For cyclic graphs `options.starting_states[node]` accumulates incoming states (when farther
than `cycle_discretization_delta`) and one is `splice!`d out at random as the incoming
state — only if the list is non-empty. When the pass terminated due to a cycle the
incoming state of the final node is pushed to that node's starting states.

### 2.7 Backward pass
Walk the trajectory from the last index to the first. For index `t` with node `i` and
sampled outgoing state `x̂_t`: skip if `i` has no children. `solve_all_children` iterates
children in `node.children` order, noises via the backward sampler (CompleteSampler =
all `noise_terms` in order), caches `(child, noise)` solutions within one item set
(matters for graphs where the same child appears twice — not in normal graphs), and
records `duals, supports (Noise), nodes, probability, objectives, belief`. Then
`refine_bellman_function` adds the cut to node `i`. With `refine_at_similar_nodes`
(default true) the same child solutions are re-used to add cuts to every other node with
the identical child set (`similar_children`), using that node's own arc probabilities
`Φ(other→child) × P(ω)`. This is what makes Markovian graphs cheap: all Markov states of
stage t share children.
`prepare_backward_pass(child)` for ContinuousConicDuality relaxes integrality on the child
before solving and restores it after (per child, per backward item).
Belief-state variant: solve children of *every* node with non-zero belief, weight items by
belief, and add the cut to *all* nodes in the partition.

### 2.8 Risk measures (`adjust_probability(measure, q_out, p, supports, V, is_min)`)
All write into `q_out` and return an `offset` (0 except Entropic). Summary:
* Expectation: `q = p`.
* WorstCase: all mass on the argmax V (min) / argmin V (max) among `p > 0`; ties → first.
* AVaR(β): β≈0 → WorstCase, β≈1 → Expectation; else sort V descending (min) / ascending
  (max) (Julia `sortperm(rev=is_min)` — stable sort, ties keep original order), fill
  `q_i = min(p_i, β − collected)/β` until `collected ≥ β`.
* ConvexCombination: weighted sum of member `q`s and offsets. `EAVaR(λ,β) = λE + (1−λ)AVaR(β)`.
* ModifiedChiSquared(r): if uncorrected std(V) < minimum_std → Expectation; uniform p →
  Algorithm 2 (sorted worst-first, closed-form, drop worst until nonneg); else Algorithm 1.
* Wasserstein(norm, solver, α): solve a small LP (transport plan). Needs a solver.
* Entropic(γ): `q ∝ p·exp(γ V)` (γ negated for max), `offset = −(Σ q log(q/p))/γ`.
  Julia uses BigFloat to avoid overflow; Python: subtract max before exp (same result).
Node-wise risk measures: `risk_measure` may be a single measure, a dict keyed by node, or a
callable `node → measure` (`to_nodal_form`).

### 2.9 Objective states (Tier 2)
`add_objective_state(update, sp; initial_value, lipschitz, lower_bound, upper_bound)`
adds N variables `μ ∈ [−L, L]` to the subproblem. `objective_state(sp)` returns the
current value `y` (updated by `update(y_prev, ω)` before each solve — forward, backward,
and bound). `set_objective` adds `Σ y_i μ_i` to the objective and marks the objective for
re-set every solve. Cuts carry `obj_y` and add `Σ y_i μ_i` to the cut LHS. Initial bound
constraints `<y,μ>+θ ≥ lb` at all 2^N corners of the y-box (if N<5).

### 2.10 Belief states (Tier 2)
Ambiguity sets partition nodes; each node in a partition gets `μ[node]` variables in
`[−L, L]`; belief `b` is updated by Bayes' rule via `construct_belief_update` using Φ and
the noise pmf of each node. Cuts carry `belief_y`. See §2.7 for the backward pass.

### 2.11 Stopping rules / train loop
`train` appends `IterationLimit`/`TimeLimit` from kwargs; if none, default is
`SimulationStoppingRule()`. `convergence_test` runs the rules in order after each
iteration; first that fires gives the status symbol. Log entry per iteration:
`(iteration, bound, simulation_value=cumulative forward value, time, pid, total_solves,
duality_key, numerical_issue)`.

### 2.12 Deterministic equivalent
Scenario tree: recursively (root children → node × noise → children …), a tree node per
`(node, noise)`, probability = product along the path. For each tree node: parameterize
the subproblem with that noise, copy *all* variables (with bounds/integrality) and *all*
constraints into one big model, add `probability × stage_objective` to the objective, link
`parent.x_out == child.x_in`, and fix root children's `x_in` to the initial state. Refuses
cyclic graphs, objective/belief states, and trained models. Because copying relies on
reading the subproblem back, my `Subproblem` layer must keep its own record of variables,
bounds, and constraint rows (it does — see §4).

## 3. Julia idioms → Python translations

| Julia idiom | Python translation |
|---|---|
| `@variable(sp, 0 <= x <= 200, SDDP.State, initial_value = 200)` | `x = sp.add_state("x", lb=0, ub=200, initial_value=200)` → `State(in, out)` with `x.in_`/`x.out` (`in` is a keyword: I use `x.incoming`/`x.outgoing` plus short aliases `x.in_`, `x.out`) |
| `@variable(sp, x[1:N] >= 0, SDDP.State, ...)` | loop: `sp.add_state(f"x[{i}]", ...)`; state names are strings, exactly as JuMP would name them (`x[1]`) so cut JSON matches |
| `@variables(sp, begin y >= 0 ... end)` | `y = sp.add_variable("y", lb=0)` |
| `@constraint(sp, name, expr == rhs)` | `sp.add_constraint(expr == rhs, name="name")`; expressions come from the solver layer's operator overloading |
| `@stageobjective(sp, expr)` | `sp.set_stage_objective(expr)` (macro removed; MutableArithmetics rewriting is irrelevant) |
| `SDDP.parameterize(sp, Ω, P) do ω ... end` | `sp.parameterize(Ω, P)(fn)` or `sp.parameterize(fn, Ω, P)` — I support `@sp.parameterize(Ω, P)` decorator and the positional form |
| `fix(v, ω)` / `set_upper_bound` / `set_normalized_rhs` / `set_normalized_coefficient` | `sp.fix(v, ω)`, `sp.set_upper_bound(v, ω)`, `sp.set_normalized_rhs(c, ω)`, `sp.set_normalized_coefficient(c, v, a)` |
| multiple dispatch on plugin types (`adjust_probability(::AVaR, ...)`) | ABCs with methods: `RiskMeasure.adjust_probability(self, ...)`, `SamplingScheme.sample_scenario(self, model)`, `StoppingRule.convergence_test(self, model, log)`, `ForwardPass.forward_pass(...)`, `DualityHandler.get_dual_solution(...)` |
| `to_nodal_form(model, x)` | `_to_nodal_form(model, x)`: scalar / dict / callable |
| `Dict{Symbol,Float64}` state dicts, insertion-ordered in Julia? **No** — Julia `Dict` iteration order is hash-based | Python dicts are insertion-ordered. This only affects: (a) tie-breaking nowhere in the algorithm (state dict order does not affect any numeric result), (b) `_solve_primal_problem` vectors (Tier 3). Safe. |
| `node.children` is a `Vector` in `graph.nodes[node]` insertion order; `graph.nodes` is a `Dict` (hash order) | Children order is user insertion order — preserved. Node iteration order in `PolicyGraph(...)` construction is hash order in Julia; irrelevant numerically. |
| 1-based indexing, `sortperm(rev=true)` | 0-based; `sorted(range(n), key=..., reverse=True)` — **Python's `sorted(reverse=True)` is stable and preserves original order among equals, same as Julia's `sortperm(rev=true)`** (both are stable; Julia's default for `sortperm` is stable). Verified in unit tests with ties. |
| `rand()` (Xoshiro) for sampling | `random.Random` instance on the sampling scheme / model (`seed=`). Trajectories cannot match Julia's RNG; parity is on converged values and statistics. Deterministic comparisons use `Historical` sampling with an explicit scenario list on both sides. |
| `isapprox(a, b)` default `rtol = sqrt(eps)` ≈ 1.49e-8, `atol=0` | `math.isclose(a, b, rel_tol=1.4901161193847656e-08, abs_tol=0.0)`; helper `_isapprox` in `utils.py` |
| `≈ 0.0` (i.e. `isapprox(x, 0.0)`) with default atol=0 is only true for exactly 0.0! (e.g. AVaR `β ≈ 0.0`) | replicate: `x == 0.0` — `math.isclose(x, 0.0)` also gives that. Note `isapprox(child.probability, 0.0; atol=1e-6)` is the explicit-atol version. |
| JuMP.fix sets `EqualTo` bound on variable; dual via `FixRef` | fix = set lb=ub; dual = reduced cost (see §4 sign check) |
| MOI dual sign: JuMP duals for MAX problems are negated relative to sensitivities | keep `dual_sign` trick but for our backend the reduced cost is already ∂obj/∂bound in both senses — **measured**, see §4 |
| `JuMP.objective_sense` MIN_SENSE/MAX_SENSE | `Sense.MIN / Sense.MAX` enum; user passes `sense="Min"|"Max"` |
| `Noise(term, probability)` struct | `@dataclass(frozen=True) Noise(term, probability)`; noise terms may be any Python object (tuples, dicts, floats). For caching `(child, noise.term)` keys must be hashable → I key the cache by `(child, index_of_noise)` instead, which is equivalent for CompleteSampler and avoids requiring hashable noise terms. |
| `nothing` noise for deterministic nodes | `None` |
| `@warn maxlog=1` | `warnings.warn(..., stacklevel=2)` once-guard |
| Threads/locks | none (Serial only) |
| `TimerOutputs` | omitted |
| `log_file` "SDDP.log" appended | `log_file=None` by default in Python (opt-in), stdout printing via `print_level` |

## 4. Solver layer decision

Requirements: persistent model, incremental rows (cuts), delete rows (cut selection),
change variable bounds (fix x_in), change RHS / coefficients (parameterize), fast
re-solve with warm start, primal values, objective, reduced costs / row duals with known
sign, MIP support with integrality relaxation (Tier 1 uses ContinuousConicDuality which
relaxes integrality on the backward pass).

Candidates: `pyoptinterface` (0.6.1, HiGHS backend) and raw `highspy` (1.15.1).
See `reference/bench_solver_layer.py` and the results recorded below.

(Results are appended below after the benchmark runs.)

## 5. Reference problems (oracle)

`reference/generate.jl` produces `reference/*.json`. Never edit those JSON files by hand.

Tier 1 (all with HiGHS, `Random.seed!(seed)`):
1. `hydro_thermal_first_steps` — 3-stage tutorial (`first_steps.jl`). Bound after 10 and
   40 iterations; deterministic equivalent; simulate 500 with recorded variables; **full
   cut set** after 3 iterations with a fixed `Historical` scenario list.
2. `fast_quickstart` — 2 stage, bound = −2 (exact).
3. `fast_hydro_thermal` — 2 stage max problem, bound = −10 (exact).
4. `fast_production_management` — 3 stage, 2 states, bound ≈ −23.96.
5. `stock_example` — 5 stages, 10 noises, bound ≈ −1.471, 1000 simulations.
6. `the_farmers_problem` — 2 stage max, bound 108 390.
Tier 2: `markov_uncertainty`, `infinite_horizon_trivial`, `infinite_horizon_hydro_thermal`,
`no_strong_duality`, `asset_management_stagewise` (EAVaR, single & multi cut),
`objective_uncertainty`, `objective_states`, `belief`.
Tier 3: `air_conditioning` (Lagrangian), `stochastic_all_blacks`, `sldp_example_one`.

## 6. Decisions log
* 2026-09-28: notes created after reading `user_interface.jl`, `algorithm.jl`, all
  `plugins/*.jl`, `deterministic_equivalent.jl`, `JuMP.jl`, `cyclic.jl`, `print.jl`.

### 4.1 Benchmark results (2026-09-28, macOS arm64, `reference/bench_solver_layer.py 1000`)

| layer | 1000 × (fix state, set RHS, add cut, solve, read obj/primal/dual) | ms/solve |
|---|---|---|
| pyoptinterface 0.6.1 + highsbox | 0.157 s | 0.157 |
| highspy 1.15.1 raw | 0.172 s | 0.172 |

Both are dominated by HiGHS itself; the Python overhead is negligible either way.

### 4.2 Dual sign check (tiny LPs, hand-computed)
`min 2x + y, x fixed at 3, y ≥ 1` → obj 7, ∂obj/∂x = +2.
`max −2x − y, x fixed at 3, y ≥ 1` → obj −7, ∂obj/∂x = −2.
Row `x ≥ 1`: min x → +1; max −x → −1.

| | min: reduced cost of fixed var | max: reduced cost of fixed var | min: row dual | max: row dual |
|---|---|---|---|---|
| pyoptinterface/HiGHS | +2 | −2 | +1 | −1 |
| highspy raw | +2 | −2 | — | — |

**Conclusion:** HiGHS (through either wrapper) reports duals as *sensitivities of the
objective in its stated sense* — i.e. already `∂obj/∂x_in` for both min and max. JuMP/MOI
instead flips the sign for MAX problems, which is why SDDP.jl multiplies by
`dual_sign = −1` for MAX_SENSE. In the Python port the solver interface contract is
"`get_fixed_dual(var)` returns ∂obj/∂(fixed value) in the model's own sense", the HiGHS
backend returns the reduced cost unchanged, and no `dual_sign` is applied in
`ContinuousConicDuality`. `tests/test_solver.py` pins this with the four LPs above so a
future backend (Gurobi etc.) must satisfy the same contract.

### 4.3 Decision
**pyoptinterface (HiGHS backend via highsbox) is the modeling/solver layer.**
Reasons: MOI-like design (variables/constraints as handles, attributes), operator-overloaded
affine expressions (gives users a JuMP-like builder), incremental add/delete of rows,
`set_normalized_rhs`/`set_normalized_coefficient`, variable domain changes (integrality
relaxation), and the same code path works for Gurobi/COPT/Mosek later. Raw highspy would
require me to write an expression layer and gains nothing measurable.
The solver is hidden behind `sddp.solver.Model` (a thin wrapper) so another backend can be
added by implementing ~20 methods.

## 7. Parity findings and justified deviations

### 7.1 Cut slopes at dual-degenerate points (markov_uncertainty deterministic run)
With fixed `Historical` scenarios, iterations 1–3 of `markov_uncertainty` match SDDP.jl
exactly (bounds 3753.3698, 7975.336, 7975.336; forward values identical). At iteration 3
the sampled outgoing volume of stage 2 is exactly 150, where the stage-3 subproblem
(`hydro + thermal == 150`, `hydro <= volume`) is **dual degenerate**: ∂V/∂volume can be any
value in `[-fuel_cost, 0]`. SDDP.jl's HiGHS returned λ = 0 for every noise; the Python
HiGHS (via pyoptinterface, HiGHS 1.13.1) returned a mix, giving an average slope of −56.25.
Both cuts have the same height (0.0) at the sampled state, and both are valid. From there the
trajectories differ (Julia reaches 8072.9167 at iteration 4, Python at a later Historical
pass; both converge to the deterministic-equivalent value 8072.916667).
Consequence: for this problem `tests/test_parity.py` compares forward values, cut heights,
sampled states and the converged bound, not the slopes. The exact cut-set comparison
(slopes included) is retained for `hydro_thermal` (5 and 20 iterations, single and multi
cut, AVaR), `fast_quickstart` and `fast_hydro_thermal`, which are non-degenerate at their
sampled points and match to 1e-6.
Evidence script: see the per-node cut listing produced during the session (node (2,1),
cut 2: py `(0.0, −56.25, 150.0)` vs jl `(0.0, 0.0, 150.0)`).

### 7.2 Iteration counts vs. convergence (objective_states)
The oracle trains `objective_states` for 60 iterations with `Random.seed!(1234)` and
reaches 5092.592593. With Python's RNG, 60 iterations give 5087.57 (seed 1234) or 5065.47
(seed 7), while 200 and 600 iterations give 5092.592592592593 for both seeds. Since
Monte Carlo trajectories cannot match Julia's RNG, the test trains for 200 iterations and
compares the converged value at 1e-6 relative. Nothing in the oracle was changed.

### 7.3 Entropic docstring
SDDP.jl's `Entropic` docstring lists the γ = 1.0 offset as −0.1203806; running
`SDDP.adjust_probability` in Julia gives −0.6671608119259349, which is also what the
hand calculation and the Python port give. The unit test pins the computed value.

## 8. Performance vs SDDP.jl (hydro-thermal family, `reference/bench_hydro_thermal.{jl,py}`)

macOS arm64, HiGHS on both sides, serial, single thread, Julia timings exclude compilation
(a warm-up train precedes the timed run). Same problem, same iteration counts; RNG differs.

| stages | noises | iterations | Julia train | Python train | Julia simulate(1000) | Python simulate(1000) | bound (Julia / Python) |
|---|---|---|---|---|---|---|---|
| 3 | 3 | 100 | 0.202 s | 0.069 s | 0.213 s | 0.087 s | 7277.7778 / 7277.7778 |
| 12 | 10 | 100 | 0.907 s | 0.545 s | 0.684 s | 0.342 s | 106299.7222 / 106299.7222 |
| 24 | 20 | 200 | 5.29 s | 3.24 s | 1.152 s | 0.683 s | 364357.92 / 364358.21 (not converged at 200 its) |

The Python port is 1.6–2.9× faster on these sizes. Both are dominated by HiGHS; the
difference is wrapper overhead (JuMP's caching-optimizer bridge layer vs pyoptinterface's
direct calls). This is not a claim about larger models.

### 7.4 belief: 100 iterations is not converged
The oracle originally trained `belief` (cyclic, 0.9 continuation, two ambiguity sets) for 100
iterations with `Random.seed!(123)` → 18.6909. Running SDDP.jl longer (script
`belief_julia.jl`, session scratch): seed 123 → 18.816721 (500), 18.816820 (1500); seed 7 →
18.773981 (100), 18.816752 (500), 18.816827 (1500). Python after 100 iterations gives
18.77–18.79 depending on seed. `generate.jl` was extended with a 1500-iteration run
(`train_1500`, `simulation_1500`); the 100-iteration entry regenerated byte-identically. The
test compares the 1500-iteration bounds at 1e-6 relative.
Python (convergence script, `tests/problems.build_belief`): seed 1234 → 18.785463 (100),
18.816762 (500), 18.816814 (1500); seed 7 → 18.797593 (100), 18.816752 (500), 18.816814
(1500). Julia vs Python at 1500 iterations: 3.4e-7 relative. Note the Python run of 1500
iterations took ~1200 s vs ~410 s in Julia: the belief backward pass solves every node of a
partition for every noise, and the Python overhead per solve (belief updater dict work,
objective rebuild every solve because of the μᵀb term) is larger than in the LP-dominated
hydro-thermal benchmark. This is the one measured case where the port is slower than SDDP.jl.

## 9. Performance work (after the port was verified)

Profiles (`cProfile`, 100 iterations): hydro-thermal 24×20 spends 35% inside HiGHS and the
rest in ~350k small pyoptinterface calls; the belief model spends 85% inside HiGHS.
pyoptinterface releases the GIL in `optimize` (2 threads: 0.089 s → 0.048 s), so a
threaded backward pass is viable. Cython is deliberately not the first lever: it cannot make
cross-boundary solver calls cheaper (see the plan in the session transcript / FINAL_REPORT).

### 9.1 Step 1: persistent objective, coefficient diffs (`Model.set_objective`)
A full `set_objective` in the HiGHS backend costs 1.49 ms vs 0.065 ms for a re-solve on the
belief node (it discards the warm start). `Model.set_objective` now diffs against the loaded
objective and calls `set_objective_coefficient` only for changed entries (full reset if the
sense or constant changes). Measured (200 iterations, seed 1):

| model | before | after |
|---|---|---|
| belief | 21.39 s | 5.63 s (3.8×) |
| objective_states | 1.33 s | 0.34 s (3.9×) |
| objective_uncertainty | 0.49 s | 0.23 s (2.1×) |
| hydro-thermal 24×20 | 3.27 s | 3.07 s |

Correctness: 85/86 parity tests unchanged (all exact cut-set tests pass). Per-solve trace of
a deterministic belief run (102 solves, old vs new code): identical objective values, states
and duals until solve 39, where the incoming state differs by 1.6e-15 (warm vs cold start
roundoff) with identical outputs; the runs then diverge through a degenerate LP. The
`test_belief` bound at 1500 iterations moved from 18.816814 to 18.816973 (Julia: 18.816820,
still rising at 1500); long runs on both sides are used to settle where the bound converges.

### 9.2 Step 2: fewer solver queries per solve
`get_outgoing_state` read 7 attributes per state per solve; the bounds and domain of the
outgoing state variables are now cached per node and invalidated by a version counter that
every bound/domain change to a *state* variable bumps (`Model.watch`). Same semantics as
reading after every solve. Hydro-thermal 24×20, 100 iterations: pyoptinterface calls
1,399,878 → 492,894 (−65%), Python function calls 6.78M → 4.66M. Suite: identical results
(85 passed, the belief value bit-identical to step 1). Wall-clock re-measured in §9.4 once the
machine was idle.

### 9.3 Step 3: cut selection — vectorisation rejected, expression caching adopted
Profiling at 400 iterations (hydro-thermal 12×10, 4,400 cuts) showed cut selection at 60% of
the run, but not in height evaluation: SDDP.jl's Level-1 selection with
`cut_deletion_minimum = 1` deletes and re-adds rows constantly (357,413 row adds and
357,304 deletes for 4,400 cuts, because ties count as domination and repeated backward passes
produce identical cuts). A numpy version of the height loops (implemented, verified
bit-identical on three 100-iteration traces, then discarded) reduced `_heights` to 0.02 s and
changed nothing else. Adopted instead: each cut caches its solver expression, and rows are
added through a fast path (`Model.add_row`) with no expression normalisation or name
registration. Cut-selection time at 400 iterations: 4.87 s → 3.32 s; whole run 7.33 s →
5.77 s (−21%). Verified bit-identical (bounds, cuts, counts, active sets) over 100 iterations
on hydro-thermal 24×20, belief, and asset_management multi-cut; suite 85 passed with the
belief value unchanged.

### 9.4 Numerical recovery (found by the long belief run)
A 5000-iteration belief run crashed at ~2000 iterations: HiGHS returned no solution for a
node with 2,489 rows, and a fresh HiGHS instance solved the dumped LP immediately. SDDP.jl's
default recovery is `MOI.Utilities.reset_optimizer` (a cold start); pyoptinterface has no
equivalent, so `Model.reset_optimizer` now calls `Highs_clearSolver` on the raw handle
through ctypes (HiGHS only), and the default recovery callback is a ladder: cold restart →
presolve off → interior point (cold, `output_flag=False`).

### 9.5 Step 4: `Threaded` parallel scheme
Ported SDDP.jl's `Threaded`: whole iterations run concurrently, one `RLock` per node (held
while a node's subproblem is used in the forward pass, backward pass, bound, and simulation),
an options lock for the log/callbacks/convergence test, and a model lock for counters.
`Historical` and `PSRSamplingScheme` counters are locked. Objective states force `Serial`
(as in SDDP.jl). Results are valid but not reproducible run to run. Measured (200
iterations, one core busy with a Julia job at the time):

| model | Serial | Threaded(2) | Threaded(4) | Threaded(8) |
|---|---|---|---|---|
| hydro-thermal 24×20 | 3.34 s | 2.13 s | 2.20 s | 2.36 s |
| belief | 6.63 s | 4.27 s | 3.24 s | 3.08 s |

The plateau is the GIL: only the solver runs in parallel, so the ceiling is
`python_time + solver_time / threads`. Models with larger LPs scale better.

### 9.6 Step 5: Cython decision
After steps 1–4 the serial profile (hydro-thermal 24×20, 200 iterations) is 44% inside the
solver; the only pure-Python hotspot left is the Level-1 cut-selection loop (~900k height
evaluations, ≈17% of profiled time, less in wall-clock). A numpy rewrite of that loop
(verified bit-identical on three 100-iteration traces) gains 6% wall-clock (3.38 s → 3.19 s).
A Cython version of the same loop cannot beat that by much and would add a compiler and
platform wheels to the install. Decision: no Cython; the scalar loop is kept for simplicity
and fidelity (the numpy variant is preserved in the session scratch as
`bellman_functions_vec.py`). Cumulative effect of steps 1–4 is summarised in FINAL_REPORT.md.

### 9.7 Belief bound validity check (open at the time of writing)
SDDP.jl, seed 123: 18.816721 (500), 18.816820 (1500), **18.816913 (5000 iterations, 5033 s)**.
Python after step 1 (warm-started solves), seed 123: 18.816518 (500), 18.816857 (1000),
18.816973 (1500), 18.817727 (2000), then a HiGHS failure (before the recovery ladder existed).
The 1500→2000 jump (+7.5e-4) is larger than any earlier increment on any run and the value
exceeds Julia's 5000-iteration bound, which raises the possibility that warm-started solves
returned slightly inaccurate duals and hence invalid (too high) cuts. Experiment in progress:
the same run with a forced cold start (`Highs_clearSolver`) before every solve vs the warm
run, 2500 iterations each, seed 123. Outcome and the resulting test decision are recorded
below when available.

### 9.8 Resolution: warm-started solves need tighter tolerances
Experiments (belief, seed 123, `Highs_clearSolver` used to force cold starts):

| run | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|
| warm (step 1 as first committed), HiGHS default tolerances 1e-7 | 18.816518 | 18.816857 | 18.816973 | 18.817727 | 18.817739 |
| cold start before every solve | 18.816566 | 18.816795 | 18.816812 | 18.816814 | 18.816815 |
| cold start before backward-pass solves only | 18.816566 | 18.816795 | 18.816810 | — | — |
| warm, tolerances 1e-9 | 18.816518 | 18.816793 | 18.816866 | 18.816870 | — |
| SDDP.jl (JuMP, warm, default tolerances) | 18.816721 | — | 18.816820 | — | 18.816913 @5000 |

The warm/default-tolerance run overshoots by ~9e-4 above every other run, i.e. it built
invalid cuts (a lower bound cannot exceed the optimum). Cold starts fix it but cost 3×.
Tightening HiGHS's primal/dual feasibility tolerances to 1e-9 fixes it at no measurable cost
at 200 iterations (belief 6.38 s, hydro 24×20 3.22 s) and ~12% at 2000 iterations on belief.
**Decision:** `sddp.HiGHS` now sets both tolerances to 1e-9 by default (override with
`sddp.HiGHS.with_options(...)`). Interpretation: warm-started dual simplex terminates as
soon as the basis is dual feasible within 1e-7; the belief model's μ variables (bounds ±100)
turn 1e-7 dual errors into 1e-5 cut errors that compound around the 0.9-discounted cycle.
SDDP.jl does not show this with its HiGHS build; the port is conservative.

Consequences for the tests:
* `test_belief` compares the 1500-iteration bounds at 1e-5 relative (Julia's own value moves
  by 5e-6 relative between 1500 and 5000 iterations, and the tight-tolerance Python value
  18.816866 lies inside that interval). This is the one tolerance in the suite above 1e-6,
  and the reason is slow convergence of a cyclic model, not a discrepancy.
* `fast_quickstart` deterministic run: with 1e-9 tolerances the forward pass picks a different
  optimal vertex at iteration 3 (any x ∈ [2, 3] is optimal, value −2; the bound matches, the
  forward value does not). The exact-match test runs that problem with Julia-equivalent
  tolerances (1e-7) through `tests.problems.set_optimizer`; every other exact-match test
  passes under the 1e-9 default.

Direct cut-validity check (each recorded cut height compared with a cold re-evaluation of
the same expected cost-to-go using the *final* cut set, which can only be higher for a valid
cut): warm run, 2000 iterations: **6 invalid cuts of 78,532**, the worst pair (nodes Ad/Bd,
cut #15545, inventory 1.9997, belief 0.858/0.142) recorded 17.348148 vs 17.347367, an excess
of 7.8e-4, which is the size of the bound overshoot; the other four exceed by 1.7e-6. Cold
run, 1000 iterations: 0 invalid of 37,812. Script: `belief_cutcheck.py` (session scratch).

## 10. Tier 4 port (value functions, inner approximation, formats, lattice, biobjective, ...)

### 10.1 Upstream defect: pyoptinterface HiGHS `set_normalized_rhs`
Found while round-tripping the StochOptFormat test model: pyoptinterface 0.6.1's HiGHS
backend implements `set_normalized_rhs` by setting *both* row bounds, so a `<=` or `>=` row
silently becomes an equality (probe: row `x+y <= 1`, after `set_normalized_rhs(3)` the range of
`x+y` over the box is `[3, 3]` instead of `[0, 3]`). Earlier parity problems were unaffected
because they change RHS only on equality rows or use `fix`/coefficients. Workaround in
`Model.set_normalized_rhs`: equality rows use pyoptinterface; inequality rows on HiGHS set
the row bounds through the HiGHS C API (`Highs_changeRowBounds`), locating the row by a unique
name that every row now receives at creation (`_sddp_row_<n>` unless the user named it).
Pinned by `tests/test_solver.py::test_set_normalized_rhs_keeps_inequality_sense`, including
the case where an earlier row was deleted.

### 10.2 SDDP.jl quirks replicated on purpose
* `_lattice_approximation` loops `for i in 1:length(states[t])` where `states[t]` is an `Int`,
  so only the first node of each stage is ever re-seeded from a sample path. The port loops
  over `range(1)` for parity; the oracle (`lattice.json`, 60 fixed sample paths) matches to
  1e-12 with this, and not without it.
* `ValueFunction` in SDDP.jl passes `belief_state, objective_state` in swapped order when
  copying local (multi-cut) thetas; the port uses the correct order (only matters for models
  that combine multi-cut with objective/belief states, which SDDP.jl does not exercise).
* The StochOptFormat reader registers states without an initial value (so the root-state
  feasibility check does not run during construction) and sets `initial_root_state` from the
  file afterwards, exactly as `Base.read` does in Julia.
