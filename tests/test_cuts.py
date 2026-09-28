"""Cut formation against a hand-solved two-stage problem.

Stage 1: choose x_out in [0, 10] at cost 1 * x_out.
Stage 2: incoming x; demand d in {2, 6} with prob 1/2 each; sell s <= x, s <= d at revenue 3;
         cost = -3 s. V2(x, d) = -3 min(x, d).
Backward pass at x̂ = 4: V2(4, 2) = -6 with dual dV/dx = 0; V2(4, 6) = -12 with dual -3.
Expected average cut: theta >= (-6 - 12)/2 + (0 - 3)/2 * (x - 4) = -9 - 1.5 (x - 4)
i.e. intercept (at x=0) = -3, coefficient -1.5. The stored "intercept" in the JSON is
the height at x̂ = -9.
"""

import math

import sddp


def build():
    def builder(sp, t):
        x = sp.add_state("x", lb=0.0, ub=10.0, initial_value=0.0)
        if t == 1:
            sp.set_stage_objective(1.0 * x.out)
        else:
            s = sp.add_variable("s", lb=0.0)
            sp.add_constraint(s <= x.in_)
            sp.parameterize(lambda d: sp.set_upper_bound(s, d), [2.0, 6.0])
            sp.set_stage_objective(-3.0 * s)
            sp.add_constraint(x.out == 0.0)

    return sddp.LinearPolicyGraph(builder, stages=2, lower_bound=-100.0, optimizer=sddp.HiGHS)


def test_backward_pass_cut_matches_hand_solution():
    model = build()
    node1, node2 = model[1], model[2]
    for node in model.nodes.values():
        node.bellman_function.cut_type = sddp.SINGLE_CUT
    options = sddp.algorithm.Options.create(model, model.initial_root_state)
    cuts = sddp.algorithm.backward_pass(model, options, [(1, None), (2, 2.0)], [{"x": 4.0}, {"x": 0.0}], [], [])
    (cut,) = cuts[1]
    assert math.isclose(cut["theta"], -9.0)
    assert math.isclose(cut["pi"]["x"], -1.5)
    stored = node1.bellman_function.global_theta.cuts[0]
    assert math.isclose(stored.intercept, -3.0)  # -9 - (-1.5 * 4)
    assert math.isclose(stored.coefficients["x"], -1.5)
    # Child duals: check the stage-2 subproblem duals individually.
    r2 = sddp.algorithm.solve_subproblem(model, node2, {"x": 4.0}, 2.0, [], duality_handler=sddp.ContinuousConicDuality())
    assert math.isclose(r2.objective, -6.0) and math.isclose(r2.duals["x"], 0.0)
    r6 = sddp.algorithm.solve_subproblem(model, node2, {"x": 4.0}, 6.0, [], duality_handler=sddp.ContinuousConicDuality())
    assert math.isclose(r6.objective, -12.0) and math.isclose(r6.duals["x"], -3.0)


def test_train_reaches_optimal_bound():
    # Optimal: choose x = 6 (buy 6, expected revenue 3*(2+6)/2 = 12) -> 6 - 12 = -6.
    # x=2 -> 2 - 6 = -4; x=6 -> -6; x between: 1*x - 1.5*(x) - 3 = -3 - 0.5x decreasing to x=6.
    model = build()
    sddp.train(model, iteration_limit=10, print_level=0)
    assert math.isclose(sddp.calculate_bound(model), -6.0, abs_tol=1e-8)


def test_max_sense_cut_signs():
    """Same problem as a maximisation of the negated objective: cuts must flip."""

    def builder(sp, t):
        x = sp.add_state("x", lb=0.0, ub=10.0, initial_value=0.0)
        if t == 1:
            sp.set_stage_objective(-1.0 * x.out)
        else:
            s = sp.add_variable("s", lb=0.0)
            sp.add_constraint(s <= x.in_)
            sp.parameterize(lambda d: sp.set_upper_bound(s, d), [2.0, 6.0])
            sp.set_stage_objective(3.0 * s)
            sp.add_constraint(x.out == 0.0)

    model = sddp.LinearPolicyGraph(builder, stages=2, sense="Max", upper_bound=100.0, optimizer=sddp.HiGHS)
    for node in model.nodes.values():
        node.bellman_function.cut_type = sddp.SINGLE_CUT
    options = sddp.algorithm.Options.create(model, model.initial_root_state)
    cuts = sddp.algorithm.backward_pass(model, options, [(1, None), (2, 2.0)], [{"x": 4.0}, {"x": 0.0}], [], [])
    (cut,) = cuts[1]
    assert math.isclose(cut["theta"], 9.0)
    assert math.isclose(cut["pi"]["x"], 1.5)
    sddp.train(model, iteration_limit=10, print_level=0)
    assert math.isclose(sddp.calculate_bound(model), 6.0, abs_tol=1e-8)


def test_write_and_read_cuts_roundtrip(tmp_path):
    model = build()
    sddp.train(model, iteration_limit=5, print_level=0)
    f = tmp_path / "cuts.json"
    sddp.write_cuts_to_file(model, str(f))
    model2 = build()
    sddp.read_cuts_from_file(model2, str(f))
    assert math.isclose(sddp.calculate_bound(model2), sddp.calculate_bound(model), abs_tol=1e-9)
