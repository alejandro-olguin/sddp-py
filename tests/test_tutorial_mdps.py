"""Markov Decision Processes tutorial: quadratic sum-of-squares MDP and the binary maze."""

from __future__ import annotations

import math

import pytest
from examples import tutorial_mdps as mdp

import sddp
from sddp.solver.model import Model, Sense
from tests.conftest import load_oracle

KW = {"print_level": 0, "run_numerical_stability_report": False}


def close(a, b, rel=1e-6, atol=1e-6):
    return math.isclose(a, b, rel_tol=rel, abs_tol=atol)


# ------------------------------------------------------------- quadratic objectives
def test_solver_layer_quadratic_objective():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x", lb=0.0)
    y = m.add_variable("y", lb=0.0)
    s = m.add_variable("s")
    m.fix(s, 5.0)
    m.add_constraint(x + y == s)
    m.set_objective(x * x + y * y, Sense.MIN)
    m.optimize()
    assert close(m.objective_value(), 12.5)
    assert close(m.value(x), 2.5) and close(m.value(y), 2.5)
    assert close(m.value(x * x + y * y), 12.5)
    assert close(m.reduced_cost(s), 5.0, atol=1e-5)  # d(obj)/ds = 2x
    # switching back to a linear objective clears the quadratic part
    m.set_objective(1.0 * x, Sense.MIN)
    m.optimize()
    assert close(m.objective_value(), 0.0)
    with pytest.raises(TypeError, match="Quadratic constraints"):
        m.add_constraint(x * x <= 1.0)


def test_sum_of_squares_optimum():
    M, N = mdp.M_BUDGET, mdp.N_STAGES
    bound, xs = mdp.run_sum_of_squares(seed=1, **KW)
    assert close(bound, M**2 / N, rel=1e-6, atol=1e-6)
    assert len(xs) == N
    for x in xs:
        assert close(x, M / N, rel=1e-4, atol=1e-4)
    assert close(sum(xs), M, atol=1e-6)


def test_sum_of_squares_matches_oracle_optimum():
    d = load_oracle("tutorials_c")["mdps"]
    bound, _ = mdp.run_sum_of_squares(seed=2, **KW)
    assert close(bound, d["sum_of_squares_optimum"])


# ---------------------------------------------------------------------- the maze
def test_maze_initial_path():
    assert mdp.path_string(mdp.maze_path()) == "1 ⋅ ⋅ ⋅\n⋅ ▩ ⋅ †\n† ⋅ ⋅ *"


def test_maze_graph():
    g = sddp.UnicyclicGraph(mdp.DISCOUNT_FACTOR)
    assert g.nodes[1] == [(1, 0.9)]


def test_maze_policy():
    d = load_oracle("tutorials_c")["mdps"]
    model = mdp.build_maze()
    sddp.train(model, seed=1, **KW)
    bound = sddp.calculate_bound(model)
    # Reward from stage 5 on: 0.9^4 / (1 - 0.9) = 6.561, the optimum (5 moves to the reward).
    assert close(bound, 0.9**4 / (1 - 0.9), rel=1e-6, atol=1e-6)
    assert close(bound, d["maze_bound"], rel=1e-6, atol=1e-6)
    sim = mdp.simulate_maze(model, seed=1)
    assert len(sim) == 5 == d["maze_stages_simulated"]
    assert [s["node_index"] for s in sim] == [1] * 5
    # never in an illegal square, exactly one square at a time
    for s in sim:
        for i, j in mdp.ILLEGAL_SQUARES:
            assert s[f"x[{i},{j}]"].out < 0.5 and s[f"x[{i},{j}]"].in_ < 0.5
        assert sum(s[f"x[{i},{j}]"].out for i in range(1, 4) for j in range(1, 5)) == pytest.approx(
            1
        )
    # reaches the reward square by step 5 (outgoing state of the fifth stage)
    assert sim[4][f"x[{mdp.REWARD[0]},{mdp.REWARD[1]}]"].out > 0.5
    assert sum(s["stage_objective"] for s in sim) == pytest.approx(1.0)
    path = mdp.maze_visits(sim)
    assert mdp.path_string(path).split("\n") == d["maze_path"]
    assert mdp.path_string(path) == "1 2 3 ⋅\n⋅ ▩ 4 †\n† ⋅ 5 *"
