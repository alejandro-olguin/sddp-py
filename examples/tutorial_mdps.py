"""Example: Markov Decision Processes (https://sddp.dev/stable/tutorial/mdps/).

A faithful port of ``docs/src/tutorial/mdps.jl``: Putterman's sum-of-squares budget problem
(a convex quadratic stage objective, solved with HiGHS's QP solver instead of Ipopt) and the
stationary binary maze on a ``UnicyclicGraph``.
"""

from __future__ import annotations

from typing import Any

import sddp

# ------------------------------------------------------------------ simple example
M_BUDGET, N_STAGES = 5, 3


def build_sum_of_squares(M: float = M_BUDGET, N: int = N_STAGES) -> sddp.PolicyGraph:
    """``min Σ x_i²  s.t. Σ x_i = M, x ≥ 0`` as an N-stage MDP over the unspent budget ``s``."""

    def builder(sp: sddp.Subproblem, node: int) -> None:
        s = sp.add_state("s", lb=0.0, initial_value=M)
        x = sp.add_variable("x", lb=0.0)
        sp.set_stage_objective(x * x)
        sp.add_constraint(x <= s.in_)
        sp.add_constraint(s.out == s.in_ - x)
        if node == N:
            sp.fix(s.out, 0.0)

    return sddp.LinearPolicyGraph(builder, stages=N, lower_bound=0.0, optimizer=sddp.HiGHS)


def run_sum_of_squares(seed: int | None = 1, **kwargs: Any) -> tuple[float, list[float]]:
    """Train with the default stopping rule; return ``(bound, [x_1, ..., x_N])``."""
    model = build_sum_of_squares()
    sddp.train(model, seed=seed, **kwargs)
    simulations = sddp.simulate(model, 1, ["x"], seed=seed)
    xs = [data["x"] for data in simulations[0]]
    return sddp.calculate_bound(model), xs


# ---------------------------------------------------------------- the maze
M_ROWS, N_COLS = 3, 4
INITIAL_SQUARE = (1, 1)
REWARD = (3, 4)
ILLEGAL_SQUARES = [(2, 2)]
PENALTIES = [(3, 1), (2, 4)]
DISCOUNT_FACTOR = 0.9


def maze_path() -> dict[tuple[int, int], str]:
    """The initial ``path`` matrix of the page (1-based ``(i, j)`` keys)."""
    path = {(i, j): "⋅" for i in range(1, M_ROWS + 1) for j in range(1, N_COLS + 1)}
    path[INITIAL_SQUARE] = "1"
    for squares, v in ((ILLEGAL_SQUARES, "▩"), (PENALTIES, "†"), ([REWARD], "*")):
        for i, j in squares:
            path[(i, j)] = v
    return path


def path_string(path: dict[tuple[int, int], str]) -> str:
    return "\n".join(
        " ".join(path[(i, j)] for j in range(1, N_COLS + 1)) for i in range(1, M_ROWS + 1)
    )


def valid_moves(i: int, j: int) -> list[tuple[int, int]]:
    moves = [(i - 1, j), (i + 1, j), (i, j), (i, j + 1), (i, j - 1)]
    return [v for v in moves if 1 <= v[0] <= M_ROWS and 1 <= v[1] <= N_COLS]


def build_maze(discount_factor: float = DISCOUNT_FACTOR) -> sddp.PolicyGraph:
    graph = sddp.UnicyclicGraph(discount_factor)
    squares = [(i, j) for i in range(1, M_ROWS + 1) for j in range(1, N_COLS + 1)]

    def builder(sp: sddp.Subproblem, _: int) -> None:
        # Our state is a binary variable for each square
        x = {
            (i, j): sp.add_state(
                f"x[{i},{j}]", binary=True, initial_value=float((i, j) == INITIAL_SQUARE)
            )
            for (i, j) in squares
        }
        # Can only be in one square at a time
        sp.add_constraint(sum(x[ij].out for ij in squares) == 1)
        # Incur rewards and penalties
        sp.set_stage_objective(x[REWARD].out - sum(x[ij].out for ij in PENALTIES))
        # Some squares are illegal
        for ij in ILLEGAL_SQUARES:
            sp.add_constraint(x[ij].out <= 0)
        # Constraints on valid moves
        for i, j in squares:
            sp.add_constraint(x[(i, j)].out <= sum(x[ab].in_ for ab in valid_moves(i, j)))

    return sddp.PolicyGraph(
        builder,
        graph,
        sense="Max",
        upper_bound=1 / (1 - discount_factor),
        optimizer=sddp.HiGHS,
    )


def simulate_maze(model: sddp.PolicyGraph, seed: int | None = 1) -> list[dict[str, Any]]:
    """One replication with ``InSampleMonteCarlo(max_depth=5, terminate_on_dummy_leaf=False)``."""
    squares = [(i, j) for i in range(1, M_ROWS + 1) for j in range(1, N_COLS + 1)]
    simulations = sddp.simulate(
        model,
        1,
        [f"x[{i},{j}]" for (i, j) in squares],
        sampling_scheme=sddp.InSampleMonteCarlo(max_depth=5, terminate_on_dummy_leaf=False),
        seed=seed,
    )
    return simulations[0]


def maze_visits(simulation: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    """Fill in ``path`` with the time-step in which each square is visited (incoming state)."""
    path = maze_path()
    for t, data in enumerate(simulation, start=1):
        for i in range(1, M_ROWS + 1):
            for j in range(1, N_COLS + 1):
                if data[f"x[{i},{j}]"].in_ > 0.5:
                    path[(i, j)] = str(t)
    return path


def run_maze(seed: int | None = 1, **kwargs: Any) -> tuple[float, dict[tuple[int, int], str]]:
    model = build_maze()
    sddp.train(model, seed=seed, **kwargs)
    sim = simulate_maze(model, seed=seed)
    return sddp.calculate_bound(model), maze_visits(sim)


def main() -> None:
    bound, xs = run_sum_of_squares()
    print(bound, M_BUDGET**2 / N_STAGES)
    for t, x in enumerate(xs, start=1):
        print(f"x_{t} = {x}")

    print(path_string(maze_path()))
    print(sddp.UnicyclicGraph(DISCOUNT_FACTOR))
    model = build_maze()
    sddp.plot_graph(model, "model_mdp.html")
    sddp.train(model, seed=1)
    sim = simulate_maze(model, seed=1)
    print(path_string(maze_visits(sim)))


if __name__ == "__main__":
    main()
