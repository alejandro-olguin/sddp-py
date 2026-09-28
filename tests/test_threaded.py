import math

import sddp
from tests.conftest import load_oracle
from tests.problems import build_belief, build_hydro_thermal, build_markov_uncertainty


def test_threaded_train_converges_to_serial_bound():
    d = load_oracle("hydro_thermal")
    model = build_hydro_thermal()
    sddp.train(model, iteration_limit=40, print_level=0, seed=1, parallel_scheme=sddp.Threaded(4))
    assert math.isclose(sddp.calculate_bound(model), d["deterministic_equivalent"], rel_tol=1e-6)
    assert sddp.termination_status(model) == "iteration_limit"
    # Threads may overshoot the iteration limit by at most (threads - 1).
    assert 40 <= len(model.most_recent_training_results.log) <= 43
    assert {log.pid for log in model.most_recent_training_results.log} <= {1, 2, 3, 4, 5}


def test_threaded_markov_and_belief():
    d = load_oracle("markov_uncertainty")
    model = build_markov_uncertainty()
    sddp.train(model, iteration_limit=60, print_level=0, seed=1, parallel_scheme=sddp.Threaded(3))
    assert math.isclose(sddp.calculate_bound(model), d["deterministic_equivalent"], rel_tol=1e-6)
    model = build_belief()
    sddp.train(
        model,
        iteration_limit=200,
        print_level=0,
        seed=1,
        cut_type=sddp.SINGLE_CUT,
        parallel_scheme=sddp.Threaded(2),
    )
    # A valid lower bound (converged value ~18.8168) that has made progress; thread
    # interleaving makes the exact value run-dependent.
    assert 18.3 < sddp.calculate_bound(model) <= 18.82


def test_threaded_simulate_matches_serial_layout():
    model = build_hydro_thermal()
    sddp.train(model, iteration_limit=10, print_level=0, seed=1)
    sims = sddp.simulate(model, 37, ["volume"], parallel_scheme=sddp.Threaded(4), seed=1)
    assert len(sims) == 37 and all(len(s) == 3 for s in sims)
    assert all("volume" in stage and "stage_objective" in stage for s in sims for stage in s)
