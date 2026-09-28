import random

import pytest

import sddp
from tests.problems import build_fast_quickstart, build_hydro_thermal, build_infinite_trivial


def test_sample_noise_is_proportional():
    rng = random.Random(0)
    terms = [sddp.Noise("a", 0.2), sddp.Noise("b", 0.8)]
    counts = {"a": 0, "b": 0}
    for _ in range(5000):
        counts[sddp.plugins.sampling_schemes.sample_noise(terms, rng)] += 1
    assert 0.15 < counts["a"] / 5000 < 0.25


def test_in_sample_monte_carlo_linear():
    model = build_hydro_thermal()
    path, cycle = sddp.InSampleMonteCarlo().sample_scenario(model, random.Random(1))
    assert [n for n, _ in path] == [1, 2, 3]
    assert not cycle
    assert all(w in (0.0, 50.0, 100.0) for _, w in path)


def test_in_sample_monte_carlo_cycle_and_max_depth():
    model = build_infinite_trivial()
    path, cycle = sddp.InSampleMonteCarlo(terminate_on_cycle=True).sample_scenario(
        model, random.Random(1)
    )
    assert cycle and [n for n, _ in path] == ["week", "week"]
    path, cycle = sddp.InSampleMonteCarlo(
        max_depth=4, terminate_on_dummy_leaf=False
    ).sample_scenario(model, random.Random(1))
    assert not cycle and len(path) == 4


def test_historical_sequential_and_probabilistic():
    model = build_hydro_thermal()
    s = sddp.Historical([[(1, 0.0), (2, 50.0), (3, 100.0)], [(1, 100.0), (2, 0.0), (3, 50.0)]])
    a, _ = s.sample_scenario(model, random.Random(0))
    b, _ = s.sample_scenario(model, random.Random(0))
    c, _ = s.sample_scenario(model, random.Random(0))
    assert a == [(1, 0.0), (2, 50.0), (3, 100.0)] and b[0] == (1, 100.0) and c == a
    single = sddp.Historical([(1, 0.0), (2, 0.0), (3, 0.0)])
    assert single.sample_scenario(model, random.Random(0))[0] == [(1, 0.0), (2, 0.0), (3, 0.0)]
    prob = sddp.Historical([[(1, 0.0), (2, 0.0), (3, 0.0)]], [1.0])
    assert prob.sample_scenario(model, random.Random(0))[0] == [(1, 0.0), (2, 0.0), (3, 0.0)]


def test_out_of_sample_monte_carlo():
    model = build_hydro_thermal()

    def f(node):
        if node == 0:
            return [sddp.Noise(1, 1.0)]
        children = [sddp.Noise(node + 1, 1.0)] if node < 3 else []
        return children, [sddp.Noise(25.0, 1.0)]

    s = sddp.OutOfSampleMonteCarlo(f, model)
    path, _ = s.sample_scenario(model, random.Random(0))
    assert path == [(1, 25.0), (2, 25.0), (3, 25.0)]


def test_psr_sampling_scheme_cycles():
    model = build_hydro_thermal()
    s = sddp.PSRSamplingScheme(2)
    rng = random.Random(3)
    a = s.sample_scenario(model, rng)[0]
    b = s.sample_scenario(model, rng)[0]
    assert s.sample_scenario(model, rng)[0] == a
    assert s.sample_scenario(model, rng)[0] == b


def test_stopping_rules():
    model = build_hydro_thermal()
    sddp.train(model, iteration_limit=3, print_level=0)
    assert sddp.termination_status(model) == "iteration_limit"
    model = build_hydro_thermal()
    sddp.train(model, stopping_rules=[sddp.BoundStalling(3, atol=1e-6)], print_level=0, seed=1)
    assert sddp.termination_status(model) == "bound_stalling"
    model = build_hydro_thermal()
    sddp.train(model, time_limit=0.0, print_level=0)
    assert sddp.termination_status(model) == "time_limit"
    model = build_hydro_thermal()
    sddp.train(model, print_level=0, seed=1)  # default SimulationStoppingRule
    assert sddp.termination_status(model) == "simulation_stopping"
    model = build_hydro_thermal()
    sddp.train(
        model,
        stopping_rules=[
            sddp.StoppingChain(sddp.IterationLimit(5), sddp.BoundStalling(2, atol=1e-6))
        ],
        print_level=0,
        seed=1,
    )
    assert sddp.termination_status(model) == "iteration_limit ∧ bound_stalling"
    model = build_fast_quickstart()  # deterministic first stage
    sddp.train(
        model, stopping_rules=[sddp.FirstStageStoppingRule(iterations=5)], print_level=0, seed=1
    )
    assert sddp.termination_status(model) == "first_stage_stopping"
    with pytest.raises(ValueError, match="not deterministic"):
        sddp.train(
            build_hydro_thermal(), stopping_rules=[sddp.FirstStageStoppingRule()], print_level=0
        )
    model = build_hydro_thermal()
    sddp.train(
        model,
        stopping_rules=[sddp.Statistical(num_replications=20, disable_warning=True, verbose=False)],
        print_level=0,
        seed=1,
    )
    assert sddp.termination_status(model) == "statistical"
