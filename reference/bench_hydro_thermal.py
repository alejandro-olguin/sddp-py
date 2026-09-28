"""Timing benchmark: hydro-thermal problem, sddp-py. Usage: python bench_hydro_thermal.py"""

import time

import numpy as np

import sddp


def build(stages: int, noises: int) -> sddp.PolicyGraph:
    omega = list(np.linspace(0.0, 100.0, noises))

    def builder(sp: sddp.Subproblem, t: int) -> None:
        volume = sp.add_state("volume", lb=0.0, ub=200.0, initial_value=200.0)
        thermal = sp.add_variable("thermal_generation", lb=0.0)
        hydro = sp.add_variable("hydro_generation", lb=0.0)
        spill = sp.add_variable("hydro_spill", lb=0.0)
        inflow = sp.add_variable("inflow")
        sp.parameterize(lambda w: sp.fix(inflow, w), omega)
        sp.add_constraint(volume.out == volume.in_ - hydro - spill + inflow)
        sp.add_constraint(hydro + thermal == 150.0)
        sp.set_stage_objective((50 + 10 * t) * thermal)

    return sddp.LinearPolicyGraph(builder, stages=stages, sense="Min", lower_bound=0.0)


if __name__ == "__main__":
    for stages, noises, iters, sims in [(3, 3, 100, 1000), (12, 10, 100, 1000), (24, 20, 200, 1000)]:
        m = build(stages, noises)
        t0 = time.perf_counter()
        sddp.train(m, iteration_limit=iters, print_level=0, seed=1)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        sddp.simulate(m, sims, ["volume"], seed=1)
        t_sim = time.perf_counter() - t0
        print(
            f"stages={stages} noises={noises} iters={iters}: train {t_train:.3f}s, "
            f"simulate({sims}) {t_sim:.3f}s, bound {sddp.calculate_bound(m)}"
        )
