"""Print the parity table (Julia oracle vs sddp-py) used in the final report."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
import sddp  # noqa: E402
from tests.conftest import load_oracle  # noqa: E402
from tests.problems import BUILDERS, asset_risk  # noqa: E402

KW = {"print_level": 0, "run_numerical_stability_report": False}
rows = []


def add(problem, quantity, jl, py, tol=1e-6):
    rel = abs(py - jl) / max(1.0, abs(jl))
    rows.append((problem, quantity, jl, py, rel, "pass" if rel <= tol else "FAIL"))


def det(name):
    m = sddp.deterministic_equivalent(BUILDERS[name]())
    m.optimize()
    return m.objective_value()


def train(name, iters, **kw):
    m = BUILDERS[name]()
    sddp.train(m, iteration_limit=iters, seed=1234, **KW, **kw)
    return sddp.calculate_bound(m)


for name, key, iters in [
    ("hydro_thermal", "train_40", 40),
    ("fast_quickstart", "train_20", 20),
    ("fast_hydro_thermal", "train_20", 20),
    ("fast_production_management", "train_50", 50),
    ("farmers", "train_40", 40),
    ("markov_uncertainty", "train_40", 40),
    ("objective_uncertainty", "train_40", 40),
]:
    d = load_oracle(name)
    add(name, "deterministic equivalent", d["deterministic_equivalent"], det(name))
    add(name, f"bound after {iters} its", d[key]["bound"], train(name, iters))
d = load_oracle("stock_example")
add("stock_example", "bound after 60 its", d["train_60"]["bound"], train("stock_example", 60))
d = load_oracle("hydro_thermal")
for rm_name, rm in [
    ("WorstCase", sddp.WorstCase()),
    ("AVaR_0.5", sddp.AVaR(0.5)),
    ("EAVaR_0.5_0.25", sddp.EAVaR(lambda_=0.5, beta=0.25)),
    ("Entropic_0.1", sddp.Entropic(0.1)),
    ("ModifiedChiSquared_0.5", sddp.ModifiedChiSquared(0.5)),
    ("Wasserstein_10", sddp.Wasserstein(lambda x, y: abs(x.term - y.term), sddp.HiGHS, alpha=10.0)),
]:
    add("hydro_thermal", f"bound, {rm_name}, 30 its", d["risk_measures"][rm_name]["bound"], train("hydro_thermal", 30, risk_measure=rm))
add("hydro_thermal", "bound, multi-cut, 30 its", d["multi_cut_30"]["bound"], train("hydro_thermal", 30, cut_type=sddp.MULTI_CUT))
d = load_oracle("infinite_trivial")
add("infinite_trivial", "bound after 30 its", d["train_30"]["bound"], train("infinite_trivial", 30))
d = load_oracle("no_strong_duality")
add("no_strong_duality", "bound after 30 its", d["train_30"]["bound"], train("no_strong_duality", 30))
d = load_oracle("infinite_hydro_thermal")
for ct, k in [(sddp.SINGLE_CUT, "single"), (sddp.MULTI_CUT, "multi")]:
    add(
        "infinite_hydro_thermal", f"bound, {k} cut, 300 its", d[f"train_300_{k}"]["bound"],
        train("infinite_hydro_thermal", 300, cut_type=ct, sampling_scheme=sddp.InSampleMonteCarlo(terminate_on_cycle=True), cycle_discretization_delta=0.1),
        tol=1e-5,
    )
d = load_oracle("asset_management_stagewise")
for ct, k in [(sddp.SINGLE_CUT, "single"), (sddp.MULTI_CUT, "multi")]:
    add("asset_management_stagewise", f"bound, EAVaR, {k} cut, 100 its", d[f"train_100_{k}"]["bound"], train("asset_management_stagewise", 100, cut_type=ct, risk_measure=asset_risk))
d = load_oracle("objective_states")
add("objective_states", "converged bound (jl 60 / py 200 its)", d["train_60"]["bound"], train("objective_states", 200))
d = load_oracle("air_conditioning")
add("air_conditioning", "deterministic equivalent (MIP)", d["deterministic_equivalent"], det("air_conditioning"))
for h, k in [(sddp.ContinuousConicDuality(), "conic"), (sddp.LagrangianDuality(), "lagrangian"), (sddp.StrengthenedConicDuality(), "strengthened")]:
    add("air_conditioning", f"bound, {k}, 30 its", d[f"train_30_{k}"]["bound"], train("air_conditioning", 30, duality_handler=h))
d = load_oracle("stochastic_all_blacks")
for h, k in [(sddp.ContinuousConicDuality(), "conic"), (sddp.LagrangianDuality(), "lagrangian")]:
    add("stochastic_all_blacks", f"bound, {k}, 30 its", d[f"train_30_{k}"]["bound"], train("stochastic_all_blacks", 30, duality_handler=h))
d = load_oracle("sldp_example_one")
add("sldp_example_one", "bound after 50 its (unconverged MIP)", d["train_50"]["bound"], train("sldp_example_one", 50), tol=2e-3)

print("| problem | quantity | Julia | Python | rel. error | result |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.10g} | {r[3]:.10g} | {r[4]:.1e} | {r[5]} |")
