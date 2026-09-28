"""Benchmark + dual-sign check for the solver layer decision (see PORTING_NOTES.md §4).

Workload: hydro-thermal-like stage LP. 1 state (volume), 3 controls, 2 rows.
Loop N times: fix x_in to a random value, change the inflow RHS, add one cut row,
solve, read objective / primal / reduced cost of x_in.
"""
import random
import time
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def bench_poi():
    import pyoptinterface as poi
    from pyoptinterface import highs

    m = highs.Model()
    m.set_raw_parameter("output_flag", False)
    x_in = m.add_variable(lb=0.0, ub=200.0, name="x_in")
    x_out = m.add_variable(lb=0.0, ub=200.0, name="x_out")
    th = m.add_variable(lb=0.0, name="thermal")
    hy = m.add_variable(lb=0.0, name="hydro")
    sp = m.add_variable(lb=0.0, name="spill")
    inflow = m.add_variable(name="inflow")
    theta = m.add_variable(lb=0.0, name="theta")
    bal = m.add_linear_constraint(x_out - x_in + hy + sp - inflow, poi.Eq, 0.0)
    dem = m.add_linear_constraint(hy + th, poi.Eq, 150.0)
    m.set_objective(50.0 * th + theta, poi.ObjectiveSense.Minimize)
    rng = random.Random(1)
    t0 = time.perf_counter()
    for k in range(N):
        v = rng.uniform(0, 200)
        m.set_variable_bounds(x_in, v, v)
        w = rng.choice([0.0, 50.0, 100.0])
        m.set_variable_bounds(inflow, w, w)
        # cut: theta >= a + b * x_out
        a, b = rng.uniform(0, 20000), -rng.uniform(0, 150)
        m.add_linear_constraint(theta - b * x_out, poi.Geq, a)
        m.optimize()
        obj = m.get_obj_value()
        _ = m.get_value(x_out)
        _ = m.get_variable_attribute(x_in, poi.VariableAttribute.ReducedCost)
    dt = time.perf_counter() - t0
    return dt, obj


def bench_highspy():
    import highspy
    import numpy as np

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    # columns: x_in, x_out, th, hy, sp, inflow, theta
    lbs = np.array([0, 0, 0, 0, 0, -inf, 0], dtype=float)
    ubs = np.array([200, 200, inf, inf, inf, inf, inf], dtype=float)
    cost = np.array([0, 0, 50, 0, 0, 0, 1], dtype=float)
    h.addVars(7, lbs, ubs)
    h.changeColsCost(7, np.arange(7, dtype=np.int32), cost)
    # rows
    h.addRow(0.0, 0.0, 5, np.array([1, 0, 3, 4, 5], dtype=np.int32), np.array([1, -1, 1, 1, -1], dtype=float))
    h.addRow(150.0, 150.0, 2, np.array([2, 3], dtype=np.int32), np.array([1, 1], dtype=float))
    rng = random.Random(1)
    t0 = time.perf_counter()
    for k in range(N):
        v = rng.uniform(0, 200)
        h.changeColBounds(0, v, v)
        w = rng.choice([0.0, 50.0, 100.0])
        h.changeColBounds(5, w, w)
        a, b = rng.uniform(0, 20000), -rng.uniform(0, 150)
        h.addRow(a, inf, 2, np.array([6, 1], dtype=np.int32), np.array([1.0, -b]))
        h.run()
        obj = h.getInfo().objective_function_value
        sol = h.getSolution()
        _ = sol.col_value[1]
        _ = sol.col_dual[0]
    dt = time.perf_counter() - t0
    return dt, obj


def dual_sign_check():
    """Tiny LPs with hand-computable duals.

    P1 (min): min 2*x + y  s.t. x fixed to 3 (lb=ub=3), y >= 1.  obj = 7.
        d obj / d x_fix = 2   → we want lambda = +2
    P2 (max): max -2*x - y  s.t. x fixed to 3, y >= 1.  obj = -7.
        d obj / d x_fix = -2  → we want lambda = -2
    Row duals: P3 (min): min x s.t. x >= 1 (row) → d obj / d rhs = +1.
               P4 (max): max -x s.t. x >= 1 (row) → d obj / d rhs = -1.
    """
    import pyoptinterface as poi
    from pyoptinterface import highs
    import highspy

    out = {}
    for sense, c in (("min", 1.0), ("max", -1.0)):
        m = highs.Model()
        m.set_raw_parameter("output_flag", False)
        x = m.add_variable(lb=3.0, ub=3.0)
        y = m.add_variable(lb=1.0)
        m.set_objective(c * (2.0 * x + y), poi.ObjectiveSense.Minimize if sense == "min" else poi.ObjectiveSense.Maximize)
        m.optimize()
        rc = m.get_variable_attribute(x, poi.VariableAttribute.ReducedCost)
        out[f"poi_{sense}_reduced_cost_fixed_var"] = (rc, m.get_obj_value())
        m2 = highs.Model()
        m2.set_raw_parameter("output_flag", False)
        z = m2.add_variable()
        con = m2.add_linear_constraint(1.0 * z, poi.Geq, 1.0)
        m2.set_objective(c * z, poi.ObjectiveSense.Minimize if sense == "min" else poi.ObjectiveSense.Maximize)
        m2.optimize()
        out[f"poi_{sense}_row_dual"] = (m2.get_constraint_attribute(con, poi.ConstraintAttribute.Dual), m2.get_obj_value())
        # raw highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.addVar(3.0, 3.0)
        h.addVar(1.0, highspy.kHighsInf)
        h.changeColCost(0, c * 2.0)
        h.changeColCost(1, c * 1.0)
        h.changeObjectiveSense(highspy.ObjSense.kMinimize if sense == "min" else highspy.ObjSense.kMaximize)
        h.run()
        out[f"highspy_{sense}_reduced_cost_fixed_var"] = (h.getSolution().col_dual[0], h.getInfo().objective_function_value)
    return out


if __name__ == "__main__":
    for k, v in dual_sign_check().items():
        print(f"{k:45s} dual={v[0]:+.3f} obj={v[1]:+.3f}")
    dt, obj = bench_poi()
    print(f"pyoptinterface: N={N} solves+cuts in {dt:.3f}s ({1e3*dt/N:.3f} ms/solve), last obj {obj:.4f}")
    dt, obj = bench_highspy()
    print(f"highspy raw   : N={N} solves+cuts in {dt:.3f}s ({1e3*dt/N:.3f} ms/solve), last obj {obj:.4f}")
