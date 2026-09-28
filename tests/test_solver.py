"""Dual sign contract of the solver layer (PORTING_NOTES.md §4.2)."""

import sddp
from sddp.solver.model import Model, Sense


def test_reduced_cost_of_fixed_variable_is_objective_sensitivity_min():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x")
    y = m.add_variable("y", lb=1.0)
    m.fix(x, 3.0)
    m.set_objective(2.0 * x + y, Sense.MIN)
    m.optimize()
    assert m.objective_value() == 7.0
    assert m.reduced_cost(x) == 2.0


def test_reduced_cost_of_fixed_variable_is_objective_sensitivity_max():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x")
    y = m.add_variable("y", lb=1.0)
    m.fix(x, 3.0)
    m.set_objective(-2.0 * x - y, Sense.MAX)
    m.optimize()
    assert m.objective_value() == -7.0
    assert m.reduced_cost(x) == -2.0


def test_row_dual_is_rhs_sensitivity():
    for sense, c, expected in ((Sense.MIN, 1.0, 1.0), (Sense.MAX, -1.0, -1.0)):
        m = Model(sddp.HiGHS)
        z = m.add_variable("z")
        con = m.add_constraint(1.0 * z >= 1.0)
        m.set_objective(c * z, sense)
        m.optimize()
        assert m.dual(con) == expected


def test_constraint_normalisation_moves_constant_to_rhs():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x", lb=0.0)
    y = m.add_variable("y", lb=0.0)
    c = m.add_constraint(x + 2.0 == y - 3.0)  # x - y == -5
    terms, sense, rhs = m.constraint_data(c)
    assert sense == "=="
    assert rhs == -5.0
    assert sorted((m.variable_name(v), coef) for v, coef in terms) == [("x", 1.0), ("y", -1.0)]


def test_constant_objective_and_expression_value():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x", lb=1.0)
    m.set_objective(x + 5.0, Sense.MIN)
    m.optimize()
    assert m.objective_value() == 6.0
    assert m.value(2 * x + 1) == 3.0
    assert m.value(7.0) == 7.0


def test_delete_constraint_and_relax_integrality():
    m = Model(sddp.HiGHS)
    x = m.add_variable("x", lb=0.0, ub=10.0, integer=True)
    c = m.add_constraint(1.0 * x >= 2.5)
    m.set_objective(1.0 * x, Sense.MIN)
    m.optimize()
    assert m.objective_value() == 3.0
    undo = m.relax_integrality()
    m.optimize()
    assert m.objective_value() == 2.5
    undo()
    m.optimize()
    assert m.objective_value() == 3.0
    m.delete_constraint(c)
    m.optimize()
    assert m.objective_value() == 0.0
    assert m.num_constraints() == 0


def test_set_normalized_rhs_keeps_inequality_sense():
    """pyoptinterface's HiGHS set_normalized_rhs makes inequalities equalities; ours must not."""
    for sense, lo_hi in (("<=", (0.0, 3.0)), (">=", (3.0, 20.0)), ("==", (3.0, 3.0))):
        m = Model(sddp.HiGHS)
        x = m.add_variable("x", lb=0.0, ub=10.0)
        y = m.add_variable("y", lb=0.0, ub=10.0)
        c = m.add_constraint_normalized(x + y, sense, 1.0)
        m.set_normalized_rhs(c, 3.0)
        assert m.get_normalized_rhs(c) == 3.0
        m.set_objective(x + y, Sense.MIN)
        m.optimize()
        lo = m.objective_value()
        m.set_objective(x + y, Sense.MAX)
        m.optimize()
        hi = m.objective_value()
        assert (lo, hi) == lo_hi, (sense, lo, hi)
        # also after a deletion shifts the HiGHS row positions
        m2 = Model(sddp.HiGHS)
        a = m2.add_variable("a", lb=0.0, ub=10.0)
        c0 = m2.add_constraint_normalized(1.0 * a, "==", 5.0)
        c1 = m2.add_constraint_normalized(1.0 * a, sense, 1.0)
        m2.delete_constraint(c0)
        m2.set_normalized_rhs(c1, 3.0)
        m2.set_objective(1.0 * a, Sense.MIN)
        m2.optimize()
        assert m2.objective_value() == {"<=": 0.0, ">=": 3.0, "==": 3.0}[sense]
