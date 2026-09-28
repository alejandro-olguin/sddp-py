"""Risk measures against hand-computed reweightings (mirrors SDDP.jl test/plugins/risk_measures.jl)."""

import math

import pytest

import sddp


def adjust(measure, p, V, is_min, supports=None):
    q = [0.0] * len(p)
    supports = supports if supports is not None else [f"s{i}" for i in range(len(p))]
    offset = measure.adjust_probability(q, p, supports, V, is_min)
    return q, offset


def approx(a, b, atol=1e-6):
    return all(math.isclose(x, y, abs_tol=atol, rel_tol=0.0) for x, y in zip(a, b))


def test_expectation():
    q, _ = adjust(sddp.Expectation(), [0.1, 0.2, 0.3, 0.4, 0.5], [5.0, 4.0, 6.0, 2.0, 1.0], True)
    assert q == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_worst_case():
    q, _ = adjust(sddp.WorstCase(), [0.1, 0.2, 0.0, 0.4, 0.5], [5.0, 4.0, 6.0, 2.0, 1.0], True)
    assert q == [1.0, 0.0, 0.0, 0.0, 0.0]  # index 2 has zero probability
    q, _ = adjust(sddp.WorstCase(), [0.1, 0.2, 0.3, 0.4, 0.5], [5.0, 4.0, 6.0, 2.0, 1.0], False)
    assert q == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_avar_bounds():
    with pytest.raises(ValueError):
        sddp.AVaR(-0.1)
    with pytest.raises(ValueError):
        sddp.AVaR(1.1)


def test_avar_02():
    q, _ = adjust(sddp.AVaR(0.2), [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], False)
    assert q == [0.5, 0.5, 0.0, 0.0]


def test_avar_0_and_1():
    q, _ = adjust(sddp.AVaR(0.0), [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], False)
    assert q == [1.0, 0.0, 0.0, 0.0]
    q, _ = adjust(sddp.AVaR(1.0), [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], False)
    assert q == [0.1, 0.2, 0.3, 0.4]


def test_avar_docstring_example():
    q, _ = adjust(sddp.AVaR(0.5), [0.1, 0.2, 0.3, 0.4], [5.0, 4.0, 6.0, 2.0], True)
    assert approx(q, [0.2, 0.2, 0.6, 0.0], atol=1e-12)


def test_avar_ties_keep_original_order():
    # Julia's sortperm is stable: with equal costs, earlier indices are filled first.
    q, _ = adjust(sddp.AVaR(0.5), [0.25, 0.25, 0.25, 0.25], [1.0, 1.0, 1.0, 1.0], True)
    assert q == [0.5, 0.5, 0.0, 0.0]


def test_eavar_constructor_and_combination():
    m = sddp.EAVaR(lambda_=0.5, beta=0.25)
    assert m.measures[0] == (0.5, sddp.Expectation())
    assert m.measures[1] == (0.5, sddp.AVaR(0.25))
    d = 0.5 * sddp.Expectation() + 0.3 * sddp.AVaR(0.5) + 0.2 * sddp.WorstCase()
    assert d.measures[0] == (0.5, sddp.Expectation())
    assert d.measures[1] == (0.3, sddp.AVaR(0.5))
    assert d.measures[2] == (0.2, sddp.WorstCase())
    with pytest.raises(ValueError):
        sddp.EAVaR(lambda_=1.1)
    with pytest.raises(ValueError):
        sddp.EAVaR(beta=-0.1)


def test_eavar_values():
    p = [0.1, 0.2, 0.3, 0.4]
    q, _ = adjust(sddp.EAVaR(lambda_=0.25, beta=0.2), p, [1.0, 2.0, 3.0, 4.0], False)
    assert approx(q, [0.25 * a + 0.75 * b for a, b in zip(p, [0.5, 0.5, 0, 0])], 1e-12)
    q, _ = adjust(sddp.EAVaR(lambda_=0.25, beta=0.2), p, [1.0, 2.0, 3.0, 4.0], True)
    assert approx(q, [0.25 * a + 0.75 * b for a, b in zip(p, [0, 0, 0, 1.0])], 1e-12)
    q, _ = adjust(sddp.EAVaR(lambda_=0.5, beta=0.0), [0.0, 0.2, 0.4, 0.4], [1.0, 2.0, 3.0, 4.0], False)
    assert approx(q, [0.5 * a + 0.5 * b for a, b in zip([0.0, 0.2, 0.4, 0.4], [0.0, 1.0, 0, 0])], 1e-12)


@pytest.mark.parametrize(
    "radius,p,V,is_min,expected",
    [
        (0.0, [0.2] * 5, [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0.2] * 5),
        (0.0, [0.1, 0.2, 0.3, 0.2, 0.2], [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0.1, 0.2, 0.3, 0.2, 0.2]),
        (6.0, [0.1, 0.2, 0.3, 0.3, 0.1], [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0.0, 1.0, 0.0, 0.0, 0.0]),
        (6.0, [0.1, 0.2, 0.3, 0.3, 0.1], [-2.0, -1.0, -3.0, -4.0, -5.0], False, [0.0, 0.0, 0.0, 0.0, 1.0]),
        (0.45, [0.1, 0.2, 0.3, 0.3, 0.1], [-2.0, -1.0, -3.0, -4.0, -0.5], True, [0.115714, 0.372861, 0.158568, 0.001421, 0.351435]),
        (0.45, [0.1, 0.2, 0.3, 0.3, 0.1], [-2.0, -1.0, -3.0, -4.0, -0.5], False, [0.0, 0.0, 0.323223, 0.676777, 0.0]),
        (0.25, [0.2] * 5, [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0.279057, 0.358114, 0.2, 0.120943, 0.0418861]),
        (0.25, [0.2] * 5, [2.0, 1.0, 3.0, 4.0, 5.0], False, [0.279057, 0.358114, 0.2, 0.120943, 0.0418861]),
        (0.4, [0.2] * 5, [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0.324162, 0.472486, 0.175838, 0.027514, 0.0]),
        (0.4, [0.2] * 5, [2.0, 1.0, 3.0, 4.0, 5.0], False, [0.324162, 0.472486, 0.175838, 0.027514, 0.0]),
        (math.sqrt(0.8), [0.2] * 5, [-2.0, -1.0, -3.0, -4.0, -5.0], True, [0, 1.0, 0, 0, 0]),
    ],
)
def test_modified_chi_squared(radius, p, V, is_min, expected):
    q, _ = adjust(sddp.ModifiedChiSquared(radius), p, V, is_min)
    assert approx(q, expected, 1e-6), q


def test_modified_chi_squared_docstring():
    q, _ = adjust(sddp.ModifiedChiSquared(0.5), [0.1, 0.2, 0.3, 0.4], [5.0, 4.0, 6.0, 2.0], True)
    assert approx(q, [0.2267731382092775, 0.1577422872635742, 0.5958039891549808, 0.019680585372167547], 1e-9)
    q, _ = adjust(sddp.ModifiedChiSquared(0.5), [0.25] * 4, [5.0, 4.0, 6.0, 2.0], True)
    assert approx(q, [0.3333333333333333, 0.044658198738520394, 0.6220084679281462, 0.0], 1e-9)


def test_modified_chi_squared_default_to_expectation():
    q, a = adjust(sddp.ModifiedChiSquared(0.1), [0.4, 0.6], [1.0, 1.0], True)
    assert a == 0.0 and q == [0.4, 0.6]


def test_entropic_docstring_values():
    p = [0.1, 0.2, 0.3, 0.4]
    V = [5.0, 4.0, 6.0, 2.0]
    q, a = adjust(sddp.Entropic(0.1), p, V, True)
    assert math.isclose(a, -0.14333892665462006, rel_tol=1e-10)
    assert approx(q, [0.1100296362588547, 0.19911786395979578, 0.3648046623591841, 0.3260478374221655], 1e-12)
    q, a = adjust(sddp.Entropic(1.0), p, V, True)
    # The SDDP.jl docstring shows -0.12038 here (a stale doctest output); running
    # SDDP.adjust_probability in Julia 1.12 / SDDP v1.15.0 gives -0.6671608119259349.
    assert math.isclose(a, -0.6671608119259349, rel_tol=1e-10)
    assert approx(q, [0.09911045746726178, 0.07292139941460454, 0.8082304666305623, 0.019737676487571337], 1e-12)
    q, a = adjust(sddp.Entropic(10.0), p, V, True)
    assert math.isclose(a, -0.12038063114659443, rel_tol=1e-10)
    assert approx(q, [1.5133080886430772e-5, 1.374081618667918e-9, 0.999984865545032, 5.664386611687232e-18], 1e-12)


def test_entropic_zero_gamma_is_expectation():
    q, a = adjust(sddp.Entropic(0.0), [0.1, 0.9], [1.0, 2.0], True)
    assert a == 0.0 and q == [0.1, 0.9]


def test_wasserstein_docstring():
    supports = [sddp.Noise(t, p) for t, p in zip([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])]
    q, _ = adjust(
        sddp.Wasserstein(lambda x, y: abs(x.term - y.term), sddp.HiGHS, alpha=0.5),
        [0.1, 0.2, 0.3, 0.4],
        [5.0, 4.0, 6.0, 2.0],
        True,
        supports,
    )
    assert approx(q, [0.1, 0.1, 0.8, 0.0], 1e-6)
