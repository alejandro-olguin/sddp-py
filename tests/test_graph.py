import pytest

import sddp


def test_linear_graph():
    g = sddp.LinearGraph(3)
    assert g.root_node == 0
    assert g.nodes[0] == [(1, 1.0)]
    assert g.nodes[2] == [(3, 1.0)]
    assert g.nodes[3] == []
    assert "0 => 1 w.p. 1.0" in repr(g)


def test_markovian_graph():
    g = sddp.MarkovianGraph([[[1.0]], [[0.5, 0.5]], [[0.8, 0.2], [0.2, 0.8]]])
    assert g.root_node == (0, 1)
    assert g.nodes[(1, 1)] == [((2, 1), 0.5), ((2, 2), 0.5)]
    assert g.nodes[(2, 2)] == [((3, 1), 0.2), ((3, 2), 0.8)]
    g2 = sddp.MarkovianGraph(
        stages=3, transition_matrix=[[0.8, 0.2], [0.2, 0.8]], root_node_transition=[0.5, 0.5]
    )
    assert g2.nodes[(0, 1)] == [((1, 1), 0.5), ((1, 2), 0.5)]
    assert len(g2.nodes) == 7


def test_unicyclic_and_is_cyclic():
    g = sddp.UnicyclicGraph(0.9, num_nodes=2)
    assert g.nodes[2] == [(1, 0.9)]

    def builder(sp, node):
        x = sp.add_state("x", initial_value=0.0)
        sp.add_constraint(x.out == x.in_)
        sp.set_stage_objective(1.0)

    model = sddp.PolicyGraph(builder, g, lower_bound=0.0)
    assert sddp.is_cyclic(model)
    assert not sddp.is_cyclic(sddp.LinearPolicyGraph(builder, stages=2, lower_bound=0.0))


def test_graph_errors():
    g = sddp.Graph(0)
    with pytest.raises(ValueError):
        g.add_edge(0, 1, 1.0)
    g.add_node(1)
    with pytest.raises(ValueError):
        g.add_node(1)
    with pytest.raises(ValueError):
        g.add_edge(1, 0, 1.0)
    g.add_edge(0, 1, -1.0)
    with pytest.raises(ValueError):
        g.validate()


def test_policy_graph_requires_bound_and_initial_value():
    def builder(sp, t):
        sp.add_state("x", initial_value=0.0)

    with pytest.raises(ValueError):
        sddp.LinearPolicyGraph(builder, stages=1)
    with pytest.raises(ValueError):
        sddp.LinearPolicyGraph(builder, stages=1, sense="Max")

    def bad(sp, t):
        sp.add_state("x", initial_value=5.0, lb=0.0, ub=1.0)

    with pytest.raises(ValueError, match="violates upper bound"):
        sddp.LinearPolicyGraph(bad, stages=1, lower_bound=0.0)
