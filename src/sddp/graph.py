# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Policy graph structure. Ported from ``src/user_interface.jl`` (Graph) and ``src/cyclic.jl``."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from sddp.policy_graph import PolicyGraph

T = TypeVar("T", bound=Hashable)


class Graph(Generic[T]):
    """A directed graph of nodes with a distinguished root node.

    ``nodes[x]`` is the list of ``(child, probability)`` arcs leaving ``x``.
    Node labels may be any hashable Python object (ints, strings, tuples, ...)
    and are preserved exactly as given.
    """

    def __init__(self, root_node: T):
        self.root_node: T = root_node
        self.nodes: dict[T, list[tuple[T, float]]] = {root_node: []}
        self.belief_partition: list[list[T]] = []
        self.belief_lipschitz: list[list[float]] = []

    @classmethod
    def from_edges(
        cls,
        root_node: T,
        nodes: Sequence[T],
        edges: Sequence[tuple[tuple[T, T], float]],
        belief_partition: Sequence[Sequence[T]] = (),
        belief_lipschitz: Sequence[Sequence[float]] = (),
    ) -> Graph[T]:
        """Construct a graph from a root, a list of nodes and ``((parent, child), prob)`` edges."""
        g: Graph[T] = cls(root_node)
        for n in nodes:
            g.add_node(n)
        for (parent, child), p in edges:
            g.add_edge(parent, child, p)
        for s, lip in zip(belief_partition, belief_lipschitz):
            g.add_ambiguity_set(list(s), list(lip))
        return g

    def add_node(self, node: T) -> None:
        if node in self.nodes or node == self.root_node:
            raise ValueError(f"Node {node!r} already exists!")
        self.nodes[node] = []

    def _add_node_if_missing(self, node: T) -> None:
        if node not in self.nodes and node != self.root_node:
            self.add_node(node)

    def add_edge(self, parent: T, child: T, probability: float) -> None:
        if not (parent == self.root_node or parent in self.nodes):
            raise ValueError(f"Node {parent!r} does not exist.")
        if child not in self.nodes:
            raise ValueError(f"Node {child!r} does not exist.")
        if child == self.root_node:
            raise ValueError("Cannot have an edge entering the root node.")
        self.nodes[parent].append((child, float(probability)))

    def _add_to_or_create_edge(self, parent: T, child: T, probability: float) -> None:
        for i, (c, p) in enumerate(self.nodes[parent]):
            if c == child:
                self.nodes[parent][i] = (child, p + probability)
                return
        self.add_edge(parent, child, probability)

    def add_ambiguity_set(
        self, nodes: Sequence[T], lipschitz: float | Sequence[float] = 1e5
    ) -> None:
        """Add ``nodes`` to the belief partition with the given Lipschitz constant(s)."""
        if isinstance(lipschitz, (int, float)):
            lip = [float(lipschitz)] * len(nodes)
        else:
            lip = [float(x) for x in lipschitz]
        if any(x < 0.0 for x in lip):
            raise ValueError(f"Cannot provide negative Lipschitz constant: {lip}")
        if len(nodes) != len(lip):
            raise ValueError(
                "You must provide one Lipschitz constant for every element in the ambiguity set."
            )
        self.belief_partition.append(list(nodes))
        self.belief_lipschitz.append(lip)

    def validate(self) -> None:
        for node, children in self.nodes.items():
            if children:
                probability = sum(p for _, p in children)
                if not (-1e-8 <= probability):
                    raise ValueError(
                        f"Probability on edges leaving node {node!r} sum to {probability}, "
                        "but this must be `>= 0.0`"
                    )
        if self.belief_partition:
            union: set[T] = set()
            for s in self.belief_partition:
                union.update(s)
            if self.root_node in union:
                raise ValueError(
                    f"Belief partition {self.belief_partition} cannot contain the root node "
                    f"{self.root_node!r}."
                )
            if len(self.nodes) - 1 != len(union):
                raise ValueError(
                    f"Belief partition {self.belief_partition} does not form a valid "
                    "partition of the nodes in the graph."
                )

    def __repr__(self) -> str:
        lines = ["Root", f" {self.root_node}", "Nodes"]
        nodes = [n for n in self.nodes if n != self.root_node]
        try:
            nodes = sorted(nodes)  # type: ignore[type-var]
        except TypeError:
            pass
        if not nodes:
            lines.append(" {}")
        lines.extend(f" {n}" for n in nodes)
        lines.append("Arcs")
        arcs = []
        for node in [self.root_node] + nodes:
            for child, p in self.nodes[node]:
                arcs.append(f" {node} => {child} w.p. {p}")
        lines.extend(arcs or [" {}"])
        if self.belief_partition:
            lines.append("Partitions")
            for s in self.belief_partition:
                lines.append(" {" + ", ".join(str(x) for x in s) + "}")
        return "\n".join(lines)


def LinearGraph(stages: int) -> Graph[int]:
    """A linear graph ``0 -> 1 -> 2 -> ... -> stages`` with root node ``0``."""
    if stages < 1:
        raise ValueError("You must create a LinearGraph with `stages >= 1`.")
    edges = [((t - 1, t), 1.0) for t in range(1, stages + 1)]
    return Graph.from_edges(0, list(range(1, stages + 1)), edges)


def MarkovianGraph(
    transition_matrices: Sequence[Any] | None = None,
    *,
    stages: int | None = None,
    transition_matrix: Any | None = None,
    root_node_transition: Sequence[float] | None = None,
) -> Graph[tuple[int, int]]:
    """A Markovian graph. Nodes are ``(stage, markov_state)`` tuples, 1-based like SDDP.jl.

    Either pass ``transition_matrices`` (a list; the first has shape ``(1, N)``), or the
    keyword form ``stages=, transition_matrix=, root_node_transition=``.
    """
    if transition_matrices is None:
        if stages is None or transition_matrix is None or root_node_transition is None:
            raise ValueError(
                "Provide `transition_matrices`, or all of `stages`, `transition_matrix`, "
                "`root_node_transition`."
            )
        tm = np.asarray(transition_matrix, dtype=float)
        if tm.shape[0] != tm.shape[1] or len(root_node_transition) != tm.shape[0]:
            raise ValueError("transition_matrix must be square and match root_node_transition.")
        transition_matrices = [np.asarray(root_node_transition, dtype=float).reshape(1, -1)] + [
            tm for _ in range(stages - 1)
        ]
    mats = [np.atleast_2d(np.asarray(m, dtype=float)) for m in transition_matrices]
    if mats[0].shape[0] != 1:
        raise ValueError(
            f"Expected the first transition matrix to be of size (1, N). It is of size "
            f"{mats[0].shape}."
        )
    nodes: list[tuple[int, int]] = []
    edges: list[tuple[tuple[tuple[int, int], tuple[int, int]], float]] = []
    for stage, transition in enumerate(mats, start=1):
        if not np.all(transition >= 0.0):
            raise ValueError("Entries in the transition matrix must be non-negative.")
        row_sums = transition.sum(axis=1)
        if not np.all((0.0 - 1e-8 <= row_sums) & (row_sums <= 1.0 + 1e-8)):
            raise ValueError("Rows in the transition matrix must sum to between 0.0 and 1.0.")
        if stage > 1 and mats[stage - 2].shape[1] != transition.shape[0]:
            raise ValueError(f"Transition matrix for stage {stage} is the wrong size.")
        for markov_state in range(1, transition.shape[1] + 1):
            nodes.append((stage, markov_state))
        for markov_state in range(1, transition.shape[1] + 1):
            for last_markov_state in range(1, transition.shape[0] + 1):
                p = float(transition[last_markov_state - 1, markov_state - 1])
                edges.append((((stage - 1, last_markov_state), (stage, markov_state)), p))
    return Graph.from_edges((0, 1), nodes, edges)


def UnicyclicGraph(discount_factor: float, num_nodes: int = 1) -> Graph[int]:
    """``num_nodes`` nodes in a single cycle, continuing with probability ``discount_factor``."""
    assert 0 < discount_factor < 1
    assert num_nodes > 0
    graph = LinearGraph(num_nodes)
    graph.add_edge(num_nodes, 1, discount_factor)
    return graph


def is_cyclic(G: PolicyGraph) -> bool:
    """Tarjan's strongly connected components, stopping at the first cycle (``src/cyclic.jl``)."""
    index_counter = 0
    S: list = []
    low_link: dict = {}
    index: dict = {}
    on_stack: dict = {}

    def strong_connect(v: Any) -> bool:
        nonlocal index_counter
        index[v] = index_counter
        low_link[v] = index_counter
        index_counter += 1
        S.append(v)
        on_stack[v] = True
        for child in G[v].children:
            w = child.term
            if v == w:
                return True  # Type I: self loop
            if w not in index:
                if strong_connect(w):
                    return True
                low_link[v] = min(low_link[v], low_link[w])
            elif on_stack[w]:
                low_link[v] = min(low_link[v], index[w])
        if low_link[v] == index[v]:
            scc = []
            w = G.root_node
            while v != w:
                w = S.pop()
                on_stack[w] = False
                scc.append(w)
            if len(scc) > 1:
                return True  # Type II: SCC with more than one node
        return False

    for v in G.nodes:
        if v not in index and strong_connect(v):
            return True
    return False
