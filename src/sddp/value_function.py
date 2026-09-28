# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Stand-alone value functions. Ported from ``src/visualization/value_functions.jl``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sddp.plugins.bellman_functions import ConvexApproximation
from sddp.policy_graph import Node, PolicyGraph
from sddp.solver.model import Model, Sense, Variable


class ValueFunction:
    """A copy of a node's value function ``V(x, b, y)`` in its own solver model.

    Build once (cost O(number of cuts)) and query repeatedly with :func:`evaluate`.
    """

    def __init__(self, model_or_node: PolicyGraph | Node, node: Any = None):
        if isinstance(model_or_node, PolicyGraph):
            if node is None:
                raise ValueError("ValueFunction(model, node=...) requires a node index")
            src = model_or_node[node]
        else:
            src = model_or_node
        b = src.bellman_function
        sense = src.model.objective_sense
        m = Model(src.optimizer)
        m.set_objective_sense(sense)
        self.index = src.index
        self.model = m
        self.states: dict[str, Variable] = {key: m.add_variable(key) for key in src.states}
        self.objective_state: tuple[Variable, ...] | None = None
        if src.objective_state is not None:
            self.objective_state = tuple(
                m.add_variable(
                    f"_objective_state_{i}",
                    lb=src.model.lower_bound(mu),
                    ub=src.model.upper_bound(mu),
                )
                for i, mu in enumerate(src.objective_state.mu)
            )
        self.belief_state: dict[Any, Variable] | None = None
        if src.belief_state is not None:
            self.belief_state = {
                key: m.add_variable(
                    f"_belief_{key}", lb=src.model.lower_bound(mu), ub=src.model.upper_bound(mu)
                )
                for key, mu in src.belief_state.mu.items()
            }
        self.theta = _add_to_value_function(
            m, self.states, self.objective_state, self.belief_state, b.global_theta, "V", src
        )
        local_thetas = [
            _add_to_value_function(
                m, self.states, self.objective_state, self.belief_state, local, f"v{i + 1}", src
            )
            for i, local in enumerate(b.local_thetas)
        ]
        for risk_set in b.risk_set_cuts:
            terms = [(self.theta, 1.0)] + [(v, -float(p)) for p, v in zip(risk_set, local_thetas)]
            if sense is Sense.MIN:
                m.add_constraint_normalized(Model.expression(terms), ">=", 0.0)
            else:
                m.add_constraint_normalized(Model.expression(terms), "<=", 0.0)

    def __repr__(self) -> str:
        return f"A value function for node {self.index}"


def _add_to_value_function(
    model: Model,
    states: dict[str, Variable],
    objective_state: tuple[Variable, ...] | None,
    belief_state: dict[Any, Variable] | None,
    approx: ConvexApproximation,
    theta_name: str,
    src: Node,
) -> Variable:
    theta = model.add_variable(theta_name)
    src_model = src.model
    if model.objective_sense is Sense.MIN:
        if src_model.has_lower_bound(approx.theta):
            model.set_lower_bound(theta, src_model.lower_bound(approx.theta))
    else:
        if src_model.has_upper_bound(approx.theta):
            model.set_upper_bound(theta, src_model.upper_bound(approx.theta))
    for cut in approx.cuts:
        # theta >= intercept + Σ coef x - Σ y μ_obj - Σ b μ_belief  (min; <= for max)
        terms = [(theta, 1.0)] + [(states[k], -c) for k, c in cut.coefficients.items()]
        if objective_state is not None:
            assert cut.obj_y is not None
            terms += [(mu, y) for y, mu in zip(cut.obj_y, objective_state)]
        if belief_state is not None:
            assert cut.belief_y is not None
            terms += [(mu, cut.belief_y[k]) for k, mu in belief_state.items()]
        expr = Model.expression(terms)
        if model.objective_sense is Sense.MIN:
            model.add_constraint_normalized(expr, ">=", cut.intercept)
        else:
            model.add_constraint_normalized(expr, "<=", cut.intercept)
    return theta


def evaluate(
    V: ValueFunction,
    point: dict[str, float],
    objective_state: float | Sequence[float] | None = None,
    belief_state: dict[Any, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate ``V`` at ``point``; returns ``(height, {state: subgradient})``.

    ``objective_state`` (scalar or tuple) and ``belief_state`` are required when the node
    has those states.
    """
    m = V.model
    for state, val in point.items():
        m.fix(V.states[str(state)], float(val))
    terms = [(V.theta, 1.0)]
    if V.objective_state is not None:
        assert objective_state is not None, "objective_state is required for this value function"
        ys = (
            (float(objective_state),)
            if isinstance(objective_state, (int, float))
            else tuple(float(v) for v in objective_state)
        )
        terms += [(x, y) for y, x in zip(ys, V.objective_state)]
    if V.belief_state is not None:
        assert belief_state is not None, "belief_state is required for this value function"
        terms += [(x, float(belief_state[key])) for key, x in V.belief_state.items()]
    m.set_objective(Model.expression(terms), m.objective_sense)
    m.optimize()
    obj = m.objective_value()
    duals = {key: m.reduced_cost(var) for key, var in V.states.items()}
    return obj, duals


def plot_value_function(
    V: ValueFunction,
    ax: Any = None,
    objective_state: Any = None,
    belief_state: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot ``V`` over one or two state variables given as ranges (others fixed at scalars).

    Example: ``plot_value_function(V, volume=np.linspace(0, 200, 50))``.
    """
    import matplotlib.pyplot as plt

    fixed = {k: float(v) for k, v in kwargs.items() if isinstance(v, (int, float))}
    varying = {k: list(v) for k, v in kwargs.items() if not isinstance(v, (int, float))}
    if len(varying) == 1:
        ((k, xs),) = varying.items()
        ys = [
            evaluate(
                V, {**fixed, k: x}, objective_state=objective_state, belief_state=belief_state
            )[0]
            for x in xs
        ]
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_xlabel(k)
        ax.set_ylabel(f"V (node {V.index})")
        return ax
    if len(varying) == 2:
        import numpy as np

        (k1, xs), (k2, ys) = varying.items()
        Z = np.array(
            [
                [
                    evaluate(
                        V,
                        {**fixed, k1: x, k2: y},
                        objective_state=objective_state,
                        belief_state=belief_state,
                    )[0]
                    for x in xs
                ]
                for y in ys
            ]
        )
        if ax is None:
            _, ax = plt.subplots()
        cs = ax.contourf(xs, ys, Z, levels=20)
        ax.figure.colorbar(cs, ax=ax)
        ax.set_xlabel(k1)
        ax.set_ylabel(k2)
        return ax
    raise ValueError(
        f"Can only plot 1- or 2-dimensional value functions. You provided {len(varying)}."
    )
