# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Publication and spaghetti plots via matplotlib (``src/visualization/*.jl``)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Simulations = Sequence[Sequence[dict[str, Any]]]


def publication_data(
    dataset: Simulations,
    quantiles: Sequence[float],
    stage_function: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    """A ``(len(quantiles), max_stages)`` array of per-stage quantiles of ``stage_function``."""
    max_stages = max(len(d) for d in dataset)
    output = np.full((len(quantiles), max_stages), math.nan)
    for t in range(max_stages):
        stage_data = []
        for i, d in enumerate(dataset):
            if len(d) <= t:
                continue
            s = float(stage_function(d[t]))
            if not math.isfinite(s):
                raise ValueError(
                    f"Unable to plot `publication_plot` because stage {t + 1} of replication "
                    f"{i + 1} contains data that is not finite. Got: {s}"
                )
            stage_data.append(s)
        output[:, t] = np.quantile(stage_data, quantiles)
    return output


def publication_plot(
    simulations: Simulations,
    data_function: Callable[[dict[str, Any]], float],
    quantile: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    ax: Any = None,
    title: str | None = None,
    xlabel: str = "Stage",
    ylabel: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot nested quantile ribbons of ``data_function`` over stages (``SDDP.publication_plot``)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    q = sorted(quantile)
    data = publication_data(simulations, q, data_function)
    x = np.arange(1, data.shape[1] + 1)
    n = data.shape[0]
    for i in range(n // 2):
        mu = 0.5 * (data[i, :] + data[n - 1 - i, :])
        r = data[n - 1 - i, :] - mu
        ax.fill_between(x, mu - r, mu + r, alpha=0.2, color=kwargs.get("color", "C0"), linewidth=0)
    if n % 2 == 1:
        ax.plot(x, data[n // 2, :], color=kwargs.get("color", "C0"))
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def spaghetti_plot(
    simulations: Simulations,
    data_function: Callable[[dict[str, Any]], float],
    ax: Any = None,
    title: str | None = None,
    xlabel: str = "Stage",
    ylabel: str | None = None,
    **kwargs: Any,
) -> Any:
    """One line per replication (``SDDP.SpaghettiPlot`` rendered with matplotlib)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    for sim in simulations:
        y = [float(data_function(s)) for s in sim]
        ax.plot(
            range(1, len(y) + 1), y, alpha=kwargs.get("alpha", 0.3), color=kwargs.get("color", "C0")
        )
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
