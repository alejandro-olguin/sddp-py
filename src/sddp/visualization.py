# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Publication and spaghetti plots via matplotlib (``src/visualization/*.jl``)."""

from __future__ import annotations

import json as _json
import math
import os as _os
import tempfile as _tempfile
import webbrowser as _webbrowser
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


# ---------------------------------------------------------------------------
# HTML outputs (templates copied from SDDP.jl's src/visualization, MPL-2.0)
# ---------------------------------------------------------------------------
_ASSETS = _os.path.join(_os.path.dirname(__file__), "assets")


def _fill_template(
    dest: str, replacements: dict[str, str], template: str, launch: bool = False
) -> str:
    with open(_os.path.join(_ASSETS, template), encoding="utf-8") as io:
        s = io.read()
    for k, v in replacements.items():
        s = s.replace(k, v)
    with open(dest, "w", encoding="utf-8") as io:
        io.write(s)
    if launch:
        _webbrowser.open("file://" + _os.path.abspath(dest))
    return dest


def _tmp_html() -> str:
    fd, path = _tempfile.mkstemp(suffix=".html")
    _os.close(fd)
    return path


class SpaghettiPlot:
    """Interactive d3 spaghetti plot (``SDDP.SpaghettiPlot``): one line per replication."""

    def __init__(self, simulations: Simulations):
        self.simulations = simulations
        self.data: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return (
            f"A spaghetti plot with {len(self.simulations)} scenarios and "
            f"{len(self.simulations[0])} stages."
        )

    def add_spaghetti(
        self,
        data_function: Callable[[dict[str, Any]], float],
        xlabel: str = "Stages",
        ylabel: str = "",
        cumulative: bool = False,
        title: str = "",
        interpolate: str = "linear",
        ymin: Any = "",
        ymax: Any = "",
    ) -> None:
        plot_dict: dict[str, Any] = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": title,
            "cumulative": cumulative,
            "interpolate": interpolate,
            "ymin": ymin,
            "ymax": ymax,
            "data": [],
        }
        for scenario in self.simulations:
            series: list[float] = []
            value = 0.0
            for stage in scenario:
                y = float(data_function(stage))
                value = value + y if cumulative else y
                series.append(value)
            plot_dict["data"].append(series)
        self.data.append(plot_dict)

    def plot(self, filename: str | None = None, open: bool = False) -> str:
        """Write the HTML file and return its path."""
        filename = filename or _tmp_html()
        with (
            open_file(_os.path.join(_ASSETS, "d3.v3.min.js")) as d3,
            open_file(_os.path.join(_ASSETS, "spaghetti_plot.js")) as js,
        ):
            return _fill_template(
                filename,
                {
                    "<!--DATA-->": _json.dumps(self.data, separators=(",", ":")),
                    "<!--D3.JS-->": d3.read(),
                    "<!--SPAGHETTI_PLOT.JS-->": js.read(),
                },
                "spaghetti_plot.html",
                launch=open,
            )


def open_file(path: str) -> Any:
    return open(path, encoding="utf-8")


def add_spaghetti(
    data_function: Callable[[dict[str, Any]], float], plt: SpaghettiPlot, **kwargs: Any
) -> None:
    plt.add_spaghetti(data_function, **kwargs)


def plot_graph(model_or_graph: Any, filename: str | None = None, open: bool = False) -> str:
    """Draw the policy graph structure with Cytoscape.js (``SDDP.plot(model)``)."""
    from sddp.policy_graph import PolicyGraph

    filename = filename or _tmp_html()
    data: list[str] = []
    if isinstance(model_or_graph, PolicyGraph):
        model = model_or_graph
        meta = "\\n".join(f"{k} = {v}" for k, v in model.initial_root_state.items())
        data.append(f"{{data: {{id: '{model.root_node}', shape: 'ellipse', meta: '{meta}'}}}}")
        names = list(model.nodes)
        try:
            names = sorted(names)
        except TypeError:
            pass
        for name in names:
            n_terms = len(model[name].noise_terms)
            meta = f"Node: {name}\\nNoise terms: {n_terms}"
            noise = ", has_noise: true" if n_terms > 1 else ""
            data.append(f"{{data: {{id: '{name}', meta: '{meta}'{noise}}}}}")
        edge_id = 0
        for child in model.root_children:
            edge_id += 1
            meta = f"From: {model.root_node}\\nTo: {child.term}\\nProbablity: {child.probability}"
            data.append(
                f"{{data: {{id: 'edge_{edge_id}', source: '{model.root_node}', "
                f"target: '{child.term}', meta: '{meta}'}}}}"
            )
        for name in names:
            for child in model[name].children:
                edge_id += 1
                meta = f"From: {name}\\nTo: {child.term}\\nProbablity: {child.probability}"
                data.append(
                    f"{{data: {{id: 'edge_{edge_id}', source: '{name}', "
                    f"target: '{child.term}', meta: '{meta}'}}}}"
                )
    else:
        graph = model_or_graph
        data.append(f"{{data: {{id: '{graph.root_node}', shape: 'ellipse'}}}}")
        names = list(graph.nodes)
        try:
            names = sorted(names)
        except TypeError:
            pass
        for name in names:
            if name != graph.root_node:
                data.append(f"{{data: {{id: '{name}', meta: 'Node: {name}'}}}}")
        edge_id = 0
        for name in names:
            for child, probability in graph.nodes[name]:
                edge_id += 1
                meta = f"From: {name}\\nTo: {child}\\nProbablity: {probability}"
                data.append(
                    f"{{data: {{id: 'edge_{edge_id}', source: '{name}', "
                    f"target: '{child}', meta: '{meta}'}}}}"
                )
    return _fill_template(filename, {"<!--DATA-->": ",\n".join(data)}, "graph.html", launch=open)


def plot_value_function_html(
    V: Any,
    filename: str | None = None,
    open: bool = False,
    objective_state: Any = None,
    belief_state: Any = None,
    **kwargs: Any,
) -> str:
    """Interactive 1-D line / 2-D surface of a value function (``SDDP.plot(V; x=..., y=...)``)."""
    from sddp.value_function import evaluate

    filename = filename or _tmp_html()
    fixed = {k: float(v) for k, v in kwargs.items() if isinstance(v, (int, float))}
    varying = {k: list(v) for k, v in kwargs.items() if not isinstance(v, (int, float))}
    if len(varying) == 1:
        ((k, xs),) = varying.items()
        x = [float(v) for v in xs]
        y = [
            evaluate(
                V, {**fixed, k: xi}, objective_state=objective_state, belief_state=belief_state
            )[0]
            for xi in x
        ]
        z: list[float] = []
    elif len(varying) == 2:
        (k1, xs), (k2, ys) = varying.items()
        x = [float(xi) for _ in ys for xi in xs]
        y = [float(yi) for yi in ys for _ in xs]
        z = [
            evaluate(
                V,
                {**fixed, k1: xi, k2: yi},
                objective_state=objective_state,
                belief_state=belief_state,
            )[0]
            for yi in ys
            for xi in xs
        ]
    else:
        raise ValueError(
            f"Can only plot 1- or 2-dimensional value functions. You provided {len(varying)}."
        )
    return _fill_template(
        filename,
        {"<!--X-->": _json.dumps(x), "<!--Y-->": _json.dumps(y), "<!--Z-->": _json.dumps(z)},
        "value_functions.html",
        launch=open,
    )


def launch_dashboard(port: int = 8000, open: bool = True) -> Callable[..., Any]:
    """Serve training logs as server-sent events for ``assets/dashboard.html``.

    Returns the ``dashboard_callback(log, close_flag)`` that ``train`` calls per iteration.
    """
    import http.server
    import queue
    import threading

    events: queue.Queue = queue.Queue()
    closing = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence
            return

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while not closing.is_set():
                try:
                    log = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                if log is None:
                    break
                payload = _json.dumps(
                    {
                        "iteration": log.iteration,
                        "bound": log.bound,
                        "simulation": log.simulation_value,
                        "time": log.time,
                        "solves": log.total_solves,
                    }
                )
                try:
                    self.wfile.write(f"event: iteration\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if open:
        _webbrowser.open("file://" + _os.path.join(_ASSETS, "dashboard.html"))

    def dashboard_callback(log: Any, close_flag: bool) -> None:
        if close_flag:
            closing.set()
            events.put(None)
            server.shutdown()
            server.server_close()
        elif log is not None:
            events.put(log)

    dashboard_callback.port = port  # type: ignore[attr-defined]
    return dashboard_callback
