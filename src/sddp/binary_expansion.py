# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Binary expansion helpers. Ported from ``src/binary_expansion.jl``."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _bitsrequired(x: int) -> int:
    return int(math.floor(math.log(x) / math.log(2))) + 1


def binexpand(x: int | float, maximum: int | float, eps: float | None = None) -> list[int]:
    """Binary coefficients of ``x``, with as many bits as ``maximum`` needs.

    For floats, ``x`` and ``maximum`` are first divided by ``eps`` (default 0.1) and rounded.
    """
    if isinstance(x, float) or isinstance(maximum, float) or eps is not None:
        eps = 0.1 if eps is None else eps
        assert eps > 0
        return binexpand(round(x / eps), round(maximum / eps))
    if x < 0:
        raise ValueError(
            "Cannot perform binary expansion on a negative number."
            "Initial values of state variables must be nonnegative."
        )
    if maximum <= 0:
        raise ValueError(
            "Cannot perform binary expansion on zero-length "
            "vector. Upper bounds of state variables must be positive."
        )
    y = [0] * _bitsrequired(maximum)
    for i in range(len(y), 0, -1):
        k = 2 ** (i - 1)
        if x >= k:
            y[i - 1] = 1
            x -= k
    if x > 0:
        raise ValueError(f"Unable to expand binary. Overflow of {x}.")
    return y


def bincontract(y: Sequence[int | float], eps: float | None = None) -> int | float:
    """``Σᵢ 2ⁱ⁻¹ yᵢ`` (times ``eps`` if given)."""
    x = 0
    for i, yi in enumerate(y):
        x += 2**i * yi
    return x if eps is None else x * eps
