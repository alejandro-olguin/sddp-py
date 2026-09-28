# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""Backward-pass sampling schemes. Ported from ``src/plugins/backward_sampling_schemes.jl``."""

from __future__ import annotations

import random

from sddp.plugins.base import BackwardSamplingScheme
from sddp.plugins.sampling_schemes import sample_noise
from sddp.policy_graph import Node, Noise


class CompleteSampler(BackwardSamplingScheme):
    """Use every noise term of the node on the backward pass."""

    def sample_backward_noise_terms(self, node: Node, rng: random.Random) -> list[Noise]:
        return node.noise_terms


class MonteCarloSampler(BackwardSamplingScheme):
    """Sample ``number_of_samples`` noise terms with replacement on the backward pass."""

    def __init__(self, number_of_samples: int):
        self.number_of_samples = number_of_samples

    def sample_backward_noise_terms(self, node: Node, rng: random.Random) -> list[Noise]:
        prob = 1 / self.number_of_samples
        return [Noise(sample_noise(node.noise_terms, rng), prob) for _ in range(self.number_of_samples)]
