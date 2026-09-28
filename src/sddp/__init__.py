# Copyright (c) 2017-26: Oscar Dowson and SDDP.jl contributors.
# Copyright (c) 2026: sddp-py contributors (Python port).
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at http://mozilla.org/MPL/2.0/.
"""sddp: a Python port of SDDP.jl (Stochastic Dual Dynamic Programming)."""

from sddp.algorithm import (
    DecisionRule,
    calculate_bound,
    confidence_interval,
    evaluate,
    set_numerical_difficulty_callback,
    simulate,
    termination_status,
    train,
)
from sddp.deterministic_equivalent import deterministic_equivalent
from sddp.graph import Graph, LinearGraph, MarkovianGraph, UnicyclicGraph, is_cyclic
from sddp.plugins.backward_sampling_schemes import CompleteSampler, MonteCarloSampler
from sddp.plugins.bellman_functions import (
    MULTI_CUT,
    SINGLE_CUT,
    BellmanFunction,
    CutType,
    add_all_cuts,
    read_cuts_from_file,
    write_cuts_to_file,
)
from sddp.plugins.duality_handlers import ContinuousConicDuality
from sddp.plugins.forward_passes import (
    DefaultForwardPass,
    RegularizedForwardPass,
    RevisitingForwardPass,
    RiskAdjustedForwardPass,
)
from sddp.plugins.parallel_schemes import Serial
from sddp.plugins.risk_measures import (
    AVaR,
    ConvexCombination,
    CVaR,
    EAVaR,
    Entropic,
    Expectation,
    ModifiedChiSquared,
    Wasserstein,
    WorstCase,
)
from sddp.plugins.sampling_schemes import (
    Historical,
    InSampleMonteCarlo,
    OutOfSampleMonteCarlo,
    PSRSamplingScheme,
)
from sddp.plugins.stopping_rules import (
    BoundStalling,
    FirstStageStoppingRule,
    IterationLimit,
    SimulationStoppingRule,
    Statistical,
    StoppingChain,
    TimeLimit,
)
from sddp.policy_graph import (
    LinearPolicyGraph,
    MarkovianPolicyGraph,
    Node,
    Noise,
    PolicyGraph,
    State,
    StateValue,
    Subproblem,
)
from sddp.solver.model import HiGHS, Model, OptimizerFactory, Sense, pyoptinterface_optimizer

__version__ = "0.1.0"

__all__ = [
    "AVaR",
    "BellmanFunction",
    "BoundStalling",
    "CVaR",
    "CompleteSampler",
    "ContinuousConicDuality",
    "ConvexCombination",
    "CutType",
    "DecisionRule",
    "DefaultForwardPass",
    "EAVaR",
    "Entropic",
    "Expectation",
    "FirstStageStoppingRule",
    "Graph",
    "HiGHS",
    "Historical",
    "InSampleMonteCarlo",
    "IterationLimit",
    "LinearGraph",
    "LinearPolicyGraph",
    "MULTI_CUT",
    "MarkovianGraph",
    "MarkovianPolicyGraph",
    "Model",
    "ModifiedChiSquared",
    "MonteCarloSampler",
    "Node",
    "Noise",
    "OptimizerFactory",
    "OutOfSampleMonteCarlo",
    "PSRSamplingScheme",
    "PolicyGraph",
    "RegularizedForwardPass",
    "RevisitingForwardPass",
    "RiskAdjustedForwardPass",
    "SINGLE_CUT",
    "Sense",
    "Serial",
    "SimulationStoppingRule",
    "State",
    "StateValue",
    "Statistical",
    "StoppingChain",
    "Subproblem",
    "TimeLimit",
    "UnicyclicGraph",
    "Wasserstein",
    "WorstCase",
    "add_all_cuts",
    "calculate_bound",
    "confidence_interval",
    "deterministic_equivalent",
    "evaluate",
    "is_cyclic",
    "pyoptinterface_optimizer",
    "read_cuts_from_file",
    "set_numerical_difficulty_callback",
    "simulate",
    "termination_status",
    "train",
    "write_cuts_to_file",
]
