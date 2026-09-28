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
    parameterize,
    set_numerical_difficulty_callback,
    simulate,
    termination_status,
    train,
    write_subproblem_to_file,
)
from sddp.binary_expansion import bincontract, binexpand
from sddp.biobjective import (
    initialize_biobjective_subproblem,
    set_biobjective_functions,
    set_trade_off_weight,
    train_biobjective,
)
from sddp.deterministic_equivalent import deterministic_equivalent
from sddp.graph import Graph, LinearGraph, MarkovianGraph, UnicyclicGraph, is_cyclic
from sddp.inner import (
    InnerBellmanFunction,
    InnerPolicyGraph,
    dp_vertices_from_visited_states,
    inner_dp,
    read_vertices_from_file,
    write_vertices_to_file,
)
from sddp.modeling_aids import (
    SimulatorSamplingScheme,
    allocate_support_budget,
    lattice_approximation,
    markovian_graph_from_simulator,
)
from sddp.msp_format import read_from_file as read_msp_format
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
from sddp.plugins.duality_handlers import (
    BanditDuality,
    ContinuousConicDuality,
    FixedDiscreteDuality,
    LagrangianDuality,
    StrengthenedConicDuality,
)
from sddp.plugins.forward_passes import (
    AlternativeForwardPass,
    AlternativePostIterationCallback,
    DefaultForwardPass,
    ImportanceSamplingForwardPass,
    LoggingForwardPass,
    RegularizedForwardPass,
    RevisitingForwardPass,
    RiskAdjustedForwardPass,
)
from sddp.plugins.local_improvement_search import BFGS, OuterApproximation
from sddp.plugins.parallel_schemes import Multiprocess, Serial, Threaded
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
    sample_noise,
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
from sddp.print import numerical_stability_report, write_log_to_csv
from sddp.solver.model import HiGHS, Model, OptimizerFactory, Sense, pyoptinterface_optimizer
from sddp.stochoptformat import (
    ValidationScenario,
    ValidationScenarios,
    read_from_file,
    write_to_file,
)
from sddp.stochoptformat import evaluate as evaluate_validation_scenarios
from sddp.value_function import (
    ValueFunction,
    plot_value_function,
)
from sddp.value_function import (
    evaluate as evaluate_value_function,
)
from sddp.visualization import (
    SpaghettiPlot,
    add_spaghetti,
    launch_dashboard,
    plot_graph,
    plot_value_function_html,
    publication_data,
    publication_plot,
    spaghetti_plot,
)

__version__ = "0.1.0"

__all__ = [
    "AlternativeForwardPass",
    "AlternativePostIterationCallback",
    "ImportanceSamplingForwardPass",
    "InnerBellmanFunction",
    "InnerPolicyGraph",
    "LoggingForwardPass",
    "Multiprocess",
    "parameterize",
    "sample_noise",
    "write_subproblem_to_file",
    "SimulatorSamplingScheme",
    "SpaghettiPlot",
    "ValidationScenario",
    "ValidationScenarios",
    "ValueFunction",
    "add_spaghetti",
    "allocate_support_budget",
    "bincontract",
    "binexpand",
    "dp_vertices_from_visited_states",
    "evaluate_validation_scenarios",
    "evaluate_value_function",
    "initialize_biobjective_subproblem",
    "inner_dp",
    "lattice_approximation",
    "launch_dashboard",
    "markovian_graph_from_simulator",
    "numerical_stability_report",
    "plot_graph",
    "plot_value_function",
    "plot_value_function_html",
    "read_from_file",
    "read_msp_format",
    "read_vertices_from_file",
    "set_biobjective_functions",
    "set_trade_off_weight",
    "train_biobjective",
    "write_log_to_csv",
    "write_to_file",
    "write_vertices_to_file",
    "AVaR",
    "BFGS",
    "BanditDuality",
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
    "FixedDiscreteDuality",
    "Graph",
    "HiGHS",
    "Historical",
    "InSampleMonteCarlo",
    "IterationLimit",
    "LagrangianDuality",
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
    "OuterApproximation",
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
    "StrengthenedConicDuality",
    "Subproblem",
    "Threaded",
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
    "publication_data",
    "publication_plot",
    "pyoptinterface_optimizer",
    "read_cuts_from_file",
    "set_numerical_difficulty_callback",
    "simulate",
    "spaghetti_plot",
    "termination_status",
    "train",
    "write_cuts_to_file",
]
