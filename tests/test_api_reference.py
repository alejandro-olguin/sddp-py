"""Every entry of https://sddp.dev/stable/apireference/ (SDDP.jl v1.15.0) has a Python home.

The headings below are the ``## `name``` sections of ``docs/src/apireference.md`` in SDDP.jl.
Each maps to the attribute path in :mod:`sddp` that provides the equivalent functionality
(see the "API mapping" table in README.md). Keeping this list in sync with SDDP.jl makes an
accidental removal from the public surface a test failure.
"""

from __future__ import annotations

import pytest

import sddp
import sddp.plugins.base as base

API_REFERENCE: dict[str, object] = {
    "Graph": sddp.Graph,
    "add_node": sddp.Graph.add_node,
    "add_edge": sddp.Graph.add_edge,
    "add_ambiguity_set": sddp.Graph.add_ambiguity_set,
    "LinearGraph": sddp.LinearGraph,
    "MarkovianGraph": (sddp.MarkovianGraph, sddp.markovian_graph_from_simulator),
    "UnicyclicGraph": sddp.UnicyclicGraph,
    "LinearPolicyGraph": sddp.LinearPolicyGraph,
    "MarkovianPolicyGraph": sddp.MarkovianPolicyGraph,
    "PolicyGraph": sddp.PolicyGraph,
    "@stageobjective": sddp.Subproblem.set_stage_objective,
    "parameterize": (sddp.Subproblem.parameterize, sddp.parameterize),
    "add_objective_state": sddp.Subproblem.add_objective_state,
    "objective_state": sddp.Subproblem.objective_state,
    "Noise": sddp.Noise,
    "numerical_stability_report": sddp.numerical_stability_report,
    "train": sddp.train,
    "termination_status": sddp.termination_status,
    "write_cuts_to_file": sddp.write_cuts_to_file,
    "read_cuts_from_file": sddp.read_cuts_from_file,
    "write_log_to_csv": sddp.write_log_to_csv,
    "set_numerical_difficulty_callback": sddp.set_numerical_difficulty_callback,
    "AbstractStoppingRule": base.StoppingRule,
    "stopping_rule_status": base.StoppingRule.stopping_rule_status,
    "convergence_test": base.StoppingRule.convergence_test,
    "IterationLimit": sddp.IterationLimit,
    "TimeLimit": sddp.TimeLimit,
    "Statistical": sddp.Statistical,
    "BoundStalling": sddp.BoundStalling,
    "StoppingChain": sddp.StoppingChain,
    "SimulationStoppingRule": sddp.SimulationStoppingRule,
    "FirstStageStoppingRule": sddp.FirstStageStoppingRule,
    "AbstractSamplingScheme": base.SamplingScheme,
    "sample_scenario": base.SamplingScheme.sample_scenario,
    "InSampleMonteCarlo": sddp.InSampleMonteCarlo,
    "OutOfSampleMonteCarlo": sddp.OutOfSampleMonteCarlo,
    "Historical": sddp.Historical,
    "PSRSamplingScheme": sddp.PSRSamplingScheme,
    "SimulatorSamplingScheme": sddp.SimulatorSamplingScheme,
    "AbstractParallelScheme": base.ParallelScheme,
    "Serial": sddp.Serial,
    "Threaded": sddp.Threaded,
    "Asynchronous": sddp.Multiprocess,  # process-based equivalent of SDDP.Asynchronous
    "AbstractForwardPass": base.ForwardPass,
    "DefaultForwardPass": sddp.DefaultForwardPass,
    "RevisitingForwardPass": sddp.RevisitingForwardPass,
    "RiskAdjustedForwardPass": sddp.RiskAdjustedForwardPass,
    "AlternativeForwardPass": sddp.AlternativeForwardPass,
    "AlternativePostIterationCallback": sddp.AlternativePostIterationCallback,
    "RegularizedForwardPass": sddp.RegularizedForwardPass,
    "ImportanceSamplingForwardPass": sddp.ImportanceSamplingForwardPass,
    "LoggingForwardPass": sddp.LoggingForwardPass,
    "AbstractRiskMeasure": base.RiskMeasure,
    "adjust_probability": base.RiskMeasure.adjust_probability,
    "Expectation": sddp.Expectation,
    "WorstCase": sddp.WorstCase,
    "AVaR": sddp.AVaR,
    "CVaR": sddp.CVaR,
    "ConvexCombination": sddp.ConvexCombination,
    "EAVaR": sddp.EAVaR,
    "ModifiedChiSquared": sddp.ModifiedChiSquared,
    "Entropic": sddp.Entropic,
    "Wasserstein": sddp.Wasserstein,
    "AbstractDualityHandler": base.DualityHandler,
    "ContinuousConicDuality": sddp.ContinuousConicDuality,
    "LagrangianDuality": sddp.LagrangianDuality,
    "StrengthenedConicDuality": sddp.StrengthenedConicDuality,
    "BanditDuality": sddp.BanditDuality,
    "FixedDiscreteDuality": sddp.FixedDiscreteDuality,
    "simulate": sddp.simulate,
    "calculate_bound": sddp.calculate_bound,
    "add_all_cuts": sddp.add_all_cuts,
    "DecisionRule": sddp.DecisionRule,
    "evaluate": sddp.evaluate,
    "SpaghettiPlot": sddp.SpaghettiPlot,
    "add_spaghetti": (sddp.add_spaghetti, sddp.SpaghettiPlot.add_spaghetti),
    "publication_plot": sddp.publication_plot,
    "ValueFunction": sddp.ValueFunction,
    "evaluate(::ValueFunction)": sddp.evaluate_value_function,
    "plot": (sddp.plot_graph, sddp.plot_value_function, sddp.SpaghettiPlot.plot),
    "write_subproblem_to_file": sddp.write_subproblem_to_file,
    "deterministic_equivalent": sddp.deterministic_equivalent,
    "write_to_file": sddp.write_to_file,
    "read_from_file": sddp.read_from_file,
    "write": sddp.write_to_file,
    "read": sddp.read_from_file,
    "evaluate(::PolicyGraph, ::ValidationScenarios)": sddp.evaluate_validation_scenarios,
    "ValidationScenarios": sddp.ValidationScenarios,
    "ValidationScenario": sddp.ValidationScenario,
}


@pytest.mark.parametrize("name", list(API_REFERENCE))
def test_api_entry_has_python_equivalent(name):
    targets = API_REFERENCE[name]
    for target in targets if isinstance(targets, tuple) else (targets,):
        assert callable(target) or isinstance(target, type), name


def test_public_names_are_exported():
    for name in ("parameterize", "write_subproblem_to_file", "sample_noise", "Multiprocess"):
        assert name in sddp.__all__, name
