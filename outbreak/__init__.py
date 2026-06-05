"""Outbreak: a realistic, age-structured stochastic SEIR epidemic simulator.

Quick start
-----------
>>> from outbreak import Simulation, preset_scenario
>>> scenario = preset_scenario("covid_like", total_population=500_000)
>>> sim = Simulation(scenario)
>>> sim.run_to_end()
>>> summary = sim.summary()
>>> round(summary.attack_rate, 2)  # doctest: +SKIP
0.78
"""

# Re-export the public API from the submodules so users can write
# ``from outbreak import Simulation`` instead of reaching into ``outbreak.simulation``.
from .config import (
    DiseaseConfig,
    HealthcareConfig,
    Intervention,
    InterventionConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
    VaccinationConfig,
    DISEASE_PRESETS,
    preset_scenario,
)
from .agents import AgentModel
from .metrics import EpidemicSummary, aggregate_ensemble, history_to_columns, summarize
from .model import EpidemicModel, StepRecord
from .simulation import RunState, Simulation, build_engine, run_ensemble

__version__ = "0.1.0"

# __all__ defines the package's public surface: it sets what
# ``from outbreak import *`` pulls in, and documents the supported names.
__all__ = [
    "ScenarioConfig",
    "PopulationConfig",
    "DiseaseConfig",
    "VaccinationConfig",
    "InterventionConfig",
    "Intervention",
    "HealthcareConfig",
    "SimulationConfig",
    "DISEASE_PRESETS",
    "preset_scenario",
    "EpidemicModel",
    "AgentModel",
    "StepRecord",
    "Simulation",
    "RunState",
    "build_engine",
    "run_ensemble",
    "EpidemicSummary",
    "summarize",
    "aggregate_ensemble",
    "history_to_columns",
    "__version__",
]
