"""Tests for the transmission-environment features: age-specific susceptibility,
seasonality, and an external (importation / reservoir) force of infection.

All three default to a no-op (uniform susceptibility, zero seasonal amplitude,
zero external rate), so they don't change baseline behaviour; these tests check
that, when switched on, they behave correctly and keep R0 calibrated.
"""

import numpy as np
import pytest

from outbreak.agents import AgentModel
from outbreak.config import (
    DiseaseConfig,
    EnvironmentConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
    preset_scenario,
)
from outbreak.model import EpidemicModel


def _scenario(engine="compartmental", **env_kw):
    sc = preset_scenario("covid_like", total_population=400_000)
    sc.simulation = SimulationConfig(engine=engine, n_agents=80_000,
                                     stochastic=(engine == "agent"),
                                     overdispersion=None, seed=1)
    if env_kw:
        sc.environment = EnvironmentConfig(**env_kw)
    return sc


# ----------------------------------------------------- age-specific susceptibility
def test_susceptibility_preserves_r0_both_engines():
    for Engine, eng in ((EpidemicModel, "compartmental"), (AgentModel, "agent")):
        sc = _scenario(eng)
        sc.disease.susceptibility = [0.4, 1.0, 1.0, 1.0]   # children less susceptible
        model = Engine(sc.validate())
        assert model.effective_rt(0.0) == pytest.approx(sc.disease.r0, rel=3e-2)


def test_lower_susceptibility_reduces_attack_rate():
    def attack(susc):
        sc = preset_scenario("covid_like", total_population=400_000)
        sc.disease.r0 = 1.6                       # lower R0 => susceptibility matters more
        sc.disease.susceptibility = susc
        sc.simulation = SimulationConfig(stochastic=False, overdispersion=None)
        m = EpidemicModel(sc.validate())
        new = sum(m.step().new_infections for _ in range(300))
        return new / m.total_population()

    assert attack([0.3, 1.0, 1.0, 1.0]) < attack([1.0, 1.0, 1.0, 1.0])


def test_zero_susceptibility_age_is_never_infected():
    # Children fully unsusceptible; the outbreak is started externally (no seeds),
    # so age group 0 must stay entirely in S.
    sc = ScenarioConfig(
        population=PopulationConfig(total_population=400_000, initial_infected=0),
        disease=DiseaseConfig(r0=2.0, susceptibility=[0.0, 1.0, 1.0, 1.0],
                              waning_immunity_days=None),
        environment=EnvironmentConfig(external_infection_rate=1e-4),
        simulation=SimulationConfig(stochastic=False, overdispersion=None, duration_days=200),
    )
    m = EpidemicModel(sc.validate())
    s0_start = m.S[0]
    for _ in range(200):
        m.step()
    assert m.S[0] == pytest.approx(s0_start)      # untouched
    assert m.S[1] < 0.5 * m.N[1]                  # adults did get infected


# --------------------------------------------------------------------- seasonality
def test_seasonal_multiplier_formula():
    env = EnvironmentConfig(seasonal_amplitude=0.4, seasonal_period_days=365,
                            seasonal_peak_day=0).validate()
    assert env.seasonal_multiplier(0.0) == pytest.approx(1.4)       # peak
    assert env.seasonal_multiplier(182.5) == pytest.approx(0.6, abs=1e-2)  # trough
    # Averages to 1 over a full period.
    days = np.arange(0, 365)
    assert np.mean([env.seasonal_multiplier(d) for d in days]) == pytest.approx(1.0, abs=1e-2)


def test_zero_amplitude_is_constant():
    env = EnvironmentConfig(seasonal_amplitude=0.0).validate()
    assert env.seasonal_multiplier(0.0) == 1.0
    assert env.seasonal_multiplier(123.0) == 1.0


def test_seasonality_modulates_beta():
    sc = _scenario("compartmental", seasonal_amplitude=0.5, seasonal_period_days=365,
                   seasonal_peak_day=0)
    m = EpidemicModel(sc.validate())
    assert m.beta_effective(0.0) > m.beta_effective(182.0)         # peak > trough


# ---------------------------------------------------- external / spillover force
def test_external_force_value_tracks_season():
    env = EnvironmentConfig(external_infection_rate=1e-3, seasonal_amplitude=0.5,
                            seasonal_peak_day=0).validate()
    assert env.external_force(0.0) == pytest.approx(1e-3 * 1.5)    # boosted at the seasonal peak
    assert EnvironmentConfig(external_infection_rate=0.0).external_force(0.0) == 0.0


def test_external_force_starts_outbreak_without_seeds():
    def total_infections(rate):
        sc = ScenarioConfig(
            population=PopulationConfig(total_population=400_000, initial_infected=0),
            disease=DiseaseConfig(r0=1.6, waning_immunity_days=None),
            environment=EnvironmentConfig(external_infection_rate=rate),
            simulation=SimulationConfig(stochastic=False, overdispersion=None, duration_days=250),
        )
        m = EpidemicModel(sc.validate())
        return sum(m.step().new_infections for _ in range(250))

    assert total_infections(0.0) == 0.0                # no seeds, no spillover -> nothing
    assert total_infections(5e-5) > 1000.0             # spillover ignites an outbreak


def test_external_force_sustains_subcritical_disease():
    # R0 < 1 can't sustain itself, but a constant reservoir keeps producing cases.
    sc = ScenarioConfig(
        population=PopulationConfig(total_population=400_000, initial_infected=0),
        disease=DiseaseConfig(r0=0.6, waning_immunity_days=120.0),
        environment=EnvironmentConfig(external_infection_rate=2e-5),
        simulation=SimulationConfig(stochastic=False, overdispersion=None, duration_days=200),
    )
    m = EpidemicModel(sc.validate())
    new = [m.step().new_infections for _ in range(200)]
    assert new[-1] > 0.0                               # still generating cases at the end


def test_external_force_in_agent_engine():
    sc = ScenarioConfig(
        population=PopulationConfig(total_population=400_000, initial_infected=0),
        disease=DiseaseConfig(r0=1.6, waning_immunity_days=None),
        environment=EnvironmentConfig(external_infection_rate=1e-4),
        simulation=SimulationConfig(engine="agent", n_agents=60_000, overdispersion=None,
                                    duration_days=150, seed=2),
    )
    m = AgentModel(sc.validate())
    total = sum(m.step().new_infections for _ in range(150))
    assert total > 0.0                                 # spillover seeds the agent model too


# ------------------------------------------------------------- config validation
def test_environment_validation():
    with pytest.raises(ValueError):
        EnvironmentConfig(seasonal_amplitude=1.0).validate()       # must be < 1
    with pytest.raises(ValueError):
        EnvironmentConfig(seasonal_amplitude=-0.1).validate()
    with pytest.raises(ValueError):
        EnvironmentConfig(seasonal_period_days=0).validate()
    with pytest.raises(ValueError):
        EnvironmentConfig(external_infection_rate=-1.0).validate()


def test_negative_susceptibility_rejected():
    with pytest.raises(ValueError):
        DiseaseConfig(susceptibility=[-0.1, 1.0, 1.0, 1.0]).validate(n_age=4)


def test_environment_survives_config_roundtrip():
    sc = preset_scenario("covid_like")
    sc.environment = EnvironmentConfig(seasonal_amplitude=0.3, external_infection_rate=1e-5)
    sc.disease.susceptibility = [0.5, 1.0, 1.0, 1.2]
    restored = ScenarioConfig.from_dict(sc.validate().to_dict())
    assert restored.environment.seasonal_amplitude == 0.3
    assert restored.environment.external_infection_rate == 1e-5
    assert list(restored.disease.susceptibility) == [0.5, 1.0, 1.0, 1.2]
