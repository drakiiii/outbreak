import numpy as np
import pytest

from outbreak.config import (
    DiseaseConfig,
    Intervention,
    InterventionConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
    VaccinationConfig,
    preset_scenario,
)
from outbreak.model import EpidemicModel


def _homogeneous_scenario(**sim_kw):
    """A simple single-age-group scenario for analytic comparisons."""
    return ScenarioConfig(
        population=PopulationConfig(
            total_population=1_000_000,
            age_group_labels=["all"],
            age_distribution=[1.0],
            initial_infected=100,
        ),
        disease=DiseaseConfig(r0=2.5, waning_immunity_days=None),
        simulation=SimulationConfig(duration_days=365, **sim_kw),
    ).validate()


def test_beta_calibration_matches_r0_at_t0():
    """With a fully susceptible population, model-implied Rt should equal R0."""
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    model = EpidemicModel(scenario)
    rt0 = model.effective_rt(0.0)
    assert rt0 == pytest.approx(scenario.disease.r0, rel=1e-3)


def test_age_structured_rt_matches_r0():
    scenario = preset_scenario("covid_like", total_population=2_000_000)
    scenario.simulation = SimulationConfig(stochastic=False, overdispersion=None)
    model = EpidemicModel(scenario.validate())
    assert model.effective_rt(0.0) == pytest.approx(scenario.disease.r0, rel=1e-2)


def test_population_is_conserved_stochastic():
    scenario = _homogeneous_scenario(stochastic=True, overdispersion=0.5, seed=1)
    model = EpidemicModel(scenario)
    total0 = model.total_population()
    for _ in range(200):
        model.step()
    assert model.total_population() == pytest.approx(total0, abs=1.0)
    # No negative compartments.
    for name in ("S", "V", "E", "Ip", "Ia", "Is", "H", "C", "R", "D"):
        assert np.all(getattr(model, name) >= 0)


def test_population_is_conserved_deterministic():
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    model = EpidemicModel(scenario)
    total0 = model.total_population()
    for _ in range(365):
        model.step()
    assert model.total_population() == pytest.approx(total0, rel=1e-9)


def test_rt_declines_as_susceptibles_deplete():
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    model = EpidemicModel(scenario)
    rt_start = model.effective_rt(0.0)
    for _ in range(120):
        model.step()
    rt_later = model.effective_rt(model.current_day())
    assert rt_later < rt_start


def test_epidemic_grows_then_burns_out_deterministic():
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    model = EpidemicModel(scenario)
    infectious = []
    for _ in range(365):
        rec = model.step()
        infectious.append(rec.Ip + rec.Ia + rec.Is)
    infectious = np.array(infectious)
    # Epidemic should rise to a peak and then decline well below the peak.
    peak_idx = int(np.argmax(infectious))
    assert 0 < peak_idx < len(infectious) - 1
    assert infectious[-1] < 0.1 * infectious[peak_idx]


def test_final_size_below_one_with_subcritical_r0():
    """An R0 < 1 epidemic should fail to take off (small attack rate)."""
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    scenario.disease.r0 = 0.7
    scenario = scenario.validate()
    model = EpidemicModel(scenario)
    total0 = model.total_population()
    new_infections = 0.0
    for _ in range(365):
        rec = model.step()
        new_infections += rec.new_infections
    assert new_infections / total0 < 0.05


def test_higher_r0_gives_higher_attack_rate():
    def attack_rate(r0):
        scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
        scenario.disease.r0 = r0
        scenario = scenario.validate()
        model = EpidemicModel(scenario)
        total0 = model.total_population()
        new = sum(model.step().new_infections for _ in range(365))
        return new / total0

    assert attack_rate(3.0) > attack_rate(1.5) > 0.0


def test_interventions_reduce_rt():
    scenario = _homogeneous_scenario(stochastic=False, overdispersion=None)
    scenario.interventions = InterventionConfig(
        [Intervention("lockdown", 0, 100, 0.5)]
    )
    scenario = scenario.validate()
    model = EpidemicModel(scenario)
    # During the lockdown window (50% reduction) and with a still-fully
    # susceptible population, Rt should be ~half of R0.
    rt_locked = model.effective_rt(10.0)
    assert rt_locked == pytest.approx(0.5 * scenario.disease.r0, rel=1e-2)


def test_seed_reproducibility():
    scenario = _homogeneous_scenario(stochastic=True, overdispersion=0.5, seed=123)
    m1 = EpidemicModel(scenario)
    m2 = EpidemicModel(scenario)
    s1 = [m1.step().new_infections for _ in range(100)]
    s2 = [m2.step().new_infections for _ in range(100)]
    assert s1 == s2


def test_different_seeds_diverge():
    sc1 = _homogeneous_scenario(stochastic=True, overdispersion=0.5, seed=1)
    sc2 = _homogeneous_scenario(stochastic=True, overdispersion=0.5, seed=2)
    m1, m2 = EpidemicModel(sc1), EpidemicModel(sc2)
    s1 = [m1.step().new_infections for _ in range(100)]
    s2 = [m2.step().new_infections for _ in range(100)]
    assert s1 != s2


def test_state_roundtrip():
    scenario = _homogeneous_scenario(stochastic=True, overdispersion=0.5, seed=7)
    model = EpidemicModel(scenario)
    for _ in range(50):
        model.step()
    state = model.get_state()

    clone = EpidemicModel(scenario)
    clone.set_state(state)
    # Both should now produce identical continuations.
    a = [model.step().new_infections for _ in range(20)]
    b = [clone.step().new_infections for _ in range(20)]
    assert a == b


def test_vaccination_reduces_attack_rate():
    def attack_rate(vaccinate):
        scenario = preset_scenario("covid_like", total_population=1_000_000)
        scenario.simulation = SimulationConfig(stochastic=False, overdispersion=None)
        if vaccinate:
            scenario.vaccination = VaccinationConfig(
                enabled=True, start_day=0, daily_rate=0.01, coverage_cap=0.8,
                ve_susceptibility=0.7,
            )
        scenario = scenario.validate()
        model = EpidemicModel(scenario)
        total0 = model.total_population()
        new = sum(model.step().new_infections for _ in range(365))
        return new / total0

    assert attack_rate(True) < attack_rate(False)


def test_icu_overflow_increases_deaths():
    def deaths(capacity):
        scenario = preset_scenario("covid_like", total_population=1_000_000)
        scenario.simulation = SimulationConfig(stochastic=False, overdispersion=None)
        scenario.healthcare.icu_capacity = capacity
        scenario.healthcare.overflow_mortality_multiplier = 3.0
        scenario = scenario.validate()
        model = EpidemicModel(scenario)
        for _ in range(365):
            model.step()
        return float(model.D.sum())

    constrained = deaths(capacity=200)
    unconstrained = deaths(capacity=None)
    assert constrained > unconstrained
