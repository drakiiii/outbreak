import numpy as np
import pytest

from outbreak.config import (
    DiseaseConfig,
    Intervention,
    InterventionConfig,
    PopulationConfig,
    ScenarioConfig,
    VaccinationConfig,
    preset_scenario,
)


def test_population_by_age_sums_to_total():
    pop = PopulationConfig(total_population=1_000_003, age_distribution=[0.2, 0.3, 0.5],
                           age_group_labels=["a", "b", "c"])
    counts = pop.population_by_age()
    assert counts.sum() == 1_000_003
    assert np.all(counts >= 0)


def test_age_distribution_is_normalised():
    pop = PopulationConfig(age_distribution=[1, 1, 2], age_group_labels=["a", "b", "c"])
    assert pytest.approx(sum(pop.age_distribution)) == 1.0


def test_invalid_population_rejected():
    with pytest.raises(ValueError):
        PopulationConfig(total_population=0)
    with pytest.raises(ValueError):
        PopulationConfig(total_population=10, initial_infected=20)


def test_disease_probability_validation():
    with pytest.raises(ValueError):
        DiseaseConfig(hospitalization_rate=1.5).validate(1)
    with pytest.raises(ValueError):
        DiseaseConfig(r0=-1).validate(1)


def test_per_age_severity_length_mismatch():
    with pytest.raises(ValueError):
        DiseaseConfig(hospitalization_rate=[0.1, 0.2]).validate(4)


def test_intervention_validation():
    with pytest.raises(ValueError):
        Intervention(start_day=10, end_day=5).validate()
    iv = Intervention(start_day=5, end_day=10, transmission_reduction=0.5).validate()
    assert iv.is_active(5) and iv.is_active(9) and not iv.is_active(10)


def test_intervention_multiplier_combines_multiplicatively():
    cfg = InterventionConfig([
        Intervention("a", 0, 100, 0.5),
        Intervention("b", 0, 100, 0.5),
    ])
    assert pytest.approx(cfg.multiplier(10)) == 0.25
    assert cfg.multiplier(200) == 1.0


def test_vaccination_validation():
    with pytest.raises(ValueError):
        VaccinationConfig(ve_susceptibility=1.2).validate()


def test_scenario_roundtrip_serialisation():
    scenario = preset_scenario("covid_like", total_population=200_000)
    scenario.interventions = InterventionConfig([Intervention("lockdown", 30, 60, 0.6)])
    d = scenario.to_dict()
    restored = ScenarioConfig.from_dict(d)
    assert restored.disease.r0 == scenario.disease.r0
    assert restored.population.total_population == 200_000
    assert restored.interventions.interventions[0].name == "lockdown"


def test_preset_unknown_raises():
    with pytest.raises(KeyError):
        preset_scenario("ebola_like")
