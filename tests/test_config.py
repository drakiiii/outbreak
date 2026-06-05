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


# Splitting a total population across age fractions must lose nobody to rounding:
# the per-age integer counts should sum back to the exact total (note the odd
# total 1_000_003 deliberately doesn't divide evenly across [0.2, 0.3, 0.5]).
def test_population_by_age_sums_to_total():
    pop = PopulationConfig(total_population=1_000_003, age_distribution=[0.2, 0.3, 0.5],
                           age_group_labels=["a", "b", "c"])
    counts = pop.population_by_age()
    assert counts.sum() == 1_000_003
    assert np.all(counts >= 0)  # and no negative buckets


# Raw weights [1, 1, 2] should be normalised to fractions summing to 1.0.
# pytest.approx wraps the value so the == comparison tolerates float rounding
# (default relative tolerance is 1e-6).
def test_age_distribution_is_normalised():
    pop = PopulationConfig(age_distribution=[1, 1, 2], age_group_labels=["a", "b", "c"])
    assert pytest.approx(sum(pop.age_distribution)) == 1.0


# Nonsensical population inputs must raise ValueError at construction:
# a zero population, and more infected people than the population holds.
# pytest.raises asserts the block raises that exception (fails if it does not).
def test_invalid_population_rejected():
    with pytest.raises(ValueError):
        PopulationConfig(total_population=0)
    with pytest.raises(ValueError):
        PopulationConfig(total_population=10, initial_infected=20)


# Disease parameters are checked by validate(n_age_groups): a probability above
# 1.0 and a negative R0 are both invalid. validate() is called separately from
# the constructor and takes the number of age groups (here 1).
def test_disease_probability_validation():
    with pytest.raises(ValueError):
        DiseaseConfig(hospitalization_rate=1.5).validate(1)
    with pytest.raises(ValueError):
        DiseaseConfig(r0=-1).validate(1)


# A per-age severity parameter given as a list must have one entry per age group:
# here a length-2 list is validated against 4 age groups and must be rejected.
def test_per_age_severity_length_mismatch():
    with pytest.raises(ValueError):
        DiseaseConfig(hospitalization_rate=[0.1, 0.2]).validate(4)


# An intervention ending before it starts is invalid. For a valid [5, 10)
# window, is_active is inclusive of the start day and exclusive of the end day:
# active on days 5 and 9 but not on day 10.
def test_intervention_validation():
    with pytest.raises(ValueError):
        Intervention(start_day=10, end_day=5).validate()
    iv = Intervention(start_day=5, end_day=10, transmission_reduction=0.5).validate()
    assert iv.is_active(5) and iv.is_active(9) and not iv.is_active(10)


# Overlapping interventions stack multiplicatively: two 50%-reduction measures
# (each a 0.5 transmission multiplier) on day 10 give 0.5 * 0.5 = 0.25. Outside
# every window (day 200) the multiplier is 1.0 (no reduction).
def test_intervention_multiplier_combines_multiplicatively():
    cfg = InterventionConfig([
        Intervention("a", 0, 100, 0.5),
        Intervention("b", 0, 100, 0.5),
    ])
    assert pytest.approx(cfg.multiplier(10)) == 0.25
    assert cfg.multiplier(200) == 1.0


# Vaccine efficacy is a fraction in [0, 1]; an efficacy of 1.2 is invalid.
def test_vaccination_validation():
    with pytest.raises(ValueError):
        VaccinationConfig(ve_susceptibility=1.2).validate()


# Round-trip test: a scenario serialised to a plain dict and rebuilt via
# from_dict must preserve its fields (disease R0, population, nested
# interventions). This guards config save/load against drift.
def test_scenario_roundtrip_serialisation():
    scenario = preset_scenario("covid_like", total_population=200_000)
    scenario.interventions = InterventionConfig([Intervention("lockdown", 30, 60, 0.6)])
    d = scenario.to_dict()
    restored = ScenarioConfig.from_dict(d)
    assert restored.disease.r0 == scenario.disease.r0
    assert restored.population.total_population == 200_000
    assert restored.interventions.interventions[0].name == "lockdown"


# Requesting a preset name that isn't registered raises KeyError (it's a lookup
# in a dict of known presets), not ValueError.
def test_preset_unknown_raises():
    with pytest.raises(KeyError):
        preset_scenario("ebola_like")
