"""Tests for the detection / reporting (surveillance) layer.

Reported cases = symptomatic onsets thinned by an ascertainment fraction and
pushed back by a reporting delay. It is a post-processing transform of the run
history: it changes only the *observed* outputs, never the dynamics. Default
reporting (ascertainment 1, no delay) leaves reported == true symptomatic.
"""

import numpy as np
import pytest

from outbreak import Simulation, preset_scenario, reported_incidence
from outbreak.config import ReportingConfig, ScenarioConfig, SimulationConfig
from outbreak.metrics import summarize


def _run(reporting=None, days=250):
    sc = preset_scenario("covid_like", total_population=400_000)
    sc.simulation = SimulationConfig(stochastic=False, overdispersion=None, duration_days=days)
    if reporting is not None:
        sc.reporting = reporting
    sim = Simulation(sc.validate())
    sim.run_to_end()
    return sim


# --------------------------------------------------------------- defaults
def test_default_reporting_equals_symptomatic():
    sim = _run()
    s = sim.summary()
    assert s.total_reported == pytest.approx(s.total_symptomatic)
    cols = sim.to_columns()
    assert cols["reported_cases"] == cols["new_symptomatic"]


def test_reported_cases_column_present():
    assert "reported_cases" in _run().to_columns()


# --------------------------------------------------------------- ascertainment
def test_ascertainment_scales_reported_total():
    sim = _run(ReportingConfig(ascertainment=0.25))
    s = sim.summary()
    assert s.total_reported == pytest.approx(0.25 * s.total_symptomatic, rel=1e-6)


def test_full_ascertainment_recovers_symptomatic():
    sim = _run(ReportingConfig(ascertainment=1.0))
    s = sim.summary()
    assert s.total_reported == pytest.approx(s.total_symptomatic)


# --------------------------------------------------------------- delay
def test_delay_shifts_the_reported_peak_later():
    sim = _run(ReportingConfig(ascertainment=1.0, reporting_delay_days=10))
    cols = sim.to_columns()
    sym_peak = int(np.argmax(cols["new_symptomatic"]))
    rep_peak = int(np.argmax(cols["reported_cases"]))
    assert rep_peak == pytest.approx(sym_peak + 10, abs=1)
    # The delay just shifts onsets forward; cases pushed past the horizon are not
    # reported, so the reported total equals the onsets up to (end - delay).
    sym = np.array(cols["new_symptomatic"])
    assert sum(cols["reported_cases"]) == pytest.approx(sym[: sym.size - 10].sum(), rel=1e-6)


def test_reported_incidence_zero_pads_front_under_delay():
    sim = _run(ReportingConfig(reporting_delay_days=5))
    rep = reported_incidence(sim.history, ReportingConfig(reporting_delay_days=5))
    assert np.all(rep[:5] == 0.0)             # nothing reported before the first onset + delay


def test_reported_incidence_no_reporting_is_symptomatic():
    sim = _run()
    rep = reported_incidence(sim.history, None)
    sym = np.array([r.new_symptomatic for r in sim.history])
    assert np.array_equal(rep, sym)


# --------------------------------------------------------------- validation
def test_reporting_validation():
    with pytest.raises(ValueError):
        ReportingConfig(ascertainment=1.5).validate()
    with pytest.raises(ValueError):
        ReportingConfig(ascertainment=-0.1).validate()
    with pytest.raises(ValueError):
        ReportingConfig(reporting_delay_days=-1).validate()


def test_reporting_survives_config_roundtrip():
    sc = preset_scenario("covid_like")
    sc.reporting = ReportingConfig(ascertainment=0.4, reporting_delay_days=6)
    restored = ScenarioConfig.from_dict(sc.validate().to_dict())
    assert restored.reporting.ascertainment == 0.4
    assert restored.reporting.reporting_delay_days == 6
