"""Tests for fitting model parameters to observed case data."""

import numpy as np
import pytest

from outbreak import Simulation, fit_to_incidence, preset_scenario
from outbreak.config import SimulationConfig


def _synthetic_symptomatic(r0, days=160, total_population=500_000):
    """Daily symptomatic incidence from a deterministic run at a known R0."""
    sc = preset_scenario("covid_like", total_population=total_population)
    sc.disease.r0 = r0
    sc.simulation = SimulationConfig(stochastic=False, overdispersion=None, duration_days=days)
    sim = Simulation(sc.validate())
    sim.run_to_end()
    return np.array(sim.to_columns()["new_symptomatic"])


def _base():
    return preset_scenario("covid_like", total_population=500_000)


# --------------------------------------------------------------- recovery
def test_recovers_known_r0_from_clean_data():
    observed = _synthetic_symptomatic(1.8)
    fit = fit_to_incidence(observed, _base(), fit_scale=False)
    assert fit.success
    assert fit.r0 == pytest.approx(1.8, rel=0.02)
    assert fit.scale == pytest.approx(1.0)            # fit_scale=False pins it


def test_recovers_r0_and_scale_from_noisy_underreported_data():
    truth_r0, truth_scale = 2.3, 0.25
    sym = _synthetic_symptomatic(truth_r0)
    rng = np.random.default_rng(0)
    observed = rng.poisson(truth_scale * sym)         # noisy, under-reported
    fit = fit_to_incidence(observed, _base(), fit_scale=True)
    assert fit.r0 == pytest.approx(truth_r0, rel=0.05)
    assert fit.scale == pytest.approx(truth_scale, rel=0.10)


def test_fitted_curve_tracks_the_data():
    observed = _synthetic_symptomatic(2.0)
    fit = fit_to_incidence(observed, _base(), fit_scale=False)
    # Predicted and observed should be highly correlated (same shape).
    corr = np.corrcoef(fit.predicted, observed)[0, 1]
    assert corr > 0.99


def test_fitted_scenario_carries_the_r0():
    observed = _synthetic_symptomatic(1.6)
    fit = fit_to_incidence(observed, _base())
    assert fit.scenario.disease.r0 == pytest.approx(fit.r0)


def test_higher_observed_growth_fits_higher_r0():
    slow = fit_to_incidence(_synthetic_symptomatic(1.5), _base(), fit_scale=False)
    fast = fit_to_incidence(_synthetic_symptomatic(2.6), _base(), fit_scale=False)
    assert fast.r0 > slow.r0


def test_respects_r0_bounds():
    observed = _synthetic_symptomatic(2.0)
    fit = fit_to_incidence(observed, _base(), r0_bounds=(1.0, 1.5))
    assert 1.0 <= fit.r0 <= 1.5                        # clamped into the search range


# --------------------------------------------------------------- validation
def test_rejects_empty_or_bad_input():
    with pytest.raises(ValueError):
        fit_to_incidence([], _base())
    with pytest.raises(ValueError):
        fit_to_incidence([[1, 2], [3, 4]], _base())    # not 1-D
    with pytest.raises(ValueError):
        fit_to_incidence([1.0, -5.0, 2.0], _base())    # negative counts
