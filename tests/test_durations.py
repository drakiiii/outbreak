"""Tests for realistic (non-exponential) stage durations in the agent engine.

Each agent samples an explicit time-in-stage from a gamma whose shape
(``duration_dispersion``) controls how peaked it is. Shape 1 reproduces the old
exponential behaviour; larger shapes cluster durations around the mean. The mean
is unchanged, so R0 and the final size are unaffected — only the epidemic's
*shape* (timing) changes.
"""

import numpy as np
import pytest

from outbreak import run_ensemble
from outbreak.agents import AgentModel
from outbreak.config import DiseaseConfig, SimulationConfig, preset_scenario
from outbreak.metrics import summarize


def _scenario(dispersion, n_agents=80_000, seed=0, **sim_kw):
    sc = preset_scenario("covid_like", total_population=200_000)
    sc.disease.duration_dispersion = dispersion
    sc.simulation = SimulationConfig(engine="agent", n_agents=n_agents, seed=seed, **sim_kw)
    return sc.validate()


def _sampled_durations(dispersion, mean_days=18.0, n=40_000):
    """The durations the engine actually assigns for a given shape."""
    model = AgentModel(_scenario(dispersion))
    idx = np.arange(n)
    model._sample_duration(idx, mean_days)
    return model.timer[idx]


# ---------------------------------------------------------- duration shape
def test_shape_one_is_exponential():
    d = _sampled_durations(1.0)
    assert d.mean() == pytest.approx(18.0, rel=0.05)
    assert d.std() / d.mean() == pytest.approx(1.0, abs=0.1)   # CV ~ 1 (exponential)


def test_larger_shape_is_peaked():
    d = _sampled_durations(6.0)
    assert d.mean() == pytest.approx(18.0, rel=0.05)
    # CV ~ 1/sqrt(6) ~ 0.41, and almost nobody leaves in under a day.
    assert d.std() / d.mean() == pytest.approx(1 / np.sqrt(6), abs=0.05)
    assert np.mean(d < 1.0) < 0.01


def test_peaked_has_far_fewer_instant_exits_than_exponential():
    expo = _sampled_durations(1.0)
    peaked = _sampled_durations(5.0)
    assert np.mean(peaked < 1.0) < np.mean(expo < 1.0)
    assert np.mean(peaked > 40.0) < np.mean(expo > 40.0)


# ------------------------------------------------- invariants (R0, final size)
def test_r0_calibration_unaffected_by_dispersion():
    for k in (1.0, 4.0, 8.0):
        model = AgentModel(_scenario(k, n_agents=80_000))
        assert model.effective_rt(0.0) == pytest.approx(model.config.disease.r0, rel=3e-2)


def test_final_size_unaffected_by_dispersion():
    """Final size depends on R0, not on the duration distribution, so the
    attack rate should be (statistically) the same for k=1 and k=4."""
    def mean_attack(k):
        sc = _scenario(k, n_agents=80_000, overdispersion=None)
        sc.disease.waning_immunity_days = None       # lifelong => true final size
        sc = sc.validate()
        histories = run_ensemble(sc, n_runs=6, base_seed=0)
        return np.mean([summarize(h, sc.disease.r0).attack_rate for h in histories])

    assert mean_attack(1.0) == pytest.approx(mean_attack(4.0), rel=0.05)


# ----------------------------------------------------------- state & config
def test_timer_persisted_across_save_load():
    model = AgentModel(_scenario(4.0, seed=3))
    for _ in range(40):
        model.step()
    clone = AgentModel(_scenario(4.0, seed=3))
    clone.set_state(model.get_state())
    assert np.array_equal(clone.timer, model.timer)
    # And the continuation matches exactly.
    a = [model.step().new_infections for _ in range(15)]
    b = [clone.step().new_infections for _ in range(15)]
    assert a == b


def test_snapshot_without_timer_still_loads():
    """Snapshots saved before this feature lack a 'timer' field; loading them
    should default the timer to zeros rather than error."""
    model = AgentModel(_scenario(4.0, seed=2))
    for _ in range(20):
        model.step()
    state = model.get_state()
    del state["timer"]                                # simulate an old snapshot
    clone = AgentModel(_scenario(4.0, seed=2))
    clone.set_state(state)                            # must not raise
    assert clone.timer.shape == (clone.n_agents,)


def test_nonpositive_dispersion_rejected():
    with pytest.raises(ValueError):
        DiseaseConfig(duration_dispersion=0.0).validate(n_age=4)


def test_presets_use_peaked_durations():
    # The built-in diseases should ship with realistic (peaked) durations.
    assert preset_scenario("covid_like").disease.duration_dispersion > 1.0
    assert preset_scenario("measles_like").disease.duration_dispersion > 1.0
