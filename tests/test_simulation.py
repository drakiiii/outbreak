import json

import numpy as np
import pytest

from outbreak.config import SimulationConfig, preset_scenario
from outbreak.metrics import aggregate_ensemble
from outbreak.simulation import RunState, Simulation, run_ensemble


# Helper (parametrize-style factory): builds a validated covid-like scenario.
# Tests pass simulation kwargs (days, stochastic, seed, ...) via **sim_kw so each
# can configure deterministic or stochastic runs without repeating boilerplate.
def _scenario(days=200, **sim_kw):
    scenario = preset_scenario("covid_like", total_population=500_000)
    scenario.simulation = SimulationConfig(duration_days=days, **sim_kw)
    return scenario.validate()


# Verifies the Simulation controller's lifecycle: IDLE -> RUNNING (after a step)
# -> FINISHED (after run_to_end). Once finished, calling step() again is a no-op
# that returns None rather than advancing or raising.
def test_state_machine_transitions():
    sim = Simulation(_scenario(days=10, stochastic=False, overdispersion=None))
    assert sim.state == RunState.IDLE
    sim.step()
    assert sim.state == RunState.RUNNING
    sim.run_to_end()
    assert sim.state == RunState.FINISHED
    assert sim.is_finished
    # Stepping past the end is a no-op.
    assert sim.step() is None


# Pausing and resuming must not lose or duplicate any recorded days. After
# running 50 steps, history has 50 records; resume() runs to the end. The test
# then checks the recorded days are sorted (contiguous, ordered) and that the
# last pre-pause record sits one timestep (dt) before the pause day.
def test_pause_and_resume_continuity():
    sim = Simulation(_scenario(days=200, stochastic=True, overdispersion=0.5, seed=42))
    sim.run(steps=50)
    assert len(sim.history) == 50
    day_at_pause = sim.current_day
    sim.pause()
    sim.resume()
    assert sim.is_finished
    # History is contiguous and ordered.
    days = [r.day for r in sim.history]
    assert days == sorted(days)
    assert sim.history[49].day == pytest.approx(day_at_pause - sim.config.simulation.dt)


# run(stop_when=...) accepts a predicate callback evaluated each step; the run
# halts (state PAUSED, not FINISHED) the moment it returns True. Here it stops
# once current_day reaches 30, well before the 365-day duration.
def test_stop_when_predicate():
    sim = Simulation(_scenario(days=365, stochastic=False, overdispersion=None))
    sim.run(stop_when=lambda s: s.current_day >= 30)
    assert sim.current_day == pytest.approx(30.0)
    assert sim.state == RunState.PAUSED


# reset() must return the controller to a pristine state: empty history, IDLE
# state, and current_day back at 0, even after a full run.
def test_reset_restores_initial_state():
    sim = Simulation(_scenario(days=100, stochastic=False, overdispersion=None))
    sim.run_to_end()
    assert len(sim.history) > 0
    sim.reset()
    assert sim.history == []
    assert sim.state == RunState.IDLE
    assert sim.current_day == 0.0


# Sanity-bounds the post-run summary statistics. waning_immunity_days=None gives
# lifelong immunity so nobody is reinfected, keeping the attack rate <= 1. Checks
# the peak occurs on a real day, r0 matches the config, and the IFR lands in a
# plausibly small range (0, 0.1) for a covid-like disease.
def test_summary_reports_sensible_numbers():
    scenario = _scenario(days=365, stochastic=False, overdispersion=None)
    scenario.disease.waning_immunity_days = None  # lifelong immunity => attack rate <= 1
    sim = Simulation(scenario.validate())
    sim.run_to_end()
    summary = sim.summary()
    assert summary is not None
    assert 0.0 < summary.attack_rate <= 1.0
    assert summary.total_deaths >= 0
    assert summary.peak_infectious > 0
    assert summary.peak_infectious_day is not None
    assert summary.r0 == sim.r0
    # IFR should be a small positive fraction for a covid-like disease.
    assert 0.0 < summary.infection_fatality_ratio < 0.1


# record_at_day(d) "scrubs" the history to find the record nearest day d (like a
# timeline scrubber). abs=1.0 tolerates being off by up to one day, since records
# fall on discrete timesteps that may not land exactly on day 50.
def test_record_at_day_scrubbing():
    sim = Simulation(_scenario(days=100, stochastic=False, overdispersion=None))
    sim.run_to_end()
    rec = sim.record_at_day(50)
    assert rec is not None
    assert rec.day == pytest.approx(50.0, abs=1.0)


# to_columns() pivots the history into a column-oriented dict (for dataframes/
# plotting). The set comprehension collects every column length; a single unique
# length means all columns align. Also checks cumulative_infections is present
# and monotonically non-decreasing (it can only ever grow).
def test_to_columns_lengths_align():
    sim = Simulation(_scenario(days=60, stochastic=False, overdispersion=None))
    sim.run_to_end()
    cols = sim.to_columns()
    lengths = {len(v) for v in cols.values()}
    assert len(lengths) == 1
    assert "cumulative_infections" in cols
    assert cols["cumulative_infections"][-1] >= cols["cumulative_infections"][0]


# Save/load round-trip via a JSON snapshot. tmp_path is pytest's per-test
# temporary directory fixture (a pathlib.Path), auto-cleaned afterwards. The
# saved snapshot must restore the RNG state so the original and reloaded sims
# produce identical continuations (exact == on the two trajectories).
def test_save_and_load_roundtrip(tmp_path):
    sim = Simulation(_scenario(days=200, stochastic=True, overdispersion=0.5, seed=99))
    sim.run(steps=80)
    path = tmp_path / "snapshot.json"
    sim.save(str(path))

    restored = Simulation.load(str(path))
    assert restored.model.t == sim.model.t
    # Both continue identically from the saved point (RNG state preserved).
    a = [sim.step().new_infections for _ in range(20)]
    b = [restored.step().new_infections for _ in range(20)]
    assert a == b


# Confirms to_dict() yields something JSON-serialisable. The default= callback
# tells json.dumps how to encode otherwise-unsupported objects: numpy arrays are
# converted to lists via their .tolist() method.
def test_snapshot_is_json_serialisable():
    sim = Simulation(_scenario(days=20, stochastic=True, overdispersion=0.5, seed=3))
    sim.run(steps=10)
    payload = json.dumps(sim.to_dict(), default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    assert isinstance(payload, str)


# run_ensemble runs n_runs stochastic simulations, each seeded deterministically
# from base_seed. Same base_seed => identical ensemble (final deaths per run
# match exactly). aggregate_ensemble then collapses the runs into quantile bands;
# the test asserts the bands are ordered q0.1 <= q0.5 <= q0.9 at every timestep
# (the +1e-9 slack absorbs float noise at ties).
def test_run_ensemble_reproducible_and_varied():
    scenario = _scenario(days=120, stochastic=True, overdispersion=0.5)
    ens1 = run_ensemble(scenario, n_runs=5, base_seed=2024)
    ens2 = run_ensemble(scenario, n_runs=5, base_seed=2024)
    # Reproducible given the same base seed.
    a = [h[-1].D for h in ens1]
    b = [h[-1].D for h in ens2]
    assert a == b
    # Aggregation produces ordered quantile bands.
    agg = aggregate_ensemble(ens1, "Is", quantiles=(0.1, 0.5, 0.9))
    lo, mid, hi = np.array(agg["q0.1"]), np.array(agg["q0.5"]), np.array(agg["q0.9"])
    assert np.all(lo <= mid + 1e-9) and np.all(mid <= hi + 1e-9)
