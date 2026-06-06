"""Tests for the agent engine's structured contact network (households/schools/
workplaces): calibration is preserved, construction is correct, the dynamics
change in the expected direction, and snapshots round-trip and stay hardened.
"""

import numpy as np
import pytest

from outbreak import Simulation, run_ensemble
from outbreak.agents import AgentModel
from outbreak.config import (
    DiseaseConfig,
    NetworkConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
)
from outbreak.metrics import summarize
from outbreak.network import build_layers, layer_mixing_matrix


def _scenario(enabled=True, r0=2.0, n_agents=60_000, duration=200, seed=0, **net_kw):
    return ScenarioConfig(
        population=PopulationConfig(total_population=200_000, initial_infected=400),
        disease=DiseaseConfig(r0=r0, waning_immunity_days=None),
        network=NetworkConfig(enabled=enabled, **net_kw),
        simulation=SimulationConfig(duration_days=duration, engine="agent",
                                    n_agents=n_agents, overdispersion=None, seed=seed),
    ).validate()


# ---------------------------------------------------------------- calibration
def test_network_rt_at_t0_matches_r0():
    """Despite structured mixing, the multilayer network is calibrated so the
    model-implied Rt at t=0 still equals the target R0."""
    model = AgentModel(_scenario(enabled=True, r0=2.0, n_agents=80_000))
    assert model.effective_rt(0.0) == pytest.approx(2.0, rel=3e-2)


def test_network_changes_beta_but_not_rt():
    """Enabling the network changes the per-contact beta (contacts are now
    concentrated) but not the calibrated R0."""
    mf = AgentModel(_scenario(enabled=False, n_agents=80_000))
    net = AgentModel(_scenario(enabled=True, n_agents=80_000))
    assert net.beta != pytest.approx(mf.beta, rel=1e-3)   # different transmission rate
    assert net.effective_rt(0.0) == pytest.approx(mf.effective_rt(0.0), rel=3e-2)


def test_custom_weights_still_calibrate():
    model = AgentModel(_scenario(enabled=True, r0=3.0, n_agents=60_000,
                                 household_weight=2.0, school_weight=0.1,
                                 workplace_weight=0.3, community_weight=1.0))
    assert model.effective_rt(0.0) == pytest.approx(3.0, rel=3e-2)


# -------------------------------------------------------------- construction
def test_layers_built_and_named():
    model = AgentModel(_scenario(enabled=True))
    assert [layer.name for layer in model.layers] == ["household", "school", "workplace"]


def test_every_agent_has_a_household():
    model = AgentModel(_scenario(enabled=True))
    hh = model.layers[0].group_id
    assert np.all(hh >= 0)                       # households partition everyone
    sizes = np.bincount(hh)
    assert sizes.sum() == model.n_agents         # sizes sum back to the population
    assert sizes.min() >= 1


def test_school_and_work_respect_age_groups():
    model = AgentModel(_scenario(enabled=True))
    school, work = model.layers[1].group_id, model.layers[2].group_id
    # Only the configured age groups are members (default: school=[0], work=[1,2]).
    assert set(np.unique(model.age[school >= 0]).tolist()) <= {0}
    assert set(np.unique(model.age[work >= 0]).tolist()) <= {1, 2}
    # Non-members are marked -1.
    assert np.all(school[model.age != 0] == -1)


def test_mean_school_size_is_respected():
    model = AgentModel(_scenario(enabled=True, mean_school_size=25))
    school = model.layers[1].group_id
    sizes = np.bincount(school[school >= 0])
    assert sizes.mean() == pytest.approx(25, rel=0.25)   # ~25 per class


def test_layer_mixing_matrix_nonnegative_and_zero_when_empty():
    rng = np.random.default_rng(0)
    age = np.array([0, 0, 1, 1, 2, 3] * 50)
    layers = build_layers(NetworkConfig(enabled=True), age, rng, n_age=4)
    n_by_age = np.bincount(age, minlength=4).astype(float)
    hh_M = layer_mixing_matrix(layers[0], age, 4, n_by_age)
    assert hh_M.shape == (4, 4)
    assert np.all(hh_M >= 0)
    # A layer with no eligible members yields an all-zero mixing matrix.
    empty = build_layers(
        NetworkConfig(enabled=True, school_age_groups=[3]),  # but age 3 present...
        np.array([0, 1, 2] * 10), rng, n_age=4,
    )
    # age 3 absent in this population => school layer empty.
    M = layer_mixing_matrix(empty[1], np.array([0, 1, 2] * 10), 4, np.bincount(np.array([0,1,2]*10), minlength=4).astype(float))
    assert np.allclose(M, 0.0)


# ------------------------------------------------------------------ dynamics
def test_population_conserved_in_network_mode():
    model = AgentModel(_scenario(enabled=True, seed=3))
    total0 = model.total_population()
    for _ in range(150):
        model.step()
    assert model.total_population() == pytest.approx(total0, rel=1e-9)


def test_network_flattens_the_peak():
    """Repeated within-household/class/work contacts cluster transmission and
    locally deplete susceptibles, so for the same R0 the epidemic peak is lower
    than under mean-field mixing. Checked on an ensemble mean for robustness."""
    def mean_peak(enabled):
        scenario = _scenario(enabled=enabled, r0=1.8, n_agents=50_000, duration=400)
        histories = run_ensemble(scenario, n_runs=6, base_seed=1)
        return np.mean([summarize(h, 1.8).peak_infectious for h in histories])

    assert mean_peak(enabled=True) < mean_peak(enabled=False)


def test_disabled_network_has_no_layers():
    model = AgentModel(_scenario(enabled=False))
    assert model.layers == []
    assert model.community_weight == 1.0


# ------------------------------------------------------- determinism & state
def test_network_reproducible_from_seed():
    m1 = AgentModel(_scenario(enabled=True, seed=7))
    m2 = AgentModel(_scenario(enabled=True, seed=7))
    a = [m1.step().new_infections for _ in range(60)]
    b = [m2.step().new_infections for _ in range(60)]
    assert a == b


def test_network_save_load_roundtrip(tmp_path):
    sim = Simulation(_scenario(enabled=True, seed=5))
    sim.run(steps=40)
    path = tmp_path / "network_run.json"
    sim.save(str(path))

    restored = Simulation.load(str(path))
    assert isinstance(restored.model, AgentModel)
    assert [l.name for l in restored.model.layers] == ["household", "school", "workplace"]
    # Identical continuation => network structure and beta were restored exactly.
    a = [sim.step().new_infections for _ in range(20)]
    b = [restored.step().new_infections for _ in range(20)]
    assert a == b


def test_snapshot_rejects_out_of_range_group_ids():
    """Network group-id arrays are untrusted input too: a tampered id outside
    [-1, n) must be rejected when restoring."""
    model = AgentModel(_scenario(enabled=True, seed=1))
    for _ in range(10):
        model.step()
    state = model.get_state()
    state["layers"][0]["group_id"] = [model.n_agents + 1] * model.n_agents
    with pytest.raises(ValueError):
        AgentModel(_scenario(enabled=True, seed=1)).set_state(state)


# ------------------------------------------------------------ config guards
def test_invalid_layer_weight_rejected():
    with pytest.raises(ValueError):
        NetworkConfig(enabled=True, household_weight=-1.0).validate(n_age=4)


def test_all_zero_weights_rejected_when_enabled():
    with pytest.raises(ValueError):
        NetworkConfig(enabled=True, household_weight=0.0, school_weight=0.0,
                      workplace_weight=0.0, community_weight=0.0).validate(n_age=4)


def test_bad_household_distribution_rejected():
    with pytest.raises(ValueError):
        NetworkConfig(household_size_distribution=[0.0, 0.0]).validate(n_age=4)


def test_default_work_groups_exclude_youngest_and_oldest():
    cfg = NetworkConfig()
    assert cfg.resolved_school_groups(4) == [0]
    assert cfg.resolved_work_groups(4) == [1, 2]
