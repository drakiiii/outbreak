"""Tests for the metapopulation (coupled regions / geography) layer."""

import numpy as np
import pytest

from outbreak import (
    MetapopulationConfig,
    MetapopulationSimulation,
    Region,
    preset_scenario,
)
from outbreak.config import SimulationConfig


def _base(days=250, engine="compartmental"):
    sc = preset_scenario("covid_like", total_population=100_000)
    sc.disease.waning_immunity_days = None
    sc.simulation = SimulationConfig(duration_days=days, engine=engine,
                                     n_agents=40_000, stochastic=(engine == "agent"),
                                     overdispersion=None)
    return sc.validate()


def _regions(seed_first_only=True):
    return [
        Region("A", 100_000, initial_infected=50 if seed_first_only else 0),
        Region("B", 100_000, initial_infected=0),
        Region("C", 100_000, initial_infected=0),
    ]


def _sim(coupling=0.02, mobility=None, regions=None, base=None, seed=0):
    cfg = MetapopulationConfig(regions=regions or _regions(), coupling=coupling, mobility=mobility)
    return MetapopulationSimulation(base or _base(), cfg, base_seed=seed)


# ----------------------------------------------------------------- structure
def test_builds_one_engine_per_region():
    sim = _sim()
    assert len(sim.engines) == 3
    assert sim.region_names == ["A", "B", "C"]
    assert sim.engines[0].total_population() == pytest.approx(100_000)


# ----------------------------------------------------------------- coupling
def test_no_coupling_keeps_regions_independent():
    sim = _sim(coupling=0.0)
    sim.run_to_end()
    # Only the seeded region (A) has an epidemic.
    assert sim.region_summary(0).attack_rate > 0.5
    assert sim.region_summary(1).total_infections == 0.0
    assert sim.region_summary(2).total_infections == 0.0


def test_coupling_spreads_to_unseeded_regions():
    sim = _sim(coupling=0.02)
    sim.run_to_end()
    for i in range(3):
        assert sim.region_summary(i).attack_rate > 0.5      # all three infected
    # The seeded region peaks first; spread reaches the others later.
    assert sim.first_infection_day(0, 100) < sim.first_infection_day(1, 100)


def test_mobility_directionality():
    """With one-way mobility B<-A, seeding A infects B, but seeding only B leaves
    A untouched (A imports from nobody)."""
    base = _base()
    regions = [Region("A", 100_000, 0), Region("B", 100_000, 0)]
    one_way = [[0.0, 0.0],   # A imports from nobody
               [1.0, 0.0]]   # B imports from A

    seed_a = [Region("A", 100_000, 50), Region("B", 100_000, 0)]
    sim = MetapopulationSimulation(base, MetapopulationConfig(seed_a, 0.03, one_way), base_seed=1)
    sim.run_to_end()
    assert sim.region_summary(1).attack_rate > 0.3          # A -> B works

    seed_b = [Region("A", 100_000, 0), Region("B", 100_000, 50)]
    sim2 = MetapopulationSimulation(base, MetapopulationConfig(seed_b, 0.03, one_way), base_seed=1)
    sim2.run_to_end()
    assert sim2.region_summary(0).total_infections == 0.0   # B -/-> A (no back-coupling)


def test_works_with_agent_engine():
    base = _base(days=120, engine="agent")
    regions = [Region("A", 100_000, 300), Region("B", 100_000, 0)]
    sim = MetapopulationSimulation(base, MetapopulationConfig(regions, 0.05), base_seed=2)
    sim.run_to_end()
    assert sim.region_summary(1).total_infections > 0.0     # spread in the agent engine too


# ----------------------------------------------------------------- aggregation
def test_combined_summary_aggregates_regions():
    sim = _sim(coupling=0.02)
    sim.run_to_end()
    combined = sim.summary()
    region_infections = sum(sim.region_summary(i).total_infections for i in range(3))
    assert combined.total_infections == pytest.approx(region_infections, rel=1e-9)
    assert combined.total_population == pytest.approx(300_000, rel=1e-6)


def test_to_columns_has_per_region_series():
    sim = _sim()
    sim.run_to_end()
    cols = sim.to_columns()
    assert "cumulative_infections::A" in cols
    assert "cumulative_infections::C" in cols


# ----------------------------------------------------------- determinism/state
def test_reproducible_from_base_seed():
    a = _sim(seed=7)
    b = _sim(seed=7)
    sa = [sum(r.new_infections for r in recs) for recs in (a.step() for _ in range(60))]
    sb = [sum(r.new_infections for r in recs) for recs in (b.step() for _ in range(60))]
    assert sa == sb


def test_save_load_roundtrip(tmp_path):
    sim = _sim(seed=3)
    for _ in range(40):
        sim.step()
    path = tmp_path / "metapop.json"
    sim.save(str(path))
    restored = MetapopulationSimulation.load(str(path))
    assert restored.region_names == sim.region_names
    a = [sum(r.new_infections for r in recs) for recs in (sim.step() for _ in range(15))]
    b = [sum(r.new_infections for r in recs) for recs in (restored.step() for _ in range(15))]
    assert a == b


# ----------------------------------------------------------------- validation
def test_validation():
    with pytest.raises(ValueError):
        MetapopulationConfig(regions=[]).validate()
    with pytest.raises(ValueError):
        MetapopulationConfig(_regions(), coupling=-0.1).validate()
    with pytest.raises(ValueError):
        MetapopulationConfig(_regions(), mobility=[[0, 1], [1, 0]]).validate()  # wrong shape
    with pytest.raises(ValueError):
        Region("bad", population=0).validate()
