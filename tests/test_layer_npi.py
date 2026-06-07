"""Tests for layer-targeted non-pharmaceutical interventions (NPIs).

A global intervention reduces all transmission (both engines); a layer-targeted
one (e.g. a school closure) reduces only its contact setting in the agent engine,
and is ignored by the compartmental engine (which has no explicit settings).
"""

import numpy as np
import pytest

from outbreak.agents import AgentModel
from outbreak.config import (
    DiseaseConfig,
    Intervention,
    InterventionConfig,
    NetworkConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
)
from outbreak.interventions import lockdown, school_closure
from outbreak.model import EpidemicModel


def _agent_scenario(interventions, seed=0):
    return ScenarioConfig(
        population=PopulationConfig(total_population=200_000, initial_infected=400),
        disease=DiseaseConfig(r0=2.0, waning_immunity_days=None),
        network=NetworkConfig(enabled=True),
        interventions=InterventionConfig(interventions),
        simulation=SimulationConfig(engine="agent", n_agents=50_000, duration_days=200,
                                    overdispersion=None, seed=seed),
    ).validate()


# --------------------------------------------------------------- multiplier
def test_multiplier_respects_layer():
    cfg = InterventionConfig([
        Intervention("global", 0, 100, 0.5, layer=None),
        Intervention("school", 0, 100, 0.4, layer="school"),
    ])
    # No layer arg (compartmental's call) -> only the global one applies.
    assert cfg.multiplier(10) == pytest.approx(0.5)
    # The community channel sees only the global one.
    assert cfg.multiplier(10, "community") == pytest.approx(0.5)
    # The school channel sees global AND school-targeted (stacked).
    assert cfg.multiplier(10, "school") == pytest.approx(0.5 * 0.6)
    # Outside the window, nothing applies.
    assert cfg.multiplier(200, "school") == pytest.approx(1.0)


# ------------------------------------------------------------ agent engine
def test_global_equals_sum_of_per_layer():
    """A global reduction r is identical to applying r to every channel
    separately — a strong check that the per-channel plumbing is correct."""
    r = 0.5
    glob = AgentModel(_agent_scenario([lockdown(0, 200, r)], seed=3))
    per_layer = AgentModel(_agent_scenario([
        Intervention("c", 0, 200, r, layer="community"),
        Intervention("h", 0, 200, r, layer="household"),
        Intervention("s", 0, 200, r, layer="school"),
        Intervention("w", 0, 200, r, layer="workplace"),
    ], seed=3))
    a = [glob.step().new_infections for _ in range(80)]
    b = [per_layer.step().new_infections for _ in range(80)]
    assert a == b


def test_targeted_intervention_reduces_its_layer():
    """A strong community-targeted lockdown (the dominant channel) markedly
    lowers the attack rate; a school-only closure changes it far less."""
    def attack(interventions):
        m = AgentModel(_agent_scenario(interventions, seed=2))
        total0 = m.total_population()
        new = sum(m.step().new_infections for _ in range(200))
        return new / total0

    base = attack([])
    community = attack([Intervention("c", 10, 200, 0.7, layer="community")])
    school = attack([school_closure(10, 200)])
    assert community < base                 # community NPI bites hard
    assert school <= base                   # school-only does no worse than nothing
    assert community < school               # and far less than a community-wide one


def test_rt_reflects_targeted_intervention():
    # During an active community closure, Rt should drop below the no-intervention
    # value at the same (fully susceptible) starting point.
    base = AgentModel(_agent_scenario([], seed=1))
    closed = AgentModel(_agent_scenario(
        [Intervention("c", 0, 100, 0.6, layer="community")], seed=1))
    assert closed.effective_rt(10.0) < base.effective_rt(10.0)


# ------------------------------------------------------- compartmental engine
def test_compartmental_ignores_layer_targeted():
    """The compartmental engine has no settings, so a layer-targeted NPI must not
    change its dynamics (only global interventions affect it)."""
    def deaths(interventions):
        sc = ScenarioConfig(
            population=PopulationConfig(total_population=200_000, initial_infected=400),
            disease=DiseaseConfig(r0=2.0, waning_immunity_days=None),
            interventions=InterventionConfig(interventions),
            simulation=SimulationConfig(engine="compartmental", stochastic=False,
                                        overdispersion=None, duration_days=200),
        ).validate()
        m = EpidemicModel(sc)
        for _ in range(200):
            m.step()
        return float(m.D.sum())

    assert deaths([school_closure(0, 200)]) == pytest.approx(deaths([]))


# ----------------------------------------------------------- config validation
def test_invalid_layer_rejected():
    with pytest.raises(ValueError):
        Intervention("bad", 0, 10, 0.5, layer="bogus").validate()


def test_valid_layers_accepted():
    for layer in (None, "community", "household", "school", "workplace"):
        Intervention("ok", 0, 10, 0.5, layer=layer).validate()
