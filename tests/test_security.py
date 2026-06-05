"""Hardening tests: restoring a saved/uploaded snapshot treats it as untrusted.

A malformed or tampered snapshot must be rejected with a clear ``ValueError``
rather than silently corrupting a run or triggering an unbounded allocation.
"""

import numpy as np
import pytest

from outbreak import Simulation
from outbreak.agents import AgentModel, SUS, D
from outbreak.config import PopulationConfig, ScenarioConfig, SimulationConfig, preset_scenario
from outbreak.model import EpidemicModel


def _compartmental():
    sc = preset_scenario("covid_like", total_population=200_000)
    sc.simulation = SimulationConfig(duration_days=50, engine="compartmental", seed=1)
    return sc.validate()


def _agent():
    sc = ScenarioConfig(
        population=PopulationConfig(total_population=200_000, initial_infected=500),
        disease=preset_scenario("influenza_like").disease,
        simulation=SimulationConfig(duration_days=50, engine="agent", n_agents=20_000, seed=1),
    )
    return sc.validate()


# ------------------------------------------------------- valid round-trips
def test_compartmental_roundtrip_still_works():
    model = EpidemicModel(_compartmental())
    for _ in range(20):
        model.step()
    clone = EpidemicModel(_compartmental())
    clone.set_state(model.get_state())
    assert [model.step().new_infections for _ in range(10)] == \
           [clone.step().new_infections for _ in range(10)]


def test_agent_roundtrip_still_works():
    model = AgentModel(_agent())
    for _ in range(20):
        model.step()
    clone = AgentModel(_agent())
    clone.set_state(model.get_state())
    assert [model.step().new_infections for _ in range(10)] == \
           [clone.step().new_infections for _ in range(10)]


# ------------------------------------------------- compartmental rejections
def test_compartmental_rejects_wrong_shape():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    state["S"] = [1.0, 2.0]  # wrong length (n_age != 2 here)
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


def test_compartmental_rejects_missing_field():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    del state["E"]
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


def test_compartmental_rejects_ragged_array():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    state["E"] = [[1.0, 2.0], [3.0]]  # ragged / non-rectangular
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


# -------------------------------------------------------- agent rejections
def test_agent_rejects_n_agents_mismatch():
    model = AgentModel(_agent())
    state = model.get_state()
    state["n_agents"] = state["n_agents"] + 5  # would imply a giant reallocation
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


def test_agent_rejects_wrong_length_array():
    model = AgentModel(_agent())
    state = model.get_state()
    state["age"] = state["age"][:-1]  # length no longer matches n_agents
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


def test_agent_rejects_out_of_range_state():
    model = AgentModel(_agent())
    state = model.get_state()
    state["state"] = [D + 7] * model.n_agents  # invalid state code
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


def test_agent_rejects_out_of_range_age():
    model = AgentModel(_agent())
    state = model.get_state()
    state["age"] = [model.n_age + 3] * model.n_agents  # invalid age group
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


def test_agent_rejects_nonpositive_scale():
    model = AgentModel(_agent())
    state = model.get_state()
    state["scale"] = 0.0
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# ------------------------------------------------ controller-level guarding
def test_simulation_from_dict_rejects_tampered_snapshot():
    sim = Simulation(_agent())
    sim.run(steps=15)
    snapshot = sim.to_dict()
    snapshot["engine_state"]["state"] = snapshot["engine_state"]["state"][:10]
    with pytest.raises(ValueError):
        Simulation.from_dict(snapshot)
