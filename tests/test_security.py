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


# Helper: a validated compartmental scenario used to build models whose snapshots
# are then deliberately corrupted by the rejection tests below.
def _compartmental():
    sc = preset_scenario("covid_like", total_population=200_000)
    sc.simulation = SimulationConfig(duration_days=50, engine="compartmental", seed=1)
    return sc.validate()


# Helper: a validated agent-based scenario, the agent-engine counterpart of the
# compartmental helper above.
def _agent():
    sc = ScenarioConfig(
        population=PopulationConfig(total_population=200_000, initial_infected=500),
        disease=preset_scenario("influenza_like").disease,
        simulation=SimulationConfig(duration_days=50, engine="agent", n_agents=20_000, seed=1),
    )
    return sc.validate()


# ------------------------------------------------------- valid round-trips
# Baseline: a well-formed (untampered) compartmental snapshot must still load and
# resume exactly. This guards the hardening below from rejecting legitimate state.
def test_compartmental_roundtrip_still_works():
    model = EpidemicModel(_compartmental())
    for _ in range(20):
        model.step()
    clone = EpidemicModel(_compartmental())
    clone.set_state(model.get_state())
    assert [model.step().new_infections for _ in range(10)] == \
           [clone.step().new_infections for _ in range(10)]


# Same baseline check for the agent engine: a clean snapshot round-trips and the
# clone continues identically. Guards the agent rejections from false positives.
def test_agent_roundtrip_still_works():
    model = AgentModel(_agent())
    for _ in range(20):
        model.step()
    clone = AgentModel(_agent())
    clone.set_state(model.get_state())
    assert [model.step().new_infections for _ in range(10)] == \
           [clone.step().new_infections for _ in range(10)]


# ------------------------------------------------- compartmental rejections
# Malformation: a compartment array with the wrong length (here 2 entries when
# the scenario has a different number of age groups). set_state must reject the
# shape mismatch with ValueError instead of loading inconsistent state.
def test_compartmental_rejects_wrong_shape():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    state["S"] = [1.0, 2.0]  # wrong length (n_age != 2 here)
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


# Malformation: a required compartment key ("E") is deleted, simulating an
# incomplete snapshot. Loading must fail with ValueError, not a later KeyError.
def test_compartmental_rejects_missing_field():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    del state["E"]
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


# Malformation: a "ragged" (non-rectangular) nested list that can't form a proper
# 2D array. set_state must detect this and raise ValueError rather than building a
# malformed numpy object array.
def test_compartmental_rejects_ragged_array():
    model = EpidemicModel(_compartmental())
    state = model.get_state()
    state["E"] = [[1.0, 2.0], [3.0]]  # ragged / non-rectangular
    with pytest.raises(ValueError):
        EpidemicModel(_compartmental()).set_state(state)


# -------------------------------------------------------- agent rejections
# Malformation: n_agents inflated beyond the actual array sizes. Trusting it could
# trigger a huge reallocation (a DoS vector), so it must be rejected with ValueError.
def test_agent_rejects_n_agents_mismatch():
    model = AgentModel(_agent())
    state = model.get_state()
    state["n_agents"] = state["n_agents"] + 5  # would imply a giant reallocation
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# Malformation: a per-agent array ("age") truncated so its length no longer
# matches n_agents. The inconsistency must be caught with ValueError.
def test_agent_rejects_wrong_length_array():
    model = AgentModel(_agent())
    state = model.get_state()
    state["age"] = state["age"][:-1]  # length no longer matches n_agents
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# Malformation: an agent state code outside the valid set (D + 7 is not a real
# compartment). Out-of-range enum values must be rejected with ValueError.
def test_agent_rejects_out_of_range_state():
    model = AgentModel(_agent())
    state = model.get_state()
    state["state"] = [D + 7] * model.n_agents  # invalid state code
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# Malformation: an age-group index beyond the configured number of groups
# (n_age + 3). Such an index would later index out of bounds, so it's rejected.
def test_agent_rejects_out_of_range_age():
    model = AgentModel(_agent())
    state = model.get_state()
    state["age"] = [model.n_age + 3] * model.n_agents  # invalid age group
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# Malformation: a zero scale factor. Since reported counts multiply by scale, a
# non-positive value would zero out / corrupt the population, so it's rejected.
def test_agent_rejects_nonpositive_scale():
    model = AgentModel(_agent())
    state = model.get_state()
    state["scale"] = 0.0
    with pytest.raises(ValueError):
        AgentModel(_agent()).set_state(state)


# ------------------------------------------------ controller-level guarding
# End-to-end check at the Simulation (controller) layer: a tampered full snapshot
# (the agent state array truncated to 10 entries) must be refused by from_dict()
# with ValueError, so validation isn't bypassable by going through the top-level API.
def test_simulation_from_dict_rejects_tampered_snapshot():
    sim = Simulation(_agent())
    sim.run(steps=15)
    snapshot = sim.to_dict()
    snapshot["engine_state"]["state"] = snapshot["engine_state"]["state"][:10]
    with pytest.raises(ValueError):
        Simulation.from_dict(snapshot)
