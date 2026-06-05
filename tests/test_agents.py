import numpy as np
import pytest

from outbreak import Simulation, run_ensemble
from outbreak.agents import AgentModel, IP, IA, IS, D
from outbreak.config import (
    DiseaseConfig,
    Intervention,
    InterventionConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
    VaccinationConfig,
    preset_scenario,
)
from outbreak.metrics import summarize


# Helper (parametrize-style factory): builds a validated influenza-like scenario,
# defaulting to the agent-based engine. Tests override engine, sizes, seed, etc.
# via the named args and **sim_kw to avoid repeating the construction boilerplate.
def _scenario(engine="agent", n_agents=50_000, total_population=200_000,
              initial_infected=500, duration=250, **sim_kw):
    return ScenarioConfig(
        population=PopulationConfig(
            total_population=total_population, initial_infected=initial_infected
        ),
        disease=preset_scenario("influenza_like").disease,
        simulation=SimulationConfig(
            duration_days=duration, engine=engine, n_agents=n_agents, **sim_kw
        ),
    ).validate()


# --------------------------------------------------------------------- basics
# With engine="agent" the Simulation must dispatch to (instantiate) an AgentModel
# rather than the compartmental EpidemicModel.
def test_engine_dispatch_builds_agent_model():
    sim = Simulation(_scenario())
    assert isinstance(sim.model, AgentModel)


# Calibration check: with everyone susceptible at t=0, the agent model's implied
# Rt should equal R0. Looser rel=2e-2 (2%) tolerance because Rt is estimated from
# a finite, discrete population of agents rather than a smooth equation.
def test_rt_at_t0_matches_r0():
    scenario = _scenario(overdispersion=None)
    model = AgentModel(scenario)
    assert model.effective_rt(0.0) == pytest.approx(scenario.disease.r0, rel=2e-2)


# The model simulates 40k agents but represents a 200k population, so reported
# counts are scaled up by the agent->population factor. Checks both the initial
# total and the per-compartment sum after one step equal the configured 200k.
def test_reported_population_matches_configuration():
    scenario = _scenario(n_agents=40_000, total_population=200_000)
    model = AgentModel(scenario)
    # Scaling reports counts at population (not agent) scale.
    assert model.total_population() == pytest.approx(200_000, rel=1e-9)
    rec = model.step()
    total = (rec.S + rec.V + rec.E + rec.Ip + rec.Ia + rec.Is
             + rec.H + rec.C + rec.R + rec.D)
    assert total == pytest.approx(200_000, rel=1e-9)


# Conservation: agents only change state, never appear or vanish, so the scaled
# total is constant across 150 steps (tight rel=1e-9). Also confirms n_agents
# equals the actual size of the underlying state array.
def test_population_conserved_over_run():
    scenario = _scenario(seed=1)
    model = AgentModel(scenario)
    total0 = model.total_population()
    for _ in range(150):
        model.step()
    # Agents are only ever moved between states, never created/destroyed.
    assert model.total_population() == pytest.approx(total0, rel=1e-9)
    assert model.n_agents == int(model.state.size)


# When requested n_agents exceeds the population, the model can't have more
# agents than people, so it caps at the population size and the scale factor
# collapses to 1.0 (one agent per person, no up-scaling).
def test_small_population_uses_full_fidelity():
    """When n_agents >= population, every person is an agent (scale == 1)."""
    scenario = _scenario(n_agents=500_000, total_population=20_000)
    model = AgentModel(scenario)
    assert model.n_agents == 20_000
    assert model.scale == pytest.approx(1.0)


# ------------------------------------------------------------- epidemic shape
# Classic outbreak shape for the agent engine: infectious count rises to an
# interior peak (not at the first or last step) then burns out to under 10% of
# the peak. Lifelong immunity (waning=None) ensures a clean single wave. The
# inner generator expression steps the model and collects infectious totals.
def test_epidemic_grows_then_burns_out():
    scenario = _scenario(seed=2, duration=300)
    scenario.disease.waning_immunity_days = None
    scenario = scenario.validate()
    model = AgentModel(scenario)
    infectious = [rec.Ip + rec.Ia + rec.Is for rec in (model.step() for _ in range(300))]
    infectious = np.array(infectious)
    peak_idx = int(np.argmax(infectious))
    assert 0 < peak_idx < len(infectious) - 1
    assert infectious[-1] < 0.1 * infectious[peak_idx]


# With R0 < 1 each case infects fewer than one other on average, so the outbreak
# dies out: the cumulative attack rate over the whole run stays under 5%.
def test_subcritical_r0_fails_to_take_off():
    scenario = _scenario(seed=4)
    scenario.disease.r0 = 0.6
    scenario = scenario.validate()
    model = AgentModel(scenario)
    total0 = model.total_population()
    new = sum(model.step().new_infections for _ in range(scenario.simulation.n_steps))
    assert new / total0 < 0.05


def test_agent_matches_compartmental_final_size():
    """In the mean-field limit the agent engine reproduces the compartmental
    attack rate (it is a stochastic realisation of the same model)."""
    def mean_attack(engine):
        scenario = _scenario(engine=engine, n_agents=100_000, duration=300,
                             stochastic=True, overdispersion=None)
        scenario.disease.waning_immunity_days = None  # lifelong => true final size
        scenario = scenario.validate()
        histories = run_ensemble(scenario, n_runs=8, base_seed=0)
        return np.mean([summarize(h, scenario.disease.r0).attack_rate for h in histories])

    # mean_attack averages the attack rate across an 8-run ensemble for a given
    # engine; the agent engine is a stochastic realisation of the same model, so
    # its mean final size should match the compartmental one within 5%.
    agent = mean_attack("agent")
    compartmental = mean_attack("compartmental")
    assert agent == pytest.approx(compartmental, rel=0.05)


# --------------------------------------------------------------- mechanisms
# A transmission-reducing intervention must lower the final attack rate. The
# inner attack_rate(reduction) reruns the sim with (or without) a distancing
# intervention; the test compares a 40% reduction against none.
def test_interventions_reduce_attack_rate():
    def attack_rate(reduction):
        scenario = _scenario(seed=5, duration=300)
        scenario.disease.waning_immunity_days = None
        if reduction:
            scenario.interventions = InterventionConfig(
                [Intervention("distancing", 0, 300, reduction)]
            )
        scenario = scenario.validate()
        model = AgentModel(scenario)
        total0 = model.total_population()
        new = sum(model.step().new_infections for _ in range(300))
        return new / total0

    assert attack_rate(0.4) < attack_rate(0.0)


# Vaccination must lower the final attack rate. attack_rate(vaccinate) reruns the
# sim with or without a rollout (1%/day up to 80% coverage, 80% efficacy against
# infection); the test asserts the vaccinated run ends with fewer infections.
def test_vaccination_reduces_attack_rate():
    def attack_rate(vaccinate):
        scenario = _scenario(seed=6, duration=300)
        scenario.disease.waning_immunity_days = None
        if vaccinate:
            scenario.vaccination = VaccinationConfig(
                enabled=True, start_day=0, daily_rate=0.02, coverage_cap=0.8,
                ve_susceptibility=0.8,
            )
        scenario = scenario.validate()
        model = AgentModel(scenario)
        total0 = model.total_population()
        new = sum(model.step().new_infections for _ in range(300))
        return new / total0

    assert attack_rate(True) < attack_rate(False)


# When ICU demand exceeds capacity, overflow patients die at a higher rate
# (mortality multiplier 3.0). deaths(capacity) counts agents in the D (dead)
# state; a tight capacity of 50 must yield more deaths than unlimited (None).
def test_icu_overflow_increases_deaths():
    def deaths(capacity):
        scenario = ScenarioConfig(
            population=PopulationConfig(total_population=300_000, initial_infected=600),
            disease=preset_scenario("covid_like").disease,
            simulation=SimulationConfig(duration_days=300, engine="agent",
                                        n_agents=100_000, seed=11, overdispersion=None),
        )
        scenario.healthcare.icu_capacity = capacity
        scenario.healthcare.overflow_mortality_multiplier = 3.0
        scenario = scenario.validate()
        model = AgentModel(scenario)
        for _ in range(300):
            model.step()
        return float((model.state == D).sum())

    assert deaths(capacity=50) > deaths(capacity=None)


def test_superspreading_heterogeneity_assigned():
    """With overdispersion enabled, infected agents carry heterogeneous loads."""
    scenario = _scenario(seed=7, overdispersion=0.3)
    model = AgentModel(scenario)
    for _ in range(20):
        model.step()
    # Overdispersion models superspreading: individuals carry different
    # infectivity ("load"). np.isin masks agents currently infectious (any of the
    # IP/IA/IS state codes), then we check their loads genuinely vary (std > 0.1)
    # rather than all being identical.
    infected = np.isin(model.state, (IP, IA, IS))
    loads = model.infectivity[infected]
    assert loads.size > 0
    assert loads.std() > 0.1  # genuine individual-level variation


# ------------------------------------------------------- determinism & state
# Determinism: two agent models built with the same seed produce byte-for-byte
# identical trajectories (exact == on the new-infections lists).
def test_seed_reproducibility():
    m1 = AgentModel(_scenario(seed=42))
    m2 = AgentModel(_scenario(seed=42))
    a = [m1.step().new_infections for _ in range(80)]
    b = [m2.step().new_infections for _ in range(80)]
    assert a == b


# The flip side: different seeds must drive different stochastic trajectories.
def test_different_seeds_diverge():
    m1 = AgentModel(_scenario(seed=1))
    m2 = AgentModel(_scenario(seed=2))
    a = [m1.step().new_infections for _ in range(80)]
    b = [m2.step().new_infections for _ in range(80)]
    assert a != b


# In-memory state round-trip: get_state() captures the full model state (incl.
# RNG); a fresh clone fed that state via set_state() must continue identically to
# the original from that point on.
def test_state_roundtrip_resumes_exactly():
    model = AgentModel(_scenario(seed=9))
    for _ in range(60):
        model.step()
    state = model.get_state()

    clone = AgentModel(_scenario(seed=9))
    clone.set_state(state)
    a = [model.step().new_infections for _ in range(25)]
    b = [clone.step().new_infections for _ in range(25)]
    assert a == b


# Persisted save/load round-trip for an agent run. tmp_path is pytest's per-test
# temp-dir fixture. The reloaded sim must be an AgentModel and continue exactly
# like the original (RNG and agent state survive the JSON round-trip).
def test_simulation_save_load_roundtrip(tmp_path):
    sim = Simulation(_scenario(seed=3))
    sim.run(steps=40)
    path = tmp_path / "agent_scenario.json"
    sim.save(str(path))

    restored = Simulation.load(str(path))
    assert isinstance(restored.model, AgentModel)
    a = [sim.step().new_infections for _ in range(15)]
    b = [restored.step().new_infections for _ in range(15)]
    assert a == b


# ------------------------------------------------------------- config guards
# An unrecognised engine name must be rejected at validate() with ValueError.
# pytest.raises asserts the block raises that exception (fails otherwise).
def test_invalid_engine_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(engine="bogus").validate()


# n_agents must be positive; zero agents is invalid and must raise ValueError.
def test_nonpositive_n_agents_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(n_agents=0).validate()
