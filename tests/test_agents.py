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
def test_engine_dispatch_builds_agent_model():
    sim = Simulation(_scenario())
    assert isinstance(sim.model, AgentModel)


def test_rt_at_t0_matches_r0():
    scenario = _scenario(overdispersion=None)
    model = AgentModel(scenario)
    assert model.effective_rt(0.0) == pytest.approx(scenario.disease.r0, rel=2e-2)


def test_reported_population_matches_configuration():
    scenario = _scenario(n_agents=40_000, total_population=200_000)
    model = AgentModel(scenario)
    # Scaling reports counts at population (not agent) scale.
    assert model.total_population() == pytest.approx(200_000, rel=1e-9)
    rec = model.step()
    total = (rec.S + rec.V + rec.E + rec.Ip + rec.Ia + rec.Is
             + rec.H + rec.C + rec.R + rec.D)
    assert total == pytest.approx(200_000, rel=1e-9)


def test_population_conserved_over_run():
    scenario = _scenario(seed=1)
    model = AgentModel(scenario)
    total0 = model.total_population()
    for _ in range(150):
        model.step()
    # Agents are only ever moved between states, never created/destroyed.
    assert model.total_population() == pytest.approx(total0, rel=1e-9)
    assert model.n_agents == int(model.state.size)


def test_small_population_uses_full_fidelity():
    """When n_agents >= population, every person is an agent (scale == 1)."""
    scenario = _scenario(n_agents=500_000, total_population=20_000)
    model = AgentModel(scenario)
    assert model.n_agents == 20_000
    assert model.scale == pytest.approx(1.0)


# ------------------------------------------------------------- epidemic shape
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

    agent = mean_attack("agent")
    compartmental = mean_attack("compartmental")
    assert agent == pytest.approx(compartmental, rel=0.05)


# --------------------------------------------------------------- mechanisms
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
    infected = np.isin(model.state, (IP, IA, IS))
    loads = model.infectivity[infected]
    assert loads.size > 0
    assert loads.std() > 0.1  # genuine individual-level variation


# ------------------------------------------------------- determinism & state
def test_seed_reproducibility():
    m1 = AgentModel(_scenario(seed=42))
    m2 = AgentModel(_scenario(seed=42))
    a = [m1.step().new_infections for _ in range(80)]
    b = [m2.step().new_infections for _ in range(80)]
    assert a == b


def test_different_seeds_diverge():
    m1 = AgentModel(_scenario(seed=1))
    m2 = AgentModel(_scenario(seed=2))
    a = [m1.step().new_infections for _ in range(80)]
    b = [m2.step().new_infections for _ in range(80)]
    assert a != b


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
def test_invalid_engine_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(engine="bogus").validate()


def test_nonpositive_n_agents_rejected():
    with pytest.raises(ValueError):
        SimulationConfig(n_agents=0).validate()
