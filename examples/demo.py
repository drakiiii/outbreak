"""A headless demonstration of the Outbreak engine (no UI required).

Run with::

    python examples/demo.py

It runs three scenarios — unmitigated, with an intervention, and with
vaccination — and prints headline statistics for each, illustrating the API.
Finally it runs the same scenario through both engines (compartmental and
agent-based) to show they agree.
"""

import numpy as np

from outbreak import Simulation, preset_scenario, run_ensemble
from outbreak.config import InterventionConfig, NetworkConfig, SimulationConfig, VaccinationConfig
from outbreak.interventions import lockdown
from outbreak.metrics import summarize


def show(title, summary):
    # Pretty-print the headline statistics from a SimulationSummary object.
    # `summary` is whatever sim.summary() / summarize() returns; the attributes
    # accessed below (total_infections, attack_rate, ...) are its fields.
    print(f"\n=== {title} ===")
    print(f"  Cumulative infections : {summary.total_infections:,.0f} "
          f"({100 * summary.attack_rate:.1f}% of population)")
    print(f"  Deaths                : {summary.total_deaths:,.0f} "
          f"(IFR {100 * summary.infection_fatality_ratio:.2f}%)")
    print(f"  Peak infectious       : {summary.peak_infectious:,.0f} "
          f"on day {summary.peak_infectious_day:.0f}")
    print(f"  Peak ICU occupancy    : {summary.peak_icu_occupancy:,.0f}")
    # Only some runs ever push Rt below 1 (e.g. a fading epidemic); skip the
    # line entirely when the engine never recorded that crossing.
    if summary.rt_crossed_one_day is not None:
        print(f"  Rt fell below 1 on day: {summary.rt_crossed_one_day:.0f}")


def main():
    population = 1_000_000

    # 1. Unmitigated epidemic.
    # preset_scenario returns a ready-made ScenarioConfig; Simulation wraps it,
    # run_to_end() advances the engine to the configured duration, and
    # sim.summary() collapses the full history into headline stats.
    base = preset_scenario("covid_like", total_population=population)
    sim = Simulation(base)
    sim.run_to_end()
    show("Unmitigated", sim.summary())

    # 2. With a 6-week lockdown starting on day 40.
    # Same preset, but we overwrite its interventions with a single lockdown
    # window. .validate() re-checks the mutated config before it is run.
    mitigated = preset_scenario("covid_like", total_population=population)
    mitigated.interventions = InterventionConfig([lockdown(start_day=40, end_day=82)])
    sim = Simulation(mitigated.validate())
    sim.run_to_end()
    show("With a 6-week lockdown (day 40-82)", sim.summary())

    # 3. With a vaccination campaign.
    # Again the same preset, this time with vaccination switched on instead of
    # an intervention, to compare its effect on the same baseline disease.
    vaccinated = preset_scenario("covid_like", total_population=population)
    vaccinated.vaccination = VaccinationConfig(
        enabled=True, start_day=20, daily_rate=0.01, coverage_cap=0.75,
        ve_susceptibility=0.7, ve_severity=0.85,
    )
    sim = Simulation(vaccinated.validate())
    sim.run_to_end()
    show("With vaccination (1%/day from day 20)", sim.summary())

    # 4. The same disease through both engines: the agent-based model is a
    #    stochastic realisation of the compartmental one, so their average
    #    final sizes agree.
    print("\n=== Compartmental vs agent-based engine (12-run mean) ===")
    # Loop over the two engine keys, building an identical scenario each time and
    # only swapping the `engine` field, so the comparison isolates the engine.
    for engine in ("compartmental", "agent"):
        scenario = preset_scenario("influenza_like", total_population=population)
        scenario.population.initial_infected = 500
        scenario.disease.waning_immunity_days = None  # lifelong => true final size
        # Rebuild the simulation section wholesale to pin duration/engine/agents.
        scenario.simulation = SimulationConfig(
            duration_days=300, engine=engine, n_agents=100_000, overdispersion=None,
        )
        # run_ensemble returns a list of histories (one per stochastic run). We
        # summarize each and average the metric across the 12 runs so the two
        # engines can be compared on their mean behaviour rather than one draw.
        histories = run_ensemble(scenario.validate(), n_runs=12, base_seed=0)
        attack = np.mean([summarize(h, scenario.disease.r0).attack_rate for h in histories])
        peak = np.mean([summarize(h, scenario.disease.r0).peak_infectious for h in histories])
        # {engine:<14} left-pads the engine name to 14 chars for column alignment.
        print(f"  {engine:<14}: attack rate {100 * attack:5.1f}%   "
              f"mean peak infectious {peak:,.0f}")

    # 5. Contact networks (agent engine): households/schools/workplaces cluster
    #    transmission. For the same R0 this flattens the peak versus mean-field
    #    mixing, even though both are calibrated to the same target R0.
    print("\n=== Agent engine: mean-field vs contact network (8-run mean) ===")
    for enabled in (False, True):
        scenario = preset_scenario("covid_like", total_population=population)
        scenario.population.initial_infected = 500
        scenario.disease.r0 = 1.8
        scenario.disease.waning_immunity_days = None
        scenario.network = NetworkConfig(enabled=enabled)
        scenario.simulation = SimulationConfig(
            duration_days=400, engine="agent", n_agents=100_000, overdispersion=None,
        )
        histories = run_ensemble(scenario.validate(), n_runs=8, base_seed=0)
        attack = np.mean([summarize(h, scenario.disease.r0).attack_rate for h in histories])
        peak = np.mean([summarize(h, scenario.disease.r0).peak_infectious for h in histories])
        peak_day = np.mean([summarize(h, scenario.disease.r0).peak_infectious_day
                            for h in histories])
        label = "contact network" if enabled else "mean-field"
        print(f"  {label:<16}: attack rate {100 * attack:5.1f}%   "
              f"peak {peak:,.0f} on day {peak_day:.0f}")

    # 6. Stage-duration realism (agent engine): exponential vs peaked sojourn
    #    times. Same R0 and (statistically) the same final size, but peaked
    #    durations give a sharper, earlier wave.
    print("\n=== Agent engine: exponential vs realistic stage durations (8-run mean) ===")
    for dispersion in (1.0, 4.0):
        scenario = preset_scenario("covid_like", total_population=population)
        scenario.population.initial_infected = 500
        scenario.disease.waning_immunity_days = None
        scenario.disease.duration_dispersion = dispersion
        scenario.simulation = SimulationConfig(
            duration_days=300, engine="agent", n_agents=100_000, overdispersion=None,
        )
        histories = run_ensemble(scenario.validate(), n_runs=8, base_seed=0)
        attack = np.mean([summarize(h, scenario.disease.r0).attack_rate for h in histories])
        peak = np.mean([summarize(h, scenario.disease.r0).peak_infectious for h in histories])
        peak_day = np.mean([summarize(h, scenario.disease.r0).peak_infectious_day
                            for h in histories])
        label = "exponential (k=1)" if dispersion == 1.0 else "peaked (k=4)"
        print(f"  {label:<18}: attack rate {100 * attack:5.1f}%   "
              f"peak {peak:,.0f} on day {peak_day:.0f}")


if __name__ == "__main__":
    main()
