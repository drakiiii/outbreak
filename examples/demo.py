"""A headless demonstration of the Outbreak engine (no UI required).

Run with::

    python examples/demo.py

It runs three scenarios — unmitigated, with an intervention, and with
vaccination — and prints headline statistics for each, illustrating the API.
"""

from outbreak import Simulation, preset_scenario
from outbreak.config import InterventionConfig, VaccinationConfig
from outbreak.interventions import lockdown


def show(title, summary):
    print(f"\n=== {title} ===")
    print(f"  Cumulative infections : {summary.total_infections:,.0f} "
          f"({100 * summary.attack_rate:.1f}% of population)")
    print(f"  Deaths                : {summary.total_deaths:,.0f} "
          f"(IFR {100 * summary.infection_fatality_ratio:.2f}%)")
    print(f"  Peak infectious       : {summary.peak_infectious:,.0f} "
          f"on day {summary.peak_infectious_day:.0f}")
    print(f"  Peak ICU occupancy    : {summary.peak_icu_occupancy:,.0f}")
    if summary.rt_crossed_one_day is not None:
        print(f"  Rt fell below 1 on day: {summary.rt_crossed_one_day:.0f}")


def main():
    population = 1_000_000

    # 1. Unmitigated epidemic.
    base = preset_scenario("covid_like", total_population=population)
    sim = Simulation(base)
    sim.run_to_end()
    show("Unmitigated", sim.summary())

    # 2. With a 6-week lockdown starting on day 40.
    mitigated = preset_scenario("covid_like", total_population=population)
    mitigated.interventions = InterventionConfig([lockdown(start_day=40, end_day=82)])
    sim = Simulation(mitigated.validate())
    sim.run_to_end()
    show("With a 6-week lockdown (day 40-82)", sim.summary())

    # 3. With a vaccination campaign.
    vaccinated = preset_scenario("covid_like", total_population=population)
    vaccinated.vaccination = VaccinationConfig(
        enabled=True, start_day=20, daily_rate=0.01, coverage_cap=0.75,
        ve_susceptibility=0.7, ve_severity=0.85,
    )
    sim = Simulation(vaccinated.validate())
    sim.run_to_end()
    show("With vaccination (1%/day from day 20)", sim.summary())


if __name__ == "__main__":
    main()
