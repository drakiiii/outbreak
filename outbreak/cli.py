"""Command-line interface for Outbreak.

Run a simulation without writing any Python or opening the web app::

    outbreak --preset covid_like --population 1000000 --days 365
    python -m outbreak --preset covid_like --csv results.csv      # equivalent

It prints a headline summary to the screen and can optionally write the full
day-by-day time series to CSV and/or the summary to JSON. With ``--ensemble N``
it runs ``N`` random repeats and reports the median and range across them.

The CLI is a thin wrapper over the same public API used everywhere else
(:class:`~outbreak.simulation.Simulation`, :func:`~outbreak.simulation.run_ensemble`,
:func:`~outbreak.metrics.summarize`), so its results match the library and UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import List, Optional, Sequence

import numpy as np

from .config import DISEASE_PRESETS, NetworkConfig, ScenarioConfig, preset_scenario
from .metrics import summarize
from .simulation import Simulation, run_ensemble


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="outbreak",
        description="Run an epidemic simulation and report the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Where the scenario comes from: a saved file, or a built-in preset + tweaks.
    src = p.add_argument_group("scenario")
    src.add_argument(
        "--scenario", metavar="PATH",
        help="Load a scenario (or a saved run snapshot) from a JSON file. "
             "When given, the preset/override options below are ignored.",
    )
    src.add_argument(
        "--preset", choices=sorted(DISEASE_PRESETS), default="covid_like",
        help="Built-in disease to start from.",
    )
    src.add_argument("--population", type=int, default=1_000_000,
                     help="Total number of people.")
    src.add_argument("--initial-infected", type=int, default=None,
                     help="Number of infections seeded at day 0.")
    src.add_argument("--r0", type=float, default=None,
                     help="Override the disease's basic reproduction number R0.")

    # How to run it.
    run = p.add_argument_group("simulation")
    run.add_argument("--days", type=int, default=None, help="Duration in days.")
    run.add_argument("--engine", choices=("compartmental", "agent"), default=None,
                     help="Fast counts-based engine, or detailed person-by-person engine.")
    run.add_argument("--n-agents", type=int, default=None,
                     help="(agent engine) number of individuals to simulate.")
    run.add_argument("--network", action="store_true",
                     help="(agent engine) enable household/school/workplace contacts.")
    run.add_argument("--no-stochastic", action="store_true",
                     help="Run the smooth, deterministic (luck-free) dynamics.")
    run.add_argument("--seed", type=int, default=None,
                     help="Random seed, for repeatable runs.")
    run.add_argument("--ensemble", type=int, default=None, metavar="N",
                     help="Run N random repeats and report the median and range.")
    run.add_argument("--ascertainment", type=float, default=None,
                     help="Surveillance: fraction of symptomatic cases reported (0-1).")
    run.add_argument("--reporting-delay", type=float, default=None,
                     help="Surveillance: mean days from symptom onset to report.")

    # What to do with the results.
    out = p.add_argument_group("output")
    out.add_argument("--csv", metavar="PATH",
                     help="Write the day-by-day time series (single run) or per-run "
                          "summaries (ensemble) to a CSV file.")
    out.add_argument("--json", metavar="PATH", dest="json_path",
                     help="Write the summary statistics to a JSON file.")
    out.add_argument("--quiet", action="store_true",
                     help="Don't print the summary to the screen.")

    p.add_argument("--list-presets", action="store_true",
                   help="List the built-in disease presets and exit.")
    return p


def _build_scenario(args: argparse.Namespace) -> ScenarioConfig:
    """Construct a validated scenario from a file or a preset plus overrides."""
    if args.scenario is not None:
        with open(args.scenario, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # A run snapshot carries an "engine_state"; a plain scenario does not.
        if isinstance(data, dict) and "engine_state" in data:
            return ScenarioConfig.from_dict(data["config"])
        return ScenarioConfig.from_dict(data)

    # Build from a preset, then apply any explicit overrides.
    scenario = preset_scenario(args.preset, total_population=args.population)

    if args.initial_infected is not None:
        scenario.population.initial_infected = args.initial_infected
    if args.r0 is not None:
        scenario.disease.r0 = args.r0

    sim = scenario.simulation
    if args.days is not None:
        sim.duration_days = args.days
    if args.engine is not None:
        sim.engine = args.engine
    if args.n_agents is not None:
        sim.n_agents = args.n_agents
    if args.seed is not None:
        sim.seed = args.seed
    if args.no_stochastic:
        sim.stochastic = False

    if args.network:
        scenario.network = NetworkConfig(enabled=True)
        if sim.engine != "agent":
            print("note: --network only affects the agent engine; ignoring for "
                  f"the {sim.engine} engine.", file=sys.stderr)

    if args.ascertainment is not None:
        scenario.reporting.ascertainment = args.ascertainment
    if args.reporting_delay is not None:
        scenario.reporting.reporting_delay_days = args.reporting_delay

    return scenario.validate()


def _summary_lines(summary) -> List[str]:
    """Human-readable headline block for a single run."""
    s = summary
    lines = [
        f"  Population            : {s.total_population:,.0f}",
        f"  Duration             : {s.duration_days:.0f} days",
        f"  Total infections     : {s.total_infections:,.0f} "
        f"({100 * s.attack_rate:.1f}% of population)",
        f"  Symptomatic cases    : {s.total_symptomatic:,.0f}",
        f"  Reported cases       : {s.total_reported:,.0f}",
        f"  Hospitalisations     : {s.total_hospitalizations:,.0f}",
        f"  ICU admissions       : {s.total_icu:,.0f}",
        f"  Deaths               : {s.total_deaths:,.0f} "
        f"(IFR {100 * s.infection_fatality_ratio:.2f}%)",
        f"  Peak infectious      : {s.peak_infectious:,.0f}"
        + (f" on day {s.peak_infectious_day:.0f}" if s.peak_infectious_day is not None else ""),
        f"  Peak ICU occupancy   : {s.peak_icu_occupancy:,.0f}",
        f"  Peak Rt              : {s.peak_rt:.2f}",
    ]
    if s.rt_crossed_one_day is not None:
        lines.append(f"  Rt fell below 1 on   : day {s.rt_crossed_one_day:.0f}")
    if s.epidemic_over_day is not None:
        lines.append(f"  Epidemic over on     : day {s.epidemic_over_day:.0f}")
    return lines


def _write_timeseries_csv(cols: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols.keys())                 # header row
        writer.writerows(zip(*cols.values()))        # transpose dict-of-lists into rows


def _run_single(scenario: ScenarioConfig, args: argparse.Namespace) -> int:
    sim = Simulation(scenario)
    sim.run_to_end()
    summary = sim.summary()

    if not args.quiet:
        print(f"\n=== {scenario.disease.name} · {scenario.simulation.engine} engine ===")
        print("\n".join(_summary_lines(summary)))

    if args.csv:
        _write_timeseries_csv(sim.to_columns(), args.csv)   # includes reported_cases
        if not args.quiet:
            print(f"\nWrote time series ({len(sim.history)} rows) to {args.csv}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(summary.as_dict(), fh, indent=2)
        if not args.quiet:
            print(f"Wrote summary to {args.json_path}")
    return 0


def _run_ensemble(scenario: ScenarioConfig, args: argparse.Namespace) -> int:
    n = args.ensemble
    if n <= 0:
        print("error: --ensemble must be a positive integer", file=sys.stderr)
        return 2
    histories = run_ensemble(scenario, n_runs=n, base_seed=scenario.simulation.seed)
    summaries = [summarize(h, scenario.disease.r0) for h in histories]

    def stat(attr):
        vals = np.array([getattr(s, attr) for s in summaries], dtype=float)
        return float(np.median(vals)), float(vals.min()), float(vals.max())

    metrics = [
        ("Attack rate (%)", "attack_rate", 100.0),
        ("Total infections", "total_infections", 1.0),
        ("Deaths", "total_deaths", 1.0),
        ("Peak infectious", "peak_infectious", 1.0),
        ("Peak ICU occupancy", "peak_icu_occupancy", 1.0),
    ]
    if not args.quiet:
        print(f"\n=== {scenario.disease.name} · {scenario.simulation.engine} engine "
              f"· {n}-run ensemble ===")
        print(f"  {'metric':<20} {'median':>14}   {'range (min–max)':>26}")
        for label, attr, mult in metrics:
            med, lo, hi = (v * mult for v in stat(attr))
            print(f"  {label:<20} {med:>14,.1f}   {lo:>11,.1f} – {hi:>11,.1f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].as_dict().keys()))
            writer.writeheader()
            for s in summaries:
                writer.writerow(s.as_dict())
        if not args.quiet:
            print(f"\nWrote {n} per-run summaries to {args.csv}")
    if args.json_path:
        agg = {label: dict(zip(("median", "min", "max"), stat(attr)))
               for label, attr, _ in metrics}
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2)
        if not args.quiet:
            print(f"Wrote ensemble statistics to {args.json_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    args = build_parser().parse_args(argv)

    if args.list_presets:
        print("Available disease presets:")
        for name in sorted(DISEASE_PRESETS):
            print(f"  {name}")
        return 0

    try:
        scenario = _build_scenario(args)
    except FileNotFoundError:
        print(f"error: scenario file not found: {args.scenario}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: could not build scenario: {exc}", file=sys.stderr)
        return 2

    if args.ensemble is not None:
        return _run_ensemble(scenario, args)
    return _run_single(scenario, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
