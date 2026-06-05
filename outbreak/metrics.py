"""Derived epidemiological metrics computed from a simulation history.

The simulation stores one :class:`~outbreak.model.StepRecord` per time step.
This module turns that raw history into the headline numbers epidemiologists
care about: the epidemic curve, cumulative incidence (attack rate), peak timing
and height, healthcare burden, and the dates the effective reproduction number
crosses one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .model import StepRecord


@dataclass
class EpidemicSummary:
    """Headline statistics for a completed (or partial) epidemic run."""

    total_population: float
    duration_days: float

    total_infections: float
    # Cumulative incidence ratio: total infections / population. Equals the
    # fraction ever infected when immunity is lifelong, but can exceed 1 when
    # waning immunity allows reinfection.
    attack_rate: float
    total_symptomatic: float
    total_hospitalizations: float
    total_icu: float
    total_deaths: float
    infection_fatality_ratio: float    # deaths / infections

    peak_infectious: float
    peak_infectious_day: Optional[float]
    peak_hospital_occupancy: float
    peak_hospital_day: Optional[float]
    peak_icu_occupancy: float
    peak_icu_day: Optional[float]

    peak_daily_infections: float
    peak_daily_infections_day: Optional[float]

    r0: float
    peak_rt: float
    rt_crossed_one_day: Optional[float]   # first day Rt drops below 1
    epidemic_over_day: Optional[float]    # first day with zero active infections

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _series(history: Sequence[StepRecord], attr: str) -> np.ndarray:
    # Pull one attribute off every record into a 1-D float array (a column /
    # time series). getattr lets us select the field by name at runtime.
    return np.array([getattr(r, attr) for r in history], dtype=float)


def summarize(history: Sequence[StepRecord], r0: float) -> Optional[EpidemicSummary]:
    """Compute an :class:`EpidemicSummary` from a list of step records."""
    if not history:
        return None  # nothing to summarise for an empty run

    # Extract each tracked quantity as its own aligned time-series array.
    days = _series(history, "day")
    new_inf = _series(history, "new_infections")
    new_sym = _series(history, "new_symptomatic")
    new_hosp = _series(history, "new_hospitalizations")
    new_icu = _series(history, "new_icu")
    new_deaths = _series(history, "new_deaths")
    rt = _series(history, "rt")

    # Arrays add element-wise, so these are per-step sums across compartments.
    infectious = _series(history, "Ip") + _series(history, "Ia") + _series(history, "Is")
    active = infectious + _series(history, "E") + _series(history, "H") + _series(history, "C")
    hosp_occ = _series(history, "H")
    icu_occ = _series(history, "C")

    # Total population is conserved, so summing the last record's compartments
    # recovers it (no need to read the original config).
    last = history[-1]
    total_pop = (
        last.S + last.V + last.E + last.Ip + last.Ia + last.Is
        + last.H + last.C + last.R + last.D
    )

    total_infections = float(new_inf.sum())  # sum of per-step new infections
    total_deaths = float(_series(history, "D")[-1])  # cumulative deaths = current D

    def _peak(values: np.ndarray):
        # Return (peak value, day of peak); (0.0, None) for an all-zero/empty series.
        if values.size == 0 or np.all(values == 0):
            return 0.0, None
        idx = int(np.argmax(values))  # argmax = index of the first maximum
        return float(values[idx]), float(days[idx])  # map that index back to a day

    peak_infectious, peak_infectious_day = _peak(infectious)
    peak_hosp, peak_hosp_day = _peak(hosp_occ)
    peak_icu, peak_icu_day = _peak(icu_occ)
    peak_daily_inf, peak_daily_inf_day = _peak(new_inf)

    # First day Rt falls below 1 after the epidemic has started growing.
    rt_cross = None
    # np.where(...) returns a tuple of index arrays; [0] takes the first axis.
    started = np.where(new_inf > 0)[0]
    if started.size:  # there was at least one day with new infections
        for k in range(started[0], len(rt)):  # scan forward from the first such day
            if rt[k] < 1.0:
                rt_cross = float(days[k])
                break

    # First day with no active infections after the epidemic began.
    over_day = None
    if started.size:
        for k in range(started[0] + 1, len(active)):
            if active[k] <= 0.5:  # fewer than one expected active case
                over_day = float(days[k])
                break

    return EpidemicSummary(
        total_population=total_pop,
        duration_days=float(days[-1]),
        total_infections=total_infections,
        attack_rate=total_infections / total_pop if total_pop > 0 else 0.0,
        total_symptomatic=float(new_sym.sum()),
        total_hospitalizations=float(new_hosp.sum()),
        total_icu=float(new_icu.sum()),
        total_deaths=total_deaths,
        infection_fatality_ratio=(
            total_deaths / total_infections if total_infections > 0 else 0.0
        ),
        peak_infectious=peak_infectious,
        peak_infectious_day=peak_infectious_day,
        peak_hospital_occupancy=peak_hosp,
        peak_hospital_day=peak_hosp_day,
        peak_icu_occupancy=peak_icu,
        peak_icu_day=peak_icu_day,
        peak_daily_infections=peak_daily_inf,
        peak_daily_infections_day=peak_daily_inf_day,
        r0=r0,
        peak_rt=float(rt.max()) if rt.size else 0.0,
        rt_crossed_one_day=rt_cross,
        epidemic_over_day=over_day,
    )


def history_to_columns(history: Sequence[StepRecord]) -> dict:
    """Return the full time series as a dict of equal-length lists.

    Convenient for building a DataFrame, plotting, or CSV export without taking
    a hard dependency on pandas in the core package.
    """
    fields = [
        "day", "S", "V", "E", "Ip", "Ia", "Is", "H", "C", "R", "D",
        "new_infections", "new_symptomatic", "new_hospitalizations",
        "new_icu", "new_deaths", "rt", "beta_effective", "icu_overflow",
    ]
    # Dict comprehension: one named column (plain list) per field.
    cols = {f: _series(history, f).tolist() for f in fields}
    # Convenience aggregates.
    cols["infectious"] = (
        _series(history, "Ip") + _series(history, "Ia") + _series(history, "Is")
    ).tolist()
    cols["active"] = (
        np.array(cols["infectious"])
        + _series(history, "E") + _series(history, "H") + _series(history, "C")
    ).tolist()
    # cumsum gives the running total of new infections (cumulative incidence).
    cols["cumulative_infections"] = np.cumsum(_series(history, "new_infections")).tolist()
    return cols


def aggregate_ensemble(
    histories: Sequence[Sequence[StepRecord]],
    attr: str,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict:
    """Aggregate one variable across many stochastic runs.

    Returns ``{"day": [...], "q0.05": [...], "q0.5": [...], ...}`` where each
    quantile band is computed pointwise across runs. Runs are truncated to the
    shortest length so the bands align.
    """
    if not histories:
        return {"day": []}
    # Runs may differ in length; truncate every run to the shortest so columns
    # line up step-for-step.
    min_len = min(len(h) for h in histories)
    days = _series(histories[0][:min_len], "day")
    # Stack the per-run series into a 2-D array, shape (n_runs, min_len): rows
    # are runs, columns are time steps.
    stacked = np.vstack([_series(h[:min_len], attr) for h in histories])
    out = {"day": days.tolist()}
    for q in quantiles:
        # axis=0 reduces across runs, giving one quantile value per time step
        # (i.e. a pointwise uncertainty band over time).
        out[f"q{q}"] = np.quantile(stacked, q, axis=0).tolist()
    out["mean"] = stacked.mean(axis=0).tolist()  # pointwise mean across runs
    return out
