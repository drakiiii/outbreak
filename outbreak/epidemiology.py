"""Shared epidemiological math used by every engine.

Both the compartmental engine (:mod:`outbreak.model`) and the agent-based engine
(:mod:`outbreak.agents`) describe the *same* disease with the *same* natural
history and the *same* R0 calibration. Keeping that math in one place guarantees
the two engines are directly comparable: given identical configuration they
calibrate to an identical transmission rate ``beta`` and report the effective
reproduction number ``Rt`` the same way.

Nothing here owns mutable simulation state; these are pure functions and an
immutable bundle of resolved parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScenarioConfig


@dataclass
class ResolvedParams:
    """Disease parameters resolved to plain arrays/scalars for one scenario.

    Produced once at engine construction from a :class:`ScenarioConfig`. The
    per-step transition *rates* (per day) are converted to per-step
    probabilities by the engines via :func:`transition_probability`.
    """

    # Transition rates (per day).
    sigma: float          # E -> infectious
    gamma_p: float        # Ip -> Is
    gamma_a: float        # Ia -> R
    gamma_s: float        # Is -> H/R
    gamma_h: float        # H -> C/R
    gamma_c: float        # C -> D/R
    omega: float          # R -> S (waning); 0 disables

    # Branching probabilities.
    p_asymp: np.ndarray       # (n_age,)  of infections that are asymptomatic
    hosp_rate: np.ndarray     # (2, n_age) of symptomatic hospitalised, per stratum
    icu_rate: np.ndarray      # (n_age,)  of hospitalised needing ICU
    death_rate: np.ndarray    # (n_age,)  of ICU patients who die (baseline)

    # Infectiousness.
    rel_p: float                      # relative infectiousness, pre-symptomatic
    rel_a: float                      # relative infectiousness, asymptomatic
    f_transmission: np.ndarray        # (2,) onward-transmission factor per stratum
    infectious_duration: np.ndarray   # (n_age,) infectiousness-weighted duration


def resolve_parameters(config: ScenarioConfig) -> ResolvedParams:
    """Resolve a scenario's disease block into arrays the engines consume."""
    d = config.disease
    n = config.population.n_age

    sigma = 1.0 / d.latent_period
    gamma_p = 1.0 / d.presymptomatic_period
    gamma_a = 1.0 / d.asymptomatic_infectious_period
    gamma_s = 1.0 / d.symptomatic_period
    gamma_h = 1.0 / d.hospital_stay
    gamma_c = 1.0 / d.icu_stay
    omega = 1.0 / d.waning_immunity_days if d.waning_immunity_days else 0.0

    p_asymp = d.asymptomatic_fraction_arr(n)
    hosp = d.hospitalization_rate_arr(n)
    icu = d.icu_rate_arr(n)
    death = d.death_rate_arr(n)

    ve_sev = config.vaccination.ve_severity
    # Severity by stratum: a single reduction applied to the probability of
    # progressing to hospitalisation (avoids triple-counting efficacy along the
    # cascade).
    hosp_rate = np.stack([hosp, hosp * (1.0 - ve_sev)])

    rel_p = d.rel_infectiousness_presymptomatic
    rel_a = d.rel_infectiousness_asymptomatic
    f_transmission = np.array([1.0, 1.0 - config.vaccination.ve_transmission])

    # Expected infectiousness-weighted duration of an infection started in each
    # age group (drives the next-generation matrix, hence R0 calibration).
    infectious_duration = (
        p_asymp * rel_a * d.asymptomatic_infectious_period
        + (1.0 - p_asymp) * (rel_p * d.presymptomatic_period + d.symptomatic_period)
    )

    return ResolvedParams(
        sigma=sigma,
        gamma_p=gamma_p,
        gamma_a=gamma_a,
        gamma_s=gamma_s,
        gamma_h=gamma_h,
        gamma_c=gamma_c,
        omega=omega,
        p_asymp=p_asymp,
        hosp_rate=hosp_rate,
        icu_rate=icu,
        death_rate=death,
        rel_p=rel_p,
        rel_a=rel_a,
        f_transmission=f_transmission,
        infectious_duration=infectious_duration,
    )


def transition_probability(rate, dt: float):
    """Convert a continuous per-day rate to a per-step transition probability."""
    return 1.0 - np.exp(-np.asarray(rate, dtype=float) * dt)


def ngm_unit(contact: np.ndarray, infectious_duration: np.ndarray) -> np.ndarray:
    """Next-generation matrix with ``beta = 1`` and full susceptibility.

    ``K0[i, j]`` is the number of secondary infections in group ``i`` produced by
    one infected individual in group ``j``: contacts a ``j``-individual has with
    ``i`` (``C[j, i]``) times ``j``'s expected infectious duration.
    """
    return contact.T * infectious_duration[None, :]


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue of a (small) square matrix."""
    if matrix.shape == (1, 1):
        return float(abs(matrix[0, 0]))
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


def calibrate_beta(contact: np.ndarray, infectious_duration: np.ndarray, r0: float) -> float:
    """Per-contact transmission rate giving the target ``r0``.

    Calibrated so the dominant eigenvalue of the next-generation matrix equals
    ``r0``.
    """
    rho = spectral_radius(ngm_unit(contact, infectious_duration))
    if rho <= 0:
        raise ValueError("degenerate contact structure: cannot calibrate beta")
    return r0 / rho


def icu_death_probability(icu_occupancy: float, death_rate: np.ndarray, healthcare):
    """Death probability for ICU leavers, raised when ICU is over capacity.

    ``icu_occupancy`` is the current number of ICU patients (at population
    scale). Returns ``(death_prob, overflow)`` where ``overflow`` is the demand
    above capacity (0 when within capacity).
    """
    cap = healthcare.icu_capacity
    death_prob = np.asarray(death_rate, dtype=float).copy()
    overflow = 0.0
    if cap is not None and icu_occupancy > cap:
        overflow = icu_occupancy - cap
        share_over = overflow / icu_occupancy
        mult = 1.0 + share_over * (healthcare.overflow_mortality_multiplier - 1.0)
        death_prob = np.minimum(1.0, death_prob * mult)
    return death_prob, overflow
