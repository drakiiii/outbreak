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

    # Shape of the per-stage sojourn-time distribution (gamma "shape" k). Used by
    # the agent engine to sample explicit, realistically-peaked stage durations;
    # the compartmental engine uses exponential sojourns and ignores it.
    duration_dispersion: float = 1.0

    # Relative susceptibility to infection by age (1.0 = baseline). Scales each
    # susceptible's force of infection and is folded into the R0 calibration.
    susceptibility: np.ndarray = None      # (n_age,)


def resolve_parameters(config: ScenarioConfig) -> ResolvedParams:
    """Resolve a scenario's disease block into arrays the engines consume."""
    d = config.disease
    n = config.population.n_age

    # A mean duration of D days corresponds to a constant transition rate of 1/D
    # per day (exponential dwell-time assumption).
    sigma = 1.0 / d.latent_period
    gamma_p = 1.0 / d.presymptomatic_period
    gamma_a = 1.0 / d.asymptomatic_infectious_period
    gamma_s = 1.0 / d.symptomatic_period
    gamma_h = 1.0 / d.hospital_stay
    gamma_c = 1.0 / d.icu_stay
    # `if d.waning_immunity_days` is falsy for both None and 0.0, so either
    # disables waning by setting the rate to 0.
    omega = 1.0 / d.waning_immunity_days if d.waning_immunity_days else 0.0

    p_asymp = d.asymptomatic_fraction_arr(n)
    hosp = d.hospitalization_rate_arr(n)
    icu = d.icu_rate_arr(n)
    death = d.death_rate_arr(n)

    ve_sev = config.vaccination.ve_severity
    # Severity by stratum: a single reduction applied to the probability of
    # progressing to hospitalisation (avoids triple-counting efficacy along the
    # cascade).
    # Stack two rows -> shape (2, n_age): row 0 = unvaccinated, row 1 = vaccinated
    # (severity scaled down by the vaccine efficacy).
    hosp_rate = np.stack([hosp, hosp * (1.0 - ve_sev)])

    rel_p = d.rel_infectiousness_presymptomatic
    rel_a = d.rel_infectiousness_asymptomatic
    # Length-2 [unvaccinated, vaccinated] onward-transmission factors. Same
    # two-row stratum convention as hosp_rate above (index 0/1 = un/vaccinated).
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
        duration_dispersion=d.duration_dispersion,
        susceptibility=d.susceptibility_arr(n),
    )


def transition_probability(rate, dt: float):
    """Convert a continuous per-day rate to a per-step transition probability."""
    # Probability that an exponential(rate) event fires within a window of dt
    # days: P = 1 - exp(-rate*dt). np.asarray means rate may be a scalar or an
    # array (the operation broadcasts either way).
    return 1.0 - np.exp(-np.asarray(rate, dtype=float) * dt)


def ngm_unit(contact: np.ndarray, infectious_duration: np.ndarray,
             susceptibility: np.ndarray = None) -> np.ndarray:
    """Next-generation matrix with ``beta = 1`` and full susceptibility.

    ``K0[i, j]`` is the number of secondary infections in group ``i`` produced by
    one infected individual in group ``j``: contacts a ``j``-individual has with
    ``i`` (``C[j, i]``) times ``j``'s expected infectious duration. If a per-age
    ``susceptibility`` is given, each row ``i`` is scaled by it (a less
    susceptible age acquires proportionally fewer infections).
    """
    # contact.T puts C[j, i] at position [i, j]. infectious_duration[None, :] is
    # a (1, n_age) row vector that broadcasts across rows, scaling each *column* j
    # by group j's infectious duration -> K0[i, j] = C[j, i] * duration[j].
    k0 = contact.T * infectious_duration[None, :]
    if susceptibility is not None:
        # Scale each row i (the acquiring age group) by its susceptibility.
        k0 = susceptibility[:, None] * k0
    return k0


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue of a (small) square matrix."""
    if matrix.shape == (1, 1):
        return float(abs(matrix[0, 0]))  # 1x1 case: the single entry is the eigenvalue
    # eigvals returns all (possibly complex) eigenvalues; the spectral radius is
    # the largest magnitude among them.
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


def calibrate_beta(contact: np.ndarray, infectious_duration: np.ndarray, r0: float,
                   susceptibility: np.ndarray = None) -> float:
    """Per-contact transmission rate giving the target ``r0``.

    Calibrated so the dominant eigenvalue of the next-generation matrix (with any
    age-specific ``susceptibility`` folded in) equals ``r0``.
    """
    rho = spectral_radius(ngm_unit(contact, infectious_duration, susceptibility))
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
    death_prob = np.asarray(death_rate, dtype=float).copy()  # copy so we never mutate the caller's array
    overflow = 0.0
    if cap is not None and icu_occupancy > cap:
        overflow = icu_occupancy - cap
        share_over = overflow / icu_occupancy  # fraction of current ICU patients that are over capacity
        # Blend baseline and elevated mortality by the over-capacity share: at
        # share_over=0 mult=1 (no change), at share_over=1 mult=the full multiplier.
        mult = 1.0 + share_over * (healthcare.overflow_mortality_multiplier - 1.0)
        death_prob = np.minimum(1.0, death_prob * mult)  # clamp so probabilities never exceed 1
    return death_prob, overflow


def restore_array(state: dict, name: str, expected_shape, dtype=float) -> np.ndarray:
    """Coerce a snapshot field into an array of an exact, expected shape.

    Used by the engines' ``set_state`` to safely rehydrate persisted state.
    Restoring is the one place a model ingests externally-supplied data (a saved
    or uploaded snapshot), so every field is shape-checked: this rejects
    malformed or hostile snapshots up front rather than letting a wrong-sized
    array cause confusing failures (or unbounded work) deep in a later step.
    """
    if name not in state:
        raise ValueError(f"snapshot is missing required field {name!r}")
    # Normalise the expected shape to a tuple of plain ints so the == comparison
    # below is reliable (accepts e.g. a list or numpy shape).
    expected_shape = tuple(int(d) for d in expected_shape)
    try:
        arr = np.asarray(state[name], dtype=dtype)
    except (ValueError, TypeError) as exc:  # ragged / non-numeric input can't be coerced
        raise ValueError(f"snapshot field {name!r} is not a valid array") from exc
    if arr.shape != expected_shape:  # reject wrong-sized data before it reaches the engine
        raise ValueError(
            f"snapshot field {name!r} has shape {arr.shape}, expected {expected_shape}"
        )
    return arr
