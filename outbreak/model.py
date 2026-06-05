"""The epidemic engine: a stochastic, age-structured SEIR-type compartmental model.

Compartments (each resolved per age group; the infectious cascade is also split
into two *strata*, unvaccinated and vaccinated, so vaccination can reduce
severity and onward transmission):

    S   susceptible (unvaccinated)
    V   vaccinated and susceptible (reduced susceptibility)
    E   exposed / latent (infected, not yet infectious)
    Ip  pre-symptomatic infectious (will develop symptoms)
    Ia  asymptomatic infectious (never develops symptoms)
    Is  symptomatic infectious
    H   hospitalised (non-ICU)
    C   critical / ICU
    R   recovered (immune; may wane back to S)
    D   dead

Transmission is frequency-dependent and driven by an age contact matrix. The
per-contact transmission rate ``beta`` is **not** specified directly: it is
calibrated so the dominant eigenvalue of the next-generation matrix equals the
target basic reproduction number ``r0``. The model also reports the
model-implied effective reproduction number ``Rt`` at every step.

Stochasticity: every transition is a binomial draw (chain-binomial model).
Superspreading/overdispersion is captured by an optional mean-one Gamma
multiplier on the daily force of infection. A deterministic mode (expected
values) is provided for validation against ODE behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import ScenarioConfig
from .contacts import default_contact_matrix, symmetrize


@dataclass
class StepRecord:
    """Per-step summary of state and flows, consumed by the metrics layer."""

    day: float
    # State totals (summed over age and strata)
    S: float
    V: float
    E: float
    Ip: float
    Ia: float
    Is: float
    H: float
    C: float
    R: float
    D: float
    # Daily incidence flows
    new_infections: float
    new_symptomatic: float
    new_hospitalizations: float
    new_icu: float
    new_deaths: float
    # Diagnostics
    rt: float
    beta_effective: float
    icu_overflow: float            # ICU demand above capacity (0 if within)
    # Per-age state snapshots (living population by compartment)
    infectious_by_age: np.ndarray = field(default=None, repr=False)
    deaths_by_age: np.ndarray = field(default=None, repr=False)


class EpidemicModel:
    """A single stochastic realisation of the epidemic.

    The model owns its state and a NumPy random generator. Call :meth:`step`
    repeatedly to advance the epidemic one time step at a time; this granularity
    is what makes pausing and inspection trivial at the controller layer.
    """

    STRATUM_UNVAX = 0
    STRATUM_VAX = 1

    def __init__(self, config: ScenarioConfig, rng: Optional[np.random.Generator] = None):
        self.config = config.validate()
        self.n_age = config.population.n_age
        self.dt = config.simulation.dt
        self.stochastic = config.simulation.stochastic

        seed = config.simulation.seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Population by age (constant denominator for frequency dependence).
        self.N = config.population.population_by_age().astype(float)
        self.N_safe = np.where(self.N > 0, self.N, 1.0)  # avoid divide-by-zero

        # Contact matrix, made reciprocal for this population.
        cm = (
            np.asarray(config.contact_matrix, dtype=float)
            if config.contact_matrix is not None
            else default_contact_matrix(self.n_age)
        )
        self.contact = symmetrize(cm, self.N)

        self._resolve_disease_parameters()
        self._calibrate_beta()

        self.t = 0
        self.cumulative_vaccinated = 0.0
        self._init_state()

    # ------------------------------------------------------------------ setup
    def _resolve_disease_parameters(self) -> None:
        d = self.config.disease
        n = self.n_age

        # Transition rates (per day) -> per-step probabilities below.
        self.sigma = 1.0 / d.latent_period                      # E -> infectious
        self.gamma_p = 1.0 / d.presymptomatic_period            # Ip -> Is
        self.gamma_a = 1.0 / d.asymptomatic_infectious_period   # Ia -> R
        self.gamma_s = 1.0 / d.symptomatic_period               # Is -> H/R
        self.gamma_h = 1.0 / d.hospital_stay                    # H -> C/R
        self.gamma_c = 1.0 / d.icu_stay                         # C -> D/R
        self.omega = (
            1.0 / d.waning_immunity_days if d.waning_immunity_days else 0.0
        )

        # Branching probabilities (per age).
        self.p_asymp = d.asymptomatic_fraction_arr(n)
        hosp = d.hospitalization_rate_arr(n)
        icu = d.icu_rate_arr(n)
        death = d.death_rate_arr(n)

        ve_sev = self.config.vaccination.ve_severity
        # Severity by stratum: vaccinated have a single reduction applied to the
        # probability of progressing to hospitalisation (avoids triple-counting
        # the efficacy across the cascade).
        self.hosp_rate = np.stack([hosp, hosp * (1.0 - ve_sev)])   # (2, n_age)
        self.icu_rate = icu                                        # (n_age,)
        self.death_rate = death                                    # (n_age,)

        # Relative infectiousness weights.
        self.rel_p = d.rel_infectiousness_presymptomatic
        self.rel_a = d.rel_infectiousness_asymptomatic
        self.f_transmission = np.array(
            [1.0, 1.0 - self.config.vaccination.ve_transmission]
        )  # per stratum infectiousness multiplier

        # Expected infectiousness-weighted duration of an infection started in
        # each age group (used by the next-generation matrix).
        self.infectious_duration = (
            self.p_asymp * self.rel_a * d.asymptomatic_infectious_period
            + (1.0 - self.p_asymp)
            * (self.rel_p * d.presymptomatic_period + d.symptomatic_period)
        )

    def _ngm_unit(self) -> np.ndarray:
        """Next-generation matrix with beta=1 and full susceptibility.

        ``K0[i, j]`` = secondary infections in group i produced by one infected
        in group j. Using ``C[j, i]`` (contacts a j-individual has with i) and
        the expected infectious duration ``T_j``.
        """
        return self.contact.T * self.infectious_duration[None, :]

    def _calibrate_beta(self) -> None:
        k0 = self._ngm_unit()
        rho = _spectral_radius(k0)
        if rho <= 0:
            raise ValueError("degenerate contact structure: cannot calibrate beta")
        self.beta = self.config.disease.r0 / rho
        self.r0_realized = self.beta * rho  # == r0 by construction; kept for clarity

    def _init_state(self) -> None:
        n = self.n_age
        self.S = self.N.copy()
        self.V = np.zeros(n)
        self.E = np.zeros((2, n))
        self.Ip = np.zeros((2, n))
        self.Ia = np.zeros((2, n))
        self.Is = np.zeros((2, n))
        self.H = np.zeros((2, n))
        self.C = np.zeros((2, n))
        self.R = np.zeros(n)
        self.D = np.zeros(n)

        # Pre-existing immunity.
        imm = self.config.population.initial_immune_fraction
        if imm > 0:
            immune = self._maybe_round(self.S * imm)
            self.S -= immune
            self.R += immune

        # Seed infections proportional to (remaining susceptible) population,
        # split into symptomatic and asymptomatic by the age-specific fraction.
        seed_total = self.config.population.initial_infected
        if seed_total > 0:
            weights = self.S / self.S.sum() if self.S.sum() > 0 else np.ones(n) / n
            seed_by_age = self._largest_remainder(seed_total, weights)
            asymp = self._maybe_round(seed_by_age * self.p_asymp)
            asymp = np.minimum(asymp, seed_by_age)
            symp = seed_by_age - asymp
            symp = np.minimum(symp, self.S)
            self.Is[self.STRATUM_UNVAX] += symp
            self.S -= symp
            asymp = np.minimum(asymp, self.S)
            self.Ia[self.STRATUM_UNVAX] += asymp
            self.S -= asymp

    # --------------------------------------------------------------- numerics
    @staticmethod
    def _prob(rate: float, dt: float) -> float:
        """Convert a continuous rate to a per-step transition probability."""
        return 1.0 - np.exp(-rate * dt)

    def _maybe_round(self, x: np.ndarray) -> np.ndarray:
        if self.stochastic:
            return np.rint(x)
        return x

    def _binom(self, n: np.ndarray, p) -> np.ndarray:
        """Draw transitions: binomial when stochastic, expectation otherwise."""
        p = np.clip(p, 0.0, 1.0)
        if not self.stochastic:
            return n * p
        n_int = np.rint(np.maximum(n, 0.0)).astype(np.int64)
        return self.rng.binomial(n_int, np.broadcast_to(p, n_int.shape)).astype(float)

    def _largest_remainder(self, total: int, weights: np.ndarray) -> np.ndarray:
        """Apportion an integer ``total`` across groups by ``weights``."""
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        raw = weights * total
        base = np.floor(raw).astype(np.int64)
        remainder = int(total - base.sum())
        if remainder > 0:
            order = np.argsort(-(raw - base))
            base[order[:remainder]] += 1
        return base.astype(float)

    # ------------------------------------------------------------------- core
    def current_day(self) -> float:
        return self.t * self.dt

    def beta_effective(self, day: float) -> float:
        return self.beta * self.config.interventions.multiplier(day)

    def susceptibility_by_age(self) -> np.ndarray:
        """Effective susceptible fraction per age (S plus leaky-protected V)."""
        ve = self.config.vaccination.ve_susceptibility
        return (self.S + (1.0 - ve) * self.V) / self.N_safe

    def effective_rt(self, day: float) -> float:
        """Model-implied effective reproduction number at ``day``."""
        sus = self.susceptibility_by_age()
        k = (sus[:, None]) * self._ngm_unit()      # diag(sus) @ K0
        return self.beta_effective(day) * _spectral_radius(k)

    def _vaccinate(self, day: float) -> None:
        vac = self.config.vaccination
        if not vac.enabled or day < vac.start_day:
            return
        total_pop = self.N.sum()
        cap_remaining = vac.coverage_cap * total_pop - self.cumulative_vaccinated
        if cap_remaining <= 0:
            return
        doses = min(vac.daily_rate * total_pop, cap_remaining, float(self.S.sum()))
        if doses <= 0:
            return

        alloc = np.zeros(self.n_age)
        if vac.prioritize_elderly:
            remaining = doses
            for i in range(self.n_age - 1, -1, -1):  # oldest first
                take = min(remaining, self.S[i])
                alloc[i] = take
                remaining -= take
                if remaining <= 0:
                    break
        else:
            if self.S.sum() > 0:
                alloc = doses * self.S / self.S.sum()
        alloc = self._maybe_round(np.minimum(alloc, self.S))
        self.S -= alloc
        self.V += alloc
        self.cumulative_vaccinated += float(alloc.sum())

    def step(self) -> StepRecord:
        """Advance the epidemic by one time step and return a summary record."""
        day = self.current_day()

        # 1. Vaccination (administrative move S -> V) at the start of the day.
        self._vaccinate(day)

        # 2. Force of infection from the current infectious population.
        beta_eff = self.beta_effective(day)
        noise = self._overdispersion_noise()
        infectious_pressure = (
            self.f_transmission[:, None]
            * (self.rel_p * self.Ip + self.rel_a * self.Ia + self.Is)
        ).sum(axis=0)                                    # (n_age,)
        prevalence = infectious_pressure / self.N_safe
        foi = beta_eff * noise * (self.contact @ prevalence)   # (n_age,)

        ve_sus = self.config.vaccination.ve_susceptibility
        p_inf_S = self._prob(foi, self.dt)
        p_inf_V = self._prob(foi * (1.0 - ve_sus), self.dt)
        new_inf_S = self._binom(self.S, p_inf_S)
        new_inf_V = self._binom(self.V, p_inf_V)

        # 3. Progression transitions (drawn from start-of-step compartments).
        p_E = self._prob(self.sigma, self.dt)
        p_Ip = self._prob(self.gamma_p, self.dt)
        p_Ia = self._prob(self.gamma_a, self.dt)
        p_Is = self._prob(self.gamma_s, self.dt)
        p_H = self._prob(self.gamma_h, self.dt)
        p_C = self._prob(self.gamma_c, self.dt)

        leave_E = self._binom(self.E, p_E)
        to_Ia = self._binom(leave_E, self.p_asymp)      # broadcast over strata
        to_Ip = leave_E - to_Ia

        leave_Ip = self._binom(self.Ip, p_Ip)           # all become symptomatic
        leave_Ia = self._binom(self.Ia, p_Ia)           # all recover

        leave_Is = self._binom(self.Is, p_Is)
        to_H = self._binom(leave_Is, self.hosp_rate)    # stratum-specific
        Is_to_R = leave_Is - to_H

        leave_H = self._binom(self.H, p_H)
        to_C = self._binom(leave_H, self.icu_rate)
        H_to_R = leave_H - to_C

        leave_C = self._binom(self.C, p_C)
        death_prob, icu_overflow = self._icu_death_probability()
        to_D = self._binom(leave_C, death_prob)
        C_to_R = leave_C - to_D

        # 4. Apply all deltas atomically.
        self.S -= new_inf_S
        self.V -= new_inf_V
        self.E[self.STRATUM_UNVAX] += new_inf_S
        self.E[self.STRATUM_VAX] += new_inf_V
        self.E -= leave_E
        self.Ip += to_Ip - leave_Ip
        self.Ia += to_Ia - leave_Ia
        self.Is += leave_Ip - leave_Is
        self.H += to_H - leave_H
        self.C += to_C - leave_C
        recovered = Is_to_R.sum(axis=0) + leave_Ia.sum(axis=0) + H_to_R.sum(axis=0) + C_to_R.sum(axis=0)
        self.R += recovered
        self.D += to_D.sum(axis=0)

        # 5. Waning immunity R -> S.
        if self.omega > 0:
            waned = self._binom(self.R, self._prob(self.omega, self.dt))
            self.R -= waned
            self.S += waned

        self._clip_negatives()
        self.t += 1

        rt = self.effective_rt(day)
        return self._make_record(
            day=day,
            new_infections=float(new_inf_S.sum() + new_inf_V.sum()),
            new_symptomatic=float(leave_Ip.sum()),
            new_hospitalizations=float(to_H.sum()),
            new_icu=float(to_C.sum()),
            new_deaths=float(to_D.sum()),
            rt=rt,
            beta_effective=beta_eff,
            icu_overflow=icu_overflow,
        )

    # ------------------------------------------------------------- mechanisms
    def _overdispersion_noise(self) -> float:
        od = self.config.simulation.overdispersion
        if od is None or not self.stochastic:
            return 1.0
        # Mean-one Gamma(shape=k, scale=1/k); small k => heavy-tailed (bursty).
        return float(self.rng.gamma(shape=od, scale=1.0 / od))

    def _icu_death_probability(self):
        """Death probability for ICU leavers, raised when ICU is over capacity."""
        cap = self.config.healthcare.icu_capacity
        death_prob = self.death_rate.copy()
        overflow = 0.0
        if cap is not None:
            occupancy = float(self.C.sum())
            if occupancy > cap:
                overflow = occupancy - cap
                share_over = overflow / occupancy
                mult = 1.0 + share_over * (
                    self.config.healthcare.overflow_mortality_multiplier - 1.0
                )
                death_prob = np.minimum(1.0, death_prob * mult)
        return death_prob, overflow

    def _clip_negatives(self) -> None:
        for name in ("S", "V", "R", "D"):
            arr = getattr(self, name)
            np.clip(arr, 0.0, None, out=arr)
        for name in ("E", "Ip", "Ia", "Is", "H", "C"):
            arr = getattr(self, name)
            np.clip(arr, 0.0, None, out=arr)

    def _make_record(self, **kw) -> StepRecord:
        infectious_by_age = (self.Ip + self.Ia + self.Is).sum(axis=0)
        return StepRecord(
            S=float(self.S.sum()),
            V=float(self.V.sum()),
            E=float(self.E.sum()),
            Ip=float(self.Ip.sum()),
            Ia=float(self.Ia.sum()),
            Is=float(self.Is.sum()),
            H=float(self.H.sum()),
            C=float(self.C.sum()),
            R=float(self.R.sum()),
            D=float(self.D.sum()),
            infectious_by_age=infectious_by_age.copy(),
            deaths_by_age=self.D.copy(),
            **kw,
        )

    # ------------------------------------------------------------ diagnostics
    def total_living(self) -> float:
        return float(
            self.S.sum() + self.V.sum() + self.E.sum() + self.Ip.sum()
            + self.Ia.sum() + self.Is.sum() + self.H.sum() + self.C.sum()
            + self.R.sum()
        )

    def total_population(self) -> float:
        return self.total_living() + float(self.D.sum())

    # ----------------------------------------------------------- (de)serialise
    def get_state(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "cumulative_vaccinated": self.cumulative_vaccinated,
            "S": self.S.tolist(),
            "V": self.V.tolist(),
            "E": self.E.tolist(),
            "Ip": self.Ip.tolist(),
            "Ia": self.Ia.tolist(),
            "Is": self.Is.tolist(),
            "H": self.H.tolist(),
            "C": self.C.tolist(),
            "R": self.R.tolist(),
            "D": self.D.tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, object]) -> None:
        self.t = int(state["t"])
        self.cumulative_vaccinated = float(state["cumulative_vaccinated"])
        for name in ("S", "V", "E", "Ip", "Ia", "Is", "H", "C", "R", "D"):
            setattr(self, name, np.asarray(state[name], dtype=float))
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]


def _spectral_radius(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue of a (small) square matrix."""
    if matrix.shape == (1, 1):
        return float(abs(matrix[0, 0]))
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))
