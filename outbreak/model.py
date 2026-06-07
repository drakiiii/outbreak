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
from .epidemiology import (
    calibrate_beta,
    icu_death_probability,
    ngm_unit,
    resolve_parameters,
    restore_array,
    spectral_radius,
    transition_probability,
)


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

    # Index into the leading axis of the (2, n_age) cascade arrays.
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
        # self.N is a 1-D array of shape (n_age,).
        self.N = config.population.population_by_age().astype(float)
        # Element-wise: keep N where positive, else substitute 1.0 so later
        # divisions by N never hit a zero denominator.
        self.N_safe = np.where(self.N > 0, self.N, 1.0)  # avoid divide-by-zero

        # Contact matrix, made reciprocal for this population.
        cm = (
            np.asarray(config.contact_matrix, dtype=float)
            if config.contact_matrix is not None
            else default_contact_matrix(self.n_age)
        )
        # symmetrize returns an (n_age, n_age) matrix.
        self.contact = symmetrize(cm, self.N)

        self._resolve_disease_parameters()
        self._calibrate_beta()

        self.t = 0
        self.cumulative_vaccinated = 0.0
        # Per-step imported infection hazard from other regions (set by a
        # metapopulation orchestrator; 0 for a standalone run).
        self.imported_force = 0.0
        self._init_state()

    # ------------------------------------------------------------------ setup
    def _resolve_disease_parameters(self) -> None:
        # Resolved once via the shared epidemiology layer so the compartmental
        # and agent-based engines describe an identical disease.
        p = resolve_parameters(self.config)
        self.sigma = p.sigma
        self.gamma_p = p.gamma_p
        self.gamma_a = p.gamma_a
        self.gamma_s = p.gamma_s
        self.gamma_h = p.gamma_h
        self.gamma_c = p.gamma_c
        self.omega = p.omega
        self.p_asymp = p.p_asymp
        self.hosp_rate = p.hosp_rate          # (2, n_age)
        self.icu_rate = p.icu_rate            # (n_age,)
        self.death_rate = p.death_rate        # (n_age,)
        self.rel_p = p.rel_p
        self.rel_a = p.rel_a
        self.f_transmission = p.f_transmission
        self.infectious_duration = p.infectious_duration
        self.susceptibility = p.susceptibility      # (n_age,) relative susceptibility

    def _ngm_unit(self) -> np.ndarray:
        """Next-generation matrix with beta=1 (age susceptibility folded in)."""
        return ngm_unit(self.contact, self.infectious_duration, self.susceptibility)

    def _calibrate_beta(self) -> None:
        self.beta = calibrate_beta(
            self.contact, self.infectious_duration, self.config.disease.r0,
            self.susceptibility,
        )
        self.r0_realized = self.config.disease.r0  # by construction

    def _init_state(self) -> None:
        n = self.n_age
        # Per-age pools are 1-D (n_age,); the infectious cascade carries a
        # leading stratum axis (2, n_age): row 0 = unvaccinated, row 1 = vaccinated.
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
            # Distribute seeds across ages in proportion to susceptibles.
            # weights is a per-age probability vector (sums to 1); fall back to
            # uniform if there are no susceptibles to weight by.
            weights = self.S / self.S.sum() if self.S.sum() > 0 else np.ones(n) / n
            seed_by_age = self._largest_remainder(seed_total, weights)
            # Split each age's seeds into asymptomatic vs symptomatic; clamp with
            # np.minimum at each step so no compartment exceeds what is available.
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
        return transition_probability(rate, dt)

    def _maybe_round(self, x: np.ndarray) -> np.ndarray:
        # np.rint rounds to nearest integer (banker's rounding) but keeps a float
        # dtype; deterministic mode leaves the fractional counts untouched.
        if self.stochastic:
            return np.rint(x)
        return x

    def _binom(self, n: np.ndarray, p) -> np.ndarray:
        """Draw transitions: binomial when stochastic, expectation otherwise."""
        p = np.clip(p, 0.0, 1.0)            # force probabilities into [0, 1]
        if not self.stochastic:
            return n * p                    # deterministic: return the expected count
        # Round counts to non-negative integers (binomial needs integer n).
        n_int = np.rint(np.maximum(n, 0.0)).astype(np.int64)
        # Element-wise binomial draw; broadcast p to n's shape so a scalar or a
        # lower-rank probability array fans out across every compartment cell.
        return self.rng.binomial(n_int, np.broadcast_to(p, n_int.shape)).astype(float)

    def _largest_remainder(self, total: int, weights: np.ndarray) -> np.ndarray:
        """Apportion an integer ``total`` across groups by ``weights``."""
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()       # normalise to a probability vector
        raw = weights * total                   # ideal (fractional) allocation
        base = np.floor(raw).astype(np.int64)   # integer floor of each share
        remainder = int(total - base.sum())     # leftover units after flooring
        if remainder > 0:
            # Hand the leftover units to the groups with the largest fractional
            # parts. argsort on the negated remainder gives descending order.
            order = np.argsort(-(raw - base))
            # Bump the top `remainder` groups by one unit each.
            base[order[:remainder]] += 1
        return base.astype(float)

    # ------------------------------------------------------------------- core
    def current_day(self) -> float:
        return self.t * self.dt

    def beta_effective(self, day: float) -> float:
        # Calibrated rate, scaled by active interventions and the seasonal cycle.
        return (
            self.beta
            * self.config.interventions.multiplier(day)
            * self.config.environment.seasonal_multiplier(day)
        )

    def susceptibility_by_age(self) -> np.ndarray:
        """Effective susceptible fraction per age (S plus leaky-protected V)."""
        ve = self.config.vaccination.ve_susceptibility
        # Vaccinated susceptibles count only partially (leaky protection);
        # element-wise division gives a per-age fraction in [0, 1].
        return (self.S + (1.0 - ve) * self.V) / self.N_safe

    def effective_rt(self, day: float) -> float:
        """Model-implied effective reproduction number at ``day``."""
        sus = self.susceptibility_by_age()
        # sus[:, None] reshapes (n_age,) -> (n_age, 1) so it broadcasts down the
        # rows of the unit NGM, scaling each row by its age's susceptibility
        # (equivalent to the left-multiply diag(sus) @ K0).
        k = (sus[:, None]) * self._ngm_unit()      # diag(sus) @ K0
        return self.beta_effective(day) * spectral_radius(k)

    def _vaccinate(self, day: float) -> None:
        vac = self.config.vaccination
        if not vac.enabled or day < vac.start_day:
            return
        total_pop = self.N.sum()
        cap_remaining = vac.coverage_cap * total_pop - self.cumulative_vaccinated
        if cap_remaining <= 0:
            return
        # Today's dose budget is the smallest of: the daily throughput, the
        # remaining coverage room, and the susceptibles actually available.
        doses = min(vac.daily_rate * total_pop, cap_remaining, float(self.S.sum()))
        if doses <= 0:
            return

        alloc = np.zeros(self.n_age)
        if vac.prioritize_elderly:
            # Greedily fill from the oldest age group down until doses run out.
            remaining = doses
            for i in range(self.n_age - 1, -1, -1):  # oldest first
                take = min(remaining, self.S[i])
                alloc[i] = take
                remaining -= take
                if remaining <= 0:
                    break
        else:
            if self.S.sum() > 0:
                # Spread doses across ages proportionally to susceptibles.
                alloc = doses * self.S / self.S.sum()
        # Never vaccinate more than the susceptibles present in each age group.
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
        # Weight each infectious compartment by relative infectiousness, scale
        # each stratum by f_transmission (shape (2,1) so it broadcasts over the
        # stratum axis of the (2, n_age) arrays), then sum over strata -> (n_age,).
        infectious_pressure = (
            self.f_transmission[:, None]
            * (self.rel_p * self.Ip + self.rel_a * self.Ia + self.Is)
        ).sum(axis=0)                                    # (n_age,)
        prevalence = infectious_pressure / self.N_safe
        # contact @ prevalence is a matrix-vector product mixing age groups via
        # the contact matrix, giving the per-age internal force of infection.
        internal = beta_eff * noise * (self.contact @ prevalence)   # (n_age,)
        # Add the external/spillover hazard (importations / reservoir) plus any
        # imported force from other regions, then scale the whole hazard by each
        # age's relative susceptibility.
        external = self.config.environment.external_force(day) + self.imported_force
        foi = self.susceptibility * (internal + external)           # (n_age,)

        ve_sus = self.config.vaccination.ve_susceptibility
        # Per-step infection probabilities (per-age, shape (n_age,)); vaccinated S
        # see a reduced FOI scaled by (1 - ve_sus).
        p_inf_S = self._prob(foi, self.dt)
        p_inf_V = self._prob(foi * (1.0 - ve_sus), self.dt)
        # Binomial draw of how many S and V get infected this step (per age).
        new_inf_S = self._binom(self.S, p_inf_S)
        new_inf_V = self._binom(self.V, p_inf_V)

        # 3. Progression transitions (drawn from start-of-step compartments).
        # Each rate is converted to a per-step probability, then a binomial draw
        # decides how many leave the compartment this step. All draws read the
        # compartments as they were at the start of the step (deltas applied later).
        p_E = self._prob(self.sigma, self.dt)
        p_Ip = self._prob(self.gamma_p, self.dt)
        p_Ia = self._prob(self.gamma_a, self.dt)
        p_Is = self._prob(self.gamma_s, self.dt)
        p_H = self._prob(self.gamma_h, self.dt)
        p_C = self._prob(self.gamma_c, self.dt)

        # Sub-binomial split of the E leavers: of those leaving E, a fraction
        # p_asymp go to Ia; the complement (leave_E - to_Ia) go to Ip.
        leave_E = self._binom(self.E, p_E)
        to_Ia = self._binom(leave_E, self.p_asymp)      # broadcast over strata
        to_Ip = leave_E - to_Ia

        leave_Ip = self._binom(self.Ip, p_Ip)           # all become symptomatic
        leave_Ia = self._binom(self.Ia, p_Ia)           # all recover

        # Symptomatic leavers split into hospitalised (hosp_rate) vs recovered.
        leave_Is = self._binom(self.Is, p_Is)
        to_H = self._binom(leave_Is, self.hosp_rate)    # stratum-specific
        Is_to_R = leave_Is - to_H

        # Hospital leavers split into ICU (icu_rate) vs recovered.
        leave_H = self._binom(self.H, p_H)
        to_C = self._binom(leave_H, self.icu_rate)
        H_to_R = leave_H - to_C

        # ICU leavers split into deaths vs recovered (death prob rises on overflow).
        leave_C = self._binom(self.C, p_C)
        death_prob, icu_overflow = self._icu_death_probability()
        to_D = self._binom(leave_C, death_prob)
        C_to_R = leave_C - to_D

        # 4. Apply all deltas atomically. Because every draw above used the
        # start-of-step values, the order of these in-place updates does not
        # matter: each compartment's net change is added in one shot.
        self.S -= new_inf_S
        self.V -= new_inf_V
        # New infections enter E in the matching stratum (unvaccinated/vaccinated).
        self.E[self.STRATUM_UNVAX] += new_inf_S
        self.E[self.STRATUM_VAX] += new_inf_V
        self.E -= leave_E
        self.Ip += to_Ip - leave_Ip
        self.Ia += to_Ia - leave_Ia
        self.Is += leave_Ip - leave_Is
        self.H += to_H - leave_H
        self.C += to_C - leave_C
        # R and D are per-age 1-D pools, so collapse the (2, n_age) inflows over
        # the stratum axis (sum(axis=0)) before adding.
        recovered = Is_to_R.sum(axis=0) + leave_Ia.sum(axis=0) + H_to_R.sum(axis=0) + C_to_R.sum(axis=0)
        self.R += recovered
        self.D += to_D.sum(axis=0)

        # 5. Waning immunity R -> S (binomial draw of recovered who lose immunity).
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
        return icu_death_probability(
            float(self.C.sum()), self.death_rate, self.config.healthcare
        )

    def _clip_negatives(self) -> None:
        # Clamp any tiny negative residue (from independent draws) to zero,
        # in place (out=arr) to avoid reallocating the arrays.
        for name in ("S", "V", "R", "D"):
            arr = getattr(self, name)
            np.clip(arr, 0.0, None, out=arr)
        for name in ("E", "Ip", "Ia", "Is", "H", "C"):
            arr = getattr(self, name)
            np.clip(arr, 0.0, None, out=arr)

    def _make_record(self, **kw) -> StepRecord:
        # Total infectious per age: add the three infectious compartments
        # (each (2, n_age)) and collapse the stratum axis -> (n_age,).
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
    def infectious_weighted_fraction(self) -> float:
        """Infectiousness-weighted infectious prevalence (for region coupling)."""
        weighted = (
            self.rel_p * self.Ip.sum() + self.rel_a * self.Ia.sum() + self.Is.sum()
        )
        total = self.N.sum()
        return float(weighted / total) if total > 0 else 0.0

    def infectious_count(self) -> float:
        """Number of currently infectious people (Ip + Ia + Is), population scale."""
        return float(self.Ip.sum() + self.Ia.sum() + self.Is.sum())

    def susceptible_fraction(self) -> float:
        """Fraction of the population currently susceptible (unvaccinated S)."""
        total = self.N.sum()
        return float(self.S.sum() / total) if total > 0 else 0.0

    def mean_infectious_duration(self) -> float:
        return float(self.infectious_duration.mean())

    def seed_exposed(self, amount: float) -> float:
        """Introduce ~``amount`` new infections (S -> E), e.g. imported by travel.

        Allocated across ages in proportion to susceptibles. Returns the number
        actually seeded (bounded by available susceptibles).
        """
        if amount <= 0:
            return 0.0
        s_total = self.S.sum()
        if s_total <= 0:
            return 0.0
        alloc = min(float(amount), float(s_total)) * self.S / s_total
        alloc = self._maybe_round(alloc)
        alloc = np.minimum(alloc, self.S)
        self.S -= alloc
        self.E[self.STRATUM_UNVAX] += alloc
        return float(alloc.sum())

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
        n = self.n_age
        self.t = max(0, int(state["t"]))
        self.cumulative_vaccinated = max(0.0, float(state["cumulative_vaccinated"]))
        # Each compartment has a fixed, configuration-determined shape; enforce
        # it so a malformed snapshot is rejected here rather than corrupting the
        # run (susceptible-style pools are per-age; the cascade is per-stratum).
        for name in ("S", "V", "R", "D"):
            setattr(self, name, restore_array(state, name, (n,)))
        for name in ("E", "Ip", "Ia", "Is", "H", "C"):
            setattr(self, name, restore_array(state, name, (2, n)))
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]
