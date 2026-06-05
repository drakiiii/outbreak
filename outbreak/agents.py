r"""The agent-based epidemic engine: an individual-level (microsimulation) model.

Where :mod:`outbreak.model` tracks *counts* in each compartment, this engine
tracks every individual as a row in a set of NumPy arrays and advances them one
at a time (vectorised, so it stays fast). It describes the same disease and is
calibrated to the same R0 via the shared :mod:`outbreak.epidemiology` layer, so
the two engines are directly comparable; in the mean-field limit (large
population, no extra heterogeneity) the agent model is a stochastic realisation
of the compartmental one.

What the individual representation buys us
------------------------------------------
* **Individual superspreading.** Each infected agent draws its own mean-one
  infectiousness multiplier, so a small fraction of people drive most onward
  transmission (realistic clustering) rather than a population-wide daily noise
  term. Because the multiplier has mean one, R0 is unchanged in expectation.
* **Demographic stochasticity.** Agents are genuine integers, so fade-out at
  small case numbers and the luck of early spread emerge naturally.
* **Per-individual attributes.** Age, vaccination status and clinical pathway
  live on the agent, ready to support richer structure (contact networks,
  households, individual histories) in future iterations.

Performance and scaling
-----------------------
Simulating hundreds of millions of agents is wasteful, so when the requested
``n_agents`` is smaller than ``total_population`` the engine simulates a
representative sample and multiplies its reported counts by
``scale = total_population / n_agents``. Absolute quantities supplied by the
user (ICU/hospital capacity) are compared against these population-scale counts,
so capacity effects behave as expected.

State machine (per agent)::

    SUS -> E -> Ip -> Is -> H -> C -> D
            \-> Ia -> R   Is/H/C -> R   R -> SUS (waning)

Vaccination is a per-agent flag rather than a separate state: a vaccinated
susceptible agent is reported as ``V`` and, if infected, carries reduced
severity and onward transmission (the "vaccinated" stratum).
"""

from __future__ import annotations

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
from .model import StepRecord

# Agent states (compact integer codes). Susceptibility and vaccination are
# separate axes: SUS + vaccinated flag == the compartmental "V" pool.
SUS, E, IP, IA, IS, H, C, R, D = range(9)
STATE_NAMES = ("SUS", "E", "Ip", "Ia", "Is", "H", "C", "R", "D")


class AgentModel:
    """A single stochastic, individual-based realisation of the epidemic.

    Exposes the same surface the :class:`~outbreak.simulation.Simulation`
    controller relies on (``t``, ``config``, :meth:`current_day`, :meth:`step`,
    :meth:`get_state`/:meth:`set_state`), so it is a drop-in alternative to
    :class:`~outbreak.model.EpidemicModel`.
    """

    STRATUM_UNVAX = 0
    STRATUM_VAX = 1

    def __init__(self, config: ScenarioConfig, rng: Optional[np.random.Generator] = None):
        self.config = config.validate()
        self.n_age = config.population.n_age
        self.dt = config.simulation.dt
        self.stochastic = config.simulation.stochastic
        self.rng = rng if rng is not None else np.random.default_rng(config.simulation.seed)

        # Population by age and the sampling scale (agents <-> people).
        pop_by_age = config.population.population_by_age().astype(float)
        total_pop = float(pop_by_age.sum())
        self.n_agents = int(min(config.simulation.n_agents, int(total_pop)))
        self.n_agents = max(self.n_agents, 1)
        self.scale = total_pop / self.n_agents

        # Agents per age group follow the population age distribution.
        self.agents_by_age = _largest_remainder(self.n_agents, pop_by_age / total_pop)
        self.N = self.agents_by_age.astype(float)          # agent-scale denominators
        self.N_safe = np.where(self.N > 0, self.N, 1.0)

        # Contact matrix (made reciprocal for the agent-scale age structure).
        cm = (
            np.asarray(config.contact_matrix, dtype=float)
            if config.contact_matrix is not None
            else default_contact_matrix(self.n_age)
        )
        self.contact = symmetrize(cm, self.N)

        params = resolve_parameters(config)
        self.p = params
        self.beta = calibrate_beta(self.contact, params.infectious_duration, config.disease.r0)
        self.r0_realized = config.disease.r0

        self.t = 0
        self.cumulative_vaccinated = 0.0
        self._init_agents()

    # ------------------------------------------------------------------ setup
    def _init_agents(self) -> None:
        n = self.n_agents
        # Per-agent age group, laid out contiguously by age for fast prioritised
        # vaccination (oldest groups occupy the highest indices).
        self.age = np.repeat(np.arange(self.n_age), self.agents_by_age).astype(np.int16)
        self.state = np.full(n, SUS, dtype=np.int8)
        self.vacc = np.zeros(n, dtype=bool)
        # Per-agent infectiousness multiplier (set when infected). Mean one.
        self.infectivity = np.ones(n, dtype=float)

        # Pre-existing immunity.
        imm = self.config.population.initial_immune_fraction
        if imm > 0:
            sus_idx = np.where(self.state == SUS)[0]
            k = int(round(imm * sus_idx.size))
            if k > 0:
                chosen = self.rng.choice(sus_idx, size=k, replace=False)
                self.state[chosen] = R

        # Seed infections (scaled to agent counts, at least one if any requested).
        seed_people = self.config.population.initial_infected
        if seed_people > 0:
            seed_agents = max(1, int(round(seed_people / self.scale)))
            sus_idx = np.where(self.state == SUS)[0]
            seed_agents = min(seed_agents, sus_idx.size)
            if seed_agents > 0:
                chosen = self.rng.choice(sus_idx, size=seed_agents, replace=False)
                # Split seeds into symptomatic / asymptomatic by age-specific rate.
                u = self.rng.random(chosen.size)
                asymp = u < self.p.p_asymp[self.age[chosen]]
                self.state[chosen[asymp]] = IA
                self.state[chosen[~asymp]] = IS
                self._assign_infectivity(chosen)

    def _assign_infectivity(self, idx: np.ndarray) -> None:
        """Draw per-agent onward infectiousness for newly infected agents."""
        od = self.config.simulation.overdispersion
        if od is None or not self.stochastic or idx.size == 0:
            self.infectivity[idx] = 1.0
        else:
            # Mean-one Gamma(shape=k, scale=1/k); small k => bursty superspreading.
            self.infectivity[idx] = self.rng.gamma(shape=od, scale=1.0 / od, size=idx.size)

    # --------------------------------------------------------------- helpers
    def current_day(self) -> float:
        return self.t * self.dt

    def beta_effective(self, day: float) -> float:
        return self.beta * self.config.interventions.multiplier(day)

    def _counts_by_age(self, mask: np.ndarray) -> np.ndarray:
        """Number of agents in each age group among those selected by ``mask``."""
        return np.bincount(self.age[mask], minlength=self.n_age).astype(float)

    def susceptibility_by_age(self) -> np.ndarray:
        """Effective susceptible fraction per age (S plus leaky-protected V)."""
        ve = self.config.vaccination.ve_susceptibility
        sus = self.state == SUS
        s_by_age = self._counts_by_age(sus & ~self.vacc)
        v_by_age = self._counts_by_age(sus & self.vacc)
        return (s_by_age + (1.0 - ve) * v_by_age) / self.N_safe

    def effective_rt(self, day: float) -> float:
        """Model-implied effective reproduction number at ``day``."""
        sus = self.susceptibility_by_age()
        k = sus[:, None] * ngm_unit(self.contact, self.p.infectious_duration)
        return self.beta_effective(day) * spectral_radius(k)

    # ----------------------------------------------------------- transitions
    def _bernoulli(self, mask: np.ndarray, prob) -> np.ndarray:
        """Indices of agents in ``mask`` that fire with per-agent ``prob``.

        Deterministic mode is undefined for individuals, so the agent engine is
        always stochastic; the per-agent draw is the model.
        """
        idx = np.where(mask)[0]
        if idx.size == 0:
            return idx
        p = np.asarray(prob, dtype=float)
        if p.ndim == 0:
            hit = self.rng.random(idx.size) < float(p)
        else:
            hit = self.rng.random(idx.size) < p[idx]
        return idx[hit]

    def _vaccinate(self, day: float) -> None:
        vac = self.config.vaccination
        if not vac.enabled or day < vac.start_day:
            return
        total_people = float(self.config.population.total_population)
        cap_remaining = vac.coverage_cap * total_people - self.cumulative_vaccinated
        if cap_remaining <= 0:
            return
        eligible = (self.state == SUS) & (~self.vacc)
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            return
        # Convert the people-scale dose budget into a number of agents.
        doses_people = min(vac.daily_rate * total_people, cap_remaining)
        n_doses = min(n_eligible, int(round(doses_people / self.scale)))
        if n_doses <= 0:
            return

        elig_idx = np.where(eligible)[0]
        if vac.prioritize_elderly:
            # Agents are laid out by age; highest indices are the oldest groups.
            order = elig_idx[np.argsort(-self.age[elig_idx], kind="stable")]
            chosen = order[:n_doses]
        else:
            chosen = self.rng.choice(elig_idx, size=n_doses, replace=False)
        self.vacc[chosen] = True
        self.cumulative_vaccinated += chosen.size * self.scale

    def step(self) -> StepRecord:
        """Advance the epidemic by one time step and return a summary record."""
        day = self.current_day()

        # 1. Vaccination (administrative S -> V flag) at the start of the day.
        self._vaccinate(day)

        # 2. Force of infection by age from the current infectious agents.
        beta_eff = self.beta_effective(day)
        weight = np.zeros(self.n_agents)
        is_ip, is_ia, is_is = self.state == IP, self.state == IA, self.state == IS
        weight[is_ip] = self.p.rel_p
        weight[is_ia] = self.p.rel_a
        weight[is_is] = 1.0
        infectious = is_ip | is_ia | is_is
        # Per-agent contribution = stratum factor * phase weight * individual load.
        f_strat = np.where(self.vacc, self.p.f_transmission[1], self.p.f_transmission[0])
        contrib = (weight * f_strat * self.infectivity)[infectious]
        pressure = np.bincount(
            self.age[infectious], weights=contrib, minlength=self.n_age
        )
        prevalence = pressure / self.N_safe
        foi = beta_eff * (self.contact @ prevalence)          # (n_age,)

        ve_sus = self.config.vaccination.ve_susceptibility
        sus = self.state == SUS
        foi_age = foi[self.age]
        foi_age = np.where(self.vacc, foi_age * (1.0 - ve_sus), foi_age)
        p_inf = transition_probability(foi_age, self.dt)
        newly = self._bernoulli(sus, p_inf)

        # 3. Progression transitions, drawn from start-of-step states.
        leave_E = self._bernoulli(self.state == E, transition_probability(self.p.sigma, self.dt))
        leave_Ip = self._bernoulli(self.state == IP, transition_probability(self.p.gamma_p, self.dt))
        leave_Ia = self._bernoulli(self.state == IA, transition_probability(self.p.gamma_a, self.dt))
        leave_Is = self._bernoulli(self.state == IS, transition_probability(self.p.gamma_s, self.dt))
        leave_H = self._bernoulli(self.state == H, transition_probability(self.p.gamma_h, self.dt))
        leave_C = self._bernoulli(self.state == C, transition_probability(self.p.gamma_c, self.dt))

        # Branch the leavers via complementary boolean masks.
        asymp = self.rng.random(leave_E.size) < self.p.p_asymp[self.age[leave_E]]
        to_Ia, to_Ip = leave_E[asymp], leave_E[~asymp]

        hosp_p = self.p.hosp_rate[self.vacc[leave_Is].astype(int), self.age[leave_Is]]
        hospitalised = self.rng.random(leave_Is.size) < hosp_p
        to_H, Is_to_R = leave_Is[hospitalised], leave_Is[~hospitalised]

        critical = self.rng.random(leave_H.size) < self.p.icu_rate[self.age[leave_H]]
        to_C, H_to_R = leave_H[critical], leave_H[~critical]

        death_prob, icu_overflow = icu_death_probability(
            float((self.state == C).sum()) * self.scale, self.p.death_rate, self.config.healthcare
        )
        died = self.rng.random(leave_C.size) < death_prob[self.age[leave_C]]
        to_D, C_to_R = leave_C[died], leave_C[~died]

        # 4. Apply all moves (start-of-step masks make ordering irrelevant).
        self.state[newly] = E
        self._assign_infectivity(newly)
        self.state[to_Ip] = IP
        self.state[to_Ia] = IA
        self.state[leave_Ip] = IS           # all pre-symptomatic become symptomatic
        self.state[leave_Ia] = R            # all asymptomatic recover
        self.state[to_H] = H
        self.state[Is_to_R] = R
        self.state[to_C] = C
        self.state[H_to_R] = R
        self.state[to_D] = D
        self.state[C_to_R] = R

        # 5. Waning immunity R -> SUS (vaccine-derived protection is not retained).
        if self.p.omega > 0:
            waned = self._bernoulli(self.state == R, transition_probability(self.p.omega, self.dt))
            self.state[waned] = SUS
            self.vacc[waned] = False

        self.t += 1
        rt = self.effective_rt(day)
        return self._make_record(
            day=day,
            new_infections=newly.size * self.scale,
            new_symptomatic=leave_Ip.size * self.scale,
            new_hospitalizations=to_H.size * self.scale,
            new_icu=to_C.size * self.scale,
            new_deaths=to_D.size * self.scale,
            rt=rt,
            beta_effective=beta_eff,
            icu_overflow=icu_overflow,
        )

    # ----------------------------------------------------------- diagnostics
    def _make_record(self, **kw) -> StepRecord:
        counts = np.bincount(self.state, minlength=9).astype(float) * self.scale
        sus = self.state == SUS
        s_count = float((sus & ~self.vacc).sum()) * self.scale
        v_count = float((sus & self.vacc).sum()) * self.scale

        infectious_mask = np.isin(self.state, (IP, IA, IS))
        infectious_by_age = self._counts_by_age(infectious_mask) * self.scale
        deaths_by_age = self._counts_by_age(self.state == D) * self.scale

        return StepRecord(
            S=s_count,
            V=v_count,
            E=float(counts[E]),
            Ip=float(counts[IP]),
            Ia=float(counts[IA]),
            Is=float(counts[IS]),
            H=float(counts[H]),
            C=float(counts[C]),
            R=float(counts[R]),
            D=float(counts[D]),
            infectious_by_age=infectious_by_age,
            deaths_by_age=deaths_by_age,
            **kw,
        )

    def total_living(self) -> float:
        return float((self.state != D).sum()) * self.scale

    def total_population(self) -> float:
        return float(self.n_agents) * self.scale

    # ----------------------------------------------------------- (de)serialise
    def get_state(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "cumulative_vaccinated": self.cumulative_vaccinated,
            "n_agents": self.n_agents,
            "scale": self.scale,
            "age": self.age.tolist(),
            "state": self.state.tolist(),
            "vacc": self.vacc.tolist(),
            "infectivity": self.infectivity.tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, object]) -> None:
        # The population/agent count is fixed by the (already validated) config
        # this engine was built from, so the snapshot must match it. Pinning the
        # length here bounds every restored array to a configuration-sanctioned
        # size and stops a tampered snapshot from forcing a huge allocation.
        n = int(state["n_agents"])
        if n != self.n_agents:
            raise ValueError(
                f"snapshot n_agents ({n}) does not match this scenario ({self.n_agents})"
            )
        self.t = max(0, int(state["t"]))
        self.cumulative_vaccinated = max(0.0, float(state["cumulative_vaccinated"]))
        scale = float(state["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("snapshot scale must be a positive, finite number")
        self.scale = scale

        age = restore_array(state, "age", (n,), dtype=np.int16)
        st = restore_array(state, "state", (n,), dtype=np.int8)
        if age.size and (age.min() < 0 or age.max() >= self.n_age):
            raise ValueError("snapshot contains out-of-range age groups")
        if st.size and (st.min() < 0 or st.max() > D):
            raise ValueError("snapshot contains out-of-range agent states")
        self.age = age
        self.state = st
        self.vacc = restore_array(state, "vacc", (n,), dtype=bool)
        self.infectivity = restore_array(state, "infectivity", (n,), dtype=float)
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """Apportion an integer ``total`` across groups by ``weights``."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    raw = weights * total
    base = np.floor(raw).astype(np.int64)
    remainder = int(total - base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base
