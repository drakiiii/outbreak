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
from .network import ContactLayer, build_layers, layer_mixing_matrix

# Agent states (compact integer codes). Susceptibility and vaccination are
# separate axes: SUS + vaccinated flag == the compartmental "V" pool.
# Tuple unpacking of range(9) assigns SUS=0, E=1, ... D=8.
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

        # Agents per age group follow the population age distribution. Result is
        # an integer (n_age,) array that sums to exactly n_agents.
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
        # Shape of the per-stage sojourn-time distribution (k=1 => exponential).
        self.duration_shape = float(params.duration_dispersion)
        self.susceptibility = params.susceptibility    # (n_age,) relative susceptibility
        self.r0_realized = config.disease.r0

        self.t = 0
        self.cumulative_vaccinated = 0.0
        # Per-step imported infection hazard from other regions (set by a
        # metapopulation orchestrator; 0 for a standalone run).
        self.imported_force = 0.0
        self._init_agents()
        # Build contact layers (if enabled) and calibrate beta on the resulting
        # effective contact structure. _build_network sets self.contact-derived
        # self.c_eff, self.layers, self.layer_scaling and self.beta. It is called
        # after agent/seed initialisation because constructing the network draws
        # from rng, and we want a stable, well-defined draw order.
        self._build_network()

    # ------------------------------------------------------------------ setup
    def _init_agents(self) -> None:
        n = self.n_agents
        # Per-agent age group, laid out contiguously by age for fast prioritised
        # vaccination (oldest groups occupy the highest indices).
        # np.repeat expands [0,1,2,...] by per-age counts, e.g. counts [2,3] ->
        # [0,0,1,1,1], so self.age is a length-n array of age-group codes.
        self.age = np.repeat(np.arange(self.n_age), self.agents_by_age).astype(np.int16)
        # Parallel per-agent arrays, all indexed by the same agent id (row).
        self.state = np.full(n, SUS, dtype=np.int8)
        self.vacc = np.zeros(n, dtype=bool)
        # Per-agent infectiousness multiplier (set when infected). Mean one.
        self.infectivity = np.ones(n, dtype=float)
        # Per-agent countdown: days left in the current timed disease stage. Only
        # meaningful while an agent is in E/Ip/Ia/Is/H/C; ignored otherwise. A
        # fresh duration is sampled each time an agent enters a timed stage.
        self.timer = np.zeros(n, dtype=float)

        # Pre-existing immunity.
        imm = self.config.population.initial_immune_fraction
        if imm > 0:
            # np.where(mask)[0] yields the integer row indices where mask is True.
            sus_idx = np.where(self.state == SUS)[0]
            k = int(round(imm * sus_idx.size))
            if k > 0:
                # Sample k distinct agents (replace=False) and mark them recovered.
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
                # u is one uniform(0,1) draw per chosen agent; comparing to the
                # per-agent asymptomatic probability gives a boolean mask.
                u = self.rng.random(chosen.size)
                asymp = u < self.p.p_asymp[self.age[chosen]]
                # chosen[asymp] / chosen[~asymp] select the two disjoint subsets.
                self.state[chosen[asymp]] = IA
                self.state[chosen[~asymp]] = IS
                self._assign_infectivity(chosen)
                # Give the seeds a remaining time in their (infectious) stage.
                self._sample_duration(chosen[asymp], 1.0 / self.p.gamma_a)
                self._sample_duration(chosen[~asymp], 1.0 / self.p.gamma_s)

    def _assign_infectivity(self, idx: np.ndarray) -> None:
        """Draw per-agent onward infectiousness for newly infected agents."""
        od = self.config.simulation.overdispersion
        if od is None or not self.stochastic or idx.size == 0:
            self.infectivity[idx] = 1.0
        else:
            # Mean-one Gamma(shape=k, scale=1/k); small k => bursty superspreading.
            self.infectivity[idx] = self.rng.gamma(shape=od, scale=1.0 / od, size=idx.size)

    def _sample_duration(self, idx: np.ndarray, mean_days: float) -> None:
        """Set the stage countdown for agents ``idx`` entering a timed stage.

        Durations follow a Gamma with the configured shape ``k`` and the given
        mean, so ``CV = 1/sqrt(k)``: ``k=1`` is the memoryless exponential (high
        spread), while larger ``k`` clusters durations tightly around the mean —
        the realistic case for incubation/infectious periods. The mean is
        unchanged either way, so R0 is unaffected.
        """
        if idx.size == 0:
            return
        k = self.duration_shape
        self.timer[idx] = self.rng.gamma(k, mean_days / k, size=idx.size)

    # --------------------------------------------------------------- network
    def _build_network(self) -> None:
        """Construct contact layers (if enabled) and finalise calibration.

        When the network is disabled this reduces exactly to the mean-field
        engine: no layers, a community weight of 1, and beta calibrated on the
        plain age contact matrix.
        """
        net = self.config.network
        if net.enabled:
            # build_layers draws from rng to assign households/schools/workplaces.
            self.layers = build_layers(net, self.age, self.rng, self.n_age)
            self.community_weight = float(net.community_weight)
        else:
            self.layers = []
            self.community_weight = 1.0
        self._finalize_network()

    def _finalize_network(self) -> None:
        """Derive the effective contact matrix, per-group scalings and beta.

        Each layer contributes an age-mixing matrix (computed from the actual
        constructed groups); summed with the weighted community matrix this gives
        an effective contact structure whose dominant eigenvalue we calibrate to
        the target R0. Idempotent: also used after restoring a snapshot.
        """
        c_eff = self.community_weight * self.contact
        self.layer_scaling = []
        self.layer_mixing = []          # each layer's age-mixing matrix, for Rt/NPIs
        for layer in self.layers:
            mixing = layer_mixing_matrix(layer, self.age, self.n_age, self.N)
            self.layer_mixing.append(mixing)
            c_eff = c_eff + layer.weight * mixing
            # Cache the per-group divisor used by the stochastic within-group FOI.
            self.layer_scaling.append(layer.group_scaling())
        # Beta is calibrated on the intervention-free contact structure, so the
        # target R0 is the "no measures" reproduction number.
        self.c_eff = c_eff
        self.beta = calibrate_beta(
            self.c_eff, self.p.infectious_duration, self.config.disease.r0,
            self.susceptibility,
        )

    def _effective_contact(self, day: float) -> np.ndarray:
        """Contact structure with each channel scaled by its active interventions.

        Global interventions hit every channel; a layer-targeted one (e.g. a
        school closure) scales only its own layer. Used for the reported Rt.
        """
        iv = self.config.interventions
        c = iv.multiplier(day, "community") * self.community_weight * self.contact
        for layer, mixing in zip(self.layers, self.layer_mixing):
            c = c + iv.multiplier(day, layer.name) * layer.weight * mixing
        return c

    # --------------------------------------------------------------- helpers
    def current_day(self) -> float:
        return self.t * self.dt

    def beta_effective(self, day: float) -> float:
        # Calibrated rate, scaled by active interventions and the seasonal cycle.
        return (
            self.beta
            * self.config.interventions.multiplier(day)
            * self.config.environment.seasonal_multiplier(day)
        )

    def _counts_by_age(self, mask: np.ndarray) -> np.ndarray:
        """Number of agents in each age group among those selected by ``mask``."""
        # bincount tallies the age codes of the masked agents; minlength pads the
        # result to a full (n_age,) vector even if some ages have zero counts.
        return np.bincount(self.age[mask], minlength=self.n_age).astype(float)

    def susceptibility_by_age(self) -> np.ndarray:
        """Effective susceptible fraction per age (S plus leaky-protected V)."""
        ve = self.config.vaccination.ve_susceptibility
        sus = self.state == SUS
        s_by_age = self._counts_by_age(sus & ~self.vacc)
        v_by_age = self._counts_by_age(sus & self.vacc)
        return (s_by_age + (1.0 - ve) * v_by_age) / self.N_safe

    def effective_rt(self, day: float) -> float:
        """Model-implied effective reproduction number at ``day``.

        Uses the *effective* contact matrix (community plus any network layers),
        so the reported Rt accounts for the structured contacts. Age-level
        susceptibility scaling is an approximation under networks (depletion is
        partly local), but remains a good summary diagnostic.
        """
        sus = self.susceptibility_by_age()
        # Use the per-day effective contact structure (interventions baked in per
        # channel); seasonality multiplies the calibrated beta.
        c_eff_t = self._effective_contact(day)
        k = sus[:, None] * ngm_unit(c_eff_t, self.p.infectious_duration, self.susceptibility)
        return self.beta * self.config.environment.seasonal_multiplier(day) * spectral_radius(k)

    # ----------------------------------------------------------- transitions
    def _bernoulli(self, mask: np.ndarray, prob) -> np.ndarray:
        """Indices of agents in ``mask`` that fire with per-agent ``prob``.

        Deterministic mode is undefined for individuals, so the agent engine is
        always stochastic; the per-agent draw is the model.
        """
        idx = np.where(mask)[0]              # agent ids where mask is True
        if idx.size == 0:
            return idx
        p = np.asarray(prob, dtype=float)
        if p.ndim == 0:
            # Scalar probability: one uniform draw per candidate vs that constant.
            hit = self.rng.random(idx.size) < float(p)
        else:
            # Per-agent probability array: index it down to just the candidates.
            hit = self.rng.random(idx.size) < p[idx]
        return idx[hit]                      # keep only the ids whose draw fired

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
            # Sort eligible ids by descending age (stable keeps relative order),
            # then take the first n_doses (the oldest available).
            order = elig_idx[np.argsort(-self.age[elig_idx], kind="stable")]
            chosen = order[:n_doses]
        else:
            # No prioritisation: pick n_doses distinct eligible agents at random.
            chosen = self.rng.choice(elig_idx, size=n_doses, replace=False)
        self.vacc[chosen] = True
        self.cumulative_vaccinated += chosen.size * self.scale

    def step(self) -> StepRecord:
        """Advance the epidemic by one time step and return a summary record."""
        day = self.current_day()

        # 1. Vaccination (administrative S -> V flag) at the start of the day.
        self._vaccinate(day)

        # 2. Force of infection from the current infectious agents. Two channels:
        #    (a) the age-mixed community layer, and (b) within-group transmission
        #    in each network layer. Each channel is scaled by its own active
        #    interventions, so a layer-targeted NPI (e.g. a school closure) hits
        #    only that layer. beta_season is the calibrated rate times seasonality.
        iv = self.config.interventions
        beta_season = self.beta * self.config.environment.seasonal_multiplier(day)
        beta_eff = beta_season * iv.multiplier(day)   # global-only, for the record field
        # Per-agent infectious-phase weight (0 for non-infectious agents).
        weight = np.zeros(self.n_agents)
        is_ip, is_ia, is_is = self.state == IP, self.state == IA, self.state == IS
        weight[is_ip] = self.p.rel_p
        weight[is_ia] = self.p.rel_a
        weight[is_is] = 1.0
        # Full-length per-agent contribution = stratum factor * phase weight *
        # individual load. Zero for non-infectious agents (weight == 0), so it can
        # be summed over everyone without masking.
        f_strat = np.where(self.vacc, self.p.f_transmission[1], self.p.f_transmission[0])
        contrib = weight * f_strat * self.infectivity        # (n_agents,)

        # (a) Community: pressure per age -> per-age FOI -> back onto each agent.
        pressure = np.bincount(self.age, weights=contrib, minlength=self.n_age)
        prevalence = pressure / self.N_safe
        comm_beta = beta_season * iv.multiplier(day, "community") * self.community_weight
        foi_comm = comm_beta * (self.contact @ prevalence)   # (n_age,)
        foi_agent = foi_comm[self.age]                       # (n_agents,)

        # (b) Network layers: each susceptible gains FOI from the infectious load
        #     in its own household/class/workplace, divided by the layer scaling.
        for layer, scaling in zip(self.layers, self.layer_scaling):
            gid = layer.group_id
            member = gid >= 0
            # Total infectious contribution in each group of this layer.
            load = np.bincount(gid[member], weights=contrib[member], minlength=scaling.size)
            # Add each member's own-group exposure (gather load/scaling by group id),
            # scaled by this layer's own active interventions.
            gm = gid[member]
            layer_beta = beta_season * iv.multiplier(day, layer.name) * layer.weight
            foi_agent[member] += layer_beta * load[gm] / scaling[gm]

        # (c) External/spillover hazard (importations / reservoir) plus any
        #     imported force from other regions, then scale the whole per-agent
        #     hazard by each agent's age-specific susceptibility.
        external = self.config.environment.external_force(day) + self.imported_force
        foi_agent = self.susceptibility[self.age] * (foi_agent + external)

        ve_sus = self.config.vaccination.ve_susceptibility
        sus = self.state == SUS
        # Vaccinated agents experience a reduced FOI.
        foi_agent = np.where(self.vacc, foi_agent * (1.0 - ve_sus), foi_agent)
        p_inf = transition_probability(foi_agent, self.dt)
        newly = self._bernoulli(sus, p_inf)

        # 3. Progression transitions. Each agent in a timed stage carries a
        #    countdown (self.timer) sampled when it entered the stage; it leaves
        #    once the countdown elapses. Decrement once per step, then a stage's
        #    leavers are the agents in that stage whose timer has run out. This
        #    gives realistically-peaked stage durations (not exponential ones).
        self.timer -= self.dt
        expired = self.timer <= 0.0
        leave_E = np.where((self.state == E) & expired)[0]
        leave_Ip = np.where((self.state == IP) & expired)[0]
        leave_Ia = np.where((self.state == IA) & expired)[0]
        leave_Is = np.where((self.state == IS) & expired)[0]
        leave_H = np.where((self.state == H) & expired)[0]
        leave_C = np.where((self.state == C) & expired)[0]

        # Branch the leavers via complementary boolean masks: each `leave_*` is an
        # array of agent ids, and masking it splits those ids into two subsets.
        asymp = self.rng.random(leave_E.size) < self.p.p_asymp[self.age[leave_E]]
        to_Ia, to_Ip = leave_E[asymp], leave_E[~asymp]

        # hosp_rate is (2, n_age): row by vaccination status, column by age. The
        # vacc flag cast to 0/1 selects the row, self.age selects the column, so
        # this advanced-indexing yields one probability per Is leaver.
        hosp_p = self.p.hosp_rate[self.vacc[leave_Is].astype(int), self.age[leave_Is]]
        hospitalised = self.rng.random(leave_Is.size) < hosp_p
        to_H, Is_to_R = leave_Is[hospitalised], leave_Is[~hospitalised]

        critical = self.rng.random(leave_H.size) < self.p.icu_rate[self.age[leave_H]]
        to_C, H_to_R = leave_H[critical], leave_H[~critical]

        # Death probability depends on ICU occupancy at population scale (current
        # C count times scale), then is looked up per leaver by age.
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

        # Sample a fresh sojourn duration for everyone entering a timed stage this
        # step (set after the decrement so it counts down from next step).
        self._sample_duration(newly, 1.0 / self.p.sigma)        # -> E
        self._sample_duration(to_Ip, 1.0 / self.p.gamma_p)      # -> Ip
        self._sample_duration(to_Ia, 1.0 / self.p.gamma_a)      # -> Ia
        self._sample_duration(leave_Ip, 1.0 / self.p.gamma_s)   # -> Is
        self._sample_duration(to_H, 1.0 / self.p.gamma_h)       # -> H
        self._sample_duration(to_C, 1.0 / self.p.gamma_c)       # -> C

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
        # Tally agents per state code (0..8) and lift to population scale; index
        # with the state constants (E, IP, ...) to read each compartment total.
        counts = np.bincount(self.state, minlength=9).astype(float) * self.scale
        sus = self.state == SUS
        s_count = float((sus & ~self.vacc).sum()) * self.scale
        v_count = float((sus & self.vacc).sum()) * self.scale

        # np.isin builds a boolean mask True where state is any of the three codes.
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

    def infectious_weighted_fraction(self) -> float:
        """Infectiousness-weighted infectious prevalence (for region coupling)."""
        ip = float((self.state == IP).sum())
        ia = float((self.state == IA).sum())
        is_ = float((self.state == IS).sum())
        # scale cancels (numerator and denominator are both agent-scale counts).
        return (self.p.rel_p * ip + self.p.rel_a * ia + is_) / self.n_agents

    def infectious_count(self) -> float:
        """Number of currently infectious agents (Ip + Ia + Is), population scale."""
        return float(np.isin(self.state, (IP, IA, IS)).sum()) * self.scale

    def susceptible_fraction(self) -> float:
        """Fraction of agents currently susceptible."""
        return float((self.state == SUS).sum()) / self.n_agents

    def mean_infectious_duration(self) -> float:
        return float(self.p.infectious_duration.mean())

    def seed_exposed(self, amount: float) -> float:
        """Move ~``amount`` (population-scale) susceptible agents to E (importation).

        Picks that many susceptible individuals at random, infects them, and gives
        them an infectiousness and a fresh latent duration. Returns the number
        actually seeded (population scale).
        """
        if amount <= 0:
            return 0.0
        n_new = int(round(amount / self.scale))            # population amount -> agents
        sus_idx = np.where(self.state == SUS)[0]
        n_new = min(n_new, sus_idx.size)
        if n_new <= 0:
            return 0.0
        chosen = self.rng.choice(sus_idx, size=n_new, replace=False)
        self.state[chosen] = E
        self._assign_infectivity(chosen)
        self._sample_duration(chosen, 1.0 / self.p.sigma)
        return n_new * self.scale

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
            "timer": self.timer.tolist(),
            # The network is random, so its group assignments are part of the
            # state and must be persisted to resume an identical run.
            "community_weight": self.community_weight,
            "layers": [
                {
                    "name": layer.name,
                    "group_id": layer.group_id.tolist(),
                    "weight": layer.weight,
                    "density_dependent": layer.density_dependent,
                }
                for layer in self.layers
            ],
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
        # Older snapshots (pre-timed-durations) won't carry a timer; default to
        # zeros, which simply makes any in-progress stages resolve promptly.
        if "timer" in state:
            self.timer = restore_array(state, "timer", (n,), dtype=float)
        else:
            self.timer = np.zeros(n, dtype=float)

        # Restore the contact network. Group-id arrays are untrusted like every
        # other field: each must be length n and hold ids in [-1, n) (-1 = not a
        # member). Then re-derive c_eff/beta/scalings from the restored layers.
        self.community_weight = float(state.get("community_weight", 1.0))
        restored = []
        for ld in state.get("layers", []):
            gid = restore_array(ld, "group_id", (n,), dtype=np.int64)
            if gid.size and (gid.min() < -1 or gid.max() >= n):
                raise ValueError("snapshot contains out-of-range network group ids")
            restored.append(ContactLayer(
                name=str(ld.get("name", "")),
                group_id=gid,
                weight=float(ld["weight"]),
                density_dependent=bool(ld["density_dependent"]),
            ))
        self.layers = restored
        self._finalize_network()

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
