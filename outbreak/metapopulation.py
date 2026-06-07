"""Geography: a metapopulation of coupled regions.

Real epidemics are spatial — they start somewhere and spread to other places over
time. :class:`MetapopulationSimulation` models that by running **one engine per
region** (each its own age-structured population and epidemic) and **coupling**
them so infection leaks between connected regions.

Coupling
--------
Each step, every region's susceptibles feel an **imported force of infection**
proportional to the infectious prevalence of the regions it is connected to:

    imported_r = coupling · beta_r · Σ_s  M[r, s] · prevalence_s

where ``M`` is a (row-normalised) **mobility matrix** — ``M[r, s]`` is how much of
region ``r``'s outside exposure comes from region ``s`` — and ``coupling`` is the
overall strength of between-region mixing (a small fraction; the bulk of
transmission stays local). Using each region's own ``beta`` puts the imported
hazard in the same units as its local force of infection.

This is the standard "coupled-patch" metapopulation: the configured **R0 is the
within-region** reproduction number, and coupling adds spatial spread on top
(seeding new regions and synchronising waves). The orchestrator reuses the
engines' imported-force hook, so the core single-population models are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .config import ScenarioConfig
from .metrics import EpidemicSummary, history_to_columns, summarize
from .model import StepRecord
from .simulation import _json_default, build_engine


@dataclass
class Region:
    """One geographic patch: a named population with its own initial seeding."""

    name: str
    population: int
    initial_infected: int = 0

    def validate(self) -> "Region":
        if self.population <= 0:
            raise ValueError(f"region {self.name!r}: population must be > 0")
        if not 0 <= self.initial_infected <= self.population:
            raise ValueError(f"region {self.name!r}: initial_infected out of range")
        return self


@dataclass
class MetapopulationConfig:
    """A set of regions plus how they're connected."""

    regions: Sequence[Region]
    coupling: float = 0.01                 # overall between-region mixing strength
    # Optional K x K mobility matrix. None => uniform mixing with every other
    # region. The diagonal is ignored (within-region spread is the local model).
    mobility: Optional[Sequence[Sequence[float]]] = None

    def __post_init__(self) -> None:
        # Allow plain dicts (e.g. from a loaded snapshot) in place of Region objects.
        self.regions = [r if isinstance(r, Region) else Region(**r) for r in self.regions]

    def validate(self) -> "MetapopulationConfig":
        if len(self.regions) < 1:
            raise ValueError("a metapopulation needs at least one region")
        for r in self.regions:
            r.validate()
        if self.coupling < 0:
            raise ValueError("coupling must be >= 0")
        if self.mobility is not None:
            m = np.asarray(self.mobility, dtype=float)
            k = len(self.regions)
            if m.shape != (k, k):
                raise ValueError(f"mobility must be {k}x{k}, got {m.shape}")
            if np.any(m < 0):
                raise ValueError("mobility entries must be non-negative")
        return self

    def mobility_matrix(self) -> np.ndarray:
        """Row-normalised coupling weights with a zero diagonal."""
        k = len(self.regions)
        if self.mobility is not None:
            m = np.asarray(self.mobility, dtype=float).copy()
        else:
            m = np.ones((k, k)) - np.eye(k)        # uniform mixing with all others
        np.fill_diagonal(m, 0.0)                    # self-coupling is the local model
        rowsum = m.sum(axis=1, keepdims=True)
        # Each region's outside-exposure weights sum to 1 (rows with no neighbours
        # stay all-zero, i.e. isolated).
        return np.divide(m, rowsum, out=np.zeros_like(m), where=rowsum > 0)


class MetapopulationSimulation:
    """Run and observe several coupled regional epidemics together."""

    def __init__(self, base_scenario: ScenarioConfig, config: MetapopulationConfig,
                 base_seed: Optional[int] = None):
        self.config = config.validate()
        self.base = base_scenario.validate()
        self.mobility = self.config.mobility_matrix()
        self.coupling = self.config.coupling
        self._n_steps = self.base.simulation.n_steps
        self.t = 0

        # Independent, reproducible RNG stream per region.
        seeds = np.random.SeedSequence(base_seed).spawn(len(self.config.regions))
        self.engines = []
        self.histories: List[List[StepRecord]] = []
        for region, seed in zip(self.config.regions, seeds):
            sc = self._region_scenario(region)
            self.engines.append(build_engine(sc, rng=np.random.default_rng(seed)))
            self.histories.append([])

    @property
    def region_names(self) -> List[str]:
        return [r.name for r in self.config.regions]

    def _region_scenario(self, region: Region) -> ScenarioConfig:
        """Base scenario specialised to a region's population and seeding."""
        sc = ScenarioConfig.from_dict(self.base.to_dict())   # deep copy
        sc.population.total_population = region.population
        sc.population.initial_infected = region.initial_infected
        return sc.validate()

    # ----------------------------------------------------------- stepping
    def step(self) -> Optional[List[StepRecord]]:
        """Advance every region one step, after applying between-region coupling."""
        if self.t >= self._n_steps:
            return None
        # 1. Current infectious prevalence in each region.
        prevalence = np.array([e.infectious_weighted_fraction() for e in self.engines])
        # 2. Imported force per region = coupling * beta_r * (M @ prevalence)_r.
        inflow = self.mobility @ prevalence
        for engine, flow in zip(self.engines, inflow):
            engine.imported_force = float(self.coupling * engine.beta * flow)
        # 3. Step every region and record.
        records = []
        for engine, history in zip(self.engines, self.histories):
            rec = engine.step()
            history.append(rec)
            records.append(rec)
        self.t += 1
        return records

    def run_to_end(self) -> None:
        while self.step() is not None:
            pass

    # --------------------------------------------------------- aggregation
    def combined_history(self) -> List[StepRecord]:
        """Per-step records summed across all regions (Rt is population-weighted)."""
        if not self.histories[0]:
            return []
        pops = np.array([e.total_population() for e in self.engines])
        total_pop = pops.sum()
        combined: List[StepRecord] = []
        for step_records in zip(*self.histories):
            agg = {f: float(sum(getattr(r, f) for r in step_records)) for f in _ADDITIVE}
            # Rt has no meaningful sum; population-weight it as a summary diagnostic.
            rt = float(np.average([r.rt for r in step_records], weights=pops)) if total_pop else 0.0
            combined.append(StepRecord(
                day=step_records[0].day, rt=rt,
                beta_effective=float(np.mean([r.beta_effective for r in step_records])),
                infectious_by_age=sum(r.infectious_by_age for r in step_records),
                deaths_by_age=sum(r.deaths_by_age for r in step_records),
                **agg,
            ))
        return combined

    def summary(self) -> Optional[EpidemicSummary]:
        """Summary of the whole metapopulation (all regions combined)."""
        return summarize(self.combined_history(), self.base.disease.r0, self.base.reporting)

    def region_summary(self, index: int) -> Optional[EpidemicSummary]:
        return summarize(self.histories[index], self.base.disease.r0, self.base.reporting)

    def to_columns(self) -> dict:
        """Combined time series plus a per-region cumulative-infection column."""
        cols = history_to_columns(self.combined_history(), self.base.reporting)
        for name, history in zip(self.region_names, self.histories):
            inf = np.cumsum([r.new_infections for r in history]).tolist()
            cols[f"cumulative_infections::{name}"] = inf
        return cols

    def first_infection_day(self, index: int, threshold: float = 1.0) -> Optional[float]:
        """Day region ``index`` first exceeds ``threshold`` cumulative infections."""
        cum = 0.0
        for rec in self.histories[index]:
            cum += rec.new_infections
            if cum >= threshold:
                return rec.day
        return None

    # ----------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "base": self.base.to_dict(),
            "metapop": {
                "regions": [vars(r) for r in self.config.regions],
                "coupling": self.config.coupling,
                "mobility": (np.asarray(self.config.mobility).tolist()
                             if self.config.mobility is not None else None),
            },
            "t": self.t,
            "engine_states": [e.get_state() for e in self.engines],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)

    @classmethod
    def from_dict(cls, data: dict) -> "MetapopulationSimulation":
        base = ScenarioConfig.from_dict(data["base"])
        m = data["metapop"]
        config = MetapopulationConfig(
            regions=[Region(**r) for r in m["regions"]],
            coupling=m["coupling"], mobility=m.get("mobility"),
        )
        sim = cls(base, config)
        states = data["engine_states"]
        if len(states) != len(sim.engines):
            raise ValueError("snapshot region count does not match the configuration")
        for engine, state in zip(sim.engines, states):
            engine.set_state(state)            # validated/hardened per engine
        sim.t = int(data.get("t", 0))
        return sim

    @classmethod
    def load(cls, path: str) -> "MetapopulationSimulation":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# Per-step record fields that are simply summed across regions.
_ADDITIVE = (
    "S", "V", "E", "Ip", "Ia", "Is", "H", "C", "R", "D",
    "new_infections", "new_symptomatic", "new_hospitalizations",
    "new_icu", "new_deaths", "icu_overflow",
)
