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
    """One geographic patch: a named population with its own initial seeding.

    ``x``/``y`` are optional map coordinates (any units) used purely for the
    spatial view; they don't affect the dynamics.
    """

    name: str
    population: int
    initial_infected: int = 0
    x: Optional[float] = None
    y: Optional[float] = None

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
    coupling: float = 0.01                 # smooth prevalence-coupling strength
    # Optional explicit K x K mobility matrix. If None, one is derived from
    # ``mobility_model`` below. The diagonal is ignored (within-region spread is
    # the local model).
    mobility: Optional[Sequence[Sequence[float]]] = None
    # How to derive mobility when none is supplied:
    #   "uniform" - mix equally with every other region.
    #   "gravity" - flow to a region grows with its population and falls with
    #               distance (needs x/y coordinates): M[i, j] ∝ N_j / d_ij^decay.
    mobility_model: str = "uniform"
    gravity_decay: float = 2.0             # distance exponent for the gravity model
    # If True, x/y are read as longitude/latitude (degrees) and the gravity model
    # uses great-circle (haversine) distances in km instead of plane Euclidean.
    geographic_coords: bool = False
    # Explicit travel: per-infectious-person daily probability of taking a trip
    # that seeds an importation in another region (discrete, stochastic). 0 = off.
    # Trip destinations follow the same mobility matrix.
    travel_rate: float = 0.0

    MOBILITY_MODELS = ("uniform", "gravity")

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
        if self.travel_rate < 0:
            raise ValueError("travel_rate must be >= 0")
        if self.mobility_model not in self.MOBILITY_MODELS:
            raise ValueError(f"mobility_model must be one of {self.MOBILITY_MODELS}")
        if self.gravity_decay <= 0:
            raise ValueError("gravity_decay must be > 0")
        if self.mobility is not None:
            m = np.asarray(self.mobility, dtype=float)
            k = len(self.regions)
            if m.shape != (k, k):
                raise ValueError(f"mobility must be {k}x{k}, got {m.shape}")
            if np.any(m < 0):
                raise ValueError("mobility entries must be non-negative")
        elif self.mobility_model == "gravity":
            if any(r.x is None or r.y is None for r in self.regions):
                raise ValueError("gravity mobility_model requires x/y on every region")
        return self

    def _gravity_matrix(self) -> np.ndarray:
        """Gravity weights M[i, j] ∝ N_j / distance(i, j)^gravity_decay."""
        coords = np.array([[r.x, r.y] for r in self.regions], dtype=float)
        pops = np.array([r.population for r in self.regions], dtype=float)
        if self.geographic_coords:
            dist = _haversine_km(coords[:, 0], coords[:, 1])   # x=lon, y=lat (degrees)
        else:
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1))           # plane Euclidean
        # Self-distance -> inf so the diagonal weight is zero, and any coincident
        # pair is treated as unconnected rather than dividing by zero.
        dist = np.where(dist == 0.0, np.inf, dist)
        w = pops[None, :] / np.power(dist, self.gravity_decay)   # M[i, j] ∝ N_j / d^decay
        w[~np.isfinite(w)] = 0.0
        return w

    def mobility_matrix(self) -> np.ndarray:
        """Row-normalised coupling weights with a zero diagonal."""
        k = len(self.regions)
        if self.mobility is not None:
            m = np.asarray(self.mobility, dtype=float).copy()
        elif self.mobility_model == "gravity":
            m = self._gravity_matrix()
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
        self.travel_rate = self.config.travel_rate
        self.dt = self.base.simulation.dt
        self._stochastic_travel = self.base.simulation.stochastic
        self._n_steps = self.base.simulation.n_steps
        self.t = 0

        # Independent, reproducible RNG stream per region, plus one for the
        # orchestrator's own travel draws.
        k = len(self.config.regions)
        seeds = np.random.SeedSequence(base_seed).spawn(k + 1)
        self.rng = np.random.default_rng(seeds[k])
        self.engines = []
        self.histories: List[List[StepRecord]] = []
        for region, seed in zip(self.config.regions, seeds[:k]):
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
        # 1. Smooth prevalence coupling: imported force = coupling*beta_r*(M@prev)_r.
        prevalence = np.array([e.infectious_weighted_fraction() for e in self.engines])
        inflow = self.mobility @ prevalence
        for engine, flow in zip(self.engines, inflow):
            engine.imported_force = float(self.coupling * engine.beta * flow)
        # 2. Explicit travel: infectious individuals take trips and seed importations.
        if self.travel_rate > 0:
            self._apply_travel()
        # 3. Step every region and record.
        records = []
        for engine, history in zip(self.engines, self.histories):
            rec = engine.step()
            history.append(rec)
            records.append(rec)
        self.t += 1
        return records

    def _apply_travel(self) -> None:
        """Seed importations caused by infectious individuals travelling.

        Each infectious person has a daily probability ``travel_rate`` of making a
        trip; trip destinations follow the mobility matrix. While visiting region
        ``s`` a traveller infects new people at the same rate a local infectious
        person would — ``(R0_s / mean_infectious_duration_s) · susceptible_frac_s``
        per day — so the expected new infections seeded in ``s`` are::

            Σ_r  infectious_r · travel_rate · M[r, s] · (R0_s/D_s) · Ssed_frac_s · dt

        drawn as a Poisson count (or used directly in deterministic mode).
        """
        infectious = np.array([e.infectious_count() for e in self.engines])
        if infectious.sum() <= 0:
            return
        # Per-destination local transmissibility: secondary infections one visiting
        # infectious person would generate per day, given the destination's
        # remaining susceptibles.
        transmissibility = np.array([
            (e.config.disease.r0 / e.mean_infectious_duration()) * e.susceptible_fraction()
            for e in self.engines
        ])
        # arriving[s] = Σ_r M[s, r] * infectious_r  (infectious trips into region s).
        # Same direction convention as the prevalence coupling (region receives
        # from its sources, weighted by its mobility row).
        arriving = self.mobility @ infectious
        expected = arriving * self.travel_rate * transmissibility * self.dt
        for engine, exp_new in zip(self.engines, expected):
            if exp_new <= 0:
                continue
            amount = self.rng.poisson(exp_new) if self._stochastic_travel else exp_new
            engine.seed_exposed(float(amount))

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
                "travel_rate": self.config.travel_rate,
                "mobility_model": self.config.mobility_model,
                "gravity_decay": self.config.gravity_decay,
                "geographic_coords": self.config.geographic_coords,
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
            coupling=m["coupling"], travel_rate=m.get("travel_rate", 0.0),
            mobility=m.get("mobility"),
            mobility_model=m.get("mobility_model", "uniform"),
            gravity_decay=m.get("gravity_decay", 2.0),
            geographic_coords=m.get("geographic_coords", False),
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


def _haversine_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) between points given in degrees.

    Returns a (K, K) matrix. The haversine formula gives the distance along the
    Earth's surface, so it's correct near the poles and across the date line —
    unlike treating longitude/latitude as a flat plane.
    """
    earth_radius_km = 6371.0
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# Per-step record fields that are simply summed across regions.
_ADDITIVE = (
    "S", "V", "E", "Ip", "Ia", "Is", "H", "C", "R", "D",
    "new_infections", "new_symptomatic", "new_hospitalizations",
    "new_icu", "new_deaths", "icu_overflow",
)
