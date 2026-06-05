"""Configuration schema for the epidemic simulation.

All tunable parameters live here, grouped into validated dataclasses. The whole
configuration is serialisable to/from plain dictionaries so scenarios can be
saved, loaded, compared and shared. A small library of disease *presets*
(COVID-like, influenza-like, measles-like) provides realistic starting points.

Design notes
------------
* Epidemiological parameters describe *biology and behaviour*, not the raw
  transmission rate ``beta``. The user specifies a target basic reproduction
  number ``r0`` and the engine calibrates ``beta`` from the next-generation
  matrix (see :mod:`outbreak.model`). This is the standard, defensible way to
  parameterise a structured compartmental model.
* Severity is expressed as a *conditional cascade*: of those infected a fraction
  are asymptomatic; of the symptomatic a fraction are hospitalised; of the
  hospitalised a fraction need ICU; of the ICU patients a fraction die. Each
  fraction may vary by age group.
* Any parameter that can vary by age accepts either a scalar (broadcast to all
  age groups) or a per-age sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Union

import numpy as np

Number = Union[int, float]
PerAge = Union[Number, Sequence[Number]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_array(value: PerAge, n_age: int, name: str) -> np.ndarray:
    """Coerce a scalar or per-age sequence into a float array of length n_age."""
    # atleast_1d turns a bare scalar (e.g. 0.35) into a length-1 array so the
    # broadcast/length logic below can treat scalar and sequence inputs uniformly.
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        arr = np.repeat(arr, n_age)  # broadcast a single scalar to one value per age group
    if arr.size != n_age:  # otherwise the caller gave a wrong-length sequence
        raise ValueError(
            f"{name!r} must be a scalar or have length {n_age}, got length {arr.size}"
        )
    return arr


def _check_prob(arr: np.ndarray, name: str) -> None:
    # np.any reduces an element-wise comparison over the whole array to one bool.
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name!r} must be a probability in [0, 1]")


def _check_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name!r} must be > 0, got {value}")


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

# @dataclass auto-generates __init__/__repr__/__eq__ from the annotated fields
# below; each field's value is its default.
@dataclass
class PopulationConfig:
    """Population size, age structure and initial seeding.

    ``age_group_labels`` and ``age_distribution`` define the age strata. If left
    as ``None`` a default four-band structure (children / young adults / older
    adults / elderly) with an illustrative contact matrix is used.
    """

    total_population: int = 1_000_000
    age_group_labels: Optional[Sequence[str]] = None
    age_distribution: Optional[Sequence[float]] = None
    initial_infected: int = 10
    # Fraction of the population already immune at t=0 (prior infection/vaccine).
    initial_immune_fraction: float = 0.0

    # __post_init__ runs automatically right after the generated __init__, so it
    # is the hook for validating/normalising the supplied field values.
    def __post_init__(self) -> None:
        if self.total_population <= 0:
            raise ValueError("total_population must be > 0")
        if self.initial_infected < 0:
            raise ValueError("initial_infected must be >= 0")
        if self.initial_infected > self.total_population:
            raise ValueError("initial_infected cannot exceed total_population")
        if not 0.0 <= self.initial_immune_fraction < 1.0:
            raise ValueError("initial_immune_fraction must be in [0, 1)")

        # The defaults are None (not a mutable list) on the dataclass; fill them
        # in here only when the caller left both unset.
        if self.age_group_labels is None or self.age_distribution is None:
            self.age_group_labels = list(DEFAULT_AGE_LABELS)
            self.age_distribution = list(DEFAULT_AGE_DISTRIBUTION)

        self.age_group_labels = list(self.age_group_labels)  # copy into a list we own
        dist = np.asarray(self.age_distribution, dtype=float)
        if len(self.age_group_labels) != dist.size:
            raise ValueError("age_group_labels and age_distribution length mismatch")
        if dist.size == 0:
            raise ValueError("at least one age group is required")
        if np.any(dist < 0):
            raise ValueError("age_distribution entries must be non-negative")
        if not np.isclose(dist.sum(), 1.0):  # tolerant float equality vs exactly 1.0
            dist = dist / dist.sum()  # normalise defensively
        self.age_distribution = dist.tolist()  # store as plain list (JSON-friendly)

    @property
    def n_age(self) -> int:
        return len(self.age_group_labels)

    def population_by_age(self) -> np.ndarray:
        """Integer population in each age group, summing to total_population."""
        dist = np.asarray(self.age_distribution, dtype=float)
        raw = dist * self.total_population         # ideal (fractional) counts
        counts = np.floor(raw).astype(np.int64)    # round down, leaving a shortfall
        # Distribute the rounding remainder to the largest fractional parts.
        remainder = self.total_population - int(counts.sum())
        if remainder > 0:
            # argsort of the negated fractional parts => indices ordered largest-first.
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1         # give +1 to the top `remainder` groups
        return counts


# ---------------------------------------------------------------------------
# Disease biology
# ---------------------------------------------------------------------------

@dataclass
class DiseaseConfig:
    """Natural history of the disease.

    Timing parameters are mean durations in days. Severity parameters are
    conditional probabilities along the cascade and may be per-age.
    """

    name: str = "custom"
    r0: float = 2.5

    # Timing (mean durations, days)
    latent_period: float = 3.0                 # E -> infectious (not yet symptomatic)
    presymptomatic_period: float = 2.0         # infectious-but-presymptomatic window
    symptomatic_period: float = 6.0            # symptomatic & infectious window
    asymptomatic_infectious_period: float = 6.0

    # Relative infectiousness of the non-symptomatic phases (symptomatic = 1.0)
    rel_infectiousness_presymptomatic: float = 1.0
    rel_infectiousness_asymptomatic: float = 0.5

    # Severity cascade (per-age conditional probabilities)
    asymptomatic_fraction: PerAge = 0.35       # of infections
    hospitalization_rate: PerAge = 0.04        # of symptomatic
    icu_rate: PerAge = 0.20                    # of hospitalised
    death_rate: PerAge = 0.30                  # of ICU patients

    # Clinical stay durations (days)
    hospital_stay: float = 8.0
    icu_stay: float = 10.0

    # Immunity waning: mean days until a recovered person becomes susceptible
    # again. ``None`` or 0 disables waning (lifelong immunity).
    waning_immunity_days: Optional[float] = None

    def validate(self, n_age: int) -> "DiseaseConfig":
        _check_positive(self.r0, "r0")
        # Loop over field names and validate each via getattr, avoiding a wall of
        # near-identical checks. Returns self so callers can chain .validate().
        for fld in (
            "latent_period",
            "presymptomatic_period",
            "symptomatic_period",
            "asymptomatic_infectious_period",
            "hospital_stay",
            "icu_stay",
        ):
            _check_positive(getattr(self, fld), fld)
        for fld in (
            "rel_infectiousness_presymptomatic",
            "rel_infectiousness_asymptomatic",
        ):
            if getattr(self, fld) < 0:
                raise ValueError(f"{fld!r} must be >= 0")
        if self.waning_immunity_days is not None and self.waning_immunity_days <= 0:
            raise ValueError("waning_immunity_days must be > 0 or None")
        for fld in (
            "asymptomatic_fraction",
            "hospitalization_rate",
            "icu_rate",
            "death_rate",
        ):
            # _as_array first broadcasts scalar-or-per-age input to length n_age,
            # then _check_prob verifies every entry lies in [0, 1].
            _check_prob(_as_array(getattr(self, fld), n_age, fld), fld)
        return self

    # Per-age resolved arrays -------------------------------------------------
    # Each accessor returns the field expanded to a length-n_age float array,
    # regardless of whether the user supplied a scalar or a per-age sequence.
    def asymptomatic_fraction_arr(self, n_age: int) -> np.ndarray:
        return _as_array(self.asymptomatic_fraction, n_age, "asymptomatic_fraction")

    def hospitalization_rate_arr(self, n_age: int) -> np.ndarray:
        return _as_array(self.hospitalization_rate, n_age, "hospitalization_rate")

    def icu_rate_arr(self, n_age: int) -> np.ndarray:
        return _as_array(self.icu_rate, n_age, "icu_rate")

    def death_rate_arr(self, n_age: int) -> np.ndarray:
        return _as_array(self.death_rate, n_age, "death_rate")


# ---------------------------------------------------------------------------
# Vaccination
# ---------------------------------------------------------------------------

@dataclass
class VaccinationConfig:
    """Leaky vaccination with reduced susceptibility, severity and onward
    transmission. Doses are administered from a start day at a constant daily
    rate (as a fraction of the total population per day) up to a coverage cap.
    """

    enabled: bool = False
    start_day: int = 0
    daily_rate: float = 0.005          # fraction of total population vaccinated / day
    coverage_cap: float = 0.7          # max fraction of population ever vaccinated
    prioritize_elderly: bool = True    # vaccinate oldest susceptible groups first

    # Vaccine efficacies (all-or-nothing leaky factors in [0, 1])
    ve_susceptibility: float = 0.6     # reduction in probability of infection
    ve_severity: float = 0.8           # reduction in hospitalisation/ICU/death
    ve_transmission: float = 0.4       # reduction in infectiousness if infected

    def validate(self) -> "VaccinationConfig":
        if self.start_day < 0:
            raise ValueError("vaccination start_day must be >= 0")
        if not 0.0 <= self.daily_rate <= 1.0:
            raise ValueError("daily_rate must be in [0, 1]")
        if not 0.0 <= self.coverage_cap <= 1.0:
            raise ValueError("coverage_cap must be in [0, 1]")
        for fld in ("ve_susceptibility", "ve_severity", "ve_transmission"):
            v = getattr(self, fld)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{fld!r} must be in [0, 1]")
        return self


# ---------------------------------------------------------------------------
# Non-pharmaceutical interventions
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    """A time-windowed reduction in transmission (masking, distancing, lockdown).

    ``transmission_reduction`` is the fractional reduction applied to the
    effective contact rate while the intervention is active (inclusive of
    ``start_day``, exclusive of ``end_day``).
    """

    name: str = "intervention"
    start_day: int = 0
    end_day: int = 0
    transmission_reduction: float = 0.0

    def validate(self) -> "Intervention":
        if self.start_day < 0 or self.end_day < 0:
            raise ValueError("intervention days must be >= 0")
        if self.end_day <= self.start_day:
            raise ValueError("intervention end_day must be after start_day")
        if not 0.0 <= self.transmission_reduction <= 1.0:
            raise ValueError("transmission_reduction must be in [0, 1]")
        return self

    def is_active(self, day: int) -> bool:
        # Half-open interval [start_day, end_day): active on start_day, not on end_day.
        return self.start_day <= day < self.end_day


@dataclass
class InterventionConfig:
    # default_factory=list gives each instance its own fresh list; a bare
    # `= []` default would be shared across all instances (the classic mutable
    # default-argument bug).
    interventions: Sequence[Intervention] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Accept either Intervention objects or plain dicts (e.g. from a loaded
        # JSON scenario); turn any dict into an Intervention via **kwargs unpacking.
        self.interventions = [
            i if isinstance(i, Intervention) else Intervention(**i)
            for i in self.interventions
        ]

    def validate(self) -> "InterventionConfig":
        for i in self.interventions:
            i.validate()
        return self

    def multiplier(self, day: int) -> float:
        """Combined transmission multiplier from all active interventions.

        Reductions combine multiplicatively (independent layers of protection).
        """
        m = 1.0
        for i in self.interventions:
            if i.is_active(day):
                m *= (1.0 - i.transmission_reduction)  # stack reductions multiplicatively
        return m


# ---------------------------------------------------------------------------
# Healthcare capacity
# ---------------------------------------------------------------------------

@dataclass
class HealthcareConfig:
    """Healthcare capacity. When ICU demand exceeds capacity, patients who
    cannot get a bed experience elevated mortality.
    """

    # Capacities as an absolute number of beds. ``None`` means unlimited.
    hospital_capacity: Optional[int] = None
    icu_capacity: Optional[int] = None
    # Mortality multiplier applied to the share of ICU need above capacity.
    overflow_mortality_multiplier: float = 2.0

    def validate(self) -> "HealthcareConfig":
        if self.hospital_capacity is not None and self.hospital_capacity < 0:
            raise ValueError("hospital_capacity must be >= 0 or None")
        if self.icu_capacity is not None and self.icu_capacity < 0:
            raise ValueError("icu_capacity must be >= 0 or None")
        if self.overflow_mortality_multiplier < 1.0:
            raise ValueError("overflow_mortality_multiplier must be >= 1")
        return self


# ---------------------------------------------------------------------------
# Simulation controls
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Numerical/runtime controls."""

    duration_days: int = 365
    dt: float = 1.0                # time step in days
    stochastic: bool = True        # binomial transitions vs deterministic expectations
    # Overdispersion of transmission (superspreading). ``None`` disables it;
    # smaller values => more overdispersion. In the compartmental engine this is
    # the shape of a mean-one Gamma multiplier on the daily force of infection;
    # in the agent engine it is the shape of a per-agent mean-one infectiousness
    # multiplier (so a few individuals drive most transmission).
    overdispersion: Optional[float] = 0.5
    seed: Optional[int] = None

    # Which engine advances the epidemic:
    #   "compartmental" - fast age-structured stochastic SEIR (default)
    #   "agent"         - individual-based model (see outbreak.agents)
    engine: str = "compartmental"
    # Agent engine only: number of simulated individuals. When smaller than the
    # total population the model simulates a representative sample and scales its
    # reported counts up to population scale (keeps large populations tractable).
    n_agents: int = 100_000

    # Class-level tuple (no type annotation), so it's a shared constant rather
    # than a per-instance dataclass field.
    ENGINES = ("compartmental", "agent")

    def validate(self) -> "SimulationConfig":
        if self.duration_days <= 0:
            raise ValueError("duration_days must be > 0")
        _check_positive(self.dt, "dt")
        if self.overdispersion is not None and self.overdispersion <= 0:
            raise ValueError("overdispersion must be > 0 or None")
        if self.engine not in self.ENGINES:
            raise ValueError(
                f"engine must be one of {self.ENGINES}, got {self.engine!r}"
            )
        if self.n_agents <= 0:
            raise ValueError("n_agents must be > 0")
        return self

    @property
    def n_steps(self) -> int:
        # Round up so the final partial step still covers the full duration.
        return int(np.ceil(self.duration_days / self.dt))


# ---------------------------------------------------------------------------
# Top-level scenario
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """The complete description of a simulation scenario."""

    # default_factory=<class> constructs a fresh sub-config per instance (calling
    # the class with no args); these nested dataclasses can't be plain defaults
    # because they are mutable.
    population: PopulationConfig = field(default_factory=PopulationConfig)
    disease: DiseaseConfig = field(default_factory=DiseaseConfig)
    vaccination: VaccinationConfig = field(default_factory=VaccinationConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    healthcare: HealthcareConfig = field(default_factory=HealthcareConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    # Optional explicit contact matrix (n_age x n_age). If None, a default is
    # derived from the number of age groups (see outbreak.contacts).
    contact_matrix: Optional[Sequence[Sequence[float]]] = None

    def validate(self) -> "ScenarioConfig":
        n_age = self.population.n_age
        self.disease.validate(n_age)
        self.vaccination.validate()
        self.interventions.validate()
        self.healthcare.validate()
        self.simulation.validate()
        if self.contact_matrix is not None:
            cm = np.asarray(self.contact_matrix, dtype=float)
            if cm.shape != (n_age, n_age):  # must be a square matrix, one row/col per age group
                raise ValueError(
                    f"contact_matrix must be {n_age}x{n_age}, got {cm.shape}"
                )
            if np.any(cm < 0):
                raise ValueError("contact_matrix entries must be non-negative")
        return self

    # Serialisation -----------------------------------------------------------
    def to_dict(self) -> dict:
        # asdict recursively converts this dataclass and all nested dataclasses
        # into plain (JSON-serialisable) dicts.
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        # Inverse of to_dict: rebuild the nested dataclasses from a plain dict.
        d = dict(d)  # shallow copy so we don't mutate the caller's dict
        # `or {}` falls back to an empty dict when the key is missing or None.
        interventions = d.get("interventions") or {}
        scenario = cls(
            # **d.get("population", {}) unpacks the sub-dict as keyword args;
            # a missing section defaults to {}, so all dataclass defaults apply.
            population=PopulationConfig(**d.get("population", {})),
            disease=DiseaseConfig(**d.get("disease", {})),
            vaccination=VaccinationConfig(**d.get("vaccination", {})),
            interventions=InterventionConfig(
                interventions=interventions.get("interventions", [])
            ),
            healthcare=HealthcareConfig(**d.get("healthcare", {})),
            simulation=SimulationConfig(**d.get("simulation", {})),
            contact_matrix=d.get("contact_matrix"),
        )
        return scenario.validate()  # re-validate after reconstruction


# ---------------------------------------------------------------------------
# Defaults and presets
# ---------------------------------------------------------------------------

# A compact, illustrative four-band age structure. The distribution is roughly
# representative of a developed-country population; the contact matrix lives in
# outbreak.contacts. These are illustrative, not calibrated to a specific
# country or study.
DEFAULT_AGE_LABELS = ("0-17", "18-49", "50-64", "65+")
DEFAULT_AGE_DISTRIBUTION = (0.22, 0.42, 0.19, 0.17)


def covid_like() -> DiseaseConfig:
    """A SARS-CoV-2-like respiratory virus (illustrative parameters)."""
    return DiseaseConfig(
        name="covid_like",
        r0=2.8,
        latent_period=3.0,
        presymptomatic_period=2.0,
        symptomatic_period=7.0,
        asymptomatic_infectious_period=7.0,
        rel_infectiousness_presymptomatic=1.0,
        rel_infectiousness_asymptomatic=0.5,
        asymptomatic_fraction=[0.60, 0.40, 0.30, 0.25],
        hospitalization_rate=[0.001, 0.012, 0.045, 0.110],
        icu_rate=[0.05, 0.10, 0.22, 0.30],
        death_rate=[0.10, 0.20, 0.35, 0.55],
        hospital_stay=8.0,
        icu_stay=10.0,
        waning_immunity_days=270.0,
    )


def influenza_like() -> DiseaseConfig:
    """A seasonal-influenza-like virus (illustrative parameters)."""
    return DiseaseConfig(
        name="influenza_like",
        r0=1.4,
        latent_period=1.5,
        presymptomatic_period=0.5,
        symptomatic_period=4.0,
        asymptomatic_infectious_period=4.0,
        rel_infectiousness_presymptomatic=0.8,
        rel_infectiousness_asymptomatic=0.4,
        asymptomatic_fraction=[0.40, 0.45, 0.40, 0.35],
        hospitalization_rate=[0.005, 0.008, 0.02, 0.06],
        icu_rate=[0.05, 0.08, 0.15, 0.25],
        death_rate=[0.05, 0.08, 0.15, 0.30],
        hospital_stay=5.0,
        icu_stay=7.0,
        waning_immunity_days=200.0,
    )


def measles_like() -> DiseaseConfig:
    """A measles-like, highly transmissible virus (illustrative parameters)."""
    return DiseaseConfig(
        name="measles_like",
        r0=14.0,
        latent_period=8.0,
        presymptomatic_period=3.0,
        symptomatic_period=6.0,
        asymptomatic_infectious_period=6.0,
        rel_infectiousness_presymptomatic=0.6,
        rel_infectiousness_asymptomatic=0.3,
        asymptomatic_fraction=[0.05, 0.05, 0.05, 0.05],
        hospitalization_rate=[0.10, 0.05, 0.06, 0.12],
        icu_rate=[0.10, 0.08, 0.10, 0.20],
        death_rate=[0.03, 0.02, 0.04, 0.10],
        hospital_stay=6.0,
        icu_stay=8.0,
        waning_immunity_days=None,  # lifelong immunity
    )


DISEASE_PRESETS = {
    "covid_like": covid_like,
    "influenza_like": influenza_like,
    "measles_like": measles_like,
}


def preset_scenario(name: str = "covid_like", **population_overrides) -> ScenarioConfig:
    """Build a ready-to-run scenario from a named disease preset."""
    if name not in DISEASE_PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; choose from {sorted(DISEASE_PRESETS)}"
        )
    scenario = ScenarioConfig(
        population=PopulationConfig(**population_overrides),
        disease=DISEASE_PRESETS[name](),
    )
    return scenario.validate()
