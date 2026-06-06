# 🦠 Outbreak — a realistic epidemic simulator

Outbreak is a program that **mimics how a contagious disease spreads through a
population over time**, so you can run "what if?" experiments: What if this virus
is twice as contagious? What if we close schools for six weeks? What if we
vaccinate the elderly first? You set the conditions, press play, and watch the
epidemic unfold day by day.

It tries to be **realistic** — capturing the things that genuinely shape real
outbreaks (people spreading the disease before they feel ill, a few "super
spreaders" causing most infections, hospitals filling up, immunity fading over
time) — while staying fast enough to play with interactively.

> **A note on the numbers.** The built-in settings are *illustrative*: they
> behave like a real respiratory virus but are **not** tuned to any specific real
> disease or country. Outbreak is a great tool for learning, teaching and
> exploring — but don't treat its numbers as predictions about a real outbreak
> unless you've fitted them to real data first.

---

## New here? A 2‑minute plain-language primer

This project comes from epidemiology (the science of how diseases spread), which
has a lot of jargon. Here's everything you need in everyday terms — every term
below appears later in this README.

| Term | In plain words |
|------|----------------|
| **Susceptible / Infected / Recovered** | The model sorts everyone into buckets by their status — never infected yet, currently infected, recovered, and so on — and moves people between buckets each day as the disease progresses. (The classic version of this is called an **"SEIR" model**; the buckets are **"compartments"**.) |
| **R₀ ("R-nought")** | How contagious the disease is: the average number of people *one* sick person infects when **everyone** around them can still catch it. Above 1, the outbreak grows; below 1, it fizzles out. R₀ ≈ 1.3 is a mild flu; ≈ 3 is COVID-like; ≈ 15 is measles. |
| **Rₜ ("R-t")** | The *current* version of R₀ on a given day — after some people have become immune or measures are in place. It tells you whether the epidemic is growing (above 1) or shrinking (below 1) **right now**. |
| **Pre-symptomatic / asymptomatic** | People can spread the disease *before* they feel sick (pre-symptomatic), and some people spread it while *never* feeling sick at all (asymptomatic). Both make a disease much harder to control. |
| **Attack rate** | The fraction of the whole population that gets infected over the course of the outbreak. |
| **Infection fatality ratio (IFR)** | Of everyone who gets infected, the fraction who die. |
| **Waning immunity** | Protection (from infection or vaccine) fades over time, so people can catch the disease again later. |
| **Interventions (NPIs)** | Measures that aren't medicine — masks, social distancing, school closures, lockdowns. ("NPI" = *non-pharmaceutical intervention*.) |
| **Healthcare capacity** | Hospitals and ICUs have a limited number of beds. When demand exceeds supply, more people die — Outbreak models this. |
| **Superspreading** | Spread is uneven: a small number of people (or events) cause most of the infections, rather than everyone spreading equally. |
| **Stochastic vs. deterministic** | *Stochastic* means chance is included, so every run comes out a little different (like real life). *Deterministic* means the smooth, luck-free average. |
| **Ensemble** | Running the chance-based version many times and looking at the **range** of possible outcomes, not just one. |
| **Contact network** | Who you actually meet: the same household, classroom and workplace each day, rather than bumping into random strangers. |

That's the whole vocabulary. The rest of this page uses these words freely, but
always means exactly what's in this table.

---

## What it can do (the highlights)

- **Two ways to run the same disease.** You can pick between two "engines" — two
  different methods that model the *same* disease and give matching results:
  - a **fast** one that tracks people as counts in each bucket (the
    *compartmental* model), great for big populations and quick experiments;
  - a **detailed** one that simulates **every individual person** (the
    *agent-based* model), which captures realistic randomness and lets a few
    people drive most of the spread.

  Everything else — the interface, saving, loading — works the same with either.
- **A realistic disease "life cycle."** A person goes from *susceptible* (can
  catch it) → *exposed* (infected but not yet contagious) → *infectious* (with a
  pre-symptomatic phase, and either a symptomatic or a never-symptomatic path) →
  and then recovers, or is hospitalised, needs intensive care, and possibly dies.
  Importantly, **people are contagious before they show symptoms**, and some
  never show symptoms at all.
- **Age matters.** People mostly mix with others their own age (children at
  school, adults at work), and older people are modelled as more likely to become
  seriously ill — just like reality.
- **You set how contagious it is, and the program does the rest.** You give it a
  target R₀ (the contagiousness dial), and Outbreak automatically works out the
  underlying transmission rate needed to hit that number. As the simulation runs,
  it reports the live Rₜ so you can see whether the outbreak is growing or
  shrinking.
- **Realistic randomness and superspreading.** Real outbreaks are bumpy, not
  smooth — chance plays a big role early on, and a few people cause most of the
  spread. Outbreak includes this (and can also run a smooth, luck-free mode for
  checking the maths).
- **Interventions.** Turn on masks, distancing, school closures or a lockdown for
  a chosen window of days and see the effect.
- **Vaccination.** Roll out a vaccine from a chosen day, oldest-first, up to a
  coverage limit. The vaccine isn't perfect ("leaky"): it separately reduces the
  chance of *catching* the disease, of *getting severely ill*, and of *passing it
  on*.
- **Hospital limits.** Set a number of ICU beds; when the outbreak overwhelms
  them, the death rate rises.
- **Fading immunity & reinfection.** Optionally let immunity wear off so people
  can be infected more than once.
- **Pause, rewind, save and replay.** The simulation moves one step at a time, so
  you can pause it, scrub back to inspect any past day, save a run to a file and
  resume it later *exactly*, and run many random versions at once to see the range
  of outcomes.

---

## Installation

```bash
pip install -r requirements.txt        # numpy, scipy, streamlit, plotly, pandas, pytest
# or, for just the engine:
pip install -e .                       # installs the `outbreak` package
```

For a **reproducible, security-checked** install (recommended if you're deploying
it), use the lockfile, which pins every dependency to an exact, fingerprinted
version:

```bash
pip install --require-hashes -r requirements.lock
```

(`requirements.txt` is the easy everyday list; `requirements.lock` is the strict,
exact one — see [SECURITY.md](SECURITY.md) for why that matters.)

Python 3.9 or newer is required.

---

## The easy way: the interactive app

```bash
streamlit run app/streamlit_app.py
```

This opens a point-and-click dashboard in your browser. You can adjust the
population, the disease, vaccination, interventions and hospital capacity with
sliders; press **Play / Pause / Step / Run to end**; and watch live charts of the
epidemic curve, the live Rₜ, daily new cases, and hospital/ICU usage. You can
rewind to any past day, download the results as a spreadsheet (CSV), save and
reload runs, and run an **ensemble** (many random versions at once) to see the
range of possible outcomes.

---

## The scriptable way: the command line

For quick runs and automation you can use the `outbreak` command — no code, no
browser. It works straight away with `python -m outbreak …`, or as a bare
`outbreak …` command once installed (`pip install -e .`):

```bash
# Run a COVID-like disease in 1,000,000 people for a year and print a summary:
python -m outbreak --preset covid_like --population 1000000 --days 365

# Save the full day-by-day results to a spreadsheet, and the headline numbers to JSON:
python -m outbreak --preset influenza_like --csv results.csv --json summary.json

# Use the detailed engine with households/schools/workplaces switched on:
python -m outbreak --engine agent --n-agents 100000 --network \
    --initial-infected 500 --r0 1.8

# Run 50 random repeats and report the median and range across them:
python -m outbreak --preset covid_like --ensemble 50
```

A single run prints a summary like:

```
=== covid_like · compartmental engine ===
  Population            : 1,000,000
  Total infections      : 780,123 (78.0% of population)
  Deaths                : 2,450 (IFR 0.31%)
  Peak infectious       : 161,000 on day 96
  Peak ICU occupancy    : 690
  Rt fell below 1 on    : day 110
```

Useful options (see `python -m outbreak --help` for the full list):

| Option | What it does |
|--------|--------------|
| `--preset {covid_like,influenza_like,measles_like}` | Disease to start from. |
| `--scenario FILE.json` | Load a scenario (or a saved run) instead of a preset. |
| `--population`, `--initial-infected`, `--r0`, `--days` | Override the basics. |
| `--engine {compartmental,agent}`, `--n-agents`, `--network` | Choose and configure the engine. |
| `--ensemble N` | Run N random repeats; report median + range. |
| `--csv FILE`, `--json FILE`, `--quiet` | Save results / suppress the printout. |
| `--seed N`, `--no-stochastic` | Make runs repeatable, or run the smooth (luck-free) version. |

> Tip: with the **agent** engine, start with a healthy number of cases (e.g.
> `--initial-infected 500`) — a handful of cases can fizzle out purely by chance.

---

## The flexible way: from Python

You can also drive Outbreak in a few lines of code. Start from a built-in disease
and run it to the end:

```python
from outbreak import Simulation, preset_scenario

# A COVID-like disease in a population of 1,000,000 people.
scenario = preset_scenario("covid_like", total_population=1_000_000)
sim = Simulation(scenario)
sim.run_to_end()

summary = sim.summary()
print(f"Attack rate: {summary.attack_rate:.1%}, deaths: {summary.total_deaths:,.0f}")
# e.g. "Attack rate: 78.0%, deaths: 2,450" — i.e. 78% of people were infected.
```

Pause, look at any past day, and resume — or save the whole run to a file:

```python
sim = Simulation(scenario)
sim.run(steps=60)                                   # advance 60 days
sim.run(stop_when=lambda s: s.current_day >= 120)   # run until day 120
record = sim.record_at_day(30)                      # inspect day 30
sim.save("run.json"); Simulation.load("run.json")   # save now, resume later, exactly
```

Run the random version many times and get the typical outcome plus an
uncertainty band (the 5th–95th percentile range across runs):

```python
from outbreak import run_ensemble, aggregate_ensemble
histories = run_ensemble(scenario, n_runs=100, base_seed=2024)  # 100 random runs
band = aggregate_ensemble(histories, "Is", quantiles=(0.05, 0.5, 0.95))
```

See [`examples/demo.py`](examples/demo.py) for worked comparisons — no
intervention vs. lockdown vs. vaccination, and the two engines side by side.

### Choosing an engine (fast vs. detailed)

Both engines model the same disease and agree on the results; you just pick which
method to use. Everything else stays the same.

```python
from outbreak import Simulation, preset_scenario
from outbreak.config import SimulationConfig

scenario = preset_scenario("covid_like", total_population=1_000_000)

# Use the detailed, person-by-person engine. To stay fast on big populations it
# simulates a representative sample of people and scales the results back up.
scenario.simulation = SimulationConfig(engine="agent", n_agents=100_000)
sim = Simulation(scenario.validate())
sim.run_to_end()
```

- **`compartmental`** (the default) — the fast one. Best for large populations
  and for running lots of experiments quickly.
- **`agent`** — the detailed one. It simulates individual people, so it captures
  realistic randomness and genuine superspreading (a few people causing most of
  the spread). `n_agents` controls how many people it simulates — more is more
  accurate but slower. With a very small number of starting cases, an outbreak
  may sometimes fizzle out purely by chance — that's realistic behaviour, not a
  bug.

### Contact networks: households, schools and workplaces

Normally the detailed engine assumes people of similar ages mix at random. You
can make it more lifelike by switching on **contact networks** — the fact that
you see the *same* household, classmates and colleagues every day:

```python
from outbreak.config import NetworkConfig, SimulationConfig

scenario.simulation = SimulationConfig(engine="agent", n_agents=100_000)
scenario.network = NetworkConfig(enabled=True)   # households + schools + workplaces
```

Because the disease now spreads through tight, repeating groups, it **clusters**
the way it does in reality — and once a household or classroom has mostly been
infected, the disease "wastes" contacts there. The visible result is that, for
the *same* contagiousness (R₀), the epidemic peaks **lower and a bit later** than
the random-mixing version — the classic "flatten the curve" effect, emerging on
its own. (Outbreak automatically re-tunes the transmission rate so switching
networks on doesn't change the R₀ you asked for.)

Households are **age-structured** — each one is seeded with an adult, so children
live with adults (realistic across-generation mixing) rather than being grouped
at random. And interventions can **target a single setting**: a school closure can
reduce school contacts only, leaving home and work untouched (see the
`layer` option on interventions). Geography/travel between regions is the next
frontier.

### Realistic stage durations

How long someone stays in each phase (incubating, infectious, in hospital, …)
isn't fixed — it varies person to person. A simple model assumes those durations
are *exponential*, which is mathematically convenient but unrealistic: it implies
some people leave a stage almost instantly and a few linger for an absurdly long
time. For example, with a 14-day incubation it would have ~7% of people turning
infectious within a day and ~10% still incubating after a month.

The detailed (agent) engine instead gives **each person an explicit, realistically
clustered duration** for every stage, controlled by a single "peakedness" knob
(`duration_dispersion`): 1 reproduces the old exponential behaviour, while higher
values (the built-in diseases use 4–6) make durations cluster tightly around their
average — almost nobody leaves in a day, almost nobody lingers for a month. The
average length is unchanged, so **R₀ and the eventual size of the epidemic stay
the same** — but the epidemic's *timing* sharpens (a taller, earlier peak), which
matters for hospital surges and the timing of interventions.

(The fast compartmental engine uses exponential durations intrinsically; this
realism is an agent-engine feature.)

---

## Project layout

```
outbreak/
├── outbreak/                 # the simulation engine (no interface needed to use it)
│   ├── config.py             # all the adjustable settings, with validation + presets
│   ├── contacts.py           # who-mixes-with-whom by age
│   ├── epidemiology.py       # shared maths (incl. the R₀ "auto-tuning")
│   ├── model.py              # the fast (compartmental) engine
│   ├── agents.py             # the detailed (person-by-person) engine
│   ├── network.py            # households / schools / workplaces for the detailed engine
│   ├── interventions.py      # ready-made measures (mask mandate, lockdown, ...)
│   ├── metrics.py            # turns a run into headline numbers (attack rate, peaks, ...)
│   ├── simulation.py         # the play/pause/step/save controller
│   ├── cli.py                # the command-line interface (`outbreak …`)
│   └── __main__.py           # lets `python -m outbreak …` work
├── app/streamlit_app.py      # the interactive browser dashboard
├── examples/demo.py          # a runnable, no-interface demonstration
├── tests/                    # the automated checks that keep it correct
└── .github/workflows/        # CI: run tests + security scans on push & weekly
```

---

## What you can adjust

| Group | What it controls (in plain terms) |
|-------|-----------------------------------|
| **Population** | How many people, their age mix, how many are infected at the start, and how many are already immune. |
| **Disease** | How contagious it is (R₀); how long each phase lasts (and how tightly those durations cluster around their average); how infectious people are before/without symptoms; how susceptible each age is to infection; and the chances — by age — of needing hospital, intensive care, or dying. Whether immunity fades. |
| **Vaccination** | When the rollout starts, how fast, the coverage limit, whether the elderly go first, and how well the vaccine blocks infection / severe illness / onward spread. |
| **Interventions** | When measures (e.g. a lockdown) start and end, and how much they cut transmission. |
| **Healthcare** | Number of hospital and ICU beds, and how much the death rate rises when they overflow. |
| **Environment** | Seasonal swing in transmissibility (amplitude, period, peak day) and an external/spillover infection rate (importations or an animal reservoir). |
| **Network** (detailed engine) | Whether to switch on households/schools/workplaces, their typical sizes, which ages attend school or work, and how much spread happens in each setting. |
| **Simulation** | How many days to run, the random seed (for repeatable runs), whether to include randomness, and which engine to use. |

Three ready-made diseases are included to start from: `covid_like`,
`influenza_like` and `measles_like`.

---

## Model details (technical reference)

This section documents the full model for readers who want the mechanics. The
plain-language sections above still apply — this just spells out what's happening
underneath. Terms like R₀, Rₜ and "compartment" are defined in the plain-language
primer near the top of this page.

### Population and demography

- The population is split into **age groups** (default four: `0-17`, `18-49`,
  `50-64`, `65+`, with population shares `0.22 / 0.42 / 0.19 / 0.17`). Labels and
  shares are configurable; any number of groups is supported.
- Group head-counts are whole numbers (apportioned by the largest-remainder rule
  so they sum exactly to the total population).
- The population is **closed**: there are no births or non-disease deaths, and no
  migration. The only way people leave is by dying of the disease. (Long-run
  demography is a possible future extension.)
- Seeding: `initial_infected` people start infectious at day 0; an optional
  `initial_immune_fraction` start already recovered/immune.

### Disease states (compartments)

Each person is in exactly one state. Susceptible people are additionally flagged
by vaccination status; the infectious/clinical states are tracked per
vaccination stratum (unvaccinated / vaccinated).

| State | Meaning |
|-------|---------|
| **S** | Susceptible, unvaccinated |
| **V** | Susceptible, vaccinated (partial protection) |
| **E** | Exposed — infected but not yet infectious (latent) |
| **Ip** | Pre-symptomatic infectious (will go on to develop symptoms) |
| **Ia** | Asymptomatic infectious (never develops symptoms) |
| **Is** | Symptomatic infectious |
| **H** | Hospitalised (non-ICU) |
| **C** | Critical — in intensive care (ICU) |
| **R** | Recovered (immune, unless immunity wanes) |
| **D** | Dead |

### Natural history (how people move between states)

```
                          ┌──────────────► Ia ──────────────┐
   S/V ──(infection)──► E ─┤                                 ├──► R ──(waning)──► S
                          └► Ip ──► Is ──┬───────────────────┘
                                         └► H ──┬─────────────► R
                                                └► C ──┬──────► R
                                                       └──────► D
```

- **E → Ip or Ia:** on leaving the latent stage, a person is asymptomatic with
  probability `asymptomatic_fraction` (per age), otherwise pre-symptomatic.
- **Ip → Is:** every pre-symptomatic person becomes symptomatic.
- **Ia → R:** asymptomatic people recover.
- **Is → H or R:** a symptomatic person is hospitalised with probability
  `hospitalization_rate` (per age, and reduced for the vaccinated — see
  vaccination), otherwise recovers.
- **H → C or R:** a hospitalised person needs ICU with probability `icu_rate`
  (per age), otherwise recovers.
- **C → D or R:** an ICU patient dies with probability `death_rate` (per age,
  raised under ICU overflow — see healthcare), otherwise recovers.
- **R → S:** if waning is enabled, recovered people return to susceptible at rate
  `1 / waning_immunity_days`, allowing reinfection.

So severity is a **conditional cascade**: overall infection-fatality ratio ≈
`P(symptomatic) × hospitalisation × ICU × death`, each factor age-specific. The
**infection fatality ratio (IFR)** and **attack rate** reported in summaries fall
straight out of this.

### How long each stage lasts (sojourn times)

Mean stage durations are set by `latent_period`, `presymptomatic_period`,
`symptomatic_period`, `asymptomatic_infectious_period`, `hospital_stay` and
`icu_stay` (all in days).

- **Compartmental engine:** transitions use a constant per-step hazard, so stage
  durations are **exponentially** distributed (memoryless).
- **Agent engine:** each individual is given an **explicit gamma-distributed
  duration** when entering a stage, with shape `duration_dispersion` (k). `k = 1`
  reproduces the exponential case; larger `k` clusters durations around the mean
  (coefficient of variation `= 1/√k`). The mean is identical either way, so this
  changes epidemic *timing* (peak height/date) but **not** R₀ or the final size.

### Transmission and the force of infection

- Transmission is **frequency-dependent** and **age-structured**, driven by a
  **contact matrix** `C` where `C[i, j]` is the mean daily number of contacts a
  person in age group `i` has with people in group `j`. The default is a
  POLYMOD-style matrix; it is made *reciprocal* for the population
  (`C[i,j]·Nᵢ = C[j,i]·Nⱼ`) so total contacts are consistent.
- The per-age **force of infection** (instantaneous infection hazard) is
  `λᵢ = sᵢ · ( β_eff · Σⱼ C[i,j] · (weighted infectious prevalence in j) + ε )`,
  where infectious people contribute with phase weights — pre-symptomatic
  `rel_infectiousness_presymptomatic`, asymptomatic
  `rel_infectiousness_asymptomatic`, symptomatic `1.0`.
- `sᵢ` is the **age-specific relative susceptibility** (`susceptibility`,
  default 1): a less-susceptible age acquires proportionally fewer infections
  (children are often less susceptible to infection, not just less severe). It is
  folded into the R₀ calibration so the target R₀ is still hit.
- `ε` is an optional **external/spillover hazard** — see the environment section.
- The per-step probability a susceptible is infected is `1 − exp(−λ·dt)`.
- `β_eff` is the calibrated per-contact transmission rate times any active
  intervention multiplier **and the seasonal multiplier** (see below).

### Setting the contagiousness: R₀ calibration and Rₜ

- You specify a target **R₀**; the engine does **not** ask for the raw
  transmission rate. It builds the **next-generation matrix** `K` (expected
  secondary infections by age, = contact structure × infectiousness-weighted
  expected infectious duration) and sets `β` so the dominant eigenvalue (spectral
  radius) of `K` equals R₀. This is the standard, defensible way to parameterise
  a structured model.
- The age-specific **infectiousness-weighted duration** is
  `p_asymp·rel_a·asym_period + (1−p_asymp)·(rel_p·presym + sympt)`.
- At every step the engine reports the **effective reproduction number**
  `Rₜ = β_eff · spectral_radius( diag(susceptible fraction by age) · K_unit )`,
  which falls below 1 as susceptibles deplete / interventions bite — the signal
  that the epidemic has turned over.

### Superspreading and randomness

- **Compartmental engine:** every flow between states is a **binomial draw**
  (a "chain-binomial" model). Over-dispersion (`overdispersion`, a gamma shape) is
  applied as a population-wide mean-one daily multiplier on the force of
  infection — bursty, but aggregate. A **deterministic** mode (expected values,
  no randomness) is available for validation.
- **Agent engine:** outcomes are realised per individual, so randomness and
  fade-out emerge naturally (it is always stochastic). Over-dispersion instead
  gives **each infected person their own mean-one infectiousness multiplier**, so
  a minority of people drive most transmission — genuine individual-level
  superspreading. Because the multiplier averages to one, R₀ is unchanged.

### Contact networks (agent engine)

Optionally (`NetworkConfig(enabled=True)`) the agent engine adds explicit,
repeated-contact **layers** on top of an age-mixed **community** layer:

- **Households** partition everyone into small groups (sizes drawn from
  `household_size_distribution`); *density-dependent* (you effectively contact all
  housemates).
- **Schools** group school-age agents; **workplaces** group working-age agents
  (configurable age bands and mean sizes); *frequency-dependent* (per-person
  contact rate doesn't grow without bound with group size).

Each layer has a relative `weight`. The engine derives each layer's age-mixing
matrix from the *actual* constructed groups, sums them (weighted) with the
community matrix into one **effective contact matrix**, and calibrates a single
global `β` on that — so a networked run still reproduces the target R₀, and the
within-group stochastic transmission matches it in expectation. The visible
effect is a **lower, later peak** for the same R₀ (clustering depletes local
susceptibles).

Households are **age-structured** by default (each seeded with an adult, so
children co-reside with adults), or random if
`NetworkConfig(age_structured_households=False)`. **Interventions can target a
single layer** via `Intervention(..., layer="school")` (or the `layer=` argument
to the NPI helpers): the agent engine then scales only that layer's transmission,
while the compartmental engine — having no explicit settings — applies only
global (untargeted) interventions.

### Vaccination

A **leaky** vaccine, rolled out from `start_day` at `daily_rate` (fraction of the
population per day) up to a `coverage_cap`, oldest groups first if
`prioritize_elderly`. It has three independent efficacies:

- `ve_susceptibility` — reduces the chance of being infected (scales the force of
  infection on vaccinated susceptibles, who sit in state **V**);
- `ve_severity` — reduces the chance of progressing to hospitalisation (applied
  once, to avoid double-counting along the cascade);
- `ve_transmission` — reduces a vaccinated infected person's onward
  infectiousness.

### Non-pharmaceutical interventions (NPIs)

Each intervention is a time window `[start_day, end_day)` with a
`transmission_reduction`. While active it multiplies the transmission rate by
`(1 − reduction)`; **multiple active interventions stack multiplicatively**.
Helpers (`mask_mandate`, `social_distancing`, `school_closure`, `lockdown`,
`test_trace_isolate`) are just named windows with typical strengths. (In the
current version interventions scale all contact settings uniformly.)

### Healthcare capacity

- `icu_capacity` sets the number of ICU beds. When ICU occupancy exceeds it, the
  death probability for the over-capacity *share* of patients is multiplied by
  `overflow_mortality_multiplier` — so an overwhelmed health system kills more
  people. Occupancy is compared at population scale (the agent engine scales its
  sampled counts up first).
- `hospital_capacity` can be set but currently only ICU overflow affects
  mortality; it does not yet change dynamics.

### Transmission environment: seasonality and external spillover

Two optional, off-by-default effects (in `EnvironmentConfig`) shared by both
engines:

- **Seasonality.** Transmissibility is multiplied by
  `1 + seasonal_amplitude · cos(2π·(day − seasonal_peak_day) / seasonal_period_days)`,
  so the effective reproduction number swings above and below its calibrated
  value through the year — the recurring winter/summer pattern of real
  respiratory diseases. Because the cosine averages to zero over a period, the
  **target R₀ is the annual average** and calibration is unchanged.
- **External / spillover force of infection.** A constant background hazard
  `external_infection_rate` (per susceptible per day, also modulated by season)
  of being infected from *outside* the modelled population — importations from
  elsewhere, or a zoonotic/environmental reservoir (e.g. rodent-borne spillover).
  This lets outbreaks **start with no initial cases**, **re-ignite after
  fade-out**, or **persist even when person-to-person spread alone (R₀ < 1) would
  die out** — the missing ingredient for reservoir-driven diseases.

### Waning immunity and reinfection

If `waning_immunity_days` is set, recovered people return to susceptible at rate
`1 / waning_immunity_days`. With waning, the cumulative **attack rate can exceed
100%** (people are counted each time they're infected) and the epidemic can
settle into recurring waves rather than burning out once.

### The two engines and population scaling

- **Compartmental** tracks real-valued counts per age group × vaccination
  stratum; it is fast and ideal for large populations, sweeps and big ensembles.
- **Agent** tracks every individual as a row in NumPy arrays (age, state,
  vaccination flag, infectiousness, stage timer, network groups). To stay fast it
  can simulate a representative sample of `n_agents` people and scale reported
  counts by `total_population / n_agents`; capacities are compared at population
  scale so overflow behaves correctly.
- They describe the **same disease** with the **same R₀ calibration**, and agree
  on the final size in the mean-field limit (the agent model is a stochastic
  realisation of the compartmental one).

### Time stepping and outputs

- The simulation advances in discrete steps of `dt` days (default 1). All
  transitions in a step are computed from the start-of-step state and applied
  together, so update order doesn't matter.
- Each step yields a record of: every compartment total; daily incidence (new
  infections, symptomatic onsets, hospitalisations, ICU admissions, deaths);
  Rₜ; the effective transmission rate; ICU overflow; and per-age infectious and
  death counts.
- A run summary derives the **attack rate**, **IFR**, peak infectious /
  hospital / ICU occupancy and their timing, peak daily incidence, peak Rₜ, the
  day Rₜ first drops below 1, and the day the epidemic ends. Ensembles report
  pointwise median and quantile bands across runs.

### Defaults are illustrative

The presets and default contact/severity parameters are qualitatively realistic
but **not fitted to any specific real disease or country**. Calibrate to data
before drawing real-world conclusions.

---

## How we know it behaves correctly

Outbreak ships with an automated test suite (run it with `pytest`) that checks,
among other things, that:

- the contagiousness dial works: when everyone is susceptible, the simulated Rₜ
  matches the R₀ you asked for;
- nobody is ever lost or double-counted, and no bucket ever goes negative;
- outbreaks behave sensibly: they grow, peak, and burn out; a more contagious
  disease infects more people; and a disease with R₀ below 1 never takes off;
- the controls have the expected effect: interventions slow spread, vaccination
  reduces infections, and overwhelmed ICUs lead to more deaths;
- runs are repeatable from a seed, and saving and reloading a run continues it
  exactly;
- **the two engines agree** — the detailed engine reproduces the fast engine's
  results to within a few percent, confirming they describe the same disease;
- **contact networks keep the maths honest** — switching them on still hits the
  R₀ you asked for, and produces the lower, later "flatten the curve" peak;
- **realistic stage durations preserve R₀ and final size** — making per-stage
  durations peaked rather than exponential leaves the R₀ and eventual attack rate
  unchanged, while sharpening the epidemic peak;
- **the transmission-environment features behave** — age-specific susceptibility
  still hits the target R₀ while sparing less-susceptible ages; seasonality swings
  transmissibility on a yearly cosine; and an external/spillover force starts an
  outbreak from zero initial cases and sustains a sub-critical (R₀ < 1) disease.

As a maths check, the smooth (luck-free) mode lands on the textbook answer for
the final size of an epidemic. For example, for R₀ = 2.5 the textbook says 89.3%
of people are eventually infected, and the model gives 89.6% (using a small time
step). Use a smaller time step when you want precise numbers; one-day steps are
fine for interactive exploration.

```bash
pytest                # run the full suite of checks
```

---

## Honest limitations

No model is reality. The main simplifications to keep in mind:

- **Who-meets-whom.** The fast engine assumes people mix at random within their
  age group. The detailed engine can do the same, or add households (with adults
  and children), schools and workplaces — and can target measures at individual
  settings. It still doesn't model **geography or travel between regions**, which
  is the obvious next step.
- **Superspreading is modelled in two slightly different ways** by the two engines
  (a population-wide "bumpiness" factor in the fast engine; genuine
  person-by-person variation in the detailed one).
- **Sampling.** When the detailed engine simulates a sample of people rather than
  everyone (for speed), very rare events are smoothed out a little; simulate more
  people for finer detail.
- **The default numbers are illustrative**, not fitted to any real disease. Tune
  them to real data before drawing real-world conclusions.

---

## Security & running it for others

Outbreak is designed to run **on your own computer, for one person at a time**,
and has no login system by default.

- **Files you load are treated as untrusted.** When you load a saved run, the app
  checks the file's size before opening it and validates every value inside, so a
  broken or tampered file produces a friendly error instead of a crash or a frozen
  computer. There is no hidden code execution or network access anywhere in the
  project.
- **Dependencies can be locked down.** `requirements.lock` pins every piece of
  supporting software to an exact, fingerprinted version.
- **Security is scanned automatically and periodically.** Two checks run on every
  change *and* weekly on a schedule (`.github/workflows/security.yml`):
  [`bandit`](https://bandit.readthedocs.io/) static-analyses the code for insecure
  patterns, and [`pip-audit`](https://pypi.org/project/pip-audit/) checks the
  pinned dependencies against known-vulnerability databases. Both are currently
  clean. Run them locally with `pip install -e ".[dev]"` then
  `bandit -r outbreak app` and `pip-audit -r requirements.lock --no-deps`.
- **Before putting it on the public internet**, add a login and limits on how much
  each visitor can run (big simulations use real computing power).

There's a full, jargon-free explanation of all of this in
[SECURITY.md](SECURITY.md).

## License

MIT.
