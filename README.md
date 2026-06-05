# 🦠 Outbreak — a realistic epidemic simulator

Outbreak is an **age-structured, stochastic epidemic simulator** with an
interactive UI and a choice of two interchangeable engines — a fast
**compartmental SEIR** model and an **agent-based** (individual-level) model. It
is built to reflect modern epidemiological understanding of how respiratory
viruses spread — pre-symptomatic and asymptomatic transmission, superspreading,
age-dependent severity, waning immunity, vaccination and non-pharmaceutical
interventions — while staying fast enough to explore interactively.

> The default parameters are **illustrative** (qualitatively realistic, not
> calibrated to a specific pathogen or country). They are a sound starting point
> for exploration and teaching; calibrate them to data before drawing
> real-world conclusions.

---

## Highlights

- **Two interchangeable engines.** A fast **compartmental** SEIR model (tracks
  counts per age group) and an **agent-based** model (simulates individuals with
  per-person superspreading and demographic stochasticity). Both describe the
  same disease, share the same R₀ calibration, and run behind the identical
  `Simulation` interface — so the UI, save/load and ensembles work with either.
- **Realistic natural history.** Susceptible → Vaccinated → Exposed (latent) →
  pre-symptomatic / asymptomatic / symptomatic infectious → hospitalised → ICU →
  recovered / dead. People are infectious *before* symptoms, and a fraction never
  develop symptoms at all.
- **Age structure & contact matrices.** Transmission is driven by an age contact
  matrix (POLYMOD-style), with age-dependent susceptibility and severity.
- **R₀-based calibration.** You specify a target basic reproduction number R₀;
  the engine calibrates the per-contact transmission rate from the dominant
  eigenvalue of the **next-generation matrix**. It reports the model-implied
  **effective reproduction number Rₜ** at every step.
- **Stochasticity & superspreading.** Every transition is a binomial draw
  (chain-binomial model); an optional over-dispersion term reproduces bursty,
  superspreading-driven dynamics. A deterministic mode is available for
  validation.
- **Interventions.** Time-windowed transmission reductions (masking, distancing,
  school closure, lockdown, test-trace-isolate).
- **Vaccination.** Leaky vaccine with separate efficacy against infection,
  severe disease and onward transmission, an age-prioritised rollout and a
  coverage cap.
- **Healthcare capacity.** ICU overflow raises mortality.
- **Waning immunity & reinfection.**
- **Pausable, inspectable engine.** The simulation advances one step at a time,
  so you can pause, scrub the timeline, save/resume a run exactly (RNG state
  included), and run reproducible stochastic ensembles with uncertainty bands.

---

## Installation

```bash
pip install -r requirements.txt        # numpy, scipy, streamlit, plotly, pandas, pytest
# or, for just the engine:
pip install -e .                       # installs the `outbreak` package
```

For a **reproducible, hash-verified** environment (recommended for deployments),
install from the lockfile instead:

```bash
pip install --require-hashes -r requirements.lock
```

`requirements.txt` lists loose minimum versions for everyday use;
`requirements.lock` pins every direct and transitive dependency to one exact,
fingerprinted version (see [SECURITY.md](SECURITY.md)).

Python 3.9+ is required.

---

## Run the interactive UI

```bash
streamlit run app/streamlit_app.py
```

The UI lets you adjust population, disease biology, vaccination, interventions
and healthcare capacity; Build / Play / Pause / Step / Run-to-end; watch the
epidemic curve, Rₜ, daily incidence and hospital/ICU occupancy update live;
scrub back to inspect any past day; export the time series to CSV; save and
reload runs; and run a stochastic **ensemble** to see the range of outcomes.

---

## Use the engine from Python

```python
from outbreak import Simulation, preset_scenario

scenario = preset_scenario("covid_like", total_population=1_000_000)
sim = Simulation(scenario)
sim.run_to_end()

summary = sim.summary()
print(f"Attack rate: {summary.attack_rate:.1%}, deaths: {summary.total_deaths:,.0f}")
```

Pause, inspect and resume:

```python
sim = Simulation(scenario)
sim.run(steps=60)                       # advance 60 days
sim.run(stop_when=lambda s: s.current_day >= 120)   # run until a condition
record = sim.record_at_day(30)          # inspect any past day
sim.save("run.json"); Simulation.load("run.json")   # exact save / resume
```

Reproducible stochastic ensembles with uncertainty bands:

```python
from outbreak import run_ensemble, aggregate_ensemble
histories = run_ensemble(scenario, n_runs=100, base_seed=2024)
band = aggregate_ensemble(histories, "Is", quantiles=(0.05, 0.5, 0.95))
```

See [`examples/demo.py`](examples/demo.py) for unmitigated vs. lockdown vs.
vaccination comparisons, plus a head-to-head of the two engines.

### Choosing an engine

The engine is selected per scenario; everything else is identical.

```python
from outbreak import Simulation, preset_scenario
from outbreak.config import SimulationConfig

scenario = preset_scenario("covid_like", total_population=1_000_000)

# Agent-based: simulate individuals. Below the population size the model
# simulates a representative sample and scales results up to population scale.
scenario.simulation = SimulationConfig(engine="agent", n_agents=100_000)
sim = Simulation(scenario.validate())
sim.run_to_end()
```

- **`compartmental`** (default) — fastest; ideal for large populations,
  parameter sweeps and big ensembles.
- **`agent`** — individual-level. Adds genuine per-person superspreading (a few
  agents drive most transmission) and demographic stochasticity (small outbreaks
  can fade out by chance). Use `n_agents` to trade fidelity for speed; very small
  seeds may stochastically go extinct — that is the model being honest, not a bug.

---

## Project layout

```
outbreak/
├── outbreak/                 # the engine (no UI dependency)
│   ├── config.py             # validated parameter schema + disease presets
│   ├── contacts.py           # age structure + contact matrices
│   ├── epidemiology.py       # shared math: NGM β-calibration, rates, ICU overflow
│   ├── model.py              # compartmental stochastic SEIR engine + Rt
│   ├── agents.py             # agent-based (individual-level) engine
│   ├── interventions.py      # named NPI builders
│   ├── metrics.py            # Rt, attack rate, peaks, ensemble aggregation
│   └── simulation.py         # run/pause/step/reset state machine + engine dispatch
├── app/streamlit_app.py      # interactive UI
├── examples/demo.py          # headless demonstration
└── tests/                    # validation + behaviour suite
```

---

## Parameters

| Group | Key parameters |
|-------|----------------|
| **Population** | total size, age distribution, initial infections, initial immune fraction |
| **Disease** | R₀, latent / pre-symptomatic / symptomatic / asymptomatic durations, relative infectiousness, asymptomatic fraction, age-specific hospitalisation / ICU / death rates, waning immunity |
| **Vaccination** | start day, daily rate, coverage cap, age prioritisation, efficacy vs. infection / severity / transmission |
| **Interventions** | per-window start/end and transmission reduction |
| **Healthcare** | hospital & ICU capacity, overflow mortality multiplier |
| **Simulation** | duration, time step `dt`, deterministic/stochastic, over-dispersion, seed, **engine** (`compartmental`/`agent`), **n_agents** |

Built-in disease presets: `covid_like`, `influenza_like`, `measles_like`.

---

## Validation

The test suite (`pytest`) checks, among other things, that:

- model-implied Rₜ equals the target R₀ when the population is fully susceptible
  (NGM calibration is correct);
- population is conserved exactly and no compartment goes negative;
- the epidemic grows, peaks and burns out; higher R₀ ⇒ higher attack rate;
  a sub-critical R₀ < 1 fails to take off;
- interventions reduce Rₜ; vaccination reduces the attack rate; ICU overflow
  increases deaths;
- runs are reproducible from a seed and survive a save/load round-trip;
- **both engines agree**: in the mean-field limit the agent-based model
  reproduces the compartmental attack rate and peak to within a few percent,
  confirming it is a stochastic realisation of the same disease.

The deterministic mode converges to the **analytic SIR final-size relation**
(`z = 1 − e^{−R₀ z}`) as the time step `dt` shrinks — e.g. for R₀ = 2.5 the
analytic final size is 0.893, and the model gives 0.896 at `dt = 0.1`. Use a
smaller `dt` when you need quantitative accuracy; `dt = 1 day` is fine for
interactive exploration.

```bash
pytest                # run the full suite
```

---

## Modelling notes & limitations

- **Mixing.** Both engines currently use age-structured **mean-field** mixing
  (an age contact matrix), not explicit individual contact networks, households
  or geography. The agent-based engine adds individual heterogeneity and
  demographic stochasticity on top of that mixing; an explicit contact
  **network** (households, schools, workplaces) is the natural next iteration and
  the agent representation is built to accommodate it.
- **Superspreading.** The compartmental engine models over-dispersion
  phenomenologically as a daily Gamma multiplier on the force of infection; the
  agent engine instead assigns each infected individual its own mean-one
  infectiousness, so superspreading is realised at the individual level.
- **Agent scaling.** When `n_agents` is below the population size the agent
  engine simulates a representative sample and scales reported counts up. This
  keeps large populations tractable but coarsens small-number effects; increase
  `n_agents` for fine-grained tail behaviour.
- Default contact matrices and severity parameters are illustrative. Calibrate to
  empirical data for any quantitative use.

---

## Security & deployment

Outbreak is built to run **locally, for a single user**, and has no
authentication by default.

- **Untrusted snapshots are handled safely.** Saved/uploaded run snapshots are
  treated as untrusted input: the UI caps the upload size *before* parsing,
  every restored field is shape- and range-checked (`outbreak/epidemiology.py`
  `restore_array`, plus each engine's `set_state`), and malformed files surface a
  friendly error instead of a crash. There is no `eval`/`exec`, `pickle`,
  `subprocess` or network I/O anywhere in the codebase.
- **Dependencies are lockable.** `requirements.lock` pins every dependency to an
  exact, hash-verified version for reproducible, tamper-evident installs.
- **Before exposing it publicly**, front the app with access control and
  per-request resource limits (large populations / ensembles are compute-heavy).

A full, jargon-free explanation of these measures is in
[SECURITY.md](SECURITY.md).

## License

MIT.
