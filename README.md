# 🦠 Outbreak — a realistic epidemic simulator

Outbreak is an **age-structured, stochastic SEIR-type epidemic simulator** with an
interactive UI. It is built to reflect modern epidemiological understanding of
how respiratory viruses spread — pre-symptomatic and asymptomatic transmission,
superspreading, age-dependent severity, waning immunity, vaccination and
non-pharmaceutical interventions — while staying fast enough to explore
interactively.

> The default parameters are **illustrative** (qualitatively realistic, not
> calibrated to a specific pathogen or country). They are a sound starting point
> for exploration and teaching; calibrate them to data before drawing
> real-world conclusions.

---

## Highlights

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
vaccination comparisons.

---

## Project layout

```
outbreak/
├── outbreak/                 # the engine (no UI dependency)
│   ├── config.py             # validated parameter schema + disease presets
│   ├── contacts.py           # age structure + contact matrices
│   ├── model.py              # stochastic SEIR engine + NGM β-calibration + Rt
│   ├── interventions.py      # named NPI builders
│   ├── metrics.py            # Rt, attack rate, peaks, ensemble aggregation
│   └── simulation.py         # run/pause/step/reset state machine + (de)serialisation
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
| **Simulation** | duration, time step `dt`, deterministic/stochastic, over-dispersion, seed |

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
- runs are reproducible from a seed and survive a save/load round-trip.

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

- This is a **compartmental** model with age strata. It captures population-level
  dynamics well but does not model explicit individual contact networks,
  households or geography. A natural next step is an **agent-based** engine behind
  the same `Simulation` interface (the engine and UI are deliberately decoupled).
- Over-dispersion is modelled phenomenologically as a daily Gamma multiplier on
  the force of infection; it reproduces aggregate variability but not
  individual-level superspreading events.
- Default contact matrices and severity parameters are illustrative. Calibrate to
  empirical data for any quantitative use.

## License

MIT.
