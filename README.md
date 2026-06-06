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

This first version forms households by grouping people at random and applies any
interventions evenly across all settings; modelling realistic family makeup and
closing *only* schools (for example) are natural next steps.

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
│   └── simulation.py         # the play/pause/step/save controller
├── app/streamlit_app.py      # the interactive browser dashboard
├── examples/demo.py          # a runnable, no-interface demonstration
└── tests/                    # the automated checks that keep it correct
```

---

## What you can adjust

| Group | What it controls (in plain terms) |
|-------|-----------------------------------|
| **Population** | How many people, their age mix, how many are infected at the start, and how many are already immune. |
| **Disease** | How contagious it is (R₀); how long each phase lasts; how infectious people are before/without symptoms; and the chances — by age — of needing hospital, intensive care, or dying. Whether immunity fades. |
| **Vaccination** | When the rollout starts, how fast, the coverage limit, whether the elderly go first, and how well the vaccine blocks infection / severe illness / onward spread. |
| **Interventions** | When measures (e.g. a lockdown) start and end, and how much they cut transmission. |
| **Healthcare** | Number of hospital and ICU beds, and how much the death rate rises when they overflow. |
| **Network** (detailed engine) | Whether to switch on households/schools/workplaces, their typical sizes, which ages attend school or work, and how much spread happens in each setting. |
| **Simulation** | How many days to run, the random seed (for repeatable runs), whether to include randomness, and which engine to use. |

Three ready-made diseases are included to start from: `covid_like`,
`influenza_like` and `measles_like`.

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
  R₀ you asked for, and produces the lower, later "flatten the curve" peak.

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
  age group. The detailed engine can do the same, or add households/schools/
  workplaces — but in this first version households are formed by random grouping
  (not realistic families), and interventions apply evenly everywhere. Realistic
  family makeup, closing individual settings, and geography/travel between regions
  are the obvious next steps.
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
- **Before putting it on the public internet**, add a login and limits on how much
  each visitor can run (big simulations use real computing power).

There's a full, jargon-free explanation of all of this in
[SECURITY.md](SECURITY.md).

## License

MIT.
