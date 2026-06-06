"""Interactive Streamlit UI for the Outbreak epidemic simulator.

Run with::

    streamlit run app/streamlit_app.py

Features
--------
* Adjustable population, disease biology, vaccination, NPIs and healthcare
  capacity, organised into collapsible sections.
* Build / Play / Pause / Step / Run-to-end controls backed by the engine's
  step-by-step state machine.
* Live epidemic curve, effective reproduction number, daily incidence and
  hospital/ICU occupancy vs. capacity.
* A timeline slider to scrub back through history and inspect any past day.
* Headline statistics, age breakdown, CSV export and scenario save/load.
* An ensemble mode that runs many stochastic realisations and shows the median
  with an uncertainty band.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from outbreak.config import (
    DiseaseConfig,
    EnvironmentConfig,
    HealthcareConfig,
    Intervention,
    InterventionConfig,
    NetworkConfig,
    PopulationConfig,
    ReportingConfig,
    ScenarioConfig,
    SimulationConfig,
    VaccinationConfig,
    DISEASE_PRESETS,
)
from outbreak.metrics import aggregate_ensemble
from outbreak.simulation import RunState, Simulation, run_ensemble

# Streamlit reruns this whole script top-to-bottom on every interaction (widget
# change, button click, st.rerun()). set_page_config must be the first Streamlit
# call in the script, so it lives here at module top level.
st.set_page_config(page_title="Outbreak — Epidemic Simulator", layout="wide")

# Upper bound on an uploaded snapshot, enforced before parsing. A legitimate
# snapshot stores only configuration plus per-agent state arrays, so tens of MB
# is generous; the cap stops a hostile file from exhausting memory in json.load.
MAX_SNAPSHOT_BYTES = 50_000_000

# Fixed colour per SEIR compartment (S=susceptible, V=vaccinated, E=exposed,
# Ip/Ia/Is=infectious variants, H=hospital, C=ICU, R=recovered, D=dead) so the
# same compartment is drawn the same colour across every chart below.
COMPARTMENT_COLORS = {
    "S": "#4c78a8", "V": "#72b7b2", "E": "#f5c518", "Ip": "#ff9d4e",
    "Ia": "#ffb86b", "Is": "#e45756", "H": "#b279a2", "C": "#8b1a1a",
    "R": "#54a24b", "D": "#333333",
}


# --------------------------------------------------------------------------- #
# Scenario construction from the sidebar
# --------------------------------------------------------------------------- #
def build_scenario() -> ScenarioConfig:
    # Reads every sidebar widget and assembles a ScenarioConfig. Because the
    # script reruns top-to-bottom, each widget call below both *renders* the
    # control and *returns its current value* in one go.
    st.sidebar.header("⚙️ Scenario parameters")

    # selectbox returns the chosen key; we look the preset factory up and call
    # it to get a DiseaseConfig whose values seed the sliders' defaults below.
    preset_name = st.sidebar.selectbox(
        "Disease preset", list(DISEASE_PRESETS.keys()), index=0,
        help="A realistic starting point you can then fine-tune below.",
    )
    base_disease = DISEASE_PRESETS[preset_name]()

    # Each `with st.sidebar.expander(...)` is a collapsible section; the widgets
    # created inside it render within that section. number_input/slider/checkbox
    # each return the user's current value for this rerun.
    with st.sidebar.expander("Population", expanded=True):
        total_population = st.number_input(
            "Total population", min_value=1_000, max_value=100_000_000,
            value=1_000_000, step=10_000,
        )
        initial_infected = st.number_input(
            "Initial infections", min_value=1, max_value=1_000_000, value=20, step=10,
        )
        initial_immune = st.slider(
            "Initial immune fraction", 0.0, 0.95, 0.0, 0.05,
            help="Population already immune at day 0 (prior infection / vaccination).",
        )

    with st.sidebar.expander("Disease biology", expanded=True):
        r0 = st.slider("Basic reproduction number R₀", 0.5, 18.0,
                       float(base_disease.r0), 0.1)
        latent = st.slider("Latent period (days)", 0.5, 14.0,
                           float(base_disease.latent_period), 0.5)
        presym = st.slider("Pre-symptomatic infectious period (days)", 0.0, 7.0,
                           float(base_disease.presymptomatic_period), 0.5)
        sympt = st.slider("Symptomatic infectious period (days)", 1.0, 21.0,
                          float(base_disease.symptomatic_period), 0.5)
        asym_period = st.slider("Asymptomatic infectious period (days)", 1.0, 21.0,
                                float(base_disease.asymptomatic_infectious_period), 0.5)
        rel_asym = st.slider("Relative infectiousness (asymptomatic)", 0.0, 1.0,
                             float(base_disease.rel_infectiousness_asymptomatic), 0.05)
        asym_frac = st.slider("Asymptomatic fraction (mean)", 0.0, 0.95,
                              float(np.mean(base_disease.asymptomatic_fraction_arr(4))), 0.05)
        duration_dispersion = st.slider(
            "Stage-duration peakedness (shape)", 1.0, 10.0,
            float(base_disease.duration_dispersion), 0.5,
            help="How tightly each stage's duration clusters around its mean. "
                 "1 = exponential (very variable); higher = realistic, peaked "
                 "durations. Used by the agent engine.",
        )
        waning_on = st.checkbox("Waning immunity",
                                value=base_disease.waning_immunity_days is not None)
        # `disabled=not waning_on` greys out the slider when the checkbox is off;
        # it still returns a value, but we ignore it later when waning is off.
        waning_days = st.slider("Immunity duration (days)", 30, 1000,
                                int(base_disease.waning_immunity_days or 270), 30,
                                disabled=not waning_on)

    with st.sidebar.expander("Severity (mean across ages)"):
        hosp = st.slider("Hospitalisation rate (of symptomatic)", 0.0, 0.5,
                         float(np.mean(base_disease.hospitalization_rate_arr(4))), 0.005)
        icu = st.slider("ICU rate (of hospitalised)", 0.0, 0.8,
                        float(np.mean(base_disease.icu_rate_arr(4))), 0.01)
        death = st.slider("Death rate (of ICU)", 0.0, 0.9,
                          float(np.mean(base_disease.death_rate_arr(4))), 0.01)

    with st.sidebar.expander("Vaccination"):
        vacc_enabled = st.checkbox("Enable vaccination", value=False)
        vacc_start = st.number_input("Start day", 0, 1000, 30, 5, disabled=not vacc_enabled)
        # Slider returns a percentage (0-3); divide by 100 to store a fraction.
        vacc_rate = st.slider("Daily coverage (% of pop/day)", 0.0, 3.0, 0.5, 0.1,
                              disabled=not vacc_enabled) / 100.0
        vacc_cap = st.slider("Coverage cap", 0.0, 1.0, 0.7, 0.05, disabled=not vacc_enabled)
        ve_sus = st.slider("Efficacy vs. infection", 0.0, 1.0, 0.6, 0.05, disabled=not vacc_enabled)
        ve_sev = st.slider("Efficacy vs. severe disease", 0.0, 1.0, 0.8, 0.05, disabled=not vacc_enabled)

    with st.sidebar.expander("Interventions (NPIs)"):
        npi_enabled = st.checkbox("Enable an intervention window", value=False)
        npi_start = st.number_input("NPI start day", 0, 1000, 30, 5, disabled=not npi_enabled)
        npi_end = st.number_input("NPI end day", 1, 2000, 120, 5, disabled=not npi_enabled)
        npi_reduction = st.slider("Transmission reduction", 0.0, 0.95, 0.5, 0.05,
                                  disabled=not npi_enabled)

    with st.sidebar.expander("Environment (seasonality & spillover)"):
        season_on = st.checkbox("Seasonal transmission", value=False)
        season_amp = st.slider("Seasonal swing (amplitude)", 0.0, 0.9, 0.3, 0.05,
                               disabled=not season_on,
                               help="Fraction by which transmissibility swings up/down "
                                    "over the year. R₀ is the annual average.")
        season_peak = st.number_input("Peak transmissibility day", 0, 365, 0, 5,
                                      disabled=not season_on)
        spillover_on = st.checkbox("External / spillover infections", value=False,
                                   help="A background infection hazard from outside the "
                                        "population (imports, or an animal reservoir). Can "
                                        "start or sustain outbreaks on its own.")
        # Slider is in cases per 100,000 susceptibles per day; convert to a rate.
        spillover_per_100k = st.slider("Spillover rate (per 100k/day)", 0.0, 50.0, 5.0, 0.5,
                                       disabled=not spillover_on)
        child_susc = st.slider("Children's relative susceptibility (0-17)", 0.1, 1.5, 1.0, 0.05,
                               help="How susceptible the youngest age group is to infection, "
                                    "relative to adults (1.0 = same).")

    with st.sidebar.expander("Surveillance (reported cases)"):
        st.caption("How true infections appear in case data: under-reporting and delay.")
        ascertainment = st.slider("Ascertainment (% of symptomatic reported)", 1, 100, 100, 1,
                                  help="Fraction of symptomatic cases that get detected.") / 100.0
        reporting_delay = st.number_input("Reporting delay (days)", 0, 30, 0, 1)

    with st.sidebar.expander("Healthcare capacity"):
        cap_enabled = st.checkbox("Limit ICU capacity", value=False)
        icu_capacity = st.number_input("ICU beds", 0, 1_000_000, 500, 50,
                                       disabled=not cap_enabled)
        overflow_mult = st.slider("Overflow mortality multiplier", 1.0, 5.0, 2.0, 0.5,
                                  disabled=not cap_enabled)

    with st.sidebar.expander("Simulation controls", expanded=True):
        duration = st.number_input("Duration (days)", 30, 2000, 365, 5)
        engine_label = st.radio(
            "Engine",
            ["Compartmental (fast)", "Agent-based (individuals)"],
            help=(
                "Compartmental tracks counts per age group (fast). Agent-based "
                "simulates individuals with per-person superspreading and "
                "demographic stochasticity (slower, richer)."
            ),
        )
        # Map the human-readable radio label to the engine key the config wants.
        engine = "agent" if engine_label.startswith("Agent") else "compartmental"
        n_agents = 100_000
        # The agent count widget is only shown when the agent engine is selected;
        # otherwise n_agents keeps its default (unused by the compartmental model).
        if engine == "agent":
            n_agents = st.number_input(
                "Number of agents", 5_000, 1_000_000, 100_000, 5_000,
                help=("Individuals simulated. Below the population size the model "
                      "simulates a representative sample and scales results up. "
                      "Very small seeds may stochastically fade out."),
            )
        stochastic = st.checkbox("Stochastic (random) dynamics", value=True,
                                 disabled=(engine == "agent"),
                                 help="The agent engine is always stochastic.")
        # Force stochastic on for the agent engine regardless of the (disabled)
        # checkbox's returned value.
        if engine == "agent":
            stochastic = True
        overdispersion = st.slider("Superspreading (lower = burstier)", 0.05, 5.0, 0.5, 0.05,
                                   disabled=not stochastic)
        seed = st.number_input("Random seed", 0, 1_000_000, 42, 1, disabled=not stochastic)

    # Contact network (agent engine only). Households/schools/workplaces add
    # repeated-contact structure on top of community mixing; the relative weights
    # control where transmission happens (the engine recalibrates beta to R0).
    with st.sidebar.expander("Contact network (agent engine)"):
        net_available = engine == "agent"
        if not net_available:
            st.caption("Switch to the agent engine to enable contact networks.")
        net_enabled = st.checkbox("Enable households / schools / workplaces",
                                  value=False, disabled=not net_available)
        hh_w = st.slider("Household weight", 0.0, 3.0, 1.0, 0.1, disabled=not net_enabled)
        sch_w = st.slider("School weight", 0.0, 3.0, 0.6, 0.1, disabled=not net_enabled)
        wrk_w = st.slider("Workplace weight", 0.0, 3.0, 0.6, 0.1, disabled=not net_enabled)
        com_w = st.slider("Community weight", 0.0, 3.0, 0.5, 0.1, disabled=not net_enabled)

    # Only add an intervention if it's enabled and the window is non-empty.
    interventions = []
    if npi_enabled and npi_end > npi_start:
        interventions.append(Intervention("intervention", int(npi_start), int(npi_end), npi_reduction))

    # Bundle all widget values into nested config dataclasses. .validate() raises
    # on inconsistent parameters and returns the (validated) ScenarioConfig.
    return ScenarioConfig(
        population=PopulationConfig(
            total_population=int(total_population),
            initial_infected=int(initial_infected),
            initial_immune_fraction=float(initial_immune),
        ),
        disease=DiseaseConfig(
            name=preset_name, r0=r0, latent_period=latent,
            presymptomatic_period=presym, symptomatic_period=sympt,
            asymptomatic_infectious_period=asym_period,
            rel_infectiousness_asymptomatic=rel_asym,
            asymptomatic_fraction=asym_frac,
            hospitalization_rate=hosp, icu_rate=icu, death_rate=death,
            waning_immunity_days=(waning_days if waning_on else None),
            duration_dispersion=duration_dispersion,
            # Youngest band gets the chosen relative susceptibility; others = 1.0.
            susceptibility=([child_susc] + [1.0] * 3 if child_susc != 1.0 else 1.0),
        ),
        vaccination=VaccinationConfig(
            enabled=vacc_enabled, start_day=int(vacc_start), daily_rate=vacc_rate,
            coverage_cap=vacc_cap, ve_susceptibility=ve_sus, ve_severity=ve_sev,
        ),
        interventions=InterventionConfig(interventions),
        healthcare=HealthcareConfig(
            icu_capacity=(int(icu_capacity) if cap_enabled else None),
            overflow_mortality_multiplier=overflow_mult,
        ),
        environment=EnvironmentConfig(
            seasonal_amplitude=(season_amp if season_on else 0.0),
            seasonal_peak_day=int(season_peak),
            # cases per 100k/day -> per-susceptible daily hazard.
            external_infection_rate=(spillover_per_100k / 100_000.0 if spillover_on else 0.0),
        ),
        reporting=ReportingConfig(
            ascertainment=ascertainment, reporting_delay_days=float(reporting_delay),
        ),
        network=NetworkConfig(
            enabled=bool(net_enabled), household_weight=hh_w, school_weight=sch_w,
            workplace_weight=wrk_w, community_weight=com_w,
        ),
        simulation=SimulationConfig(
            duration_days=int(duration), stochastic=stochastic,
            overdispersion=(overdispersion if stochastic else None),
            seed=int(seed) if stochastic else None,
            engine=engine, n_agents=int(n_agents),
        ),
    ).validate()


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_epidemic_curve(df: pd.DataFrame, capacity=None) -> go.Figure:
    # Builds a Plotly figure with one line per compartment found in the DataFrame
    # (which comes from sim.to_columns()). Returned for st.plotly_chart to render.
    fig = go.Figure()
    series = [
        ("S", "Susceptible"), ("E", "Exposed"), ("infectious", "Infectious"),
        ("H", "Hospitalised"), ("C", "ICU"), ("R", "Recovered"), ("D", "Dead"),
    ]
    # Merge a colour for the synthetic "infectious" series into the compartment
    # palette (dict unpacking; later keys win, but there's no overlap here).
    palette = {"infectious": "#e45756", **COMPARTMENT_COLORS}
    for key, label in series:
        # Skip any series the current engine/run didn't produce a column for.
        if key in df:
            fig.add_trace(go.Scatter(
                x=df["day"], y=df[key], name=label, mode="lines",
                line=dict(color=palette.get(key, None)),
            ))
    fig.update_layout(
        title="Epidemic curve", xaxis_title="Day", yaxis_title="People",
        hovermode="x unified", legend=dict(orientation="h"), height=420,
    )
    return fig


def plot_incidence(df: pd.DataFrame) -> go.Figure:
    # Daily new infections (bars) and new deaths (line on a secondary y-axis).
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["day"], y=df["new_infections"], name="New infections (true)",
                         marker_color="#e45756"))
    # Reported cases — what surveillance would actually observe (under-reported /
    # delayed). Only distinct from true symptomatic onsets when reporting is set.
    if "reported_cases" in df:
        fig.add_trace(go.Scatter(x=df["day"], y=df["reported_cases"], name="Reported cases",
                                 line=dict(color="#f58518", dash="dot")))
    fig.add_trace(go.Scatter(x=df["day"], y=df["new_deaths"], name="New deaths",
                             yaxis="y2", line=dict(color="#333333")))
    fig.update_layout(
        title="Daily incidence", xaxis_title="Day",
        yaxis=dict(title="New infections"),
        yaxis2=dict(title="New deaths", overlaying="y", side="right"),
        hovermode="x unified", legend=dict(orientation="h"), height=320,
    )
    return fig


def plot_rt(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["rt"], name="Rt",
                             line=dict(color="#4c78a8")))
    # Reference line at Rt = 1, the epidemic growth/decline threshold.
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
                  annotation_text="Rt = 1")
    fig.update_layout(title="Effective reproduction number Rₜ", xaxis_title="Day",
                      yaxis_title="Rt", height=320, hovermode="x unified")
    return fig


def plot_hospital(df: pd.DataFrame, icu_capacity=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["H"], name="Hospital (non-ICU)",
                             line=dict(color="#b279a2")))
    fig.add_trace(go.Scatter(x=df["day"], y=df["C"], name="ICU",
                             line=dict(color="#8b1a1a")))
    # Draw the capacity line only when a finite ICU cap was configured.
    if icu_capacity is not None:
        fig.add_hline(y=icu_capacity, line_dash="dash", line_color="red",
                      annotation_text="ICU capacity")
    fig.update_layout(title="Healthcare burden", xaxis_title="Day",
                      yaxis_title="Occupancy", height=320, hovermode="x unified",
                      legend=dict(orientation="h"))
    return fig


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def ensure_sim(scenario: ScenarioConfig):
    # st.session_state persists across reruns (unlike ordinary locals, which are
    # recreated each run). Create a fresh Simulation only on first load or when
    # the "dirty" flag says the scenario changed; otherwise keep the running one
    # so its accumulated history survives the rerun.
    if "sim" not in st.session_state or st.session_state.get("dirty", False):
        st.session_state.sim = Simulation(scenario)
        st.session_state.playing = False
        st.session_state.dirty = False


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():
    st.title("🦠 Outbreak — Realistic Epidemic Simulator")
    st.caption(
        "Age-structured stochastic SEIR model with pre-symptomatic and "
        "asymptomatic transmission, vaccination, interventions, waning immunity "
        "and healthcare-capacity effects. Parameters are illustrative."
    )

    scenario = build_scenario()

    # Rebuild the simulation whenever the scenario signature changes.
    # `sig` is a canonical fingerprint of the scenario: dumping the config dict
    # with sort_keys=True makes the string identical whenever the parameters are
    # identical (key order doesn't matter). If it differs from last rerun's
    # stored signature, the user changed a parameter, so mark the sim "dirty" so
    # ensure_sim() rebuilds it below.
    sig = json.dumps(scenario.to_dict(), sort_keys=True, default=str)
    if st.session_state.get("sig") != sig:
        st.session_state.sig = sig
        st.session_state.dirty = True
    ensure_sim(scenario)
    # Pull the persisted Simulation back out of session_state for use this rerun.
    sim: Simulation = st.session_state.sim

    # st.tabs returns one container per tab; widgets created under each `with`
    # block render in that tab. All three tabs' code runs every rerun.
    tab_sim, tab_ensemble, tab_io = st.tabs(["▶ Simulate", "📊 Ensemble", "💾 Save / Load"])

    # ---------------------------------------------------------------- Simulate
    with tab_sim:
        # st.columns(6) lays out six side-by-side control cells. st.button
        # returns True only on the rerun triggered by *this* click, so each
        # `if cN.button(...)` block fires once per press.
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        if c1.button("⟲ Build / Reset", use_container_width=True):
            sim.reset()
            st.session_state.playing = False
        if c2.button("▶ Play", use_container_width=True):
            # Just flip the persistent "playing" flag; the animation loop at the
            # bottom of this tab does the actual stepping.
            st.session_state.playing = True
        if c3.button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False
            sim.pause()
        if c4.button("⏭ Step +7d", use_container_width=True):
            # Advance one week. dt is the step size in days, so 7/dt is the
            # number of integration steps in a week.
            st.session_state.playing = False
            sim.run(steps=int(7 / sim.config.simulation.dt))
        if c5.button("⏩ Run to end", use_container_width=True):
            st.session_state.playing = False
            sim.run_to_end()
        speed = c6.selectbox("Speed", ["Fast", "Medium", "Slow"], index=0)

        if sim.config.simulation.engine == "agent":
            engine_tag = f"agent-based · {sim.model.n_agents:,} agents (×{sim.model.scale:.0f})"
            if sim.config.network.enabled:
                engine_tag += " · networked"
        else:
            engine_tag = "compartmental"
        st.progress(sim.progress, text=f"Day {sim.current_day:.0f} / "
                                       f"{sim.config.simulation.duration_days}  •  "
                                       f"{engine_tag}  •  {sim.state.value}")

        # Only render charts/metrics once at least one day has been simulated.
        if sim.history:
            # to_columns() turns the per-day history into a dict of equal-length
            # lists (column name -> values), which DataFrame consumes directly.
            df = pd.DataFrame(sim.to_columns())
            summary = sim.summary()

            # Five metric cards across the top. st.metric(label, value, delta).
            m = st.columns(5)
            m[0].metric("Cumulative infections", f"{summary.total_infections:,.0f}",
                        f"{100*summary.attack_rate:.1f}% of pop")
            m[1].metric("Deaths", f"{summary.total_deaths:,.0f}",
                        f"IFR {100*summary.infection_fatality_ratio:.2f}%")
            m[2].metric("Peak infectious", f"{summary.peak_infectious:,.0f}",
                        f"day {summary.peak_infectious_day:.0f}" if summary.peak_infectious_day else None)
            m[3].metric("Peak ICU", f"{summary.peak_icu_occupancy:,.0f}",
                        f"day {summary.peak_icu_day:.0f}" if summary.peak_icu_day else None)
            m[4].metric("Current Rt", f"{df['rt'].iloc[-1]:.2f}",
                        f"R₀ = {summary.r0:.1f}")

            st.plotly_chart(plot_epidemic_curve(df), use_container_width=True)
            cc = st.columns(2)
            cc[0].plotly_chart(plot_rt(df), use_container_width=True)
            cc[1].plotly_chart(plot_incidence(df), use_container_width=True)
            st.plotly_chart(
                plot_hospital(df, scenario.healthcare.icu_capacity),
                use_container_width=True,
            )

            # Timeline scrubbing -------------------------------------------------
            with st.expander("🔎 Inspect a specific day"):
                # Slider ranges from day 0 to the last simulated day and defaults
                # to that last day; record_at_day() fetches that day's snapshot.
                day = st.slider("Day", 0, int(df["day"].iloc[-1]),
                                int(df["day"].iloc[-1]))
                rec = sim.record_at_day(day)
                if rec is not None:
                    cols = st.columns(5)
                    cols[0].metric("Susceptible", f"{rec.S:,.0f}")
                    cols[1].metric("Infectious", f"{rec.Ip+rec.Ia+rec.Is:,.0f}")
                    cols[2].metric("In hospital", f"{rec.H+rec.C:,.0f}")
                    cols[3].metric("Recovered", f"{rec.R:,.0f}")
                    cols[4].metric("Rt", f"{rec.rt:.2f}")
                    if rec.infectious_by_age is not None and len(rec.infectious_by_age) > 1:
                        labels = scenario.population.age_group_labels
                        st.bar_chart(pd.DataFrame(
                            {"infectious": rec.infectious_by_age}, index=labels))

            # Serialise the whole DataFrame to CSV text in memory; the button
            # offers it as a download without writing any file server-side.
            st.download_button(
                "⬇ Download time series (CSV)", df.to_csv(index=False),
                file_name="epidemic_timeseries.csv", mime="text/csv",
            )
        else:
            st.info("Press **Play** or **Run to end** to start the simulation.")

        # Animation loop: advance a chunk and rerun while playing.
        # This is how Streamlit "animates": when playing, step the sim by a
        # speed-dependent chunk of days, pause briefly, then st.rerun() to
        # restart the script from the top — which redraws the charts and re-enters
        # this block, stepping again, until paused or the run finishes.
        if st.session_state.get("playing") and not sim.is_finished:
            chunk = {"Fast": 14, "Medium": 5, "Slow": 1}[speed]
            sim.run(steps=int(chunk / sim.config.simulation.dt))
            time.sleep(0.05)
            st.rerun()

    # ---------------------------------------------------------------- Ensemble
    with tab_ensemble:
        st.subheader("Stochastic ensemble")
        st.caption("Run many independent realisations to see the range of "
                   "plausible outcomes, not just one trajectory.")
        n_runs = st.slider("Number of runs", 5, 200, 30, 5)
        variable = st.selectbox(
            "Variable", ["Is", "infectious", "C", "H", "new_infections", "D"], index=1,
            format_func=lambda k: {
                "Is": "Symptomatic infectious", "infectious": "All infectious",
                "C": "ICU occupancy", "H": "Hospital occupancy",
                "new_infections": "Daily new infections", "D": "Cumulative deaths",
            }[k],
        )
        if st.button("Run ensemble"):
            # Run many independent stochastic realisations. The progress callback
            # updates the bar as each run completes (i of n).
            prog = st.progress(0.0)
            histories = run_ensemble(
                scenario, n_runs=n_runs,
                base_seed=scenario.simulation.seed,
                progress=lambda i, n: prog.progress(i / n),
            )
            if variable == "infectious":
                # Derive total infectious per run before aggregating. There is no
                # single "infectious" column, so we attach a temporary attribute
                # on each per-day record and aggregate that synthetic field.
                for h in histories:
                    for r in h:
                        r.infectious_tmp = r.Ip + r.Ia + r.Is
                agg = aggregate_ensemble(histories, "infectious_tmp", (0.05, 0.5, 0.95))
            else:
                # Collapse the runs into 5th/50th/95th percentile curves per day.
                agg = aggregate_ensemble(histories, variable, (0.05, 0.5, 0.95))
            days = agg["day"]
            # Shaded uncertainty band: an invisible upper-quantile trace, then a
            # lower-quantile trace with fill="tonexty" to colour the gap between
            # them, and finally the median line drawn on top.
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=agg["q0.95"], line=dict(width=0),
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=days, y=agg["q0.05"], fill="tonexty",
                                     fillcolor="rgba(228,87,86,0.2)", line=dict(width=0),
                                     name="90% band"))
            fig.add_trace(go.Scatter(x=days, y=agg["q0.5"], line=dict(color="#e45756"),
                                     name="Median"))
            fig.update_layout(title=f"{n_runs}-run ensemble", xaxis_title="Day",
                              yaxis_title="People", height=460, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            finals = np.array([h[-1].D for h in histories])
            st.write(f"**Deaths across runs** — median {np.median(finals):,.0f}, "
                     f"range {finals.min():,.0f}–{finals.max():,.0f}")

    # ------------------------------------------------------------- Save / Load
    with tab_io:
        st.subheader("Save / load a scenario or run")
        st.download_button(
            "⬇ Download scenario (JSON)",
            json.dumps(scenario.to_dict(), indent=2, default=str),
            file_name="scenario.json", mime="application/json",
        )
        if sim.history:
            st.download_button(
                "⬇ Download run snapshot (JSON)",
                json.dumps(sim.to_dict(), indent=2,
                           default=lambda o: o.tolist() if hasattr(o, "tolist") else o),
                file_name="run_snapshot.json", mime="application/json",
            )
        uploaded = st.file_uploader("Load a run snapshot (JSON)", type="json")
        if uploaded is not None and st.button("Load snapshot"):
            # Treat the upload as untrusted: cap its size before parsing (so a
            # huge file can't exhaust memory in json.load) and surface any
            # malformed/inconsistent snapshot as a friendly error rather than an
            # unhandled traceback. from_dict validates the config and the
            # engines' set_state shape-check the restored state.
            if uploaded.size > MAX_SNAPSHOT_BYTES:
                st.error(
                    f"File is too large ({uploaded.size / 1e6:.1f} MB); the limit "
                    f"is {MAX_SNAPSHOT_BYTES // 1_000_000} MB."
                )
            else:
                try:
                    data = json.loads(uploaded.getvalue().decode("utf-8"))
                    st.session_state.sim = Simulation.from_dict(data)
                    st.session_state.playing = False
                    st.success("Snapshot loaded. Switch to the Simulate tab to continue.")
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                    st.error(f"Could not load snapshot: {exc}")


if __name__ == "__main__":
    main()
