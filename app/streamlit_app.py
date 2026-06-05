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
    HealthcareConfig,
    Intervention,
    InterventionConfig,
    PopulationConfig,
    ScenarioConfig,
    SimulationConfig,
    VaccinationConfig,
    DISEASE_PRESETS,
)
from outbreak.metrics import aggregate_ensemble
from outbreak.simulation import RunState, Simulation, run_ensemble

st.set_page_config(page_title="Outbreak — Epidemic Simulator", layout="wide")

COMPARTMENT_COLORS = {
    "S": "#4c78a8", "V": "#72b7b2", "E": "#f5c518", "Ip": "#ff9d4e",
    "Ia": "#ffb86b", "Is": "#e45756", "H": "#b279a2", "C": "#8b1a1a",
    "R": "#54a24b", "D": "#333333",
}


# --------------------------------------------------------------------------- #
# Scenario construction from the sidebar
# --------------------------------------------------------------------------- #
def build_scenario() -> ScenarioConfig:
    st.sidebar.header("⚙️ Scenario parameters")

    preset_name = st.sidebar.selectbox(
        "Disease preset", list(DISEASE_PRESETS.keys()), index=0,
        help="A realistic starting point you can then fine-tune below.",
    )
    base_disease = DISEASE_PRESETS[preset_name]()

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
        waning_on = st.checkbox("Waning immunity",
                                value=base_disease.waning_immunity_days is not None)
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

    with st.sidebar.expander("Healthcare capacity"):
        cap_enabled = st.checkbox("Limit ICU capacity", value=False)
        icu_capacity = st.number_input("ICU beds", 0, 1_000_000, 500, 50,
                                       disabled=not cap_enabled)
        overflow_mult = st.slider("Overflow mortality multiplier", 1.0, 5.0, 2.0, 0.5,
                                  disabled=not cap_enabled)

    with st.sidebar.expander("Simulation controls", expanded=True):
        duration = st.number_input("Duration (days)", 30, 2000, 365, 5)
        stochastic = st.checkbox("Stochastic (random) dynamics", value=True)
        overdispersion = st.slider("Superspreading (lower = burstier)", 0.05, 5.0, 0.5, 0.05,
                                   disabled=not stochastic)
        seed = st.number_input("Random seed", 0, 1_000_000, 42, 1, disabled=not stochastic)

    interventions = []
    if npi_enabled and npi_end > npi_start:
        interventions.append(Intervention("intervention", int(npi_start), int(npi_end), npi_reduction))

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
        simulation=SimulationConfig(
            duration_days=int(duration), stochastic=stochastic,
            overdispersion=(overdispersion if stochastic else None),
            seed=int(seed) if stochastic else None,
        ),
    ).validate()


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_epidemic_curve(df: pd.DataFrame, capacity=None) -> go.Figure:
    fig = go.Figure()
    series = [
        ("S", "Susceptible"), ("E", "Exposed"), ("infectious", "Infectious"),
        ("H", "Hospitalised"), ("C", "ICU"), ("R", "Recovered"), ("D", "Dead"),
    ]
    palette = {"infectious": "#e45756", **COMPARTMENT_COLORS}
    for key, label in series:
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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["day"], y=df["new_infections"], name="New infections",
                         marker_color="#e45756"))
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
    sig = json.dumps(scenario.to_dict(), sort_keys=True, default=str)
    if st.session_state.get("sig") != sig:
        st.session_state.sig = sig
        st.session_state.dirty = True
    ensure_sim(scenario)
    sim: Simulation = st.session_state.sim

    tab_sim, tab_ensemble, tab_io = st.tabs(["▶ Simulate", "📊 Ensemble", "💾 Save / Load"])

    # ---------------------------------------------------------------- Simulate
    with tab_sim:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        if c1.button("⟲ Build / Reset", use_container_width=True):
            sim.reset()
            st.session_state.playing = False
        if c2.button("▶ Play", use_container_width=True):
            st.session_state.playing = True
        if c3.button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False
            sim.pause()
        if c4.button("⏭ Step +7d", use_container_width=True):
            st.session_state.playing = False
            sim.run(steps=int(7 / sim.config.simulation.dt))
        if c5.button("⏩ Run to end", use_container_width=True):
            st.session_state.playing = False
            sim.run_to_end()
        speed = c6.selectbox("Speed", ["Fast", "Medium", "Slow"], index=0)

        st.progress(sim.progress, text=f"Day {sim.current_day:.0f} / "
                                       f"{sim.config.simulation.duration_days}  •  {sim.state.value}")

        if sim.history:
            df = pd.DataFrame(sim.to_columns())
            summary = sim.summary()

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

            st.download_button(
                "⬇ Download time series (CSV)", df.to_csv(index=False),
                file_name="epidemic_timeseries.csv", mime="text/csv",
            )
        else:
            st.info("Press **Play** or **Run to end** to start the simulation.")

        # Animation loop: advance a chunk and rerun while playing.
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
            prog = st.progress(0.0)
            histories = run_ensemble(
                scenario, n_runs=n_runs,
                base_seed=scenario.simulation.seed,
                progress=lambda i, n: prog.progress(i / n),
            )
            if variable == "infectious":
                # Derive total infectious per run before aggregating.
                for h in histories:
                    for r in h:
                        r.infectious_tmp = r.Ip + r.Ia + r.Is
                agg = aggregate_ensemble(histories, "infectious_tmp", (0.05, 0.5, 0.95))
            else:
                agg = aggregate_ensemble(histories, variable, (0.05, 0.5, 0.95))
            days = agg["day"]
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
            data = json.load(uploaded)
            st.session_state.sim = Simulation.from_dict(data)
            st.session_state.playing = False
            st.success("Snapshot loaded. Switch to the Simulate tab to continue.")


if __name__ == "__main__":
    main()
