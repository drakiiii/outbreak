"""Spatial / map view for the metapopulation (coupled regions).

Run with::

    streamlit run app/metapop_app.py

Lays a set of regions out in space, runs a coupled metapopulation epidemic, and
lets you scrub through time on a map: each region is a bubble sized by population
and coloured by its current infection prevalence, with the mobility links drawn
between them — so you can watch the epidemic spread geographically from the seed
region outward. Below the map are the combined epidemic curve and a table of when
each region's outbreak arrives.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from outbreak import (
    MetapopulationConfig,
    MetapopulationSimulation,
    Region,
    preset_scenario,
)
from outbreak.config import SimulationConfig

st.set_page_config(page_title="Outbreak — Spatial spread", layout="wide")

# A handful of major world cities (name, longitude, latitude) for the real-map
# layout. Coordinates are stored as (x=lon, y=lat) so the gravity model and the
# geographic plot both work from the same array.
CITIES = [
    ("London", -0.13, 51.51), ("New York", -74.01, 40.71), ("Tokyo", 139.69, 35.69),
    ("Paris", 2.35, 48.85), ("São Paulo", -46.63, -23.55), ("Cairo", 31.24, 30.04),
    ("Mumbai", 72.88, 19.08), ("Moscow", 37.62, 55.75), ("Lagos", 3.38, 6.52),
    ("Beijing", 116.41, 39.90), ("Los Angeles", -118.24, 34.05), ("Mexico City", -99.13, 19.43),
    ("Johannesburg", 28.05, -26.20), ("Singapore", 103.82, 1.35), ("Sydney", 151.21, -33.87),
    ("Toronto", -79.38, 43.65),
]


# --------------------------------------------------------------------------- #
# Region layouts: coordinates + a neighbour mobility matrix so the wave spreads
# geographically (nearby regions are connected).
# --------------------------------------------------------------------------- #
def build_layout(layout: str, k: int):
    """Return (coords (k,2), default mobility matrix or None, names, is_geographic)."""
    names = [f"R{i+1}" for i in range(k)]
    geographic = False
    if layout == "Line":
        coords = np.array([[i, 0.0] for i in range(k)])
        mob = _neighbour_matrix(k, lambda i, j: abs(i - j) == 1)
    elif layout == "Ring":
        ang = 2 * np.pi * np.arange(k) / k
        coords = np.column_stack([np.cos(ang), np.sin(ang)])
        mob = _neighbour_matrix(k, lambda i, j: (i - j) % k in (1, k - 1))
    elif layout == "Grid":
        side = int(np.ceil(np.sqrt(k)))
        coords = np.array([[i % side, i // side] for i in range(k)], dtype=float)
        mob = _neighbour_matrix(
            k, lambda i, j: abs(coords[i, 0] - coords[j, 0]) + abs(coords[i, 1] - coords[j, 1]) == 1
        )
    elif layout == "World cities (map)":
        k = min(k, len(CITIES))
        names = [CITIES[i][0] for i in range(k)]
        coords = np.array([[CITIES[i][1], CITIES[i][2]] for i in range(k)], dtype=float)
        mob = None          # gravity is the natural default on a real map
        geographic = True
    else:  # "Random (all-to-all)"
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 10, size=(k, 2))
        mob = None          # None => uniform mixing with every other region
    return coords, mob, names, geographic


def _neighbour_matrix(k: int, is_neighbour) -> np.ndarray:
    m = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j and is_neighbour(i, j):
                m[i, j] = 1.0
    return m


# --------------------------------------------------------------------------- #
# Sidebar -> scenario + metapopulation
# --------------------------------------------------------------------------- #
def build_metapop():
    st.sidebar.header("🗺️ Spatial scenario")
    preset = st.sidebar.selectbox("Disease", ["covid_like", "influenza_like", "measles_like"])
    k = st.sidebar.slider("Number of regions", 2, 16, 6)
    layout = st.sidebar.selectbox(
        "Layout", ["Line", "Ring", "Grid", "Random (all-to-all)", "World cities (map)"])
    pop = st.sidebar.number_input("Population per region", 10_000, 2_000_000, 100_000, 10_000)
    duration = st.sidebar.slider("Duration (days)", 60, 730, 300, 10)

    st.sidebar.subheader("Spread between regions")
    mobility_choice = st.sidebar.radio(
        "Mobility", ["Layout default", "Gravity (distance × population)"],
        index=1 if layout == "World cities (map)" else 0,
        help="How regions are connected. Gravity: flow grows with a region's "
             "population and falls with distance.")
    gravity_decay = st.sidebar.slider("Gravity distance decay", 1.0, 4.0, 2.0, 0.5,
                                      disabled=mobility_choice == "Layout default")
    coupling = st.sidebar.slider("Prevalence coupling", 0.0, 0.1, 0.0, 0.005,
                                 help="Smooth, mean-field leakage between connected regions.")
    travel_rate = st.sidebar.slider("Explicit travel rate", 0.0, 0.02, 0.004, 0.001,
                                    help="Per-infectious-person daily chance of a trip that "
                                         "seeds an importation in a connected region.")
    seed = st.sidebar.number_input("Random seed", 0, 1_000_000, 0, 1)

    coords, layout_mobility, names, geographic = build_layout(layout, k)
    k = len(names)          # world layout may cap k to the city list
    # Seed only the first region (the others start empty so we can watch arrival).
    regions = [
        Region(name=names[i], population=int(pop),
               initial_infected=(50 if i == 0 else 0),
               x=float(coords[i, 0]), y=float(coords[i, 1]))
        for i in range(k)
    ]
    base = preset_scenario(preset, total_population=int(pop))
    base.disease.waning_immunity_days = None
    base.simulation = SimulationConfig(duration_days=int(duration), stochastic=False,
                                       overdispersion=None)
    if mobility_choice == "Layout default":
        config = MetapopulationConfig(regions=regions, coupling=coupling,
                                      travel_rate=travel_rate, mobility=layout_mobility)
    else:
        config = MetapopulationConfig(regions=regions, coupling=coupling,
                                      travel_rate=travel_rate, mobility=None,
                                      mobility_model="gravity", gravity_decay=gravity_decay)
    return base.validate(), config.validate(), int(seed), geographic


@st.cache_data(show_spinner="Running the spatial epidemic…")
def run_metapop(base_dict, config_dict, seed):
    """Run the metapopulation and return the data the map needs (cached)."""
    from outbreak.config import ScenarioConfig
    base = ScenarioConfig.from_dict(base_dict)
    regions = [Region(**r) for r in config_dict["regions"]]
    config = MetapopulationConfig(regions=regions, coupling=config_dict["coupling"],
                                  travel_rate=config_dict["travel_rate"],
                                  mobility=config_dict["mobility"],
                                  mobility_model=config_dict["mobility_model"],
                                  gravity_decay=config_dict["gravity_decay"])
    sim = MetapopulationSimulation(base, config, base_seed=seed)
    sim.run_to_end()

    # prevalence[region, day] = infectious fraction; plus combined infectious curve.
    prevalence = np.array([
        [(r.Ip + r.Ia + r.Is) / max(reg.population, 1) for r in hist]
        for hist, reg in zip(sim.histories, config.regions)
    ])
    days = np.array([r.day for r in sim.histories[0]])
    combined_infectious = prevalence.T @ np.array([reg.population for reg in config.regions])
    arrivals = [sim.first_infection_day(i, 100) for i in range(len(config.regions))]
    coords = np.array([[reg.x, reg.y] for reg in config.regions], dtype=float)
    pops = np.array([reg.population for reg in config.regions], dtype=float)
    names = [reg.name for reg in config.regions]
    mob = sim.mobility
    return prevalence, days, combined_infectious, arrivals, coords, pops, names, mob


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_map(coords, pops, names, mob, prevalence_at_day, vmax):
    fig = go.Figure()
    # Mobility links (drawn once, light grey).
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if mob[i, j] > 0 or mob[j, i] > 0:
                fig.add_trace(go.Scatter(
                    x=[coords[i, 0], coords[j, 0]], y=[coords[i, 1], coords[j, 1]],
                    mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1),
                    hoverinfo="skip", showlegend=False))
    # Region bubbles: size ~ population, colour ~ current prevalence.
    sizes = 12 + 40 * np.sqrt(pops / pops.max())
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers+text", text=names,
        textposition="top center",
        marker=dict(size=sizes, color=prevalence_at_day, cmin=0, cmax=vmax,
                    colorscale="Reds", showscale=True,
                    colorbar=dict(title="Infectious<br>prevalence"),
                    line=dict(color="#333", width=1)),
        customdata=(100 * prevalence_at_day),
        hovertemplate="%{text}<br>%{customdata:.2f}% infectious<extra></extra>",
        showlegend=False))
    fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False,
                      scaleanchor="x", scaleratio=1), margin=dict(l=10, r=10, t=30, b=10),
                      title="Regions (bubble size = population, colour = infection prevalence)")
    return fig


def plot_geo_map(coords, pops, names, mob, prevalence_at_day, vmax):
    """Render regions on a real world map (offline natural-earth projection)."""
    lon, lat = coords[:, 0], coords[:, 1]
    fig = go.Figure()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if mob[i, j] > 0 or mob[j, i] > 0:
                fig.add_trace(go.Scattergeo(
                    lon=[lon[i], lon[j]], lat=[lat[i], lat[j]], mode="lines",
                    line=dict(color="rgba(150,150,150,0.4)", width=1),
                    hoverinfo="skip", showlegend=False))
    sizes = 10 + 30 * np.sqrt(pops / pops.max())
    fig.add_trace(go.Scattergeo(
        lon=lon, lat=lat, text=names, mode="markers",
        marker=dict(size=sizes, color=prevalence_at_day, cmin=0, cmax=vmax,
                    colorscale="Reds", showscale=True,
                    colorbar=dict(title="Infectious<br>prevalence"),
                    line=dict(color="#333", width=1)),
        customdata=(100 * prevalence_at_day),
        hovertemplate="%{text}<br>%{customdata:.2f}% infectious<extra></extra>",
        showlegend=False))
    fig.update_geos(projection_type="natural earth", showcountries=True,
                    showcoastlines=True, showland=True, landcolor="#f5f5f5",
                    countrycolor="#dddddd")
    fig.update_layout(height=540, margin=dict(l=0, r=0, t=30, b=0),
                      title="Regions on the map (bubble colour = infection prevalence)")
    return fig


def plot_combined(days, combined_infectious, current_day):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=combined_infectious, fill="tozeroy",
                             line=dict(color="#e45756"), name="Infectious (all regions)"))
    fig.add_vline(x=current_day, line=dict(color="#333", dash="dash"))
    fig.update_layout(height=260, xaxis_title="Day", yaxis_title="People infectious",
                      margin=dict(l=10, r=10, t=30, b=10), title="Combined epidemic curve")
    return fig


# --------------------------------------------------------------------------- #
def main():
    st.title("🗺️ Outbreak — Spatial spread across regions")
    st.caption("Watch an epidemic seeded in one region spread to the others through "
               "travel and mixing. Scrub the day slider to animate it.")

    base, config, seed, geographic = build_metapop()
    # Cache key: serialise the inputs.
    base_dict = base.to_dict()
    config_dict = {
        "regions": [vars(r) for r in config.regions],
        "coupling": config.coupling, "travel_rate": config.travel_rate,
        "mobility": (np.asarray(config.mobility).tolist() if config.mobility is not None else None),
        "mobility_model": config.mobility_model, "gravity_decay": config.gravity_decay,
    }
    (prevalence, days, combined_infectious, arrivals,
     coords, pops, names, mob) = run_metapop(base_dict, config_dict, seed)

    vmax = float(max(prevalence.max(), 1e-6))
    day_idx = st.slider("Day", 0, len(days) - 1, min(len(days) - 1, len(days) // 3),
                        format="%d")
    plot = plot_geo_map if geographic else plot_map
    st.plotly_chart(plot(coords, pops, names, mob, prevalence[:, day_idx], vmax),
                    use_container_width=True)
    st.plotly_chart(plot_combined(days, combined_infectious, days[day_idx]),
                    use_container_width=True)

    # Arrival table: when each region's outbreak took off.
    st.subheader("Arrival times (day each region passed 100 cumulative cases)")
    cols = st.columns(min(len(names), 8))
    for i, name in enumerate(names):
        arrival = arrivals[i]
        cols[i % len(cols)].metric(name, f"day {arrival:.0f}" if arrival is not None else "—")


if __name__ == "__main__":
    main()
