/* app.js — UI for the browser epidemic simulator.
   Drives the ported compartmental SEIR engine: builds a scenario from the
   controls, runs it entirely on-device, and renders the epidemic curve,
   a play/scrub timeline and headline statistics. */

import { makeScenario, runSimulation, summarize, DISEASE_PRESETS } from "./engine.js";
import { LineChart } from "./chart.js";

const $ = (id) => document.getElementById(id);
const els = {
  preset: $("ob-preset"),
  r0: $("ob-r0"), r0v: $("ob-r0-val"),
  pop: $("ob-pop"),
  inf: $("ob-inf"), infv: $("ob-inf-val"),
  duration: $("ob-duration"), durationv: $("ob-duration-val"),
  vax: $("ob-vax"), vaxOpts: $("ob-vax-opts"),
  vaxStart: $("ob-vax-start"), vaxStartv: $("ob-vax-start-val"),
  vaxRate: $("ob-vax-rate"), vaxRatev: $("ob-vax-rate-val"),
  vaxCov: $("ob-vax-cov"), vaxCovv: $("ob-vax-cov-val"),
  npi: $("ob-npi"), npiOpts: $("ob-npi-opts"),
  npiStart: $("ob-npi-start"), npiStartv: $("ob-npi-start-val"),
  npiLen: $("ob-npi-len"), npiLenv: $("ob-npi-len-val"),
  npiRed: $("ob-npi-red"), npiRedv: $("ob-npi-red-val"),
  icu: $("ob-icu"), icuOpts: $("ob-icu-opts"),
  icuCap: $("ob-icu-cap"), icuCapv: $("ob-icu-cap-val"),
  season: $("ob-season"), seasonv: $("ob-season-val"),
  stoch: $("ob-stoch"),
  seed: $("ob-seed"),
  run: $("ob-run"), play: $("ob-play"),
  chart: $("ob-chart"), legend: $("ob-legend"),
  scrub: $("ob-scrub"), day: $("ob-day"), read: $("ob-read"),
  stats: $("ob-stats"),
};

Object.entries(DISEASE_PRESETS).forEach(([key, p]) => {
  const opt = document.createElement("option");
  opt.value = key; opt.textContent = p.label;
  els.preset.appendChild(opt);
});
els.preset.value = "covid_like";
els.preset.addEventListener("change", () => {
  els.r0.value = DISEASE_PRESETS[els.preset.value].r0;
  syncLabels();
});

function syncLabels() {
  els.r0v.textContent = Number(els.r0.value).toFixed(1);
  els.infv.textContent = Number(els.inf.value).toLocaleString();
  els.durationv.textContent = els.duration.value + " d";
  els.vaxStartv.textContent = "day " + els.vaxStart.value;
  els.vaxRatev.textContent = Number(els.vaxRate.value).toFixed(1) + "%/d";
  els.vaxCovv.textContent = els.vaxCov.value + "%";
  els.npiStartv.textContent = "day " + els.npiStart.value;
  els.npiLenv.textContent = els.npiLen.value + " d";
  els.npiRedv.textContent = els.npiRed.value + "%";
  els.icuCapv.textContent = els.icuCap.value + " / 100k";
  els.seasonv.textContent = els.season.value + "%";
}
[els.r0, els.inf, els.duration, els.vaxStart, els.vaxRate, els.vaxCov,
 els.npiStart, els.npiLen, els.npiRed, els.icuCap, els.season]
  .forEach((el) => el.addEventListener("input", syncLabels));

function wireToggle(toggle, panel) {
  const sync = () => { panel.hidden = !toggle.checked; };
  toggle.addEventListener("change", sync);
  sync();
}
wireToggle(els.vax, els.vaxOpts);
wireToggle(els.npi, els.npiOpts);
wireToggle(els.icu, els.icuOpts);

const chart = new LineChart(els.chart);
let history = null;

const SERIES_DEFS = [
  { key: "infectious", label: "Infectious", color: "#ff4324", axis: "left", width: 2.4 },
  { key: "new_infections", label: "New infections / day", color: "#ff6a4d", axis: "left" },
  { key: "H", label: "Hospitalised", color: "#e8b34a", axis: "left" },
  { key: "C", label: "In ICU", color: "#d65db1", axis: "left" },
  { key: "D", label: "Deaths (cumulative)", color: "#9b978e", axis: "left" },
  { key: "rt", label: "Rₜ", color: "#5bbf8a", axis: "right", width: 1.6 },
];
const hidden = new Set(["new_infections"]);

function buildScenario() {
  const total = parseInt(els.pop.value, 10);
  const interventions = [];
  if (els.npi.checked) {
    const start = parseInt(els.npiStart.value, 10);
    interventions.push({
      start_day: start,
      end_day: start + parseInt(els.npiLen.value, 10),
      transmission_reduction: parseInt(els.npiRed.value, 10) / 100,
    });
  }
  const seedRaw = els.seed.value.trim();
  return makeScenario({
    preset: els.preset.value,
    r0: parseFloat(els.r0.value),
    total_population: total,
    initial_infected: parseInt(els.inf.value, 10),
    duration_days: parseInt(els.duration.value, 10),
    stochastic: els.stoch.checked,
    seed: seedRaw === "" ? null : parseInt(seedRaw, 10) >>> 0,
    vaccination: {
      enabled: els.vax.checked,
      start_day: parseInt(els.vaxStart.value, 10),
      daily_rate: parseFloat(els.vaxRate.value) / 100,
      coverage_cap: parseInt(els.vaxCov.value, 10) / 100,
    },
    interventions,
    healthcare: els.icu.checked
      ? { icu_capacity: (parseInt(els.icuCap.value, 10) * total) / 100000 }
      : {},
    environment: { seasonal_amplitude: parseInt(els.season.value, 10) / 100 },
  });
}

const seriesValue = (key, r) => (key === "infectious" ? r.Ip + r.Ia + r.Is : r[key]);

function run() {
  els.run.disabled = true;
  els.run.textContent = "Simulating…";
  setTimeout(() => {
    const scenario = buildScenario();
    history = runSimulation(scenario);
    const days = history.map((r) => r.day);
    const series = SERIES_DEFS.map((def) => ({
      ...def,
      values: history.map((r) => seriesValue(def.key, r)),
      hidden: hidden.has(def.key),
    }));
    const hlines = [];
    if (els.icu.checked && scenario.healthcare.icu_capacity != null) {
      hlines.push({ y: scenario.healthcare.icu_capacity, axis: "left", color: "#d65db1", label: "ICU capacity" });
    }
    chart.setData({ days, series, hlines });

    els.scrub.max = String(days.length - 1);
    els.scrub.value = String(days.length - 1);
    renderLegend();
    renderStats(summarize(history, scenario.disease.r0));
    updateScrub(days.length - 1);

    els.run.disabled = false;
    els.run.textContent = "Run simulation";
    els.play.disabled = false;
  }, 20);
}
els.run.addEventListener("click", run);

function renderLegend() {
  els.legend.innerHTML = "";
  SERIES_DEFS.forEach((def) => {
    const b = document.createElement("button");
    if (hidden.has(def.key)) b.classList.add("off");
    b.innerHTML = `<span class="swatch" style="background:${def.color}"></span>${def.label}`;
    b.addEventListener("click", () => {
      if (hidden.has(def.key)) hidden.delete(def.key); else hidden.add(def.key);
      b.classList.toggle("off");
      const s = chart.data.series.find((x) => x.key === def.key);
      if (s) s.hidden = hidden.has(def.key);
      chart.render(parseInt(els.scrub.value, 10));
    });
    els.legend.appendChild(b);
  });
}

function fmtN(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return Math.round(v).toLocaleString();
}
function updateScrub(idx) {
  if (!history) return;
  const r = history[idx];
  els.day.textContent = "Day " + Math.round(r.day);
  const infectious = r.Ip + r.Ia + r.Is;
  els.read.textContent =
    `infectious ${fmtN(infectious)} · hosp ${fmtN(r.H)} · ICU ${fmtN(r.C)} · deaths ${fmtN(r.D)} · Rₜ ${r.rt.toFixed(2)}`;
  chart.render(idx);
}
els.scrub.addEventListener("input", () => { stopPlay(); updateScrub(parseInt(els.scrub.value, 10)); });

let playing = false, rafId = null;
function stopPlay() {
  playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  els.play.textContent = "▶ Play";
}
function startPlay() {
  if (!history) return;
  const n = history.length - 1;
  let idx = parseInt(els.scrub.value, 10);
  if (idx >= n) idx = 0;
  playing = true;
  els.play.textContent = "❚❚ Pause";
  const perFrame = Math.max(1, Math.round(n / 180));
  const tick = () => {
    if (!playing) return;
    idx = Math.min(n, idx + perFrame);
    els.scrub.value = String(idx);
    updateScrub(idx);
    if (idx >= n) { stopPlay(); return; }
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}
els.play.addEventListener("click", () => (playing ? stopPlay() : startPlay()));

function renderStats(s) {
  if (!s) { els.stats.innerHTML = ""; return; }
  const pct = (x) => (x * 100).toFixed(1) + "%";
  const tiles = [
    { label: "Attack rate", value: pct(s.attack_rate), accent: true },
    { label: "Total infections", value: fmtN(s.total_infections) },
    { label: "Total deaths", value: fmtN(s.total_deaths) },
    { label: "Infection fatality", value: pct(s.infection_fatality_ratio) },
    { label: "Peak infectious", value: fmtN(s.peak_infectious), sub: s.peak_infectious_day != null ? "day " + Math.round(s.peak_infectious_day) : "" },
    { label: "Peak ICU", value: fmtN(s.peak_icu_occupancy), sub: s.peak_icu_day != null ? "day " + Math.round(s.peak_icu_day) : "" },
    { label: "Peak Rₜ", value: s.peak_rt.toFixed(2) },
    { label: "Rₜ < 1", value: s.rt_crossed_one_day != null ? "day " + Math.round(s.rt_crossed_one_day) : "—" },
  ];
  els.stats.innerHTML = tiles
    .map((t) =>
      `<div class="stat${t.accent ? " stat--accent" : ""}">
        <div class="stat__label">${t.label}</div>
        <div class="stat__value">${t.value}${t.sub ? `<small>${t.sub}</small>` : ""}</div>
      </div>`)
    .join("");
}

syncLabels();
run();
