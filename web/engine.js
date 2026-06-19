/*
 * outbreak engine (browser, ES module).
 *
 * A JavaScript port of this project's compartmental engine (outbreak/model.py,
 * epidemiology.py, contacts.py and config.py): a stochastic, age-structured
 * SEIR-type model. It mirrors the Python package's math — natural history,
 * next-generation-matrix R0 calibration,
 * chain-binomial transitions, vaccination, interventions, healthcare
 * overflow, seasonality and waning immunity — so the same scenarios produce
 * the same behaviour, but running entirely on the client.
 *
 * Compartments (per age group; the infectious cascade is split into two
 * strata, unvaccinated [0] and vaccinated [1]):
 *   S  susceptible            V  vaccinated-susceptible (reduced susceptibility)
 *   E  exposed/latent         Ip pre-symptomatic   Ia asymptomatic   Is symptomatic
 *   H  hospitalised           C  critical/ICU       R  recovered      D  dead
 *
 * Only the compartmental engine is ported. The agent-based and metapopulation
 * engines in the Python project rely on NumPy/SciPy and stay server-side.
 */

/* ============================================================
   Random number generation
   ============================================================ */
export function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  // Box–Muller; one draw per call is fine for our volume.
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function samplePoisson(rng, lam) {
  // Knuth's algorithm — only used for modest lambda (< ~30).
  if (lam <= 0) return 0;
  const limit = Math.exp(-lam);
  let k = 0, p = 1.0;
  do { k++; p *= rng(); } while (p > limit);
  return k - 1;
}

function sampleGamma(rng, shape, scale) {
  // Marsaglia & Tsang; handles shape < 1 via the boosting trick.
  if (shape < 1) {
    const u = rng();
    return sampleGamma(rng, shape + 1, scale) * Math.pow(u, 1.0 / shape);
  }
  const d = shape - 1.0 / 3.0;
  const c = 1.0 / Math.sqrt(9.0 * d);
  for (;;) {
    let x, vv;
    do { x = gaussian(rng); vv = 1.0 + c * x; } while (vv <= 0);
    vv = vv * vv * vv;
    const u = rng();
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * vv * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1.0 - vv + Math.log(vv))) return d * vv * scale;
  }
}

// Binomial(n, p) draw, with approximations chosen by scale so large
// populations stay fast while small early counts keep their stochasticity.
function sampleBinomial(rng, n, p) {
  n = Math.round(n > 0 ? n : 0);
  if (n <= 0 || p <= 0) return 0;
  if (p >= 1) return n;
  if (n < 50) {
    let k = 0;
    for (let i = 0; i < n; i++) if (rng() < p) k++;
    return k;
  }
  const mean = n * p;
  if (mean < 30) {
    const k = samplePoisson(rng, mean);
    return k > n ? n : k;
  }
  const sd = Math.sqrt(mean * (1 - p));
  let x = Math.round(mean + sd * gaussian(rng));
  if (x < 0) x = 0;
  if (x > n) x = n;
  return x;
}

/* ============================================================
   Small array helpers (n is tiny — typically 4 age groups)
   ============================================================ */
const vec = (n, fill = 0) => new Array(n).fill(fill);
const mat2 = (n) => [vec(n), vec(n)];
const asArray = (value, n) => (Array.isArray(value) ? value.slice() : vec(n, value));

/* ============================================================
   Contacts (mirrors outbreak/contacts.py)
   ============================================================ */
const DEFAULT_CONTACT_MATRIX_4 = [
  [9.0, 4.0, 1.2, 0.6],
  [3.0, 8.0, 2.5, 1.0],
  [1.2, 3.5, 4.0, 1.5],
  [0.6, 1.5, 1.5, 2.5],
];

function defaultContactMatrix(nAge) {
  if (nAge === 1) return [[10.0]];
  if (nAge === 4) return DEFAULT_CONTACT_MATRIX_4.map((r) => r.slice());
  const base = [];
  const ageScale = [];
  for (let i = 0; i < nAge; i++) ageScale.push(1.0 - (0.4 * i) / (nAge - 1));
  for (let i = 0; i < nAge; i++) {
    const row = [];
    for (let j = 0; j < nAge; j++) {
      row.push((8.0 * Math.exp(-Math.abs(i - j) / 1.5) + 1.0) * ageScale[i]);
    }
    base.push(row);
  }
  return base;
}

function symmetrize(C, N) {
  const n = C.length;
  const out = [];
  for (let i = 0; i < n; i++) out.push(vec(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const totalIJ = C[i][j] * N[i];
      const totalJI = C[j][i] * N[j];
      const total = 0.5 * (totalIJ + totalJI);
      out[i][j] = N[i] > 0 ? total / N[i] : 0.0;
    }
  }
  return out;
}

/* ============================================================
   Epidemiology math (mirrors outbreak/epidemiology.py)
   ============================================================ */
function transitionProbability(rate, dt) {
  return 1.0 - Math.exp(-rate * dt);
}

// K0[i][j] = C[j][i] * duration[j], optionally scaled by susceptibility[i].
function ngmUnit(contact, infectiousDuration, susceptibility) {
  const n = contact.length;
  const k0 = [];
  for (let i = 0; i < n; i++) {
    const row = vec(n);
    for (let j = 0; j < n; j++) {
      let v = contact[j][i] * infectiousDuration[j];
      if (susceptibility) v *= susceptibility[i];
      row[j] = v;
    }
    k0.push(row);
  }
  return k0;
}

// Spectral radius of a small non-negative matrix via power iteration
// (the next-generation matrix is non-negative, so the dominant eigenvalue
// is its real Perron root).
function spectralRadius(matrix) {
  const n = matrix.length;
  if (n === 1) return Math.abs(matrix[0][0]);
  let v = vec(n, 1.0 / Math.sqrt(n));
  let lambda = 0;
  for (let iter = 0; iter < 300; iter++) {
    const w = vec(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += matrix[i][j] * v[j];
      w[i] = s;
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm);
    if (norm === 0) return 0;
    const next = w.map((x) => x / norm);
    if (Math.abs(norm - lambda) < 1e-10 * (norm || 1)) { lambda = norm; break; }
    lambda = norm;
    v = next;
  }
  return lambda;
}

function calibrateBeta(contact, infectiousDuration, r0, susceptibility) {
  const rho = spectralRadius(ngmUnit(contact, infectiousDuration, susceptibility));
  if (rho <= 0) throw new Error("degenerate contact structure: cannot calibrate beta");
  return r0 / rho;
}

function icuDeathProbability(icuOccupancy, deathRate, healthcare) {
  const out = deathRate.slice();
  let overflow = 0.0;
  const cap = healthcare.icu_capacity;
  if (cap != null && icuOccupancy > cap) {
    overflow = icuOccupancy - cap;
    const shareOver = overflow / icuOccupancy;
    const mult = 1.0 + shareOver * (healthcare.overflow_mortality_multiplier - 1.0);
    for (let i = 0; i < out.length; i++) out[i] = Math.min(1.0, out[i] * mult);
  }
  return { deathProb: out, overflow };
}

/* ============================================================
   Disease presets (mirrors outbreak/config.py)
   ============================================================ */
export const AGE_LABELS = ["0–17", "18–49", "50–64", "65+"];
export const AGE_DISTRIBUTION = [0.22, 0.42, 0.19, 0.17];

export const DISEASE_PRESETS = {
  covid_like: {
    name: "covid_like", label: "COVID-like", r0: 2.8,
    latent_period: 3.0, presymptomatic_period: 2.0, symptomatic_period: 7.0,
    asymptomatic_infectious_period: 7.0,
    rel_infectiousness_presymptomatic: 1.0, rel_infectiousness_asymptomatic: 0.5,
    asymptomatic_fraction: [0.6, 0.4, 0.3, 0.25],
    hospitalization_rate: [0.001, 0.012, 0.045, 0.11],
    icu_rate: [0.05, 0.1, 0.22, 0.3],
    death_rate: [0.1, 0.2, 0.35, 0.55],
    hospital_stay: 8.0, icu_stay: 10.0, waning_immunity_days: 270.0,
    susceptibility: 1.0,
  },
  influenza_like: {
    name: "influenza_like", label: "Influenza-like", r0: 1.4,
    latent_period: 1.5, presymptomatic_period: 0.5, symptomatic_period: 4.0,
    asymptomatic_infectious_period: 4.0,
    rel_infectiousness_presymptomatic: 0.8, rel_infectiousness_asymptomatic: 0.4,
    asymptomatic_fraction: [0.4, 0.45, 0.4, 0.35],
    hospitalization_rate: [0.005, 0.008, 0.02, 0.06],
    icu_rate: [0.05, 0.08, 0.15, 0.25],
    death_rate: [0.05, 0.08, 0.15, 0.3],
    hospital_stay: 5.0, icu_stay: 7.0, waning_immunity_days: 200.0,
    susceptibility: 1.0,
  },
  measles_like: {
    name: "measles_like", label: "Measles-like", r0: 14.0,
    latent_period: 8.0, presymptomatic_period: 3.0, symptomatic_period: 6.0,
    asymptomatic_infectious_period: 6.0,
    rel_infectiousness_presymptomatic: 0.6, rel_infectiousness_asymptomatic: 0.3,
    asymptomatic_fraction: [0.05, 0.05, 0.05, 0.05],
    hospitalization_rate: [0.1, 0.05, 0.06, 0.12],
    icu_rate: [0.1, 0.08, 0.1, 0.2],
    death_rate: [0.03, 0.02, 0.04, 0.1],
    hospital_stay: 6.0, icu_stay: 8.0, waning_immunity_days: null,
    susceptibility: 1.0,
  },
};

/* ============================================================
   Scenario assembly + parameter resolution
   ============================================================ */
export function makeScenario(opts = {}) {
  const preset = DISEASE_PRESETS[opts.preset || "covid_like"];
  const disease = { ...preset };
  if (opts.r0 != null) disease.r0 = opts.r0;

  return {
    population: {
      total_population: opts.total_population ?? 1_000_000,
      age_distribution: AGE_DISTRIBUTION.slice(),
      initial_infected: opts.initial_infected ?? 20,
      initial_immune_fraction: opts.initial_immune_fraction ?? 0.0,
    },
    disease,
    vaccination: {
      enabled: opts.vaccination?.enabled ?? false,
      start_day: opts.vaccination?.start_day ?? 30,
      daily_rate: opts.vaccination?.daily_rate ?? 0.005,
      coverage_cap: opts.vaccination?.coverage_cap ?? 0.7,
      prioritize_elderly: opts.vaccination?.prioritize_elderly ?? true,
      ve_susceptibility: opts.vaccination?.ve_susceptibility ?? 0.6,
      ve_severity: opts.vaccination?.ve_severity ?? 0.8,
      ve_transmission: opts.vaccination?.ve_transmission ?? 0.4,
    },
    interventions: opts.interventions ?? [],
    healthcare: {
      hospital_capacity: opts.healthcare?.hospital_capacity ?? null,
      icu_capacity: opts.healthcare?.icu_capacity ?? null,
      overflow_mortality_multiplier: opts.healthcare?.overflow_mortality_multiplier ?? 2.0,
    },
    environment: {
      seasonal_amplitude: opts.environment?.seasonal_amplitude ?? 0.0,
      seasonal_period_days: 365.0,
      seasonal_peak_day: opts.environment?.seasonal_peak_day ?? 0.0,
      external_infection_rate: opts.environment?.external_infection_rate ?? 0.0,
    },
    simulation: {
      duration_days: opts.duration_days ?? 300,
      dt: 1.0,
      stochastic: opts.stochastic ?? true,
      overdispersion: opts.overdispersion ?? 0.5,
      seed: opts.seed ?? null,
    },
  };
}

function populationByAge(total, dist) {
  const n = dist.length;
  const raw = dist.map((d) => d * total);
  const base = raw.map((x) => Math.floor(x));
  let remainder = total - base.reduce((a, b) => a + b, 0);
  if (remainder > 0) {
    const order = raw
      .map((x, i) => [x - base[i], i])
      .sort((a, b) => b[0] - a[0]);
    for (let k = 0; k < remainder; k++) base[order[k][1]] += 1;
  }
  return base;
}

function resolveParameters(scenario) {
  const d = scenario.disease;
  const n = scenario.population.age_distribution.length;
  const sigma = 1.0 / d.latent_period;
  const gamma_p = 1.0 / d.presymptomatic_period;
  const gamma_a = 1.0 / d.asymptomatic_infectious_period;
  const gamma_s = 1.0 / d.symptomatic_period;
  const gamma_h = 1.0 / d.hospital_stay;
  const gamma_c = 1.0 / d.icu_stay;
  const omega = d.waning_immunity_days ? 1.0 / d.waning_immunity_days : 0.0;

  const p_asymp = asArray(d.asymptomatic_fraction, n);
  const hosp = asArray(d.hospitalization_rate, n);
  const icu = asArray(d.icu_rate, n);
  const death = asArray(d.death_rate, n);
  const susceptibility = asArray(d.susceptibility, n);

  const veSev = scenario.vaccination.ve_severity;
  const hosp_rate = [hosp.slice(), hosp.map((x) => x * (1.0 - veSev))];

  const rel_p = d.rel_infectiousness_presymptomatic;
  const rel_a = d.rel_infectiousness_asymptomatic;
  const f_transmission = [1.0, 1.0 - scenario.vaccination.ve_transmission];

  const infectious_duration = vec(n);
  for (let i = 0; i < n; i++) {
    infectious_duration[i] =
      p_asymp[i] * rel_a * d.asymptomatic_infectious_period +
      (1.0 - p_asymp[i]) * (rel_p * d.presymptomatic_period + d.symptomatic_period);
  }

  return {
    sigma, gamma_p, gamma_a, gamma_s, gamma_h, gamma_c, omega,
    p_asymp, hosp_rate, icu_rate: icu, death_rate: death,
    rel_p, rel_a, f_transmission, infectious_duration, susceptibility,
  };
}

/* ============================================================
   The epidemic engine
   ============================================================ */
export class EpidemicModel {
  constructor(scenario) {
    this.config = scenario;
    this.n = scenario.population.age_distribution.length;
    this.dt = scenario.simulation.dt;
    this.stochastic = scenario.simulation.stochastic;
    const seed = scenario.simulation.seed;
    this.rng = makeRng(seed == null ? (Math.random() * 4294967296) >>> 0 : seed);

    this.N = populationByAge(scenario.population.total_population, scenario.population.age_distribution)
      .map((x) => x * 1.0);
    this.N_safe = this.N.map((x) => (x > 0 ? x : 1.0));

    this.contact = symmetrize(defaultContactMatrix(this.n), this.N);

    const p = resolveParameters(scenario);
    Object.assign(this, p);

    this.beta = calibrateBeta(this.contact, this.infectious_duration, scenario.disease.r0, this.susceptibility);

    this.t = 0;
    this.cumulativeVaccinated = 0.0;
    this._initState();
  }

  _maybeRound(arr) {
    if (!this.stochastic) return arr;
    return arr.map((x) => Math.round(x));
  }

  _binom(n, p) {
    // n and p are equal-length arrays (or p a scalar applied to each).
    const out = new Array(n.length);
    for (let i = 0; i < n.length; i++) {
      const pi = Array.isArray(p) ? p[i] : p;
      const clamped = pi < 0 ? 0 : pi > 1 ? 1 : pi;
      out[i] = this.stochastic ? sampleBinomial(this.rng, n[i], clamped) : n[i] * clamped;
    }
    return out;
  }

  _largestRemainder(total, weights) {
    const sum = weights.reduce((a, b) => a + b, 0) || 1;
    const w = weights.map((x) => x / sum);
    const raw = w.map((x) => x * total);
    const base = raw.map((x) => Math.floor(x));
    let remainder = total - base.reduce((a, b) => a + b, 0);
    if (remainder > 0) {
      const order = raw.map((x, i) => [x - base[i], i]).sort((a, b) => b[0] - a[0]);
      for (let k = 0; k < remainder; k++) base[order[k][1]] += 1;
    }
    return base.map((x) => x * 1.0);
  }

  _initState() {
    const n = this.n;
    this.S = this.N.slice();
    this.V = vec(n);
    this.E = mat2(n); this.Ip = mat2(n); this.Ia = mat2(n);
    this.Is = mat2(n); this.H = mat2(n); this.C = mat2(n);
    this.R = vec(n); this.D = vec(n);

    const imm = this.config.population.initial_immune_fraction;
    if (imm > 0) {
      const immune = this._maybeRound(this.S.map((s) => s * imm));
      for (let i = 0; i < n; i++) { this.S[i] -= immune[i]; this.R[i] += immune[i]; }
    }

    const seedTotal = this.config.population.initial_infected;
    if (seedTotal > 0) {
      const sSum = this.S.reduce((a, b) => a + b, 0);
      const weights = sSum > 0 ? this.S.slice() : vec(n, 1);
      const seedByAge = this._largestRemainder(seedTotal, weights);
      let asymp = this._maybeRound(seedByAge.map((s, i) => s * this.p_asymp[i]));
      asymp = asymp.map((a, i) => Math.min(a, seedByAge[i]));
      let symp = seedByAge.map((s, i) => s - asymp[i]);
      symp = symp.map((x, i) => Math.min(x, this.S[i]));
      for (let i = 0; i < n; i++) { this.Is[0][i] += symp[i]; this.S[i] -= symp[i]; }
      asymp = asymp.map((a, i) => Math.min(a, this.S[i]));
      for (let i = 0; i < n; i++) { this.Ia[0][i] += asymp[i]; this.S[i] -= asymp[i]; }
    }
  }

  currentDay() { return this.t * this.dt; }

  _interventionMultiplier(day) {
    let m = 1.0;
    for (const iv of this.config.interventions) {
      if (day >= iv.start_day && day < iv.end_day) m *= (1.0 - iv.transmission_reduction);
    }
    return m;
  }

  _seasonalMultiplier(day) {
    const env = this.config.environment;
    if (env.seasonal_amplitude === 0) return 1.0;
    const phase = (2.0 * Math.PI * (day - env.seasonal_peak_day)) / env.seasonal_period_days;
    return 1.0 + env.seasonal_amplitude * Math.cos(phase);
  }

  _externalForce(day) {
    const env = this.config.environment;
    if (env.external_infection_rate === 0) return 0.0;
    return env.external_infection_rate * this._seasonalMultiplier(day);
  }

  betaEffective(day) {
    return this.beta * this._interventionMultiplier(day) * this._seasonalMultiplier(day);
  }

  _overdispersionNoise() {
    const od = this.config.simulation.overdispersion;
    if (od == null || !this.stochastic) return 1.0;
    return sampleGamma(this.rng, od, 1.0 / od);
  }

  effectiveRt(day) {
    const ve = this.config.vaccination.ve_susceptibility;
    const sus = vec(this.n);
    for (let i = 0; i < this.n; i++) {
      sus[i] = (this.S[i] + (1.0 - ve) * this.V[i]) / this.N_safe[i];
    }
    const k0 = ngmUnit(this.contact, this.infectious_duration, this.susceptibility);
    const k = k0.map((row, i) => row.map((x) => sus[i] * x));
    return this.betaEffective(day) * spectralRadius(k);
  }

  _vaccinate(day) {
    const vac = this.config.vaccination;
    if (!vac.enabled || day < vac.start_day) return;
    const totalPop = this.N.reduce((a, b) => a + b, 0);
    const capRemaining = vac.coverage_cap * totalPop - this.cumulativeVaccinated;
    if (capRemaining <= 0) return;
    const sSum = this.S.reduce((a, b) => a + b, 0);
    const doses = Math.min(vac.daily_rate * totalPop, capRemaining, sSum);
    if (doses <= 0) return;

    let alloc = vec(this.n);
    if (vac.prioritize_elderly) {
      let remaining = doses;
      for (let i = this.n - 1; i >= 0; i--) {
        const take = Math.min(remaining, this.S[i]);
        alloc[i] = take;
        remaining -= take;
        if (remaining <= 0) break;
      }
    } else if (sSum > 0) {
      alloc = this.S.map((s) => (doses * s) / sSum);
    }
    alloc = this._maybeRound(alloc.map((a, i) => Math.min(a, this.S[i])));
    for (let i = 0; i < this.n; i++) { this.S[i] -= alloc[i]; this.V[i] += alloc[i]; }
    this.cumulativeVaccinated += alloc.reduce((a, b) => a + b, 0);
  }

  step() {
    const n = this.n;
    const day = this.currentDay();
    this._vaccinate(day);

    const betaEff = this.betaEffective(day);
    const noise = this._overdispersionNoise();

    // Infection pressure and force of infection per age.
    const prevalence = vec(n);
    for (let a = 0; a < n; a++) {
      let pressure = 0;
      for (let s = 0; s < 2; s++) {
        pressure += this.f_transmission[s] * (this.rel_p * this.Ip[s][a] + this.rel_a * this.Ia[s][a] + this.Is[s][a]);
      }
      prevalence[a] = pressure / this.N_safe[a];
    }
    const external = this._externalForce(day);
    const foi = vec(n);
    for (let a = 0; a < n; a++) {
      let internal = 0;
      for (let j = 0; j < n; j++) internal += this.contact[a][j] * prevalence[j];
      internal *= betaEff * noise;
      foi[a] = this.susceptibility[a] * (internal + external);
    }

    const veSus = this.config.vaccination.ve_susceptibility;
    const pInfS = foi.map((f) => transitionProbability(f, this.dt));
    const pInfV = foi.map((f) => transitionProbability(f * (1.0 - veSus), this.dt));
    const newInfS = this._binom(this.S, pInfS);
    const newInfV = this._binom(this.V, pInfV);

    const pE = transitionProbability(this.sigma, this.dt);
    const pIp = transitionProbability(this.gamma_p, this.dt);
    const pIa = transitionProbability(this.gamma_a, this.dt);
    const pIs = transitionProbability(this.gamma_s, this.dt);
    const pH = transitionProbability(this.gamma_h, this.dt);
    const pC = transitionProbability(this.gamma_c, this.dt);

    const cSum = this.C[0].reduce((a, b) => a + b, 0) + this.C[1].reduce((a, b) => a + b, 0);
    const { deathProb, overflow } = icuDeathProbability(cSum, this.death_rate, this.config.healthcare);

    // Per-stratum transitions.
    const leaveE = [this._binom(this.E[0], pE), this._binom(this.E[1], pE)];
    const toIa = [this._binom(leaveE[0], this.p_asymp), this._binom(leaveE[1], this.p_asymp)];
    const toIp = [leaveE[0].map((x, i) => x - toIa[0][i]), leaveE[1].map((x, i) => x - toIa[1][i])];
    const leaveIp = [this._binom(this.Ip[0], pIp), this._binom(this.Ip[1], pIp)];
    const leaveIa = [this._binom(this.Ia[0], pIa), this._binom(this.Ia[1], pIa)];
    const leaveIs = [this._binom(this.Is[0], pIs), this._binom(this.Is[1], pIs)];
    const toH = [this._binom(leaveIs[0], this.hosp_rate[0]), this._binom(leaveIs[1], this.hosp_rate[1])];
    const isToR = [leaveIs[0].map((x, i) => x - toH[0][i]), leaveIs[1].map((x, i) => x - toH[1][i])];
    const leaveH = [this._binom(this.H[0], pH), this._binom(this.H[1], pH)];
    const toC = [this._binom(leaveH[0], this.icu_rate), this._binom(leaveH[1], this.icu_rate)];
    const hToR = [leaveH[0].map((x, i) => x - toC[0][i]), leaveH[1].map((x, i) => x - toC[1][i])];
    const leaveC = [this._binom(this.C[0], pC), this._binom(this.C[1], pC)];
    const toD = [this._binom(leaveC[0], deathProb), this._binom(leaveC[1], deathProb)];
    const cToR = [leaveC[0].map((x, i) => x - toD[0][i]), leaveC[1].map((x, i) => x - toD[1][i])];

    // Apply deltas (all draws used start-of-step values).
    for (let a = 0; a < n; a++) {
      this.S[a] -= newInfS[a];
      this.V[a] -= newInfV[a];
      this.E[0][a] += newInfS[a] - leaveE[0][a];
      this.E[1][a] += newInfV[a] - leaveE[1][a];
      for (let s = 0; s < 2; s++) {
        this.Ip[s][a] += toIp[s][a] - leaveIp[s][a];
        this.Ia[s][a] += toIa[s][a] - leaveIa[s][a];
        this.Is[s][a] += leaveIp[s][a] - leaveIs[s][a];
        this.H[s][a] += toH[s][a] - leaveH[s][a];
        this.C[s][a] += toC[s][a] - leaveC[s][a];
      }
      this.R[a] += isToR[0][a] + isToR[1][a] + leaveIa[0][a] + leaveIa[1][a] +
        hToR[0][a] + hToR[1][a] + cToR[0][a] + cToR[1][a];
      this.D[a] += toD[0][a] + toD[1][a];
    }

    if (this.omega > 0) {
      const waned = this._binom(this.R, transitionProbability(this.omega, this.dt));
      for (let a = 0; a < n; a++) { this.R[a] -= waned[a]; this.S[a] += waned[a]; }
    }

    this._clipNegatives();
    this.t += 1;

    const sum2 = (m) => m[0].reduce((a, b) => a + b, 0) + m[1].reduce((a, b) => a + b, 0);
    const sum1 = (v) => v.reduce((a, b) => a + b, 0);
    const newSym = sum2(leaveIp);

    return {
      day,
      S: sum1(this.S), V: sum1(this.V), E: sum2(this.E),
      Ip: sum2(this.Ip), Ia: sum2(this.Ia), Is: sum2(this.Is),
      H: sum2(this.H), C: sum2(this.C), R: sum1(this.R), D: sum1(this.D),
      new_infections: sum1(newInfS) + sum1(newInfV),
      new_symptomatic: newSym,
      new_hospitalizations: sum2(toH),
      new_icu: sum2(toC),
      new_deaths: sum2(toD),
      rt: this.effectiveRt(day),
      beta_effective: betaEff,
      icu_overflow: overflow,
    };
  }

  _clipNegatives() {
    const clip1 = (v) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; };
    [this.S, this.V, this.R, this.D].forEach(clip1);
    [this.E, this.Ip, this.Ia, this.Is, this.H, this.C].forEach((m) => { clip1(m[0]); clip1(m[1]); });
  }
}

/* ============================================================
   Driver + metrics
   ============================================================ */
export function runSimulation(scenario) {
  const model = new EpidemicModel(scenario);
  const steps = Math.ceil(scenario.simulation.duration_days / scenario.simulation.dt);
  const history = [];
  for (let i = 0; i < steps; i++) history.push(model.step());
  return history;
}

export function summarize(history, r0) {
  if (!history.length) return null;
  const last = history[history.length - 1];
  const totalPop = last.S + last.V + last.E + last.Ip + last.Ia + last.Is + last.H + last.C + last.R + last.D;

  const peak = (key) => {
    let best = 0, day = null;
    for (const r of history) if (r[key] > best) { best = r[key]; day = r.day; }
    return { value: best, day };
  };
  const infectiousPeak = (() => {
    let best = 0, day = null;
    for (const r of history) {
      const v = r.Ip + r.Ia + r.Is;
      if (v > best) { best = v; day = r.day; }
    }
    return { value: best, day };
  })();

  const totalInfections = history.reduce((a, r) => a + r.new_infections, 0);
  const totalDeaths = last.D;
  const totalHosp = history.reduce((a, r) => a + r.new_hospitalizations, 0);
  const totalIcu = history.reduce((a, r) => a + r.new_icu, 0);

  let rtCrossOne = null;
  for (const r of history) {
    if (r.day > 0 && r.rt < 1 && (r.Ip + r.Ia + r.Is) > 0) { rtCrossOne = r.day; break; }
  }
  let peakRt = 0;
  for (const r of history) if (r.rt > peakRt) peakRt = r.rt;

  return {
    total_population: totalPop,
    total_infections: totalInfections,
    attack_rate: totalPop > 0 ? totalInfections / totalPop : 0,
    total_hospitalizations: totalHosp,
    total_icu: totalIcu,
    total_deaths: totalDeaths,
    infection_fatality_ratio: totalInfections > 0 ? totalDeaths / totalInfections : 0,
    peak_infectious: infectiousPeak.value,
    peak_infectious_day: infectiousPeak.day,
    peak_hospital_occupancy: peak("H").value,
    peak_hospital_day: peak("H").day,
    peak_icu_occupancy: peak("C").value,
    peak_icu_day: peak("C").day,
    peak_daily_infections: peak("new_infections").value,
    peak_daily_infections_day: peak("new_infections").day,
    r0,
    peak_rt: peakRt,
    rt_crossed_one_day: rtCrossOne,
  };
}
