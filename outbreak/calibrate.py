"""Fit model parameters to observed case data.

Turns Outbreak from a "what if?" explorer into a tool that can be **matched to a
real outbreak**: given a series of observed daily cases, it finds the basic
reproduction number R0 (and, optionally, an observation/ascertainment scale) that
best reproduces the data.

Method
------
Fitting uses the fast **deterministic compartmental** engine (smooth, no Monte
Carlo noise) and compares the model's daily symptomatic incidence to the observed
counts under a **Poisson** likelihood — the natural choice for case counts.

The observation scale is *profiled out*: for any candidate R0, the
likelihood-maximising scale has the closed form ``Σobserved / Σmodel``, so only a
single one-dimensional search over R0 remains (robust and fast, via SciPy's
bounded ``minimize_scalar``).

Assumptions / limitations
-------------------------
* ``observed[t]`` is assumed to line up with model day ``t`` (same start). Align
  or trim your series first if it doesn't.
* It fits R0 and a scalar reporting factor; everything else (durations, severity,
  age structure, ...) is taken from the supplied base scenario. Set those to your
  best estimates before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .config import ScenarioConfig, SimulationConfig
from .model import EpidemicModel


@dataclass
class FitResult:
    """Outcome of fitting a scenario to observed cases."""

    r0: float                     # best-fit basic reproduction number
    scale: float                  # best-fit observation scale (observed / true symptomatic)
    loss: float                   # Poisson negative log-likelihood at the optimum
    observed: np.ndarray          # the data that was fitted
    predicted: np.ndarray         # model's fitted observed-case curve (same length)
    success: bool                 # did the optimiser converge?
    scenario: ScenarioConfig      # a copy of the base scenario with the fitted R0

    def summary(self) -> str:
        return (f"fitted R0 = {self.r0:.3f}, observation scale = {self.scale:.3f} "
                f"(Poisson NLL {self.loss:,.1f}, {'converged' if self.success else 'did not converge'})")


def _eval_scenario(base: ScenarioConfig, r0: float, n_days: int) -> ScenarioConfig:
    """A deterministic compartmental copy of ``base`` with the given R0/duration."""
    # Round-trip through dict to deep-copy, then override the fit-relevant fields.
    sc = ScenarioConfig.from_dict(base.to_dict())
    sc.disease.r0 = float(r0)
    sc.simulation = SimulationConfig(
        duration_days=int(n_days), dt=base.simulation.dt,
        engine="compartmental", stochastic=False, overdispersion=None,
    )
    return sc.validate()


def _model_symptomatic(base: ScenarioConfig, r0: float, n_days: int) -> np.ndarray:
    """Daily symptomatic incidence from a deterministic run at this R0."""
    model = EpidemicModel(_eval_scenario(base, r0, n_days))
    return np.array([model.step().new_symptomatic for _ in range(n_days)])


def _poisson_nll(predicted: np.ndarray, observed: np.ndarray) -> float:
    # Poisson negative log-likelihood up to a constant: Σ (λ - y·log λ).
    lam = np.maximum(predicted, 1e-9)
    return float(np.sum(lam - observed * np.log(lam)))


def fit_to_incidence(observed, base_scenario: ScenarioConfig,
                     fit_scale: bool = True,
                     r0_bounds=(0.5, 8.0)) -> FitResult:
    """Fit R0 (and optionally an observation scale) to a daily case series.

    ``observed`` is a 1-D array of daily case counts aligned to model day 0. When
    ``fit_scale`` is True the cases are treated as a (constant) fraction of true
    symptomatic incidence and that fraction is fitted too; when False they are
    treated as true symptomatic counts.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed must be a non-empty 1-D sequence of daily counts")
    if np.any(observed < 0):
        raise ValueError("observed counts must be non-negative")
    n_days = observed.size
    total_obs = observed.sum()

    def evaluate(r0: float):
        model = _model_symptomatic(base_scenario, r0, n_days)
        model_sum = model.sum()
        if fit_scale:
            # Closed-form Poisson-optimal scale for this R0.
            scale = total_obs / model_sum if model_sum > 0 else 1.0
        else:
            scale = 1.0
        predicted = scale * model
        return scale, predicted, _poisson_nll(predicted, observed)

    result = minimize_scalar(lambda r0: evaluate(r0)[2],
                             bounds=r0_bounds, method="bounded")
    scale, predicted, loss = evaluate(result.x)

    fitted = ScenarioConfig.from_dict(base_scenario.to_dict())
    fitted.disease.r0 = float(result.x)
    fitted.validate()

    return FitResult(
        r0=float(result.x), scale=float(scale), loss=float(loss),
        observed=observed, predicted=predicted,
        success=bool(result.success), scenario=fitted,
    )
