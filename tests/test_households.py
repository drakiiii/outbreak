"""Tests for age-structured households (network engine).

By default each household is seeded with an adult, so children live with adults
(inter-generational mixing) rather than being grouped at random. This must not
break household sizes/coverage or the R0 calibration.
"""

import numpy as np
import pytest

from outbreak import preset_scenario
from outbreak.agents import AgentModel
from outbreak.config import NetworkConfig, SimulationConfig


def _model(age_structured, seed=0):
    sc = preset_scenario("covid_like", total_population=200_000)
    sc.network = NetworkConfig(enabled=True, age_structured_households=age_structured)
    sc.simulation = SimulationConfig(engine="agent", n_agents=60_000, seed=seed)
    return AgentModel(sc.validate())


def _fraction_children_with_adult(model):
    hh, age = model.layers[0].group_id, model.age
    n_groups = hh.max() + 1
    has_adult = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(has_adult, hh, age >= 1)          # ages 1+ are adults
    child_idx = np.where(age == 0)[0]
    return has_adult[hh[child_idx]].mean()


def test_age_structured_puts_every_child_with_an_adult():
    model = _model(age_structured=True)
    assert _fraction_children_with_adult(model) == pytest.approx(1.0)


def test_random_leaves_some_children_without_adults():
    # Purely random grouping strands a meaningful share of children in
    # adult-free households; the age-structured version fixes that.
    structured = _fraction_children_with_adult(_model(True))
    random = _fraction_children_with_adult(_model(False))
    assert random < structured
    assert random < 0.95


def test_households_still_partition_everyone():
    for structured in (True, False):
        model = _model(structured)
        hh = model.layers[0].group_id
        assert np.all(hh >= 0)                          # everyone in a household
        assert np.bincount(hh).sum() == model.n_agents  # sizes sum to the population


def test_household_size_distribution_preserved():
    # Age structuring rearranges *who* is in each household, not the sizes.
    model = _model(age_structured=True)
    sizes = np.bincount(model.layers[0].group_id)
    assert sizes.min() >= 1
    assert sizes.mean() == pytest.approx(2.4, abs=0.4)  # ~ the configured distribution


def test_r0_calibration_holds_for_age_structured_households():
    assert _model(age_structured=True).effective_rt(0.0) == pytest.approx(2.8, rel=3e-2)


def test_age_structured_is_reproducible():
    a = _model(True, seed=5).layers[0].group_id
    b = _model(True, seed=5).layers[0].group_id
    assert np.array_equal(a, b)
