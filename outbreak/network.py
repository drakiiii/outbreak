"""Contact-network construction for the agent engine.

The agent engine can mix people purely within age groups (mean field) or through
explicit, repeated-contact *settings*. This module builds those settings and
exposes, for each one, the two things the engine needs:

1. a per-agent ``group_id`` array (which household / class / workplace each agent
   belongs to, or ``-1`` if it is not in that layer) — used by the stochastic
   step to compute who-infects-whom within shared settings; and
2. an age-mixing matrix derived from the *actual* constructed network — used to
   calibrate the global transmission rate so the whole multilayer network still
   reproduces the target R0.

A "layer" here is a partition of (some) agents into disjoint groups that mix
fully within each group:

* **households** — every agent is in exactly one; small; *density-dependent*
  (you effectively contact all housemates regardless of household size);
* **schools** — only school-age agents; large; *frequency-dependent* (per-person
  contacts don't grow without bound as the school grows);
* **workplaces** — only working-age agents; large; frequency-dependent.

Why the age-mixing matrix matters
---------------------------------
For calibration the engine treats each layer like an extra contact matrix. If a
layer puts an age-``i`` person in groups whose other members are, on average,
``M[i, j]`` people of age ``j`` (after the layer's size scaling), then in the
mean-field limit that layer contributes transmission exactly as a contact matrix
``M`` would. So we can sum the community contact matrix and every layer's ``M``
into one effective matrix and calibrate the global beta on its dominant
eigenvalue — the same next-generation-matrix machinery used everywhere else.
:func:`layer_mixing_matrix` computes ``M`` for a constructed layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import NetworkConfig


@dataclass
class ContactLayer:
    """One constructed contact layer (households, a school system, ...).

    ``group_id[a]`` is the group index of agent ``a`` within this layer, or ``-1``
    if the agent does not participate (e.g. an adult has no school group).
    ``density_dependent`` selects how within-group force of infection scales with
    group size (see module docstring).
    """

    name: str
    group_id: np.ndarray         # (n_agents,) int, -1 == not a member
    weight: float
    density_dependent: bool

    def group_scaling(self) -> np.ndarray:
        """Per-group divisor applied to within-group infectious load.

        Density-dependent layers (households) use 1 — every infectious member
        contributes fully. Frequency-dependent layers (schools/workplaces) divide
        by ``size - 1`` so a susceptible's expected exposure does not grow with
        the size of the setting (bounded per-capita contact rate).
        """
        sizes = np.bincount(self.group_id[self.group_id >= 0]) if np.any(self.group_id >= 0) \
            else np.zeros(0)
        if self.density_dependent:
            return np.ones(sizes.size)
        return np.maximum(sizes - 1.0, 1.0)


def build_layers(config: NetworkConfig, age: np.ndarray, rng: np.random.Generator,
                 n_age: int) -> List[ContactLayer]:
    """Construct all enabled contact layers for a population of agents.

    ``age`` is the per-agent age-group array. Draws from ``rng`` (so the network
    is reproducible from the engine's seed and must be persisted with a snapshot).
    """
    n = age.size
    layers: List[ContactLayer] = []

    # Households: partition *all* agents into small groups by sampling sizes from
    # the configured distribution.
    hh = _household_ids(n, config.household_size_distribution, rng)
    layers.append(ContactLayer("household", hh, config.household_weight, density_dependent=True))

    # Schools: only school-age agents; the rest get -1.
    school_ages = config.resolved_school_groups(n_age)
    school = _setting_ids(age, school_ages, config.mean_school_size, rng)
    layers.append(ContactLayer("school", school, config.school_weight, density_dependent=False))

    # Workplaces: only working-age agents.
    work_ages = config.resolved_work_groups(n_age)
    work = _setting_ids(age, work_ages, config.mean_workplace_size, rng)
    layers.append(ContactLayer("workplace", work, config.workplace_weight, density_dependent=False))

    return layers


def _household_ids(n: int, size_dist, rng: np.random.Generator) -> np.ndarray:
    """Assign each of ``n`` agents to a household by sampling household sizes."""
    probs = np.asarray(size_dist, dtype=float)
    probs = probs / probs.sum()
    sizes_values = np.arange(1, probs.size + 1)        # household sizes 1..K

    # Sample more sizes than we could possibly need (mean size >= 1, so n draws
    # is always enough), then take the prefix whose cumulative size first reaches n.
    draw = rng.choice(sizes_values, size=n, p=probs)
    cumulative = np.cumsum(draw)
    n_households = int(np.searchsorted(cumulative, n, side="left") + 1)
    sizes = draw[:n_households].copy()
    # Trim the final household so the sizes sum to exactly n.
    overshoot = int(sizes.sum() - n)
    sizes[-1] -= overshoot

    # Expand [s0, s1, ...] into per-agent household ids [0]*s0 + [1]*s1 + ...,
    # then shuffle so household membership is not correlated with agent order
    # (agents are laid out by age, which we don't want to leak into households).
    ids = np.repeat(np.arange(n_households), sizes)
    rng.shuffle(ids)
    return ids.astype(np.int64)


def _setting_ids(age: np.ndarray, eligible_ages, mean_size: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Partition agents whose age is in ``eligible_ages`` into ~``mean_size`` groups.

    Non-eligible agents receive ``-1`` (not a member of this layer).
    """
    n = age.size
    group_id = np.full(n, -1, dtype=np.int64)
    eligible = np.isin(age, list(eligible_ages))
    elig_idx = np.where(eligible)[0]
    if elig_idx.size == 0:
        return group_id

    n_groups = max(1, int(round(elig_idx.size / mean_size)))
    # Shuffle eligible agents, then split into n_groups near-equal contiguous
    # chunks; chunk index becomes the group id.
    shuffled = rng.permutation(elig_idx)
    chunks = np.array_split(shuffled, n_groups)
    for g, chunk in enumerate(chunks):
        group_id[chunk] = g
    return group_id


def layer_mixing_matrix(layer: ContactLayer, age: np.ndarray, n_age: int,
                        n_by_age: np.ndarray) -> np.ndarray:
    """Age-mixing matrix ``M`` induced by a constructed layer.

    ``M[i, j]`` is the (size-scaled) expected number of layer co-members of age
    ``j`` that an age-``i`` member has. This plays the exact role of a contact
    matrix in the next-generation matrix, so the engine can fold it into the R0
    calibration. Returns a zero matrix if the layer has no members.
    """
    gid = layer.group_id
    member = gid >= 0
    if not np.any(member):
        return np.zeros((n_age, n_age))

    g = gid[member]
    a = age[member]
    n_groups = int(g.max()) + 1

    # comp[group, age] = number of members of each age in each group. add.at does
    # an unbuffered scatter-add, so repeated (group, age) pairs accumulate.
    comp = np.zeros((n_groups, n_age))
    np.add.at(comp, (g, a), 1.0)

    sizes = comp.sum(axis=1)                              # (n_groups,)
    if layer.density_dependent:
        scaling = np.ones(n_groups)
    else:
        scaling = np.maximum(sizes - 1.0, 1.0)
    weighted = comp / scaling[:, None]                   # divide each group's row

    # cross[i, j] = sum over groups of comp[g,i] * comp[g,j] / scaling[g].
    # comp.T @ weighted contracts the group axis, leaving an (age, age) matrix.
    cross = comp.T @ weighted
    # Remove self-pairs (a member is not its own contact): subtract, on the
    # diagonal, sum_g comp[g,i] / scaling[g].
    self_pairs = (comp / scaling[:, None]).sum(axis=0)
    cross[np.diag_indices(n_age)] -= self_pairs

    # Normalise by the number of agents of each age to get per-person contacts.
    # Rows for absent ages stay zero (guard divide-by-zero).
    denom = np.where(n_by_age > 0, n_by_age, 1.0)
    return cross / denom[:, None]
