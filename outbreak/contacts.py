"""Age structure and contact matrices.

A contact matrix ``C`` has ``C[i, j]`` = the mean number of contacts a single
individual in age group ``i`` has with individuals in age group ``j`` per day.
Realistic matrices are *assortative* (people mix mostly within their own age
band) with strong inter-generational mixing between children and adults.

The default matrices here are POLYMOD-*style* illustrative values: they capture
the qualitative structure (school-age clustering, working-age mixing, lower
elderly contacts) without claiming to reproduce a specific empirical study.

Symmetry of total contacts
---------------------------
Physical contacts are reciprocal: the total number of contacts between groups i
and j must match from both directions, i.e. ``C[i, j] * N_i == C[j, i] * N_j``.
:func:`symmetrize` enforces this for a given population so the transmission
model is internally consistent.
"""

from __future__ import annotations

import numpy as np


# A 4x4 illustrative contact matrix matching DEFAULT_AGE_LABELS
# ("0-17", "18-49", "50-64", "65+"). Rows = ego age group, columns = contact
# age group, units = contacts/day. Values are illustrative.
DEFAULT_CONTACT_MATRIX_4 = np.array(
    [
        [9.0, 4.0, 1.2, 0.6],   # children: heavy within-group (school), parents
        [3.0, 8.0, 2.5, 1.0],   # young adults: work + family
        [1.2, 3.5, 4.0, 1.5],   # older adults
        [0.6, 1.5, 1.5, 2.5],   # elderly: fewer contacts overall
    ],
    dtype=float,
)


def default_contact_matrix(n_age: int) -> np.ndarray:
    """Return a plausible default contact matrix for ``n_age`` age groups.

    * ``n_age == 1``: a single homogeneous-mixing group. The absolute value is
      arbitrary because the transmission rate is calibrated to R0 downstream.
    * ``n_age == 4``: the illustrative POLYMOD-style matrix above.
    * otherwise: a synthetic assortative matrix (stronger diagonal, decaying
      off-diagonal mixing) so any age resolution still produces sensible mixing.
    """
    if n_age == 1:
        return np.array([[10.0]], dtype=float)
    if n_age == 4:
        return DEFAULT_CONTACT_MATRIX_4.copy()

    # Synthetic assortative matrix for arbitrary resolutions.
    idx = np.arange(n_age)
    distance = np.abs(idx[:, None] - idx[None, :])
    base = 8.0 * np.exp(-distance / 1.5)          # within-group mixing decays
    base += 1.0                                   # baseline community mixing
    # Slightly reduce contacts for the oldest groups.
    age_scale = np.linspace(1.0, 0.6, n_age)
    base *= age_scale[:, None]
    return base


def symmetrize(contact_matrix: np.ndarray, population_by_age: np.ndarray) -> np.ndarray:
    """Enforce reciprocity of total contacts for a given population.

    Returns a matrix ``C`` satisfying ``C[i, j] * N_i == C[j, i] * N_j`` while
    preserving the overall scale of mixing. Groups with zero population are left
    untouched (they carry no contacts).
    """
    C = np.asarray(contact_matrix, dtype=float)
    N = np.asarray(population_by_age, dtype=float)
    n = C.shape[0]
    if C.shape != (n, n):
        raise ValueError("contact_matrix must be square")
    if N.shape != (n,):
        raise ValueError("population_by_age length must match contact matrix")

    # Total contacts from i to j and from j to i; average them.
    total_ij = C * N[:, None]                     # contacts i->j weighted by N_i
    total = 0.5 * (total_ij + total_ij.T)         # symmetric total contacts
    with np.errstate(divide="ignore", invalid="ignore"):
        C_sym = np.where(N[:, None] > 0, total / N[:, None], 0.0)
    return C_sym
