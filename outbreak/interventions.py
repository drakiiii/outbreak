"""Convenience builders for common non-pharmaceutical interventions (NPIs).

These are thin wrappers around :class:`outbreak.config.Intervention` that name
typical real-world measures with plausible default strengths. They make
scenarios more readable, e.g.::

    from outbreak.interventions import lockdown, mask_mandate
    scenario.interventions = InterventionConfig([
        mask_mandate(start_day=30, end_day=120),
        lockdown(start_day=45, end_day=75),
    ])

The transmission-reduction figures are illustrative; adjust them to your
context.
"""

from __future__ import annotations

from .config import Intervention


def mask_mandate(start_day: int, end_day: int, reduction: float = 0.25) -> Intervention:
    """Population-wide masking. Modest, broad reduction in transmission."""
    # Positional args map to Intervention's fields (name, start_day, end_day,
    # transmission_reduction); .validate() returns the same instance (so it can
    # be chained), raising if any value is out of range. All builders below
    # follow this identical pattern with different name/default-reduction.
    return Intervention("mask_mandate", start_day, end_day, reduction).validate()


def social_distancing(start_day: int, end_day: int, reduction: float = 0.4) -> Intervention:
    """Voluntary distancing and reduced gatherings."""
    return Intervention("social_distancing", start_day, end_day, reduction).validate()


def school_closure(start_day: int, end_day: int, reduction: float = 0.2) -> Intervention:
    """School closures. Removes a slice of high-contact mixing."""
    return Intervention("school_closure", start_day, end_day, reduction).validate()


def lockdown(start_day: int, end_day: int, reduction: float = 0.7) -> Intervention:
    """Stay-at-home order. Strong, broad reduction in contacts."""
    return Intervention("lockdown", start_day, end_day, reduction).validate()


def test_trace_isolate(start_day: int, end_day: int, reduction: float = 0.3) -> Intervention:
    """Testing, contact tracing and isolation of cases."""
    return Intervention("test_trace_isolate", start_day, end_day, reduction).validate()
