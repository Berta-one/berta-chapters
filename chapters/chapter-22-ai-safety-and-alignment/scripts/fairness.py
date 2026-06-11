"""Group fairness metrics: demographic parity and disparate impact."""

from __future__ import annotations

import numpy as np


def positive_rates(y_pred, group) -> dict:
    """Positive-prediction rate per group."""
    y = np.asarray(y_pred, dtype=float)
    g = np.asarray(group)
    return {val: float(y[g == val].mean()) for val in np.unique(g)}


def demographic_parity_difference(y_pred, group) -> float:
    """Max minus min positive rate across groups (0 = perfect parity)."""
    rates = list(positive_rates(y_pred, group).values())
    return float(max(rates) - min(rates))


def disparate_impact_ratio(y_pred, group) -> float:
    """Min/Max positive rate. Below 0.8 fails the four-fifths rule."""
    rates = list(positive_rates(y_pred, group).values())
    hi = max(rates)
    return float(min(rates) / hi) if hi > 0 else 0.0
