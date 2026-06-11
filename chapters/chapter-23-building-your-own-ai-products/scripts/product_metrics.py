"""
Product analytics math: prioritization, unit economics, retention, funnels.

Pure NumPy + stdlib. These are the numbers that decide whether an AI feature
is worth building and whether the product around it is a business.
"""

from __future__ import annotations

import numpy as np


def rice_score(reach: float, impact: float, confidence: float, effort: float) -> float:
    """RICE = Reach * Impact * Confidence / Effort. Higher is better."""
    return reach * impact * confidence / max(effort, 1e-9)


def rank_backlog(features: list[dict]) -> list[tuple[str, float]]:
    """Sort features (each a dict with name + RICE inputs) by score, descending."""
    scored = [
        (f["name"], rice_score(f["reach"], f["impact"], f["confidence"], f["effort"]))
        for f in features
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def ltv(arpu: float, gross_margin: float, monthly_churn: float) -> float:
    """Lifetime value = ARPU * margin / churn (geometric-series lifetime)."""
    return arpu * gross_margin / monthly_churn if monthly_churn > 0 else float("inf")


def cac_payback_months(cac: float, arpu: float, gross_margin: float) -> float:
    """Months of gross margin to recover customer acquisition cost."""
    monthly_margin = arpu * gross_margin
    return cac / monthly_margin if monthly_margin > 0 else float("inf")


def ltv_cac_ratio(ltv_value: float, cac: float) -> float:
    return ltv_value / cac if cac > 0 else float("inf")


def retention_curve(monthly_churn: float, months: int, initial: float = 1.0):
    """Fraction of a cohort retained over time under constant churn."""
    return np.array([initial * ((1.0 - monthly_churn) ** m) for m in range(months)])


def funnel_conversion(counts: list[float]):
    """Return (overall_conversion, per_step_rates) for an ordered funnel."""
    counts = list(counts)
    overall = counts[-1] / counts[0] if counts and counts[0] else 0.0
    steps = [
        counts[i + 1] / counts[i] if counts[i] else 0.0
        for i in range(len(counts) - 1)
    ]
    return overall, steps


def biggest_leak(stage_names: list[str], counts: list[float]) -> str:
    """Name the funnel step with the lowest conversion (the biggest leak)."""
    _, steps = funnel_conversion(counts)
    if not steps:
        return ""
    worst = int(np.argmin(steps))
    return f"{stage_names[worst]} -> {stage_names[worst + 1]}"
