"""
Financial metrics: returns, risk, and risk-adjusted performance.

Pure NumPy and deterministic. Conventions follow standard quant practice;
read the docstrings for the assumptions behind each number.
"""

from __future__ import annotations

import numpy as np


def simple_returns(prices):
    """Period-over-period simple returns: p[t]/p[t-1] - 1."""
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def log_returns(prices):
    """Continuously compounded returns: log(p[t]/p[t-1])."""
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def volatility(returns, periods: int = 252) -> float:
    """Annualized standard deviation of returns."""
    r = np.asarray(returns, dtype=float)
    return float(r.std(ddof=1) * np.sqrt(periods)) if r.size > 1 else 0.0


def sharpe_ratio(returns, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio. `risk_free` is an annual rate."""
    r = np.asarray(returns, dtype=float) - risk_free / periods
    sd = r.std(ddof=1)
    return float(np.sqrt(periods) * r.mean() / sd) if sd > 0 else 0.0


def equity_curve(returns, initial: float = 1.0):
    """Cumulative wealth from a return stream."""
    r = np.asarray(returns, dtype=float)
    return initial * np.cumprod(1.0 + r)


def max_drawdown(equity) -> float:
    """Largest peak-to-trough decline (a non-positive number)."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float(np.min((e - peak) / peak))
