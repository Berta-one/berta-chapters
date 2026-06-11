"""Moving-average crossover strategy and a look-ahead-safe backtester."""

from __future__ import annotations

import numpy as np

from metrics import simple_returns, equity_curve, max_drawdown, sharpe_ratio


def rolling_mean(a, window: int):
    """Trailing simple moving average; leading values are NaN."""
    a = np.asarray(a, dtype=float)
    out = np.full(len(a), np.nan)
    if window <= len(a):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def ma_crossover_signal(prices, fast: int = 3, slow: int = 5):
    """1 = long, 0 = flat. Signal at bar t uses only data up to bar t."""
    f = rolling_mean(prices, fast)
    s = rolling_mean(prices, slow)
    sig = np.where(f > s, 1, 0)
    sig[np.isnan(s)] = 0
    return sig


def backtest(prices, signal, cost_per_trade: float = 0.0):
    """Apply yesterday's signal to today's return (no look-ahead). Returns dict."""
    prices = np.asarray(prices, dtype=float)
    rets = simple_returns(prices)
    pos = np.asarray(signal, dtype=float)[:-1]  # lag: position held into next bar
    strat = rets * pos
    if cost_per_trade:
        trades = np.abs(np.diff(np.concatenate([[0.0], pos])))
        strat = strat - trades * cost_per_trade
    eq = equity_curve(strat)
    return {
        "equity": eq,
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": sharpe_ratio(strat),
        "max_drawdown": max_drawdown(eq),
    }
