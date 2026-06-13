"""Structured logging and golden-signal aggregation."""

from __future__ import annotations

import json
from collections import deque


class StructuredLogger:
    """Collects JSON log events in memory (stand-in for a log shipper)."""

    def __init__(self, capacity: int = 1000) -> None:
        self.events: deque = deque(maxlen=capacity)

    def log(self, **fields) -> str:
        line = json.dumps(fields, sort_keys=True)
        self.events.append(fields)
        return line


def percentile(values, p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    import math

    rank = max(1, math.ceil(p / 100.0 * len(s)))
    return float(s[rank - 1])


def golden_signals(latencies_ms, errors, total):
    """Latency p50/p99, error rate, and a simple saturation proxy."""
    return {
        "p50_ms": percentile(latencies_ms, 50),
        "p99_ms": percentile(latencies_ms, 99),
        "error_rate": errors / total if total else 0.0,
        "requests": total,
    }
