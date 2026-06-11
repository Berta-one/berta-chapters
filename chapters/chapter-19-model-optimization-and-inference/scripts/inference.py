"""Inference cost model: latency vs throughput under batching, plus a profiler."""

from __future__ import annotations

import time


def batch_latency(batch_size: int, fixed_ms: float = 5.0, per_item_ms: float = 1.0) -> float:
    """Total latency for a batch: fixed overhead + per-item cost."""
    return fixed_ms + per_item_ms * batch_size


def throughput(batch_size: int, fixed_ms: float = 5.0, per_item_ms: float = 1.0) -> float:
    """Items per second for a given batch size (higher batch amortizes overhead)."""
    latency_s = batch_latency(batch_size, fixed_ms, per_item_ms) / 1000.0
    return batch_size / latency_s


def profile(fn, *args, repeats: int = 3, **kwargs):
    """Return (result, best_seconds) over a few runs."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return result, best
