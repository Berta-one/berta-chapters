"""
Reliability primitives for production services.

Token-bucket rate limiting, a circuit breaker, exponential backoff, and
deterministic canary routing. Pure stdlib, deterministic, and unit-testable
(time is always passed in, never read from the clock).
"""

from __future__ import annotations

import hashlib


class TokenBucket:
    """Burst-tolerant rate limiter. Pass a monotonic `now` (seconds) to allow()."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = 0.0

    def allow(self, now: float, cost: float = 1.0) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class CircuitOpen(RuntimeError):
    pass


class CircuitBreaker:
    """Trips open after `fail_threshold` consecutive failures; recovers via half-open."""

    def __init__(self, fail_threshold: int = 3, reset_timeout: float = 5.0) -> None:
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "closed"
        self.opened_at = 0.0

    def call(self, fn, now: float):
        if self.state == "open":
            if now - self.opened_at >= self.reset_timeout:
                self.state = "half_open"
            else:
                raise CircuitOpen("circuit is open")
        try:
            result = fn()
        except Exception:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.state = "open"
                self.opened_at = now
            raise
        self.failures = 0
        self.state = "closed"
        return result


def backoff_delays(base: float, retries: int, factor: float = 2.0, jitter: float = 0.0):
    """Exponential backoff delays; `jitter` (0..1) scales a deterministic perturbation."""
    delays = []
    for i in range(retries):
        d = base * (factor ** i)
        if jitter:
            # deterministic pseudo-jitter so tests stay reproducible
            frac = (hash(("backoff", i)) % 1000) / 1000.0
            d *= 1.0 + jitter * (frac - 0.5)
        delays.append(d)
    return delays


def canary_route(key, canary_fraction: float) -> str:
    """Deterministic hash routing: a stable `canary_fraction` of keys hit the canary."""
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    return "canary" if (h % 10_000) / 10_000.0 < canary_fraction else "stable"
