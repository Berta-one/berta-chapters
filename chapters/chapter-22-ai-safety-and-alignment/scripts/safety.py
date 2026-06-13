"""
Measurable-safety primitives: content filtering, reward-model evaluation,
and a red-team harness. Pure stdlib + NumPy and deterministic.

These are teaching tools. Production safety uses trained classifiers and
human review, but the *measurement discipline* shown here is identical.
"""

from __future__ import annotations

import re

import numpy as np


def content_filter(text: str, banned: list[str] | None = None) -> dict:
    """Block text containing any banned pattern. Returns blocked flag + matches."""
    from config import BANNED_PATTERNS

    patterns = banned if banned is not None else BANNED_PATTERNS
    low = text.lower()
    matches = [p for p in patterns if re.search(re.escape(p), low)]
    return {"blocked": bool(matches), "matches": matches}


def reward_model_accuracy(chosen_scores, rejected_scores) -> float:
    """Fraction of pairs where the reward model prefers the chosen response."""
    c = np.asarray(chosen_scores, dtype=float)
    r = np.asarray(rejected_scores, dtype=float)
    return float(np.mean(c > r))


def red_team(model_fn, prompts, judge) -> dict:
    """Run adversarial prompts through a model; tally unsafe outputs by a judge."""
    records = []
    unsafe = 0
    for p in prompts:
        out = model_fn(p)
        bad = bool(judge(out))
        unsafe += bad
        records.append({"prompt": p, "output": out, "unsafe": bad})
    return {"records": records, "unsafe_rate": unsafe / len(prompts) if prompts else 0.0}
