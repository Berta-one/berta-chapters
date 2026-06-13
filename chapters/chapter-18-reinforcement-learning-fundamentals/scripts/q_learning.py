"""Tabular Q-learning with epsilon-greedy exploration."""

from __future__ import annotations

import numpy as np


def train(env, episodes=500, alpha=0.5, gamma=0.95, epsilon=0.1, max_steps=100, seed=18):
    """Train a Q-table on a gridworld-style env. Returns (Q, returns)."""
    rng = np.random.default_rng(seed)
    Q: dict = {}

    def q(s):
        return Q.setdefault(s, np.zeros(4))

    returns = []
    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            if rng.random() < epsilon:
                a = int(rng.integers(4))
            else:
                a = int(np.argmax(q(s)))
            ns, r, done = env.step(a)
            total += r
            target = r + (0.0 if done else gamma * np.max(q(ns)))
            q(s)[a] += alpha * (target - q(s)[a])
            s = ns
            if done:
                break
        returns.append(total)
    return Q, returns


def greedy_policy(Q: dict) -> dict:
    return {s: int(np.argmax(qv)) for s, qv in Q.items()}


def run_policy(env, policy: dict, max_steps=100) -> int:
    """Follow a greedy policy; return number of steps to the goal (or max_steps)."""
    s = env.reset()
    for t in range(1, max_steps + 1):
        a = policy.get(s, 0)
        s, _, done = env.step(a)
        if done:
            return t
    return max_steps
