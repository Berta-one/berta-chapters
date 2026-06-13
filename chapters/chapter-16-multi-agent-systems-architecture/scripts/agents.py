"""
Multi-agent coordination primitives.

Deterministic, dependency-light building blocks: a message type, a shared
blackboard, agents that bid for work, the Contract-Net Protocol for task
allocation, and gossip averaging for distributed consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Message:
    sender: str
    recipient: str
    performative: str  # inform | request | propose | accept | reject
    content: dict[str, Any] = field(default_factory=dict)


class Blackboard:
    """A shared key-value store agents read from and write to."""

    def __init__(self) -> None:
        self.facts: dict[str, Any] = {}
        self.writes = 0

    def write(self, key: str, value: Any) -> None:
        self.facts[key] = value
        self.writes += 1

    def read(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)


@dataclass
class Agent:
    """An agent with a skill level and a per-unit cost."""

    name: str
    skill: float
    cost: float

    def bid(self, difficulty: float) -> float:
        """Lower is better: cheaper and more skilled agents bid lower."""
        return self.cost * difficulty / max(self.skill, 1e-6)


def contract_net(tasks: list[dict], agents: list[Agent]) -> dict[str, str | None]:
    """Greedy Contract-Net: award each task to the lowest capable bidder."""
    assignments: dict[str, str | None] = {}
    for task in tasks:
        bids = [
            (a.bid(task["difficulty"]), a.name)
            for a in agents
            if a.skill >= task["min_skill"]
        ]
        assignments[task["id"]] = min(bids)[1] if bids else None
    return assignments


def gossip_consensus(values, adjacency, rounds: int = 200, eps: float = 0.2):
    """Distributed averaging on an undirected graph (Laplacian consensus).

    Each round every node moves by ``eps`` times the *sum* of differences
    to its neighbours: ``v <- v - eps * L @ v`` with the symmetric graph
    Laplacian ``L``. Because ``L`` is symmetric the update preserves the
    global mean, so on a connected graph the values converge to that mean.
    Pick ``eps < 2 / lambda_max(L)`` (roughly ``eps < 1 / max_degree``).
    Returns the full history with shape (rounds + 1, n_nodes).
    """
    v = np.asarray(values, dtype=float)
    A = np.asarray(adjacency, dtype=float)
    history = [v.copy()]
    for _ in range(rounds):
        nv = v.copy()
        for i in range(len(v)):
            neigh = np.where(A[i] > 0)[0]
            if neigh.size:
                nv[i] = v[i] + eps * np.sum(v[neigh] - v[i])
        v = nv
        history.append(v.copy())
    return np.asarray(history)


def majority_vote(votes: list[Any]) -> Any:
    """Return the most common vote (ties broken by first-seen order)."""
    from collections import Counter

    return Counter(votes).most_common(1)[0][0]
