"""
Orchestration topologies for multi-agent workflows.

Each orchestrator runs a list of worker callables and tracks a simple
cost/latency budget so you can compare topologies on more than output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RunStats:
    steps: int = 0
    cost: float = 0.0
    transcript: list[str] = field(default_factory=list)

    def log(self, who: str, msg: str, cost: float = 1.0) -> None:
        self.steps += 1
        self.cost += cost
        self.transcript.append(f"{who}: {msg}")


def pipeline(stages: list[Callable[[str], str]], x: str) -> tuple[str, RunStats]:
    """Sequential pipeline: each stage transforms the previous output."""
    stats = RunStats()
    for i, stage in enumerate(stages, 1):
        x = stage(x)
        stats.log(f"stage{i}", x)
    return x, stats


def supervisor(workers: dict[str, Callable[[str], str]], plan: list[str],
               goal: str, max_steps: int = 20) -> tuple[list[str], RunStats]:
    """Supervisor dispatches the plan to named workers, with a step cap to
    prevent runaway delegation loops.
    """
    stats = RunStats()
    results = []
    for worker_name in plan:
        if stats.steps >= max_steps:
            stats.log("supervisor", "HALT: step budget exhausted", cost=0.0)
            break
        out = workers[worker_name](goal)
        stats.log(worker_name, out)
        results.append(out)
    return results, stats


def debate(propose: Callable[[str], str], critique: Callable[[str], str],
           question: str, rounds: int = 3) -> tuple[str, RunStats]:
    """Two-agent debate: a proposer answers, a critic pushes back, repeat."""
    stats = RunStats()
    answer = propose(question)
    stats.log("proposer", answer)
    for _ in range(rounds):
        objection = critique(answer)
        stats.log("critic", objection)
        answer = propose(f"{question} | revise given: {objection}")
        stats.log("proposer", answer)
    return answer, stats
