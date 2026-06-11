"""
Deterministic gridworld MDP.

States are (row, col); actions are up/down/left/right. Hitting a wall or the
border leaves the agent in place. Reaching the goal yields +1 and ends the
episode; every other step yields a small negative reward to encourage short
paths. Includes value iteration as a model-based reference.
"""

from __future__ import annotations

import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
STEP_REWARD = -0.04
GOAL_REWARD = 1.0


class GridWorld:
    def __init__(self, layout: list[str]) -> None:
        self.grid = [list(row) for row in layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")
        self.state = self.start

    def _find(self, ch: str) -> tuple[int, int]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == ch:
                    return (r, c)
        raise ValueError(f"missing {ch}")

    def is_wall(self, r: int, c: int) -> bool:
        return not (0 <= r < self.rows and 0 <= c < self.cols) or self.grid[r][c] == "#"

    def reset(self) -> tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int):
        dr, dc = ACTIONS[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        if self.is_wall(nr, nc):
            nr, nc = r, c
        self.state = (nr, nc)
        done = self.state == self.goal
        reward = GOAL_REWARD if done else STEP_REWARD
        return self.state, reward, done

    def states(self):
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] != "#"
        ]

    def value_iteration(self, gamma: float = 0.95, theta: float = 1e-6):
        """Model-based optimal values via Bellman optimality backups."""
        V = {s: 0.0 for s in self.states()}
        while True:
            delta = 0.0
            for s in self.states():
                if s == self.goal:
                    continue
                best = -1e9
                for a in range(4):
                    self.state = s
                    ns, r, _ = self.step(a)
                    best = max(best, r + gamma * V[ns])
                delta = max(delta, abs(best - V[s]))
                V[s] = best
            if delta < theta:
                break
        return V
