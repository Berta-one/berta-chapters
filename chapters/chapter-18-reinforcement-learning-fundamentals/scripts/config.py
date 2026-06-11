"""Configuration for Chapter 18: Reinforcement Learning."""

from pathlib import Path

SEED = 18
CHAPTER_ROOT = Path(__file__).resolve().parent.parent

GAMMA = 0.95      # discount factor
ALPHA = 0.5       # learning rate
EPSILON = 0.1     # exploration rate
EPISODES = 500

# 5x5 gridworld: '#' = wall, 'S' = start, 'G' = goal, '.' = free
LAYOUT = [
    "S....",
    ".##..",
    ".#.#.",
    "...#.",
    "..#.G",
]
