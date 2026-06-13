"""Configuration for Chapter 16: Multi-Agent Systems."""

from pathlib import Path

SEED = 16
CHAPTER_ROOT = Path(__file__).resolve().parent.parent
DATASETS = CHAPTER_ROOT / "datasets"

# Default agent roster (name, skill in [0,1], cost per unit difficulty)
ROSTER = [
    ("researcher", 0.90, 1.00),
    ("coder", 0.80, 0.80),
    ("writer", 0.60, 0.40),
    ("critic", 0.75, 0.55),
]

TOPOLOGIES = ("supervisor", "pipeline", "debate")
