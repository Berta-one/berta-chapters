"""Configuration for Chapter 17: Advanced RAG."""

from pathlib import Path

SEED = 17
CHAPTER_ROOT = Path(__file__).resolve().parent.parent
DATASETS = CHAPTER_ROOT / "datasets"

# Reciprocal Rank Fusion constant (60 is the value from the original paper).
RRF_K = 60
