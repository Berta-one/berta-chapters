"""Configuration for Chapter 19: Model Optimization."""

from pathlib import Path

SEED = 19
CHAPTER_ROOT = Path(__file__).resolve().parent.parent

# Bytes per parameter by precision.
BYTES = {"float32": 4, "float16": 2, "int8": 1}
