"""Configuration for Chapter 23: Building AI Products."""

from pathlib import Path

CHAPTER_ROOT = Path(__file__).resolve().parent.parent
DATASETS = CHAPTER_ROOT / "datasets"

# A healthy SaaS rule of thumb: LTV:CAC >= 3 and payback <= 12 months.
TARGET_LTV_CAC = 3.0
TARGET_PAYBACK_MONTHS = 12.0
