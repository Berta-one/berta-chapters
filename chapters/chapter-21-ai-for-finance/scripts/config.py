"""Configuration for Chapter 21: AI for Finance."""

from pathlib import Path

SEED = 21
CHAPTER_ROOT = Path(__file__).resolve().parent.parent
DATASETS = CHAPTER_ROOT / "datasets"

TRADING_DAYS = 252       # annualization factor for daily data
COST_PER_TRADE = 0.0005  # 5 bps per position change
