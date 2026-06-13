"""Configuration for Chapter 22: AI Safety & Alignment."""

# Illustrative banned patterns (toy). Real systems use trained classifiers.
BANNED_PATTERNS = [
    "build a bomb",
    "steal a car",
    "make a weapon",
    "credit card number",
]

# Disparate impact below this ratio fails the US EEOC "four-fifths" rule.
FOUR_FIFTHS = 0.8
