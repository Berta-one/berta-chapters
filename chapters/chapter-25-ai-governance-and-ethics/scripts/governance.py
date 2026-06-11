"""
Operational governance tools: risk tiering, model cards, and risk matrices.

Rule-based and transparent — governance must be explainable. Pure stdlib.
"""

from __future__ import annotations


def classify_risk(system: dict) -> str:
    """Map a system descriptor to an EU AI Act-style risk tier.

    Tiers: 'unacceptable', 'high', 'limited', 'minimal'.
    """
    from config import HIGH_RISK_DOMAINS, PROHIBITED_DOMAINS

    domain = system.get("domain", "")
    if system.get("prohibited") or domain in PROHIBITED_DOMAINS:
        return "unacceptable"
    if domain in HIGH_RISK_DOMAINS:
        return "high"
    if system.get("interacts_with_users") or system.get("generates_content"):
        return "limited"
    return "minimal"


def generate_model_card(meta: dict) -> str:
    """Render structured metadata as a Markdown model card."""
    from config import REQUIRED_CARD_FIELDS

    lines = [f"# Model Card: {meta.get('model_name', 'Unnamed Model')}", ""]
    for field in REQUIRED_CARD_FIELDS:
        title = field.replace("_", " ").title()
        value = meta.get(field, "_Not provided_")
        lines += [f"## {title}", str(value), ""]
    return "\n".join(lines).rstrip() + "\n"


def validate_model_card(meta: dict) -> dict:
    """Report which required fields are missing or empty."""
    from config import REQUIRED_CARD_FIELDS

    missing = [f for f in REQUIRED_CARD_FIELDS if not meta.get(f)]
    return {"complete": not missing, "missing": missing}


def risk_matrix(likelihood: int, severity: int) -> str:
    """Likelihood x severity (each 1..5) into a risk level."""
    score = likelihood * severity
    if score >= 20:
        return "critical"
    if score >= 12:
        return "high"
    if score >= 6:
        return "medium"
    return "low"
