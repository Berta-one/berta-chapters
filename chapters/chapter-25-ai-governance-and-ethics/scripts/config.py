"""Configuration for Chapter 25: AI Governance & Ethics."""

# Domains the EU AI Act treats as high-risk (illustrative subset).
HIGH_RISK_DOMAINS = {
    "biometric_id",
    "critical_infrastructure",
    "employment",
    "credit_scoring",
    "law_enforcement",
    "education",
}

# Use-cases the Act prohibits outright (illustrative subset).
PROHIBITED_DOMAINS = {"social_scoring", "manipulative_targeting"}

# Fields every model card must contain before release.
REQUIRED_CARD_FIELDS = [
    "model_name",
    "intended_use",
    "training_data",
    "metrics",
    "limitations",
    "ethical_considerations",
]
