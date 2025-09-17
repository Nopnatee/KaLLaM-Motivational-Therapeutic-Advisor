from __future__ import annotations

"""Shared helpers for configuring SEA-Lion access.

Supports both the public SEA-Lion API and the Amazon Bedrock access gateway
that exposes an OpenAI-compatible endpoint. Configuration is driven entirely by
environment variables so individual agents do not have to duplicate the same
initialisation logic.
"""

from dataclasses import dataclass
import hashlib
import os
from typing import Optional


@dataclass
class SeaLionSettings:
    base_url: str
    token: Optional[str]
    model: str
    mode: str  # "gateway", "api-key", or "disabled"


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def fingerprint_secret(secret: Optional[str]) -> str:
    """Return a short fingerprint for a secret without exposing it."""
    if not secret:
        return "unset"
    try:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return "error"


def load_sea_lion_settings(*, default_model: str) -> SeaLionSettings:
    """Load SEA-Lion configuration from environment variables."""
    gateway_url = _strip_or_none(os.getenv("SEA_LION_GATEWAY_URL"))
    direct_url = _strip_or_none(os.getenv("SEA_LION_BASE_URL"))
    base_url = gateway_url or direct_url or "https://api.sea-lion.ai/v1"
    base_url = base_url.rstrip("/")

    gateway_token = _strip_or_none(os.getenv("SEA_LION_GATEWAY_TOKEN"))
    api_key = _strip_or_none(os.getenv("SEA_LION_API_KEY"))

    if gateway_token:
        mode = "gateway"
        token = gateway_token
    elif api_key:
        mode = "api-key"
        token = api_key
    else:
        mode = "disabled"
        token = None

    model = _strip_or_none(os.getenv("SEA_LION_MODEL_ID")) or default_model

    return SeaLionSettings(
        base_url=base_url,
        token=token,
        model=model,
        mode=mode,
    )
