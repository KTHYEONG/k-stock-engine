"""Configuration failures that fail closed without echoing secrets."""
from __future__ import annotations


class ConfigError(ValueError):
    """Raised when configuration is missing, unknown, or inconsistent."""
