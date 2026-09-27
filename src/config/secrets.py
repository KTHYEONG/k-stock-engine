"""Process-environment secret access; the only module that may touch ``os.environ``."""
from __future__ import annotations

import os

from src.config.errors import ConfigError

__all__ = ["read_secret"]


def read_secret(env_name: str, default: str | None = None) -> str:
    """Return a secret from the process environment; the value is never logged or echoed.

    Raises:
        ConfigError: the variable is missing or empty and no default was given.
            The message names the variable and contains no value.
    """
    if not env_name or not env_name.strip():
        raise ConfigError("secret name must be a non-empty environment variable name")
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        if default is not None:
            return default
        raise ConfigError(f"secret {env_name} is not set (source ~/.quant_env.sh)")
    return value
