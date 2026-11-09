"""Synthetic multi-key DART policy for tests; the shipped config declares one key."""
from __future__ import annotations

from typing import Any

SECOND_KEY_ENV = "OPENDART_API_KEY_2"


def with_second_dart_key(provider: Any) -> Any:
    """Return ``provider`` with an unmarked second key, leaving the default on the primary."""
    primary = provider.dart_key(provider.primary_key_env)
    second = primary.model_copy(update={"daily_budget": 19500, "daily_reserve": 500, "default": False})
    keys = {**provider.dart.keys, SECOND_KEY_ENV: second}
    return provider.model_copy(update={"dart": provider.dart.model_copy(update={"keys": keys})})
