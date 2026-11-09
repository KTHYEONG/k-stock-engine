"""Credential expiry entries kse owns, read from the published credential registry file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

__all__ = ["ExpiringCredential", "load_kse_credentials"]

_SCHEMA: Final = "credential-registry.v1"
_CONSUMER: Final = "kse"
_ACTIVE: Final = "active"
_ESTIMATED: Final = "estimated"


@dataclass(frozen=True, slots=True)
class ExpiringCredential:
    """One active kse-owned credential with a dated expiry."""

    name: str
    days_left: int
    expires_on: date
    estimated: bool


def _entry(raw: Any) -> tuple[str, str, list[str], str, date | None]:
    if not isinstance(raw, dict):
        raise ValueError("registry entry is not a table")
    name, status, consumers, confidence = raw.get("name"), raw.get("status"), raw.get("consumers"), raw.get("confidence")
    if not isinstance(name, str) or not isinstance(status, str) or not isinstance(confidence, str):
        raise ValueError("registry entry has a malformed scalar field")
    if not isinstance(consumers, list) or not all(isinstance(item, str) for item in consumers):
        raise ValueError("registry entry has malformed consumers")
    expires_on = raw.get("expires_on")
    if expires_on is not None and not isinstance(expires_on, date):
        raise ValueError("registry entry has a malformed expires_on")
    return name, status, consumers, confidence, expires_on


def load_kse_credentials(path: Path, today: date) -> tuple[ExpiringCredential, ...]:
    """Active kse entries with an expiry date, sorted by ``(days_left, name)``; ValueError on a bad registry."""
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema") != _SCHEMA:
        raise ValueError("registry schema is not credential-registry.v1")
    raw_entries = payload.get("credential", [])
    if not isinstance(raw_entries, list):
        raise ValueError("registry credential section is malformed")
    found: list[ExpiringCredential] = []
    for raw in raw_entries:
        name, status, consumers, confidence, expires_on = _entry(raw)
        if status != _ACTIVE or _CONSUMER not in consumers or expires_on is None:
            continue
        found.append(ExpiringCredential(
            name=name, days_left=(expires_on - today).days, expires_on=expires_on,
            estimated=confidence == _ESTIMATED,
        ))
    return tuple(sorted(found, key=lambda item: (item.days_left, item.name)))
