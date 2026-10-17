"""Typed thresholds for live fail-closed data-freshness gates."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.config.errors import ConfigError

__all__ = ["LiveGatePolicy", "load_live_gate_policy"]


class LiveGatePolicy(BaseModel):
    """Typed thresholds for live fail-closed data-freshness gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_missing_flow_fraction: Annotated[float, Field(ge=0.0, lt=1.0)]


def load_live_gate_policy(path: Path) -> LiveGatePolicy:
    """Load ``config/research/live_gate.toml``; raises ConfigError on missing file, bad TOML, unknown keys or out-of-range values."""
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"live gate config is missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"live gate config is invalid TOML: {path}") from exc
    unknown = set(raw) - {"flow"}
    if unknown:
        raise ConfigError(f"live gate config has unknown keys: {sorted(unknown)}")
    section = raw.get("flow")
    if not isinstance(section, dict):
        raise ConfigError(f"live gate config must declare [flow]: {path}")
    try:
        return LiveGatePolicy.model_validate(section)
    except ValueError as exc:
        raise ConfigError(f"live gate config has an unknown key or invalid value: {exc}") from exc
