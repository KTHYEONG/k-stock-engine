"""Execution bounded-context settings (primitive fields only).

Named ``ExecutionSettings`` to avoid colliding with the legacy
``ExecutionConfig`` type that lives in the quarantined live trading code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Execution boundary configuration; live submission is disabled by default."""

    default_mode: str = "paper"
    session_open: str = "09:00"
    session_close: str = "15:30"
    version: str = "v1"


DEFAULT_EXECUTION = ExecutionSettings()
