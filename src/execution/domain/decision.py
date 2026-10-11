"""Structural view of a sealed target decision consumed by the execution layer."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Protocol


class TargetDecision(Protocol):
    @property
    def execution_session(self) -> date: ...

    @property
    def sealed_at(self) -> datetime: ...

    @property
    def models_id(self) -> str: ...

    @property
    def targets(self) -> Mapping[str, float]: ...

    @property
    def reference(self) -> Mapping[str, float]: ...

    @property
    def hash(self) -> str: ...
