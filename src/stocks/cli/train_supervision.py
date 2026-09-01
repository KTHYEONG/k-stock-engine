# mypy: ignore-errors
"""Cgroup/RSS sampling, journal, subprocess supervision."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class TrainSupervisor:
    """Fail-closed supervision state for a single training invocation."""

    timeout_seconds: float = 0.0
    started_at: float = field(default_factory=monotonic)
    return_code: int | None = None

    def expired(self, now: float | None = None) -> bool:
        if self.timeout_seconds <= 0.0:
            return False
        current = monotonic() if now is None else float(now)
        return current - self.started_at >= self.timeout_seconds

    def finish(self, return_code: int) -> int:
        if not isinstance(return_code, int):
            raise TypeError("return_code must be int")
        self.return_code = return_code
        return return_code
__all__ = ["TrainSupervisor"]
