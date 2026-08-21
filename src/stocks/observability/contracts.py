"""Flat diagnostic event vocabulary and sink protocol.

Every event is a flat record with mandatory keys: ``run_id``, ``sequence``,
``category``, ``component``, ``stage``, ``event``, ``status``, and
``elapsed_ms``.  Payload values are scalar only; strings are bounded to 512
characters.  The ``RunDiagnostics`` protocol is the only sink interface;
concrete implementations live in ``recorder.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast


class DiagnosticCategory(StrEnum):
    SYS = "SYS"
    DATA = "DATA"
    ALGO = "ALGO"
    EVAL = "EVAL"


class DiagnosticStatus(StrEnum):
    START = "START"
    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    SKIP = "SKIP"
    INFO = "INFO"


class DiagnosticStage(StrEnum):
    INPUT = "input"
    DATA = "data"
    SPLIT_FIT = "split_fit"
    CALIBRATION = "calibration"
    SELECTION = "selection"
    ALLOCATION = "allocation"
    EXECUTION = "execution"
    COSTS = "costs"
    SETTLEMENT = "settlement"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    project: str = "stocks"


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    run_id: str
    sequence: int
    category: DiagnosticCategory
    component: str
    stage: DiagnosticStage
    event: str
    status: DiagnosticStatus
    elapsed_ms: float = 0.0
    payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError(f"sequence must be non-negative, got {self.sequence}")
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")
        if self.payload:
            for k, v in self.payload.items():
                if isinstance(v, str) and len(v) > 512:
                    raise ValueError(
                        f"payload string value for {k!r} exceeds 512 characters"
                    )
                if isinstance(v, float) and (v != v):  # NaN check
                    self.payload[k] = None


class RunDiagnostics(Protocol):
    def emit(self, event: DiagnosticEvent) -> None: ...
    def close(self, status: str) -> None: ...


def emit_checkpoint(
    sink: RunDiagnostics | None,
    *,
    run_id: str,
    category: DiagnosticCategory,
    component: str,
    stage: DiagnosticStage,
    event: str,
    status: DiagnosticStatus = DiagnosticStatus.INFO,
    elapsed_ms: float = 0.0,
    payload: dict[str, object] | None = None,
) -> None:
    """Emit one bounded checkpoint without touching quantitative state."""
    if sink is None:
        return
    state = cast(Any, sink)
    sequence = int(getattr(state, "_diagnostic_sequence", 0)) + 1
    state._diagnostic_sequence = sequence
    sink.emit(
        DiagnosticEvent(
            run_id=run_id,
            sequence=sequence,
            category=category,
            component=component,
            stage=stage,
            event=event,
            status=status,
            elapsed_ms=max(0.0, elapsed_ms),
            payload=payload,
        )
    )
