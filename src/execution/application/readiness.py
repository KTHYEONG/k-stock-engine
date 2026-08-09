"""Live-submission readiness gate.

New live submission is disabled until the dedicated readiness gate passes.
Paper requests require a complete intent; live requests require all readiness
evidence, broker/account reconciliation, and an idempotency key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.execution.domain.intents import TradeIntent


class LiveExecutionNotReadyError(RuntimeError):
    """Raised when a live submission is attempted before the readiness gate."""


class DuplicateIntentError(ValueError):
    """Raised when an intent's idempotency key was already processed."""


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """A request to execute a single approved intent."""

    intent: TradeIntent
    mode: str = "paper"  # "paper" | "live"
    account: str = "paper"
    submitted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    """Independent evidence that must be satisfied for live trading."""

    broker_reconciliation: bool = False
    order_state_transitions: bool = False
    paper_acceptance: bool = False
    idempotency_verified: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return all(
            [
                self.broker_reconciliation,
                self.order_state_transitions,
                self.paper_acceptance,
                self.idempotency_verified,
            ]
        )


class SubmissionGate:
    """Gates intent submission: paper is the default and always allowed."""

    def __init__(self, readiness: ReadinessEvidence | None = None):
        self._readiness = readiness or ReadinessEvidence()
        self._processed_keys: set[str] = set()

    def authorize(self, request: SubmissionRequest) -> None:
        if request.mode not in ("paper", "live"):
            raise ValueError(f"invalid mode {request.mode!r}")
        if request.intent.idempotency_key in self._processed_keys:
            raise DuplicateIntentError(
                f"duplicate intent for idempotency key {request.intent.idempotency_key!r}"
            )
        if request.mode == "live" and not self._readiness.complete:
            raise LiveExecutionNotReadyError(
                "live submission disabled: readiness gate not satisfied "
                f"(reconciliation={self._readiness.broker_reconciliation}, "
                f"transitions={self._readiness.order_state_transitions}, "
                f"paper_acceptance={self._readiness.paper_acceptance}, "
                f"idempotency={self._readiness.idempotency_verified})"
            )
        self._processed_keys.add(request.intent.idempotency_key)
