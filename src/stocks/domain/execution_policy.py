"""Immutable execution-outcome policy shared by labels, replay, and backtesting.

The policy is the one declared execution definition consumed by label
construction, ML replay, and the event backtester. It never changes a missing
open into another OHLC field or a fabricated return: a leg fills only at a
verified, finite, strictly positive open at or after its scheduled session.

``scheduled_open_v1`` preserves the existing semantics and is fully
fail-closed: entry is required at the scheduled entry open and exit at the
scheduled exit open. ``first_tradable_open_v1`` is an optional operational
strategy and is legal only when its non-negative maximum entry/exit delays are
explicitly configured; it may only look forward, never backwards, and never
use an unverified bar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

SCHEDULED_OPEN_POLICY_ID = "scheduled_open_v1"
FIRST_TRADABLE_OPEN_POLICY_ID = "first_tradable_open_v1"

DEFAULT_ENTRY_OFFSET_SESSIONS = 1


@dataclass(frozen=True, slots=True)
class ExecutionOutcomePolicy:
    """Versioned execution policy binding scheduled and deferred fills.

    ``policy_id`` pins the semantic version; ``canonical_hash`` binds the exact
    configuration. ``entry_offset_sessions`` is the decision-session offset of
    the scheduled entry (1 = the next KRX session open). ``max_entry_delay`` /
    ``max_exit_delay`` are the maximum forward delay sessions for a first valid
    open; both are zero for ``scheduled_open_v1`` and must be non-negative and
    explicitly set for ``first_tradable_open_v1``.
    """

    policy_id: str
    entry_offset_sessions: int = DEFAULT_ENTRY_OFFSET_SESSIONS
    max_entry_delay_sessions: int = 0
    max_exit_delay_sessions: int = 0

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if self.entry_offset_sessions < 1:
            raise ValueError("entry_offset_sessions must be at least 1")
        if self.max_entry_delay_sessions < 0:
            raise ValueError("max_entry_delay_sessions must be non-negative")
        if self.max_exit_delay_sessions < 0:
            raise ValueError("max_exit_delay_sessions must be non-negative")
        deferred = self.max_entry_delay_sessions or self.max_exit_delay_sessions
        if deferred and self.policy_id == SCHEDULED_OPEN_POLICY_ID:
            raise ValueError(
                f"{SCHEDULED_OPEN_POLICY_ID} is fully fail-closed and must not "
                "carry explicit delay bounds"
            )

    @property
    def permits_deferral(self) -> bool:
        return bool(self.max_entry_delay_sessions or self.max_exit_delay_sessions)

    @property
    def canonical_hash(self) -> str:
        payload = json.dumps(
            {
                "policy_id": self.policy_id,
                "entry_offset_sessions": int(self.entry_offset_sessions),
                "max_entry_delay_sessions": int(self.max_entry_delay_sessions),
                "max_exit_delay_sessions": int(self.max_exit_delay_sessions),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


SCHEDULED_OPEN_V1 = ExecutionOutcomePolicy(policy_id=SCHEDULED_OPEN_POLICY_ID)
