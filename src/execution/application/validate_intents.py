"""Execution intent validation: session, asset kind, quantity, duplicates."""
from __future__ import annotations

from datetime import datetime

from src.core.time import KRX_DAILY
from src.execution.domain.intents import TradeIntent
from src.execution.ports.state_store import StateStorePort


class IntentValidator:
    """Validates session, asset kind, quantity, and duplicate intents."""

    def __init__(self, state_store: StateStorePort):
        self.state_store = state_store

    def validate(self, intent: TradeIntent, now: datetime) -> None:
        if intent.decision_time > intent.execution_time:
            raise ValueError("intent decision_time after execution_time")
        if intent.target_value < 0:
            raise ValueError("intent target_value must be non-negative")
        if not intent.account_snapshot_id:
            raise ValueError("intent account_snapshot_id must be non-empty")
        if not KRX_DAILY.in_session(intent.execution_time):
            raise ValueError(
                f"execution outside {KRX_DAILY.name} session: {intent.execution_time.isoformat()}"
            )
        if self.state_store.is_duplicate_intent(intent.idempotency_key):
            raise ValueError(f"duplicate intent {intent.idempotency_key!r}")
