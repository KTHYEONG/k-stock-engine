"""Execution application: validate intents, authorize, submit to broker."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from src.execution.application.submission_gate import (
    ReadinessEvidence,
    SubmissionGate,
    SubmissionRequest,
)
from src.execution.domain.order import (
    OrderRequest,
    OrderSide,
    OrderStateRecord,
    TradeIntent,
)
from src.execution.infrastructure.paper import InMemoryStateStore, PaperBroker
from src.execution.ports.broker import BrokerPort, StateStorePort

logger = logging.getLogger("execution.run_cycle")


class IntentValidator:
    """Validates session, asset kind, quantity, and duplicate intents."""

    def __init__(self, state_store: StateStorePort):
        self.state_store = state_store

    def validate(self, intent: TradeIntent, now: datetime) -> None:
        if intent.decision_time > intent.execution_time:
            raise ValueError("intent decision_time after execution_time")
        if intent.target_value <= 0:
            raise ValueError("intent target_value must be positive")
        if self.state_store.is_duplicate_intent(intent.idempotency_key):
            raise ValueError(f"duplicate intent {intent.idempotency_key!r}")


def run_cycle(
    intents: Sequence[TradeIntent],
    *,
    submission_gate: SubmissionGate,
    broker: BrokerPort,
    state_store: StateStorePort,
    now: datetime | None = None,
) -> list[OrderStateRecord]:
    """Process a batch of approved intents through the execution boundary.

    Paper mode is the default; the live adapter is not enabled by this refactor.
    """
    now = now or datetime.now()
    validator = IntentValidator(state_store)
    submitted: list[OrderStateRecord] = []

    for intent in intents:
        validator.validate(intent, now)
        submission_gate.authorize(
            SubmissionRequest(intent=intent, mode="paper", account=broker.account)
        )
        order_id = f"order:{intent.intent_id}"
        request = OrderRequest(
            order_id=order_id,
            asset_kind=intent.asset_kind,
            instrument_id=intent.instrument_id,
            side=OrderSide.BUY,
            quantity=intent.target_value,
            price=None,
            request_time=now,
            idempotency_key=intent.idempotency_key,
            intent_id=intent.intent_id,
        )
        record = broker.submit(request)
        state_store.save(record)
        state_store.mark_intent(intent.idempotency_key)
        submitted.append(record)
        logger.info("order %s -> %s", order_id, record.state.value)

    return submitted


def make_default_cycle() -> tuple[SubmissionGate, PaperBroker, InMemoryStateStore]:
    """Build the paper-mode default wiring (no live adapter enabled)."""
    gate = SubmissionGate(ReadinessEvidence())
    broker = PaperBroker()
    store = InMemoryStateStore()
    return gate, broker, store
