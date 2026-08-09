"""Execution application: validate intents, authorize, plan, submit to broker."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.execution.application.readiness import SubmissionGate, SubmissionRequest
from src.execution.application.validate_intents import IntentValidator
from src.execution.domain.intents import TradeIntent
from src.execution.domain.orders import OrderRequest, OrderSide, OrderStateRecord
from src.execution.ports.broker import BrokerPort, PriceProvider
from src.execution.ports.state_store import StateStorePort
from src.execution.settings import ExecutionSettings

logger = logging.getLogger("execution.submit_intents")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Wiring for one submission cycle. Paper mode is the only default."""

    settings: ExecutionSettings
    gate: SubmissionGate
    broker: BrokerPort
    state_store: StateStorePort
    price_provider: PriceProvider
    account: str = "paper"
    now: datetime | None = None


def plan_order_request(
    intent: TradeIntent,
    *,
    order_id: str,
    request_time: datetime,
    reference_price: float,
) -> OrderRequest:
    """Plan a validated order from an approved intent.

    ``target_value`` is a currency amount, never a share quantity: the order
    quantity is derived by dividing by the reference price. Side is derived
    from the intent's sign, not hard-coded to BUY.
    """
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    side = OrderSide.BUY if intent.target_value >= 0 else OrderSide.SELL
    quantity = abs(intent.target_value) / reference_price
    return OrderRequest(
        order_id=order_id,
        asset_kind=intent.asset_kind,
        instrument_id=intent.instrument_id,
        side=side,
        quantity=quantity,
        price=reference_price,
        request_time=request_time,
        idempotency_key=intent.idempotency_key,
        intent_id=intent.intent_id,
    )


def submit_intents(
    intents: Sequence[TradeIntent],
    context: ExecutionContext,
) -> list[OrderStateRecord]:
    """Process a batch of approved intents through the execution boundary.

    Each intent is validated, authorized by the readiness gate, sized with the
    reference price, submitted to the broker, and recorded in the state store.
    """
    now = context.now or datetime.now(UTC)
    validator = IntentValidator(context.state_store)
    submitted: list[OrderStateRecord] = []

    for intent in intents:
        validator.validate(intent, now)
        context.gate.authorize(
            SubmissionRequest(
                intent=intent,
                mode=context.settings.default_mode,
                account=context.account,
            )
        )
        price = context.price_provider.price_of(intent.instrument_id)
        order = plan_order_request(
            intent,
            order_id=f"order:{intent.intent_id}",
            request_time=now,
            reference_price=price,
        )
        record = context.broker.submit(order)
        context.state_store.save(record)
        context.state_store.mark_intent(intent.idempotency_key)
        submitted.append(record)
        logger.info("order %s -> %s", order.order_id, record.state.value)

    return submitted
