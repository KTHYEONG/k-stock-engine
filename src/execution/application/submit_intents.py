"""Execution application: validate intents, authorize, plan, submit to broker."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.core.instruments import Instrument
from src.core.portfolio import PortfolioSnapshot
from src.execution.application.readiness import SubmissionGate, SubmissionRequest
from src.execution.application.validate_intents import IntentValidator
from src.execution.domain.intents import TradeIntent  # TradeIntent.target_quantity
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
    positions: PortfolioSnapshot | None = None
    instruments: Mapping[str, Instrument] = field(default_factory=dict)


def plan_order_request(
    intent: TradeIntent,
    *,
    order_id: str,
    request_time: datetime,
    reference_price: float,
    current_quantity: int,
    lot_size: int = 1,
) -> OrderRequest | None:
    """Plan a validated order from an approved target-position intent.

    ``target_value`` is a non-negative desired position notional, never an order
    delta. The integer target quantity is the floor of the target notional at
    the reference price, rounded to the lot. The signed delta is the target
    quantity minus the broker-reconciled current quantity:

    - ``delta > 0``: BUY order of the delta quantity;
    - ``delta < 0``: SELL order of the absolute delta quantity;
    - ``delta == 0``: no order, returns ``None`` (an explicit ``AT_TARGET``).

    ``None`` is also returned when the target is unreachable only because the
    lot floor yields zero additional shares. Sells must be planned before buys
    so sale proceeds are not treated as spendable before settlement.
    """
    if not intent.account_snapshot_id:
        raise ValueError("stale intent: account_snapshot_id is missing")
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if lot_size < 1:
        raise ValueError("lot_size must be positive")
    if current_quantity < 0:
        raise ValueError("current_quantity must be non-negative")

    # reject non-lot-multiple hard target; otherwise use it before legacy target_value conversion
    if intent.target_quantity is not None:
        tq = intent.target_quantity
        if isinstance(tq, bool) or not isinstance(tq, int) or tq < 0:
            raise ValueError("target_quantity must be non-negative")
        if tq % lot_size != 0:
            raise ValueError(f"target_quantity {tq} not a multiple of lot_size {lot_size}")
        target_quantity = tq
    else:
        target_quantity = int(intent.target_value / reference_price / lot_size) * lot_size
    delta = target_quantity - current_quantity
    if delta == 0:
        return None
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    if side is OrderSide.SELL and -delta > current_quantity:
        raise ValueError("quantity shortfall: cannot sell more than broker holdings")
    return OrderRequest(
        order_id=order_id,
        asset_kind=intent.asset_kind,
        instrument_id=intent.instrument_id,
        side=side,
        quantity=abs(delta),
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
    reference price and reconciled current quantity, submitted to the broker,
    and recorded in the state store. Sells are planned before buys.
    """
    now = context.now or datetime.now(UTC)
    validator = IntentValidator(context.state_store)
    submitted: list[OrderStateRecord] = []
    positions = context.positions or PortfolioSnapshot(
        account_snapshot_id="unknown",
        as_of=now,
        settled_cash=0.0,
        unsettled_cash=0.0,
        positions=(),
    )

    for intent in sorted(intents, key=lambda i: (i.instrument_id, i.intent_id)):
        validator.validate(intent, now)
        context.gate.authorize(
            SubmissionRequest(
                intent=intent,
                mode=context.settings.default_mode,
                account=context.account,
            )
        )
        price = context.price_provider.price_of(intent.instrument_id)
        instrument = context.instruments.get(intent.instrument_id)
        lot_size = instrument.lot_size if instrument is not None else 1
        order = plan_order_request(
            intent,
            order_id=f"order:{intent.intent_id}",
            request_time=now,
            reference_price=price,
            current_quantity=positions.quantity_of(intent.instrument_id),
            lot_size=lot_size,
        )
        if order is None:
            logger.info("intent %s at target; no order", intent.intent_id)
            continue
        record = context.broker.submit(order)
        context.state_store.save(record)
        context.state_store.mark_intent(intent.idempotency_key)
        submitted.append(record)
        logger.info("order %s -> %s", order.order_id, record.state.value)

    return submitted
