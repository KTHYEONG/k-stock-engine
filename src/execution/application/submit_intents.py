"""Execution application: validate intents, authorize, plan, submit to broker."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.core.costs import CostSchedule
from src.core.instruments import Instrument
from src.core.ledger import Ledger, LedgerFill, LedgerSide
from src.core.portfolio import PortfolioSnapshot
from src.core.time import SessionCalendar
from src.execution.application.readiness import SubmissionGate, SubmissionRequest
from src.execution.application.validate_intents import IntentValidator
from src.execution.domain.intents import TradeIntent  # TradeIntent.target_quantity
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState, OrderStateRecord
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
    ledger: Ledger | None = None
    cost_schedule: CostSchedule | None = None
    calendar: SessionCalendar | None = None


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
    if context.ledger is not None and (context.cost_schedule is None or context.calendar is None):
        raise ValueError("cost_schedule and calendar required when ledger is supplied")
    validator = IntentValidator(context.state_store)
    submitted: list[OrderStateRecord] = []
    positions = context.positions or PortfolioSnapshot(
        account_snapshot_id="unknown",
        as_of=now,
        settled_cash=0.0,
        unsettled_cash=0.0,
        positions=(),
    )

    # Ledger path uses holdings for reconciliation and records confirmed fills.
    # Sells are planned before buys.
    if context.ledger is not None:
        # classify sells before buys using initial holdings
        ledger = context.ledger
        calendar = context.calendar
        cost_schedule = context.cost_schedule
        assert ledger is not None
        assert calendar is not None
        assert cost_schedule is not None
        intents_sorted = sorted(intents, key=lambda i: (i.instrument_id, i.intent_id))
        # determine side for ordering
        def _side_key(intent: TradeIntent) -> int:
            price = context.price_provider.price_of(intent.instrument_id)
            instrument = context.instruments.get(intent.instrument_id)
            lot_size = instrument.lot_size if instrument is not None else 1
            current_qty = ledger.quantity_of(intent.instrument_id)
            if intent.target_quantity is not None:
                tq = intent.target_quantity
                target_quantity = int(tq)
            else:
                target_quantity = int(intent.target_value / price / lot_size) * lot_size
            delta = target_quantity - current_qty
            if delta < 0:
                return 0
            if delta > 0:
                return 1
            return 2

        intents_sorted.sort(key=_side_key)
        for intent in intents_sorted:
            validator.validate(intent, now)
            price = context.price_provider.price_of(intent.instrument_id)
            instrument = context.instruments.get(intent.instrument_id)
            lot_size = instrument.lot_size if instrument is not None else 1
            current_qty = ledger.quantity_of(intent.instrument_id)
            order = plan_order_request(
                intent,
                order_id=f"order:{intent.intent_id}",
                request_time=now,
                reference_price=price,
                current_quantity=current_qty,
                lot_size=lot_size,
            )
            if order is None:
                context.gate.authorize(
                    SubmissionRequest(
                        intent=intent,
                        mode=context.settings.default_mode,
                        account=context.account,
                    )
                )
                logger.info("intent %s at target; no order", intent.intent_id)
                continue

            cost_point = cost_schedule.cost_for(now)
            expected_quantity = int(order.quantity)
            expected_price = float(order.price or price)
            expected_tax = (
                expected_quantity * expected_price * float(cost_point.tax_rate)
                if order.side is OrderSide.SELL
                else 0.0
            )
            expected_fill = LedgerFill(
                fill_id=order.order_id,
                instrument_id=order.instrument_id,
                side=LedgerSide.BUY if order.side is OrderSide.BUY else LedgerSide.SELL,
                quantity=expected_quantity,
                price=expected_price,
                commission=expected_quantity * expected_price * float(cost_point.commission_rate),
                tax=expected_tax,
                slippage_cost=0.0,
                trade_time=now,
                settlement_time=calendar.advance(now, int(cost_point.settlement_days)),
            )
            # Reject unaffordable orders before they reach a broker.
            ledger.validate_fill(expected_fill)
            context.gate.authorize(
                SubmissionRequest(
                    intent=intent,
                    mode=context.settings.default_mode,
                    account=context.account,
                )
            )
            record = context.broker.submit(order)
            logger.info("order %s -> %s", order.order_id, record.state.value)
            if record.state is OrderState.FILLED:
                if (
                    isinstance(record.filled_quantity, bool)
                    or not float(record.filled_quantity).is_integer()
                    or record.filled_quantity <= 0
                    or record.filled_price is None
                    or record.filled_price <= 0
                ):
                    raise ValueError("broker returned invalid filled quantity or price")
                filled_qty = int(record.filled_quantity)
                filled_price = float(record.filled_price)
                commission = filled_qty * filled_price * float(cost_point.commission_rate)
                tax = 0.0
                if record.side is OrderSide.SELL:
                    tax = filled_qty * filled_price * float(cost_point.tax_rate)
                slippage_cost = abs(filled_price - float(price)) * filled_qty
                settlement_time = calendar.advance(now, int(cost_point.settlement_days))
                side = LedgerSide.BUY if record.side is OrderSide.BUY else LedgerSide.SELL
                ledger.record_fill(
                    LedgerFill(
                        fill_id=record.order_id,
                        instrument_id=record.instrument_id,
                        side=side,
                        quantity=filled_qty,
                        price=filled_price,
                        commission=commission,
                        tax=tax,
                        slippage_cost=slippage_cost,
                        trade_time=now,
                        settlement_time=settlement_time,
                    )
                )
            context.state_store.save(record)
            context.state_store.mark_intent(intent.idempotency_key)
            submitted.append(record)
        return submitted

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
