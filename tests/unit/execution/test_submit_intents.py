"""submit_intents application contract tests."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from src.execution.adapters.in_memory_state_store import InMemoryStateStore
from src.execution.adapters.paper_broker import ConstantPriceProvider, PaperBroker
from src.execution.application.readiness import SubmissionGate
from src.execution.application.submit_intents import (
    ExecutionContext,
    plan_order_request,
    submit_intents,
)
from src.execution.domain.orders import OrderSide, OrderState
from src.execution.settings import DEFAULT_EXECUTION
from tests.unit.execution.test_intents import make_intent


def paper_context(*, store: InMemoryStateStore | None = None) -> ExecutionContext:
    return ExecutionContext(
        settings=DEFAULT_EXECUTION,
        gate=SubmissionGate(),
        broker=PaperBroker(),
        state_store=store or InMemoryStateStore(),
        price_provider=ConstantPriceProvider(5_000.0),
    )


class TestSubmitIntents:
    def test_submit_intents_fills_in_paper_mode(self) -> None:
        store = InMemoryStateStore()
        context = paper_context(store=store)
        intents = [make_intent("x"), replace(make_intent("y"), intent_id="intent-y", idempotency_key="key-y")]
        records = submit_intents(intents, context)
        assert len(records) == 2
        assert all(r.state is OrderState.FILLED for r in records)

    def test_order_quantity_is_target_notional_over_price(self) -> None:
        context = paper_context()
        records = submit_intents([make_intent("q")], context)
        assert records[0].submitted_quantity == 1_000_000.0 / 5_000.0

    def test_at_target_intent_submits_no_order(self) -> None:
        context = paper_context()
        store = InMemoryStateStore()
        context = ExecutionContext(
            settings=DEFAULT_EXECUTION,
            gate=SubmissionGate(),
            broker=PaperBroker(),
            state_store=store,
            price_provider=ConstantPriceProvider(5_000.0),
            positions=paper_positions(1_000_000.0 / 5_000.0),
        )
        records = submit_intents([make_intent("at-target")], context)
        assert records == []

    def test_duplicate_intent_rejected(self) -> None:
        store = InMemoryStateStore()
        context = paper_context(store=store)
        intent = make_intent()
        submit_intents([intent], context)
        with pytest.raises(ValueError, match="duplicate"):
            submit_intents([intent], context)


class TestPlanOrderRequest:
    def test_zero_target_plans_full_exit_sell(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        exit_intent = make_intent("exit", target_value=0.0)
        order = plan_order_request(
            exit_intent,
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=10,
        )
        assert order is not None
        assert order.side is OrderSide.SELL
        assert order.quantity == 10

    def test_partial_exit_sells_remaining_delta(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        order = plan_order_request(
            make_intent("partial", target_value=300.0),
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=10,
        )
        assert order is not None
        assert order.side is OrderSide.SELL
        assert order.quantity == 7

    def test_target_above_current_buys_delta(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        order = plan_order_request(
            make_intent("buy", target_value=1_000.0),
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=2,
        )
        assert order is not None
        assert order.side is OrderSide.BUY
        assert order.quantity == 8

    def test_at_target_returns_none(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        order = plan_order_request(
            make_intent("keep", target_value=1_000.0),
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=10,
        )
        assert order is None

    def test_quantity_is_floor_to_lot(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        order = plan_order_request(
            make_intent("lot", target_value=10_050.0),
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=0,
            lot_size=100,
        )
        assert order is not None
        assert order.side is OrderSide.BUY
        assert order.quantity == 100

    def test_target_below_one_lot_returns_none(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        order = plan_order_request(
            make_intent("lot-below", target_value=1_050.0),
            order_id="o",
            request_time=execution,
            reference_price=100.0,
            current_quantity=0,
            lot_size=100,
        )
        assert order is None

    def test_rejects_stale_account_snapshot(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        with pytest.raises(ValueError, match="account_snapshot_id"):
            plan_order_request(
                make_intent("stale", account_snapshot_id=""),
                order_id="o",
                request_time=execution,
                reference_price=100.0,
                current_quantity=0,
            )

    def test_rejects_non_positive_price(self) -> None:
        execution = datetime(2024, 6, 3, 1, 15, tzinfo=UTC)
        with pytest.raises(ValueError, match="reference_price"):
            plan_order_request(
                make_intent("price"),
                order_id="o",
                request_time=execution,
                reference_price=0.0,
                current_quantity=0,
            )


def paper_positions(quantity: float):
    from src.core.instruments import AssetKind, Instrument
    from src.core.portfolio import PortfolioSnapshot, Position

    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    return PortfolioSnapshot(
        account_snapshot_id="account-a",
        as_of=datetime(2024, 6, 3, 1, 0, tzinfo=UTC),
        settled_cash=0.0,
        unsettled_cash=0.0,
        positions=(Position(instrument=instrument, quantity=quantity, average_cost=5_000.0),),
    )

def test_SCENARIO_SMALL_ACCOUNT_LOT_04_LIVE_LOT_REJECTION():
    """SCENARIO_SMALL_ACCOUNT_LOT_04_LIVE_LOT_REJECTION"""
    from src.execution.domain.intents import TradeIntent
    from src.core.instruments import AssetKind
    from src.execution.application.submit_intents import plan_order_request
    from datetime import UTC, datetime
    import pytest
    def make(qty, tv=1700):
        return TradeIntent(intent_id='i', asset_kind=AssetKind.STOCK, instrument_id='KRX:005930', target_value=float(tv), target_quantity=qty, decision_time=datetime(2024,6,3,1,0,tzinfo=UTC), execution_time=datetime(2024,6,3,1,15,tzinfo=UTC), strategy_id='s', reason='r', idempotency_key='k', account_snapshot_id='a')
    with pytest.raises(ValueError, match="multiple"):
        plan_order_request(make(17, 1700), order_id='o', request_time=datetime(2024,6,3,1,15,tzinfo=UTC), reference_price=100.0, current_quantity=0, lot_size=10)
    order = plan_order_request(make(20, 2000), order_id='o', request_time=datetime(2024,6,3,1,15,tzinfo=UTC), reference_price=100.0, current_quantity=0, lot_size=10)
    assert order is not None
    assert order.quantity == 20
    order_none = plan_order_request(make(None, 1050), order_id='o', request_time=datetime(2024,6,3,1,15,tzinfo=UTC), reference_price=100.0, current_quantity=0, lot_size=10)
    # floor(1050/100/10)*10 = 10
    assert order_none is not None
    assert order_none.quantity == 10
