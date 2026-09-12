"""Fill model tests."""
from __future__ import annotations


def test_historical_fill_uses_next_open_and_sell_only_tax() -> None:
    from datetime import UTC, datetime, timedelta
    from math import inf

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.engine.fill_model import BacktestOrder, ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.orders import OrderSide

    decision = datetime(2024, 1, 2, 6, 30, tzinfo=UTC)
    execution = decision + timedelta(days=1)
    schedule = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.001, 0.002, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, inf, 1.0),))
    model = HistoricalFillModel(schedule, LiquiditySlippageModel(0.1, ticks), ExecutionScenario.BASE, target_participation_cap=0.01, hard_participation_cap=0.02)
    order = BacktestOrder("sell-1", "intent-1", Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW"), OrderSide.SELL, 10, decision, execution)
    outcome = model.execute(order, HistoricalBar(execution, "KRX:005930", 100.0, 101.0, 1_000_000.0, 0.02))

    assert outcome.fill.trade_time == execution
    assert outcome.fill.price < 100.0
    assert outcome.fill.tax > 0.0


def test_historical_fill_caps_oversize_order_without_aborting_run() -> None:
    from datetime import UTC, datetime
    from math import inf

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.engine.fill_model import (
        BacktestOrder,
        BacktestReject,
        ExecutionScenario,
        FillOutcome,
        HistoricalBar,
        HistoricalFillModel,
    )
    from src.execution.domain.orders import OrderSide

    # Given: target cap 1%, hard cap 2% of a 1,000,000 ADTV at price 100 -> 100 shares at target.
    now = datetime(2024, 1, 3, tzinfo=UTC)
    schedule = CostSchedule('fixture', (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule('all', datetime(2020, 1, 1, tzinfo=UTC), 0.0, inf, 1.0),))
    model = HistoricalFillModel(schedule, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)
    instrument = Instrument('KRX:005930', AssetKind.STOCK, 'KRX', '005930', 'KRW')
    bar = HistoricalBar(now, instrument.instrument_id, 100.0, 100.0, 1_000_000.0, 0.02)

    # When: request 250 shares, far beyond the hard cap that previously aborted the whole run.
    outcome = model.execute(BacktestOrder('o2', 'i2', instrument, OrderSide.BUY, 250, now, now), bar)

    # Then: capped, not fatal, and the remainder is reported for the reject receipt.
    assert isinstance(outcome, FillOutcome)
    assert outcome.fill.quantity == 100
    assert outcome.requested_quantity == 250
    assert outcome.unfilled_quantity == 150
    assert outcome.participation <= 0.01 + 1e-12

    # And: a bar with no target capacity still fails closed as a reject.
    thin = HistoricalBar(now, instrument.instrument_id, 100.0, 100.0, 100.0, 0.02)
    rejected = model.execute(BacktestOrder('o3', 'i3', instrument, OrderSide.BUY, 250, now, now), thin)
    assert isinstance(rejected, BacktestReject)
    assert rejected.rejected_quantity == 250
