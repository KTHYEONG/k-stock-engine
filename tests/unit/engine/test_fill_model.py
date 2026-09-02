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


def test_historical_fill_partial_and_hard_participation_fail_closed() -> None:
    from datetime import UTC, datetime
    from math import inf

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.engine.backtest import BacktestIntegrityError
    from src.engine.fill_model import BacktestOrder, ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.orders import OrderSide

    now = datetime(2024, 1, 3, tzinfo=UTC)
    schedule = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, inf, 1.0),))
    model = HistoricalFillModel(schedule, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    partial = model.execute(BacktestOrder("o1", "i1", instrument, OrderSide.BUY, 150, now, now), HistoricalBar(now, instrument.instrument_id, 100.0, 100.0, 1_000_000.0, 0.02))

    assert partial.fill.quantity == 100
    with pytest.raises(BacktestIntegrityError, match="hard participation"):
        model.execute(BacktestOrder("o2", "i2", instrument, OrderSide.BUY, 250, now, now), HistoricalBar(now, instrument.instrument_id, 100.0, 100.0, 1_000_000.0, 0.02))
