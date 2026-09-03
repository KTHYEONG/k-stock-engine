"""Backtest tests."""
from __future__ import annotations


def test_backtester_separates_close_decision_from_next_open_fill() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    day_one = datetime(2024, 1, 2, tzinfo=UTC)
    day_two = day_one + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, day_two, "champion-v1", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    sessions = (BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()), BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, instrument.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()))
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    result = EventBacktester(BacktestConfig("bt-1", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, BuyOnce())

    assert len(result.fills) == 1
    assert result.fills[0].trade_time == day_two


def test_backtester_requires_explicit_execution_dependencies() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario

    opened = datetime(2024, 1, 2, tzinfo=UTC)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    session = BacktestSession(opened, opened + timedelta(hours=6, minutes=30), (), (), object())
    config = BacktestConfig("missing-dependencies", 100.0, {}, ExecutionScenario.IDEAL)

    with pytest.raises(BacktestIntegrityError, match="cost_schedule"):
        EventBacktester(config).run((session,), NoTrade())


def test_backtester_rejects_fill_model_scenario_mismatch() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    base_model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.BASE, target_participation_cap=0.01, hard_participation_cap=0.02)
    session = BacktestSession(opened, opened + timedelta(hours=6), (HistoricalBar(opened, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object())
    config = BacktestConfig("scenario-mismatch", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((opened,)), base_model)

    with pytest.raises(BacktestIntegrityError, match="scenario"):
        EventBacktester(config).run((session,), lambda context: ())


def test_backtester_rejects_duplicate_or_misaligned_session_bars() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    duplicate = HistoricalBar(opened, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02)
    session = BacktestSession(opened, opened + timedelta(hours=6), (duplicate, duplicate), (), object())
    config = BacktestConfig("duplicate-bars", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((opened,)), model)

    with pytest.raises(BacktestIntegrityError, match="duplicate bar"):
        EventBacktester(config).run((session,), NoTrade())


def test_backtester_records_partial_remainder_as_reject() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    first = datetime(2024, 1, 2, tzinfo=UTC)
    second = first + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time != first + timedelta(hours=6):
                return ()
            return (TradeIntent("buy", AssetKind.STOCK, instrument.instrument_id, 150.0, context.decision_time, second, "champion-v1", "fixture", "buy-key", context.portfolio.account_snapshot_id, target_quantity=150),)

    sessions = (
        BacktestSession(first, first + timedelta(hours=6), (HistoricalBar(first, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
        BacktestSession(second, second + timedelta(hours=6), (HistoricalBar(second, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
    )
    config = BacktestConfig("partial", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((first, second)), model)

    result = EventBacktester(config).run(sessions, BuyOnce())

    assert result.fills[0].quantity == 100
    assert result.rejects[0].rejected_quantity == 50
    assert result.capacity_diagnostics[0].requested_quantity == 150
    assert result.capacity_diagnostics[0].filled_quantity == 100


def test_backtester_rejects_intent_not_created_at_current_decision() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestIntegrityError, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    first = datetime(2024, 1, 2, tzinfo=UTC)
    second = first + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class StaleIntent:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            return (TradeIntent("stale", AssetKind.STOCK, instrument.instrument_id, 10.0, context.decision_time - timedelta(days=1), second, "champion-v1", "fixture", "stale-key", context.portfolio.account_snapshot_id),)

    sessions = (
        BacktestSession(first, first + timedelta(hours=6), (HistoricalBar(first, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
        BacktestSession(second, second + timedelta(hours=6), (HistoricalBar(second, instrument.instrument_id, 1.0, 1.0, 10_000.0, 0.02),), (), object()),
    )
    config = BacktestConfig("stale", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar((first, second)), model)

    with pytest.raises(BacktestIntegrityError, match="decision_time"):
        EventBacktester(config).run(sessions, StaleIntent())
