"""Backtest tests."""
from __future__ import annotations


def test_backtester_separates_close_decision_from_next_open_fill() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar
    from src.execution.domain.intents import TradeIntent

    day_one = datetime(2024, 1, 2, tzinfo=UTC)
    day_two = day_one + timedelta(days=1)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")

    class BuyOnce:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() != day_one.date():
                return ()
            return (TradeIntent("intent-1", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, day_two, "champion-v1", "fixture", "key-1", context.portfolio.account_snapshot_id),)

    sessions = (BacktestSession(day_one, day_one + timedelta(hours=6, minutes=30), (HistoricalBar(day_one, instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()), BacktestSession(day_two, day_two + timedelta(hours=6, minutes=30), (HistoricalBar(day_two, instrument.instrument_id, 11.0, 11.0, 1_000_000.0, 0.02),), (), object()))
    calendar = SessionCalendar((day_one, day_two, day_two + timedelta(days=1), day_two + timedelta(days=2)))
    result = EventBacktester(BacktestConfig("bt-1", 1_000.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, calendar=calendar)).run(sessions, BuyOnce())

    assert len(result.fills) == 1
    assert result.fills[0].trade_time == day_two
