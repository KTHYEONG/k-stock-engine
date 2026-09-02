"""Unified event backtester integration tests."""
from __future__ import annotations


def test_backtester_multisession_replay_reconciles_t2_and_exact_nav() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.instruments import AssetKind, Instrument
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar
    from src.execution.domain.intents import TradeIntent

    start = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")

    class EnterThenExit:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            offset = (context.decision_time.date() - start.date()).days
            if offset == 0:
                return (TradeIntent("buy", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, start + timedelta(days=1), "champion-v1", "entry", "buy-key", context.portfolio.account_snapshot_id),)
            if offset == 1:
                return (TradeIntent("sell", AssetKind.STOCK, instrument.instrument_id, 0.0, context.decision_time, start + timedelta(days=2), "champion-v1", "exit", "sell-key", context.portfolio.account_snapshot_id),)
            return ()

    sessions = tuple(BacktestSession(start + timedelta(days=i), start + timedelta(days=i, hours=6, minutes=30), (HistoricalBar(start + timedelta(days=i), instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()) for i in range(5))
    result = EventBacktester(BacktestConfig("bt-1", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL)).run(sessions, EnterThenExit())

    assert len(result.daily_nav) == 5
    assert result.daily_nav[-1].nav == pytest.approx(100.0)
    assert result.daily_nav[-1].unsettled_cash == pytest.approx(0.0)
