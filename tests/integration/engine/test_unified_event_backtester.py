"""Unified event backtester integration tests."""
from __future__ import annotations


def test_backtester_multisession_replay_reconciles_t2_and_exact_nav() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    start = datetime(2024, 1, 2, tzinfo=UTC)
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.1, hard_participation_cap=0.2)
    calendar = SessionCalendar(tuple(start + timedelta(days=i) for i in range(7)))

    class EnterThenExit:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            offset = (context.decision_time.date() - start.date()).days
            if offset == 0:
                return (TradeIntent("buy", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, start + timedelta(days=1), "champion-v1", "entry", "buy-key", context.portfolio.account_snapshot_id),)
            if offset == 1:
                return (TradeIntent("sell", AssetKind.STOCK, instrument.instrument_id, 0.0, context.decision_time, start + timedelta(days=2), "champion-v1", "exit", "sell-key", context.portfolio.account_snapshot_id),)
            return ()

    sessions = tuple(BacktestSession(start + timedelta(days=i), start + timedelta(days=i, hours=6, minutes=30), (HistoricalBar(start + timedelta(days=i), instrument.instrument_id, 10.0, 10.0, 1_000_000.0, 0.02),), (), object()) for i in range(5))
    result = EventBacktester(BacktestConfig("bt-1", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, calendar, model)).run(sessions, EnterThenExit())

    assert len(result.daily_nav) == 5
    assert result.daily_nav[-1].nav == pytest.approx(100.0)
    assert result.daily_nav[-1].unsettled_cash == pytest.approx(0.0)


def test_backtester_actions_settlement_and_nav_identity() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.instruments import AssetKind, Instrument
    from src.core.ledger import LedgerActionType, LedgerCorporateAction
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession, EventBacktester
    from src.engine.decision import DecisionContext
    from src.engine.fill_model import ExecutionScenario, HistoricalBar, HistoricalFillModel
    from src.execution.domain.intents import TradeIntent

    start = datetime(2024, 1, 2, tzinfo=UTC)
    sessions_open = tuple(start + timedelta(days=offset) for offset in range(5))
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    costs = CostSchedule("fixture", (CostPoint(datetime(2020, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),))
    ticks = TickSizeSchedule((TickSizeRule("all", datetime(2020, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1.0),))
    model = HistoricalFillModel(costs, LiquiditySlippageModel(0.0, ticks), ExecutionScenario.IDEAL, target_participation_cap=0.01, hard_participation_cap=0.02)

    class EnterThenExit:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            if context.decision_time.date() == sessions_open[0].date():
                return (TradeIntent("buy", AssetKind.STOCK, instrument.instrument_id, 100.0, context.decision_time, sessions_open[1], "champion-v1", "entry", "buy-key", context.portfolio.account_snapshot_id),)
            if context.decision_time.date() == sessions_open[1].date():
                return (TradeIntent("sell", AssetKind.STOCK, instrument.instrument_id, 0.0, context.decision_time, sessions_open[2], "champion-v1", "exit", "sell-key", context.portfolio.account_snapshot_id),)
            return ()

    sessions = tuple(
        BacktestSession(
            session_open,
            session_open + timedelta(hours=6),
            (HistoricalBar(session_open, instrument.instrument_id, 5.0 if session_open >= sessions_open[2] else 10.0, 5.0 if session_open >= sessions_open[2] else 10.0, 1_000_000.0, 0.02),),
            (
                LedgerCorporateAction("split", instrument.instrument_id, LedgerActionType.SPLIT, session_open, 2.0, 0.0),
                LedgerCorporateAction("dividend", instrument.instrument_id, LedgerActionType.DIVIDEND, session_open, 1.0, 1.0),
            ) if session_open == sessions_open[2] else (),
            object(),
        )
        for session_open in sessions_open
    )
    config = BacktestConfig("identity", 100.0, {instrument.instrument_id: instrument}, ExecutionScenario.IDEAL, costs, SessionCalendar(sessions_open), model)

    result = EventBacktester(config).run(sessions, EnterThenExit())

    assert result.daily_nav[2].unsettled_cash == pytest.approx(100.0)
    assert result.daily_nav[4].settled_cash == pytest.approx(110.0)
    assert {entry.event_type for entry in result.journal} >= {"split", "dividend", "settlement"}
    for nav in result.daily_nav:
        assert nav.nav == pytest.approx(nav.settled_cash + nav.unsettled_cash + nav.marked_value)

