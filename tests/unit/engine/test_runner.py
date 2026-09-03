from __future__ import annotations


def test_run_backtest_delegates_to_event_engine() -> None:
    from datetime import UTC, datetime, timedelta

    from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
    from src.core.time import SessionCalendar
    from src.engine.backtest import BacktestConfig, BacktestSession
    from src.engine.fill_model import ExecutionScenario, HistoricalFillModel
    from src.engine.runner import run_backtest

    now = datetime(2024, 1, 2, tzinfo=UTC)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    costs = CostSchedule("runner-fixture", (CostPoint(now, 0.0, 0.0, 0.0, 0),))
    ticks = TickSizeSchedule((TickSizeRule("all", now, 0.0, float("inf"), 1.0),))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.0, ticks),
        ExecutionScenario.IDEAL,
        target_participation_cap=0.01,
        hard_participation_cap=0.02,
    )
    result = run_backtest(
        BacktestConfig(
            "runner-test",
            100.0,
            {},
            ExecutionScenario.IDEAL,
            costs,
            SessionCalendar((now,)),
            fill_model,
        ),
        (BacktestSession(now, now + timedelta(hours=6), (), (), object()),),
        NoTrade(),
    )

    assert len(result.daily_nav) == 1
