from __future__ import annotations


def test_run_backtest_delegates_to_event_engine() -> None:
    from datetime import UTC, datetime, timedelta

    from src.engine.backtest import BacktestConfig, BacktestSession
    from src.engine.fill_model import ExecutionScenario
    from src.engine.runner import run_backtest

    now = datetime(2024, 1, 2, tzinfo=UTC)

    class NoTrade:
        def decide(self, context: object) -> tuple[object, ...]:
            del context
            return ()

    result = run_backtest(
        BacktestConfig("runner-test", 100.0, {}, ExecutionScenario.IDEAL),
        (BacktestSession(now, now + timedelta(hours=6), (), (), object()),),
        NoTrade(),
    )

    assert len(result.daily_nav) == 1
