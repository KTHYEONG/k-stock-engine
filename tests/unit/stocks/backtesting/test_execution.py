from __future__ import annotations

from src.stocks.backtesting.execution import ScenarioExecutor


def test_scenario_executor_delegates_run() -> None:
    class Runner:
        def run(self, value: int) -> int:
            return value + 1

    assert ScenarioExecutor(Runner()).run(1) == 2
