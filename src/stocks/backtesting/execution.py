"""Backtest scenario execution extracted from engine.py.

``ScenarioExecutor`` manages the chronological execution of backtest
scenarios with fills, costs, and settlement.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ScenarioExecutor:
    """Chronological backtest scenario executor.

    Manages the lifecycle of one backtest scenario: initialization,
    step-by-step execution, fill resolution, cost attribution, and
    settlement.
    """

    def __init__(self, backtester: object) -> None:
        self._backtester = backtester

    def run(self, *args: object, **kwargs: object) -> object:
        """Delegate to the underlying backtester's run method."""
        return self._backtester.run(*args, **kwargs)  # type: ignore[attr-defined]
