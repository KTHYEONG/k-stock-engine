"""Application entry point for the unified event-driven backtester."""
from __future__ import annotations

from src.engine.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestSession,
    EventBacktester,
)
from src.engine.decision import StrategyDecisionPort


def run_backtest(
    config: BacktestConfig,
    sessions: tuple[BacktestSession, ...],
    strategy: StrategyDecisionPort,
) -> BacktestResult:
    """Run a configured historical replay through the shared engine."""
    return EventBacktester(config).run(sessions, strategy)
