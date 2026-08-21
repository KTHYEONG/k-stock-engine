"""Direct contract tests for the decomposed backtesting module."""
from __future__ import annotations

from src.stocks.backtesting.contracts import BacktestAttribution, BacktestResult


def test_backtest_attribution_cost_components_are_immutable() -> None:
    attribution = BacktestAttribution(base_commission=1.0, base_total=1.0)

    assert attribution.base_commission == 1.0
    assert attribution.base_total == 1.0


def test_backtest_result_has_empty_safe_defaults() -> None:
    result = BacktestResult()

    assert result.ledger == ()
    assert result.trades == ()
    assert result.metrics == {}
