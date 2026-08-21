"""Tests for backtesting decomposition and attribution.

Scenarios:
- BACKTEST_05: Attribution and result contracts are correctly structured.
"""
from __future__ import annotations

from src.stocks.backtesting.contracts import BacktestAttribution, BacktestResult
from src.stocks.backtesting.metrics import build_backtest_attribution


class TestBacktestAttribution:
    """BacktestAttribution captures typed cost components."""

    def test_default_attribution(self) -> None:
        attr = BacktestAttribution()
        assert attr.base_commission == 0.0
        assert attr.base_total == 0.0
        assert attr.gross_return == 0.0

    def test_attribution_with_values(self) -> None:
        attr = BacktestAttribution(
            base_commission=100.0,
            base_tax=50.0,
            base_spread=25.0,
            base_impact=10.0,
            base_total=185.0,
            gross_return=0.15,
            net_return=0.12,
            cost_drag_bps=30.0,
        )
        assert attr.base_total == 185.0
        assert attr.net_return == 0.12


class TestBacktestResult:
    """BacktestResult captures replay outcome."""

    def test_default_result(self) -> None:
        result = BacktestResult()
        assert result.ledger == ()
        assert result.trades == ()
        assert result.metrics == {}

    def test_result_with_attribution(self) -> None:
        attr = BacktestAttribution(base_total=100.0)
        result = BacktestResult(
            attribution=attr,
            terminal_equity=1_000_000.0,
            turnover=3.5,
            total_cost=100.0,
        )
        assert result.attribution is not None
        assert result.attribution.base_total == 100.0
        assert result.terminal_equity == 1_000_000.0


class TestBuildBacktestAttribution:
    """build_backtest_attribution constructs attribution from ledger and trades."""

    def test_empty_trades(self) -> None:
        attr = build_backtest_attribution([], [])
        assert attr.base_total == 0.0
        assert attr.stress_total == 0.0

    def test_trades_with_cost_breakdown(self) -> None:
        class MockCostBreakdown:
            def __init__(self, commission: float, tax: float, spread: float, impact: float) -> None:
                self.commission = commission
                self.tax = tax
                self.spread = spread
                self.impact = impact

        class MockTrade:
            def __init__(self, cb: MockCostBreakdown) -> None:
                self.cost_breakdown = cb

        trades = [
            MockTrade(MockCostBreakdown(10.0, 5.0, 2.0, 1.0)),
            MockTrade(MockCostBreakdown(20.0, 10.0, 4.0, 2.0)),
        ]
        attr = build_backtest_attribution([], trades)
        assert attr.base_commission == 30.0
        assert attr.base_tax == 15.0
        assert attr.base_spread == 6.0
        assert attr.base_impact == 3.0
        assert attr.base_total == 54.0
