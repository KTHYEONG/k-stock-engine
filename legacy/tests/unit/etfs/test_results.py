"""ETF backtest result value type tests."""
from __future__ import annotations

from legacy.etfs.backtesting.results import EtfBacktestResult


def make_result() -> EtfBacktestResult:
    return EtfBacktestResult(
        market="KOSPI",
        total_return_pct=12.5,
        mdd_pct=8.0,
        total_trades=10,
        win_rate=60.0,
        profit_factor=1.5,
        final_balance=11_250_000.0,
        equity_curve=[10_000_000.0, 10_500_000.0],
        trades=[{"entry_idx": 0, "exit_idx": 5, "pnl": 50000.0}],
    )


class TestEtfBacktestResult:
    def test_result_carries_typed_metrics(self) -> None:
        result = make_result()
        assert result.total_return_pct == 12.5
        assert result.total_trades == 10
        assert result.trades[0]["pnl"] == 50000.0

    def test_result_is_immutable_slots(self) -> None:
        result = make_result()
        try:
            result.total_trades = 99
            raised = False
        except AttributeError:
            raised = True
        assert raised
