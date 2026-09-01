"""ETF backtest result value types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EtfBacktestResult:
    """Aggregated backtest outcome for one market/universe."""

    market: str
    total_return_pct: float
    mdd_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    final_balance: float
    equity_curve: list[float]
    trades: list[dict[str, float | int]]
