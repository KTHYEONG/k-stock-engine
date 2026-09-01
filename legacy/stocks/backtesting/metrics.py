"""Backtest metrics and attribution extracted from engine.py.

``build_backtest_attribution`` constructs the typed cost attribution
from ledger rows and trades.
"""
from __future__ import annotations

from collections.abc import Sequence

from legacy.stocks.backtesting.contracts import BacktestAttribution


def build_backtest_attribution(
    ledger: Sequence[object],
    trades: Sequence[object],
) -> BacktestAttribution:
    """Build typed cost attribution from ledger and trade sequences.

    Parameters
    ----------
    ledger:
        Sequence of ``BacktestLedgerRow`` objects.
    trades:
        Sequence of ``BacktestTrade`` objects.

    Returns
    -------
    BacktestAttribution
        Aggregated cost attribution.
    """
    base_commission = 0.0
    base_tax = 0.0
    base_spread = 0.0
    base_impact = 0.0
    stress_commission = 0.0
    stress_tax = 0.0
    stress_spread = 0.0
    stress_impact = 0.0

    for trade in trades:
        if hasattr(trade, "cost_breakdown"):
            cb = trade.cost_breakdown
            if hasattr(cb, "commission"):
                base_commission += cb.commission
            if hasattr(cb, "tax"):
                base_tax += cb.tax
            if hasattr(cb, "spread"):
                base_spread += cb.spread
            if hasattr(cb, "impact"):
                base_impact += cb.impact

    base_total = base_commission + base_tax + base_spread + base_impact

    return BacktestAttribution(
        base_commission=base_commission,
        base_tax=base_tax,
        base_spread=base_spread,
        base_impact=base_impact,
        base_total=base_total,
        stress_commission=stress_commission,
        stress_tax=stress_tax,
        stress_spread=stress_spread,
        stress_impact=stress_impact,
        stress_total=stress_commission + stress_tax + stress_spread + stress_impact,
    )
