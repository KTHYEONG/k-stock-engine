"""Backtest contracts extracted from engine.py.

``BacktestAttribution`` and ``BacktestResult`` are the typed contracts
for backtest outcomes. Re-exports ``BacktestResult`` from engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BacktestAttribution:
    """Typed cost attribution for a backtest run.

    Separates base and stress cost components for gross-to-net bridge.
    """

    base_commission: float = 0.0
    base_tax: float = 0.0
    base_spread: float = 0.0
    base_impact: float = 0.0
    base_other: float = 0.0
    base_total: float = 0.0
    stress_commission: float = 0.0
    stress_tax: float = 0.0
    stress_spread: float = 0.0
    stress_impact: float = 0.0
    stress_other: float = 0.0
    stress_total: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    cost_drag_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of historical replay: ledger, fills, derived metrics.

    Re-exported from engine for module-level access.
    """

    ledger: tuple[object, ...] = field(default_factory=tuple)
    trades: tuple[object, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)
    attribution: BacktestAttribution | None = None
    terminal_equity: float = 0.0
    turnover: float = 0.0
    total_cost: float = 0.0
