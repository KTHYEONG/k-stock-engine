"""Trading transition evidence extracted from portfolio_constructor.py.

``TransitionEvidence`` captures the bounded evidence for one turnover
transition decision.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """Bounded evidence for one turnover transition decision.

    Captures the counts and reasons for entries, exits, and retains
    during a single rebalance cycle.
    """

    retained_count: int = 0
    entry_count: int = 0
    exit_count: int = 0
    entry_reasons: dict[str, int] | None = None
    exit_reasons: dict[str, int] | None = None
    turnover_bps: float = 0.0
    delta_cost_bps: float = 0.0
