"""Trading allocation decisions extracted from portfolio_constructor.py.

``AllocationDecision`` captures the typed output of the allocation planner.
``plan_target_allocations`` and ``plan_target_allocations_prepared`` are
thin facades delegating to the existing allocation logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

    from src.core.instruments import Instrument
    from src.core.portfolio import Allocation as _Allocation
    from src.core.portfolio import PortfolioSnapshot
    from legacy.stocks.trading.portfolio_constructor import (
        PreparedAllocationMarket,
        StockRiskPolicy,
    )


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    """Typed output of the allocation planner.

    Captures candidate/ranked/selected counts, cash reason, and the
    final allocation tuple.
    """

    candidate_count: int = 0
    ranked_count: int = 0
    selected_count: int = 0
    cash_reason: str = ""
    allocations: tuple[_Allocation, ...] = field(default_factory=tuple)
    exposure: float = 0.0
    gross_exposure: float = 0.0


def plan_target_allocations(
    panel: pl.DataFrame,
    instruments: dict[str, Instrument],
    portfolio: PortfolioSnapshot,
    policy: StockRiskPolicy,
) -> AllocationDecision:
    """Plan target allocations from a scores panel.

    Thin facade delegating to ``construct_target_allocations``.
    """
    from legacy.stocks.trading.portfolio_constructor import (
        construct_target_allocations,
    )

    allocations = construct_target_allocations(panel, instruments, portfolio, policy)
    selected = [a for a in allocations if a.target_value > 0]
    return AllocationDecision(
        candidate_count=len(allocations),
        ranked_count=len(allocations),
        selected_count=len(selected),
        cash_reason="target_met" if selected else "no_candidates",
        allocations=tuple(allocations),
    )


def plan_target_allocations_prepared(
    market: PreparedAllocationMarket,
    decision_index: int,
    score_overlay: np.ndarray,
    calibration_state: dict[str, object] | None,
    instruments: dict[str, Instrument],
    portfolio: PortfolioSnapshot,
    policy: StockRiskPolicy,
) -> AllocationDecision:
    """Plan target allocations from prepared market data.

    Thin facade delegating to ``construct_target_allocations_prepared``.
    """
    from legacy.stocks.trading.portfolio_constructor import (
        construct_target_allocations_prepared,
    )

    allocations = construct_target_allocations_prepared(
        market, decision_index, score_overlay, calibration_state,
        instruments, portfolio, policy,
    )
    selected = [a for a in allocations if a.target_value > 0]
    return AllocationDecision(
        candidate_count=len(allocations),
        ranked_count=len(allocations),
        selected_count=len(selected),
        cash_reason="target_met" if selected else "no_candidates",
        allocations=tuple(allocations),
    )
