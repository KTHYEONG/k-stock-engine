"""Production strategy orchestration entry points."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from src.core.portfolio import PortfolioSnapshot
from src.strategy.portfolio import (
    ChampionPortfolioPolicy,
    ChampionPortfolioResult,
    PortfolioSecurityInput,
    construct_champion_portfolio,
)
from src.strategy.selection import ChampionSelectionResult


def build_champion_portfolio(
    selection: ChampionSelectionResult,
    security_inputs: tuple[PortfolioSecurityInput, ...],
    portfolio: PortfolioSnapshot,
    mark_prices: Mapping[str, float],
    market_volatility: float,
    *,
    decision_time: datetime,
    policy: ChampionPortfolioPolicy = ChampionPortfolioPolicy(),  # noqa: B008
) -> ChampionPortfolioResult:
    """Build constrained targets from a selected universe and reconciled state."""
    return construct_champion_portfolio(
        selection,
        security_inputs,
        portfolio,
        mark_prices,
        market_volatility,
        decision_time=decision_time,
        policy=policy,
    )
