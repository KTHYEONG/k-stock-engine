"""Session-driven Champion v1 strategy behind StrategyDecisionPort."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, cast

from src.engine.decision import DecisionContext
from src.execution.domain.intents import TradeIntent
from src.strategy.portfolio import (
    ChampionPortfolioPolicy,
    PortfolioSecurityInput,
    construct_champion_portfolio,
)
from src.strategy.scoring import ChampionScoreRow
from src.strategy.selection import ChampionSelectionPolicy, select_champion_targets


class ChampionStrategy:
    """Monthly-rebalanced Champion v1 decision policy (session-by-session)."""

    def __init__(
        self,
        *,
        scores_by_session: Mapping[date, tuple[ChampionScoreRow, ...]],
        selection_policy: ChampionSelectionPolicy | None = None,
        portfolio_policy: ChampionPortfolioPolicy | None = None,
        rebalance_frequency: str = 'monthly',
    ) -> None:
        self._scores_by_session: dict[date, tuple[ChampionScoreRow, ...]] = dict(scores_by_session)
        self._selection_policy = selection_policy if selection_policy is not None else ChampionSelectionPolicy()
        self._portfolio_policy = portfolio_policy if portfolio_policy is not None else ChampionPortfolioPolicy()
        self.rebalance_frequency = rebalance_frequency
        self._last_rebalance_month: tuple[int, int] | None = None

    def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
        decision_time = context.decision_time
        month_key = (decision_time.year, decision_time.month)
        scores = self._scores_by_session.get(decision_time.date())
        # Missing score rows keep holdings safely; intra-month sessions skip.
        if self._last_rebalance_month == month_key or not scores:
            return ()
        selection = select_champion_targets(
            scores, context.portfolio, decision_time=decision_time, policy=self._selection_policy
        )
        snapshot = cast(Mapping[str, Any], context.market_snapshot)
        raw_marks = cast(Mapping[str, Any], snapshot['mark_prices'])
        mark_prices = {str(k): float(v) for k, v in dict(raw_marks).items()}
        volatilities = cast(Mapping[str, Any], snapshot['volatilities'])
        adtv20 = cast(Mapping[str, Any], snapshot['adtv20'])
        sectors = cast(Mapping[str, Any], snapshot['sectors'])
        instruments = cast(Mapping[str, Any], snapshot['instruments'])
        market_volatility = float(snapshot['market_volatility'])
        held_ids = [p.instrument.instrument_id for p in context.portfolio.positions]
        ordered_ids = tuple(dict.fromkeys((*selection.selected_instrument_ids, *held_ids)))
        security_inputs = tuple(
            PortfolioSecurityInput(
                instrument=instruments[iid],
                sector=str(sectors[iid]),
                annualized_volatility=float(volatilities[iid]),
                adtv20=float(adtv20[iid]),
            )
            for iid in ordered_ids
        )
        # Caps, vol scaling, and PortfolioExclusionReason handling live in construct.
        result = construct_champion_portfolio(
            selection,
            security_inputs,
            context.portfolio,
            mark_prices,
            market_volatility,
            decision_time=decision_time,
            policy=self._portfolio_policy,
        )
        account_id = context.portfolio.account_snapshot_id
        session_tag = decision_time.date().isoformat()
        intents = tuple(
            TradeIntent(
                intent_id=f'champion-{target.allocation.instrument.instrument_id}-{session_tag}',
                asset_kind=target.allocation.instrument.asset_kind,
                instrument_id=target.allocation.instrument.instrument_id,
                target_value=float(target.allocation.target_value),
                decision_time=decision_time,
                execution_time=decision_time,
                strategy_id='champion-v1',
                reason=target.allocation.reason,
                idempotency_key=f'champion_{target.allocation.instrument.instrument_id}_{session_tag}',
                account_snapshot_id=account_id,
            )
            for target in result.targets
        )
        self._last_rebalance_month = month_key
        return intents
