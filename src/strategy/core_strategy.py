"""Deterministic Korean-equity compounding core (20-stock large-cap/low-vol)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

from src.core.time import SessionCalendar
from src.engine.decision import DecisionContext
from src.execution.domain.intents import TradeIntent
from src.strategy.portfolio import (
    ChampionPortfolioPolicy,
    PortfolioSecurityInput,
    construct_champion_portfolio,
)
from src.strategy.scoring import ChampionScoreRow
from src.strategy.selection import ChampionSelectionPolicy, select_champion_targets


@dataclass(frozen=True, slots=True)
class CoreStrategyPolicy:
    version: str = "korean-core-v1"
    score_policy_version: str = "korean-core-v1-scoring-v1"
    selection_policy_version: str = "korean-core-v1-selection-v1"
    max_positions: int = 20
    entry_rank: int = 20
    retention_rank: int = 40
    rebalance_sessions: int = 20
    market_cap_weight: float = 0.5
    inverse_volatility_weight: float = 0.5

    def __post_init__(self) -> None:
        if (
            self.version != "korean-core-v1"
            or self.score_policy_version != "korean-core-v1-scoring-v1"
            or self.selection_policy_version != "korean-core-v1-selection-v1"
            or self.max_positions != 20
            or self.entry_rank != 20
            or self.retention_rank != 40
            or self.rebalance_sessions != 20
            or self.market_cap_weight != 0.5
            or self.inverse_volatility_weight != 0.5
            or abs(float(self.market_cap_weight) + float(self.inverse_volatility_weight) - 1.0) >= 1e-12
        ):
            raise ValueError("CoreStrategyPolicy constants are immutable")


def score_core_candidates(
    *,
    candidate_ids: tuple[str, ...],
    decision_time: datetime,
    market_caps: Mapping[str, float],
    volatilities: Mapping[str, float],
    policy: CoreStrategyPolicy,
) -> tuple[ChampionScoreRow, ...]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if not candidate_ids:
        raise ValueError("candidate_ids must be non-empty")
    seen: set[str] = set()
    for iid in candidate_ids:
        if not iid or not iid.strip():
            raise ValueError("instrument_id must be non-empty")
        if iid in seen:
            raise ValueError(f"duplicate instrument_id {iid!r}")
        seen.add(iid)
        if iid not in market_caps or iid not in volatilities:
            raise ValueError(f"missing snapshot inputs for {iid!r}")
        cap = float(market_caps[iid])
        vol = float(volatilities[iid])
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"non-positive market_cap for {iid!r}")
        if not math.isfinite(vol) or vol <= 0:
            raise ValueError(f"non-positive volatility for {iid!r}")
    n = len(candidate_ids)
    by_cap = sorted(candidate_ids, key=lambda v: (-float(market_caps[v]), v))
    cap_rank = {iid: idx + 1 for idx, iid in enumerate(by_cap)}
    by_vol = sorted(candidate_ids, key=lambda v: (float(volatilities[v]), v))
    vol_rank = {iid: idx + 1 for idx, iid in enumerate(by_vol)}
    scored: list[tuple[str, float]] = []
    for iid in candidate_ids:
        cap_norm = (n - cap_rank[iid] + 1) / n
        vol_norm = (n - vol_rank[iid] + 1) / n
        score = float(policy.market_cap_weight) * cap_norm + float(policy.inverse_volatility_weight) * vol_norm
        scored.append((iid, score))
    ordered = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
    rows: list[ChampionScoreRow] = []
    for rank, (iid, score) in enumerate(ordered, start=1):
        rows.append(
            ChampionScoreRow(
                decision_session=decision_time,
                instrument_id=iid,
                eligible=True,
                champion_score=float(score),
                rank=rank,
                exclusion_reasons=(),
                feature_policy_version="korean-core-v1-features-v1",
                score_policy_version=policy.score_policy_version,
            )
        )
    return tuple(rows)


def is_core_rebalance_session(
    *, calendar: SessionCalendar, decision_time: datetime, policy: CoreStrategyPolicy
) -> bool:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    target = decision_time.date()
    index = -1
    for i, session in enumerate(calendar.sessions):
        if session.date() == target:
            index = i
            break
    if index < 0:
        raise ValueError(f"decision_time {decision_time.isoformat()} not on calendar")
    return index % int(policy.rebalance_sessions) == 0


class CoreStrategy:
    def __init__(
        self,
        *,
        eligible_by_session: Mapping[date, tuple[str, ...]],
        calendar: SessionCalendar,
        policy: CoreStrategyPolicy | None = None,
        portfolio_policy: ChampionPortfolioPolicy | None = None,
    ) -> None:
        self._eligible_by_session: dict[date, tuple[str, ...]] = dict(eligible_by_session)
        self._calendar = calendar
        self._policy = policy if policy is not None else CoreStrategyPolicy()
        self._selection_policy = ChampionSelectionPolicy(
            version=self._policy.selection_policy_version,
            required_score_policy_version=self._policy.score_policy_version,
            max_positions=self._policy.max_positions,
            entry_rank=self._policy.entry_rank,
            retention_rank=self._policy.retention_rank,
        )
        if portfolio_policy is not None:
            self._portfolio_policy = portfolio_policy
        else:
            self._portfolio_policy = ChampionPortfolioPolicy(
                required_selection_policy_version=self._policy.selection_policy_version
            )

    def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
        decision_time = context.decision_time
        if not is_core_rebalance_session(
            calendar=self._calendar, decision_time=decision_time, policy=self._policy
        ):
            return ()
        candidates = self._eligible_by_session.get(decision_time.date(), ())
        if not candidates:
            return ()
        snapshot = cast(Mapping[str, Any], context.market_snapshot)
        raw_caps = snapshot["market_caps"]
        market_caps = {str(k): float(v) for k, v in dict(cast(Mapping[str, Any], raw_caps)).items()}
        volatilities = {str(k): float(v) for k, v in dict(cast(Mapping[str, Any], snapshot["volatilities"])).items()}
        scores = score_core_candidates(
            candidate_ids=tuple(candidates),
            decision_time=decision_time,
            market_caps=market_caps,
            volatilities=volatilities,
            policy=self._policy,
        )
        selection = select_champion_targets(
            scores, context.portfolio, decision_time=decision_time, policy=self._selection_policy
        )
        raw_marks = cast(Mapping[str, Any], snapshot["mark_prices"])
        mark_prices = {str(k): float(v) for k, v in dict(raw_marks).items()}
        adtv20 = cast(Mapping[str, Any], snapshot["adtv20"])
        sectors = cast(Mapping[str, Any], snapshot["sectors"])
        instruments = cast(Mapping[str, Any], snapshot["instruments"])
        market_volatility = float(snapshot["market_volatility"])
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
        execution_time = self._calendar.advance(decision_time, 1)
        return tuple(
            TradeIntent(
                intent_id=f"core-{target.allocation.instrument.instrument_id}-{session_tag}",
                asset_kind=target.allocation.instrument.asset_kind,
                instrument_id=target.allocation.instrument.instrument_id,
                target_value=float(target.allocation.target_value),
                decision_time=decision_time,
                execution_time=execution_time,
                strategy_id="core-v1",
                reason=target.allocation.reason,
                idempotency_key=f"core_{target.allocation.instrument.instrument_id}_{session_tag}",
                account_snapshot_id=account_id,
            )
            for target in result.targets
        )
