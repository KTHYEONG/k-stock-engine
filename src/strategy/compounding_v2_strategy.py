"""Deterministic Korean-equity compounding strategy v2 (PIT champion-score hysteresis)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

from src.core.instruments import AssetKind, Instrument
from src.core.time import SessionCalendar
from src.engine.decision import DecisionContext
from src.execution.domain.intents import TradeIntent
from src.strategy.scoring import ChampionScoreRow


@dataclass(frozen=True, slots=True)
class CompoundingV2Policy:
    version: str = "compounding-v2"
    max_positions: int = 12
    entry_rank: int = 12
    retention_rank: int = 24
    volatility_sessions: int = 60
    security_weight_cap: float = 0.10
    selection_rebalance_sessions: int = 20
    risk_rebalance_sessions: int = 10
    short_sma_sessions: int = 100
    long_sma_sessions: int = 200
    target_market_volatility: float = 0.15

    def __post_init__(self) -> None:
        if (
            self.version != "compounding-v2"
            or self.max_positions != 12
            or self.entry_rank != 12
            or self.retention_rank != 24
            or self.volatility_sessions != 60
            or self.security_weight_cap != 0.10
            or self.selection_rebalance_sessions != 20
            or self.risk_rebalance_sessions != 10
            or self.short_sma_sessions != 100
            or self.long_sma_sessions != 200
            or self.target_market_volatility != 0.15
        ):
            raise ValueError("CompoundingV2Policy constants are immutable")


def select_compounding_v2_weights(
    *,
    scores: tuple[ChampionScoreRow, ...],
    held_ids: tuple[str, ...],
    marks: Mapping[str, float],
    volatilities: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    decision_time: datetime,
    policy: CompoundingV2Policy,
) -> dict[str, float]:
    if not scores:
        raise ValueError("scores must be non-empty")
    sessions = {row.decision_session for row in scores}
    if len(sessions) != 1: raise ValueError("scores must share exactly one decision_session")  # noqa: E701
    session = next(iter(sessions))
    if session > decision_time:
        raise ValueError("decision_session must not be after decision_time")
    if any(not row.instrument_id or not row.instrument_id.strip() for row in scores): raise ValueError("instrument_id must be non-empty")  # noqa: E701
    if len({row.instrument_id for row in scores}) != len(scores):
        raise ValueError("duplicate instrument_id in scores")
    if any(row.eligible and (row.champion_score is None or not math.isfinite(float(row.champion_score)) or not isinstance(row.rank, int) or isinstance(row.rank, bool) or row.rank <= 0) for row in scores):
        raise ValueError("invalid score/rank state for eligible row")
    if any(not row.eligible and (row.champion_score is not None or row.rank is not None) for row in scores): raise ValueError("invalid score/rank state for ineligible row")  # noqa: E701
    held_set = set(held_ids)
    valid = [
        row
        for row in scores
        if row.eligible
        and row.instrument_id in marks
        and math.isfinite(float(marks[row.instrument_id]))
        and float(marks[row.instrument_id]) > 0
        and row.instrument_id in volatilities
        and math.isfinite(float(volatilities[row.instrument_id]))
        and float(volatilities[row.instrument_id]) > 0
        and row.instrument_id in instruments
        and instruments[row.instrument_id].asset_kind == AssetKind.STOCK
    ]
    retained = sorted(
        (row for row in valid if row.instrument_id in held_set and cast(int, row.rank) <= policy.retention_rank),
        key=lambda row: (cast(int, row.rank), row.instrument_id),
    )
    newcomers = sorted(
        (row for row in valid if row.instrument_id not in held_set and cast(int, row.rank) <= policy.entry_rank),
        key=lambda row: (cast(int, row.rank), row.instrument_id),
    )
    selected = sorted(
        (*retained, *newcomers[: max(0, policy.max_positions - len(retained))]),
        key=lambda row: (cast(int, row.rank), row.instrument_id),
    )[: policy.max_positions]
    if len(selected) < policy.max_positions:
        return {}
    weight = 1.0 / float(len(selected))
    return {row.instrument_id: weight for row in selected}


class CompoundingV2Strategy:
    def __init__(
        self,
        *,
        scores_by_session: Mapping[date, tuple[ChampionScoreRow, ...]],
        calendar: SessionCalendar,
        policy: CompoundingV2Policy | None = None,
    ) -> None:
        self._scores_by_session: dict[date, tuple[ChampionScoreRow, ...]] = dict(scores_by_session)
        self._calendar = calendar
        self._policy = policy if policy is not None else CompoundingV2Policy()
        self._frozen_weights: dict[str, float] = {}
        self._entries_deferred = False

    @staticmethod
    def _is_positive_finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0

    def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
        decision_time = context.decision_time
        dates = [session.date() for session in self._calendar.sessions]
        index = dates.index(decision_time.date())
        policy = self._policy
        if index % policy.risk_rebalance_sessions != 0:
            return ()
        snapshot = cast(Mapping[str, Any], context.market_snapshot)
        marks = {str(key): float(value) for key, value in dict(snapshot["mark_prices"]).items()}
        vols = {str(key): float(value) for key, value in dict(snapshot["volatilities"]).items()}
        instruments = cast(Mapping[str, Instrument], snapshot["instruments"])
        portfolio = context.portfolio
        held_ids = tuple(position.instrument.instrument_id for position in portfolio.positions)
        if index % policy.selection_rebalance_sessions == 0:
            rows: tuple[ChampionScoreRow, ...] = ()
            for day in sorted(self._scores_by_session, reverse=True):
                if day <= decision_time.date():
                    batch = self._scores_by_session[day]
                    if batch and all(row.decision_session < decision_time for row in batch):
                        rows = batch
                        break
            updated = select_compounding_v2_weights(scores=rows, held_ids=held_ids, marks=marks, volatilities=vols, instruments=instruments, decision_time=decision_time, policy=policy) if rows else {}
            if updated != self._frozen_weights:
                self._frozen_weights = updated
                self._entries_deferred = True
        level = snapshot.get("market_index_level")
        sma_short = snapshot.get("market_index_sma100")
        sma_long = snapshot.get("market_index_sma200")
        market_vol = snapshot.get("market_volatility")
        trend_on = self._is_positive_finite(level) and self._is_positive_finite(sma_short) and self._is_positive_finite(sma_long) and float(cast(Any, level)) >= float(cast(Any, sma_short)) and float(cast(Any, level)) >= float(cast(Any, sma_long))
        risk_on = bool(trend_on) and self._is_positive_finite(market_vol)
        held = {position.instrument.instrument_id: position.instrument for position in portfolio.positions}
        frozen = self._frozen_weights
        if not risk_on or not frozen:
            targets = dict.fromkeys(held, 0.0)
        elif self._entries_deferred and float(portfolio.unsettled_cash) > 0:
            targets = {iid: float(portfolio.quantity_of(iid)) * marks[iid] if iid in frozen else 0.0 for iid in sorted(held)}
        else:
            nav = portfolio.equity(marks)
            exposure = min(1.0, float(policy.target_market_volatility) / float(cast(Any, market_vol)))
            investable_nav = nav * exposure
            targets = {iid: float(frozen.get(iid, 0.0)) * investable_nav for iid in sorted(set(frozen) | set(held))}
            self._entries_deferred = False
        execution_time = self._calendar.advance(decision_time, 1)
        account_id = portfolio.account_snapshot_id
        tag = decision_time.date().isoformat()
        all_instruments = {str(key): value for key, value in dict(instruments).items()} | dict(held)
        return tuple(
            TradeIntent(
                intent_id=f"compounding-v2-{iid}-{tag}",
                asset_kind=all_instruments[iid].asset_kind,
                instrument_id=iid,
                target_value=float(targets[iid]),
                decision_time=decision_time,
                execution_time=execution_time,
                strategy_id="compounding-v2",
                reason="compounding-v2-target" if targets[iid] > 0 else "compounding-v2-liquidate",
                idempotency_key=f"compounding-v2_{iid}_{tag}",
                account_snapshot_id=account_id,
            )
            for iid in sorted(targets)
        )
