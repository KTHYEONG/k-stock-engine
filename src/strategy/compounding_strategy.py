"""Deterministic Korean-equity compounding strategy (compounding-v1)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

from src.core.instruments import AssetKind
from src.core.time import SessionCalendar
from src.engine.decision import DecisionContext
from src.execution.domain.intents import TradeIntent


@dataclass(frozen=True, slots=True)
class CompoundingStrategyPolicy:
    version: str = "compounding-v1"
    max_positions: int = 15
    volatility_sessions: int = 60
    security_weight_cap: float = 0.075
    selection_rebalance_sessions: int = 40
    risk_rebalance_sessions: int = 10
    short_sma_sessions: int = 100
    long_sma_sessions: int = 200
    execution_cash_buffer: float = 0.30

    def __post_init__(self) -> None:
        if (
            self.version != "compounding-v1"
            or self.max_positions != 15
            or self.volatility_sessions != 60
            or self.security_weight_cap != 0.075
            or self.selection_rebalance_sessions != 40
            or self.risk_rebalance_sessions != 10
            or self.short_sma_sessions != 100
            or self.long_sma_sessions != 200
            or self.execution_cash_buffer != 0.30
        ):
            raise ValueError("CompoundingStrategyPolicy constants are immutable")


def capped_inverse_volatility_weights(
    *,
    instrument_ids: tuple[str, ...],
    volatilities: Mapping[str, float],
    policy: CompoundingStrategyPolicy,
) -> tuple[tuple[str, float], ...]:
    seen: set[str] = set()
    for iid in instrument_ids:
        if not iid or not iid.strip() or iid in seen:
            raise ValueError(f"duplicate instrument_id {iid!r}")
        seen.add(iid)
    vols: list[float] = []
    for iid in instrument_ids:
        raw = volatilities.get(iid, float("nan"))
        vol = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else float("nan")
        if not math.isfinite(vol) or vol <= 0:
            raise ValueError(f"invalid volatility for {iid!r}")
        vols.append(vol)
    cap = float(policy.security_weight_cap)
    raw_weights = [1.0 / vol for vol in vols]
    total = sum(raw_weights)
    weights = [value / total for value in raw_weights]
    capped_flags = [False] * len(weights)
    for _ in range(len(weights) + 1):
        over = [not flag and weight > cap for weight, flag in zip(weights, capped_flags, strict=True)]
        if not any(over):
            break
        capped_flags = [flag or is_over for flag, is_over in zip(capped_flags, over, strict=True)]
        fixed = float(sum(capped_flags)) * cap
        remaining = sum(weight for weight, flag in zip(weights, capped_flags, strict=True) if not flag)
        weights = [cap if flag else weight * (1.0 - fixed) / remaining for weight, flag in zip(weights, capped_flags, strict=True)]
    order = sorted(range(len(instrument_ids)), key=lambda pos: instrument_ids[pos])
    return tuple((instrument_ids[pos], float(weights[pos])) for pos in order)


class CompoundingStrategy:
    def __init__(
        self,
        *,
        eligible_by_session: Mapping[date, tuple[str, ...]],
        calendar: SessionCalendar,
        policy: CompoundingStrategyPolicy | None = None,
    ) -> None:
        self._eligible_by_session: dict[date, tuple[str, ...]] = dict(eligible_by_session)
        self._calendar = calendar
        self._policy = policy if policy is not None else CompoundingStrategyPolicy()
        self._frozen_weights: dict[str, float] = {}
        self._entries_deferred = False

    @staticmethod
    def _is_positive_finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0

    def _select_weights(
        self,
        decision_time: datetime,
        candidates: tuple[str, ...],
        marks: dict[str, float],
        caps: dict[str, float],
        vols: dict[str, float],
        instruments: Mapping[str, Any],
    ) -> dict[str, float]:
        valid = [
            iid
            for iid in candidates
            if iid in marks
            and iid in caps
            and iid in vols
            and iid in instruments
            and math.isfinite(marks[iid])
            and marks[iid] > 0
            and math.isfinite(caps[iid])
            and caps[iid] > 0
            and math.isfinite(vols[iid])
            and vols[iid] > 0
            and instruments[iid].asset_kind == AssetKind.STOCK
        ]
        ranked = sorted(valid, key=lambda iid: (-caps[iid], iid))[: self._policy.max_positions]
        return {} if len(ranked) < self._policy.max_positions else dict(capped_inverse_volatility_weights(instrument_ids=tuple(ranked), volatilities=vols, policy=self._policy))

    def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
        decision_time = context.decision_time
        dates = [session.date() for session in self._calendar.sessions]
        index = dates.index(decision_time.date())
        policy = self._policy
        if index % policy.risk_rebalance_sessions != 0:
            return ()
        snapshot = cast(Mapping[str, Any], context.market_snapshot)
        marks = {str(key): float(value) for key, value in dict(snapshot["mark_prices"]).items()}
        caps = {str(key): float(value) for key, value in dict(snapshot["market_caps"]).items()}
        vols = {str(key): float(value) for key, value in dict(snapshot["volatilities"]).items()}
        instruments = cast(Mapping[str, Any], snapshot["instruments"])
        if index % policy.selection_rebalance_sessions == 0:
            self._frozen_weights = self._select_weights(decision_time, tuple(self._eligible_by_session.get(decision_time.date(), ())), marks, caps, vols, instruments)
        level = snapshot.get("market_index_level")
        sma_short = snapshot.get("market_index_sma100")
        sma_long = snapshot.get("market_index_sma200")
        risk_on = self._is_positive_finite(level) and self._is_positive_finite(sma_short) and self._is_positive_finite(sma_long) and float(cast(Any, level)) >= float(cast(Any, sma_short)) and float(cast(Any, level)) >= float(cast(Any, sma_long))
        held = {position.instrument.instrument_id: position.instrument for position in context.portfolio.positions}
        frozen = self._frozen_weights
        nav = context.portfolio.equity(marks) if risk_on and frozen else 0.0
        investable_nav = nav * (1.0 - self._policy.execution_cash_buffer)
        selection_event = index % policy.selection_rebalance_sessions == 0
        if selection_event and held:
            self._entries_deferred = True
        if not risk_on or not frozen:
            targets = dict.fromkeys(held, 0.0)
        elif self._entries_deferred:
            targets = {
                iid: float(context.portfolio.quantity_of(iid)) * marks[iid]
                if iid in frozen
                else 0.0
                for iid in sorted(held)
            }
            if not selection_event:
                self._entries_deferred = False
        else:
            targets = {
                iid: frozen.get(iid, 0.0) * investable_nav
                for iid in sorted(set(frozen) | set(held))
            }
        execution_time = self._calendar.advance(decision_time, 1)
        account_id = context.portfolio.account_snapshot_id
        tag = decision_time.date().isoformat()
        all_instruments = {str(key): value for key, value in dict(instruments).items()} | dict(held)
        return tuple(
            TradeIntent(
                intent_id=f"compounding-{iid}-{tag}",
                asset_kind=all_instruments[iid].asset_kind,
                instrument_id=iid,
                target_value=float(targets[iid]),
                decision_time=decision_time,
                execution_time=execution_time,
                strategy_id="compounding-v1",
                reason="compounding-target" if targets[iid] > 0 else "compounding-liquidate",
                idempotency_key=f"compounding_{iid}_{tag}",
                account_snapshot_id=account_id,
            )
            for iid in sorted(targets)
        )
