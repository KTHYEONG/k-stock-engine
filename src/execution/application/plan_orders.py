"""Sealed-target intent planning for the next open."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from src.core.instruments import AssetKind
from src.core.time import KRX_TZ
from src.execution.domain.decision import TargetDecision
from src.execution.domain.intents import TradeIntent
from src.execution.domain.portfolio import PortfolioSnapshot


def plan_intents(
    decision: TargetDecision,
    snapshot: PortfolioSnapshot,
    *,
    nav_krw: int,
    reference_prices: Mapping[str, int],
    cash_buffer: float,
    rebalance_band: float,
    strategy_id: str,
    planning_time: datetime | None = None,
) -> tuple[TradeIntent, ...]:
    """Share-quantity intents moving the snapshot to sealed stock targets."""
    if isinstance(nav_krw, bool) or not isinstance(nav_krw, int) or nav_krw <= 0:
        raise ValueError("nav_krw must be a positive int")
    if isinstance(cash_buffer, bool) or not isinstance(cash_buffer, (int, float)):
        raise ValueError("cash_buffer must be numeric")
    if not 0.0 <= float(cash_buffer) < 1.0:
        raise ValueError("cash_buffer must be in [0, 1)")
    if isinstance(rebalance_band, bool) or not isinstance(rebalance_band, (int, float)):
        raise ValueError("rebalance_band must be numeric")
    if not math.isfinite(rebalance_band) or float(rebalance_band) < 0.0:
        raise ValueError("rebalance_band must be non-negative")
    if not strategy_id:
        raise ValueError("strategy_id must be non-empty")
    observed_at = decision.sealed_at if planning_time is None else planning_time
    if observed_at.utcoffset() is None or decision.sealed_at.utcoffset() is None:
        raise ValueError("planning and decision times must be timezone-aware")
    if observed_at < decision.sealed_at:
        raise ValueError("planning_time must not precede the sealed decision")
    snapshot.validate_as_of(observed_at)

    scale = 1.0 - float(cash_buffer)
    band = float(rebalance_band)
    session = decision.execution_session
    execution_time = datetime(session.year, session.month, session.day, 9, 0, tzinfo=KRX_TZ)
    if observed_at > execution_time:
        raise ValueError("planning_time must not follow execution_time")
    prefix = decision.hash[:16]
    held_kind: dict[str, AssetKind] = {}
    for position in snapshot.positions:
        held_kind[position.instrument.instrument_id] = position.instrument.asset_kind

    def _reference_price(instrument_id: str) -> int:
        if instrument_id not in reference_prices:
            raise ValueError(f"missing reference price for {instrument_id!r}")
        price = reference_prices[instrument_id]
        if isinstance(price, bool) or not isinstance(price, int) or price <= 0:
            raise ValueError(f"invalid reference price for {instrument_id!r}")
        if decision.reference.get(instrument_id) != price:
            raise ValueError(f"reference price differs from sealed decision for {instrument_id!r}")
        return int(price)

    intents: list[TradeIntent] = []
    for instrument_id in sorted(set(decision.targets) | {p.instrument.instrument_id for p in snapshot.positions}):
        price = _reference_price(instrument_id)
        held = snapshot.quantity_of(instrument_id)
        if instrument_id in decision.targets:
            weight = decision.targets[instrument_id]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                raise ValueError(f"invalid target weight for {instrument_id!r}")
            if float(weight) < 0.0:
                raise ValueError(f"negative target weight for {instrument_id!r}")
            target_quantity = math.floor(float(weight) * float(nav_krw) * scale / float(price))
        else:
            weight = 0.0
            target_quantity = 0
        delta = target_quantity - held
        if delta == 0:
            continue
        if (
            band > 0.0
            and held > 0
            and float(weight) > 0.0
            and target_quantity > 0
            and abs(delta) * float(price) < band * float(target_quantity) * float(price)
        ):
            continue
        kind = held_kind.get(instrument_id, AssetKind.STOCK)
        intents.append(
            TradeIntent(
                intent_id=f"{session.isoformat()}:{instrument_id}",
                asset_kind=kind,
                instrument_id=instrument_id,
                target_value=float(target_quantity * price),
                target_quantity=int(target_quantity),
                decision_time=decision.sealed_at,
                execution_time=execution_time,
                strategy_id=strategy_id,
                reason="sealed-target",
                idempotency_key=f"{prefix}:{instrument_id}",
                account_snapshot_id=snapshot.account_snapshot_id,
            )
        )
    return tuple(intents)
