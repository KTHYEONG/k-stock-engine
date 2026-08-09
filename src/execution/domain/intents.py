"""Execution intent contract: ``TradeIntent`` is the source of truth here.

An intent represents a *target position*, not a signed currency order. It
carries a non-negative ``target_value`` and the account-snapshot identity used
to reconcile current holdings at execution. Buy/sell/exit sides are derived at
execution by comparing the target quantity with the broker-reconciled current
quantity.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from src.core.instruments import AssetKind

INTENT_SCHEMA_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """An approved target position for one instrument at a decision time.

    Execution validates session, account, tick/lot, cash/settlement, price
    guard, and duplicate intent before calling a broker. It does not calculate
    features or contain strategy exit rules.
    """

    intent_id: str
    asset_kind: AssetKind
    instrument_id: str
    target_value: float
    decision_time: datetime
    execution_time: datetime
    strategy_id: str
    reason: str
    idempotency_key: str
    account_snapshot_id: str
    reference_price_guard_bps: float | None = None

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")
        if not self.account_snapshot_id:
            raise ValueError("account_snapshot_id must be non-empty")
        if self.target_value < 0 or not math.isfinite(self.target_value):
            raise ValueError("target_value must be a non-negative finite number")
        if self.decision_time > self.execution_time:
            raise ValueError("decision_time must not be after execution_time")
        if (
            self.reference_price_guard_bps is not None
            and (self.reference_price_guard_bps < 0 or not math.isfinite(self.reference_price_guard_bps))
        ):
            raise ValueError("reference_price_guard_bps must be a non-negative finite number")


def read_v1_intent(payload: Mapping[str, object]) -> TradeIntent:
    """Migrate a serialized v1 intent (target-value order semantics).

    v1 payloads carried a signed ``target_value`` used as an order delta, no
    account snapshot identity, and required a positive value. The migration
    preserves the old absolute notional as the new target position and rejects
    an exit that cannot be reconciled: v1 exits (zero/negative) require an
    explicit ``account_snapshot_id`` because a target position cannot be
    guessed from local order state.
    """
    raw_value = payload.get("target_value")
    if not isinstance(raw_value, (int, float)):
        raise ValueError("v1 intent must carry a numeric target_value")
    target_value = float(raw_value)
    if target_value < 0:
        raise ValueError(
            "v1 negative target_value cannot be migrated without broker state"
        )
    account_snapshot_id = payload.get("account_snapshot_id")
    if not account_snapshot_id:
        raise ValueError(
            "v1 intent migration requires account_snapshot_id for reconciled exits"
        )
    decision_raw = payload["decision_time"]
    execution_raw = payload["execution_time"]
    if not isinstance(decision_raw, datetime):
        decision_time = datetime.fromisoformat(str(decision_raw))
    else:
        decision_time = decision_raw
    if not isinstance(execution_raw, datetime):
        execution_time = datetime.fromisoformat(str(execution_raw))
    else:
        execution_time = execution_raw
    return TradeIntent(
        intent_id=str(payload["intent_id"]),
        asset_kind=AssetKind(str(payload["asset_kind"])),
        instrument_id=str(payload["instrument_id"]),
        target_value=max(0.0, target_value),
        decision_time=decision_time,
        execution_time=execution_time,
        strategy_id=str(payload["strategy_id"]),
        reason=str(payload.get("reason", "migrated-v1")),
        idempotency_key=str(payload["idempotency_key"]),
        account_snapshot_id=str(account_snapshot_id),
    )
