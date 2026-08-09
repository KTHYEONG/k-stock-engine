"""Execution intent contract: ``TradeIntent`` is the source of truth here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.instruments import AssetKind


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """An approved intent to trade one instrument at a target value.

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

    def __post_init__(self) -> None:
        if self.target_value <= 0:
            raise ValueError("target_value must be positive")
        if self.decision_time > self.execution_time:
            raise ValueError("decision_time must not be after execution_time")
