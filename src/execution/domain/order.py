"""Execution domain contracts: TradeIntent, OrderRequest, OrderState."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

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


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderState(StrEnum):
    NEW = "NEW"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """The validated order handed to a ``BrokerPort``."""

    order_id: str
    asset_kind: AssetKind
    instrument_id: str
    side: OrderSide
    quantity: float
    price: float | None
    request_time: datetime
    idempotency_key: str
    intent_id: str

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True, slots=True)
class OrderStateRecord:
    """Full order state-machine record; never invents price or history."""

    order_id: str
    intent_id: str
    instrument_id: str
    asset_kind: AssetKind
    side: OrderSide
    state: OrderState
    submitted_quantity: float
    filled_quantity: float = 0.0
    filled_price: float | None = None
    reason: str = ""
