"""Execution order contracts: OrderRequest, OrderState, and records."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.instruments import AssetKind


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
