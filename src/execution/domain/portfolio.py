"""Portfolio primitives shared by stock and ETF subsystems.

Fill-side cost contracts live in ``core.costs``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from src.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class Position:
    """A current holding of one instrument."""

    instrument: Instrument
    quantity: float
    average_cost: float

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if not math.isfinite(self.quantity):
            raise ValueError("quantity must be finite")


@dataclass(frozen=True, slots=True)
class Allocation:
    """A target allocation to one instrument."""

    instrument: Instrument
    target_value: float
    reason: str = ""
    target_quantity: int | None = None

    def __post_init__(self) -> None:
        if self.target_value < 0 or not math.isfinite(self.target_value):
            raise ValueError("target_value must be a non-negative finite number")
        if self.target_quantity is not None:
            if isinstance(self.target_quantity, bool) or not isinstance(self.target_quantity, int) or self.target_quantity < 0:
                raise ValueError("target_quantity must be a non-negative integer when provided")
            if self.target_quantity % self.instrument.lot_size != 0:
                raise ValueError("target_quantity must be a multiple of lot_size")


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Immutable reconciled account state used by a trading cycle.

    ``positions`` are broker-confirmed holdings, never inferred from local
    orders. ``equity`` marks held positions with a supplied price mapping so a
    snapshot can be valued at any decision session without holding prices
    itself.
    """

    account_snapshot_id: str
    as_of: datetime
    settled_cash: float
    unsettled_cash: float
    positions: tuple[Position, ...]
    open_order_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.account_snapshot_id:
            raise ValueError("account_snapshot_id must be non-empty")
        if not math.isfinite(self.settled_cash) or not math.isfinite(self.unsettled_cash):
            raise ValueError("cash balances must be finite")
        seen: set[str] = set()
        for position in self.positions:
            instrument_id = position.instrument.instrument_id
            if instrument_id in seen:
                raise ValueError(f"duplicate position for {instrument_id}")
            seen.add(instrument_id)

    def quantity_of(self, instrument_id: str) -> int:
        for position in self.positions:
            if position.instrument.instrument_id == instrument_id:
                return int(position.quantity)
        return 0

    def equity(self, cash_prices: Mapping[str, float]) -> float:
        """Mark the snapshot to ``cash_prices`` (instrument_id -> price)."""
        positions_value = 0.0
        for position in self.positions:
            instrument_id = position.instrument.instrument_id
            if instrument_id not in cash_prices:
                raise ValueError(f"no price to value held instrument {instrument_id!r}")
            price = cash_prices[instrument_id]
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"invalid mark price for {instrument_id!r}")
            positions_value += position.quantity * price
        return self.settled_cash + self.unsettled_cash + positions_value

    def validate_as_of(self, decision_time: datetime) -> None:
        if self.as_of > decision_time:
            raise ValueError(
                "account snapshot is newer than decision_time "
                f"({self.as_of.isoformat()} > {decision_time.isoformat()})"
            )
