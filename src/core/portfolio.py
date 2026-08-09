"""Portfolio and cost primitives shared by stock and ETF subsystems."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fee/tax/slippage assumptions as typed inputs, not strategy constants."""

    commission_rate: float = 0.0
    tax_rate: float = 0.0
    slippage_bps: float = 0.0

    def round_trip_cost(self, notional: float) -> float:
        return notional * (self.commission_rate * 2 + self.tax_rate + self.slippage_bps / 10_000)


@dataclass(frozen=True, slots=True)
class Position:
    """A current holding of one instrument."""

    instrument: Instrument
    quantity: float
    average_cost: float


@dataclass(frozen=True, slots=True)
class Allocation:
    """A target allocation to one instrument."""

    instrument: Instrument
    target_value: float
    reason: str = ""
