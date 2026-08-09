"""Fill-side cost contracts shared by stock and ETF simulations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fee/tax/slippage assumptions as typed inputs, not strategy constants.

    ``round_trip_cost`` models one entry and one exit fill: commission on both
    sides, tax once (sell-side only), and slippage as basis points of notional.
    """

    commission_rate: float = 0.0
    tax_rate: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        if self.tax_rate < 0:
            raise ValueError("tax_rate must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

    def round_trip_cost(self, notional: float) -> float:
        return notional * (self.commission_rate * 2 + self.tax_rate + self.slippage_bps / 10_000)
