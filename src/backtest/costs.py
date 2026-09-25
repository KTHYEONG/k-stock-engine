"""Broker commission, statutory sell tax, and square-root market impact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Broker and market-impact parameters; all values come from run config.

    Attributes:
        commission_rate: Broker commission per side on notional.
        impact_k: Coefficient of the square-root impact term.
    """

    commission_rate: Decimal
    impact_k: float


@dataclass(frozen=True, slots=True)
class FillCost:
    commission: int
    sell_tax: int


def fill_cost(
    *, side: Side, quantity: int, price: int, sell_tax_rate: Decimal, config: CostConfig
) -> FillCost:
    """Commission and statutory sell tax for one fill, truncated to whole KRW (Korean practice truncates sub-won amounts)."""
    if side is not Side.BUY and side is not Side.SELL:
        raise ValueError(f"side must be Side.BUY or Side.SELL, got {side!r}")
    notional = Decimal(quantity * price)
    commission = int((notional * config.commission_rate).to_integral_value(rounding=ROUND_FLOOR))
    tax = (
        int((notional * sell_tax_rate).to_integral_value(rounding=ROUND_FLOOR))
        if side is Side.SELL
        else 0
    )
    return FillCost(commission=commission, sell_tax=tax)


def impact_fraction(*, notional: float, adtv20: float, vol60: float, config: CostConfig) -> float:
    """Adverse price fraction ``k · σ₆₀ · sqrt(notional / adtv20)``; inputs must be finite and adtv20 > 0."""
    if (
        not math.isfinite(notional)
        or not math.isfinite(adtv20)
        or not math.isfinite(vol60)
        or adtv20 <= 0.0
        or notional < 0.0
        or vol60 < 0.0
    ):
        raise ValueError(
            "impact inputs must be finite with adtv20 > 0, "
            f"got notional={notional!r}, adtv20={adtv20!r}, vol60={vol60!r}"
        )
    return config.impact_k * vol60 * math.sqrt(notional / adtv20)
