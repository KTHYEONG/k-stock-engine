"""Broker commission, statutory sell tax, and square-root market impact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Broker, auction-slippage and market-impact parameters; all values come from run config.

    Attributes:
        commission_rate: Broker commission per side on notional.
        impact_k: Coefficient of the square-root impact term ``k·σ₆₀·√(notional/adtv20)``.
        dividend_withholding_rate: Tax withheld on cash dividends.
        auction_slippage_ticks: Adverse price offset per side at the open auction, in ticks of the base price
            (fractional allowed). Why explicit: the auction has no spread, but our order moves the clearing
            price; the size of that move cannot be measured from historical data, so it is a scenario parameter evaluated on a grid.
        extra_slippage: Additional adverse price fraction per side (stress scenarios only; 0 in base runs).
        cash_yield_tax_rate: Tax on the positive annual yield of the cash sweep (bond-ETF distribution tax).
    """

    commission_rate: Decimal
    impact_k: float
    dividend_withholding_rate: Decimal
    auction_slippage_ticks: float = 0.0
    extra_slippage: float = 0.0
    cash_yield_tax_rate: Decimal = Decimal("0.154")

    def __post_init__(self) -> None:
        rate = self.dividend_withholding_rate
        if not Decimal(0) <= rate < Decimal(1):
            raise ValueError(f"dividend_withholding_rate must be in [0, 1), got {rate!r}")
        for name in ("auction_slippage_ticks", "extra_slippage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
            out = float(value)
            if not math.isfinite(out) or out < 0.0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        tax = self.cash_yield_tax_rate
        if not Decimal(0) <= tax < Decimal(1):
            raise ValueError(f"cash_yield_tax_rate must be in [0, 1), got {tax!r}")


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


def dividend_withholding(*, gross_krw: int, config: CostConfig) -> int:
    """Tax withheld from one cash dividend payment, truncated to whole KRW.

    Args:
        gross_krw: Gross dividend owed for one position (shares * DPS), non-negative.
        config: Run cost configuration.

    Returns:
        Withheld KRW, ``0 <= withheld <= gross_krw``.

    Raises:
        ValueError: ``gross_krw`` is negative.
    """
    if gross_krw < 0:
        raise ValueError(f"gross_krw must be >= 0, got {gross_krw!r}")
    return int((Decimal(gross_krw) * config.dividend_withholding_rate).to_integral_value(rounding=ROUND_FLOOR))


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
