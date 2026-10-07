"""KQ150 futures + inverse-ETF overlay terms and causal policy view."""

from __future__ import annotations

import bisect
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from src.core.pit import PITDataError


def _reject_bool(value: object) -> bool:
    return isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class DerivativeConfig:
    """Contract, margin, cost and tax terms of the index-futures + inverse-ETF overlay.

    Attributes:
        contract_multiplier_krw: KRW per index point per contract (KQ150: 10,000).
        initial_margin_rate: Initial margin as a fraction of notional.
        margin_buffer_rate: Extra reserve held above initial margin at every overlay rebalance.
        margin_topup_trigger_fraction: Top up when margin < trigger · initial_margin_rate · notional.
        futures_cost_rate: Commission + slippage per side on futures notional (also charged twice per roll).
        inverse_cost_rate: Commission + slippage per side on inverse-ETF notional (no sell tax for ETFs).
        futures_tax_rate: Derivative capital-gains tax on the annual net futures P&L above the deduction.
        futures_annual_deduction_krw: Annual deduction; losses are not carried forward.
        inverse_tax_rate: Tax on each inverse-ETF sale's realized gain; losses are not offset.
    """

    contract_multiplier_krw: int
    initial_margin_rate: float
    margin_buffer_rate: float
    margin_topup_trigger_fraction: float
    futures_cost_rate: float
    inverse_cost_rate: float
    futures_tax_rate: Decimal
    futures_annual_deduction_krw: int
    inverse_tax_rate: Decimal

    def __post_init__(self) -> None:
        mult = self.contract_multiplier_krw
        if _reject_bool(mult) or not isinstance(mult, numbers.Integral) or int(mult) <= 0:
            raise ValueError(f"contract_multiplier_krw must be a positive integer, got {mult!r}")
        for name in ("initial_margin_rate", "margin_buffer_rate"):
            value = getattr(self, name)
            if (
                _reject_bool(value)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must be in (0, 1), got {value!r}")
        trigger = self.margin_topup_trigger_fraction
        if (
            _reject_bool(trigger)
            or not isinstance(trigger, (int, float))
            or not math.isfinite(float(trigger))
            or not 0.0 < float(trigger) <= 1.0
        ):
            raise ValueError(
                f"margin_topup_trigger_fraction must be in (0, 1], got {trigger!r}"
            )
        for name in ("futures_cost_rate", "inverse_cost_rate"):
            value = getattr(self, name)
            if (
                _reject_bool(value)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        for name in ("futures_tax_rate", "inverse_tax_rate"):
            value = getattr(self, name)
            if _reject_bool(value) or not isinstance(value, Decimal):
                raise ValueError(f"{name} must be a Decimal in [0, 1), got {value!r}")
            if not Decimal(0) <= value < Decimal(1):
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        deduction = self.futures_annual_deduction_krw
        if (
            _reject_bool(deduction)
            or not isinstance(deduction, numbers.Integral)
            or int(deduction) < 0
        ):
            raise ValueError(
                f"futures_annual_deduction_krw must be >= 0, got {deduction!r}"
            )


PRIMARY_LEG: Final = "primary"


def check_leg_name(name: object) -> None:
    if not isinstance(name, str) or not name or name == PRIMARY_LEG:
        raise ValueError(f"invalid leg name: {name!r}")


def _check_leg_pairs(pairs: object, *, what: str) -> None:
    """Secondary-leg ``(name, contracts)`` pairs: valid unique names in strictly ascending order, integer counts."""
    if not isinstance(pairs, tuple):
        raise ValueError(f"{what} must be a tuple of (name, contracts) pairs, got {pairs!r}")
    prev_name: str | None = None
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"leg pair must be a 2-tuple (name, contracts), got {item!r}")
        name, count = item
        check_leg_name(name)
        if name == prev_name:
            raise ValueError(f"duplicate leg name in {what}: {name!r}")
        if prev_name is not None and name < prev_name:
            raise ValueError(f"{what} must be sorted by name: {pairs!r}")
        prev_name = name
        if _reject_bool(count) or not isinstance(count, numbers.Integral):
            raise ValueError(f"leg {name!r} contracts must be an integer, got {count!r}")


@dataclass(frozen=True, slots=True)
class OverlayMarket:
    """Closes aligned to the engine sessions: futures underlying level, inverse-ETF close (NaN before listing),
    and the underlying level of each secondary futures leg by name."""

    index_level: NDArray[np.float64]
    inverse_close: NDArray[np.float64]
    legs: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.legs, Mapping):
            raise ValueError(f"legs must be a Mapping, got {self.legs!r}")
        for name, arr in self.legs.items():
            check_leg_name(name)
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"leg {name!r} must be an ndarray, got {arr!r}")


@dataclass(frozen=True, slots=True)
class OverlayTarget:
    """Net short futures contracts (signed: negative = net long), inverse-ETF value in KRW (>= 0), and signed
    net-short contracts of each secondary leg as ``(name, contracts)`` pairs sorted by name."""

    contracts: int
    inverse_value_krw: int
    legs: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        contracts = self.contracts
        if _reject_bool(contracts) or not isinstance(contracts, numbers.Integral):
            raise ValueError(f"contracts must be an integer, got {contracts!r}")
        inverse = self.inverse_value_krw
        if _reject_bool(inverse) or not isinstance(inverse, numbers.Integral):
            raise ValueError(f"inverse_value_krw must be an integer >= 0, got {inverse!r}")
        if int(inverse) < 0:
            raise ValueError(f"inverse_value_krw must be >= 0, got {inverse!r}")
        _check_leg_pairs(self.legs, what="legs")


@dataclass(frozen=True, slots=True)
class OverlayState:
    """Causal view handed to the overlay policy after session ``session_idx``'s close.

    ``stock_book_returns`` / ``index_returns`` cover the run's sessions up to and including
    ``session_idx`` (decisions are made after the 18:00 data release, so the session's own close is known).
    ``stock_book_nav`` = free cash + dividend receivable + stock market value - tax_payable (margin and inverse
    excluded) — the same SB used for ``stock_book_returns``.
    ``contracts`` is the signed net short count (negative = net long).
    """

    session_idx: int
    nav: int
    stock_book_nav: int
    stock_book_returns: NDArray[np.float64]
    index_returns: NDArray[np.float64]
    index_level: float
    contracts: int
    inverse_units: int
    leg_contracts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _check_leg_pairs(self.leg_contracts, what="leg_contracts")


class OverlayPolicy(Protocol):
    def target(self, state: OverlayState) -> OverlayTarget | None:
        """New overlay target, or None to keep the current position (non-rebalance session)."""
        ...


def ceil_amount_krw(rate: float | Decimal, *factors: int | float) -> int:
    """``ceil(rate · factors...)`` in whole KRW, evaluated in exact decimal arithmetic.

    Why not float: binary rounding turns ``0.15 · 10,000 · 1,050`` into ``1,575,000.0000000002``, so the ceiling
    overcharges by 1 KRW and the ledger stops conserving whole KRW. Every factor is a decimal the caller wrote
    down (a rate, a multiplier, a quoted level), so the product is exact and only the ceiling rounds.
    """
    product = rate if isinstance(rate, Decimal) else Decimal(str(rate))
    for factor in factors:
        product *= factor if isinstance(factor, Decimal) else Decimal(str(factor))
    return int(product.to_integral_value(rounding=ROUND_CEILING))


def required_reserve_krw(*, contracts: int, level: float, config: DerivativeConfig) -> int:
    """``ceil((initial_margin_rate + margin_buffer_rate) · |contracts| · multiplier · level)``; margin is charged on
    gross exposure whichever the side."""
    if _reject_bool(contracts) or not isinstance(contracts, numbers.Integral):
        raise ValueError(f"contracts must be an integer, got {contracts!r}")
    count = int(contracts)
    if count == 0:
        return 0
    count = abs(count)
    if _reject_bool(level) or not isinstance(level, (int, float)):
        raise PITDataError(f"index level must be finite, got {level!r}")
    level_f = float(level)
    if not math.isfinite(level_f) or level_f <= 0.0:
        raise PITDataError(f"index level must be finite and positive, got {level!r}")
    rate = Decimal(str(float(config.initial_margin_rate))) + Decimal(
        str(float(config.margin_buffer_rate))
    )
    return ceil_amount_krw(rate, count, int(config.contract_multiplier_krw), level_f)


def _second_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (3 - first.weekday()) % 7
    return date(year, month, 1 + offset + 7)


def futures_expiry_rows(sessions: Sequence[date]) -> frozenset[int]:
    """Row of the last session on or before the second Thursday of March, June, September and December.

    Why: KRX index futures expire on the second Thursday of the quarter month (the prior session when it is
    a holiday); a held hedge is rolled there and pays the roll cost.
    """
    ordered = list(sessions)
    if not ordered:
        return frozenset()
    rows: set[int] = set()
    years = range(ordered[0].year, ordered[-1].year + 1)
    for year in years:
        for month in (3, 6, 9, 12):
            expiry = _second_thursday(year, month)
            if expiry < ordered[0] or expiry > ordered[-1]:
                continue
            idx = bisect.bisect_right(ordered, expiry) - 1
            if idx >= 0:
                rows.add(idx)
    return frozenset(rows)
