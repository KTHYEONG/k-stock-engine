"""Exact cash-and-shares ledger for a long-only KRX cash account."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from numpy.typing import NDArray

from src.backtest.events import DividendEvent
from src.core.pit import PITDataError


class JournalKind(StrEnum):
    DEPOSIT = "deposit"
    BUY = "buy"
    SELL = "sell"
    COMMISSION = "commission"
    SELL_TAX = "sell_tax"
    CASH_IN_LIEU = "cash_in_lieu"
    DIVIDEND = "dividend"
    EXIT_PROCEEDS = "exit_proceeds"


@dataclass(frozen=True, slots=True)
class JournalEntry:
    session_idx: int
    kind: JournalKind
    instrument_idx: int | None
    cash_delta: int
    quantity_delta: int


@dataclass(frozen=True, slots=True)
class NavRecord:
    session_idx: int
    cash: int
    dividend_receivable: int
    market_value: int
    nav: int
    external_flow: int


def _checked_int(value: int, *, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    amount = int(value)
    if amount < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value!r}")
    return amount


class Ledger:
    """Exact cash-and-shares ledger for a long-only KRX cash account.

    Sale proceeds are usable for same-session buys (KRX settles both legs
    T+2), so one cash balance suffices; cash never goes negative because no
    margin is modeled. Every cash change is journaled, so cash equals the
    initial balance plus the journal sum exactly.
    """

    __slots__ = (
        "_cash",
        "_external_flow",
        "_initial_cash",
        "_journal",
        "_positions",
        "_receivable",
        "_receivable_by_pay",
    )

    def __init__(self, *, initial_cash: int) -> None:
        self._initial_cash = _checked_int(initial_cash, what="initial_cash", minimum=0)
        self._cash = self._initial_cash
        self._positions: dict[int, int] = {}
        self._receivable = 0
        self._receivable_by_pay: dict[int, int] = {}
        self._external_flow = 0
        self._journal: list[JournalEntry] = []

    def deposit(self, *, session_idx: int, amount: int) -> None:
        amount = _checked_int(amount, what="amount", minimum=1)
        self._cash += amount
        self._external_flow += amount
        self._journal.append(JournalEntry(session_idx, JournalKind.DEPOSIT, None, amount, 0))

    def buy(
        self, *, session_idx: int, instrument_idx: int, quantity: int, price: int, commission: int
    ) -> None:
        instrument_idx = _checked_int(instrument_idx, what="instrument_idx", minimum=0)
        quantity = _checked_int(quantity, what="quantity", minimum=1)
        price = _checked_int(price, what="price", minimum=1)
        commission = _checked_int(commission, what="commission", minimum=0)
        total = quantity * price + commission
        if total > self._cash:
            raise ValueError(f"buy costs {total} KRW but cash is {self._cash} KRW")
        self._cash -= total
        self._positions[instrument_idx] = self._positions.get(instrument_idx, 0) + quantity
        self._journal.append(
            JournalEntry(session_idx, JournalKind.BUY, instrument_idx, -quantity * price, quantity)
        )
        self._journal.append(
            JournalEntry(session_idx, JournalKind.COMMISSION, instrument_idx, -commission, 0)
        )

    def sell(
        self,
        *,
        session_idx: int,
        instrument_idx: int,
        quantity: int,
        price: int,
        commission: int,
        sell_tax: int,
    ) -> None:
        instrument_idx = _checked_int(instrument_idx, what="instrument_idx", minimum=0)
        quantity = _checked_int(quantity, what="quantity", minimum=1)
        price = _checked_int(price, what="price", minimum=1)
        commission = _checked_int(commission, what="commission", minimum=0)
        sell_tax = _checked_int(sell_tax, what="sell_tax", minimum=0)
        held = self._positions.get(instrument_idx, 0)
        if quantity > held:
            raise ValueError(f"sell {quantity} exceeds holding {held}")
        remaining = held - quantity
        if remaining:
            self._positions[instrument_idx] = remaining
        else:
            del self._positions[instrument_idx]
        self._cash += quantity * price - commission - sell_tax
        self._journal.append(
            JournalEntry(session_idx, JournalKind.SELL, instrument_idx, quantity * price, -quantity)
        )
        self._journal.append(
            JournalEntry(session_idx, JournalKind.COMMISSION, instrument_idx, -commission, 0)
        )
        self._journal.append(
            JournalEntry(session_idx, JournalKind.SELL_TAX, instrument_idx, -sell_tax, 0)
        )

    def apply_share_factor(
        self, *, session_idx: int, instrument_idx: int, factor: float, base_price: int
    ) -> None:
        instrument_idx = _checked_int(instrument_idx, what="instrument_idx", minimum=0)
        held = self._positions.get(instrument_idx, 0)
        if held == 0:
            return
        scaled = held * factor
        new_quantity = math.floor(scaled)
        cash_in_lieu = math.floor((scaled - new_quantity) * base_price)
        if new_quantity:
            self._positions[instrument_idx] = new_quantity
        else:
            del self._positions[instrument_idx]
        self._cash += cash_in_lieu
        self._journal.append(
            JournalEntry(
                session_idx,
                JournalKind.CASH_IN_LIEU,
                instrument_idx,
                cash_in_lieu,
                new_quantity - held,
            )
        )

    def record_dividend_entitlements(
        self, *, session_idx: int, events: Sequence[DividendEvent]
    ) -> None:
        for event in events:
            amount = self._positions.get(event.instrument_idx, 0) * event.dps_krw
            self._receivable += amount
            pay_idx = event.pay_session_idx
            self._receivable_by_pay[pay_idx] = self._receivable_by_pay.get(pay_idx, 0) + amount

    def settle_dividends(self, *, session_idx: int) -> None:
        due = self._receivable_by_pay.pop(session_idx, 0)
        if due == 0:
            return
        self._receivable -= due
        self._cash += due
        self._journal.append(JournalEntry(session_idx, JournalKind.DIVIDEND, None, due, 0))

    def close_exit(self, *, session_idx: int, instrument_idx: int, price: int) -> None:
        instrument_idx = _checked_int(instrument_idx, what="instrument_idx", minimum=0)
        price = _checked_int(price, what="price", minimum=0)
        quantity = self._positions.pop(instrument_idx, 0)
        proceeds = quantity * price
        self._cash += proceeds
        self._journal.append(
            JournalEntry(session_idx, JournalKind.EXIT_PROCEEDS, instrument_idx, proceeds, -quantity)
        )

    def positions(self) -> Mapping[int, int]:
        return dict(self._positions)

    @property
    def cash(self) -> int:
        """Current cash balance in whole KRW."""
        return self._cash

    def mark(self, *, session_idx: int, close: NDArray[Any], present: NDArray[Any]) -> NavRecord:
        market_value = 0
        for instrument_idx, quantity in self._positions.items():
            if not present[instrument_idx]:
                raise PITDataError(f"cannot mark missing instrument {instrument_idx}")
            market_value += quantity * int(close[instrument_idx])
        return NavRecord(
            session_idx=session_idx,
            cash=self._cash,
            dividend_receivable=self._receivable,
            market_value=market_value,
            nav=self._cash + self._receivable + market_value,
            external_flow=self._external_flow,
        )

    @property
    def journal(self) -> tuple[JournalEntry, ...]:
        return tuple(self._journal)
