"""Exact cash-and-shares ledger for a long-only KRX cash account with a futures/inverse overlay."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from src.backtest.costs import CostConfig, dividend_withholding
from src.backtest.events import DividendEvent
from src.backtest.overlay import PRIMARY_LEG, ceil_amount_krw
from src.core.pit import PITDataError

if TYPE_CHECKING:
    from src.backtest.overlay import DerivativeConfig


class LedgerAccount(StrEnum):
    CASH = "cash"
    MARGIN = "margin"
    PAYABLE = "payable"


class JournalKind(StrEnum):
    DEPOSIT = "deposit"
    BUY = "buy"
    SELL = "sell"
    COMMISSION = "commission"
    SELL_TAX = "sell_tax"
    CASH_IN_LIEU = "cash_in_lieu"
    DIVIDEND = "dividend"
    DIVIDEND_TAX = "dividend_tax"
    EXIT_PROCEEDS = "exit_proceeds"
    CASH_YIELD = "cash_yield"
    CASH_YIELD_TAX = "cash_yield_tax"
    TAX_PAYMENT = "tax_payment"
    MARGIN_TRANSFER = "margin_transfer"
    VARIATION_MARGIN = "variation_margin"
    FUTURES_TRADE = "futures_trade"
    FUTURES_COMMISSION = "futures_commission"
    FUTURES_ROLL = "futures_roll"
    FUTURES_TAX = "futures_tax"
    INVERSE_BUY = "inverse_buy"
    INVERSE_SELL = "inverse_sell"
    INVERSE_COMMISSION = "inverse_commission"
    INVERSE_TAX = "inverse_tax"


@dataclass(frozen=True, slots=True)
class JournalEntry:
    session_idx: int
    kind: JournalKind
    instrument_idx: int | None
    cash_delta: int
    quantity_delta: int
    account: LedgerAccount = LedgerAccount.CASH
    leg: str | None = None


@dataclass(frozen=True, slots=True)
class NavRecord:
    session_idx: int
    cash: int
    dividend_receivable: int
    market_value: int
    nav: int
    external_flow: int
    margin: int = 0
    inverse_value: int = 0
    tax_payable: int = 0


def _checked_int(value: int, *, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    amount = int(value)
    if amount < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value!r}")
    return amount


def _checked_level(value: float, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PITDataError(f"{what} must be finite, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise PITDataError(f"{what} must be finite and positive, got {value!r}")
    return out


def _floored_mark(contracts: int, multiplier: int, level: float) -> int:
    """Signed net-short mark ``floor(-contracts · multiplier · level)`` in whole KRW, exactly."""
    return -ceil_amount_krw(contracts * multiplier, level)


def _checked_cost_rate(value: float, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be in [0, 1), got {value!r}")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out < 1.0:
        raise ValueError(f"{what} must be in [0, 1), got {value!r}")
    return out


class Ledger:
    """Exact KRW-and-shares ledger for a KRX cash account with a derivatives overlay.

    Sale proceeds are usable for same-session buys (KRX settles both legs
    T+2), so one cash balance suffices. Every KRW movement is journaled against the
    account it moved in, so each account equals the journal sum exactly: ``cash`` equals
    the initial balance plus the ``CASH`` sum, ``margin`` the ``MARGIN`` sum, and
    ``-tax_payable`` the ``PAYABLE`` sum. Tax that cannot be paid from cash (a fully
    invested book at year end) accrues on the ``PAYABLE`` account, which nets the full
    liability out of NAV until it is settled.

    The ``MARGIN`` account holds the signed index-futures position: margin transfers,
    daily variation, commission, quarterly roll and the annual capital-gains tax all
    land there, so a hedge pays its own costs without touching free cash. Free cash
    never goes negative; margin may go negative (a liability restored by maintenance)
    but no futures method ever refuses a trade for lack of funds.
    """

    __slots__ = (
        "_cash",
        "_contracts",
        "_external_flow",
        "_initial_cash",
        "_inverse_basis",
        "_inverse_units",
        "_journal",
        "_margin",
        "_positions",
        "_receivable",
        "_receivable_by_pay",
        "_secondary_contracts",
        "_tax_payable",
        "_ytd_cash_yield",
        "_ytd_futures_pnl",
    )

    def __init__(self, *, initial_cash: int) -> None:
        self._initial_cash = _checked_int(initial_cash, what="initial_cash", minimum=0)
        self._cash = self._initial_cash
        self._positions: dict[int, int] = {}
        self._receivable = 0
        self._receivable_by_pay: dict[int, int] = {}
        self._external_flow = 0
        self._journal: list[JournalEntry] = []
        self._ytd_cash_yield = 0
        self._tax_payable = 0
        self._margin = 0
        self._contracts = 0
        self._secondary_contracts: dict[str, int] = {}
        self._inverse_units = 0
        self._inverse_basis = 0
        self._ytd_futures_pnl = 0

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

    def settle_dividends(self, *, session_idx: int, config: CostConfig) -> None:
        due = self._receivable_by_pay.pop(session_idx, 0)
        if due == 0:
            return
        tax = dividend_withholding(gross_krw=due, config=config)
        self._receivable -= due
        self._cash += due - tax
        self._journal.append(JournalEntry(session_idx, JournalKind.DIVIDEND, None, due, 0))
        self._journal.append(JournalEntry(session_idx, JournalKind.DIVIDEND_TAX, None, -tax, 0))

    def close_exit(self, *, session_idx: int, instrument_idx: int, price: int) -> None:
        instrument_idx = _checked_int(instrument_idx, what="instrument_idx", minimum=0)
        price = _checked_int(price, what="price", minimum=0)
        quantity = self._positions.pop(instrument_idx, 0)
        proceeds = quantity * price
        self._cash += proceeds
        self._journal.append(
            JournalEntry(session_idx, JournalKind.EXIT_PROCEEDS, instrument_idx, proceeds, -quantity)
        )

    def accrue_cash_yield(self, *, session_idx: int, gross_return: float) -> int:
        """Credit (or debit) one session's cash-sweep return on the cash held since the previous close.

        Amount = ``floor(cash * gross_return)`` (floor toward -inf so a negative return never under-charges);
        zero when cash <= 0 or the amount rounds to 0 (no journal entry then). Adds the amount to the
        year-to-date sweep yield. Returns the amount.

        Raises: ValueError when ``gross_return`` is not finite or <= -1.
        """
        if isinstance(gross_return, bool) or not isinstance(gross_return, (int, float)):
            raise ValueError(f"gross_return must be a finite number > -1, got {gross_return!r}")
        rate = float(gross_return)
        if not math.isfinite(rate) or rate <= -1.0:
            raise ValueError(f"gross_return must be a finite number > -1, got {gross_return!r}")
        if self._cash <= 0:
            return 0
        amount = math.floor(self._cash * rate)
        if amount == 0:
            return 0
        self._cash += amount
        self._ytd_cash_yield += amount
        self._journal.append(JournalEntry(session_idx, JournalKind.CASH_YIELD, None, amount, 0))
        return amount

    def settle_cash_yield_tax(self, *, session_idx: int, config: CostConfig) -> int:
        """Assess ``floor(max(ytd_yield, 0) * cash_yield_tax_rate)``, reset the year-to-date yield, and pay it
        from free cash up to the available balance; any unpaid remainder becomes ``tax_payable``.

        Why yearly netting: the sweep models one held bond ETF whose gain is taxed on disposal; netting the
        year's daily accruals approximates that without tracking lots. Why a payable instead of an error: a
        fully invested book can hold less cash at year end than the year's accrued tax; the account must
        record the liability, never crash and never go negative. Returns the assessed tax (0 → no entry).
        """
        from decimal import ROUND_FLOOR

        ytd = self._ytd_cash_yield
        self._ytd_cash_yield = 0
        if ytd <= 0:
            return 0
        tax = int((Decimal(ytd) * config.cash_yield_tax_rate).to_integral_value(rounding=ROUND_FLOOR))
        if tax <= 0:
            return 0
        paid = min(tax, max(self._cash, 0))
        if paid:
            self._cash -= paid
            self._journal.append(JournalEntry(session_idx, JournalKind.CASH_YIELD_TAX, None, -paid, 0))
        unpaid = tax - paid
        if unpaid:
            self._tax_payable += unpaid
            self._journal.append(
                JournalEntry(session_idx, JournalKind.CASH_YIELD_TAX, None, -unpaid, 0, LedgerAccount.PAYABLE)
            )
        return tax

    def settle_tax_payable(self, *, session_idx: int) -> int:
        """Pay the outstanding ``tax_payable`` from free cash up to the available balance; returns the amount paid."""
        due = self._tax_payable
        if due <= 0:
            return 0
        paid = min(due, max(self._cash, 0))
        if paid <= 0:
            return 0
        self._tax_payable -= paid
        self._cash -= paid
        self._journal.append(JournalEntry(session_idx, JournalKind.TAX_PAYMENT, None, -paid, 0))
        self._journal.append(
            JournalEntry(session_idx, JournalKind.TAX_PAYMENT, None, paid, 0, LedgerAccount.PAYABLE)
        )
        return paid

    def transfer_margin(self, *, session_idx: int, amount: int) -> None:
        """Move ``amount`` from free cash to margin (positive) or back (negative); never makes cash negative.

        Why the only cash-raising call: a margin top-up that free cash cannot fund is a sizing bug the
        engine must avoid, so it surfaces here instead of silently borrowing.
        """
        if isinstance(amount, bool) or not isinstance(amount, numbers.Integral):
            raise ValueError(f"amount must be an integer, got {amount!r}")
        delta = int(amount)
        if delta == 0:
            return
        if self._cash - delta < 0:
            raise ValueError(f"margin transfer {delta} exceeds cash {self._cash}")
        self._cash -= delta
        self._margin += delta
        self._journal.append(
            JournalEntry(
                session_idx, JournalKind.MARGIN_TRANSFER, None, -delta, 0, LedgerAccount.CASH
            )
        )
        self._journal.append(
            JournalEntry(
                session_idx, JournalKind.MARGIN_TRANSFER, None, delta, 0, LedgerAccount.MARGIN
            )
        )

    def settle_variation(
        self,
        *,
        session_idx: int,
        prev_level: float,
        level: float,
        multiplier: int,
        leg: str = PRIMARY_LEG,
    ) -> int:
        """Daily settlement of the held signed position into margin: a long gains when the level rises.

        Delta = ``mark(level) - mark(prev_level)`` with ``mark(L) = floor(-contracts · multiplier · L)`` so the
        cumulative settled P&L equals the floored mark exactly. Adds delta to the year-to-date futures P&L.
        """
        mult = _checked_int(multiplier, what="multiplier", minimum=1)
        held = self.leg_contracts(leg)
        if held == 0:
            return 0
        prev = _checked_level(prev_level, what="prev_level")
        cur = _checked_level(level, what="level")
        delta = _floored_mark(held, mult, cur) - _floored_mark(held, mult, prev)
        if delta == 0:
            return 0
        self._margin += delta
        self._ytd_futures_pnl += delta
        journal_leg = None if leg == PRIMARY_LEG else leg
        self._journal.append(
            JournalEntry(
                session_idx,
                JournalKind.VARIATION_MARGIN,
                None,
                delta,
                0,
                LedgerAccount.MARGIN,
                leg=journal_leg,
            )
        )
        return delta

    def trade_futures(
        self,
        *,
        session_idx: int,
        contracts: int,
        level: float,
        multiplier: int,
        cost_rate: float,
        leg: str = PRIMARY_LEG,
    ) -> None:
        """Set the net short position to ``contracts`` (signed; negative = net long). Commission
        ``ceil(cost_rate · |Δ| · multiplier · level)`` is debited from MARGIN."""
        if isinstance(contracts, bool) or not isinstance(contracts, numbers.Integral):
            raise ValueError(f"contracts must be an integer, got {contracts!r}")
        target = int(contracts)
        mult = _checked_int(multiplier, what="multiplier", minimum=1)
        rate = _checked_cost_rate(cost_rate, what="cost_rate")
        mark_level = _checked_level(level, what="level")
        held = self.leg_contracts(leg)
        delta = target - held
        if delta == 0:
            return
        if leg == PRIMARY_LEG:
            self._contracts = target
        else:
            if target != 0:
                self._secondary_contracts[leg] = target
            else:
                self._secondary_contracts.pop(leg, None)
        journal_leg = None if leg == PRIMARY_LEG else leg
        self._journal.append(
            JournalEntry(
                session_idx, JournalKind.FUTURES_TRADE, None, 0, delta, leg=journal_leg
            )
        )
        commission = ceil_amount_krw(rate, abs(delta), mult, mark_level)
        if commission > 0:
            self._margin -= commission
            self._journal.append(
                JournalEntry(
                    session_idx,
                    JournalKind.FUTURES_COMMISSION,
                    None,
                    -commission,
                    0,
                    LedgerAccount.MARGIN,
                    leg=journal_leg,
                )
            )

    def roll_futures(
        self,
        *,
        session_idx: int,
        level: float,
        multiplier: int,
        cost_rate: float,
        leg: str = PRIMARY_LEG,
    ) -> None:
        """Debit ``ceil(2 · cost_rate · |contracts| · multiplier · level)`` from MARGIN (close + reopen)."""
        mult = _checked_int(multiplier, what="multiplier", minimum=1)
        rate = _checked_cost_rate(cost_rate, what="cost_rate")
        mark_level = _checked_level(level, what="level")
        held = self.leg_contracts(leg)
        if held == 0:
            return
        cost = ceil_amount_krw(2.0 * rate, abs(held), mult, mark_level)
        if cost <= 0:
            return
        self._margin -= cost
        journal_leg = None if leg == PRIMARY_LEG else leg
        self._journal.append(
            JournalEntry(
                session_idx,
                JournalKind.FUTURES_ROLL,
                None,
                -cost,
                0,
                LedgerAccount.MARGIN,
                leg=journal_leg,
            )
        )

    def buy_inverse(
        self, *, session_idx: int, units: int, price: int, cost_rate: float
    ) -> None:
        quantity = _checked_int(units, what="units", minimum=1)
        px = _checked_int(price, what="price", minimum=1)
        rate = _checked_cost_rate(cost_rate, what="cost_rate")
        commission = ceil_amount_krw(rate, quantity, px)
        total = quantity * px + commission
        if total > self._cash:
            raise ValueError(f"inverse buy costs {total} but cash is {self._cash}")
        self._cash -= total
        self._inverse_units += quantity
        self._inverse_basis += quantity * px
        self._journal.append(
            JournalEntry(session_idx, JournalKind.INVERSE_BUY, None, -quantity * px, quantity)
        )
        if commission > 0:
            self._journal.append(
                JournalEntry(session_idx, JournalKind.INVERSE_COMMISSION, None, -commission, 0)
            )

    def sell_inverse(
        self, *, session_idx: int, units: int, price: int, cost_rate: float, tax_rate: Decimal
    ) -> int:
        """Sell inverse units at average cost basis; tax ``floor(max(proceeds - basis_sold, 0) · tax_rate)``.

        Why per-sale with no offset: inverse-ETF gains are taxed as dividend income per disposal and losses are
        not netted in an ordinary account. Returns the tax charged.
        """
        quantity = _checked_int(units, what="units", minimum=1)
        px = _checked_int(price, what="price", minimum=1)
        rate = _checked_cost_rate(cost_rate, what="cost_rate")
        if isinstance(tax_rate, bool) or not isinstance(tax_rate, Decimal):
            raise ValueError(f"tax_rate must be a Decimal in [0, 1), got {tax_rate!r}")
        if not Decimal(0) <= tax_rate < Decimal(1):
            raise ValueError(f"tax_rate must be in [0, 1), got {tax_rate!r}")
        if quantity > self._inverse_units:
            raise ValueError(f"sell {quantity} exceeds inverse holding {self._inverse_units}")
        from decimal import ROUND_FLOOR

        proceeds = quantity * px
        commission = ceil_amount_krw(rate, proceeds)
        basis_sold = (self._inverse_basis * quantity) // self._inverse_units
        gain = proceeds - basis_sold
        tax = (
            int((Decimal(gain) * tax_rate).to_integral_value(rounding=ROUND_FLOOR))
            if gain > 0
            else 0
        )
        self._inverse_units -= quantity
        self._inverse_basis -= basis_sold
        self._cash += proceeds - commission - tax
        self._journal.append(
            JournalEntry(session_idx, JournalKind.INVERSE_SELL, None, proceeds, -quantity)
        )
        if commission > 0:
            self._journal.append(
                JournalEntry(session_idx, JournalKind.INVERSE_COMMISSION, None, -commission, 0)
            )
        if tax > 0:
            self._journal.append(
                JournalEntry(session_idx, JournalKind.INVERSE_TAX, None, -tax, 0)
            )
        return tax

    def settle_futures_tax(self, *, session_idx: int, config: DerivativeConfig) -> int:
        """Debit ``floor(max(ytd_pnl - deduction, 0) · futures_tax_rate)`` from MARGIN and reset ytd (no carry-forward).

        Why margin: the taxable gains were settled into the futures account, so the tax is its own liability; a
        resulting margin shortfall is restored by the next maintenance step like any other variation loss.
        """
        from decimal import ROUND_FLOOR

        ytd = self._ytd_futures_pnl
        self._ytd_futures_pnl = 0
        taxable = ytd - int(config.futures_annual_deduction_krw)
        if taxable <= 0:
            return 0
        tax = int(
            (Decimal(taxable) * config.futures_tax_rate).to_integral_value(rounding=ROUND_FLOOR)
        )
        if tax <= 0:
            return 0
        self._margin -= tax
        self._journal.append(
            JournalEntry(session_idx, JournalKind.FUTURES_TAX, None, -tax, 0, LedgerAccount.MARGIN)
        )
        return tax

    def leg_contracts(self, leg: str) -> int:
        """Net-short contracts held in the named leg (primary leg if ``PRIMARY_LEG``)."""
        if leg == PRIMARY_LEG:
            return self._contracts
        return self._secondary_contracts.get(leg, 0)

    @property
    def legs(self) -> tuple[tuple[str, int], ...]:
        """Secondary legs with a non-zero position, sorted by name."""
        return tuple(sorted((k, v) for k, v in self._secondary_contracts.items() if v != 0))

    @property
    def contracts(self) -> int:
        return self._contracts

    @property
    def margin(self) -> int:
        return self._margin

    @property
    def inverse_units(self) -> int:
        return self._inverse_units

    def positions(self) -> Mapping[int, int]:
        return dict(self._positions)

    @property
    def cash(self) -> int:
        """Current cash balance in whole KRW."""
        return self._cash

    @property
    def tax_payable(self) -> int:
        """Outstanding tax liability in whole KRW (0 when fully settled)."""
        return self._tax_payable

    def mark(
        self,
        *,
        session_idx: int,
        close: NDArray[Any],
        present: NDArray[Any],
        inverse_price: int | None = None,
    ) -> NavRecord:
        market_value = 0
        for instrument_idx, quantity in self._positions.items():
            if not present[instrument_idx]:
                raise PITDataError(f"cannot mark missing instrument {instrument_idx}")
            market_value += quantity * int(close[instrument_idx])
        if self._inverse_units > 0:
            if inverse_price is None:
                raise PITDataError("cannot mark inverse holding without a price")
            if isinstance(inverse_price, bool) or not isinstance(inverse_price, numbers.Integral):
                raise PITDataError(f"invalid inverse price: {inverse_price!r}")
            px = int(inverse_price)
            if px < 0:
                raise PITDataError(f"invalid inverse price: {inverse_price!r}")
            inverse_value = self._inverse_units * px
        else:
            inverse_value = 0
        return NavRecord(
            session_idx=session_idx,
            cash=self._cash,
            dividend_receivable=self._receivable,
            market_value=market_value,
            nav=(
                self._cash + self._receivable + market_value + self._margin + inverse_value
                - self._tax_payable
            ),
            external_flow=self._external_flow,
            margin=self._margin,
            inverse_value=inverse_value,
            tax_payable=self._tax_payable,
        )

    @property
    def journal(self) -> tuple[JournalEntry, ...]:
        return tuple(self._journal)

    @property
    def journal_size(self) -> int:
        """Number of entries booked so far; pair with :meth:`journal_since` to read one session's slice."""
        return len(self._journal)

    def journal_since(self, index: int) -> Sequence[JournalEntry]:
        """Entries booked at or after ``index`` (a ``journal_size`` captured before the session)."""
        return tuple(self._journal[index:])
