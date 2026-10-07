"""Integer ledger: journal equality, splits, dividends, exits, and marks."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

from src.backtest.costs import CostConfig
from src.backtest.events import DividendEvent
from src.backtest.ledger import JournalKind, Ledger, LedgerAccount
from src.backtest.overlay import DerivativeConfig
from src.core.pit import PITDataError


def _costs(withholding: str = "0") -> CostConfig:
    return CostConfig(
        commission_rate=Decimal("0.00015"),
        impact_k=1.0,
        dividend_withholding_rate=Decimal(withholding),
    )


def _entitlement(instrument_idx: int, ex: int, pay: int, dps: int) -> DividendEvent:
    return DividendEvent(
        instrument_idx=instrument_idx, ex_session_idx=ex, pay_session_idx=pay, dps_krw=dps
    )


def _cash(ledger: Ledger, initial_cash: int) -> int:
    return initial_cash + sum(
        entry.cash_delta for entry in ledger.journal if entry.account is LedgerAccount.CASH
    )


def _quantity_sums(ledger: Ledger) -> dict[int, int]:
    totals: dict[int, int] = {}
    for entry in ledger.journal:
        if entry.instrument_idx is not None:
            totals[entry.instrument_idx] = totals.get(entry.instrument_idx, 0) + entry.quantity_delta
    return totals


def test_cash_equals_journal_sum_after_replay() -> None:
    initial = 10_000_000
    ledger = Ledger(initial_cash=initial)
    ledger.deposit(session_idx=0, amount=1_000_000)
    ledger.buy(session_idx=1, instrument_idx=0, quantity=100, price=10_000, commission=1_500)
    ledger.buy(session_idx=1, instrument_idx=1, quantity=50, price=20_000, commission=1_500)
    ledger.sell(
        session_idx=2, instrument_idx=0, quantity=40, price=11_000, commission=700, sell_tax=1_320
    )
    ledger.apply_share_factor(session_idx=3, instrument_idx=1, factor=2.0, base_price=21_000)
    ledger.record_dividend_entitlements(
        session_idx=4,
        events=(_entitlement(0, 4, 6, 100), _entitlement(1, 4, 6, 50)),
    )
    ledger.sell(
        session_idx=5, instrument_idx=0, quantity=60, price=9_000, commission=500, sell_tax=1_620
    )
    ledger.settle_dividends(session_idx=6, config=_costs())
    ledger.close_exit(session_idx=7, instrument_idx=1, price=15_000)

    assert _cash(ledger, initial) == 11_483_860
    assert ledger.positions() == {}
    assert _quantity_sums(ledger) == {0: 0, 1: 0}
    kinds = [entry.kind for entry in ledger.journal]
    assert kinds.count(JournalKind.DEPOSIT) == 1
    assert kinds.count(JournalKind.DIVIDEND) == 1
    assert kinds.count(JournalKind.EXIT_PROCEEDS) == 1
    record = ledger.mark(
        session_idx=7,
        close=np.zeros(2, dtype=np.int64),
        present=np.ones(2, dtype=bool),
    )
    assert record.nav == record.cash == 11_483_860
    assert record.external_flow == 1_000_000


def test_overspend_leaves_state_unchanged() -> None:
    ledger = Ledger(initial_cash=1_000)
    journal_before = ledger.journal
    with pytest.raises(ValueError, match="buy costs"):
        ledger.buy(session_idx=0, instrument_idx=0, quantity=1, price=1_000, commission=1)
    assert ledger.journal == journal_before
    assert ledger.positions() == {}
    assert _cash(ledger, 1_000) == 1_000


def test_split_pays_cash_in_lieu() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=3, price=10_000, commission=0)
    ledger.apply_share_factor(session_idx=1, instrument_idx=0, factor=1.5, base_price=10_000)
    assert ledger.positions() == {0: 4}
    assert _cash(ledger, 1_000_000) == 975_000
    lieu = [entry for entry in ledger.journal if entry.kind is JournalKind.CASH_IN_LIEU]
    assert [(entry.cash_delta, entry.quantity_delta) for entry in lieu] == [(5_000, 1)]


def test_reverse_split_below_one_share() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=3, price=100_000, commission=0)
    ledger.apply_share_factor(session_idx=1, instrument_idx=0, factor=0.2, base_price=50_000)
    assert ledger.positions() == {}
    assert _cash(ledger, 1_000_000) == 730_000


def test_dividend_accrues_then_pays() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=10, price=10_000, commission=0)
    ledger.record_dividend_entitlements(session_idx=3, events=(_entitlement(0, 3, 20, 500),))
    ledger.sell(
        session_idx=4, instrument_idx=0, quantity=10, price=10_500, commission=0, sell_tax=0
    )
    before = ledger.mark(
        session_idx=5, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool)
    )
    assert before.dividend_receivable == 5_000
    assert before.nav == 1_010_000
    ledger.settle_dividends(session_idx=20, config=_costs())
    after = ledger.mark(
        session_idx=20, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool)
    )
    assert after.dividend_receivable == 0
    assert after.cash == 1_010_000
    assert after.nav == before.nav


def test_missing_mark_fails_closed() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=1, price=10_000, commission=0)
    with pytest.raises(PITDataError):
        ledger.mark(
            session_idx=1,
            close=np.asarray([10_000], dtype=np.int64),
            present=np.asarray([False]),
        )


def test_non_integer_and_range_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        Ledger(initial_cash=-1)
    with pytest.raises(ValueError, match="initial_cash"):
        Ledger(initial_cash=1.5)
    ledger = Ledger(initial_cash=100)
    with pytest.raises(ValueError, match="amount"):
        ledger.deposit(session_idx=0, amount=0)
    with pytest.raises(ValueError, match="quantity"):
        ledger.buy(session_idx=0, instrument_idx=0, quantity=True, price=10, commission=0)


def test_sell_above_holding_rejected() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=5, price=10_000, commission=0)
    with pytest.raises(ValueError, match="exceeds holding"):
        ledger.sell(
            session_idx=1, instrument_idx=0, quantity=6, price=10_000, commission=0, sell_tax=0
        )


def test_share_factor_without_position_is_noop() -> None:
    ledger = Ledger(initial_cash=100)
    ledger.apply_share_factor(session_idx=0, instrument_idx=3, factor=50.0, base_price=100)
    assert ledger.positions() == {}
    assert ledger.journal == ()


def test_settle_without_due_is_noop() -> None:
    ledger = Ledger(initial_cash=100)
    ledger.settle_dividends(session_idx=9, config=_costs())
    assert ledger.journal == ()


def test_settle_dividends_pays_net_and_records_tax() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=100, price=10_000, commission=0)
    ledger.record_dividend_entitlements(session_idx=3, events=(_entitlement(0, 3, 20, 1_400),))
    ledger.settle_dividends(session_idx=20, config=_costs("0.154"))

    assert ledger.cash == 118_440
    kinds = [entry.kind for entry in ledger.journal]
    assert kinds.count(JournalKind.DIVIDEND) == 1
    assert kinds.count(JournalKind.DIVIDEND_TAX) == 1
    gross = next(entry.cash_delta for entry in ledger.journal if entry.kind is JournalKind.DIVIDEND)
    tax = next(entry.cash_delta for entry in ledger.journal if entry.kind is JournalKind.DIVIDEND_TAX)
    assert (gross, tax) == (140_000, -21_560)
    assert gross == 118_440 - tax
    assert _cash(ledger, 1_000_000) == ledger.cash
    assert ledger.mark(
        session_idx=20, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool)
    ).dividend_receivable == 0


def test_settle_dividends_zero_rate_pays_gross() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=100, price=10_000, commission=0)
    ledger.record_dividend_entitlements(session_idx=3, events=(_entitlement(0, 3, 20, 1_400),))
    ledger.settle_dividends(session_idx=20, config=_costs("0"))

    assert ledger.cash == 140_000
    tax = next(entry.cash_delta for entry in ledger.journal if entry.kind is JournalKind.DIVIDEND_TAX)
    assert tax == 0


def test_settle_dividends_conserves_gross_across_payments() -> None:
    ledger = Ledger(initial_cash=10_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=100, price=10_000, commission=0)
    ledger.buy(session_idx=0, instrument_idx=1, quantity=50, price=20_000, commission=0)
    ledger.record_dividend_entitlements(
        session_idx=1,
        events=(_entitlement(0, 1, 5, 1_400), _entitlement(1, 1, 9, 333)),
    )
    ledger.settle_dividends(session_idx=5, config=_costs("0.154"))
    ledger.settle_dividends(session_idx=9, config=_costs("0.154"))

    gross = sum(entry.cash_delta for entry in ledger.journal if entry.kind is JournalKind.DIVIDEND)
    withheld = -sum(entry.cash_delta for entry in ledger.journal if entry.kind is JournalKind.DIVIDEND_TAX)
    assert gross == 100 * 1_400 + 50 * 333
    assert _cash(ledger, 10_000_000) == 10_000_000 - 2_000_000 + gross - withheld


def test_close_exit_at_zero_records_total_loss() -> None:
    ledger = Ledger(initial_cash=100_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=10, price=5_000, commission=0)
    ledger.close_exit(session_idx=1, instrument_idx=0, price=0)
    assert ledger.positions() == {}
    assert _cash(ledger, 100_000) == 50_000


def _yield_costs(rate: str = "0.154") -> CostConfig:
    return CostConfig(
        commission_rate=Decimal("0.00015"), impact_k=0.0, dividend_withholding_rate=Decimal("0"),
        cash_yield_tax_rate=Decimal(rate),
    )


def test_cash_yield_credited_on_overnight_cash() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    assert ledger.accrue_cash_yield(session_idx=1, gross_return=0.0001) == 100
    assert ledger.cash == 1_000_100
    kinds = [entry.kind for entry in ledger.journal]
    assert kinds == [JournalKind.CASH_YIELD]
    assert _cash(ledger, 1_000_000) == ledger.cash


def test_cash_yield_zero_cash_or_dust_is_noop() -> None:
    empty = Ledger(initial_cash=0)
    assert empty.accrue_cash_yield(session_idx=0, gross_return=0.01) == 0
    assert empty.journal == ()
    dust = Ledger(initial_cash=100)
    assert dust.accrue_cash_yield(session_idx=0, gross_return=0.000001) == 0
    assert dust.journal == ()
    assert dust.cash == 100


def test_negative_yield_floors_toward_minus_inf() -> None:
    ledger = Ledger(initial_cash=1_000_001)
    assert ledger.accrue_cash_yield(session_idx=1, gross_return=-0.0001) == -101
    assert ledger.cash == 1_000_001 - 101
    assert _cash(ledger, 1_000_001) == ledger.cash


def test_cash_yield_invalid_return_rejected() -> None:
    ledger = Ledger(initial_cash=1_000)
    for bad in (float("nan"), float("inf"), -1.0, -2.0, True, "0.01"):
        with pytest.raises(ValueError, match="gross_return"):
            ledger.accrue_cash_yield(session_idx=0, gross_return=bad)  # type: ignore[arg-type]


def test_year_end_tax_nets_the_year() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.accrue_cash_yield(session_idx=0, gross_return=0.0005)
    ledger.accrue_cash_yield(session_idx=1, gross_return=-0.0001)
    assert ledger.settle_cash_yield_tax(session_idx=2, config=_yield_costs()) == 61
    assert ledger.cash == 1_000_000 + 500 - 101 - 61
    tax_entries = [e for e in ledger.journal if e.kind is JournalKind.CASH_YIELD_TAX]
    assert [(e.cash_delta, e.session_idx) for e in tax_entries] == [(-61, 2)]
    assert ledger.settle_cash_yield_tax(session_idx=3, config=_yield_costs()) == 0
    assert _cash(ledger, 1_000_000) == ledger.cash


def test_negative_year_pays_no_tax() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.accrue_cash_yield(session_idx=0, gross_return=-0.0003)
    assert ledger.settle_cash_yield_tax(session_idx=1, config=_yield_costs()) == 0
    assert not [e for e in ledger.journal if e.kind is JournalKind.CASH_YIELD_TAX]


def test_dust_yield_tax_rounds_to_zero() -> None:
    ledger = Ledger(initial_cash=10_000)
    assert ledger.accrue_cash_yield(session_idx=0, gross_return=0.0001) == 1
    assert ledger.settle_cash_yield_tax(session_idx=1, config=_yield_costs()) == 0
    assert not [e for e in ledger.journal if e.kind is JournalKind.CASH_YIELD_TAX]
    assert _cash(ledger, 10_000) == ledger.cash


def _payable_sum(ledger: Ledger) -> int:
    return sum(e.cash_delta for e in ledger.journal if e.account is LedgerAccount.PAYABLE)


def _invested_book(*, cash_left: int) -> Ledger:
    """A fully invested book: 100 accruals bank a ytd sweep yield of 10,000, then buys leave ``cash_left``."""
    ledger = Ledger(initial_cash=10_000_000)
    for session_idx in range(100):
        ledger.accrue_cash_yield(session_idx=session_idx, gross_return=0.00001)
    ledger.buy(session_idx=100, instrument_idx=0, quantity=10_010_000 - cash_left, price=1, commission=0)
    assert ledger.cash == cash_left
    return ledger


def test_tax_beyond_cash_becomes_a_payable() -> None:
    """A fully invested book owes more than it holds: the account records the liability, cash stays >= 0."""
    ledger = _invested_book(cash_left=99)

    before = ledger.mark(session_idx=101, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool))
    assert ledger.settle_cash_yield_tax(session_idx=101, config=_yield_costs()) == 1_540
    after = ledger.mark(session_idx=101, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool))

    assert ledger.cash == 0
    assert ledger.tax_payable == 1_441
    assert after.nav == before.nav - 1_540
    entries = [e for e in ledger.journal if e.kind is JournalKind.CASH_YIELD_TAX]
    assert [(e.account, e.cash_delta) for e in entries] == [
        (LedgerAccount.CASH, -99),
        (LedgerAccount.PAYABLE, -1_441),
    ]
    assert _cash(ledger, 10_000_000) == ledger.cash
    assert _payable_sum(ledger) == -ledger.tax_payable


def test_next_session_sales_fund_the_tax_payable() -> None:
    ledger = _invested_book(cash_left=99)
    ledger.settle_cash_yield_tax(session_idx=101, config=_yield_costs())

    ledger.sell(
        session_idx=102, instrument_idx=0, quantity=1, price=2_000_000, commission=0, sell_tax=0
    )
    assert ledger.settle_tax_payable(session_idx=102) == 1_441
    assert ledger.tax_payable == 0
    assert ledger.cash == 2_000_000 - 1_441
    payments = [e for e in ledger.journal if e.kind is JournalKind.TAX_PAYMENT]
    assert [(e.account, e.cash_delta) for e in payments] == [
        (LedgerAccount.CASH, -1_441),
        (LedgerAccount.PAYABLE, 1_441),
    ]
    assert _payable_sum(ledger) == -ledger.tax_payable
    assert _cash(ledger, 10_000_000) == ledger.cash


def test_tax_payment_capped_by_available_cash() -> None:
    ledger = _invested_book(cash_left=499)
    ledger.settle_cash_yield_tax(session_idx=101, config=_yield_costs())
    assert (ledger.cash, ledger.tax_payable) == (0, 1_041)

    ledger.deposit(session_idx=102, amount=600)
    assert ledger.settle_tax_payable(session_idx=102) == 600
    assert (ledger.cash, ledger.tax_payable) == (0, 441)
    assert ledger.settle_tax_payable(session_idx=103) == 0
    ledger.deposit(session_idx=104, amount=500)
    assert ledger.settle_tax_payable(session_idx=104) == 441
    assert (ledger.cash, ledger.tax_payable) == (59, 0)
    assert _payable_sum(ledger) == 0
    assert _cash(ledger, 10_000_000) == ledger.cash


def test_settle_tax_payable_without_liability_is_noop() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    assert ledger.settle_tax_payable(session_idx=0) == 0
    assert ledger.journal == ()
    assert ledger.tax_payable == 0


def _derivatives(**over: object) -> DerivativeConfig:
    base: dict[str, object] = {
        "contract_multiplier_krw": 10_000,
        "initial_margin_rate": 0.1,
        "margin_buffer_rate": 0.05,
        "margin_topup_trigger_fraction": 1.0,
        "futures_cost_rate": 0.0,
        "inverse_cost_rate": 0.0,
        "futures_tax_rate": Decimal("0.11"),
        "futures_annual_deduction_krw": 2_500_000,
        "inverse_tax_rate": Decimal("0.154"),
    }
    base.update(over)
    return DerivativeConfig(**base)  # type: ignore[arg-type]


def test_variation_settles_floored_marks_exactly() -> None:
    """Two short contracts settle to the floored mark: the sum of daily deltas equals the mark difference."""
    ledger = Ledger(initial_cash=10_000_000)
    ledger.transfer_margin(session_idx=0, amount=1_000_000)
    ledger.trade_futures(session_idx=0, contracts=2, level=1000.00, multiplier=10, cost_rate=0.0)

    first = ledger.settle_variation(
        session_idx=1, prev_level=1000.00, level=1001.37, multiplier=10
    )
    second = ledger.settle_variation(
        session_idx=2, prev_level=1001.37, level=999.99, multiplier=10
    )

    mark_start = math.floor(-2 * 10 * 1000.00)
    mark_up = math.floor(-2 * 10 * 1001.37)
    mark_end = math.floor(-2 * 10 * 999.99)
    assert (first, second) == (mark_up - mark_start, mark_end - mark_up)
    assert first + second == mark_end - mark_start
    assert ledger.margin == 1_000_000 + first + second
    assert all(
        entry.account is LedgerAccount.MARGIN
        for entry in ledger.journal
        if entry.kind is JournalKind.VARIATION_MARGIN
    )
    assert _cash(ledger, 10_000_000) == ledger.cash


def test_short_gains_when_the_index_falls() -> None:
    """A 10-point fall on one 10,000-KRW contract settles exactly +100,000 and feeds the year-to-date P&L."""
    ledger = Ledger(initial_cash=10_000_000)
    ledger.transfer_margin(session_idx=0, amount=1_000_000)
    ledger.trade_futures(session_idx=0, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)

    assert ledger.settle_variation(
        session_idx=1, prev_level=1000.0, level=990.0, multiplier=10_000
    ) == 100_000
    assert ledger.margin == 1_100_000

    config = _derivatives(futures_annual_deduction_krw=0)
    assert ledger.settle_futures_tax(session_idx=2, config=config) == math.floor(100_000 * 0.11)
    assert ledger.margin == 1_100_000 - math.floor(100_000 * 0.11)


def test_transfers_conserve_both_accounts() -> None:
    """A transfer moves KRW between the two accounts only; the cash side is the sole guard rail."""
    ledger = Ledger(initial_cash=1_000_000)
    ledger.transfer_margin(session_idx=0, amount=50_000)
    ledger.transfer_margin(session_idx=1, amount=-50_000)

    cash_entries = [e for e in ledger.journal if e.account is LedgerAccount.CASH]
    margin_entries = [e for e in ledger.journal if e.account is LedgerAccount.MARGIN]
    assert ledger.cash == 1_000_000 + sum(e.cash_delta for e in cash_entries)
    assert ledger.margin == sum(e.cash_delta for e in margin_entries) == 0
    assert [e.cash_delta for e in cash_entries] == [-50_000, 50_000]

    ledger.transfer_margin(session_idx=2, amount=1_000_000)
    assert (ledger.cash, ledger.margin) == (0, 1_000_000)
    with pytest.raises(ValueError, match="exceeds"):
        ledger.transfer_margin(session_idx=3, amount=1)
    assert (ledger.cash, ledger.margin) == (0, 1_000_000)


def test_inverse_tax_charged_only_on_gains_per_sale() -> None:
    """Disposal tax with no loss offset: the winning half pays, the losing half pays nothing."""
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy_inverse(session_idx=0, units=10, price=10_000, cost_rate=0.0)

    assert ledger.sell_inverse(
        session_idx=1, units=5, price=11_000, cost_rate=0.0, tax_rate=Decimal("0.154")
    ) == math.floor(5_000 * 0.154)
    assert ledger.sell_inverse(
        session_idx=2, units=5, price=9_000, cost_rate=0.0, tax_rate=Decimal("0.154")
    ) == 0
    assert ledger.inverse_units == 0
    assert ledger.cash == 1_000_000 - 100_000 + 55_000 + 45_000 - math.floor(5_000 * 0.154)
    assert _cash(ledger, 1_000_000) == ledger.cash


def test_futures_tax_deduction_without_carry_forward() -> None:
    """The annual deduction applies per year: year 1's loss neither taxes nor reduces year 2's base."""
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(session_idx=0, contracts=4, level=1000.0, multiplier=1_000, cost_rate=0.0)
    assert ledger.settle_variation(
        session_idx=1, prev_level=1000.0, level=1750.0, multiplier=1_000
    ) == -3_000_000
    assert ledger.margin == -3_000_000

    config = _derivatives()
    assert ledger.settle_futures_tax(session_idx=2, config=config) == 0
    assert ledger.settle_variation(
        session_idx=3, prev_level=1750.0, level=750.0, multiplier=1_000
    ) == 4_000_000

    assert ledger.settle_futures_tax(session_idx=4, config=config) == 165_000
    assert ledger.margin == 1_000_000 - 165_000
    taxes = [e for e in ledger.journal if e.kind is JournalKind.FUTURES_TAX]
    assert [(e.account, e.cash_delta) for e in taxes] == [(LedgerAccount.MARGIN, -165_000)]
    assert _cash(ledger, 0) == ledger.cash == 0
    assert ledger.settle_futures_tax(session_idx=5, config=config) == 0


def test_futures_no_ops_and_argument_validation() -> None:
    """Flat positions and a zero transfer are silent no-ops; bad arguments are rejected at the boundary."""
    ledger = Ledger(initial_cash=1_000_000)

    assert ledger.settle_variation(
        session_idx=0, prev_level=1000.0, level=900.0, multiplier=10_000
    ) == 0
    assert ledger.journal == ()

    ledger.trade_futures(session_idx=0, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)
    ledger.trade_futures(session_idx=1, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)
    assert [e.kind for e in ledger.journal] == [JournalKind.FUTURES_TRADE]
    assert ledger.margin == 0

    ledger.roll_futures(session_idx=2, level=1000.0, multiplier=10_000, cost_rate=0.0)
    assert [e.kind for e in ledger.journal] == [JournalKind.FUTURES_TRADE]
    ledger.roll_futures(session_idx=3, level=1000.0, multiplier=10_000, cost_rate=0.001)
    assert ledger.journal[-1].kind is JournalKind.FUTURES_ROLL
    assert ledger.journal[-1].cash_delta == -math.ceil(2 * 0.001 * 10_000 * 1000.0)

    flat = Ledger(initial_cash=1_000_000)
    flat.roll_futures(session_idx=0, level=1000.0, multiplier=10_000, cost_rate=0.001)
    assert flat.journal == ()

    before = ledger.journal
    ledger.transfer_margin(session_idx=4, amount=0)
    assert ledger.journal == before
    with pytest.raises(ValueError, match="cost_rate must be"):
        ledger.roll_futures(session_idx=4, level=1000.0, multiplier=10_000, cost_rate=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="amount must be an integer"):
        ledger.transfer_margin(session_idx=4, amount=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contracts must be"):
        ledger.trade_futures(
            session_idx=4, contracts=True, level=1000.0, multiplier=10_000, cost_rate=0.0  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cost_rate must be"):
        ledger.trade_futures(
            session_idx=4, contracts=1, level=1000.0, multiplier=10_000, cost_rate=1.0
        )
    for bad in (True, float("nan"), -1.0):
        with pytest.raises(PITDataError, match="level must be finite"):
            ledger.trade_futures(
                session_idx=4, contracts=2, level=bad, multiplier=10_000, cost_rate=0.0  # type: ignore[arg-type]
            )
    with pytest.raises(PITDataError, match="prev_level must be finite"):
        ledger.settle_variation(
            session_idx=4, prev_level="x", level=1000.0, multiplier=10_000  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cost_rate must be"):
        ledger.roll_futures(session_idx=4, level=1000.0, multiplier=10_000, cost_rate=2.0)
    with pytest.raises(PITDataError, match="level must be finite"):
        ledger.roll_futures(session_idx=4, level=None, multiplier=10_000, cost_rate=0.0)  # type: ignore[arg-type]
    assert ledger.margin == -math.ceil(2 * 0.001 * 10_000 * 1000.0)


def test_inverse_order_validation_rejects_bad_arguments() -> None:
    """Units, price and the disposal tax rate are validated at the ledger boundary."""
    ledger = Ledger(initial_cash=1_000_000)
    with pytest.raises(ValueError, match="units must be"):
        ledger.buy_inverse(session_idx=0, units=0, price=10_000, cost_rate=0.0)
    with pytest.raises(ValueError, match="units must be"):
        ledger.buy_inverse(session_idx=0, units=1.5, price=10_000, cost_rate=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inverse buy costs"):
        ledger.buy_inverse(session_idx=0, units=200, price=10_000, cost_rate=0.0)

    ledger.buy_inverse(session_idx=0, units=1, price=10_000, cost_rate=0.0)
    with pytest.raises(ValueError, match="exceeds inverse holding"):
        ledger.sell_inverse(
            session_idx=1, units=2, price=10_000, cost_rate=0.0, tax_rate=Decimal("0.154")
        )
    for bad in (0.154, True):
        with pytest.raises(ValueError, match="tax_rate must be"):
            ledger.sell_inverse(
                session_idx=1, units=1, price=10_000, cost_rate=0.0, tax_rate=bad  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="tax_rate must be in"):
        ledger.sell_inverse(
            session_idx=1, units=1, price=10_000, cost_rate=0.0, tax_rate=Decimal("1.0")
        )
    assert ledger.inverse_units == 1
    assert _cash(ledger, 1_000_000) == ledger.cash


def test_mark_requires_a_price_for_an_inverse_holding() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy_inverse(session_idx=0, units=1, price=10_000, cost_rate=0.0)

    with pytest.raises(PITDataError, match="without a price"):
        ledger.mark(
            session_idx=1, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool)
        )
    with pytest.raises(PITDataError, match="invalid inverse price"):
        ledger.mark(
            session_idx=1, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool),
            inverse_price=True,  # type: ignore[arg-type]
        )
    with pytest.raises(PITDataError, match="invalid inverse price"):
        ledger.mark(
            session_idx=1, close=np.zeros(1, dtype=np.int64), present=np.ones(1, dtype=bool),
            inverse_price=-1,
        )


def test_futures_tax_below_one_won_is_not_journaled() -> None:
    """A positive but sub-won tax rounds away, leaving the year-to-date account reset and margin untouched."""
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(session_idx=0, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)
    ledger.settle_variation(session_idx=1, prev_level=1000.0, level=990.0, multiplier=10_000)

    assert ledger.settle_futures_tax(
        session_idx=2,
        config=_derivatives(futures_annual_deduction_krw=0, futures_tax_rate=Decimal("0.000005")),
    ) == 0
    assert ledger.margin == 100_000
    assert ledger.settle_futures_tax(session_idx=3, config=_derivatives()) == 0


def test_long_position_gains_on_a_rising_index() -> None:
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(session_idx=0, contracts=-2, level=300.0, multiplier=50_000, cost_rate=0.0)
    delta = ledger.settle_variation(
        session_idx=1, prev_level=300.0, level=303.0, multiplier=50_000
    )
    assert delta == 2 * 50_000 * 3
    assert ledger.margin == 2 * 50_000 * 3
    assert ledger.settle_futures_tax(
        session_idx=2, config=_derivatives(futures_annual_deduction_krw=0)
    ) == 33_000


def test_long_and_short_marks_are_mirror_images() -> None:
    levels = [300.0, 303.0, 301.5]
    deltas: dict[int, list[int]] = {}
    for contracts in (2, -2):
        ledger = Ledger(initial_cash=0)
        ledger.trade_futures(
            session_idx=0, contracts=contracts, level=levels[0], multiplier=50_000, cost_rate=0.0
        )
        seq = [
            ledger.settle_variation(
                session_idx=i + 1, prev_level=levels[i], level=levels[i + 1], multiplier=50_000
            )
            for i in range(len(levels) - 1)
        ]
        deltas[contracts] = seq
    assert deltas[2] == [-d for d in deltas[-2]]
    for contracts in (2, -2):
        mark = math.floor(-contracts * 50_000 * levels[-1]) - math.floor(
            -contracts * 50_000 * levels[0]
        )
        assert sum(deltas[contracts]) == mark


def test_flip_commission_uses_the_gross_change() -> None:
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(session_idx=0, contracts=1, level=1000.0, multiplier=10_000, cost_rate=0.0)
    rate, level, mult = 0.0005, 1000.0, 10_000
    ledger.trade_futures(
        session_idx=1, contracts=-2, level=level, multiplier=mult, cost_rate=rate
    )
    trades = [e for e in ledger.journal if e.kind is JournalKind.FUTURES_TRADE]
    assert trades[-1].quantity_delta == -3
    commissions = [e for e in ledger.journal if e.kind is JournalKind.FUTURES_COMMISSION]
    assert commissions[-1].cash_delta == -math.ceil(rate * 3 * mult * level)


def test_roll_cost_is_size_symmetric() -> None:
    debits = []
    for contracts in (-3, 3):
        ledger = Ledger(initial_cash=0)
        ledger.trade_futures(
            session_idx=0, contracts=contracts, level=1000.0, multiplier=10_000, cost_rate=0.0
        )
        ledger.roll_futures(session_idx=1, level=1000.0, multiplier=10_000, cost_rate=0.001)
        rolls = [e for e in ledger.journal if e.kind is JournalKind.FUTURES_ROLL]
        debits.append(rolls[-1].cash_delta)
    assert debits[0] == debits[1]


@pytest.mark.parametrize("contracts", [-3, 3])
def test_signed_fractional_marks_conserve_whole_krw(contracts: int) -> None:
    from decimal import ROUND_FLOOR

    levels = [300.01, 303.137, 299.999, 301.123]
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(
        session_idx=0, contracts=contracts, level=levels[0], multiplier=7, cost_rate=0.0
    )
    settled = 0
    for idx, level in enumerate(levels[1:], start=1):
        settled += ledger.settle_variation(
            session_idx=idx, prev_level=levels[idx - 1], level=level, multiplier=7
        )
        marks = [
            int((Decimal(-contracts * 7) * Decimal(str(value))).to_integral_value(rounding=ROUND_FLOOR))
            for value in (levels[0], level)
        ]
        assert settled == ledger.margin == marks[1] - marks[0]


def test_short_only_replay_is_unchanged() -> None:
    ledger = Ledger(initial_cash=0)
    ledger.trade_futures(session_idx=0, contracts=2, level=1000.0, multiplier=10_000, cost_rate=0.0)
    assert ledger.settle_variation(
        session_idx=1, prev_level=1000.0, level=990.0, multiplier=10_000
    ) == 200_000
    ledger.roll_futures(session_idx=2, level=990.0, multiplier=10_000, cost_rate=0.001)
    assert ledger.margin == 200_000 - math.ceil(2 * 0.001 * 2 * 10_000 * 990.0)
    assert [e.quantity_delta for e in ledger.journal if e.kind is JournalKind.FUTURES_TRADE] == [2]
