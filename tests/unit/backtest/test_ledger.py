"""Integer ledger: journal equality, splits, dividends, exits, and marks."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.events import DividendEvent
from src.backtest.ledger import JournalKind, Ledger
from src.core.pit import PITDataError


def _entitlement(instrument_idx: int, ex: int, pay: int, dps: int) -> DividendEvent:
    return DividendEvent(
        instrument_idx=instrument_idx, ex_session_idx=ex, pay_session_idx=pay, dps_krw=dps
    )


def _cash(ledger: Ledger, initial_cash: int) -> int:
    return initial_cash + sum(entry.cash_delta for entry in ledger.journal)


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
    ledger.settle_dividends(session_idx=6)
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
    with pytest.raises(ValueError):
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
    ledger.settle_dividends(session_idx=20)
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
    with pytest.raises(ValueError):
        Ledger(initial_cash=-1)
    with pytest.raises(ValueError):
        Ledger(initial_cash=1.5)
    ledger = Ledger(initial_cash=100)
    with pytest.raises(ValueError):
        ledger.deposit(session_idx=0, amount=0)
    with pytest.raises(ValueError):
        ledger.buy(session_idx=0, instrument_idx=0, quantity=True, price=10, commission=0)


def test_sell_above_holding_rejected() -> None:
    ledger = Ledger(initial_cash=1_000_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=5, price=10_000, commission=0)
    with pytest.raises(ValueError):
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
    ledger.settle_dividends(session_idx=9)
    assert ledger.journal == ()


def test_close_exit_at_zero_records_total_loss() -> None:
    ledger = Ledger(initial_cash=100_000)
    ledger.buy(session_idx=0, instrument_idx=0, quantity=10, price=5_000, commission=0)
    ledger.close_exit(session_idx=1, instrument_idx=0, price=0)
    assert ledger.positions() == {}
    assert _cash(ledger, 100_000) == 50_000
