"""Ledger tests."""
from __future__ import annotations


def test_ledger_buy_sell_settlement_and_nav() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.ledger import Ledger, LedgerFill, LedgerSide

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    ledger = Ledger("account-a", 100_000.0, opened)
    ledger.record_fill(LedgerFill("buy-1", "KRX:005930", LedgerSide.BUY, 10, 1_000.0, 10.0, 0.0, 2.0, opened, datetime(2024, 1, 4, tzinfo=UTC)))
    sold = datetime(2024, 1, 3, tzinfo=UTC)
    due = datetime(2024, 1, 5, tzinfo=UTC)
    ledger.record_fill(LedgerFill("sell-1", "KRX:005930", LedgerSide.SELL, 4, 1_100.0, 4.0, 4.4, 1.0, sold, due))

    before = ledger.snapshot(sold)
    before_nav = before.settled_cash + before.unsettled_cash + before.positions[0].quantity * 1_100.0
    ledger.settle(due)
    ledger.settle(due)
    after = ledger.snapshot(due)
    after_nav = after.settled_cash + after.unsettled_cash + after.positions[0].quantity * 1_100.0

    assert before.settled_cash == pytest.approx(89_990.0)
    assert before.unsettled_cash == pytest.approx(4_391.6)
    assert before.positions[0].quantity == 6
    assert before_nav == pytest.approx(100_981.6)
    assert after.settled_cash == pytest.approx(94_381.6)
    assert after.unsettled_cash == pytest.approx(0.0)
    assert after_nav == pytest.approx(before_nav)


def test_ledger_rejects_cash_short_duplicate_and_buy_tax_atomically() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime
    import pytest
    from src.core.ledger import Ledger, LedgerFill, LedgerSide

    now = datetime(2024, 1, 2, tzinfo=UTC)
    base = LedgerFill("buy", "KRX:005930", LedgerSide.BUY, 1, 1_000.0, 0.0, 0.0, 0.0, now, now)
    ledger = Ledger("account-a", 1_000.0, now)
    ledger.record_fill(base)
    accepted = ledger.snapshot(now)

    with pytest.raises(ValueError, match="duplicate"):
        ledger.record_fill(base)
    with pytest.raises(ValueError, match="holdings"):
        ledger.record_fill(replace(base, fill_id="sell-too-much", side=LedgerSide.SELL, quantity=2))
    with pytest.raises(ValueError, match="tax"):
        replace(base, fill_id="taxed-buy", tax=1.0)
    with pytest.raises(ValueError, match="settled cash"):
        ledger.record_fill(replace(base, fill_id="no-cash"))

    assert ledger.snapshot(now) == accepted


def test_ledger_buy_average_cost_includes_commission() -> None:
    from datetime import UTC, datetime

    from src.core.ledger import Ledger, LedgerFill, LedgerSide

    now = datetime(2024, 1, 2, tzinfo=UTC)
    ledger = Ledger("account-a", 100_000.0, now)
    ledger.record_fill(LedgerFill("buy-1", "KRX:005930", LedgerSide.BUY, 10, 1_000.0, 10.0, 0.0, 0.0, now, now))
    ledger.record_fill(LedgerFill("buy-2", "KRX:005930", LedgerSide.BUY, 10, 1_100.0, 20.0, 0.0, 0.0, now, now))

    assert ledger.snapshot(now).positions[0].average_cost == 1_051.5


def test_ledger_applies_dps_from_opening_quantity_and_split_cash_in_lieu() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerFill, LedgerSide

    opened = datetime(2024, 1, 2, tzinfo=UTC)
    action_open = datetime(2024, 1, 3, tzinfo=UTC)
    ledger = Ledger("account-a", 100.0, opened)
    ledger.record_fill(LedgerFill("buy", "KRX:005930", LedgerSide.BUY, 3, 10.0, 0.0, 0.0, 0.0, opened, opened))
    ledger.apply_corporate_actions((LedgerCorporateAction("dividend-1", "KRX:005930", LedgerActionType.DIVIDEND, action_open, 1.0, 2.0), LedgerCorporateAction("split-1", "KRX:005930", LedgerActionType.SPLIT, action_open, 1.5, 0.0)), session_open=action_open, cash_in_lieu_prices={"KRX:005930": 100.0})

    snapshot = ledger.snapshot(action_open)
    assert snapshot.positions[0].quantity == 4
    assert snapshot.settled_cash == pytest.approx(126.0)


def test_ledger_mark_nav_is_exact_and_does_not_mutate_balances() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.core.ledger import Ledger, LedgerFill, LedgerMark, LedgerSide

    now = datetime(2024, 1, 2, tzinfo=UTC)
    ledger = Ledger("account-a", 100.0, now)
    ledger.record_fill(LedgerFill("buy", "KRX:005930", LedgerSide.BUY, 3, 10.0, 0.0, 0.0, 0.0, now, now))
    before = ledger.snapshot(now)
    nav = ledger.record_mark(LedgerMark("mark-1", now, (("KRX:005930", 12.0),)))
    after = ledger.snapshot(now)

    assert nav.nav == pytest.approx(106.0)
    assert after.settled_cash == before.settled_cash
    assert after.positions == before.positions
