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


def test_lifecycle_cash_out_removes_position_without_invented_cost() -> None:
    from datetime import datetime
    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerFill, LedgerSide
    from src.core.time import KRX_TZ

    last = datetime(2016,5,18,9,tzinfo=KRX_TZ)
    removed = datetime(2016,5,19,9,tzinfo=KRX_TZ)
    ledger = Ledger('x', 10000.0, last)
    ledger.apply_fill(LedgerFill('buy','KRX:008020',LedgerSide.BUY,2,100.0,0.0,0.0,0.0,last,last))
    entries = ledger.apply_corporate_actions((LedgerCorporateAction('delist','KRX:008020',LedgerActionType.DELISTING_CASH_OUT,removed,1.0,10200.0),), session_open=removed, cash_in_lieu_prices={})
    snap = ledger.snapshot(removed)
    assert ledger.quantity_of('KRX:008020') == 0
    assert snap.settled_cash == 30200.0
    assert dict(entries[0].payload)['valuation_source'] == 'disclosed_settlement'
    assert snap.commission == snap.tax == snap.slippage_cost == 0.0


def test_ledger_exchange_entitlement_then_successor_delivery_preserves_quantity_and_cost() -> None:
    from datetime import datetime
    from decimal import Decimal
    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerFill, LedgerSide, LedgerSuccessorAllocation
    from src.core.time import KRX_TZ

    source_day = datetime(2016, 10, 31, 9, tzinfo=KRX_TZ)
    delivery_day = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    allocation = LedgerSuccessorAllocation('KRX:105560', Decimal('0.5'), Decimal('1'))
    ledger = Ledger('exchange', 10000.0, source_day)
    ledger.record_fill(LedgerFill('buy-source', 'KRX:003450', LedgerSide.BUY, 10, 100.0, 0.0, 0.0, 0.0, source_day, source_day))
    entitlement = LedgerCorporateAction('evt-1:entitlement', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, source_day, 1.0, 0.0, lifecycle_event_id='evt-1', successor_allocations=(allocation,))
    ledger.apply_corporate_actions((entitlement,), session_open=source_day, cash_in_lieu_prices={})
    assert ledger.quantity_of('KRX:003450') == 0
    assert ledger.quantity_of('KRX:105560') == 0
    delivery = LedgerCorporateAction('evt-1:delivery', 'KRX:003450', LedgerActionType.SUCCESSOR_DELIVERY, delivery_day, 1.0, 0.0, lifecycle_event_id='evt-1', successor_allocations=(allocation,))
    entries = ledger.apply_corporate_actions((delivery,), session_open=delivery_day, cash_in_lieu_prices={})
    successor = ledger.snapshot(delivery_day).positions[0]
    assert successor.instrument_id == 'KRX:105560'
    assert successor.quantity == 5
    assert successor.average_cost == 200.0
    assert dict(entries[0].payload)['action_type'] == 'successor_delivery'
    assert ledger.snapshot(delivery_day).settled_cash == 9000.0


def test_ledger_successor_fraction_requires_disclosed_cash_in_lieu() -> None:
    from datetime import datetime
    from decimal import Decimal
    import pytest
    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerFill, LedgerSide, LedgerSuccessorAllocation
    from src.core.pit import PITDataError
    from src.core.time import KRX_TZ

    first = datetime(2016, 10, 31, 9, tzinfo=KRX_TZ)
    second = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    third = datetime(2016, 11, 2, 9, tzinfo=KRX_TZ)
    allocation = LedgerSuccessorAllocation('KRX:105560', Decimal('0.5'), Decimal('1'))
    ledger = Ledger('fraction', 1000.0, first)
    ledger.record_fill(LedgerFill('buy', 'KRX:003450', LedgerSide.BUY, 3, 100.0, 0.0, 0.0, 0.0, first, first))
    ledger.apply_corporate_actions((LedgerCorporateAction('evt-2:e', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, first, 1.0, 0.0, lifecycle_event_id='evt-2', successor_allocations=(allocation,)),), session_open=first, cash_in_lieu_prices={})
    ledger.apply_corporate_actions((LedgerCorporateAction('evt-2:d', 'KRX:003450', LedgerActionType.SUCCESSOR_DELIVERY, second, 1.0, 0.0, lifecycle_event_id='evt-2', successor_allocations=(allocation,)),), session_open=second, cash_in_lieu_prices={})
    assert ledger.quantity_of('KRX:105560') == 1
    with pytest.raises(PITDataError, match='cash-in-lieu'):
        ledger.apply_corporate_actions((LedgerCorporateAction('evt-2:bad-cil', 'KRX:003450', LedgerActionType.CASH_IN_LIEU_SETTLEMENT, third, 1.0, 0.0, lifecycle_event_id='evt-2', settlement_instrument_id='KRX:105560'),), session_open=third, cash_in_lieu_prices={'KRX:105560': 99999.0})
    entries = ledger.apply_corporate_actions((LedgerCorporateAction('evt-2:cil', 'KRX:003450', LedgerActionType.CASH_IN_LIEU_SETTLEMENT, third, 1.0, 0.0, lifecycle_event_id='evt-2', settlement_instrument_id='KRX:105560', cash_settlement_per_entitlement_unit=Decimal('20')),), session_open=third, cash_in_lieu_prices={})
    assert ledger.snapshot(third).settled_cash == 710.0
    assert dict(entries[0].payload)['cash'] == 10.0


def test_ledger_rejects_unmatched_or_conflicting_successor_action_atomically() -> None:
    from datetime import datetime
    from decimal import Decimal
    import pytest
    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerSuccessorAllocation
    from src.core.pit import PITDataError
    from src.core.time import KRX_TZ

    now = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    allocation = LedgerSuccessorAllocation('KRX:105560', Decimal('0.5'), Decimal('1'))
    ledger = Ledger('reject', 100.0, now)
    delivery = LedgerCorporateAction('evt-3:d', 'KRX:003450', LedgerActionType.SUCCESSOR_DELIVERY, now, 1.0, 0.0, lifecycle_event_id='evt-3', successor_allocations=(allocation,))
    with pytest.raises(PITDataError, match='entitlement'):
        ledger.apply_corporate_actions((delivery,), session_open=now, cash_in_lieu_prices={})
    assert ledger.snapshot(now).settled_cash == 100.0
    conflict = LedgerCorporateAction('evt-3:cash', 'KRX:003450', LedgerActionType.DELISTING_CASH_OUT, now, 1.0, 1.0, lifecycle_event_id='evt-3')
    with pytest.raises(PITDataError, match='conflicting lifecycle'):
        ledger.apply_corporate_actions((delivery, conflict), session_open=now, cash_in_lieu_prices={})
    assert ledger.quantity_of('KRX:105560') == 0


def test_ledger_successor_validation_rejects_invalid_terms_before_mutation() -> None:
    from datetime import datetime
    from decimal import Decimal

    import pytest

    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerSuccessorAllocation
    from src.core.pit import PITDataError
    from src.core.time import KRX_TZ

    now = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    with pytest.raises(ValueError, match='non-empty'):
        LedgerSuccessorAllocation('', Decimal('1'), Decimal('1'))
    with pytest.raises(ValueError, match='Decimal'):
        LedgerSuccessorAllocation('KRX:105560', Decimal('1'), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='positive finite'):
        LedgerSuccessorAllocation('KRX:105560', Decimal('0'), Decimal('1'))

    valid = LedgerSuccessorAllocation('KRX:105560', Decimal('1'), Decimal('1'))
    cases = (
        (LedgerCorporateAction('missing-event', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 0.0, successor_allocations=(valid,)), 'lifecycle_event_id'),
        (LedgerCorporateAction('bad-factor', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 2.0, 0.0, lifecycle_event_id='evt', successor_allocations=(valid,)), 'factor'),
        (LedgerCorporateAction('bad-cash', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 1.0, lifecycle_event_id='evt', successor_allocations=(valid,)), 'cash_amount'),
        (LedgerCorporateAction('bad-settlement-id', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(valid,), settlement_instrument_id='KRX:105560'), 'settlement fields'),
        (LedgerCorporateAction('bad-settlement-cash', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(valid,), cash_settlement_per_entitlement_unit=Decimal('1')), 'settlement fields'),
        (LedgerCorporateAction('missing-allocation', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 0.0, lifecycle_event_id='evt'), 'allocations'),
        (LedgerCorporateAction('bad-weight', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, now, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(LedgerSuccessorAllocation('KRX:105560', Decimal('1'), Decimal('0.5')),)), 'sum to 1'),
        (LedgerCorporateAction('missing-cil-instrument', 'KRX:003450', LedgerActionType.CASH_IN_LIEU_SETTLEMENT, now, 1.0, 0.0, lifecycle_event_id='evt', cash_settlement_per_entitlement_unit=Decimal('1')), 'settlement_instrument_id'),
    )
    for action, message in cases:
        ledger = Ledger(f'validate-{action.action_id}', 100.0, now)
        with pytest.raises((PITDataError, ValueError), match=message):
            ledger.apply_corporate_actions((action,), session_open=now, cash_in_lieu_prices={})
        assert ledger.snapshot(now).settled_cash == 100.0


def test_ledger_successor_preconditions_cover_claim_conflicts_and_residuals() -> None:
    from datetime import datetime, timedelta
    from decimal import Decimal

    import pytest

    from src.core.ledger import Ledger, LedgerActionType, LedgerCorporateAction, LedgerFill, LedgerSide, LedgerSuccessorAllocation
    from src.core.pit import PITDataError
    from src.core.time import KRX_TZ

    first = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    second = first + timedelta(days=1)
    allocation = LedgerSuccessorAllocation('KRX:105560', Decimal('0.5'), Decimal('1'))
    no_position = Ledger('no-position', 100.0, first)
    action = LedgerCorporateAction('no-position:e', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, first, 1.0, 0.0, lifecycle_event_id='no-position', successor_allocations=(allocation,))
    with pytest.raises(PITDataError, match='opening position'):
        no_position.apply_corporate_actions((action,), session_open=first, cash_in_lieu_prices={})

    ledger = Ledger('claims', 1000.0, first)
    ledger.record_fill(LedgerFill('buy-1', 'KRX:003450', LedgerSide.BUY, 2, 100.0, 0.0, 0.0, 0.0, first, first))
    entitlement = LedgerCorporateAction('evt:e', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, first, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(allocation,))
    ledger.apply_corporate_actions((entitlement,), session_open=first, cash_in_lieu_prices={})
    mismatch = LedgerCorporateAction('evt:mismatch', 'KRX:003450', LedgerActionType.SUCCESSOR_DELIVERY, second, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(LedgerSuccessorAllocation('KRX:005930', Decimal('0.5'), Decimal('1')),))
    with pytest.raises(PITDataError, match='mismatch'):
        ledger.apply_corporate_actions((mismatch,), session_open=second, cash_in_lieu_prices={})
    ledger.apply_corporate_actions((LedgerCorporateAction('evt:d', 'KRX:003450', LedgerActionType.SUCCESSOR_DELIVERY, second, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(allocation,)),), session_open=second, cash_in_lieu_prices={})
    integral_cil = LedgerCorporateAction('evt:cil', 'KRX:003450', LedgerActionType.CASH_IN_LIEU_SETTLEMENT, second, 1.0, 0.0, lifecycle_event_id='evt', settlement_instrument_id='KRX:105560', cash_settlement_per_entitlement_unit=Decimal('1'))
    with pytest.raises(PITDataError, match='fractional'):
        ledger.apply_corporate_actions((integral_cil,), session_open=second, cash_in_lieu_prices={})
    absent_cil = LedgerCorporateAction('absent:cil', 'KRX:003450', LedgerActionType.CASH_IN_LIEU_SETTLEMENT, second, 1.0, 0.0, lifecycle_event_id='absent', settlement_instrument_id='KRX:105560', cash_settlement_per_entitlement_unit=Decimal('1'))
    with pytest.raises(PITDataError, match='without residual'):
        ledger.apply_corporate_actions((absent_cil,), session_open=second, cash_in_lieu_prices={})
    ledger.record_fill(LedgerFill('buy-2', 'KRX:003450', LedgerSide.BUY, 1, 100.0, 0.0, 0.0, 0.0, second, second))
    duplicate = LedgerCorporateAction('evt:e2', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, second, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(allocation,))
    with pytest.raises(ValueError, match='duplicate entitlement'):
        ledger.apply_corporate_actions((duplicate,), session_open=second, cash_in_lieu_prices={})
    conflict_time = second + timedelta(days=1)
    conflicting_cash = LedgerCorporateAction('evt:cash', 'KRX:003450', LedgerActionType.DELISTING_CASH_OUT, conflict_time, 1.0, 1.0, lifecycle_event_id='evt')
    with pytest.raises(PITDataError, match='conflicting lifecycle'):
        ledger.apply_corporate_actions((conflicting_cash,), session_open=conflict_time, cash_in_lieu_prices={})

    legacy = Ledger('legacy', 100.0, first)
    legacy.apply_corporate_actions((LedgerCorporateAction('legacy:cash', 'KRX:999991', LedgerActionType.DELISTING_CASH_OUT, first, 1.0, 0.0, lifecycle_event_id='legacy-cash'),), session_open=first, cash_in_lieu_prices={})
    legacy.apply_corporate_actions((LedgerCorporateAction('legacy:unsettled', 'KRX:999992', LedgerActionType.DELISTING_UNSETTLED, second, 1.0, 0.0, lifecycle_event_id='legacy-unsettled'),), session_open=second, cash_in_lieu_prices={})
