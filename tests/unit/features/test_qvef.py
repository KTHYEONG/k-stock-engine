def test_build_qvef_features_uses_ttm_and_ignores_future_correction() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    identifiers = [f'KRX:{i:06d}' for i in range(10)]
    universe = tuple(UniverseDecision(decision, instrument_id, True, (), 252, 2_000_000_000.0) for instrument_id in identifiers)
    master = pl.DataFrame({'instrument_id': identifiers, 'company_id': identifiers, 'sector': ['Technology'] * 10, 'valid_from': [sessions[0]] * 10, 'valid_to': [None] * 10, 'available_at': [sessions[0]] * 10})
    market = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'trading_value': 100.0 + index, 'market_cap': 200.0 + index, 'available_at': session} for index, instrument_id in enumerate(identifiers) for session in sessions[-21:]])
    flow = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'foreign_net_value': float(index + 1), 'available_at': session} for index, instrument_id in enumerate(identifiers) for session in sessions[-21:-1]])
    facts = []
    for index, company_id in enumerate(identifiers):
        for quarter in range(1, 5):
            for fact, value in {'gross_profit': 10.0 + index, 'net_income': 5.0 + index, 'operating_cash_flow': 4.0 + index, 'assets': 100.0, 'equity': 50.0, 'operating_profit': 20.0 + index, 'sales': 120.0 + index}.items():
                facts.append({'company_id': company_id, 'fiscal_period': f'2024Q{quarter}', 'filing_id': f'{company_id}-24-{quarter}', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': decision})
        for fact, value in {'assets': 100.0, 'operating_profit': 10.0 + index, 'sales': 100.0 + index}.items():
            facts.append({'company_id': company_id, 'fiscal_period': '2023Q4', 'filing_id': f'{company_id}-23-4', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': sessions[0]})
    facts.append({'company_id': identifiers[0], 'fiscal_period': '2024Q4', 'filing_id': 'future-correction', 'fact': 'net_income', 'consolidated': True, 'value': 999.0, 'unit': 'KRW', 'restatement_id': 'r1', 'available_at': decision + timedelta(days=1)})

    rows = build_qvef_features(decision_session=decision, decision_time=decision, calendar=SessionCalendar(sessions), universe=universe, security_master=master, daily_market=market, investor_flow=flow, financial_facts=pl.DataFrame(facts))

    first = next(row for row in rows if row.instrument_id == identifiers[0])
    assert first.gross_profitability == 0.4
    assert first.roe == 0.4
    assert first.cfo_to_assets == 0.16
    assert first.earnings_to_price == 0.1
    assert first.operating_income_change == 0.1
    assert first.sales_growth == 0.2
    assert round(first.operating_margin_change, 12) == round(20.0 / 120.0 - 10.0 / 100.0, 12)
    assert first.foreign_flow_5 == 5.0 / 100.0
    assert first.foreign_flow_20 == 20.0 / 100.0
    assert first.quality_score == -1.0
    assert first.source_available_at[0][1] <= decision


def test_build_qvef_features_neutralizes_negative_earnings_and_rejects_incomplete_flow() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    identifiers = [f'KRX:{i:06d}' for i in range(10)]
    universe = tuple(UniverseDecision(decision, instrument_id, True, (), 252, 2_000_000_000.0) for instrument_id in identifiers)
    master = pl.DataFrame({'instrument_id': identifiers, 'company_id': identifiers, 'sector': ['Technology'] * 10, 'valid_from': [sessions[0]] * 10, 'valid_to': [None] * 10, 'available_at': [sessions[0]] * 10})
    market = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'trading_value': 100.0, 'market_cap': 200.0, 'available_at': session} for instrument_id in identifiers for session in sessions[-21:]])
    flow = pl.DataFrame([{'session': session, 'instrument_id': instrument_id, 'foreign_net_value': 1.0, 'available_at': session} for instrument_id in identifiers for session in sessions[-21:-1] if not (instrument_id == identifiers[1] and session == sessions[-2])])
    facts = []
    for index, company_id in enumerate(identifiers):
        for quarter in range(1, 5):
            values = {'gross_profit': 10.0 + index, 'net_income': -1.0 if index == 0 else 5.0 + index, 'operating_cash_flow': 4.0 + index, 'assets': 100.0, 'equity': 50.0, 'operating_profit': 20.0 + index, 'sales': 120.0 + index}
            facts.extend({'company_id': company_id, 'fiscal_period': f'2024Q{quarter}', 'filing_id': f'{company_id}-{quarter}', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': decision} for fact, value in values.items())
        facts.extend({'company_id': company_id, 'fiscal_period': '2023Q4', 'filing_id': f'{company_id}-old', 'fact': fact, 'consolidated': True, 'value': value, 'unit': 'KRW', 'restatement_id': 'r0', 'available_at': sessions[0]} for fact, value in {'assets': 100.0, 'operating_profit': 10.0 + index, 'sales': 100.0 + index}.items())

    rows = build_qvef_features(decision_session=decision, decision_time=decision, calendar=SessionCalendar(sessions), universe=universe, security_master=master, daily_market=market, investor_flow=flow, financial_facts=pl.DataFrame(facts))

    negative_earnings = next(row for row in rows if row.instrument_id == identifiers[0])
    incomplete_flow = next(row for row in rows if row.instrument_id == identifiers[1])
    assert negative_earnings.value_score is not None
    assert 'earnings_to_price_neutral' in negative_earnings.component_presence
    assert incomplete_flow.foreign_flow_score is None
    assert 'foreign_flow_incomplete' in incomplete_flow.component_presence


def test_qvef_flow_lookback_excludes_same_session_unpublished_flow() -> None:
    from datetime import datetime, timedelta
    from src.core.time import KRX_TZ, SessionCalendar
    from src.features.qvef import _eligible_flow_sessions

    sessions = tuple(datetime(2016, 1, 1, 9, tzinfo=KRX_TZ) + timedelta(days=i) for i in range(21))
    result = _eligible_flow_sessions(calendar=SessionCalendar(sessions), decision_session=sessions[-1], lookback=20)

    assert result == sessions[:-1]
    assert sessions[-1] not in result


def test_qvef_flow_lookback_rejects_invalid_requests() -> None:
    from datetime import datetime
    import pytest
    from src.core.time import KRX_TZ, SessionCalendar
    from src.features.qvef import _eligible_flow_sessions

    session = datetime(2016, 1, 1, 9, tzinfo=KRX_TZ)
    calendar = SessionCalendar((session,))
    with pytest.raises(ValueError, match='positive'):
        _eligible_flow_sessions(calendar=calendar, decision_session=session, lookback=0)
    with pytest.raises(ValueError, match='does not contain'):
        _eligible_flow_sessions(calendar=calendar, decision_session=datetime(2016, 1, 2, 9, tzinfo=KRX_TZ), lookback=1)
    assert _eligible_flow_sessions(calendar=calendar, decision_session=session, lookback=2) == ()


def test_resolve_master_for_eligible_pit_dedup_correct() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.features.qvef import _resolve_master_for_eligible

    base = datetime(2024, 6, 1, tzinfo=UTC)
    decision_time = base + timedelta(days=10)
    decision_session = base + timedelta(days=10)

    # 3 rows for iid_A: one future, two past (latest valid_from wins)
    # 1 row for iid_B: past
    # iid_C not in eligible_iids
    sm = pl.DataFrame([
        {'instrument_id': 'KRX:A', 'company_id': 'CA', 'sector': 'Tech', 'valid_from': base, 'valid_to': None, 'available_at': base},
        {'instrument_id': 'KRX:A', 'company_id': 'CA', 'sector': 'TechNew', 'valid_from': base + timedelta(days=5), 'valid_to': None, 'available_at': base + timedelta(days=5)},
        {'instrument_id': 'KRX:A', 'company_id': 'CA', 'sector': 'TechFuture', 'valid_from': base + timedelta(days=20), 'valid_to': None, 'available_at': base + timedelta(days=20)},
        {'instrument_id': 'KRX:B', 'company_id': 'CB', 'sector': 'Finance', 'valid_from': base, 'valid_to': None, 'available_at': base},
        {'instrument_id': 'KRX:C', 'company_id': 'CC', 'sector': 'Energy', 'valid_from': base, 'valid_to': None, 'available_at': base},
    ])

    result = _resolve_master_for_eligible(
        sm,
        decision_time=decision_time,
        decision_session=decision_session,
        eligible_iids=['KRX:A', 'KRX:B'],
    )

    # KRX:A: future row excluded (available_at+20 > decision_time+10); latest past row has sector='TechNew'
    assert 'KRX:A' in result
    assert result['KRX:A']['sector'] == 'TechNew'
    # KRX:B: included
    assert 'KRX:B' in result
    assert result['KRX:B']['company_id'] == 'CB'
    # KRX:C: not in eligible_iids -> excluded
    assert 'KRX:C' not in result


def test_resolve_master_for_eligible_accepts_same_krx_date_open_snapshot() -> None:
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.features.qvef import _resolve_master_for_eligible

    kst = ZoneInfo("Asia/Seoul")
    session = datetime(2017, 12, 28, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:005930"],
            "company_id": ["005930"],
            "sector": ["Technology"],
            "valid_from": [datetime(2017, 12, 28, 9, tzinfo=kst)],
            "valid_to": [datetime(2017, 12, 28, 9, tzinfo=kst)],
            "available_at": [datetime(2017, 12, 28, 9, tzinfo=kst)],
        }
    )

    result = _resolve_master_for_eligible(
        master,
        decision_time=datetime(2017, 12, 28, 15, 30, tzinfo=UTC),
        decision_session=session,
        eligible_iids=["KRX:005930"],
    )

    assert result["KRX:005930"]["company_id"] == "005930"

def test_staleness_bisect_matches_linear_scan() -> None:
    """Verifies bisect_right replacement produces identical staleness index."""
    from bisect import bisect_right
    from datetime import UTC, datetime, timedelta

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100))

    def legacy_find_latest_session_idx(sessions: tuple, latest_av: datetime) -> int:
        idx = -1
        for i, s in enumerate(sessions):
            if s <= latest_av:
                idx = i
            else:
                break
        return idx

    test_cases = [
        sessions[0],                    # exactly first session
        sessions[-1],                   # exactly last session
        sessions[50],                   # mid session
        sessions[50] + timedelta(hours=12),  # between sessions
        sessions[0] - timedelta(hours=1),    # before all sessions -> -1
        sessions[-1] + timedelta(hours=1),   # after all sessions -> last
    ]
    for latest_av in test_cases:
        expected = legacy_find_latest_session_idx(sessions, latest_av)
        bisect_idx = bisect_right(sessions, latest_av) - 1
        assert bisect_idx == expected, f'latest_av={latest_av}: expected={expected}, got={bisect_idx}'


def test_build_qvef_features_pit_correctness_preserved_after_opt() -> None:
    """Regression: optimized Polars dedup produces same scores as original small-frame path."""
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(70))
    decision = sessions[-1]
    identifiers = ['KRX:000001', 'KRX:000002']
    universe = tuple(UniverseDecision(decision, iid, True, (), 252, 5_000_000_000.0) for iid in identifiers)

    # Build bloated master: 50 duplicate snapshots per instrument
    master_rows_bloated = []
    for iid, cid, sector in zip(identifiers, ['C1', 'C2'], ['Tech', 'Tech']):  # noqa: B905
        for j in range(50):
            master_rows_bloated.append({'instrument_id': iid, 'company_id': cid, 'sector': sector,  # noqa: PERF401
                                        'valid_from': sessions[0], 'valid_to': None,
                                        'available_at': sessions[j]})
    master_bloated = pl.DataFrame(master_rows_bloated)
    assert master_bloated.height == 100  # 2 instruments x 50

    # Clean master (1 row per instrument)
    master_clean = pl.DataFrame([{'instrument_id': iid, 'company_id': cid, 'sector': sector,
                                   'valid_from': sessions[0], 'valid_to': None,
                                   'available_at': sessions[0]}
                                  for iid, cid, sector in zip(identifiers, ['C1', 'C2'], ['Tech', 'Tech'])])  # noqa: B905

    # Shared market and facts
    market = pl.DataFrame([{'session': s, 'instrument_id': iid, 'trading_value': 100.0, 'market_cap': 200.0, 'available_at': s}
                            for iid in identifiers for s in sessions[-21:]])
    flow = pl.DataFrame([{'session': s, 'instrument_id': iid, 'foreign_net_value': 1.0, 'available_at': s}
                          for iid in identifiers for s in sessions[-21:-1]])
    facts_list = []
    for iid, cid in zip(identifiers, ['C1', 'C2']):  # noqa: B007, B905
        for q in range(1, 5):
            for fact, val in {'gross_profit': 10.0, 'net_income': 5.0, 'operating_cash_flow': 4.0,
                              'assets': 100.0, 'equity': 50.0, 'operating_profit': 20.0, 'sales': 120.0}.items():
                facts_list.append({'company_id': cid, 'fiscal_period': f'2024Q{q}', 'filing_id': f'{cid}-{q}',
                                   'fact': fact, 'consolidated': True, 'value': val, 'unit': 'KRW',
                                   'restatement_id': 'r0', 'available_at': decision})
        for fact, val in {'assets': 100.0, 'operating_profit': 10.0, 'sales': 100.0}.items():
            facts_list.append({'company_id': cid, 'fiscal_period': '2023Q4', 'filing_id': f'{cid}-23-4',
                               'fact': fact, 'consolidated': True, 'value': val, 'unit': 'KRW',
                               'restatement_id': 'r0', 'available_at': sessions[0]})
    facts = pl.DataFrame(facts_list)
    cal = SessionCalendar(sessions)

    rows_clean = build_qvef_features(decision_session=decision, decision_time=decision, calendar=cal,
                                     universe=universe, security_master=master_clean,
                                     daily_market=market, investor_flow=flow, financial_facts=facts)
    rows_bloated = build_qvef_features(decision_session=decision, decision_time=decision, calendar=cal,
                                       universe=universe, security_master=master_bloated,
                                       daily_market=market, investor_flow=flow, financial_facts=facts)

    assert len(rows_clean) == len(rows_bloated) == len(identifiers)
    by_clean = {r.instrument_id: r for r in rows_clean}
    by_bloated = {r.instrument_id: r for r in rows_bloated}
    for iid in identifiers:
        assert by_clean[iid].gross_profitability == by_bloated[iid].gross_profitability, f'{iid} gross_profitability mismatch'
        assert by_clean[iid].quality_score == by_bloated[iid].quality_score, f'{iid} quality_score mismatch'


def test_resolve_facts_pit_by_basis_keeps_both_accounting_bases() -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.features.qvef import resolve_facts_pit_by_basis

    # Given: one filing reported on both bases plus a future-dated row.
    decision_time = datetime(2024, 5, 1, 6, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            'company_id': ['C1', 'C1', 'C1'],
            'fiscal_period': ['2024Q1', '2024Q1', '2024Q1'],
            'fact': ['net_income', 'net_income', 'assets'],
            'consolidated': [True, False, True],
            'value': [100.0, 70.0, 900.0],
            'unit': ['KRW', 'KRW', 'KRW'],
            'available_at': [
                datetime(2024, 4, 1, tzinfo=UTC),
                datetime(2024, 4, 1, tzinfo=UTC),
                datetime(2024, 6, 1, tzinfo=UTC),
            ],
        }
    )

    # When
    resolved = resolve_facts_pit_by_basis(
        frame, decision_time=decision_time, eligible_company_ids=frozenset({'C1'})
    )

    # Then: both bases survive; the future row is excluded.
    assert resolved[('C1', '2024Q1', 'net_income', 'consolidated')][0] == 100.0
    assert resolved[('C1', '2024Q1', 'net_income', 'separate')][0] == 70.0
    assert ('C1', '2024Q1', 'assets', 'consolidated') not in resolved


def test_resolve_facts_pit_by_basis_nulls_ambiguous_and_non_krw_rows() -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.features.qvef import resolve_facts_pit_by_basis

    # Given: an ambiguous tie on one key and a foreign-currency row on another.
    decision_time = datetime(2024, 5, 1, 6, 30, tzinfo=UTC)
    tie = datetime(2024, 4, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            'company_id': ['C1', 'C1', 'C1'],
            'fiscal_period': ['2024Q1', '2024Q1', '2024Q1'],
            'fact': ['net_income', 'net_income', 'equity'],
            'consolidated': [True, True, True],
            'value': [100.0, 120.0, 500.0],
            'unit': ['KRW', 'KRW', 'USD'],
            'available_at': [tie, tie, tie],
        }
    )

    # When
    resolved = resolve_facts_pit_by_basis(
        frame, decision_time=decision_time, eligible_company_ids=frozenset({'C1'})
    )

    # Then: ambiguity and non-KRW both fail closed to an unavailable value.
    assert resolved[('C1', '2024Q1', 'net_income', 'consolidated')] == (None, None)
    assert resolved[('C1', '2024Q1', 'equity', 'consolidated')] == (None, None)
    assert resolve_facts_pit_by_basis(
        pl.DataFrame(), decision_time=decision_time, eligible_company_ids=frozenset({'C1'})
    ) == {}


def test_select_company_basis_prefers_the_basis_covering_the_window() -> None:
    from datetime import UTC, datetime

    from src.features.qvef import select_company_basis

    # Given: 4 separate quarters versus a single stray consolidated quarter.
    avail = datetime(2024, 4, 1, tzinfo=UTC)
    periods = ('2023Q2', '2023Q3', '2023Q4', '2024Q1')
    facts = ('net_income', 'assets')
    by_basis: dict[tuple[str, str, str, str], tuple[float | None, datetime | None]] = {}
    for period in periods:
        for fact in facts:
            by_basis[('C1', period, fact, 'separate')] = (10.0, avail)
    by_basis[('C1', '2024Q1', 'net_income', 'consolidated')] = (99.0, avail)

    # When
    selection = select_company_basis(
        facts_by_basis=by_basis, company_id='C1', window_quarters=4, facts=facts
    )

    # Then
    assert selection is not None
    assert selection.basis == 'separate'
    assert selection.latest_period == '2024Q1'
    assert selection.covered_cells == 8


def test_select_company_basis_breaks_ties_by_preference_and_returns_none_when_empty() -> None:
    from datetime import UTC, datetime

    from src.features.qvef import ACCOUNTING_BASIS_PREFERENCE, select_company_basis

    # Given: identical coverage on both bases.
    avail = datetime(2024, 4, 1, tzinfo=UTC)
    facts = ('net_income',)
    by_basis: dict[tuple[str, str, str, str], tuple[float | None, datetime | None]] = {}
    for basis in ACCOUNTING_BASIS_PREFERENCE:
        for period in ('2023Q4', '2024Q1'):
            by_basis[('C1', period, 'net_income', basis)] = (5.0, avail)

    # When
    tied = select_company_basis(
        facts_by_basis=by_basis, company_id='C1', window_quarters=2, facts=facts
    )

    # Then: preference order decides, never dict iteration order.
    assert tied is not None
    assert tied.basis == 'consolidated'
    assert tied.covered_cells == 2

    # And: an unknown company yields no selection at all.
    assert select_company_basis(
        facts_by_basis=by_basis, company_id='C2', window_quarters=2, facts=facts
    ) is None


def test_select_company_basis_ignores_null_valued_cells() -> None:
    from datetime import UTC, datetime

    from src.features.qvef import select_company_basis

    # Given: a later consolidated period whose only cell is unavailable.
    avail = datetime(2024, 4, 1, tzinfo=UTC)
    facts = ('net_income',)
    by_basis: dict[tuple[str, str, str, str], tuple[float | None, datetime | None]] = {
        ('C1', '2023Q4', 'net_income', 'consolidated'): (7.0, avail),
        ('C1', '2024Q1', 'net_income', 'consolidated'): (None, None),
    }

    # When
    selection = select_company_basis(
        facts_by_basis=by_basis, company_id='C1', window_quarters=2, facts=facts
    )

    # Then: the null cell neither counts nor becomes the latest period.
    assert selection is not None
    assert selection.latest_period == '2023Q4'
    assert selection.covered_cells == 1


def test_build_qvef_features_never_mixes_accounting_bases_in_one_ttm() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    # Given: 4 quarters where only 2 are consolidated and the other 2 separate.
    kst_sessions = tuple(
        datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(40)
    )
    calendar = SessionCalendar(kst_sessions)
    decision_session = kst_sessions[30]
    decision_time = decision_session.replace(hour=15, minute=30)
    avail = datetime(2023, 12, 1, tzinfo=UTC)
    quarters = ('2023Q1', '2023Q2', '2023Q3', '2023Q4')
    bases = [True, False, True, False]
    rows: list[dict[str, object]] = []
    for quarter, consolidated in zip(quarters, bases, strict=True):
        for fact, value in (
            ('gross_profit', 50.0),
            ('net_income', 30.0),
            ('operating_cash_flow', 40.0),
            ('assets', 1000.0),
            ('equity', 600.0),
        ):
            rows.append(
                {
                    'company_id': 'C1',
                    'fiscal_period': quarter,
                    'fact': fact,
                    'consolidated': consolidated,
                    'value': value,
                    'unit': 'KRW',
                    'available_at': avail,
                }
            )
    financial_facts = pl.DataFrame(rows)
    security_master = pl.DataFrame(
        {
            'instrument_id': ['KRX:000001'],
            'ticker': ['000001'],
            'company_id': ['C1'],
            'sector': ['Technology'],
            'valid_from': [kst_sessions[0]],
            'valid_to': [kst_sessions[-1]],
            'available_at': [kst_sessions[0]],
        }
    )
    daily_market = pl.DataFrame(
        {
            'session': list(kst_sessions),
            'instrument_id': ['KRX:000001'] * len(kst_sessions),
            'close': [1000.0] * len(kst_sessions),
            'trading_value': [1.0e9] * len(kst_sessions),
            'market_cap': [1.0e11] * len(kst_sessions),
            'available_at': [s.replace(hour=15, minute=30) for s in kst_sessions],
        }
    )
    universe = (UniverseDecision(decision_session, 'KRX:000001', True, (), 400, 1.0e9),)

    # When
    built = build_qvef_features(
        decision_session=decision_session,
        decision_time=decision_time,
        calendar=calendar,
        universe=universe,
        security_master=security_master,
        daily_market=daily_market,
        investor_flow=pl.DataFrame(),
        financial_facts=financial_facts,
    )

    # Then: no cross-basis TTM is fabricated.
    assert len(built) == 1
    assert built[0].gross_profitability is None
    assert built[0].cfo_to_assets is None
    assert built[0].quality_score is None


def test_select_company_basis_rejects_unparsable_fiscal_period() -> None:
    from datetime import UTC, datetime

    from src.features.qvef import select_company_basis

    # Given: a period string that cannot be decremented into a prior quarter.
    avail = datetime(2024, 4, 1, tzinfo=UTC)
    by_basis: dict[tuple[str, str, str, str], tuple[float | None, datetime | None]] = {
        ("C1", "FY2023", "net_income", "consolidated"): (7.0, avail),
    }

    # When: a 4-quarter window is requested from an unparsable period.
    selection = select_company_basis(
        facts_by_basis=by_basis, company_id="C1", window_quarters=4, facts=("net_income",)
    )

    # Then: no quarter window can be built, so the basis fails closed to no selection.
    assert selection is None


def test_build_qvef_features_leaves_fundamentals_none_when_company_has_no_facts() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.features.qvef import build_qvef_features
    from src.strategy.universe import UniverseDecision

    # Given: a universe name whose company has no financial facts at all.
    sessions = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(40))
    calendar = SessionCalendar(sessions)
    decision_session = sessions[30]
    decision_time = decision_session.replace(hour=15, minute=30)
    security_master = pl.DataFrame(
        {
            "instrument_id": ["KRX:000001"],
            "ticker": ["000001"],
            "company_id": ["C1"],
            "sector": ["Technology"],
            "valid_from": [sessions[0]],
            "valid_to": [sessions[-1]],
            "available_at": [sessions[0]],
        }
    )
    daily_market = pl.DataFrame(
        {
            "session": list(sessions),
            "instrument_id": ["KRX:000001"] * len(sessions),
            "close": [1000.0] * len(sessions),
            "trading_value": [1.0e9] * len(sessions),
            "market_cap": [1.0e11] * len(sessions),
            "available_at": [s.replace(hour=15, minute=30) for s in sessions],
        }
    )
    universe = (UniverseDecision(decision_session, "KRX:000001", True, (), 400, 1.0e9),)

    # When: no basis can be selected for the company.
    built = build_qvef_features(
        decision_session=decision_session,
        decision_time=decision_time,
        calendar=calendar,
        universe=universe,
        security_master=security_master,
        daily_market=daily_market,
        investor_flow=pl.DataFrame(),
        financial_facts=pl.DataFrame(),
    )

    # Then: every fundamental stays unavailable rather than defaulting.
    assert len(built) == 1
    assert built[0].book_to_price is None
    assert built[0].roe is None
    assert built[0].quality_score is None
    assert built[0].value_score is None
