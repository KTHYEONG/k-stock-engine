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


def test_resolve_facts_pit_conflict_returns_none() -> None:
    from datetime import UTC, datetime
    import polars as pl
    from src.features.qvef import _resolve_facts_pit

    base = datetime(2024, 6, 1, tzinfo=UTC)
    eligible = frozenset(['C1'])
    ff = pl.DataFrame([
        # Conflict: two rows for (C1, 2024Q1, sales) at same available_at
        {'company_id': 'C1', 'fiscal_period': '2024Q1', 'fact': 'sales', 'value': 100.0, 'unit': 'KRW', 'consolidated': True, 'available_at': base, 'restatement_id': 'r0'},
        {'company_id': 'C1', 'fiscal_period': '2024Q1', 'fact': 'sales', 'value': 200.0, 'unit': 'KRW', 'consolidated': True, 'available_at': base, 'restatement_id': 'r1'},
        # Non-conflict: unique row for (C1, 2024Q1, assets)
        {'company_id': 'C1', 'fiscal_period': '2024Q1', 'fact': 'assets', 'value': 500.0, 'unit': 'KRW', 'consolidated': True, 'available_at': base, 'restatement_id': 'r0'},
    ])

    result = _resolve_facts_pit(ff, decision_time=base, eligible_company_ids=eligible)

    # sales conflict -> (None, None)
    assert result.get(('C1', '2024Q1', 'sales'), 'MISSING') == (None, None)
    # assets non-conflict -> (500.0, base)
    val, av = result[('C1', '2024Q1', 'assets')]
    assert val == 500.0
    assert av == base


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
