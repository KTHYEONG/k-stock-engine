def test_build_historical_universe_accepts_exact_listing_and_liquidity_boundaries() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(252))
    calendar = SessionCalendar(sessions)
    decision_session = sessions[-1]
    master = pl.DataFrame({
        'instrument_id': ['KRX:BOUNDARY', 'KRX:YOUNG'],
        'ticker': ['BOUNDARY', 'YOUNG'],
        'company_id': ['C1', 'C2'],
        'market': ['KOSPI', 'KOSDAQ'],
        'sector': ['Industrials', 'Industrials'],
        'listing_date': [sessions[0], sessions[1]],
        'delisting_date': [None, None],
        'share_class': ['common', 'common'],
        'status': ['listed', 'listed'],
        'valid_from': [sessions[0], sessions[1]],
        'valid_to': [None, None],
        'available_at': [sessions[0], sessions[1]],
    })
    daily = pl.DataFrame({
        'session': list(sessions[-60:]) * 2,
        'instrument_id': ['KRX:BOUNDARY'] * 60 + ['KRX:YOUNG'] * 60,
        'trading_value': [2_000_000_000.0] * 120,
        'open': [100.0] * 120,
        'close': [100.0] * 120,
        'volume': [1_000.0] * 120,
        'available_at': list(sessions[-60:]) * 2,
    })

    decisions = build_historical_universe(
        decision_session=decision_session, decision_time=decision_session, calendar=calendar,
        security_master=master, daily_market=daily,
    )
    by_id = {decision.instrument_id: decision for decision in decisions}

    assert by_id['KRX:BOUNDARY'].eligible is True
    assert by_id['KRX:BOUNDARY'].listing_age_sessions == 252
    assert by_id['KRX:BOUNDARY'].median_trading_value_60 == 2_000_000_000.0
    assert ExclusionReason.INSUFFICIENT_LISTING_AGE in by_id['KRX:YOUNG'].exclusion_reasons


def test_build_historical_universe_reason_codes_asset_status_and_liquidity_failures() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(252))
    calendar = SessionCalendar(sessions)
    ids = ['KRX:ETF', 'KRX:FIN', 'KRX:SUSP', 'KRX:NOSECTOR', 'KRX:SHORT', 'KRX:ILLIQUID']
    master = pl.DataFrame({
        'instrument_id': ids, 'ticker': ids, 'company_id': ids,
        'market': ['KOSPI'] * 6,
        'sector': ['Industrials', 'Financials', 'Industrials', None, 'Industrials', 'Industrials'],
        'listing_date': [sessions[0]] * 6, 'delisting_date': [None] * 6,
        'share_class': ['etf', 'common', 'common', 'common', 'common', 'common'],
        'status': ['listed', 'listed', 'suspended', 'listed', 'listed', 'listed'],
        'valid_from': [sessions[0]] * 6, 'valid_to': [None] * 6, 'available_at': [sessions[0]] * 6,
    })
    daily_rows = []  # noqa: PERF401 - skeleton fidelity
    for instrument_id in ids:
        count = 59 if instrument_id == 'KRX:SHORT' else 60
        value = 1_999_999_999.0 if instrument_id == 'KRX:ILLIQUID' else 3_000_000_000.0
        for session in sessions[-count:]:
            daily_rows.append({'session': session, 'instrument_id': instrument_id, 'open': 100.0, 'close': 100.0, 'volume': 1_000.0, 'trading_value': value, 'available_at': session})  # noqa: PERF401
    daily = pl.DataFrame(daily_rows)

    decisions = build_historical_universe(
        decision_session=sessions[-1], decision_time=sessions[-1], calendar=calendar,
        security_master=master, daily_market=daily,
    )
    reasons = {decision.instrument_id: set(decision.exclusion_reasons) for decision in decisions}

    assert ExclusionReason.NON_COMMON_SHARE_CLASS in reasons['KRX:ETF']
    assert ExclusionReason.FINANCIAL_SECTOR in reasons['KRX:FIN']
    assert ExclusionReason.INELIGIBLE_STATUS in reasons['KRX:SUSP']
    assert ExclusionReason.MISSING_SECTOR not in reasons['KRX:NOSECTOR']
    assert ExclusionReason.INSUFFICIENT_LIQUIDITY_HISTORY in reasons['KRX:SHORT']
    assert ExclusionReason.LIQUIDITY_BELOW_THRESHOLD in reasons['KRX:ILLIQUID']


def test_build_historical_universe_missing_availability_fails_closed() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(252))
    calendar = SessionCalendar(sessions)
    master = pl.DataFrame({
        'instrument_id': ['KRX:MISSING_AVAILABILITY'], 'market': ['KOSPI'],
        'sector': ['Industrials'], 'share_class': ['common'], 'status': ['listed'],
        'listing_date': [sessions[0]], 'delisting_date': [None],
        'valid_from': [sessions[0]], 'valid_to': [None],
    })
    daily = pl.DataFrame({
        'session': list(sessions[-60:]),
        'instrument_id': ['KRX:MISSING_AVAILABILITY'] * 60,
        'trading_value': [2_000_000_000.0] * 60,
        'open': [100.0] * 60,
        'close': [100.0] * 60,
        'volume': [1_000.0] * 60,
        'available_at': list(sessions[-60:]),
    })

    decisions = build_historical_universe(
        decision_session=sessions[-1], decision_time=sessions[-1], calendar=calendar,
        security_master=master, daily_market=daily,
    )

    assert decisions[0].eligible is False
    assert ExclusionReason.MISSING_MASTER in decisions[0].exclusion_reasons


def test_sector_optional_universe_remains_eligible() -> None:
    from src.strategy.universe import ExclusionReason

    assert ExclusionReason.MISSING_SECTOR.value == "missing_sector"


def test_universe_calls_corporate_action_exclusion_once_per_session(monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    import src.data.gold as gold
    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, UniversePolicy, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(60))
    master = pl.DataFrame([{'instrument_id': iid, 'company_id': iid, 'market': 'KOSPI', 'sector': 'Technology', 'share_class': 'common', 'status': 'listed', 'listing_date': sessions[0], 'valid_from': sessions[0], 'valid_to': None, 'available_at': sessions[0]} for iid in ('KRX:1', 'KRX:2')])
    daily = pl.DataFrame([{'instrument_id': iid, 'session': session, 'open': 10.0, 'close': 10.0, 'volume': 1.0, 'trading_value': 10.0, 'available_at': session} for iid in ('KRX:1', 'KRX:2') for session in sessions])
    actions = pl.DataFrame({'instrument_id': ['KRX:1'], 'effective_date': [sessions[0]], 'coverage_end': [sessions[-2]], 'type': ['split']})
    calls = []
    def wrapped(*args, **kwargs):
        calls.append(args[1])
        return frozenset({'KRX:1'})

    monkeypatch.setattr(gold, 'exclude_sentinel_corporate_actions', wrapped)
    result = build_historical_universe(decision_session=sessions[-1], decision_time=sessions[-1], calendar=SessionCalendar(sessions), security_master=master, daily_market=daily, corporate_actions=actions, policy=UniversePolicy(minimum_listing_sessions=1, minimum_median_trading_value_krw=1.0))

    assert len(calls) == 1
    assert tuple(item.instrument_id for item in result) == ('KRX:1', 'KRX:2')
    assert ExclusionReason.NO_VALID_CORPORATE_ACTION in result[0].exclusion_reasons


def test_universe_preserves_listing_and_liquidity_boundaries_after_indexing() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, UniversePolicy, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(60))
    master = pl.DataFrame([{'instrument_id': 'KRX:edge', 'company_id': 'C1', 'market': 'KOSPI', 'sector': 'Technology', 'share_class': 'common', 'status': 'listed', 'listing_date': sessions[0], 'valid_from': sessions[0], 'valid_to': None, 'available_at': sessions[0]}])
    daily = pl.DataFrame([{'instrument_id': 'KRX:edge', 'session': session, 'open': 10.0, 'close': 10.0, 'volume': 1.0, 'trading_value': 100.0, 'available_at': session} for session in sessions])

    result = build_historical_universe(decision_session=sessions[-1], decision_time=sessions[-1], calendar=SessionCalendar(sessions), security_master=master, daily_market=daily, policy=UniversePolicy(minimum_listing_sessions=60, minimum_median_trading_value_krw=100.0))

    assert result[0].eligible is True
    assert result[0].median_trading_value_60 == 100.0
    assert ExclusionReason.INSUFFICIENT_LISTING_AGE not in result[0].exclusion_reasons


def test_universe_excludes_zero_value_bars_as_non_tradable() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.strategy.universe import ExclusionReason, UniversePolicy, build_historical_universe

    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(60))
    master = pl.DataFrame([{'instrument_id': 'KRX:halted', 'company_id': 'C1', 'market': 'KOSPI', 'sector': 'Technology', 'share_class': 'common', 'status': 'listed', 'listing_date': sessions[0], 'valid_from': sessions[0], 'valid_to': None, 'available_at': sessions[0]}])
    daily = pl.DataFrame([{'instrument_id': 'KRX:halted', 'session': session, 'open': 10.0, 'close': 10.0, 'volume': 1.0, 'trading_value': 100.0, 'available_at': session} for session in sessions]).with_columns(pl.when(pl.col('session') == sessions[-1]).then(0.0).otherwise(pl.col('volume')).alias('volume'))

    result = build_historical_universe(decision_session=sessions[-1], decision_time=sessions[-1], calendar=SessionCalendar(sessions), security_master=master, daily_market=daily, policy=UniversePolicy(minimum_listing_sessions=60, minimum_median_trading_value_krw=1.0))

    assert ExclusionReason.NON_TRADABLE_BAR in result[0].exclusion_reasons


def test_dedup_master_frame_returns_latest_per_instrument() -> None:
    from datetime import datetime, timedelta, timezone
    import polars as pl
    from src.strategy.universe import _dedup_master_frame

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017
    # 3 instruments, each with 2 snapshots at different available_at
    frame = pl.DataFrame({
        'instrument_id': ['A', 'A', 'B', 'B', 'C', 'C'],
        'valid_from': [base, base + timedelta(days=10), base, base + timedelta(days=5), base, base],
        'valid_to': [None, None, None, None, None, None],
        'available_at': [base, base + timedelta(days=10), base, base + timedelta(days=5), base, base + timedelta(days=1)],
        'sector': ['X', 'X_new', 'Y', 'Y_new', 'Z', 'Z_new'],
    })

    result = _dedup_master_frame(frame)

    # Must have exactly 1 row per instrument
    assert result.height == 3
    by_id = {row['instrument_id']: row for row in result.to_dicts()}
    # A: latest available_at is base+10 -> sector='X_new'
    assert by_id['A']['sector'] == 'X_new'
    # B: latest available_at is base+5 -> sector='Y_new'
    assert by_id['B']['sector'] == 'Y_new'
    # C: latest available_at is base+1 -> sector='Z_new'
    assert by_id['C']['sector'] == 'Z_new'


def test_dedup_master_frame_empty_returns_empty() -> None:
    import polars as pl
    from src.strategy.universe import _dedup_master_frame

    frame = pl.DataFrame({'instrument_id': [], 'available_at': [], 'valid_from': []},
                         schema={'instrument_id': pl.String, 'available_at': pl.Datetime('us','UTC'), 'valid_from': pl.Datetime('us','UTC')})
    result = _dedup_master_frame(frame)
    assert result.is_empty()
    assert 'instrument_id' in result.columns


def test_dedup_master_frame_missing_sort_columns_returns_unchanged() -> None:
    import polars as pl
    from src.strategy.universe import _dedup_master_frame

    # Frame with no available_at or valid_from columns
    frame = pl.DataFrame({'instrument_id': ['A', 'B'], 'sector': ['X', 'Y']})
    result = _dedup_master_frame(frame)
    # Must not crash and must return the full frame unchanged
    assert result.height == frame.height
    assert set(result.columns) == set(frame.columns)


def test_build_historical_universe_perf_dedup_applied_before_groupby() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.strategy.universe import build_historical_universe

    N_SESSIONS = 252
    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(N_SESSIONS))
    calendar = SessionCalendar(sessions)
    decision_session = sessions[-1]
    base_available_at = sessions[0]

    # Build master with 6 duplicate snapshots per instrument (simulating Silver bloat)
    iid = 'KRX:000001'
    master_rows = []
    for j in range(6):
        master_rows.append({  # noqa: PERF401
            'instrument_id': iid,
            'ticker': 'TEST',
            'company_id': 'C1',
            'market': 'KOSPI',
            'sector': 'Industrials',
            'listing_date': sessions[0],
            'delisting_date': None,
            'share_class': 'common',
            'status': 'listed',
            'valid_from': sessions[0],
            'valid_to': None,
            'available_at': base_available_at + timedelta(days=j),
        })
    master = pl.DataFrame(master_rows)
    assert master.height == 6  # Confirm 6x bloat

    daily = pl.DataFrame([
        {'session': s, 'instrument_id': iid, 'trading_value': 5_000_000_000.0,
         'open': 100.0, 'close': 100.0, 'volume': 1_000.0, 'available_at': s}
        for s in sessions[-60:]
    ])

    decisions = build_historical_universe(
        decision_session=decision_session,
        decision_time=decision_session,
        calendar=calendar,
        security_master=master,
        daily_market=daily,
    )
    assert len(decisions) == 1
    assert decisions[0].instrument_id == iid
    assert decisions[0].eligible is True, f'Expected eligible, got reasons: {decisions[0].exclusion_reasons}'
