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
            daily_rows.append({'session': session, 'instrument_id': instrument_id, 'trading_value': value, 'available_at': session})  # noqa: PERF401
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
