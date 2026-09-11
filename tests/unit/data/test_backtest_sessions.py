def test_backtest_session_builder_rejects_missing_next_open_bar(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    day = datetime(2024, 1, 2, tzinfo=UTC)
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: pl.DataFrame({'session': [day], 'instrument_id': ['KRX:1'], 'open': [1.0], 'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1.0], 'trading_value': [1.0], 'market_cap': [1.0], 'shares_outstanding': [1.0], 'available_at': [day]})}, root=tmp_path)
    calendar = SessionCalendar((day, datetime(2024, 1, 3, tzinfo=UTC)))

    with pytest.raises(PITDataError, match=r'next.session.*bar'):
        build_backtest_sessions(snapshot_repository=repository, calendar=calendar, start=day, end=day, decision_time_of=lambda session: session)


def test_backtest_session_builder_allows_suspended_zero_volume_bar(tmp_path) -> None:
    """Suspended bars (volume=0, trading_value=0) with valid open/close are allowed for mark-to-market.
    Execution rejection on these bars is handled by fill_model, not session builder."""
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    day = datetime(2024, 1, 2, tzinfo=UTC)
    next_day = datetime(2024, 1, 3, tzinfo=UTC)
    frame = pl.DataFrame({'session': [day, next_day], 'instrument_id': ['KRX:1', 'KRX:1'], 'open': [1.0, 1.0], 'high': [1.0, 1.0], 'low': [1.0, 1.0], 'close': [1.0, 1.0], 'volume': [0.0, 1.0], 'trading_value': [0.0, 1.0], 'market_cap': [1.0, 1.0], 'shares_outstanding': [1.0, 1.0], 'available_at': [day, next_day]})
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)

    with pytest.raises(PITDataError, match='missing PIT security master'):
        build_backtest_sessions(snapshot_repository=repository, calendar=SessionCalendar((day, next_day)), start=day, end=day, decision_time_of=lambda session: session)


def test_backtest_sessions_optimized_builder_fast_instantiation(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    d1 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
    d2 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    frame = pl.DataFrame({
        'session': [d1, d1, d2, d2],
        'instrument_id': ['KRX:005930', 'KRX:000660', 'KRX:005930', 'KRX:000660'],
        'open': [70000.0, 120000.0, 71000.0, 121000.0],
        'high': [71000.0, 122000.0, 72000.0, 123000.0],
        'low': [69500.0, 119000.0, 70500.0, 120000.0],
        'close': [70500.0, 121000.0, 71500.0, 122000.0],
        'volume': [1000.0, 500.0, 1100.0, 550.0],
        'trading_value': [70500000.0, 60500000.0, 78650000.0, 67100000.0],
        'available_at': [
            d1.replace(hour=15, minute=30),
            d1.replace(hour=15, minute=30),
            d2.replace(hour=15, minute=30),
            d2.replace(hour=15, minute=30),
        ],
    })
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    calendar = SessionCalendar((d1, d2))
    with pytest.raises(PITDataError, match='missing PIT security master'):
        build_backtest_sessions(
            snapshot_repository=repo,
            calendar=calendar,
            start=d1,
            end=d1,
            decision_time_of=lambda s: s.replace(hour=15, minute=30),
        )


def test_backtest_sessions_rejects_duplicate_bars(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    d1 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
    d2 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    frame = pl.DataFrame({
        'session': [d1, d1, d2],
        'instrument_id': ['KRX:005930', 'KRX:005930', 'KRX:005930'],
        'open': [70000.0, 70000.0, 71000.0],
        'high': [71000.0, 71000.0, 72000.0],
        'low': [69500.0, 69500.0, 70500.0],
        'close': [70500.0, 70500.0, 71500.0],
        'volume': [1000.0, 1000.0, 1100.0],
        'trading_value': [70500000.0, 70500000.0, 78650000.0],
        'available_at': [
            d1.replace(hour=15, minute=30),
            d1.replace(hour=15, minute=30),
            d2.replace(hour=15, minute=30),
        ],
    })
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    calendar = SessionCalendar((d1, d2))
    with pytest.raises(PITDataError, match=r'duplicate bar'):
        build_backtest_sessions(
            snapshot_repository=repo,
            calendar=calendar,
            start=d1,
            end=d1,
            decision_time_of=lambda s: s.replace(hour=15, minute=30),
        )


def test_backtest_sessions_rejects_available_at_leak(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    d1 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
    d2 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
    frame = pl.DataFrame({
        'session': [d1, d2],
        'instrument_id': ['KRX:005930', 'KRX:005930'],
        'open': [70000.0, 71000.0],
        'high': [71000.0, 72000.0],
        'low': [69500.0, 70500.0],
        'close': [70500.0, 71500.0],
        'volume': [1000.0, 1100.0],
        'trading_value': [70500000.0, 78650000.0],
        'available_at': [
            d1.replace(hour=16, minute=0),
            d2.replace(hour=15, minute=30),
        ],
    })
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    calendar = SessionCalendar((d1, d2))
    with pytest.raises(PITDataError, match=r'available_at after decision time'):
        build_backtest_sessions(
            snapshot_repository=repo,
            calendar=calendar,
            start=d1,
            end=d1,
            decision_time_of=lambda s: s.replace(hour=15, minute=30),
        )


from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

def test_market_input_policy_fixed() -> None:
    policy = BacktestMarketInputsPolicy()
    assert (policy.adtv_sessions, policy.volatility_sessions, policy.market_volatility_sessions) == (20, 60, 60)
    assert (policy.annualization_sessions, policy.unexplained_price_jump_threshold) == (252, 0.5)


def test_unresolved_action_blocks_certification() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.schemas import PITDataError

    first = datetime(2024, 1, 2, 9, tzinfo=UTC)
    second = datetime(2024, 1, 3, 9, tzinfo=UTC)
    bars = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:000001', 'KRX:000001'], 'close': [100.0, 40.0], 'available_at': [first, second]})
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='UTC')})
    with pytest.raises(PITDataError, match='unexplained'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=actions, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())


def test_validate_action_split_explains_jump_and_rejects_bad_rows() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage
    from src.data.schemas import PITDataError

    first = datetime(2024, 1, 2, 9, tzinfo=UTC)
    second = datetime(2024, 1, 3, 9, tzinfo=UTC)
    bars = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:000001', 'KRX:000001'], 'close': [100.0, 40.0], 'available_at': [first, second]})
    calendar = SessionCalendar((first, second))
    decide = lambda value: value.replace(hour=15, minute=30)  # noqa: E731
    policy = BacktestMarketInputsPolicy()
    good = pl.DataFrame({'effective_session': [second], 'instrument_id': ['KRX:000001'], 'action_type': ['split'], 'available_at': [second], 'factor': [2.0], 'cash_amount': [0.0]})
    covered = validate_corporate_action_coverage(daily_market=bars, corporate_actions=good, calendar=calendar, decision_time_of=decide, policy=policy)
    assert len(covered.actions_by_session[second]) == 1
    bad_type = pl.DataFrame({'effective_session': [second], 'instrument_id': ['KRX:000001'], 'action_type': ['merger'], 'available_at': [second]})
    with pytest.raises(PITDataError, match='unsupported'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=bad_type, calendar=calendar, decision_time_of=decide, policy=policy)
    late = pl.DataFrame({'effective_session': [second], 'instrument_id': ['KRX:000001'], 'action_type': ['split'], 'available_at': [second + timedelta(days=1)]})
    with pytest.raises(PITDataError, match='late'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=late, calendar=calendar, decision_time_of=decide, policy=policy)
    ambiguous = pl.DataFrame({'effective_session': [second, second], 'instrument_id': ['KRX:000001', 'KRX:000001'], 'action_type': ['split', 'split'], 'available_at': [second, second]})
    with pytest.raises(PITDataError, match='ambiguous'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=ambiguous, calendar=calendar, decision_time_of=decide, policy=policy)
    naive_eff = pl.DataFrame({'effective_session': [second.replace(tzinfo=None)], 'instrument_id': ['KRX:000001'], 'action_type': ['split'], 'available_at': [second]})
    with pytest.raises(PITDataError, match='invalid corporate action timing'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=naive_eff, calendar=calendar, decision_time_of=decide, policy=policy)
    naive_avail = pl.DataFrame({'effective_session': [second], 'instrument_id': ['KRX:000001'], 'action_type': ['split'], 'available_at': [second.replace(tzinfo=None)]})
    with pytest.raises(PITDataError, match='invalid corporate action timing'):
        validate_corporate_action_coverage(daily_market=bars, corporate_actions=naive_avail, calendar=calendar, decision_time_of=decide, policy=policy)
    dividend = pl.DataFrame({'effective_session': [second.isoformat()], 'instrument_id': ['KRX:000001'], 'action_type': ['dividend'], 'available_at': [second], 'cash_amount': [50.0]})
    calm = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:000001', 'KRX:000001'], 'close': [100.0, 101.0], 'available_at': [first, second]})
    div_map = validate_corporate_action_coverage(daily_market=calm, corporate_actions=dividend, calendar=calendar, decision_time_of=decide, policy=policy)
    assert div_map.actions_by_session[second][0].cash_amount == 50.0
    zero_prev = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:000001', 'KRX:000001'], 'close': [0.0, 40.0], 'available_at': [first, second]})
    actions_empty = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='UTC')})
    with pytest.raises(PITDataError, match='invalid corporate-action market value'):
        validate_corporate_action_coverage(daily_market=zero_prev, corporate_actions=actions_empty, calendar=calendar, decision_time_of=decide, policy=policy)


def test_backtest_sessions_rejects_global_sector_and_covers_zero_cap_market(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import (
        BacktestMarketInputsPolicy,
        _rolling_inputs,
        build_backtest_sessions,
    )
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    first = datetime(2024, 1, 2, 9, tzinfo=UTC)
    second = datetime(2024, 1, 3, 9, tzinfo=UTC)
    frame = pl.DataFrame({
        'session': [first, second],
        'instrument_id': ['KRX:000001', 'KRX:000001'],
        'open': [100.0, 101.0],
        'close': [100.0, 101.0],
        'volume': [10.0, 10.0],
        'trading_value': [1000.0, 1010.0],
        'market_cap': [1.0, 1.0],
        'available_at': [first, second],
    })
    _adtv, _vols, market_vol = _rolling_inputs(frame, BacktestMarketInputsPolicy(), (first, second))
    assert second not in market_vol
    master = pl.DataFrame({
        'instrument_id': ['KRX:000001'],
        'sector': ['__GLOBAL__'],
        'valid_from': [first],
        'valid_to': [second],
        'available_at': [first],
    })
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='UTC')})
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    with pytest.raises(PITDataError, match='invalid PIT sector'):
        build_backtest_sessions(
            snapshot_repository=repo,
            calendar=SessionCalendar((first, second)),
            start=first,
            end=first,
            decision_time_of=lambda s: s.replace(hour=15, minute=30),
            security_master=master,
            corporate_actions=actions,
        )


def test_build_sessions_pit_rolling_inputs(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import SilverTable
    from src.data.snapshot import PITSnapshotRepository

    from datetime import timedelta

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(62))
    avail = [day.replace(hour=15, minute=30) for day in days]
    frame = pl.DataFrame({
        'session': [day for day in days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * len(days),
        'open': [100.0, 50.0] * len(days),
        'close': [value for index in range(len(days)) for value in (100.0 + index + (index % 3) * 0.1, 50.0 + index + (index % 4) * 0.1)],
        'volume': [1000.0] * (2 * len(days)),
        'trading_value': [101000.0, 51000.0] * len(days),
        'market_cap': [1e10, 5e9] * len(days),
        'available_at': [slot for slot in avail for _ in ('KRX:A', 'KRX:B')],
    })
    master = pl.DataFrame({
        'instrument_id': ['KRX:A', 'KRX:B'],
        'sector': ['Technology', 'Healthcare'],
        'valid_from': [days[0], days[0]],
        'valid_to': [days[-1], days[-1]],
        'available_at': [days[0], days[0]],
    })
    actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='UTC')})
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    sessions = build_backtest_sessions(
        snapshot_repository=repo,
        calendar=SessionCalendar(days),
        start=days[60],
        end=days[60],
        decision_time_of=lambda s: s.replace(hour=15, minute=30),
        security_master=master,
        corporate_actions=actions,
    )
    assert len(sessions) == 1
    snapshot = sessions[0].market_snapshot
    assert snapshot['sectors'] == {'KRX:A': 'Technology', 'KRX:B': 'Healthcare'}
    assert snapshot['adtv20']['KRX:A'] == 101000.0
    assert snapshot['market_caps']['KRX:A'] == 1e10
    assert sessions[0].actions == ()


def test_backtest_market_inputs_fail_closed_on_invalid_metadata(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import (
        BacktestMarketInputsPolicy,
        _resolve_sector,
        _rolling_inputs,
        build_backtest_sessions,
        validate_corporate_action_coverage,
    )
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    first = datetime(2024, 1, 2, 9, tzinfo=UTC)
    second = datetime(2024, 1, 3, 9, tzinfo=UTC)
    calendar = SessionCalendar((first, second))
    policy = BacktestMarketInputsPolicy()
    with pytest.raises(ValueError, match='immutable'):
        BacktestMarketInputsPolicy(adtv_sessions=2)
    no_action = pl.DataFrame({'effective_session': [first], 'instrument_id': ['KRX:A'], 'action_type': ['no_action'], 'available_at': [first]})
    calm = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 101.0], 'available_at': [first, second]})
    with pytest.raises(PITDataError, match='legacy no_action corporate-action evidence requires rebuild'):
        validate_corporate_action_coverage(daily_market=calm, corporate_actions=no_action, calendar=calendar, decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=policy)
    for action, message in (
        (pl.DataFrame({'action_type': ['split'], 'instrument_id': ['KRX:A'], 'available_at': [first]}), 'invalid corporate action timing'),
        (pl.DataFrame({'effective_session': [datetime(2025, 1, 1, tzinfo=UTC)], 'instrument_id': ['KRX:A'], 'action_type': ['split'], 'available_at': [first]}), 'outside calendar'),
    ):
        with pytest.raises(PITDataError, match=message):
            validate_corporate_action_coverage(daily_market=calm, corporate_actions=action, calendar=calendar, decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=policy)
    with pytest.raises(PITDataError, match='missing rolling columns'):
        _rolling_inputs(calm, policy, (first, second))
    with pytest.raises(PITDataError, match='duplicate sessions'):
        _rolling_inputs(pl.DataFrame({'session': [first], 'instrument_id': ['KRX:A'], 'close': [100.0], 'trading_value': [1.0], 'market_cap': [1.0]}), policy, (first, first))
    outside = pl.DataFrame({'session': [datetime(2025, 1, 1, tzinfo=UTC)], 'instrument_id': ['KRX:A'], 'close': [100.0], 'trading_value': [1.0], 'market_cap': [1.0]})
    with pytest.raises(PITDataError, match='outside calendar'):
        _rolling_inputs(outside, policy, (first, second))
    invalid = pl.DataFrame({'session': [first], 'instrument_id': ['KRX:A'], 'close': [100.0], 'trading_value': [1.0], 'market_cap': [0.0]})
    with pytest.raises(PITDataError, match='invalid rolling market input'):
        _rolling_inputs(invalid, policy, (first, second))
    with pytest.raises(PITDataError, match='missing PIT security master'):
        _resolve_sector(security_master=None, instrument_id='KRX:A', session=first, decision_time=first)
    frame = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'open': [100.0, 101.0], 'close': [100.0, 101.0], 'volume': [1.0, 1.0], 'trading_value': [1.0, 1.0], 'market_cap': [1.0, 1.0], 'available_at': [first, second]})
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    master = pl.DataFrame({'instrument_id': ['KRX:A'], 'sector': ['Technology'], 'valid_from': [first], 'valid_to': [second], 'available_at': [first]})
    with pytest.raises(PITDataError, match='missing PIT corporate actions'):
        build_backtest_sessions(snapshot_repository=repository, calendar=calendar, start=first, end=first, decision_time_of=lambda value: value.replace(hour=15, minute=30), security_master=master, corporate_actions=None)
    empty_actions = pl.DataFrame(schema={'effective_session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'action_type': pl.String, 'available_at': pl.Datetime(time_zone='UTC')})
    with pytest.raises(PITDataError, match='insufficient rolling'):
        build_backtest_sessions(snapshot_repository=repository, calendar=calendar, start=first, end=first, decision_time_of=lambda value: value.replace(hour=15, minute=30), security_master=master, corporate_actions=empty_actions)


# test_backtest_reconciles_bonus_and_uses_adjusted_returns
def test_validate_corporate_action_coverage_reconciles_bonus_and_uses_adjusted_return() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.ledger import LedgerActionType
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    krx = ZoneInfo('Asia/Seoul')
    previous = datetime(2016, 11, 22, 9, tzinfo=krx)
    current = datetime(2016, 11, 23, 9, tzinfo=krx)
    daily = pl.DataFrame({'session': [previous, current], 'instrument_id': ['KRX:027410', 'KRX:027410'], 'close': [168500.0, 82600.0], 'shares_outstanding': [24773964.0, 49547928.0], 'market_cap': [4174406934000.0, 4092658852800.0]})
    actions = pl.DataFrame({'effective_session': [current], 'share_listing_date': [current], 'share_delta': [24773964], 'instrument_id': ['KRX:027410'], 'action_type': ['bonus_issue'], 'action_id': ['20161107000214'], 'factor': [2.0], 'cash_amount': [0.0], 'available_at': [datetime(2016, 11, 8, 9, tzinfo=krx)]})

    coverage = validate_corporate_action_coverage(daily_market=daily, corporate_actions=actions, calendar=SessionCalendar((previous, current)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())

    assert coverage.actions_by_session[current][0].action_type is LedgerActionType.SPLIT
    assert coverage.research_returns_by_key[(current, 'KRX:027410')] == pytest.approx(2.0 * 82600.0 / 168500.0 - 1.0)


# test_backtest_rejects_missing_late_or_legacy_sentinel_coverage
def test_validate_corporate_action_coverage_rejects_legacy_no_action() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage
    from src.data.schemas import PITDataError

    krx = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=krx)
    second = datetime(2024, 1, 3, 9, tzinfo=krx)
    daily = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 49.0], 'shares_outstanding': [10.0, 10.0], 'market_cap': [1000.0, 490.0]})
    sentinel = pl.DataFrame({'effective_session': [first], 'coverage_end': [second], 'instrument_id': ['KRX:A'], 'action_type': ['no_action'], 'action_id': ['legacy'], 'factor': [1.0], 'cash_amount': [0.0], 'available_at': [first]})

    with pytest.raises(PITDataError, match='legacy no_action corporate-action evidence requires rebuild'):
        validate_corporate_action_coverage(daily_market=daily, corporate_actions=sentinel, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())


def test_validate_corporate_action_coverage_exercises_factor_and_value_guards() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage
    from src.data.schemas import PITDataError

    tz = ZoneInfo("Asia/Seoul")
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    second = datetime(2024, 1, 3, 9, tzinfo=tz)
    daily = pl.DataFrame({"session": [first, second], "instrument_id": ["KRX:A", "KRX:A"], "close": [100.0, 50.0], "shares_outstanding": [10.0, 20.0], "market_cap": [1000.0, 1000.0]})
    base = {"effective_session": [second], "instrument_id": ["KRX:A"], "action_id": ["a"], "available_at": [first]}
    for action_type, factor, cash, pattern in (("split", 1.0, 0.0, "factor"), ("reverse_split", 2.0, 0.0, "factor"), ("dividend", 1.0, -1.0, "cash")):
        row = {**base, "action_type": [action_type], "factor": [factor], "cash_amount": [cash]}
        with pytest.raises(PITDataError, match=pattern):
            validate_corporate_action_coverage(daily_market=daily, corporate_actions=pl.DataFrame(row), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())
    bad = daily.with_columns(pl.Series("close", ["bad", "50"]))
    with pytest.raises(PITDataError, match="market value"):
        validate_corporate_action_coverage(daily_market=bad, corporate_actions=pl.DataFrame(), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())

    reverse_daily = pl.DataFrame({"session": [first, second], "instrument_id": ["KRX:R", "KRX:R"], "close": [50.0, 100.0], "shares_outstanding": [20.0, 10.0], "market_cap": [1000.0, 1000.0]})
    reverse = pl.DataFrame({**base, "instrument_id": ["KRX:R"], "action_type": ["reverse_split"], "factor": [0.5], "cash_amount": [0.0]})
    coverage = validate_corporate_action_coverage(daily_market=reverse_daily, corporate_actions=reverse, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())
    assert coverage.actions_by_session[second][0].action_type.value == "reverse_split"

    dividend_daily = pl.DataFrame({"session": [first, second], "instrument_id": ["KRX:D", "KRX:D"], "close": [100.0, 40.0], "shares_outstanding": [10.0, 10.0], "market_cap": [1000.0, 400.0]})
    dividend = pl.DataFrame({**base, "instrument_id": ["KRX:D"], "action_type": ["dividend"], "factor": [1.0], "cash_amount": [60.0]})
    assert validate_corporate_action_coverage(daily_market=dividend_daily, corporate_actions=dividend, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy()).research_returns_by_key


def test_validate_bonus_issue_accepts_delayed_listing_share_delta() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    tz = ZoneInfo('Asia/Seoul')
    price_previous = datetime(2016, 11, 22, 9, tzinfo=tz)
    price_session = datetime(2016, 11, 23, 9, tzinfo=tz)
    listing_previous = datetime(2016, 12, 13, 9, tzinfo=tz)
    listing_session = datetime(2016, 12, 14, 9, tzinfo=tz)
    calendar = SessionCalendar((price_previous, price_session, listing_previous, listing_session))
    daily = pl.DataFrame({'session': list(calendar.sessions), 'instrument_id': ['KRX:027410'] * 4, 'close': [168500.0, 82600.0, 86200.0, 88000.0], 'shares_outstanding': [24773964.0, 24773964.0, 24773964.0, 49547625.0], 'market_cap': [4174412934000.0, 2046329426400.0, 2135515696800.0, 4360191000000.0]})
    actions = pl.DataFrame({'effective_session': [price_session], 'share_listing_date': [listing_session], 'share_delta': [24773661], 'instrument_id': ['KRX:027410'], 'action_type': ['bonus_issue'], 'action_id': ['20161107000214'], 'factor': [2.0], 'cash_amount': [0.0], 'available_at': [datetime(2016, 11, 8, 9, tzinfo=tz)]})

    coverage = validate_corporate_action_coverage(daily_market=daily, corporate_actions=actions, calendar=calendar, decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())

    assert tuple(coverage.actions_by_session) == (price_session,)
    assert coverage.research_returns_by_key[(listing_session, 'KRX:027410')] == 88000.0 / 86200.0 - 1.0


def test_validate_bonus_issue_rejects_listing_delta_or_market_cap_mismatch() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage
    from src.data.schemas import PITDataError

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    price = datetime(2024, 1, 3, 9, tzinfo=tz)
    listing = datetime(2024, 1, 4, 9, tzinfo=tz)
    calendar = SessionCalendar((first, price, listing))
    daily = pl.DataFrame({'session': [first, price, listing], 'instrument_id': ['KRX:A'] * 3, 'close': [100.0, 49.0, 50.0], 'shares_outstanding': [10.0, 10.0, 19.0], 'market_cap': [1000.0, 490.0, 950.0]})
    actions = pl.DataFrame({'effective_session': [price], 'share_listing_date': [listing], 'share_delta': [10], 'instrument_id': ['KRX:A'], 'action_type': ['bonus_issue'], 'action_id': ['a'], 'factor': [2.0], 'cash_amount': [0.0], 'available_at': [first], 'evidence_status': ['verified'], 'evidence_reason': [None]})

    with pytest.raises(PITDataError, match='unreconciled corporate action listed shares'):
        validate_corporate_action_coverage(daily_market=daily, corporate_actions=actions, calendar=calendar, decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())

    legacy = actions.drop(['share_listing_date', 'share_delta'])
    with pytest.raises(PITDataError, match='legacy bonus_issue corporate-action evidence requires rebuild'):
        validate_corporate_action_coverage(daily_market=daily, corporate_actions=legacy, calendar=calendar, decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())


def test_resolve_backtest_evidence_excludes_unknown_without_adjusting_verified() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    second = datetime(2024, 1, 3, 9, tzinfo=tz)
    daily = pl.DataFrame({'session': [first, second, first, second], 'instrument_id': ['KRX:A', 'KRX:A', 'KRX:B', 'KRX:B'], 'close': [100.0, 50.0, 100.0, 40.0], 'shares_outstanding': [10.0, 20.0, 10.0, 10.0], 'market_cap': [1000.0, 1000.0, 1000.0, 400.0]})
    actions = pl.DataFrame({'instrument_id': ['KRX:A', 'KRX:B'], 'action_id': ['split-a', 'unknown-b'], 'action_type': ['split', 'unresolved'], 'effective_session': [second, second], 'factor': [2.0, 1.0], 'cash_amount': [0.0, 0.0], 'available_at': [first, first], 'share_listing_date': [None, None], 'share_delta': [None, None], 'evidence_status': ['verified', 'unresolved'], 'evidence_reason': [None, 'unsupported_merger']})
    resolution = resolve_backtest_corporate_action_evidence(daily_market=daily, corporate_actions=actions, calendar=SessionCalendar((first, second)), policy=BacktestMarketInputsPolicy())
    assert resolution.excluded_instruments == frozenset()
    assert resolution.exclusion_reasons['KRX:B'] == ('unsupported_merger',)
    assert set(resolution.eligible_daily_market['instrument_id'].unique().to_list()) == {'KRX:A', 'KRX:B'}
    assert resolution.quarantine_sessions_by_instrument['KRX:B'] == (second,)
    assert resolution.eligible_daily_market.filter(pl.col('instrument_id') == 'KRX:B')['session'].to_list() == [first]
    assert resolution.verified_corporate_actions['action_id'].to_list() == ['split-a']


def test_build_backtest_sessions_uses_evidence_resolution_before_rolling_inputs(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import SilverTable
    from src.data.snapshot import PITSnapshotRepository

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(62))
    frame = pl.DataFrame({'session': [day for day in days for _ in ('KRX:A', 'KRX:B')], 'instrument_id': ['KRX:A', 'KRX:B'] * len(days), 'open': [100.0, 100.0] * len(days), 'close': [100.0 + index for index in range(len(days)) for _ in ('KRX:A', 'KRX:B')], 'volume': [1000.0] * (2 * len(days)), 'trading_value': [100000.0] * (2 * len(days)), 'market_cap': [1000000.0] * (2 * len(days)), 'shares_outstanding': [10000.0] * (2 * len(days)), 'available_at': [day.replace(hour=15, minute=30) for day in days for _ in ('KRX:A', 'KRX:B')]})
    master = pl.DataFrame({'instrument_id': ['KRX:A', 'KRX:B'], 'sector': ['Technology', 'Healthcare'], 'valid_from': [days[0], days[0]], 'valid_to': [days[-1], days[-1]], 'available_at': [days[0], days[0]]})
    actions = pl.DataFrame({'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'], 'effective_session': [days[30]], 'factor': [1.0], 'cash_amount': [0.0], 'available_at': [days[0]], 'share_listing_date': [None], 'share_delta': [None], 'evidence_status': ['unresolved'], 'evidence_reason': ['unsupported_merger']})
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)
    sessions = build_backtest_sessions(snapshot_repository=repository, calendar=SessionCalendar(days), start=days[60], end=days[60], decision_time_of=lambda value: value.replace(hour=15, minute=30), security_master=master, corporate_actions=actions)
    assert len(sessions) == 1
    assert set(sessions[0].market_snapshot['mark_prices']) == {'KRX:A'}


def test_find_unexplained_price_discontinuities_returns_only_uncovered_jump() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    from src.data.backtest_sessions import find_unexplained_price_discontinuities

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    second = datetime(2024, 1, 3, 9, tzinfo=tz)
    daily = pl.DataFrame({'session': [first, second, first, second], 'instrument_id': ['KRX:V', 'KRX:V', 'KRX:U', 'KRX:U'], 'close': [100.0, 50.0, 100.0, 40.0], 'shares_outstanding': [10.0, 20.0, 10.0, 10.0], 'market_cap': [1000.0, 1000.0, 1000.0, 400.0]})
    verified = pl.DataFrame({'instrument_id': ['KRX:V'], 'effective_session': [second]})
    jumps = find_unexplained_price_discontinuities(daily_market=daily, verified_actions=verified, threshold=0.5)
    assert jumps.to_dicts() == [{'instrument_id': 'KRX:U', 'session': second}]


def test_resolve_backtest_evidence_quarantines_event_window_not_whole_instrument() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(64))
    daily = pl.DataFrame({
        'session': [day for day in days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * len(days),
        'close': [100.0, 100.0] * len(days),
    })
    actions = pl.DataFrame({
        'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'],
        'effective_session': [days[1]], 'factor': [1.0], 'cash_amount': [0.0],
        'available_at': [days[0]], 'share_listing_date': [None], 'share_delta': [None],
        'evidence_status': ['unresolved'], 'evidence_reason': ['unsupported_merger'],
    })

    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
    )

    kept_b = result.eligible_daily_market.filter(pl.col('instrument_id') == 'KRX:B')['session'].to_list()
    assert days[0] in kept_b
    assert days[1] not in kept_b
    assert days[61] not in kept_b
    assert days[62] in kept_b
    assert result.quarantine_sessions_by_instrument['KRX:B'] == days[1:62]
    assert 'KRX:B' not in result.excluded_instruments


def test_resolve_backtest_evidence_rejects_unresolved_event_without_calendar_time() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    daily = pl.DataFrame({'session': [session], 'instrument_id': ['KRX:B'], 'close': [100.0]})
    actions = pl.DataFrame({
        'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'],
        'effective_session': [None], 'factor': [1.0], 'cash_amount': [0.0],
        'available_at': [session], 'evidence_status': ['unresolved'],
        'evidence_reason': ['unsupported_merger'],
    })

    with pytest.raises(PITDataError, match='effective'):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily, corporate_actions=actions, calendar=SessionCalendar((session,)),
            policy=BacktestMarketInputsPolicy(),
        )


def test_resolve_backtest_evidence_accepts_iso_string_effective_session() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(4))
    daily = pl.DataFrame({
        'session': list(days),
        'instrument_id': ['KRX:B'] * len(days),
        'close': [100.0] * len(days),
    })
    actions = pl.DataFrame({
        'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'],
        'effective_session': [days[1].isoformat()], 'factor': [1.0], 'cash_amount': [0.0],
        'available_at': [days[0]], 'share_listing_date': [None], 'share_delta': [None],
        'evidence_status': ['unresolved'], 'evidence_reason': ['unsupported_merger'],
    })
    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
    )
    assert result.quarantine_sessions_by_instrument['KRX:B'] == days[1:]
    assert result.eligible_daily_market['session'].to_list() == [days[0]]


def test_resolve_backtest_evidence_rejects_bad_event_timing() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence
    from src.data.schemas import PITDataError

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    other = datetime(2024, 2, 1, 9, tzinfo=UTC)
    daily = pl.DataFrame({'session': [session], 'instrument_id': ['KRX:B'], 'close': [100.0]})
    base = {
        'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'],
        'factor': [1.0], 'cash_amount': [0.0], 'available_at': [session],
        'share_listing_date': [None], 'share_delta': [None],
        'evidence_status': ['unresolved'], 'evidence_reason': ['unsupported_merger'],
    }
    naive = pl.DataFrame({**base, 'effective_session': [session.replace(tzinfo=None)]})
    with pytest.raises(PITDataError, match='effective'):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily, corporate_actions=naive, calendar=SessionCalendar((session,)),
            policy=BacktestMarketInputsPolicy(),
        )
    garbage = pl.DataFrame({**base, 'effective_session': ['not-a-date']})
    with pytest.raises(PITDataError, match='effective'):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily, corporate_actions=garbage, calendar=SessionCalendar((session,)),
            policy=BacktestMarketInputsPolicy(),
        )
    outside = pl.DataFrame({**base, 'effective_session': [other]})
    with pytest.raises(PITDataError, match='outside calendar'):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily, corporate_actions=outside, calendar=SessionCalendar((session,)),
            policy=BacktestMarketInputsPolicy(),
        )


def test_resolve_backtest_evidence_filters_verified_on_effective_date() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(4))
    daily = pl.DataFrame({
        'session': [day for day in days for _ in ('KRX:A', 'KRX:B')],
        'instrument_id': ['KRX:A', 'KRX:B'] * len(days),
        'close': [100.0, 100.0] * len(days),
    })
    actions = pl.DataFrame({
        'instrument_id': ['KRX:A', 'KRX:B'],
        'action_id': ['split-a', 'unknown-b'],
        'action_type': ['split', 'unresolved'],
        'effective_date': [days[1], days[1]],
        'factor': [2.0, 1.0],
        'cash_amount': [0.0, 0.0],
        'available_at': [days[0], days[0]],
        'evidence_status': ['verified', 'unresolved'],
        'evidence_reason': [None, 'unsupported_merger'],
    })
    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
    )
    assert result.verified_corporate_actions['action_id'].to_list() == ['split-a']
    assert result.quarantine_sessions_by_instrument['KRX:B'] == days[1:]
    assert 'KRX:B' not in result.excluded_instruments


def test_resolve_backtest_evidence_rejects_jump_outside_calendar() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence
    from src.data.schemas import PITDataError

    first = datetime(2024, 1, 2, 9, tzinfo=UTC)
    second = datetime(2024, 1, 3, 9, tzinfo=UTC)
    outsider = datetime(2024, 5, 5, 9, tzinfo=UTC)
    daily = pl.DataFrame({
        'session': [first, outsider],
        'instrument_id': ['KRX:A', 'KRX:A'],
        'close': [100.0, 10.0],
    })
    actions = pl.DataFrame({'instrument_id': [], 'action_id': []})
    with pytest.raises(PITDataError, match='outside calendar'):
        resolve_backtest_corporate_action_evidence(
            daily_market=daily, corporate_actions=actions, calendar=SessionCalendar((first, second)),
            policy=BacktestMarketInputsPolicy(),
        )


def test_resolve_backtest_evidence_handles_empty_daily_market() -> None:
    import polars as pl

    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    session = datetime(2024, 1, 2, 9, tzinfo=UTC)
    daily = pl.DataFrame(schema={'session': pl.Datetime(time_zone='UTC'), 'instrument_id': pl.String, 'close': pl.Float64})
    actions = pl.DataFrame({'instrument_id': [], 'action_id': []})
    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar((session,)),
        policy=BacktestMarketInputsPolicy(),
    )
    assert result.excluded_instruments == frozenset()
    assert result.quarantine_sessions_by_instrument == {}


def test_resolve_lifecycle_only_covers_verified_pit_cleanup_interval() -> None:
    from datetime import datetime
    import polars as pl
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.backtest_sessions import resolve_backtest_lifecycle_evidence

    last = datetime(2016,5,18,tzinfo=KRX_TZ)
    removed = datetime(2016,5,19,tzinfo=KRX_TZ)
    lifecycle = pl.DataFrame({'instrument_id':['KRX:008020'],'event_type':['delisting'],'published_at':[datetime(2016,5,2,tzinfo=KRX_TZ)],'available_at':[datetime(2016,5,2,tzinfo=KRX_TZ)],'cleanup_start':[last],'cleanup_end':[last],'last_tradable_session':[last],'delisting_date':[removed.date()],'cash_settlement_per_share':[10200.0],'source_url':['https://kind.krx.co.kr/a'],'source_hash':['k'],'evidence_status':['verified'],'evidence_reason':['matched']})
    daily = pl.DataFrame({'session':[last],'instrument_id':['KRX:008020'],'close':[10200.0]})
    out = resolve_backtest_lifecycle_evidence(daily_market=daily,lifecycle_events=lifecycle,calendar=SessionCalendar((last,removed)),decision_time_of=lambda s: s.replace(hour=15,minute=30))
    assert out.actions_by_session[removed][0].action_type.value == 'delisting_cash_out'
    assert out.actions_by_session[removed][0].cash_amount == 10200.0


from datetime import datetime

import polars as pl
import pytest

from src.core.time import KRX_TZ, SessionCalendar
from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage  # noqa: F811
from src.data.schemas import PITDataError  # noqa: F811

def test_validate_corporate_action_coverage_exempts_only_verified_cleanup_key():
    first = datetime(2016, 5, 10, 9, tzinfo=KRX_TZ); second = datetime(2016, 5, 11, 9, tzinfo=KRX_TZ)  # noqa: E702
    frame = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:008020', 'KRX:008020'], 'close': [10000.0, 26000.0]})
    policy = BacktestMarketInputsPolicy(unexplained_price_jump_threshold=0.5)
    coverage = validate_corporate_action_coverage(daily_market=frame, corporate_actions=pl.DataFrame(), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15), policy=policy, lifecycle_cleanup_keys=frozenset({('KRX:008020', second)}))
    assert coverage.research_returns_by_key[(second, 'KRX:008020')] == pytest.approx(1.6)
    with pytest.raises(PITDataError, match='unexplained price discontinuity'):
        validate_corporate_action_coverage(daily_market=frame, corporate_actions=pl.DataFrame(), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15), policy=policy, lifecycle_cleanup_keys=frozenset())


from datetime import date, datetime  # noqa: F811

import polars as pl  # noqa: F811

from src.core.ledger import LedgerActionType
from src.core.time import KRX_TZ, SessionCalendar  # noqa: F811
from src.data.backtest_sessions import resolve_backtest_lifecycle_evidence  # noqa: F811

def test_resolve_lifecycle_evidence_never_uses_last_close_for_unpriced_delisting():
    last = datetime(2016, 5, 18, 9, tzinfo=KRX_TZ); absent = datetime(2016, 5, 19, 9, tzinfo=KRX_TZ)  # noqa: E702
    events = pl.DataFrame({'instrument_id': ['KRX:074150'], 'evidence_status': ['verified'], 'resolution_kind': ['unsettled_delisting'], 'available_at': [datetime(2016, 5, 1, 9, tzinfo=KRX_TZ)], 'last_tradable_session': [last], 'delisting_date': [date(2016, 5, 19)], 'cash_settlement_per_share': [None]})
    bars = pl.DataFrame({'session': [last], 'instrument_id': ['KRX:074150'], 'close': [9000.0]})
    coverage = resolve_backtest_lifecycle_evidence(daily_market=bars, lifecycle_events=events, calendar=SessionCalendar((last, absent)), decision_time_of=lambda value: value.replace(hour=15))
    action = coverage.actions_by_session[absent][0]
    assert action.action_type is LedgerActionType.DELISTING_UNSETTLED
    assert action.cash_amount == 0.0


def test_resolve_sector_accepts_duplicate_snapshots_with_same_sector() -> None:
    from datetime import datetime

    import polars as pl

    from src.data.backtest_sessions import _resolve_sector

    session = datetime(2016, 1, 4, 9, tzinfo=KRX_TZ)
    decision = datetime(2016, 1, 4, 15, 30, tzinfo=KRX_TZ)
    rows = pl.DataFrame(
        {
            'instrument_id': ['KRX:A', 'KRX:A'],
            'sector': ['__UNKNOWN__', '__UNKNOWN__'],
            'available_at': [session, session],
            'valid_from': [session, session],
            'valid_to': [session, session],
        }
    )
    assert _resolve_sector(security_master=rows, instrument_id='KRX:A', session=session, decision_time=decision) == '__UNKNOWN__'


def test_ledger_delisting_unsettled_records_no_cash_and_rejects_open_position():
    from datetime import datetime
    from src.core.ledger import Ledger, LedgerCorporateAction, LedgerActionType
    from src.core.time import KRX_TZ
    from src.data.schemas import PITDataError
    import pytest
    session = datetime(2016, 5, 19, 9, tzinfo=KRX_TZ)
    ledger = Ledger(ledger_id="cov", initial_cash=100.0, opened_at=datetime(2016, 5, 18, 9, tzinfo=KRX_TZ))
    action = LedgerCorporateAction(action_id="cov-unsettled", instrument_id="KRX:074150", action_type=LedgerActionType.DELISTING_UNSETTLED, effective_time=session, factor=1.0, cash_amount=0.0)
    ledger.apply_corporate_actions((action,), session_open=session, cash_in_lieu_prices={})
    assert ledger.quantity_of("KRX:074150") == 0
    from src.core.ledger import LedgerFill, LedgerSide
    ledger2 = Ledger(ledger_id="cov2", initial_cash=100000.0, opened_at=datetime(2016, 5, 18, 9, tzinfo=KRX_TZ))
    ledger2.record_fill(LedgerFill(fill_id="f1", instrument_id="KRX:074150", side=LedgerSide.BUY, quantity=1, price=9000.0, commission=0.0, tax=0.0, slippage_cost=0.0, trade_time=datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), settlement_time=datetime(2016, 5, 18, 9, tzinfo=KRX_TZ)))
    with pytest.raises(PITDataError, match="unsettled"):
        ledger2.apply_corporate_actions((action,), session_open=session, cash_in_lieu_prices={})


def test_resolve_lifecycle_exchange_emits_entitlement_and_delivery_with_pit_terms() -> None:
    from datetime import date, datetime
    import json
    import polars as pl
    from src.core.ledger import LedgerActionType
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.backtest_sessions import resolve_backtest_lifecycle_evidence

    delisting = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    delivery = datetime(2016, 11, 2, 9, tzinfo=KRX_TZ)
    allocations = json.dumps([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': '0.1907312', 'cost_basis_weight': '1'}])
    events = pl.DataFrame({'lifecycle_event_id': ['evt-003450'], 'instrument_id': ['KRX:003450'], 'source_security_id': ['KR7003450004'], 'evidence_status': ['verified'], 'resolution_kind': ['merger_or_exchange'], 'available_at': [datetime(2016, 10, 20, 9, tzinfo=KRX_TZ)], 'delisting_date': [date(2016, 11, 1)], 'successor_delivery_date': [date(2016, 11, 2)], 'successor_allocations_json': [allocations], 'cash_settlement_per_share': [None]})
    out = resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=events, calendar=SessionCalendar((delisting, delivery)), decision_time_of=lambda value: value.replace(hour=15, minute=30))
    assert out.actions_by_session[delisting][0].action_type is LedgerActionType.EXCHANGE_ENTITLEMENT
    assert out.actions_by_session[delivery][0].action_type is LedgerActionType.SUCCESSOR_DELIVERY
    assert out.actions_by_session[delivery][0].successor_allocations[0].successor_instrument_id == 'KRX:105560'


def test_resolve_lifecycle_exchange_rejects_missing_or_late_terms() -> None:
    from datetime import date, datetime
    import polars as pl
    import pytest
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.backtest_sessions import resolve_backtest_lifecycle_evidence
    from src.data.schemas import PITDataError

    session = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    events = pl.DataFrame({'lifecycle_event_id': ['evt-bad'], 'instrument_id': ['KRX:003450'], 'source_security_id': ['KR7003450004'], 'evidence_status': ['verified'], 'resolution_kind': ['merger_or_exchange'], 'available_at': [datetime(2016, 11, 1, 16, tzinfo=KRX_TZ)], 'delisting_date': [date(2016, 11, 1)], 'successor_delivery_date': [None], 'successor_allocations_json': [None]})
    with pytest.raises(PITDataError, match='lifecycle'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=events, calendar=SessionCalendar((session,)), decision_time_of=lambda value: value.replace(hour=15, minute=30))
    timely = events.with_columns(pl.lit(datetime(2016, 10, 20, 9, tzinfo=KRX_TZ)).alias('available_at'))
    missing_event = timely.with_columns(pl.lit(None).alias('lifecycle_event_id'))
    with pytest.raises(PITDataError, match='lifecycle_event_id'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=missing_event, calendar=SessionCalendar((session,)), decision_time_of=lambda value: value.replace(hour=15, minute=30))
    outside_delivery = timely.with_columns(pl.lit('evt-outside').alias('lifecycle_event_id'), pl.lit(date(2016, 11, 2)).alias('successor_delivery_date'))
    with pytest.raises(PITDataError, match='outside calendar'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=outside_delivery, calendar=SessionCalendar((session,)), decision_time_of=lambda value: value.replace(hour=15, minute=30))


def test_build_backtest_sessions_wires_successor_lifecycle_actions(monkeypatch, tmp_path) -> None:
    from datetime import datetime
    from decimal import Decimal
    import polars as pl
    import src.data.backtest_sessions as mod
    from src.core.ledger import LedgerActionType, LedgerCorporateAction, LedgerSuccessorAllocation
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.snapshot import PITSnapshotRepository
    from src.data.schemas import SilverTable

    first = datetime(2016, 10, 31, 9, tzinfo=KRX_TZ)
    second = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    third = datetime(2016, 11, 2, 9, tzinfo=KRX_TZ)
    calendar = SessionCalendar((first, second, third))
    bars = pl.DataFrame({'session': [first, second, third], 'instrument_id': ['KRX:003450'] * 3, 'open': [100.0] * 3, 'close': [100.0] * 3, 'volume': [1000.0] * 3, 'trading_value': [100000.0] * 3, 'market_cap': [1000000.0] * 3, 'available_at': [first, first, first]})
    master = pl.DataFrame({'instrument_id': ['KRX:003450'], 'sector': ['finance'], 'valid_from': [first], 'valid_to': [third], 'available_at': [first]})
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: bars}, root=tmp_path)
    allocation = LedgerSuccessorAllocation('KRX:105560', Decimal('0.5'), Decimal('1'))
    successor = LedgerCorporateAction('evt:wired', 'KRX:003450', LedgerActionType.EXCHANGE_ENTITLEMENT, second, 1.0, 0.0, lifecycle_event_id='evt', successor_allocations=(allocation,))
    resolution = mod.CorporateActionEvidenceResolution(bars, pl.DataFrame(), frozenset(), {}, {})
    monkeypatch.setattr(mod, 'resolve_backtest_corporate_action_evidence', lambda **_: resolution)
    monkeypatch.setattr(mod, 'resolve_backtest_lifecycle_evidence', lambda **_: mod.CorporateActionCoverage({second: (successor,)}, {}))
    monkeypatch.setattr(mod, 'validate_corporate_action_coverage', lambda **_: mod.CorporateActionCoverage({}, {}))
    monkeypatch.setattr(mod, '_rolling_inputs', lambda *_args: ({(s, 'KRX:003450'): 1.0 for s in (first, second)}, {(s, 'KRX:003450'): 0.1 for s in (first, second)}, {first: 0.1, second: 0.1}))
    built = mod.build_backtest_sessions(snapshot_repository=repository, calendar=calendar, start=first, end=second, decision_time_of=lambda value: value.replace(hour=15, minute=30), security_master=master, corporate_actions=pl.DataFrame(), lifecycle_events=pl.DataFrame({'instrument_id': ['KRX:003450']}))
    assert built[1].actions == (successor,)


def test_decode_successor_allocations_rejects_noncanonical_terms() -> None:
    import pytest

    from src.data.backtest_sessions import decode_successor_allocations
    from src.data.schemas import PITDataError

    bad_values = (
        (None, 'missing'),
        ('not-json', 'malformed'),
        ('[]', 'non-empty'),
        ([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': 0.5, 'cost_basis_weight': '1'}], 'Decimal string'),
        ([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': '0', 'cost_basis_weight': '1'}], 'positive finite'),
        ('[{"successor_security_id":"KR7105560007","successor_instrument_id":"KRX:105560","ratio":"bad","cost_basis_weight":"1"}]', 'malformed'),
        ([None], 'object'),
        ([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': '1', 'cost_basis_weight': '1', 'extra': 'x'}], 'unknown keys'),
        ([{'successor_security_id': '', 'successor_instrument_id': 'KRX:105560', 'ratio': '1', 'cost_basis_weight': '1'}], 'identity'),
        ([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': '', 'ratio': '1', 'cost_basis_weight': '1'}], 'identity'),
        ([{'successor_security_id': 'SAME', 'successor_instrument_id': 'SAME', 'ratio': '1', 'cost_basis_weight': '1'}], 'equal'),
        ([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': '1', 'cost_basis_weight': '1'}, {'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:000001', 'ratio': '1', 'cost_basis_weight': '1'}], 'unique'),
    )
    for raw, message in bad_values:
        with pytest.raises(PITDataError, match=message):
            decode_successor_allocations(raw=raw, lifecycle_event_id='evt-invalid')
    with pytest.raises(PITDataError, match='lifecycle_event_id'):
        decode_successor_allocations(raw='[]', lifecycle_event_id='')


def test_resolve_lifecycle_exchange_rejects_identity_delivery_pit_and_duplicates() -> None:
    from datetime import date, datetime

    import json
    import polars as pl
    import pytest

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.backtest_sessions import resolve_backtest_lifecycle_evidence
    from src.data.schemas import PITDataError

    first = datetime(2016, 11, 1, 9, tzinfo=KRX_TZ)
    second = datetime(2016, 11, 2, 9, tzinfo=KRX_TZ)
    allocations = json.dumps([{'successor_security_id': 'KR7105560007', 'successor_instrument_id': 'KRX:105560', 'ratio': '1', 'cost_basis_weight': '1'}])
    base = {'lifecycle_event_id': 'evt-identity', 'instrument_id': 'KRX:003450', 'evidence_status': 'verified', 'resolution_kind': 'merger_or_exchange', 'available_at': datetime(2016, 10, 20, 9, tzinfo=KRX_TZ), 'delisting_date': date(2016, 11, 2), 'successor_delivery_date': date(2016, 11, 2), 'successor_allocations_json': allocations, 'source_security_id': 'KR7003450004'}
    with pytest.raises(PITDataError, match='source_security_id'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=pl.DataFrame([{**base, 'source_security_id': ''}]), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30))
    late_delivery = {**base, 'lifecycle_event_id': 'evt-late', 'delisting_date': date(2016, 11, 2), 'successor_delivery_date': date(2016, 11, 1), 'available_at': datetime(2016, 11, 1, 16, tzinfo=KRX_TZ)}
    with pytest.raises(PITDataError, match='delivery is not PIT'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=pl.DataFrame([late_delivery]), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30))
    with pytest.raises(PITDataError, match='duplicate lifecycle'):
        resolve_backtest_lifecycle_evidence(daily_market=pl.DataFrame(), lifecycle_events=pl.DataFrame([{**base, 'action_id': 'source-a'}, {**base, 'action_id': 'source-b'}]), calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30))


def test_resolve_backtest_evidence_joins_jump_by_market_date() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    opens = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(4))
    closes = tuple(day.replace(hour=15, minute=30) for day in opens)
    daily = pl.DataFrame({
        'session': list(closes),
        'instrument_id': ['KRX:A'] * len(closes),
        'close': [100.0, 100.0, 49.0, 49.0],
        'shares_outstanding': [10.0, 10.0, 10.0, 10.0],
        'market_cap': [1000.0, 1000.0, 490.0, 490.0],
    })
    actions = pl.DataFrame({'instrument_id': [], 'action_id': []})
    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(opens),
        policy=BacktestMarketInputsPolicy(),
    )
    assert result.quarantine_sessions_by_instrument['KRX:A'] == opens[2:]
    assert result.eligible_daily_market['session'].to_list() == [closes[0], closes[1]]
    assert 'KRX:A' not in result.excluded_instruments


def test_resolve_backtest_evidence_keeps_slot_without_matching_bar() -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, resolve_backtest_corporate_action_evidence

    days = tuple(datetime(2024, 1, 2, 9, tzinfo=UTC) + timedelta(days=index) for index in range(4))
    bars = (days[0], days[1], days[3])
    daily = pl.DataFrame({
        'session': list(bars),
        'instrument_id': ['KRX:B'] * len(bars),
        'close': [100.0] * len(bars),
    })
    actions = pl.DataFrame({
        'instrument_id': ['KRX:B'], 'action_id': ['unknown-b'], 'action_type': ['unresolved'],
        'effective_session': [days[2]], 'factor': [1.0], 'cash_amount': [0.0],
        'available_at': [days[0]], 'share_listing_date': [None], 'share_delta': [None],
        'evidence_status': ['unresolved'], 'evidence_reason': ['unsupported_merger'],
    })
    result = resolve_backtest_corporate_action_evidence(
        daily_market=daily, corporate_actions=actions, calendar=SessionCalendar(days),
        policy=BacktestMarketInputsPolicy(),
    )
    assert result.quarantine_sessions_by_instrument['KRX:B'] == days[2:]
    assert result.eligible_daily_market['session'].to_list() == [days[0], days[1]]


def test_validate_corporate_action_coverage_ignores_out_of_window_legacy_rows() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage

    krx = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=krx)
    second = datetime(2024, 1, 3, 9, tzinfo=krx)
    daily = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 100.0], 'shares_outstanding': [10.0, 10.0], 'market_cap': [1000.0, 1000.0]})
    legacy = pl.DataFrame({
        'effective_session': [datetime(2023, 12, 1, 9, tzinfo=krx), second],
        'instrument_id': ['KRX:A', 'KRX:ZZZ'],
        'action_type': ['no_action', 'no_action'],
        'action_id': ['legacy-old', 'legacy-ghost'],
        'factor': [1.0, 1.0],
        'cash_amount': [0.0, 0.0],
        'available_at': [first, first],
    })
    coverage = validate_corporate_action_coverage(daily_market=daily, corporate_actions=legacy, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())
    assert coverage.actions_by_session == {}


def test_validate_corporate_action_coverage_stays_fail_closed_on_unparseable_legacy_row() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import BacktestMarketInputsPolicy, validate_corporate_action_coverage
    from src.data.schemas import PITDataError

    krx = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=krx)
    second = datetime(2024, 1, 3, 9, tzinfo=krx)
    daily = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 100.0], 'shares_outstanding': [10.0, 10.0], 'market_cap': [1000.0, 1000.0]})
    base = {'instrument_id': ['KRX:A'], 'action_type': ['no_action'], 'action_id': ['legacy-bad'], 'factor': [1.0], 'cash_amount': [0.0], 'available_at': [first]}
    garbage = pl.DataFrame({**base, 'effective_session': ['not-a-date']})
    with pytest.raises(PITDataError, match='legacy no_action'):
        validate_corporate_action_coverage(daily_market=daily, corporate_actions=garbage, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())
    naive = pl.DataFrame({**base, 'effective_session': [first.replace(tzinfo=None)]})
    with pytest.raises(PITDataError, match='legacy no_action'):
        validate_corporate_action_coverage(daily_market=daily, corporate_actions=naive, calendar=SessionCalendar((first, second)), decision_time_of=lambda value: value.replace(hour=15, minute=30), policy=BacktestMarketInputsPolicy())
