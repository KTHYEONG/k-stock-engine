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


def test_backtest_session_builder_rejects_zero_volume_execution_bar(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import PITDataError, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    day = datetime(2024, 1, 2, tzinfo=UTC)
    next_day = datetime(2024, 1, 3, tzinfo=UTC)
    frame = pl.DataFrame({'session': [day, next_day], 'instrument_id': ['KRX:1', 'KRX:1'], 'open': [1.0, 1.0], 'high': [1.0, 1.0], 'low': [1.0, 1.0], 'close': [1.0, 1.0], 'volume': [0.0, 1.0], 'trading_value': [1.0, 1.0], 'market_cap': [1.0, 1.0], 'shares_outstanding': [1.0, 1.0], 'available_at': [day, next_day]})
    repository = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: frame}, root=tmp_path)

    with pytest.raises(PITDataError, match='non-positive execution bar'):
        build_backtest_sessions(snapshot_repository=repository, calendar=SessionCalendar((day, next_day)), start=day, end=day, decision_time_of=lambda session: session)


def test_backtest_sessions_optimized_builder_fast_instantiation(tmp_path) -> None:
    from datetime import UTC, datetime
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.backtest_sessions import build_backtest_sessions
    from src.data.schemas import SilverTable
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
    sessions = build_backtest_sessions(
        snapshot_repository=repo,
        calendar=calendar,
        start=d1,
        end=d1,
        decision_time_of=lambda s: s.replace(hour=15, minute=30),
    )
    assert len(sessions) == 1
    s = sessions[0]
    assert len(s.bars) == 2
    assert s.bars[0].instrument_id == 'KRX:000660'
    assert s.bars[1].instrument_id == 'KRX:005930'
    assert isinstance(s.market_snapshot, dict)
    assert 'mark_prices' in s.market_snapshot
    assert s.market_snapshot['mark_prices']['KRX:005930'] == 70500.0
    assert 'instruments' in s.market_snapshot
    assert 'market_volatility' in s.market_snapshot


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
