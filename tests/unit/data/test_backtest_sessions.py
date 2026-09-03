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
