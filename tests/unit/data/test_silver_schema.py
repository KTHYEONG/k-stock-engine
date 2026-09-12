def test_canonicalize_session_keys_reanchors_midnight_calendar_session() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame({"session": [dt.datetime(2024, 1, 2, 0, 0, tzinfo=kst)], "available_at": [dt.datetime(2024, 1, 2, 15, 30, tzinfo=kst)]})

    result = canonicalize_session_keys(frame)

    assert result["session"].to_list() == [dt.datetime(2024, 1, 2, 9, 0, tzinfo=kst)]
    assert str(result.schema["session"].time_zone) == "Asia/Seoul"


def test_canonicalize_session_keys_preserves_market_date_across_utc_encoding() -> None:
    """UTC@00:00 세션키는 같은 KRX 거래일의 09:00 KST 로 재라벨링된다."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame({"session": [dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.UTC)], "instrument_id": ["KRX:A"]})

    result = canonicalize_session_keys(frame)

    assert result["session"].to_list() == [dt.datetime(2024, 1, 2, 9, 0, tzinfo=kst)]


def test_canonicalize_session_keys_leaves_availability_instants_untouched() -> None:
    """가용시각은 instant 보존 — 재앵커링 금지 (look-ahead 방지)."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    avail = dt.datetime(2024, 1, 2, 14, 0, tzinfo=dt.UTC)
    published = dt.datetime(2024, 1, 2, 18, 0, tzinfo=dt.UTC)
    frame = pl.DataFrame({"session": [dt.datetime(2024, 1, 2, 0, 0, tzinfo=kst)], "available_at": [avail], "published_at": [published]})

    result = canonicalize_session_keys(frame)

    assert result["available_at"].to_list()[0] == avail
    assert result["published_at"].to_list()[0] == published
    assert result["session"].to_list() == [dt.datetime(2024, 1, 2, 9, 0, tzinfo=kst)]


def test_canonicalize_session_keys_handles_all_declared_key_columns() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import SESSION_KEY_COLUMNS, canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    midnight = dt.datetime(2024, 1, 2, 0, 0, tzinfo=kst)
    expected = dt.datetime(2024, 1, 2, 9, 0, tzinfo=kst)
    frame = pl.DataFrame({name: [midnight] for name in sorted(SESSION_KEY_COLUMNS)})

    result = canonicalize_session_keys(frame)

    for name in SESSION_KEY_COLUMNS:
        assert result[name].to_list() == [expected], name


def test_canonicalize_session_keys_is_idempotent() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame({"session": [dt.datetime(2024, 1, 2, 0, 0, tzinfo=kst), dt.datetime(2024, 1, 3, 0, 0, tzinfo=kst)]})

    once = canonicalize_session_keys(frame)
    twice = canonicalize_session_keys(once)

    assert once["session"].to_list() == twice["session"].to_list()


def test_canonicalize_session_keys_ignores_non_temporal_and_absent_columns() -> None:
    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    frame = pl.DataFrame({"instrument_id": ["KRX:A"], "delisting_date": ["2024-01-02"], "close": [100.0]})

    result = canonicalize_session_keys(frame)

    assert result.to_dicts() == frame.to_dicts()
    assert result.columns == frame.columns


def test_canonicalize_session_keys_rejects_naive_session_key() -> None:
    import datetime as dt

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError
    from src.data.silver_schema import canonicalize_session_keys

    frame = pl.DataFrame({"session": [dt.datetime(2024, 1, 2, 0, 0)]})

    with pytest.raises(PITDataError, match="timezone-aware"):
        canonicalize_session_keys(frame)


def test_canonicalize_session_keys_returns_empty_frame_with_same_schema() -> None:
    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    schema = {"session": pl.Datetime(time_zone="UTC"), "instrument_id": pl.String}
    frame = pl.DataFrame(schema=schema)

    result = canonicalize_session_keys(frame)

    assert result.height == 0
    assert result.columns == list(schema)


def test_observe_time_semantics_reports_encoding_per_temporal_column() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import observe_time_semantics

    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame({
        "session": [dt.datetime(2024, 1, 2, 9, 0, tzinfo=kst)],
        "available_at": [dt.datetime(2024, 1, 2, 14, 0, tzinfo=dt.UTC)],
        "instrument_id": ["KRX:A"],
    })

    observations = observe_time_semantics(frame)
    by_column = {o.column: o for o in observations}

    assert set(by_column) == {"session", "available_at"}
    assert by_column["session"].time_zone == "Asia/Seoul"
    assert by_column["session"].hour_anchors == (9,)
    assert by_column["session"].canonical is True
    assert by_column["available_at"].time_zone == "UTC"
    assert by_column["available_at"].hour_anchors == (14,)
    assert by_column["available_at"].canonical is True


def test_observe_time_semantics_flags_non_canonical_session_key() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import observe_time_semantics

    # Polars stores one tz per Datetime column, so a genuinely non-canonical
    # encoding is represented as same-tz midnight sessions (the actual
    # Phase 1 finding: one Silver root anchors "session" at 00:00 KST instead
    # of the canonical 09:00 KST) rather than mixed input tzinfo, which would
    # be silently converted into the first value's tz and no longer test the
    # anchor-hour check this scenario targets.
    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame(
        {"session": [dt.datetime(2024, 1, 2, 0, 0, tzinfo=kst), dt.datetime(2024, 1, 3, 0, 0, tzinfo=kst)]}
    )

    observations = observe_time_semantics(frame)

    assert len(observations) == 1
    assert observations[0].column == "session"
    assert observations[0].time_zone == "Asia/Seoul"
    assert observations[0].canonical is False
    assert observations[0].hour_anchors == (0,)


def test_observe_time_semantics_returns_empty_for_frame_without_temporal_columns() -> None:
    import polars as pl

    from src.data.silver_schema import observe_time_semantics

    frame = pl.DataFrame({"instrument_id": ["KRX:A"], "close": [100.0]})

    assert observe_time_semantics(frame) == ()


def test_silver_session_key_column_declares_every_silver_table() -> None:
    from src.data.schemas import SilverTable
    from src.data.silver_schema import SESSION_KEY_COLUMNS, SILVER_SESSION_KEY_COLUMN

    assert set(SILVER_SESSION_KEY_COLUMN) == set(SilverTable)
    for table, column in SILVER_SESSION_KEY_COLUMN.items():
        assert column is None or column in SESSION_KEY_COLUMNS, table
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.CALENDAR] == "session"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.DAILY_MARKET] == "session"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.INVESTOR_FLOW] == "session"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.SECURITY_MASTER] == "valid_from"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.CORPORATE_ACTIONS] == "effective_date"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.HISTORICAL_COSTS] == "effective_date"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.LIFECYCLE_EVENTS] == "last_tradable_session"
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.FINANCIAL_FACTS] is None
    assert SILVER_SESSION_KEY_COLUMN[SilverTable.DISCLOSURES] is None


def test_canonical_session_constants_are_declared() -> None:
    from src.data.silver_schema import (
        AVAILABILITY_COLUMNS,
        CANONICAL_SESSION_HOUR,
        CANONICAL_SESSION_TZ,
        SESSION_KEY_COLUMNS,
    )

    assert CANONICAL_SESSION_TZ == "Asia/Seoul"
    assert CANONICAL_SESSION_HOUR == 9
    assert "session" in SESSION_KEY_COLUMNS
    assert "available_at" in AVAILABILITY_COLUMNS
    assert not (SESSION_KEY_COLUMNS & AVAILABILITY_COLUMNS)


def test_canonicalize_session_keys_ignores_non_datetime_session_key_column() -> None:
    """이름은 세션키 후보(SESSION_KEY_COLUMNS)지만 dtype 이 Datetime 이 아니면 그대로 통과한다."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.silver_schema import canonicalize_session_keys

    kst = ZoneInfo("Asia/Seoul")
    frame = pl.DataFrame(
        {"session": ["2024-01-02"], "available_at": [dt.datetime(2024, 1, 2, 15, 30, tzinfo=kst)]}
    )

    result = canonicalize_session_keys(frame)

    assert result["session"].to_list() == ["2024-01-02"]
    assert result.columns == frame.columns


def test_observe_time_semantics_skips_non_datetime_session_key_column() -> None:
    """이름은 세션키 후보지만 dtype 이 Datetime 이 아닌 컬럼은 관측 대상에서 제외된다."""
    import polars as pl

    from src.data.silver_schema import observe_time_semantics

    frame = pl.DataFrame({"session": ["2024-01-02"], "instrument_id": ["KRX:A"]})

    observations = observe_time_semantics(frame)

    assert observations == ()
