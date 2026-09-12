def test_compact_security_master_intervals_merges_unchanged_attribute_run() -> None:
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    import datetime as dt

    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(3))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 3,
            "sector": ["Technology"] * 3,
            "status": ["listed"] * 3,
            "valid_from": list(days),
            "valid_to": list(days),
            "available_at": list(days),
            "source_hash": ["h0", "h1", "h2"],
        }
    )

    result = compact_security_master_intervals(master, sessions=days)

    assert result.height == 1
    row = result.to_dicts()[0]
    assert row["valid_from"] == days[0]
    assert row["valid_to"] == days[2]
    assert row["available_at"] == days[0]
    assert row["source_hash"] == "h0"
    assert result.columns == master.columns

def test_compact_security_master_intervals_splits_on_attribute_change() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(3))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 3,
            "sector": ["Technology"] * 3,
            "status": ["listed", "listed", "suspended"],
            "valid_from": list(days),
            "valid_to": list(days),
            "available_at": list(days),
            "source_hash": ["h", "h", "h"],
        }
    )

    result = compact_security_master_intervals(master, sessions=days).sort("valid_from")

    assert result.height == 2
    rows = result.to_dicts()
    assert (rows[0]["valid_from"], rows[0]["valid_to"], rows[0]["status"]) == (days[0], days[1], "listed")
    assert (rows[1]["valid_from"], rows[1]["valid_to"], rows[1]["status"]) == (days[2], days[2], "suspended")

def test_compact_security_master_intervals_splits_on_session_gap() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    present = [days[0], days[1], days[3]]
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 3,
            "sector": ["Technology"] * 3,
            "status": ["listed"] * 3,
            "valid_from": present,
            "valid_to": present,
            "available_at": present,
            "source_hash": ["h"] * 3,
        }
    )

    result = compact_security_master_intervals(master, sessions=days).sort("valid_from")

    assert result.height == 2
    rows = result.to_dicts()
    assert (rows[0]["valid_from"], rows[0]["valid_to"]) == (days[0], days[1])
    assert (rows[1]["valid_from"], rows[1]["valid_to"]) == (days[3], days[3])

def test_compact_security_master_intervals_isolates_late_receipt_row() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(3))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 3,
            "sector": ["Technology"] * 3,
            "status": ["listed"] * 3,
            "valid_from": list(days),
            "valid_to": list(days),
            "available_at": [days[0], days[2], days[2]],
            "source_hash": ["h"] * 3,
        }
    )

    result = compact_security_master_intervals(master, sessions=days).sort("valid_from")

    assert result.height == 3
    assert result["available_at"].to_list() == [days[0], days[2], days[2]]

def test_compact_security_master_intervals_keeps_multi_session_interval_rows_unfolded() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(4))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A", "KRX:A"],
            "sector": ["Technology", "Technology"],
            "status": ["listed", "listed"],
            "valid_from": [days[0], days[2]],
            "valid_to": [days[1], days[3]],
            "available_at": [days[0], days[2]],
            "source_hash": ["h", "h"],
        }
    )

    result = compact_security_master_intervals(master, sessions=days).sort("valid_from")

    assert result.height == 2
    assert result["valid_to"].to_list() == [days[1], days[3]]

def test_compact_security_master_intervals_separates_instruments() -> None:
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(2))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A", "KRX:B", "KRX:A", "KRX:B"],
            "sector": ["Technology", "Healthcare", "Technology", "Healthcare"],
            "status": ["listed"] * 4,
            "valid_from": [days[0], days[0], days[1], days[1]],
            "valid_to": [days[0], days[0], days[1], days[1]],
            "available_at": [days[0], days[0], days[1], days[1]],
            "source_hash": ["h"] * 4,
        }
    )

    result = compact_security_master_intervals(master, sessions=days).sort("instrument_id")

    assert result.height == 2
    assert result["instrument_id"].to_list() == ["KRX:A", "KRX:B"]
    assert result["valid_to"].to_list() == [days[1], days[1]]

def test_compact_security_master_intervals_returns_empty_frame_with_same_schema() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    schema = {
        "instrument_id": pl.String,
        "sector": pl.String,
        "valid_from": pl.Datetime(time_zone="Asia/Seoul"),
        "valid_to": pl.Datetime(time_zone="Asia/Seoul"),
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
    }
    master = pl.DataFrame(schema=schema)

    result = compact_security_master_intervals(master, sessions=(day,))

    assert result.height == 0
    assert result.columns == list(schema)

def test_compact_security_master_intervals_rejects_missing_key_column() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    master = pl.DataFrame({"instrument_id": ["KRX:A"], "valid_from": [day], "valid_to": [day]})

    with pytest.raises(PITDataError, match="available_at"):
        compact_security_master_intervals(master, sessions=(day,))

def test_compact_security_master_intervals_rejects_duplicate_primary_key() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A", "KRX:A"],
            "sector": ["Technology", "Utilities"],
            "valid_from": [day, day],
            "valid_to": [day, day],
            "available_at": [day, day],
        }
    )

    with pytest.raises(PITDataError, match="duplicate"):
        compact_security_master_intervals(master, sessions=(day,))

def test_compact_security_master_intervals_rejects_valid_from_outside_calendar() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    other = dt.datetime(2024, 1, 3, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "sector": ["Technology"],
            "valid_from": [other],
            "valid_to": [other],
            "available_at": [other],
        }
    )

    with pytest.raises(PITDataError, match="calendar"):
        compact_security_master_intervals(master, sessions=(day,))

def test_compact_security_master_intervals_rejects_naive_sessions() -> None:
    import datetime as dt

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals
    from src.data.schemas import PITDataError

    naive = dt.datetime(2024, 1, 2, 9)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "sector": ["Technology"],
            "valid_from": [naive],
            "valid_to": [naive],
            "available_at": [naive],
        }
    )

    with pytest.raises(PITDataError, match="timezone-aware"):
        compact_security_master_intervals(master, sessions=(naive,))

def test_compact_security_master_intervals_rejects_unordered_sessions() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    first = dt.datetime(2024, 1, 2, 9, tzinfo=kst)
    second = dt.datetime(2024, 1, 3, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "sector": ["Technology"],
            "valid_from": [first],
            "valid_to": [first],
            "available_at": [first],
        }
    )

    with pytest.raises(PITDataError, match="strictly increasing"):
        compact_security_master_intervals(master, sessions=(second, first))

def test_compact_security_master_intervals_preserves_eligibility_for_every_session() -> None:
    """압축 전후로 (session, instrument) 적격 판정 집합이 완전히 동일해야 한다."""
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2024, 1, 2, 9, tzinfo=kst) + timedelta(days=i) for i in range(6))
    present = [days[0], days[1], days[2], days[4], days[5]]
    statuses = ["listed", "listed", "suspended", "listed", "listed"]
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 5,
            "sector": ["Technology"] * 5,
            "status": statuses,
            "valid_from": present,
            "valid_to": present,
            "available_at": present,
            "source_hash": ["h"] * 5,
        }
    )

    def eligible(frame: pl.DataFrame) -> set[tuple[object, str]]:
        found: set[tuple[object, str]] = set()
        for row in frame.to_dicts():
            for session in days:
                decision = session.replace(hour=15, minute=30)
                if row["available_at"] <= decision and row["valid_from"] <= session <= row["valid_to"]:
                    found.add((session, row["status"]))
        return found

    result = compact_security_master_intervals(master, sessions=days)

    assert eligible(result) == eligible(master)
    assert result.height < master.height



def test_resolve_sentinel_master_conflicts_drops_dominated_placeholder_row() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import resolve_sentinel_master_conflicts

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2016, 1, 4, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:000670", "KRX:000670"],
            "market": ["__UNKNOWN__", "KOSPI"],
            "status": ["__UNKNOWN__", "listed"],
            "sector": ["__UNKNOWN__", "__UNKNOWN__"],
            "valid_from": [day, day],
            "valid_to": [day, day],
            "available_at": [day, day],
            "source_hash": ["placeholder", "krx"],
        }
    )

    result = resolve_sentinel_master_conflicts(master)

    assert result.height == 1
    assert result.columns == master.columns
    assert result.to_dicts()[0]["market"] == "KOSPI"
    assert result.to_dicts()[0]["source_hash"] == "krx"


def test_resolve_sentinel_master_conflicts_keeps_row_with_null_sentinel_columns() -> None:
    """market/status/sector 가 null 인 정상 행이 null==null 전파로 조용히 삭제되지 않는다."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import resolve_sentinel_master_conflicts

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2016, 1, 4, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:005930"],
            "market": [None],
            "status": [None],
            "sector": [None],
            "valid_from": [day],
            "valid_to": [day],
            "available_at": [day],
            "source_hash": ["krx"],
        }
    )

    result = resolve_sentinel_master_conflicts(master)

    assert result.height == 1
    assert result.to_dicts()[0]["source_hash"] == "krx"


def test_resolve_sentinel_master_conflicts_keeps_equal_evidence_tie_for_primary_key_gate() -> None:
    """센티넬 밀도가 같은 상충은 진짜 모호성이므로 행 순서로 임의 해소하지 않는다."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.data.master_intervals import compact_security_master_intervals, resolve_sentinel_master_conflicts
    from src.data.schemas import PITDataError

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2016, 1, 4, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A", "KRX:A"],
            "market": ["KOSPI", "KOSDAQ"],
            "status": ["listed", "listed"],
            "sector": ["Technology", "Utilities"],
            "valid_from": [day, day],
            "valid_to": [day, day],
            "available_at": [day, day],
        }
    )

    assert resolve_sentinel_master_conflicts(master).height == 2

    with pytest.raises(PITDataError, match="duplicate"):
        compact_security_master_intervals(master, sessions=(day,))


def test_resolve_sentinel_master_conflicts_preserves_frame_without_sentinel_columns() -> None:
    import datetime as dt
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import resolve_sentinel_master_conflicts

    kst = ZoneInfo("Asia/Seoul")
    day = dt.datetime(2016, 1, 4, 9, tzinfo=kst)
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"],
            "valid_from": [day],
            "valid_to": [day],
            "available_at": [day],
        }
    )

    result = resolve_sentinel_master_conflicts(master)

    assert result.height == 1
    assert result.columns == master.columns


def test_resolve_sentinel_master_conflicts_returns_empty_frame_unchanged() -> None:
    import polars as pl

    from src.data.master_intervals import resolve_sentinel_master_conflicts

    schema = {
        "instrument_id": pl.String,
        "market": pl.String,
        "status": pl.String,
        "sector": pl.String,
        "valid_from": pl.Datetime(time_zone="Asia/Seoul"),
        "valid_to": pl.Datetime(time_zone="Asia/Seoul"),
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
    }
    master = pl.DataFrame(schema=schema)

    result = resolve_sentinel_master_conflicts(master)

    assert result.height == 0
    assert result.columns == list(schema)


def test_compact_security_master_intervals_folds_across_resolved_sentinel_conflict() -> None:
    """센티넬 상충이 해소된 뒤 실측 행들이 하나의 구간으로 접혀야 한다."""
    import datetime as dt
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.data.master_intervals import compact_security_master_intervals

    kst = ZoneInfo("Asia/Seoul")
    days = tuple(dt.datetime(2016, 1, 4, 9, tzinfo=kst) + timedelta(days=i) for i in range(3))
    master = pl.DataFrame(
        {
            "instrument_id": ["KRX:A"] * 4,
            "market": ["__UNKNOWN__", "KOSPI", "KOSPI", "KOSPI"],
            "status": ["__UNKNOWN__", "listed", "listed", "listed"],
            "sector": ["__UNKNOWN__", "Technology", "Technology", "Technology"],
            "valid_from": [days[0], days[0], days[1], days[2]],
            "valid_to": [days[0], days[0], days[1], days[2]],
            "available_at": [days[0], days[0], days[1], days[2]],
            "source_hash": ["placeholder", "krx0", "krx1", "krx2"],
        }
    )

    result = compact_security_master_intervals(master, sessions=days)

    assert result.height == 1
    row = result.to_dicts()[0]
    assert (row["valid_from"], row["valid_to"], row["market"]) == (days[0], days[2], "KOSPI")
    assert row["source_hash"] == "krx0"
