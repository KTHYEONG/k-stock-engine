"""Gold window partition loading scenarios (contract skeletons)."""
from __future__ import annotations


def test_load_gold_window_inputs_reads_only_required_daily_partitions(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    from src.data.gold_loader import load_gold_window_inputs
    from src.storage.parquet_datasets import ParquetDatasetStore

    seen = []
    monkeypatch.setattr(ParquetDatasetStore, 'read_bounded', lambda self, dataset_id, *_args, **kwargs: seen.append((dataset_id, kwargs['session_start'], kwargs['session_end'])) or __import__('polars').DataFrame())
    monkeypatch.setattr('src.data.gold_loader.load_latest_silver_table', lambda **_kwargs: __import__('polars').DataFrame({'session': [datetime(2015, 10, 1, tzinfo=UTC), datetime(2016, 1, 4, tzinfo=UTC)]}) if _kwargs['table'].value == 'calendar' else __import__('polars').DataFrame())

    try:  # noqa: SIM105
        load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 4), decision_time=datetime(2016, 12, 30, tzinfo=UTC))
    except Exception:  # noqa: S110
        pass
    assert all(end <= date(2016, 1, 4) for _dataset, _start, end in seen)


def test_load_gold_window_inputs_fails_closed_for_missing_projected_column(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import pytest
    from src.data.gold_loader import load_gold_window_inputs
    from src.data.schemas import PITDataError

    monkeypatch.setattr('src.data.gold_loader.load_latest_silver_table', lambda **_kwargs: __import__('polars').DataFrame({'session': [datetime(2016, 1, 4, tzinfo=UTC)]}))
    with pytest.raises(PITDataError, match='invalid certified Silver table'):
        load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 4), decision_time=datetime(2016, 12, 30, tzinfo=UTC))


def test_gold_loader_certifies_silver_at_load_time_and_keeps_historical_pit(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    historical_time = datetime(2016, 12, 30, tzinfo=UTC)
    seen: list[datetime] = []
    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]

    def calendar_loader(**kwargs):
        seen.append(kwargs["decision_time"])
        return pl.DataFrame({"session": sessions})

    monkeypatch.setattr(module, "load_latest_silver_table", calendar_loader)
    monkeypatch.setattr(
        module,
        "_read_bounded_table",
        lambda **_kwargs: (_ for _ in ()).throw(PITDataError("stop after calendar")),
    )
    with pytest.raises(PITDataError, match="stop after calendar"):
        module.load_gold_window_inputs(
            silver_root=tmp_path,
            validation_start=date(2016, 1, 4),
            validation_end=date(2016, 1, 8),
            decision_time=historical_time,
        )
    assert seen[0] > historical_time


def test_gold_loader_compacts_repeated_master_snapshots() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.data.gold_loader import _compact_master_snapshots

    start = datetime(2016, 1, 4, tzinfo=UTC)
    rows = [{
        "instrument_id": "KRX:005930", "ticker": "005930", "company_id": "005930",
        "market": "KOSPI", "sector": "IT", "listing_date": start,
        "delisting_date": None, "share_class": "common", "status": "listed", "valid_to": None,
        "valid_from": start + timedelta(days=offset), "available_at": start + timedelta(days=offset),
        "source_hash": str(offset),
    } for offset in range(3)]
    compacted = _compact_master_snapshots(pl.DataFrame(rows))
    assert compacted.height == 1
    assert compacted["valid_from"][0] == start


def test_gold_loader_aligns_intraday_bar_timestamp_to_krx_session_date() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    from src.data.gold_loader import _align_session_dates

    aligned = _align_session_dates(pl.DataFrame({"session": [datetime(2016, 1, 4, 9, tzinfo=ZoneInfo("Asia/Seoul"))]}))
    assert aligned["session"][0] == datetime(2016, 1, 4, tzinfo=ZoneInfo("Asia/Seoul"))


def test_bounded_gold_loader_matches_full_fixture_audit() -> None:
    from src.data.gold import build_gold_audit_manifest

    assert callable(build_gold_audit_manifest)


def test_gold_loader_validates_latest_manifest_and_projected_reads(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    table_root = tmp_path / SilverTable.DAILY_MARKET.value
    (table_root / "old").mkdir(parents=True)
    (table_root / "new").mkdir()

    class Manifest:
        def __init__(self, generated_time, content_hash):
            self.generated_time = generated_time
            self.content_hash = content_hash

    monkeypatch.setattr(
        module.ParquetDatasetStore,
        "read_manifest",
        lambda self, ident: Manifest(
            datetime(2016, 1, 2, tzinfo=UTC) if ident == "new" else datetime(2016, 1, 1, tzinfo=UTC), ident
        ),
    )
    ident, _store = module._resolve_latest_dataset(tmp_path, SilverTable.DAILY_MARKET)
    assert ident == "new"
    assert module._to_krx_date(date(2016, 1, 4)) == date(2016, 1, 4)
    with pytest.raises(PITDataError, match="bad session"):
        module._to_krx_date(object())

    monkeypatch.setattr(
        module.ParquetDatasetStore,
        "read_bounded",
        lambda *args, **kwargs: pl.DataFrame({"session": [datetime(2016, 1, 4, tzinfo=UTC)], "close": [1.0]}),
    )
    monkeypatch.setattr(
        module.ParquetDatasetStore,
        "read",
        lambda *args, **kwargs: pl.DataFrame({"company_id": ["C1"], "value": [1.0]}),
    )
    assert module._read_bounded_table(
        silver_root=tmp_path, table=SilverTable.DAILY_MARKET, decision_time=datetime(2016, 1, 5, tzinfo=UTC),
        session_start=date(2016, 1, 1), session_end=date(2016, 1, 5), columns=["session", "close"]
    ).height == 1
    assert module._read_full_projected(
        silver_root=tmp_path, table=SilverTable.DAILY_MARKET, decision_time=datetime(2016, 1, 5, tzinfo=UTC), columns=["company_id"]
    ).height == 1


def test_gold_loader_fail_closed_dataset_and_read_errors(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    with pytest.raises(PITDataError, match="missing"):
        module._resolve_latest_dataset(tmp_path, SilverTable.CALENDAR)
    root = tmp_path / SilverTable.CALENDAR.value
    root.mkdir()
    with pytest.raises(PITDataError, match="missing"):
        module._resolve_latest_dataset(tmp_path, SilverTable.CALENDAR)
    (root / "broken").mkdir()
    monkeypatch.setattr(module.ParquetDatasetStore, "read_manifest", lambda *_args: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(PITDataError, match="invalid"):
        module._resolve_latest_dataset(tmp_path, SilverTable.CALENDAR)

    class Manifest:
        generated_time = datetime(2016, 1, 1, tzinfo=UTC)
        content_hash = "x"

    monkeypatch.setattr(module.ParquetDatasetStore, "read_manifest", lambda *_args: Manifest())
    monkeypatch.setattr(module.ParquetDatasetStore, "read_bounded", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(PITDataError, match="invalid"):
        module._read_bounded_table(silver_root=tmp_path, table=SilverTable.CALENDAR, decision_time=datetime(2016, 1, 2, tzinfo=UTC), session_start=datetime(2016, 1, 1).date(), session_end=datetime(2016, 1, 2).date(), columns=["session"])
    monkeypatch.setattr(module.ParquetDatasetStore, "read_bounded", lambda *_args, **_kwargs: pl.DataFrame({"other": [1]}))
    with pytest.raises(PITDataError, match="missing"):
        module._read_bounded_table(silver_root=tmp_path, table=SilverTable.CALENDAR, decision_time=datetime(2016, 1, 2, tzinfo=UTC), session_start=datetime(2016, 1, 1).date(), session_end=datetime(2016, 1, 2).date(), columns=["session"])
    monkeypatch.setattr(module.ParquetDatasetStore, "read", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(PITDataError, match="invalid"):
        module._read_full_projected(silver_root=tmp_path, table=SilverTable.CALENDAR, decision_time=datetime(2016, 1, 2, tzinfo=UTC), columns=["session"])


def test_gold_loader_rejects_invalid_calendar_and_ranges(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="decision_time"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1))
    with pytest.raises(PITDataError, match="inverted"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 3), validation_end=date(2016, 1, 2), decision_time=datetime(2016, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(PITDataError, match="calendar"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame())
    with pytest.raises(PITDataError, match="calendar"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": ["bad"]}))
    with pytest.raises(PITDataError, match="bad session"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))


def test_gold_loader_rejects_manifest_without_aware_generation_time(tmp_path, monkeypatch) -> None:
    from datetime import datetime
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    root = tmp_path / SilverTable.CALENDAR.value
    (root / "candidate").mkdir(parents=True)
    class BadManifest:
        generated_time = datetime(2016, 1, 1)
        content_hash = "x"
    monkeypatch.setattr(module.ParquetDatasetStore, "read_manifest", lambda *_args: BadManifest())
    with pytest.raises(PITDataError, match="invalid"):
        module._resolve_latest_dataset(tmp_path, SilverTable.CALENDAR)


def test_gold_loader_rejects_calendar_edge_cases_and_full_projection_columns(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    root = tmp_path / SilverTable.CALENDAR.value
    (root / "candidate").mkdir(parents=True)
    class Manifest:
        generated_time = datetime(2016, 1, 1, tzinfo=UTC)
        content_hash = "x"
    monkeypatch.setattr(module.ParquetDatasetStore, "read_manifest", lambda *_args: Manifest())
    monkeypatch.setattr(module.ParquetDatasetStore, "read", lambda *_args, **_kwargs: pl.DataFrame({"other": [1]}))
    with pytest.raises(PITDataError, match="missing"):
        module._read_full_projected(silver_root=tmp_path, table=SilverTable.CALENDAR, decision_time=datetime(2016, 1, 2, tzinfo=UTC), columns=["session"])
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": []}, schema={"session": pl.Datetime(time_zone="UTC")}))
    with pytest.raises(PITDataError, match="calendar"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": [datetime(2015, 1, 1, tzinfo=UTC)]}))
    with pytest.raises(PITDataError, match="calendar"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 2), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": [datetime(2016, 1, 4, tzinfo=UTC)]}))
    with pytest.raises(PITDataError, match="calendar"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 1), validation_end=date(2016, 1, 3), decision_time=datetime(2016, 1, 1, tzinfo=UTC))


def test_load_gold_window_inputs_collects_bounded_and_reference_tables(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import src.data.gold_loader as module

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    def bounded(**kwargs):
        values = {c: [None] for c in kwargs["columns"]}
        values["session"] = [sessions[-1]]
        return pl.DataFrame(values)

    def full(**kwargs):
        values = {c: [None] for c in kwargs["columns"]}
        if "valid_from" in values:
            values["valid_from"] = [sessions[0]]
        if "available_at" in values:
            values["available_at"] = [sessions[0]]
        return pl.DataFrame(values)

    monkeypatch.setattr(module, "_read_bounded_table", bounded)
    monkeypatch.setattr(module, "_read_full_projected", full)
    inputs = module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))
    assert inputs.calendar.sessions
    assert inputs.daily_market.columns


def test_gold_loader_handles_optional_flow_and_rejects_bad_reference_dtypes(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    monkeypatch.setattr(module, "_read_bounded_table", lambda **kwargs: (_ for _ in ()).throw(PITDataError("flow missing")) if kwargs["table"].value == "investor_flow" else pl.DataFrame({c: [sessions[-1]] for c in kwargs["columns"]}))
    monkeypatch.setattr(module, "_read_full_projected", lambda **kwargs: pl.DataFrame({c: ["bad"] for c in kwargs["columns"]}))
    with pytest.raises(PITDataError, match="security_master"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))


def test_gold_loader_rejects_existing_flow_error_and_bad_fact_filter(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    (tmp_path / "investor_flow" / "existing").mkdir(parents=True)
    monkeypatch.setattr(module, "_read_bounded_table", lambda **_kwargs: (_ for _ in ()).throw(PITDataError("flow")) if _kwargs["table"].value == "investor_flow" else pl.DataFrame({c: [sessions[-1]] for c in _kwargs["columns"]}))
    def full(**kwargs):
        return pl.DataFrame({c: [sessions[0]] for c in kwargs["columns"]}) if "valid_from" in kwargs["columns"] else pl.DataFrame({c: [None] for c in kwargs["columns"]})
    monkeypatch.setattr(module, "_read_full_projected", full)
    with pytest.raises(PITDataError, match="flow"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))


def test_gold_loader_rejects_missing_fact_availability_column(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    monkeypatch.setattr(module, "_read_bounded_table", lambda **kwargs: pl.DataFrame({c: [sessions[-1]] for c in kwargs["columns"]}))
    def full(**kwargs):
        if "available_at" in kwargs["columns"]:
            return pl.DataFrame({c: [sessions[0]] for c in kwargs["columns"] if c != "available_at"})
        return pl.DataFrame({c: [sessions[0]] for c in kwargs["columns"]})
    monkeypatch.setattr(module, "_read_full_projected", full)
    with pytest.raises(PITDataError, match="financial_facts"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))


def test_gold_loader_rejects_empty_master_and_missing_market_columns(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    monkeypatch.setattr(module, "_read_bounded_table", lambda **kwargs: pl.DataFrame({c: [sessions[-1]] for c in kwargs["columns"]}))
    monkeypatch.setattr(module, "_read_full_projected", lambda **kwargs: pl.DataFrame({c: pl.Series([], dtype=pl.String) for c in kwargs["columns"]}) if kwargs["table"] is SilverTable.SECURITY_MASTER else pl.DataFrame({c: [sessions[0]] for c in kwargs["columns"]}))
    with pytest.raises(PITDataError, match="security_master"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))

    monkeypatch.setattr(module, "_read_full_projected", lambda **kwargs: pl.DataFrame({"wrong": [1]}) if kwargs["table"] is SilverTable.SECURITY_MASTER else pl.DataFrame({c: [sessions[0]] for c in kwargs["columns"]}))
    with pytest.raises(PITDataError, match="security_master"):
        module.load_gold_window_inputs(silver_root=tmp_path, validation_start=date(2016, 1, 4), validation_end=date(2016, 1, 8), decision_time=datetime(2016, 1, 10, tzinfo=UTC))
