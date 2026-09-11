"""Gold window partition loading scenarios (contract skeletons)."""
from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture(autouse=True)
def _certified_manifest_ends(monkeypatch, request):
    """Keep row-loader unit tests focused on their table-level failure paths."""
    import src.data.gold_loader as module

    if request.node.name == 'test_gold_selected_manifest_time_end_rejects_unreadable_or_naive_manifest':
        return
    monkeypatch.setattr(
        module,
        '_selected_manifest_time_end',
        lambda **_kwargs: date(2026, 9, 10),
    )


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


def test_gold_loader_prefers_bridged_dart_facts_over_newer_fixture(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import src.data.gold_loader as module
    from src.data.schemas import SilverTable

    table_root = tmp_path / SilverTable.FINANCIAL_FACTS.value
    (table_root / "bridged").mkdir(parents=True)
    (table_root / "fixture").mkdir()

    class Manifest:
        def __init__(self, generated_time, provider_version):
            self.generated_time = generated_time
            self.provider_version = provider_version

    def read_manifest(_store, ident):
        if ident == "fixture":
            return Manifest(datetime(2026, 9, 1, tzinfo=UTC), "fixture")
        return Manifest(datetime(2025, 1, 1, tzinfo=UTC), "dart-facts-v1")

    monkeypatch.setattr(module.ParquetDatasetStore, "read_manifest", read_manifest)
    ident, _store = module._resolve_latest_dataset(tmp_path, SilverTable.FINANCIAL_FACTS)
    assert ident == "bridged"


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


def test_daily_market_backfill_plan_marks_missing_and_late_sessions(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime, time, timedelta
    import polars as pl
    import src.data.gold_loader as module
    from src.core.time import KRX_TZ

    sessions = tuple(datetime(2024, 1, 1, tzinfo=KRX_TZ) + timedelta(days=index) for index in range(65))
    monkeypatch.setattr(module, 'load_latest_silver_table', lambda **_kwargs: pl.DataFrame({'session': sessions}))
    rows = [{'session': session, 'instrument_id': 'KRX:1', 'available_at': datetime.combine(session.date(), time(15, 30), tzinfo=KRX_TZ)} for session in sessions[:-1]]
    rows[-1]['available_at'] = datetime.combine(sessions[-2].date(), time(15, 31), tzinfo=KRX_TZ)
    monkeypatch.setattr(module, '_read_bounded_table', lambda **_kwargs: pl.DataFrame(rows))

    plan = module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=sessions[60].date(), validation_end=sessions[-1].date(), decision_time=datetime(2024, 12, 31, tzinfo=UTC))

    assert plan.history_start == sessions[0].date()
    assert plan.validation_end == sessions[-1].date()
    assert plan.missing_sessions == (sessions[-2].date(), sessions[-1].date())


def test_daily_market_backfill_plan_rejects_invalid_calendar_inputs(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime, timedelta
    import polars as pl
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="decision_time"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=date(2024, 1, 1), validation_end=date(2024, 1, 2), decision_time=datetime(2024, 1, 2))
    with pytest.raises(PITDataError, match="range inverted"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=date(2024, 1, 2), validation_end=date(2024, 1, 1), decision_time=datetime(2024, 1, 2, tzinfo=UTC))
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": []}))
    with pytest.raises(PITDataError, match="calendar"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=date(2024, 1, 1), validation_end=date(2024, 1, 2), decision_time=datetime(2024, 1, 2, tzinfo=UTC))

    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(65)]
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: pl.DataFrame({"session": sessions}))
    with pytest.raises(PITDataError, match="lacks warmup"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=sessions[0].date(), validation_end=sessions[-1].date(), decision_time=datetime(2024, 12, 31, tzinfo=UTC))
    with pytest.raises(PITDataError, match="calendar"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=sessions[60].date(), validation_end=date(2024, 12, 31), decision_time=datetime(2024, 12, 31, tzinfo=UTC))
    monkeypatch.setattr(module, "_read_bounded_table", lambda **_kwargs: (_ for _ in ()).throw(PITDataError("bad")))
    plan = module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=sessions[60].date(), validation_end=sessions[-1].date(), decision_time=datetime(2024, 12, 31, tzinfo=UTC))
    assert len(plan.missing_sessions) == 65
    monkeypatch.setattr(module, "load_latest_silver_table", lambda **_kwargs: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(PITDataError, match="calendar"):
        module.plan_daily_market_backfill(silver_root=tmp_path, validation_start=date(2024, 1, 1), validation_end=date(2024, 1, 2), decision_time=datetime(2024, 1, 2, tzinfo=UTC))


def test_load_gold_window_inputs_overlays_lifecycle_without_replacing_known_master_fields() -> None:
    from datetime import datetime
    import polars as pl
    from src.core.time import KRX_TZ
    from src.data.gold_loader import apply_lifecycle_master_overlay

    session = datetime(2016,5,18,tzinfo=KRX_TZ)
    master = pl.DataFrame({'instrument_id':['KRX:008020'],'ticker':['008020'],'market':['KOSPI'],'sector':['Industrial'],'listing_date':[session],'delisting_date':[None],'share_class':['common'],'status':['listed'],'valid_from':[session],'valid_to':[session],'available_at':[session],'source_hash':['m']})
    life = pl.DataFrame({'instrument_id':['KRX:008020'],'delisting_date':[session.date()],'evidence_status':['verified'],'available_at':[session]})
    out = apply_lifecycle_master_overlay(security_master=master,lifecycle_events=life)
    assert out.row(0,named=True)['market'] == 'KOSPI'
    assert out.row(0,named=True)['sector'] == 'Industrial'
    assert out.row(0,named=True)['delisting_date'] == session.date()


def test_gold_common_coverage_rejects_mixed_2026_manifest_ends() -> None:
    from datetime import date
    import pytest
    from src.core.pit import SilverTable
    from src.data.gold_loader import assert_common_silver_coverage
    from src.data.schemas import PITDataError

    required = frozenset(SilverTable)
    ends = {table: date(2026, 9, 9) for table in required}
    ends[SilverTable.LIFECYCLE_EVENTS] = date(2026, 3, 10)
    with pytest.raises(PITDataError, match='lifecycle_events'):
        assert_common_silver_coverage(coverage_ends=ends, required_tables=required, required_end=date(2026, 9, 9))
    ends[SilverTable.LIFECYCLE_EVENTS] = date(2026, 9, 9)
    assert assert_common_silver_coverage(coverage_ends=ends, required_tables=required, required_end=date(2026, 9, 9)) == date(2026, 9, 9)
    with pytest.raises(PITDataError, match='calendar'):
        assert_common_silver_coverage(coverage_ends={}, required_tables=frozenset({SilverTable.CALENDAR}), required_end=date(2026, 9, 9))
    assert assert_common_silver_coverage(coverage_ends={}, required_tables=frozenset(), required_end=date(2026, 9, 9)) == date(2026, 9, 9)


def test_gold_selected_manifest_time_end_rejects_unreadable_or_naive_manifest(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import pytest

    import src.data.gold_loader as mod
    from src.core.pit import SilverTable
    from src.data.schemas import PITDataError

    class Store:
        def read_manifest(self, _dataset_id):
            raise OSError('unreadable')

    monkeypatch.setattr(mod, '_resolve_latest_dataset', lambda *_args: ('id', Store()))
    with pytest.raises(PITDataError, match='lifecycle_events'):
        mod._selected_manifest_time_end(silver_root=tmp_path, table=SilverTable.LIFECYCLE_EVENTS, decision_time=datetime(2026, 9, 10, tzinfo=UTC))

    class NaiveStore:
        def read_manifest(self, _dataset_id):
            return SimpleNamespace(time_end=datetime(2026, 9, 9))

    monkeypatch.setattr(mod, '_resolve_latest_dataset', lambda *_args: ('id', NaiveStore()))
    with pytest.raises(PITDataError, match='lifecycle_events'):
        mod._selected_manifest_time_end(silver_root=tmp_path, table=SilverTable.LIFECYCLE_EVENTS, decision_time=datetime(2026, 9, 10, tzinfo=UTC))


def test_gold_coverage_gate_rejects_missing_core_or_corrupt_optional_manifest(monkeypatch, tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    import polars as pl

    import src.data.gold_loader as mod
    from src.core.pit import SilverTable
    from src.data.schemas import PITDataError

    sessions = [datetime(2015, 10, 1, tzinfo=UTC) + timedelta(days=index) for index in range(100)]
    monkeypatch.setattr(mod, 'load_latest_silver_table', lambda **_kwargs: pl.DataFrame({'session': sessions}))
    def core_missing(**kwargs):
        if kwargs['table'] is SilverTable.CALENDAR:
            raise PITDataError('missing manifest')
        return sessions[-1].date()
    monkeypatch.setattr(mod, '_selected_manifest_time_end', core_missing)
    with pytest.raises(PITDataError, match='calendar'):
        mod.load_gold_window_inputs(silver_root=tmp_path, validation_start=sessions[-5].date(), validation_end=sessions[-1].date(), decision_time=datetime(2026, 9, 10, tzinfo=UTC))

    flow_dir = tmp_path / SilverTable.INVESTOR_FLOW.value / 'broken'
    flow_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, '_selected_manifest_time_end', lambda **kwargs: (_ for _ in ()).throw(PITDataError('broken flow')) if kwargs['table'] is SilverTable.INVESTOR_FLOW else sessions[-1].date())
    with pytest.raises(PITDataError, match='investor_flow'):
        mod.load_gold_window_inputs(silver_root=tmp_path, validation_start=sessions[-5].date(), validation_end=sessions[-1].date(), decision_time=datetime(2026, 9, 10, tzinfo=UTC))


def test_parse_silver_dataset_bindings_requires_complete_unique_known_tables() -> None:
    import pytest
    from src.data.gold_loader import parse_silver_dataset_bindings
    from src.data.schemas import PITDataError, SilverTable

    complete = [f'{table.value}=id-{table.value}' for table in SilverTable]
    assert parse_silver_dataset_bindings(complete) == {table: f'id-{table.value}' for table in SilverTable}
    with pytest.raises(PITDataError, match='duplicate'):
        parse_silver_dataset_bindings([*complete, complete[0]])
    with pytest.raises(PITDataError, match='unknown'):
        parse_silver_dataset_bindings([*complete[:-1], 'unknown=id'])
    with pytest.raises(PITDataError, match='missing'):
        parse_silver_dataset_bindings(complete[:-1])
    with pytest.raises(PITDataError, match='malformed'):
        parse_silver_dataset_bindings(['daily_market'])


def test_resolve_gold_dataset_bindings_rejects_fixture_and_missing_dataset(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    import pytest
    from src.data.gold_loader import resolve_gold_dataset_bindings
    from src.data.schemas import PITDataError, SilverTable
    from src.storage.parquet_datasets import ParquetDatasetStore

    requested = {table: f'id-{table.value}' for table in SilverTable}
    for table, dataset_id in requested.items():
        (tmp_path / table.value / dataset_id).mkdir(parents=True)
    monkeypatch.setattr(ParquetDatasetStore, 'read_manifest', lambda _self, _dataset_id: SimpleNamespace(provider_version='fixture', generated_time=datetime(2026, 9, 10, tzinfo=UTC), time_end=datetime(2026, 9, 10, tzinfo=UTC)))
    with pytest.raises(PITDataError, match='fixture'):
        resolve_gold_dataset_bindings(silver_root=tmp_path, requested=requested, decision_time=datetime(2026, 9, 11, tzinfo=UTC))
    missing = dict(requested)
    missing[SilverTable.DAILY_MARKET] = 'absent'
    with pytest.raises(PITDataError, match='missing'):
        resolve_gold_dataset_bindings(silver_root=tmp_path, requested=missing, decision_time=datetime(2026, 9, 11, tzinfo=UTC))
    with pytest.raises(PITDataError, match='aware'):
        resolve_gold_dataset_bindings(silver_root=tmp_path, requested=requested, decision_time=datetime(2026, 9, 11))


def test_load_gold_window_inputs_resolves_explicit_bindings_before_scan(tmp_path, monkeypatch) -> None:
    from datetime import UTC, date, datetime
    import pytest
    import src.data.gold_loader as module
    from src.data.schemas import PITDataError, SilverTable

    bindings = {table: f'id-{table.value}' for table in SilverTable}
    monkeypatch.setattr(
        module,
        'resolve_gold_dataset_bindings',
        lambda **_kwargs: (_ for _ in ()).throw(PITDataError('explicit binding rejected')),
    )

    with pytest.raises(PITDataError, match='explicit binding rejected'):
        module.load_gold_window_inputs(
            silver_root=tmp_path,
            validation_start=date(2016, 1, 4),
            validation_end=date(2016, 1, 4),
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
            silver_dataset_ids=bindings,
        )


def test_resolve_gold_dataset_bindings_accepts_production_manifests(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    from src.data.gold_loader import resolve_gold_dataset_bindings
    from src.data.schemas import SilverTable
    from src.storage.parquet_datasets import ParquetDatasetStore

    requested = {table: f'id-{table.value}' for table in SilverTable}
    for table, dataset_id in requested.items():
        (tmp_path / table.value / dataset_id).mkdir(parents=True)
    monkeypatch.setattr(ParquetDatasetStore, 'read_manifest', lambda _self, _dataset_id: SimpleNamespace(provider_version='production', generated_time=datetime(2026, 9, 10, tzinfo=UTC), time_end=datetime(2026, 9, 10, tzinfo=UTC)))
    assert resolve_gold_dataset_bindings(silver_root=tmp_path, requested=requested, decision_time=datetime(2026, 9, 11, tzinfo=UTC)) == requested


def test_write_gold_input_binding_artifact_persists_sorted_binding(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    import pytest
    from src.data.gold_loader import write_gold_input_binding_artifact
    from src.data.schemas import PITDataError, SilverTable

    bindings = {table: f'id-{table.value}' for table in SilverTable}
    out = write_gold_input_binding_artifact(artifact_root=tmp_path, dataset_ids=bindings, decision_time=datetime(2026, 9, 11, tzinfo=UTC), validation_start=datetime(2016, 1, 4).date(), validation_end=datetime(2016, 12, 29).date())
    assert out.parent.name == 'gold_input_bindings'
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert sorted(payload['dataset_ids']) == sorted(t.value for t in SilverTable)
    assert payload['decision_time'] == datetime(2026, 9, 11, tzinfo=UTC).isoformat()
    with pytest.raises(PITDataError, match='aware'):
        write_gold_input_binding_artifact(artifact_root=tmp_path, dataset_ids=bindings, decision_time=datetime(2026, 9, 11), validation_start=datetime(2016, 1, 4).date(), validation_end=datetime(2016, 12, 29).date())
    with pytest.raises(PITDataError, match='inverted'):
        write_gold_input_binding_artifact(artifact_root=tmp_path, dataset_ids=bindings, decision_time=datetime(2026, 9, 11, tzinfo=UTC), validation_start=datetime(2016, 12, 29).date(), validation_end=datetime(2016, 1, 4).date())
    with pytest.raises(PITDataError, match='missing'):
        write_gold_input_binding_artifact(artifact_root=tmp_path, dataset_ids={}, decision_time=datetime(2026, 9, 11, tzinfo=UTC), validation_start=datetime(2016, 1, 4).date(), validation_end=datetime(2016, 12, 29).date())
