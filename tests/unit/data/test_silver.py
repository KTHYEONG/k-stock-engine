def test_dart_receipt_is_available_next_krx_session() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.silver import next_krx_session_open

    friday = datetime(2024, 1, 5, tzinfo=KRX_TZ)
    monday = datetime(2024, 1, 8, tzinfo=KRX_TZ)
    calendar = SessionCalendar((friday, monday))

    assert next_krx_session_open(friday, calendar) == datetime(2024, 1, 8, 9, tzinfo=KRX_TZ)


def test_certification_rejects_duplicate_daily_market_primary_key() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import complete_minimal_fixture, validate_table

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, _ = complete_minimal_fixture(decision_time=decision)
    duplicate = pl.concat([tables[SilverTable.DAILY_MARKET], tables[SilverTable.DAILY_MARKET]])
    with pytest.raises(PITDataError, match=r'duplicate.*daily_market'):
        validate_table(SilverTable.DAILY_MARKET, duplicate, decision_time=decision)


def test_certification_rejects_fact_available_after_decision_time() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import complete_minimal_fixture, validate_table

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, _ = complete_minimal_fixture(decision_time=decision)
    late = tables[SilverTable.DAILY_MARKET].with_columns(pl.lit(datetime(2024, 1, 4, tzinfo=UTC)).alias('available_at'))
    with pytest.raises(PITDataError, match='available_at'):
        validate_table(SilverTable.DAILY_MARKET, late, decision_time=decision)


def test_certification_rejects_incomplete_required_evidence() -> None:
    from datetime import date

    import pytest

    from src.core.datasets import DatasetCertification
    from src.data.schemas import PITDataError
    from src.data.silver import certify_silver

    with pytest.raises(PITDataError, match=r'investor_flow.*financial_facts'):
        certify_silver({}, receipts={}, coverage_start=date(2024, 1, 2), coverage_end=date(2024, 1, 2), certification=DatasetCertification.RESEARCH)


def test_load_latest_silver_table_uses_manifest_time_not_directory_name(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import polars as pl
    from src.data.schemas import SilverTable
    from src.data.silver import load_latest_silver_table
    from src.storage.parquet_datasets import ParquetDatasetStore

    root = tmp_path / 'financial_facts'; (root / 'zzz').mkdir(parents=True); (root / 'aaa').mkdir()  # noqa: E702
    class Manifest: 
        def __init__(self, at): self.generated_time = at
    monkeypatch.setattr(ParquetDatasetStore, 'read_manifest', lambda self, ident: Manifest(datetime(2016, 2, 1, tzinfo=UTC) if ident == 'aaa' else datetime(2016, 1, 1, tzinfo=UTC)))
    monkeypatch.setattr(ParquetDatasetStore, 'read', lambda self, ident, *_args: pl.DataFrame({'id': [ident]}))

    assert load_latest_silver_table(root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30, tzinfo=UTC)).item(0, 'id') == 'aaa'


def test_load_latest_silver_table_rejects_missing_table(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_latest_silver_table

    with pytest.raises(PITDataError, match="missing certified Silver table"):
        load_latest_silver_table(
            root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30, tzinfo=UTC)
        )
    with pytest.raises(PITDataError, match="decision_time must be timezone-aware"):
        load_latest_silver_table(
            root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30)
        )


def test_load_latest_silver_table_skips_invalid_manifests(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_latest_silver_table
    from src.storage.parquet_datasets import ParquetDatasetStore

    table_root = tmp_path / "financial_facts"
    (table_root / "broken").mkdir(parents=True)
    (table_root / "naive").mkdir()

    class Manifest:
        def __init__(self, at):
            self.generated_time = at

    def fake_manifest(self, ident):
        if ident == "broken":
            raise FileNotFoundError("no manifest")
        if ident == "naive":
            return Manifest(datetime(2016, 3, 1))
        return Manifest(datetime(2016, 2, 1, tzinfo=UTC))

    monkeypatch.setattr(ParquetDatasetStore, "read_manifest", fake_manifest)

    with pytest.raises(PITDataError, match="missing certified Silver table"):
        load_latest_silver_table(
            root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30, tzinfo=UTC)
        )


def test_load_latest_silver_table_breaks_ties_by_content_hash(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.data.schemas import SilverTable
    from src.data.silver import load_latest_silver_table
    from src.storage.parquet_datasets import ParquetDatasetStore

    table_root = tmp_path / "financial_facts"
    (table_root / "low").mkdir(parents=True)
    (table_root / "high").mkdir()

    class Manifest:
        def __init__(self, at, content_hash):
            self.generated_time = at
            self.content_hash = content_hash

    moment = datetime(2016, 2, 1, tzinfo=UTC)
    monkeypatch.setattr(
        ParquetDatasetStore,
        "read_manifest",
        lambda self, ident: Manifest(moment, "a" if ident == "low" else "b"),
    )
    monkeypatch.setattr(
        ParquetDatasetStore, "read", lambda self, ident, *_args: pl.DataFrame({"id": [ident]})
    )

    frame = load_latest_silver_table(
        root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30, tzinfo=UTC)
    )
    assert frame.item(0, "id") == "high"


def test_load_latest_silver_table_rejects_unreadable_dataset(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_latest_silver_table
    from src.storage.parquet_datasets import ParquetDatasetStore

    table_root = tmp_path / "financial_facts"
    (table_root / "only").mkdir(parents=True)

    class Manifest:
        def __init__(self, at):
            self.generated_time = at

    monkeypatch.setattr(
        ParquetDatasetStore, "read_manifest", lambda self, ident: Manifest(datetime(2016, 2, 1, tzinfo=UTC))
    )

    def fake_read(self, ident, *_args):
        raise ValueError("tampered partition")

    monkeypatch.setattr(ParquetDatasetStore, "read", fake_read)

    with pytest.raises(PITDataError, match="invalid certified Silver table"):
        load_latest_silver_table(
            root=tmp_path, table=SilverTable.FINANCIAL_FACTS, decision_time=datetime(2016, 12, 30, tzinfo=UTC)
        )


def test_load_latest_silver_table_rejects_empty_or_undated_candidates(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_latest_silver_table
    from src.storage.parquet_datasets import ParquetDatasetStore

    empty_root = tmp_path / "empty" / "financial_facts"
    empty_root.mkdir(parents=True)
    with pytest.raises(PITDataError, match="missing certified Silver table"):
        load_latest_silver_table(
            root=tmp_path / "empty",
            table=SilverTable.FINANCIAL_FACTS,
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )

    undated_root = tmp_path / "undated" / "financial_facts"
    (undated_root / "v1").mkdir(parents=True)
    monkeypatch.setattr(
        ParquetDatasetStore, "read_manifest", lambda self, ident: SimpleNamespace()
    )
    with pytest.raises(PITDataError, match="missing certified Silver table"):
        load_latest_silver_table(
            root=tmp_path / "undated",
            table=SilverTable.FINANCIAL_FACTS,
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )


def _materialized_silver_root_for_by_id(tmp_path):
    from datetime import UTC, datetime

    from src.data.silver import SilverStore, complete_minimal_fixture

    decision = datetime(2026, 9, 6, tzinfo=UTC)
    tables, _receipts, report = complete_minimal_fixture(decision_time=decision)
    silver_root = tmp_path / "silver"
    SilverStore(silver_root).materialize_all(tables, report=report, decision_time=decision)
    return silver_root, decision


def test_silver_dataset_path_and_load_by_id_round_trip(tmp_path) -> None:
    from src.data.schemas import SilverTable
    from src.data.silver import load_silver_table_by_dataset_id, silver_dataset_path_by_id

    silver_root, decision = _materialized_silver_root_for_by_id(tmp_path)
    dataset_id = next(p.name for p in (silver_root / "calendar").iterdir() if p.is_dir())

    path = silver_dataset_path_by_id(
        root=silver_root, table=SilverTable.CALENDAR, dataset_id=dataset_id, decision_time=decision
    )

    assert path == silver_root / "calendar" / dataset_id
    frame = load_silver_table_by_dataset_id(
        root=silver_root, table=SilverTable.CALENDAR, dataset_id=dataset_id, decision_time=decision
    )
    assert frame.height == 1


def test_silver_by_id_rejects_invalid_and_missing(tmp_path) -> None:
    from datetime import datetime

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_silver_table_by_dataset_id, silver_dataset_path_by_id

    silver_root, decision = _materialized_silver_root_for_by_id(tmp_path)
    with pytest.raises(PITDataError, match="timezone-aware"):
        silver_dataset_path_by_id(
            root=silver_root,
            table=SilverTable.CALENDAR,
            dataset_id="x",
            decision_time=datetime(2024, 1, 2, 9),
        )
    for bad in ("a/b", "a\\b", ".", "..", "   ", "", " padded "):
        with pytest.raises(PITDataError, match="dataset id"):
            silver_dataset_path_by_id(
                root=silver_root, table=SilverTable.CALENDAR, dataset_id=bad, decision_time=decision
            )
    with pytest.raises(PITDataError, match="missing certified"):
        silver_dataset_path_by_id(
            root=silver_root,
            table=SilverTable.CALENDAR,
            dataset_id="absent-id",
            decision_time=decision,
        )
    (silver_root / "daily_market" / "empty-id").mkdir(parents=True, exist_ok=True)
    with pytest.raises(PITDataError, match="invalid certified"):
        silver_dataset_path_by_id(
            root=silver_root, table=SilverTable.DAILY_MARKET, dataset_id="empty-id", decision_time=decision
        )
    with pytest.raises(PITDataError, match="missing certified"):
        silver_dataset_path_by_id(
            root=tmp_path / "absent-root",
            table=SilverTable.CALENDAR,
            dataset_id="x",
            decision_time=decision,
        )
    import shutil as _shutil

    calendar_id = next(p.name for p in (silver_root / "calendar").iterdir() if p.is_dir())
    _shutil.copytree(silver_root / "calendar" / calendar_id, silver_root / "daily_market" / calendar_id)
    with pytest.raises(PITDataError, match="invalid certified"):
        silver_dataset_path_by_id(
            root=silver_root, table=SilverTable.DAILY_MARKET, dataset_id=calendar_id, decision_time=decision
        )
    with pytest.raises(PITDataError, match="invalid certified"):
        load_silver_table_by_dataset_id(
            root=silver_root, table=SilverTable.DAILY_MARKET, dataset_id=calendar_id, decision_time=decision
        )


def test_silver_load_by_id_rejects_tampered_partitions(tmp_path) -> None:
    import shutil

    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import load_silver_table_by_dataset_id

    from datetime import datetime as _dt

    silver_root, decision = _materialized_silver_root_for_by_id(tmp_path)
    dataset_id = next(p.name for p in (silver_root / "calendar").iterdir() if p.is_dir())
    with pytest.raises(PITDataError, match="timezone-aware"):
        load_silver_table_by_dataset_id(
            root=silver_root, table=SilverTable.CALENDAR, dataset_id=dataset_id, decision_time=_dt(2024, 1, 2, 9)
        )
    shutil.rmtree(silver_root / "calendar" / dataset_id / "partitions", ignore_errors=True)
    with pytest.raises(PITDataError, match="invalid certified"):
        load_silver_table_by_dataset_id(
            root=silver_root, table=SilverTable.CALENDAR, dataset_id=dataset_id, decision_time=decision
        )
