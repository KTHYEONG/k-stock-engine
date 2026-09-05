def test_refresh_dart_facts_rejects_tampered_receipt_before_publish(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    receipt_dir = tmp_path / 'bronze' / 'financial_facts' / 'bad'
    receipt_dir.mkdir(parents=True)
    (receipt_dir / 'payload.json').write_text('{"records": []}', encoding='utf-8')
    (receipt_dir / 'receipt.json').write_text('{"kind": "financial_facts", "content_hash": "0" * 64, "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}', encoding='utf-8')

    with pytest.raises(PITDataError, match='hash mismatch'):
        refresh_dart_financial_facts(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts', decision_time=datetime(2016, 12, 30, tzinfo=UTC))
    assert not (tmp_path / 'silver' / 'financial_facts').exists()


def test_refresh_dart_facts_keeps_all_filings_and_excludes_later_rows() -> None:
    from datetime import UTC, datetime
    from src.data.normalization import normalize_dart_financial_facts
    from src.core.time import SessionCalendar

    rows = normalize_dart_financial_facts(pages=[{'records': [{'ticker': '005930', 'corp_code': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'A', 'fact': 'sales', 'published_at': '2015-11-16T00:00:00+00:00', 'value': 1.0, 'unit': 'KRW'}, {'ticker': '005930', 'corp_code': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'B', 'fact': 'sales', 'published_at': '2017-01-01T00:00:00+00:00', 'value': 2.0, 'unit': 'KRW'}]}], disclosure_rows=(), source_hash='a' * 64, calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)), decision_time=datetime(2016, 12, 30, tzinfo=UTC))

    assert rows['filing_id'].to_list() == ['A']


def _write_fact_receipt(bronze_root, name, payload_text, retrieved="2016-01-01T00:00:00+00:00"):
    import hashlib
    import json

    raw = payload_text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    receipt_dir = bronze_root / "financial_facts" / name
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_bytes(raw)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "financial_facts",
                "content_hash": digest,
                "source_path": name,
                "retrieved_at": retrieved,
                "ingested_at": retrieved,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return digest


def _write_reference_silver(silver_root, decision_time):
    from datetime import datetime

    import polars as pl

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    session = datetime(2015, 11, 17, tzinfo=decision_time.tzinfo)
    published = datetime(2015, 11, 16, tzinfo=decision_time.tzinfo)
    tables = {
        "calendar": pl.DataFrame(
            {"session": [session], "available_at": [session], "source_hash": ["r"]}
        ),
        "disclosures": pl.DataFrame(
            {
                "company_id": ["005930"],
                "filing_id": ["F1"],
                "filing_type": ["annual"],
                "published_at": [published],
                "available_at": [session],
                "correction_of": [None],
                "source_hash": ["r"],
            }
        ),
    }
    for table, frame in tables.items():
        content_hash = canonical_content_hash(frame, frame.columns)
        manifest = make_manifest(
            asset_kind=AssetKind.STOCK,
            columns=frame.columns,
            feature_set=f"stock_pit_{table}_v1",
            label_definition="none",
            label_horizon_sessions=1,
            time_start=session,
            time_end=session,
            provider_version="t",
            universe_policy_version="v1",
            row_count=frame.height,
            schema_version="v2",
            content_hash=content_hash,
            storage_layout=HIVE_PARTITION_LAYOUT,
            certification=DatasetCertification.RESEARCH,
        )
        ParquetDatasetStore(silver_root / table).write_partitioned(
            frame,
            dataset_id=content_hash,
            manifest=manifest,
            expected_feature_set=f"stock_pit_{table}_v1",
            decision_time=decision_time,
            content_manifest={},
        )


_FACT_PAGE = '{"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3", "filing_id": "F1", "fact": "sales", "published_at": "2015-11-16T00:00:00+00:00", "value": 10.0, "unit": "KRW"}]}'


def test_refresh_publishes_new_facts_and_returns_artifact(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        batch_size=500,
    )

    assert artifact.row_count == 1
    assert len(artifact.receipt_hashes) == 1
    assert artifact.output_hash
    assert artifact.report_hash
    assert (tmp_path / "silver" / "financial_facts" / artifact.output_hash).exists()
    assert (tmp_path / "artifacts" / f"dart_fact_refresh_{artifact.output_hash}.json").exists()
    published = pl.read_parquet(
        tmp_path / "silver" / "financial_facts" / artifact.output_hash / "partitions"
    )
    assert published["value"].dtype == pl.Float64
    assert published.item(0, "filing_id") == "F1"


def test_refresh_is_idempotent_for_same_receipts(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)
    kwargs = {
        "bronze_root": tmp_path / "bronze",
        "silver_root": tmp_path / "silver",
        "artifact_root": tmp_path / "artifacts",
        "decision_time": decision_time,
    }

    first = refresh_dart_financial_facts(**kwargs)
    second = refresh_dart_financial_facts(**kwargs, batch_size=1)

    assert second.output_hash == first.output_hash
    assert second.prior_dataset_hash == first.output_hash


def test_refresh_rejects_conflicting_duplicate_payloads(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "a", _FACT_PAGE)
    _write_fact_receipt(
        tmp_path / "bronze",
        "b",
        _FACT_PAGE.replace('"value": 10.0', '"value": 99.0'),
        retrieved="2016-02-01T00:00:00+00:00",
    )
    _write_reference_silver(tmp_path / "silver", decision_time)

    with pytest.raises(PITDataError, match="conflicting"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_rejects_missing_payload_before_publish(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    receipt_dir = tmp_path / "bronze" / "financial_facts" / "ghost"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "receipt.json").write_text(
        '{"kind": "financial_facts", "content_hash": "a", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )

    with pytest.raises(PITDataError, match="missing Bronze payload"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_rejects_invalid_batch_size_and_naive_decision_time(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)

    with pytest.raises(PITDataError, match="batch_size"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
            batch_size=0,
        )
    with pytest.raises(PITDataError, match="timezone-aware"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30),
        )


def test_refresh_rejects_empty_bronze_scope(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    with pytest.raises(PITDataError, match="financial_facts"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "missing",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )
    (tmp_path / "bronze" / "financial_facts").mkdir(parents=True)
    with pytest.raises(PITDataError, match="financial_facts"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )


def test_refresh_rejects_unparseable_payload_with_valid_hash(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    _write_fact_receipt(tmp_path / "bronze", "broken", "not json at all {{")

    with pytest.raises(PITDataError, match="invalid Bronze payload"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_rejects_malformed_receipt_metadata(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    cases = {
        "not_dict": ("[1, 2]", "malformed Bronze receipt"),
        "kind_mismatch": (
            '{"kind": "daily_market", "content_hash": "a", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
            "kind mismatch",
        ),
        "bad_time": (
            '{"kind": "financial_facts", "content_hash": "a", "retrieved_at": "nope", "ingested_at": "2016-01-01T00:00:00+00:00"}',
            "malformed Bronze receipt",
        ),
        "empty_hash": (
            '{"kind": "financial_facts", "content_hash": "", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
            "malformed Bronze receipt",
        ),
    }
    for name, (receipt_text, match) in cases.items():
        scope = tmp_path / name
        receipt_dir = scope / "bronze" / "financial_facts" / "r"
        receipt_dir.mkdir(parents=True)
        (receipt_dir / "payload.json").write_text("{}", encoding="utf-8")
        (receipt_dir / "receipt.json").write_text(receipt_text, encoding="utf-8")
        with pytest.raises(PITDataError, match=match):
            refresh_dart_financial_facts(
                bronze_root=scope / "bronze",
                silver_root=scope / "silver",
                artifact_root=scope / "artifacts",
                decision_time=decision_time,
            )
        assert not (scope / "silver" / "financial_facts").exists()


def test_refresh_bootstraps_without_prior_silver(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
    )

    assert artifact.row_count == 1
    assert artifact.prior_dataset_hash == ""


def test_refresh_rejects_empty_result_when_no_rows(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "empty", '{"records": []}')

    with pytest.raises(PITDataError, match="empty"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_rejects_dataset_beyond_decision_time(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    future = datetime(2017, 6, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {"session": [future], "available_at": [future], "source_hash": ["r"]}
    )
    content_hash = canonical_content_hash(frame, frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="stock_pit_calendar_v1",
        label_definition="none",
        label_horizon_sessions=1,
        time_start=datetime(2015, 11, 17, tzinfo=UTC),
        time_end=datetime(2015, 11, 17, tzinfo=UTC),
        provider_version="t",
        universe_policy_version="v1",
        row_count=frame.height,
        schema_version="v2",
        content_hash=content_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
        certification=DatasetCertification.RESEARCH,
    )
    ParquetDatasetStore(tmp_path / "silver" / "calendar").write_partitioned(
        frame,
        dataset_id=content_hash,
        manifest=manifest,
        expected_feature_set="stock_pit_calendar_v1",
        decision_time=decision_time,
        content_manifest={},
    )

    with pytest.raises(PITDataError, match="not available at decision_time"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )


def test_refresh_rejects_unreadable_or_garbage_receipt(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    unreadable = tmp_path / "unreadable"
    receipt_dir = unreadable / "bronze" / "financial_facts" / "r"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text("{}", encoding="utf-8")
    (receipt_dir / "receipt.json").mkdir()
    with pytest.raises(PITDataError, match="malformed Bronze receipt"):
        refresh_dart_financial_facts(
            bronze_root=unreadable / "bronze",
            silver_root=unreadable / "silver",
            artifact_root=unreadable / "artifacts",
            decision_time=decision_time,
        )
    garbage = tmp_path / "garbage"
    receipt_dir = garbage / "bronze" / "financial_facts" / "r"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text("{}", encoding="utf-8")
    (receipt_dir / "receipt.json").write_text("{{{ not json", encoding="utf-8")
    with pytest.raises(PITDataError, match="hash mismatch"):
        refresh_dart_financial_facts(
            bronze_root=garbage / "bronze",
            silver_root=garbage / "silver",
            artifact_root=garbage / "artifacts",
            decision_time=decision_time,
        )


def test_refresh_rejects_receipt_with_bad_ingested_at(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    receipt_dir = tmp_path / "bronze" / "financial_facts" / "r"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text("{}", encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(
        '{"kind": "financial_facts", "content_hash": "a", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "nope"}',
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="malformed Bronze receipt"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )


def test_refresh_maps_publish_failures_to_pit_error(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError
    from src.storage.parquet_datasets import ParquetDatasetStore

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    def fake_write(self, *args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(ParquetDatasetStore, "write_partitioned", fake_write)
    with pytest.raises(PITDataError, match="boom"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
        )


def test_merge_replaces_unbridged_legacy_rows_for_refreshed_filings() -> None:
    import polars as pl

    from src.data.incremental_normalization import _merge_fact_frames

    common = {
        "fiscal_period": "2015Q3", "fact": "sales", "restatement_id": "r0",
        "value": 1.0, "unit": "KRW", "consolidated": True,
        "source_kind": "legacy_document", "mapping_version": "v1", "raw_document_hash": "h",
    }
    existing = pl.DataFrame([{
        **common, "company_id": "00126380", "ticker": "", "dart_corp_code": "",
        "filing_id": "F1", "available_at": "2015-11-17T00:00:00+00:00",
    }]).with_columns(pl.col("available_at").str.to_datetime(time_zone="UTC"))
    refreshed = pl.DataFrame([{
        **common, "company_id": "005930", "ticker": "005930", "dart_corp_code": "00126380",
        "filing_id": "F1", "available_at": "2015-11-17T00:00:00+00:00",
    }]).with_columns(pl.col("available_at").str.to_datetime(time_zone="UTC"))
    merged = _merge_fact_frames(existing, refreshed)
    assert merged.select("company_id").to_series().to_list() == ["005930"]
