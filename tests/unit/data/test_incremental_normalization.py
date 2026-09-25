from pathlib import Path


def _artifact_dataset_path(artifact) -> Path:
    return Path(str(artifact.dataset_path))


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
        refresh_dart_financial_facts(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts', decision_time=datetime(2016, 12, 30, tzinfo=UTC), calendar=_covering_calendar())
    assert not (tmp_path / 'silver' / 'financial_facts').exists()


def test_refresh_dart_facts_keeps_all_filings_and_excludes_later_rows() -> None:
    from datetime import UTC, datetime
    from src.data.normalization import normalize_dart_financial_facts_with_quarantine
    from src.core.time import SessionCalendar

    rows, _ = normalize_dart_financial_facts_with_quarantine(pages=[{'records': [{'ticker': '005930', 'corp_code': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'A', 'fact': 'sales', 'published_at': '2015-11-16T00:00:00+00:00', 'value': 1.0, 'unit': 'KRW'}, {'ticker': '005930', 'corp_code': '00126380', 'fiscal_period': '2015Q3', 'filing_id': 'B', 'fact': 'sales', 'published_at': '2017-01-01T00:00:00+00:00', 'value': 2.0, 'unit': 'KRW'}]}], disclosure_rows=(), source_hash='a' * 64, calendar=SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),)), decision_time=datetime(2016, 12, 30, tzinfo=UTC))

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


def _covering_calendar():
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar

    return SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC),))


def _write_reference_silver(silver_root, decision_time):
    from datetime import datetime

    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

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
        publish_dataset(
            layer_root=silver_root,
            identity=DatasetIdentity(
                kind=table,
                layer=DatasetLayer.SILVER,
                policy_version="fixture-v1",
                inputs={},
                params={},
            ),
            partitions={"part-00000.parquet": frame},
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
        calendar=_covering_calendar(),
        batch_size=500,
    )

    assert artifact.row_count == 1
    assert len(artifact.receipt_hashes) == 1
    assert artifact.output_hash
    assert artifact.report_hash
    dataset_path = _artifact_dataset_path(artifact)
    assert dataset_path.is_dir()
    assert (tmp_path / "artifacts" / f"dart_fact_refresh_{artifact.output_hash}.json").exists()
    published = pl.read_parquet(dataset_path / "part-00000.parquet")
    assert published["value"].dtype == pl.Float64
    assert published.item(0, "filing_id") == "F1"


def test_refresh_uses_retained_dart_ticker_bridge(tmp_path) -> None:
    from datetime import UTC, datetime
    import hashlib
    import json

    import polars as pl

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)
    payload = json.dumps(
        [{"corp_code": "00126380", "ticker": "005930"}],
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    bridge_path = tmp_path / "bronze" / "dart_corp_codes" / digest / "payload.json"
    bridge_path.parent.mkdir(parents=True)
    bridge_path.write_text(payload, encoding="utf-8")

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    published = pl.read_parquet(
        _artifact_dataset_path(artifact) / "part-00000.parquet"
    )
    assert published.item(0, "mapping_version").endswith(f"+bridge:{digest}")


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
        "calendar": _covering_calendar(),
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
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
            batch_size=0,
        )
    with pytest.raises(PITDataError, match="timezone-aware"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2016, 12, 30),
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
        )
    (tmp_path / "bronze" / "financial_facts").mkdir(parents=True)
    with pytest.raises(PITDataError, match="financial_facts"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
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
                calendar=_covering_calendar(),
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
        calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_manifest_time_range_bounded_by_data(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)
    calendar = SessionCalendar(
        (
            datetime(2015, 11, 17, tzinfo=UTC),
            datetime(2017, 12, 29, tzinfo=UTC),
        )
    )

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=calendar,
    )

    dataset_dir = _artifact_dataset_path(artifact)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    published = pl.read_parquet(dataset_dir / "part-00000.parquet")
    assert datetime.fromisoformat(manifest["params"]["decision_time"]) == decision_time
    assert published["available_at"].max() <= decision_time
    assert published["available_at"].min() <= published["available_at"].max()


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
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
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
            calendar=_covering_calendar(),
        )


def test_refresh_maps_publish_failures_to_pit_error(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    def fake_write(self, *args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write)
    with pytest.raises(PITDataError, match="cannot stage financial facts partition"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
            calendar=_covering_calendar(),
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


def test_refresh_dart_facts_rejects_future_or_missing_frozen_bridge(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import load_frozen_dart_ticker_bridge
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match='ticker bridge'):
        load_frozen_dart_ticker_bridge(
            bronze_root=tmp_path,
            decision_time=datetime(2016, 12, 30, tzinfo=UTC),
        )


def test_load_frozen_dart_ticker_bridge_returns_retained_receipt_mapping(tmp_path) -> None:
    from datetime import UTC, datetime
    import hashlib
    import json

    from src.data.incremental_normalization import load_frozen_dart_ticker_bridge

    payload = json.dumps(
        [
            {'corp_code': '00126380', 'ticker': '005930'},
            {'corp_code': '00266961', 'ticker': '000660'},
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    receipt_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    payload_path = tmp_path / 'dart_corp_codes' / receipt_hash / 'payload.json'
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(payload, encoding='utf-8')

    mapping, actual_hash = load_frozen_dart_ticker_bridge(
        bronze_root=tmp_path,
        decision_time=datetime(2016, 12, 30, tzinfo=UTC),
    )

    assert mapping == {'00126380': '005930', '00266961': '000660'}
    assert actual_hash == receipt_hash


def test_refresh_rebuild_ignores_prior_rows(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "f1", _FACT_PAGE)
    _write_fact_receipt(
        tmp_path / "bronze",
        "f2",
        _FACT_PAGE.replace('"filing_id": "F1"', '"filing_id": "F2"'),
        retrieved="2016-02-01T00:00:00+00:00",
    )
    _write_reference_silver(tmp_path / "silver", decision_time)
    kwargs = {
        "bronze_root": tmp_path / "bronze",
        "silver_root": tmp_path / "silver",
        "artifact_root": tmp_path / "artifacts",
        "decision_time": decision_time,
        "calendar": _covering_calendar(),
    }
    first = refresh_dart_financial_facts(**kwargs)

    import shutil

    shutil.rmtree(tmp_path / "bronze" / "financial_facts" / "f2")
    second = refresh_dart_financial_facts(**kwargs)

    published = pl.read_parquet(
        _artifact_dataset_path(second) / "part-00000.parquet"
    )
    assert set(published["filing_id"].to_list()) == {"F1"}
    assert second.prior_dataset_hash == first.output_hash


def test_refresh_records_availability_policy_in_content_manifest(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.incremental_normalization import (
        AVAILABILITY_POLICY,
        refresh_dart_financial_facts,
    )

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    manifest = json.loads(
        (_artifact_dataset_path(artifact) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["details"]["availability_policy"] == AVAILABILITY_POLICY


def test_refresh_rows_all_lag_their_receipt(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2026, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "original", _FACT_PAGE)
    _write_fact_receipt(
        tmp_path / "bronze",
        "restated",
        '{"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3", '
        '"filing_id": "20260701000123", "rcept_no": "20260701000123", "fact": "sales", '
        '"published_at": "2015-11-16T00:00:00+00:00", "value": 11.0, "unit": "KRW"}]}',
        retrieved="2026-07-02T00:00:00+00:00",
    )
    calendar = SessionCalendar(
        (
            datetime(2015, 11, 17, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
        )
    )

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=calendar,
    )

    published = pl.read_parquet(
        _artifact_dataset_path(artifact) / "part-00000.parquet"
    )
    assert artifact.row_count == 2
    assert (published["available_at"] > published["published_at"]).all()


def test_refresh_republish_is_idempotent_with_single_dataset(tmp_path) -> None:
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
        "calendar": _covering_calendar(),
    }

    first = refresh_dart_financial_facts(**kwargs)
    second = refresh_dart_financial_facts(**kwargs)

    assert second.output_hash == first.output_hash
    dataset_path = _artifact_dataset_path(first)
    datasets = [
        path
        for path in dataset_path.parent.iterdir()
        if path.is_dir() and path.name.startswith("financial_facts_")
    ]
    assert datasets == [dataset_path]


def test_refresh_excludes_superseded_receipts_and_records_them(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "good", _FACT_PAGE.replace('"value": 10.0', '"value": 10000.0'),
                        retrieved="2016-02-01T00:00:00+00:00")
    stale = _write_fact_receipt(tmp_path / "bronze", "stale", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
        superseded_receipt_hashes=frozenset({stale}),
    )

    assert stale not in artifact.receipt_hashes
    manifest = json.loads(
        (_artifact_dataset_path(artifact) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["details"]["superseded_receipt_hashes"] == [stale]


def test_refresh_rejects_unknown_superseded_receipt(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "ok", _FACT_PAGE)
    _write_reference_silver(tmp_path / "silver", decision_time)

    with pytest.raises(PITDataError, match="superseded receipts not found"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
            calendar=_covering_calendar(),
            superseded_receipt_hashes=frozenset({"0" * 64}),
        )


def _legacy_receipt_payload(*, filing_id, published, fiscal_period="2015Q3"):
    import json

    return json.dumps(
        {
            "source_kind": "legacy_document",
            "records": [
                {
                    "ticker": "005930",
                    "corp_code": "00126380",
                    "fiscal_period": fiscal_period,
                    "filing_id": filing_id,
                    "fact": "sales",
                    "published_at": published,
                    "value": 10.0,
                    "unit": "KRW",
                }
            ],
        },
        sort_keys=True,
    )


def _standard_receipt_payload(*, filing_id, published, fiscal_period="2015Q3"):
    import json

    return json.dumps(
        {
            "source_kind": "opendart_standard",
            "records": [
                {
                    "ticker": "005930",
                    "corp_code": "00126380",
                    "fiscal_period": fiscal_period,
                    "filing_id": filing_id,
                    "fact": "sales",
                    "published_at": published,
                    "value": 10.0,
                    "unit": "KRW",
                }
            ],
        },
        sort_keys=True,
    )


def test_refresh_withholds_untrusted_rows_and_lists_quarantine(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    import polars as pl

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(tmp_path / "bronze", "leg", _legacy_receipt_payload(filing_id="F2", published="2015-11-16T00:00:00+00:00"))

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    published = pl.read_parquet(_artifact_dataset_path(artifact) / "part-00000.parquet")
    assert published["filing_id"].to_list() == ["F1"]
    assert set(published["source_kind"].to_list()) == {"opendart_standard"}
    assert artifact.quarantined_filings == 1
    quarantine = json.loads((tmp_path / "artifacts" / f"dart_fact_quarantine_{artifact.output_hash}.json").read_text(encoding="utf-8"))
    assert [entry["filing_id"] for entry in quarantine] == ["F2"]
    assert artifact.quarantine_path.endswith(f"dart_fact_quarantine_{artifact.output_hash}.json")


def test_refresh_manifest_records_quarantine_exclusion(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(tmp_path / "bronze", "leg", _legacy_receipt_payload(filing_id="F2", published="2015-11-16T00:00:00+00:00"))

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    manifest = json.loads(
        (_artifact_dataset_path(artifact) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["details"]["quarantined_filings"] == 1
    assert manifest["details"]["trusted_source_kinds"] == ["legacy_document_verified", "opendart_standard"]


def test_refresh_quarantine_deterministic_and_idempotent(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(tmp_path / "bronze", "leg", _legacy_receipt_payload(filing_id="F2", published="2015-11-16T00:00:00+00:00"))
    kwargs = {
        "bronze_root": tmp_path / "bronze",
        "silver_root": tmp_path / "silver",
        "artifact_root": tmp_path / "artifacts",
        "decision_time": decision_time,
        "calendar": _covering_calendar(),
    }
    first = refresh_dart_financial_facts(**kwargs)
    before = (tmp_path / "artifacts" / f"dart_fact_quarantine_{first.output_hash}.json").read_bytes()
    second = refresh_dart_financial_facts(**kwargs)
    after = (tmp_path / "artifacts" / f"dart_fact_quarantine_{second.output_hash}.json").read_bytes()

    assert second.output_hash == first.output_hash
    assert before == after
    dataset_path = _artifact_dataset_path(first)
    assert [p for p in dataset_path.parent.iterdir() if p.is_dir() and p.name.startswith("financial_facts_")] == [dataset_path]


def test_refresh_duplicate_legacy_receipts_collapse_to_later(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.core.time import SessionCalendar
    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    calendar = SessionCalendar((datetime(2015, 11, 17, tzinfo=UTC), datetime(2015, 11, 18, tzinfo=UTC)))
    _write_fact_receipt(tmp_path / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(tmp_path / "bronze", "leg-old", _legacy_receipt_payload(filing_id="F9", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(
        tmp_path / "bronze",
        "leg-new",
        _legacy_receipt_payload(filing_id="F9", published="2015-11-17T00:00:00+00:00"),
        retrieved="2016-02-01T00:00:00+00:00",
    )

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=calendar,
    )

    quarantine = json.loads((tmp_path / "artifacts" / f"dart_fact_quarantine_{artifact.output_hash}.json").read_text(encoding="utf-8"))
    assert [entry["filing_id"] for entry in quarantine] == ["F9"]
    assert quarantine[0]["available_at"] == "2015-11-18T00:00:00+00:00"


def test_refresh_all_untrusted_input_refused(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.incremental_normalization import refresh_dart_financial_facts
    from src.data.schemas import PITDataError

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    _write_fact_receipt(tmp_path / "bronze", "leg", _legacy_receipt_payload(filing_id="F2", published="2015-11-16T00:00:00+00:00"))

    with pytest.raises(PITDataError, match="certification blocked"):
        refresh_dart_financial_facts(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=decision_time,
            calendar=_covering_calendar(),
        )
    assert not (tmp_path / "silver" / "financial_facts").exists()


def test_refresh_output_hash_independent_of_quarantine(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    only = tmp_path / "only"
    extra = tmp_path / "extra"
    _write_fact_receipt(only / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(extra / "bronze", "std", _standard_receipt_payload(filing_id="F1", published="2015-11-16T00:00:00+00:00"))
    _write_fact_receipt(extra / "bronze", "leg", _legacy_receipt_payload(filing_id="F2", published="2015-11-16T00:00:00+00:00"))

    base = refresh_dart_financial_facts(
        bronze_root=only / "bronze",
        silver_root=only / "silver",
        artifact_root=only / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )
    extended = refresh_dart_financial_facts(
        bronze_root=extra / "bronze",
        silver_root=extra / "silver",
        artifact_root=extra / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    assert extended.output_hash == base.output_hash
    assert extended.quarantined_filings == 1


def test_refresh_does_not_quarantine_a_filing_that_also_has_trusted_rows(tmp_path) -> None:
    import json
    from datetime import UTC, datetime

    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    published = "2015-11-16T00:00:00+00:00"
    _write_fact_receipt(tmp_path / "bronze", "old-legacy", _legacy_receipt_payload(filing_id="F1", published=published), retrieved="2016-01-01T00:00:00+00:00")
    _write_fact_receipt(tmp_path / "bronze", "new-standard", _standard_receipt_payload(filing_id="F1", published=published), retrieved="2016-02-01T00:00:00+00:00")
    _write_fact_receipt(tmp_path / "bronze", "other-legacy", _legacy_receipt_payload(filing_id="F2", published=published), retrieved="2016-01-01T00:00:00+00:00")

    artifact = refresh_dart_financial_facts(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision_time,
        calendar=_covering_calendar(),
    )

    quarantine = json.loads((tmp_path / "artifacts" / f"dart_fact_quarantine_{artifact.output_hash}.json").read_text(encoding="utf-8"))
    assert [entry["filing_id"] for entry in quarantine] == ["F2"]
    assert artifact.quarantined_filings == 1


def test_reference_table_loader_skips_invalid_candidates_and_validates_bridge(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import hashlib
    import json

    import polars as pl
    import pytest

    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
    from src.data.incremental_normalization import (
        _load_reference_tables,
        _validate_fact_frame,
        load_frozen_dart_ticker_bridge,
    )
    from src.data.schemas import PITDataError

    silver = tmp_path / "silver"
    disclosure = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity("disclosures", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"filing_id": ["F1"]})},
    )
    facts = publish_dataset(
        layer_root=silver,
        identity=DatasetIdentity("financial_facts", DatasetLayer.SILVER, "fixture-v1", {}, {}),
        partitions={"part.parquet": pl.DataFrame({"value": [1]})},
    )
    bad = silver / "disclosures_0123456789abcdef"
    bad.mkdir()
    (bad / "manifest.json").write_text("not-json", encoding="utf-8")
    rows, prior_hash, disclosure_digest = _load_reference_tables(
        silver,
        datetime(2024, 1, 1, tzinfo=UTC),
        disclosures_dataset_id=bad.name,
        financial_facts_dataset_id="financial_facts_0123456789abcdef",
    )
    assert rows == []
    assert prior_hash == ""
    assert disclosure_digest.startswith("bronze:")
    rows, prior_hash, disclosure_digest = _load_reference_tables(
        silver,
        datetime(2024, 1, 1, tzinfo=UTC),
        disclosures_dataset_id=disclosure.dataset_id,
        financial_facts_dataset_id=facts.dataset_id,
    )
    assert rows == [{"filing_id": "F1"}]
    assert prior_hash
    assert disclosure_digest == disclosure.dataset_id

    decision_cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    def _bridge_case(name: str, raw: bytes, *, directory_hash: str | None = None) -> tuple[Path, Path]:
        root = tmp_path / name
        payload = root / "dart_corp_codes" / (directory_hash or hashlib.sha256(raw).hexdigest()) / "payload.json"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(raw)
        return root, payload

    invalid_root, _ = _bridge_case("bridge-invalid-hash", b"[]", directory_hash="z" * 64)
    with pytest.raises(PITDataError, match="receipt hash"):
        load_frozen_dart_ticker_bridge(bronze_root=invalid_root, decision_time=decision_cutoff)

    mismatch_root, _ = _bridge_case(
        "bridge-mismatch", b"[]", directory_hash=hashlib.sha256(b"different").hexdigest()
    )
    with pytest.raises(PITDataError, match="hash mismatch"):
        load_frozen_dart_ticker_bridge(bronze_root=mismatch_root, decision_time=decision_cutoff)

    invalid_json_root, _ = _bridge_case("bridge-invalid-json", b"not-json")
    with pytest.raises(PITDataError, match="payload is invalid"):
        load_frozen_dart_ticker_bridge(bronze_root=invalid_json_root, decision_time=decision_cutoff)

    non_list_root, _ = _bridge_case("bridge-non-list", b"{}")
    with pytest.raises(PITDataError, match="must be a list"):
        load_frozen_dart_ticker_bridge(bronze_root=non_list_root, decision_time=decision_cutoff)

    row_root, _ = _bridge_case("bridge-row", json.dumps([1]).encode())
    with pytest.raises(PITDataError, match="row must be an object"):
        load_frozen_dart_ticker_bridge(bronze_root=row_root, decision_time=decision_cutoff)

    missing_root, _ = _bridge_case("bridge-missing", json.dumps([{"corp_code": "1"}]).encode())
    with pytest.raises(PITDataError, match="lacks corp_code"):
        load_frozen_dart_ticker_bridge(bronze_root=missing_root, decision_time=decision_cutoff)

    conflict_root, _ = _bridge_case(
        "bridge-conflict", json.dumps([{"corp_code": "1", "ticker": "A"}, {"corp_code": "1", "ticker": "B"}]).encode()
    )
    with pytest.raises(PITDataError, match="multiple tickers"):
        load_frozen_dart_ticker_bridge(bronze_root=conflict_root, decision_time=decision_cutoff)

    empty_root, _ = _bridge_case("bridge-empty", b"[]")
    with pytest.raises(PITDataError, match="payload is empty"):
        load_frozen_dart_ticker_bridge(bronze_root=empty_root, decision_time=decision_cutoff)

    readable_root, readable_path = _bridge_case("bridge-unreadable", b"[]")
    original_read_bytes = Path.read_bytes

    def fail_read(path: Path) -> bytes:
        if path == readable_path:
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(PITDataError, match="unreadable"):
        load_frozen_dart_ticker_bridge(bronze_root=readable_root, decision_time=decision_cutoff)

    decision_time = datetime(2024, 1, 1, tzinfo=UTC)
    valid = {
        "company_id": ["A"], "fiscal_period": ["2024Q1"], "filing_id": ["F1"],
        "fact": ["sales"], "published_at": [decision_time], "available_at": [decision_time],
        "value": [1.0], "unit": ["KRW"], "consolidated": [True], "restatement_id": ["r0"],
    }
    for bad_frame, message in (
        (pl.DataFrame({"x": [1]}), "lacks columns"),
        (pl.DataFrame({**valid, "company_id": [None]}), "null primary"),
        (pl.concat([pl.DataFrame(valid), pl.DataFrame(valid)]), "duplicate primary"),
        (pl.DataFrame({**valid, "available_at": [decision_time.replace(year=2025)]}), "after decision_time"),
    ):
        with pytest.raises(PITDataError, match=message):
            _validate_fact_frame(bad_frame, decision_time=decision_time)
