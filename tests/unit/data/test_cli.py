import sys

from src.data.cli import _parse_args


def test_collect_command_requires_immutable_plan_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stock-data", "collect", "--plan-id", "plan-a"])

    args = _parse_args()

    assert args.command == "collect"
    assert args.plan_id == "plan-a"


def test_run_backtest_refuses_without_resolved_execution_components(tmp_path) -> None:
    from argparse import Namespace

    import pytest

    from src.data.cli import _dispatch_backtest
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="requires resolved Gold artifact"):
        _dispatch_backtest(Namespace(gold_root=tmp_path))


def _write_cli_fact_receipt(bronze_root, payload_text) -> None:
    import hashlib
    import json

    raw = payload_text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    receipt_dir = bronze_root / "financial_facts" / digest
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_bytes(raw)
    (receipt_dir / "receipt.json").write_text(
        json.dumps(
            {
                "kind": "financial_facts",
                "content_hash": digest,
                "source_path": "cli",
                "retrieved_at": "2016-01-01T00:00:00+00:00",
                "ingested_at": "2016-01-01T00:00:00+00:00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_normalize_dart_facts_command_parses_all_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            "b",
            "--silver-root",
            "s",
            "--artifact-root",
            "a",
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
            "--batch-size",
            "7",
        ],
    )

    args = _parse_args()

    assert args.command == "normalize-dart-facts"
    assert args.batch_size == 7


def test_normalize_dart_facts_dispatch_publishes(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    _write_cli_fact_receipt(
        tmp_path / "bronze",
        '{"records": [{"ticker": "005930", "corp_code": "00126380", "fiscal_period": "2015Q3", "filing_id": "F1", "fact": "sales", "published_at": "2015-11-16T00:00:00+00:00", "value": 10.0, "unit": "KRW"}]}',
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 0
    captured = capsys.readouterr()
    assert "output_hash" in captured.out
    assert (tmp_path / "silver" / "financial_facts").exists()


def test_load_silver_table_reads_latest_manifest_dataset(tmp_path) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from src.core.datasets import DatasetCertification, HIVE_PARTITION_LAYOUT, make_manifest
    from src.core.instruments import AssetKind
    from src.data.cli import _load_silver_table
    from src.data.schemas import SilverTable
    from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

    decision_time = datetime(2016, 12, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "session": [datetime(2015, 11, 17, tzinfo=UTC)],
            "available_at": [datetime(2015, 11, 17, tzinfo=UTC)],
            "source_hash": ["r"],
        }
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

    loaded = _load_silver_table(tmp_path / "silver", SilverTable.CALENDAR)

    assert loaded.height == 1


def test_normalize_dart_facts_dispatch_reports_failure(tmp_path, monkeypatch, capsys) -> None:
    from src.data import cli as cli_module

    receipt_dir = tmp_path / "bronze" / "financial_facts" / "bad"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text('{"records": []}', encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(
        '{"kind": "financial_facts", "content_hash": "f", "retrieved_at": "2016-01-01T00:00:00+00:00", "ingested_at": "2016-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stock-data",
            "normalize-dart-facts",
            "--bronze-root",
            str(tmp_path / "bronze"),
            "--silver-root",
            str(tmp_path / "silver"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--decision-time",
            "2016-12-30T00:00:00+00:00",
        ],
    )

    assert cli_module.main() == 1
    assert not (tmp_path / "silver" / "financial_facts").exists()
