from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.bronze import BronzeStore
from src.data.ordinary_universe_price_audit import audit_ordinary_universe_price_availability
from src.data.schemas import EvidenceKind, PITDataError


def _universe(root: Path) -> Path:
    dataset = root / "ordinary_universe_test"
    part = dataset / "session=2020-01-02" / "part.parquet"
    part.parent.mkdir(parents=True)
    frame = pl.DataFrame({"ticker": ["000001", "000002"], "eligible": [True, True]})
    frame.write_parquet(part)
    manifest = {"dataset_id": dataset.name, "partitions": [{"session": "2020-01-02", "path": "session=2020-01-02/part.parquet", "row_count": 2, "eligible_count": 2, "parquet_sha256": hashlib.sha256(part.read_bytes()).hexdigest()}]}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    return dataset


def _daily(store: BronzeStore, records: list[dict[str, str]], label: str = "a") -> None:
    store.import_bytes(json.dumps({"session": "2020-01-02", "records": records}).encode(), kind=EvidenceKind.DAILY_MARKET, retrieved_at=datetime(2020, 1, 2, tzinfo=UTC), source_label=label)


def _bar(ticker: str, *, volume: str = "1", close: str = "10") -> dict[str, str]:
    return {"ISU_CD": ticker, "BAS_DD": "20200102", "TDD_OPNPRC": "10", "TDD_HGPRC": "11", "TDD_LWPRC": "9", "TDD_CLSPRC": close, "ACC_TRDVOL": volume, "ACC_TRDVAL": "10"}


def test_audit_counts_tradable_zero_volume_and_missing(tmp_path) -> None:
    _universe(tmp_path / "universe")
    store = BronzeStore(tmp_path / "bronze")
    _daily(store, [_bar("000001"), _bar("000002", volume="0")])
    audit = audit_ordinary_universe_price_availability(universe_root=tmp_path / "universe", bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts")
    assert (audit.eligible_rows, audit.price_rows, audit.tradable_rows, audit.zero_volume_rows, audit.missing_price_rows) == (2, 2, 1, 1, 0)
    assert (tmp_path / "artifacts" / "ordinary-universe-price-audit" / f"{audit.dataset_id}.json").is_file()


def test_audit_rejects_duplicate_daily_ticker_and_tampered_universe(tmp_path) -> None:
    dataset = _universe(tmp_path / "universe")
    store = BronzeStore(tmp_path / "bronze")
    _daily(store, [_bar("000001"), _bar("000001")])
    with pytest.raises(PITDataError, match="duplicate daily-market ticker"):
        audit_ordinary_universe_price_availability(universe_root=tmp_path / "universe", bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts")
    (dataset / "session=2020-01-02" / "part.parquet").write_bytes(b"tampered")
    with pytest.raises(PITDataError, match="partition hash mismatch"):
        audit_ordinary_universe_price_availability(universe_root=tmp_path / "universe", bronze_root=tmp_path / "bronze", artifact_root=tmp_path / "artifacts")


def test_audit_helper_validation_rejects_malformed_source_boundaries(tmp_path) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod

    with pytest.raises(PITDataError, match="invalid daily-market session"):
        audit_mod._parse_date("not-a-date")
    with pytest.raises(PITDataError, match="exactly one"):
        audit_mod._only_universe_dataset(tmp_path / "missing")

    first = _universe(tmp_path / "universe")
    second = (tmp_path / "universe" / "ordinary_universe_other")
    second.mkdir()
    with pytest.raises(PITDataError, match="exactly one"):
        audit_mod._only_universe_dataset(tmp_path / "universe")
    second.rmdir()

    manifest = first / "manifest.json"
    manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        audit_mod._load_universe_manifest(first)
    manifest.write_text(json.dumps({"dataset_id": first.name, "partitions": [{}]}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid daily-market session"):
        audit_mod._load_universe_manifest(first)
    manifest.write_text(json.dumps({"dataset_id": "wrong", "partitions": []}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe manifest"):
        audit_mod._load_universe_manifest(first)
    manifest.write_text(json.dumps({"dataset_id": first.name, "partitions": [1]}), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid ordinary-universe partition"):
        audit_mod._load_universe_manifest(first)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": first.name,
                "partitions": [
                    {"session": "2020-01-02"},
                    {"session": "2020-01-02"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PITDataError, match="strictly ordered"):
        audit_mod._load_universe_manifest(first)


def test_audit_rejects_invalid_price_payload_states_and_reports_missing_price(tmp_path) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod

    _universe(tmp_path / "universe")
    store = BronzeStore(tmp_path / "bronze")
    _daily(store, [{"BAS_DD": "20200102"}])
    with pytest.raises(PITDataError, match="lacks ticker"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "universe",
            bronze_root=tmp_path / "bronze",
            artifact_root=tmp_path / "artifacts",
        )

    assert audit_mod._price_state(_bar("000001", close="0")) == "invalid"
    with pytest.raises(PITDataError, match="invalid daily-market numeric"):
        audit_mod._price_state({**_bar("000001"), "TDD_OPNPRC": "not-a-number"})
    with pytest.raises(PITDataError, match="invalid daily-market numeric"):
        audit_mod._price_state({**_bar("000001"), "TDD_OPNPRC": "nan"})
    with pytest.raises(PITDataError, match="missing daily-market numeric"):
        audit_mod._price_state({"TDD_OPNPRC": "1"})

    missing = audit_ordinary_universe_price_availability(
        universe_root=tmp_path / "universe",
        bronze_root=tmp_path / "empty-bronze",
        artifact_root=tmp_path / "artifacts",
    )
    assert missing.missing_price_rows == 2


def test_daily_payload_reader_rejects_bad_receipts_and_ambiguous_pages(tmp_path) -> None:
    import src.data.ordinary_universe_price_audit as audit_mod

    root = tmp_path / "bronze" / "daily_market"
    bad = root / "bad"
    bad.mkdir(parents=True)
    (bad / "payload.json").write_text("{broken", encoding="utf-8")
    (bad / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid daily-market Bronze receipt"):
        audit_mod._daily_payloads_by_session(tmp_path / "bronze", {datetime(2020, 1, 2).date()})

    (bad / "payload.json").write_text(json.dumps({"session": "2020-01-02", "records": []}), encoding="utf-8")
    (bad / "receipt.json").write_text(json.dumps({"content_hash": "0" * 64}), encoding="utf-8")
    with pytest.raises(PITDataError, match="hash mismatch"):
        audit_mod._daily_payloads_by_session(tmp_path / "bronze", {datetime(2020, 1, 2).date()})

    payload = bad / "payload.json"
    receipt = bad / "receipt.json"

    def write_page(body: object) -> None:
        raw = json.dumps(body).encode()
        payload.write_bytes(raw)
        receipt.write_text(json.dumps({"content_hash": hashlib.sha256(raw).hexdigest()}), encoding="utf-8")

    write_page([])
    with pytest.raises(PITDataError, match="invalid daily-market payload"):
        audit_mod._daily_payloads_by_session(tmp_path / "bronze", {datetime(2020, 1, 2).date()})
    write_page({"records": [{}]})
    assert audit_mod._daily_payloads_by_session(tmp_path / "bronze", {datetime(2020, 1, 2).date()}) == {}
    write_page({"session": "2020-01-02", "records": {}})
    with pytest.raises(PITDataError, match="invalid daily-market payload"):
        audit_mod._daily_payloads_by_session(tmp_path / "bronze", {datetime(2020, 1, 2).date()})


def test_audit_rejects_ambiguous_records_partition_count_and_output_collision(tmp_path) -> None:
    _universe(tmp_path / "universe")
    store = BronzeStore(tmp_path / "bronze")
    _daily(store, [_bar("000001")], label="a")
    _daily(store, [{**_bar("000001"), "extra": "other"}], label="b")
    with pytest.raises(PITDataError, match="ambiguous daily-market page"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "universe",
            bronze_root=tmp_path / "bronze",
            artifact_root=tmp_path / "artifacts",
        )

    _universe(tmp_path / "single")
    clean = BronzeStore(tmp_path / "clean-bronze")
    _daily(clean, [42])
    with pytest.raises(PITDataError, match="invalid daily-market record"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "single",
            bronze_root=tmp_path / "clean-bronze",
            artifact_root=tmp_path / "artifacts",
        )

    _universe(tmp_path / "conflict")
    conflict = BronzeStore(tmp_path / "conflict-bronze")
    _daily(conflict, [{**_bar("000001"), "BAS_DD": "20200103"}])
    with pytest.raises(PITDataError, match="conflicting daily-market session"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "conflict",
            bronze_root=tmp_path / "conflict-bronze",
            artifact_root=tmp_path / "artifacts",
        )

    _universe(tmp_path / "stable")
    stable = BronzeStore(tmp_path / "stable-bronze")
    _daily(stable, [_bar("000001")])
    audit = audit_ordinary_universe_price_availability(
        universe_root=tmp_path / "stable",
        bronze_root=tmp_path / "stable-bronze",
        artifact_root=tmp_path / "artifacts",
    )
    report = tmp_path / "artifacts" / "ordinary-universe-price-audit" / f"{audit.dataset_id}.json"
    report.write_text("tampered", encoding="utf-8")
    with pytest.raises(PITDataError, match="collision"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "stable",
            bronze_root=tmp_path / "stable-bronze",
            artifact_root=tmp_path / "artifacts",
        )

    bad_count = _universe(tmp_path / "bad-count")
    manifest = json.loads((bad_count / "manifest.json").read_text(encoding="utf-8"))
    manifest["partitions"][0]["row_count"] = 99
    (bad_count / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PITDataError, match="partition count mismatch"):
        audit_ordinary_universe_price_availability(
            universe_root=tmp_path / "bad-count",
            bronze_root=tmp_path / "empty",
            artifact_root=tmp_path / "artifacts",
        )
