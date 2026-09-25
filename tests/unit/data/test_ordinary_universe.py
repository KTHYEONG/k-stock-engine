"""Ordinary-universe v2 identity and publication tests."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.pit import BronzeReceipt, EvidenceKind
from src.data.datasets import load_manifest, verify_dataset
from src.data.ordinary_universe import ordinary_universe_snapshot, write_ordinary_universe_silver


def _snapshot(tmp_path: Path, session: date):
    payload = {
        "as_of": session.isoformat(),
        "records": [{
            "ISU_SRT_CD": "005930",
            "ISU_CD": "KR7005930003",
            "KIND_STKCERT_TP_NM": "보통주",
            "SECUGRP_NM": "주권",
            "MKT_TP_NM": "KOSPI",
            "SECT_TP_NM": "",
        }],
    }
    payload_path = tmp_path / "payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    raw = payload_path.read_bytes()
    receipt = BronzeReceipt(
        kind=EvidenceKind.SECURITY_MASTER,
        content_hash=__import__("hashlib").sha256(raw).hexdigest(),
        source_path=f"KRX:historical-master:{session.isoformat()}",
        retrieved_at=datetime(session.year, session.month, session.day, tzinfo=UTC),
        ingested_at=datetime(session.year, session.month, session.day, tzinfo=UTC),
        payload_path=payload_path,
        metadata_path=tmp_path / "receipt.json",
    )
    return ordinary_universe_snapshot(receipt)


def test_ordinary_universe_is_v2_and_identity_scoped_to_sessions(tmp_path: Path) -> None:
    first_snapshot = _snapshot(tmp_path / "first", date(2024, 1, 2))
    second_snapshot = _snapshot(tmp_path / "second", date(2024, 1, 3))
    first = write_ordinary_universe_silver([first_snapshot], root=tmp_path / "silver")
    repeated = write_ordinary_universe_silver([first_snapshot], root=tmp_path / "silver")
    second = write_ordinary_universe_silver([second_snapshot], root=tmp_path / "silver")

    assert first == repeated
    assert first.name != second.name
    assert load_manifest(first).kind == "ordinary_universe"
    assert load_manifest(first).layer.value == "silver"
    assert verify_dataset(first, known_ids=lambda _dataset_id: True).passed
    frame = pl.read_parquet(first / "session=2024-01-02/part.parquet")
    assert frame["instrument_id"].to_list() == ["KRX:005930"]


def test_ordinary_universe_loader_boundaries_and_catalog_publication(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    import pytest

    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.ordinary_universe import (
        BronzeReceipt,
        EvidenceKind,
        PITDataError,
        build_ordinary_universe,
        catalog_master_receipts,
        catalog_master_sessions,
        dated_master_receipts,
        materialize_ordinary_universe_from_catalog,
        ordinary_universe_snapshot,
        write_ordinary_universe_silver,
    )

    missing = BronzeReceipt(
        kind=EvidenceKind.SECURITY_MASTER,
        content_hash="a" * 64,
        source_path="KRX:historical-master:2024-01-02",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        ingested_at=datetime(2024, 1, 2, tzinfo=UTC),
        payload_path=tmp_path / "missing.json",
        metadata_path=tmp_path / "receipt.json",
    )
    with pytest.raises(PITDataError, match="unreadable"):
        ordinary_universe_snapshot(missing)

    bad_listing_root = tmp_path / "bad-listing"
    bad_listing_root.mkdir()
    payload = {
        "as_of": "2024-01-02",
        "records": [{
            "ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003",
            "KIND_STKCERT_TP_NM": "보통주", "SECUGRP_NM": "주권", "MKT_TP_NM": "KOSPI",
            "LIST_DD": "not-a-date",
        }],
    }
    payload_path = bad_listing_root / "payload.json"
    raw = json.dumps(payload, ensure_ascii=False).encode()
    payload_path.write_bytes(raw)
    bad_listing = BronzeReceipt(
        kind=EvidenceKind.SECURITY_MASTER,
        content_hash=hashlib.sha256(raw).hexdigest(),
        source_path="KRX:historical-master:2024-01-02",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        ingested_at=datetime(2024, 1, 2, tzinfo=UTC),
        payload_path=payload_path,
        metadata_path=bad_listing_root / "receipt.json",
    )
    with pytest.raises(PITDataError, match="listing date"):
        ordinary_universe_snapshot(bad_listing)

    receipt_dir = tmp_path / "bronze" / "security_master" / ("c" * 64)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "payload.json").write_text("{}", encoding="utf-8")
    (receipt_dir / "receipt.json").write_text(
        json.dumps({
            "source_path": "KRX:historical-master:2024-01-02",
            "kind": "security_master",
            "content_hash": "c" * 64,
            "retrieved_at": "2024-01-02T00:00:00+00:00",
            "ingested_at": "2024-01-02T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    selected = dated_master_receipts(tmp_path / "bronze", sessions=(date(2024, 1, 2),))
    assert selected[0].content_hash == "c" * 64

    with pytest.raises(PITDataError, match="dated snapshots"):
        write_ordinary_universe_silver([], root=tmp_path / "silver")
    snapshot = _snapshot(tmp_path / "ordered", date(2024, 1, 2))
    with pytest.raises(PITDataError, match="ascending"):
        write_ordinary_universe_silver([snapshot, snapshot], root=tmp_path / "silver")

    catalog_root = tmp_path / "catalog"
    catalog_bronze = tmp_path / "catalog-bronze"
    catalog_payload = catalog_bronze / "security_master" / ("d" * 64) / "payload.json"
    catalog_payload.parent.mkdir(parents=True)
    catalog_raw = json.dumps({
        "as_of": "2024-01-02",
        "records": [{
            "ISU_SRT_CD": "005930", "ISU_CD": "KR7005930003",
            "KIND_STKCERT_TP_NM": "보통주", "SECUGRP_NM": "주권", "MKT_TP_NM": "KOSPI",
        }],
    }, ensure_ascii=False).encode()
    catalog_payload.write_bytes(catalog_raw)
    catalog = ReceiptCatalog(catalog_root)
    catalog.publish([
        ReceiptIndexEntry(
            source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
            fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=hashlib.sha256(catalog_raw).hexdigest(),
            retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=catalog_payload,
        )
    ])
    assert catalog_master_sessions(catalog) == (date(2024, 1, 2),)
    assert catalog_master_receipts(catalog, sessions=(date(2024, 1, 2),))[0].payload_path == catalog_payload
    result = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=(date(2024, 1, 2),), silver_root=tmp_path / "silver-catalog"
    )
    assert result.is_dir()

    class _SingleEntryCatalog:
        def __init__(self, entry):
            self.entry = entry

        def latest(self, **_kwargs):
            return {"2024-01-02": self.entry}

    missing_entry = ReceiptIndexEntry(
        source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
        fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash="e" * 64,
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=tmp_path / "missing-payload.json",
    )
    with pytest.raises(PITDataError, match="payload is missing"):
        catalog_master_receipts(_SingleEntryCatalog(missing_entry), sessions=(date(2024, 1, 2),))

    mismatch_entry = ReceiptIndexEntry(
        source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
        fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash="e" * 64,
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=catalog_payload,
    )
    with pytest.raises(PITDataError, match="hash mismatch"):
        catalog_master_receipts(_SingleEntryCatalog(mismatch_entry), sessions=(date(2024, 1, 2),))

    original_read_bytes = Path.read_bytes

    def fail_catalog_read(path: Path) -> bytes:
        if path == catalog_payload:
            raise OSError("closed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_catalog_read)
    valid_entry = ReceiptIndexEntry(
        source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
        fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=hashlib.sha256(catalog_raw).hexdigest(),
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=catalog_payload,
    )
    with pytest.raises(PITDataError, match="payload is missing"):
        catalog_master_receipts(_SingleEntryCatalog(valid_entry), sessions=(date(2024, 1, 2),))
    monkeypatch.undo()

    with pytest.raises(PITDataError, match="requires requested sessions"):
        build_ordinary_universe([], sessions=())
    with pytest.raises(PITDataError, match="no certified"):
        catalog_master_sessions(ReceiptCatalog(tmp_path / "empty-catalog"))
