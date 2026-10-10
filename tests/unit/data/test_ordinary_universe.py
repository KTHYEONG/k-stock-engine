"""Ordinary-universe identity and publication tests."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.pit import BronzeReceipt, EvidenceKind
from src.data.datasets import load_manifest, verify_dataset
from src.data.ordinary_universe import ordinary_universe_snapshot, write_ordinary_universe_silver
from tests.fixtures import seed_receipts


def _snapshot(tmp_path: Path, session: date):
    (tmp_path / "catalog").mkdir(parents=True, exist_ok=True)
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
    (tmp_path / "catalog").mkdir(parents=True, exist_ok=True)
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
    seed_receipts(
        catalog,
        [
            ReceiptIndexEntry(
                source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
                fiscal_period=None, status=EvidenceStatus.SUCCESS,
                content_hash=hashlib.sha256(catalog_raw).hexdigest(),
                retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=catalog_payload,
            )
        ],
    )
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
    resolved_missing = catalog_master_receipts(_SingleEntryCatalog(missing_entry), sessions=(date(2024, 1, 2),))
    assert resolved_missing[0].content_hash == "e" * 64
    with pytest.raises(PITDataError, match="unreadable"):
        ordinary_universe_snapshot(resolved_missing[0])

    mismatch_entry = ReceiptIndexEntry(
        source="krx_security_master", natural_key="2024-01-02", as_of=date(2024, 1, 2),
        fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash="e" * 64,
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC), payload_path=catalog_payload,
    )
    resolved_mismatch = catalog_master_receipts(_SingleEntryCatalog(mismatch_entry), sessions=(date(2024, 1, 2),))
    assert resolved_mismatch[0].content_hash == "e" * 64
    with pytest.raises(PITDataError, match="hash mismatch"):
        ordinary_universe_snapshot(resolved_mismatch[0])

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
    with pytest.raises(PITDataError, match="unreadable"):
        ordinary_universe_snapshot(
            catalog_master_receipts(_SingleEntryCatalog(valid_entry), sessions=(date(2024, 1, 2),))[0]
        )
    monkeypatch.undo()

    with pytest.raises(PITDataError, match="requires requested sessions"):
        build_ordinary_universe([], sessions=())
    with pytest.raises(PITDataError, match="no certified"):
        catalog_master_sessions(ReceiptCatalog(tmp_path / "empty-catalog"))


def test_catalog_master_sessions_exclude_exchange_confirmed_closures(tmp_path) -> None:
    """A master snapshot exists on holidays; KRX's empty daily answer proves the closure."""
    import hashlib

    from src.data.ordinary_universe import catalog_master_sessions
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

    root = tmp_path / "bronze"
    payload = root / "payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"{}")
    content_hash = hashlib.sha256(b"{}").hexdigest()

    def _entry(source: str, day: date, status: EvidenceStatus) -> ReceiptIndexEntry:
        return ReceiptIndexEntry(
            source=source, natural_key=day.isoformat(), as_of=day, fiscal_period=None, status=status,
            content_hash=content_hash, retrieved_at=datetime(2026, 9, 26, tzinfo=UTC), payload_path=payload,
        )

    catalog = ReceiptCatalog(root / "catalog")
    seed_receipts(
        catalog,
        [
            _entry("krx_security_master", date(2026, 6, 2), EvidenceStatus.SUCCESS),
            _entry("krx_security_master", date(2026, 6, 3), EvidenceStatus.SUCCESS),
            _entry("krx_daily_market", date(2026, 6, 2), EvidenceStatus.SUCCESS),
            _entry("krx_daily_market", date(2026, 6, 3), EvidenceStatus.EMPTY),
        ],
    )

    assert catalog_master_sessions(catalog) == (date(2026, 6, 2),)


def _publish_master(bronze_root: Path, session: date, *, retrieved_at: datetime, ticker: str = "005930") -> str:
    import hashlib
    import json

    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from tests.fixtures import seed_receipts

    raw = json.dumps({
        "as_of": session.isoformat(),
        "records": [{
            "ISU_SRT_CD": ticker, "ISU_CD": "KR7005930003",
            "KIND_STKCERT_TP_NM": "보통주", "SECUGRP_NM": "주권", "MKT_TP_NM": "KOSPI",
        }],
    }, ensure_ascii=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload_path = bronze_root / "security_master" / digest / "payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(raw)
    seed_receipts(
        ReceiptCatalog(bronze_root / "catalog"),
        [ReceiptIndexEntry(
            source="krx_security_master", natural_key=session.isoformat(), as_of=session,
            fiscal_period=None, status=EvidenceStatus.SUCCESS, content_hash=digest,
            retrieved_at=retrieved_at, payload_path=payload_path,
        )],
    )
    return digest


def _universe_fingerprint(path: Path) -> object:
    import json

    manifest = load_manifest(path)
    return (
        manifest.dataset_id,
        tuple((partition.path, partition.sha256) for partition in manifest.partitions),
        json.dumps(manifest.details, sort_keys=True, default=str),
    )


def test_reuse_equals_full_rebuild(tmp_path: Path) -> None:
    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    _publish_master(bronze, date(2024, 1, 5), retrieved_at=datetime(2024, 1, 6, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    sessions = (*days, date(2024, 1, 5))
    reused = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=sessions, silver_root=tmp_path / "silver-reuse", reuse_from=first
    )
    full = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=sessions, silver_root=tmp_path / "silver-full"
    )
    assert _universe_fingerprint(reused) == _universe_fingerprint(full)


def test_reused_sessions_read_no_payload(tmp_path: Path, monkeypatch) -> None:
    import src.data.ordinary_universe as universe_module
    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    digests = [(_publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))) for day in days]
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    _publish_master(bronze, date(2024, 1, 5), retrieved_at=datetime(2024, 1, 6, tzinfo=UTC))
    sessions = (*days, date(2024, 1, 5))
    full = materialize_ordinary_universe_from_catalog(
        catalog=ReceiptCatalog(bronze / "catalog"), sessions=sessions, silver_root=tmp_path / "silver-full"
    )
    for digest in digests:
        (bronze / "security_master" / digest / "payload.json").unlink()
    calls = 0
    real_read = universe_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(universe_module, "read_payload", _counted)
    catalog = ReceiptCatalog(bronze / "catalog")
    rebuilt = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=sessions,
        silver_root=tmp_path / "silver-reuse", reuse_from=first,
    )
    assert calls == 1
    assert _universe_fingerprint(rebuilt) == _universe_fingerprint(full)


def test_changed_source_hash_is_rebuilt(tmp_path: Path, monkeypatch) -> None:
    import src.data.ordinary_universe as universe_module
    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    _publish_master(bronze, date(2024, 1, 3), retrieved_at=datetime(2024, 1, 6, tzinfo=UTC), ticker="000660")
    calls = 0
    real_read = universe_module.read_payload

    def _counted(path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_read(path)

    monkeypatch.setattr(universe_module, "read_payload", _counted)
    catalog = ReceiptCatalog(bronze / "catalog")
    reused = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-reuse", reuse_from=first
    )
    assert calls == 1
    full = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-full"
    )
    assert _universe_fingerprint(reused) == _universe_fingerprint(full)


def test_corrupt_prior_partition_is_not_trusted(tmp_path: Path) -> None:
    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    with (first / "session=2024-01-02" / "part.parquet").open("ab") as handle:
        handle.write(b"\x00" * 64)
    reused = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-reuse", reuse_from=first
    )
    full = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-full"
    )
    assert _universe_fingerprint(reused) == _universe_fingerprint(full)


def test_packed_bronze_builds_identical_universe(tmp_path: Path) -> None:
    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.payload_store import pack_loose
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    pack_loose(bronze, kinds=["security_master"], offsite_paths=None, remove=True)
    second = materialize_ordinary_universe_from_catalog(
        catalog=ReceiptCatalog(bronze / "catalog"), sessions=days, silver_root=tmp_path / "silver-b"
    )
    assert _universe_fingerprint(first) == _universe_fingerprint(second)


def test_stale_or_foreign_prior_dataset_falls_back_to_full_rebuild(tmp_path: Path) -> None:
    import json
    import shutil

    import polars as pl

    from src.data.datasets import DatasetIdentity, DatasetLayer, load_manifest, publish_dataset
    from src.data.ordinary_universe import REVISION, materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    expected = _universe_fingerprint(
        materialize_ordinary_universe_from_catalog(
            catalog=catalog, sessions=days, silver_root=tmp_path / "silver-full"
        )
    )

    foreign = publish_dataset(
        layer_root=tmp_path / "silver-foreign",
        identity=DatasetIdentity(
            kind="daily_market", layer=DatasetLayer.SILVER, policy_version="krx-daily-market-v1",
            inputs={}, params={},
        ),
        partitions={"part-00000.parquet": pl.DataFrame({"a": [1]})},
    ).path
    for reuse_from in (tmp_path / "missing-dir", foreign):
        rebuilt = materialize_ordinary_universe_from_catalog(
            catalog=catalog, sessions=days, silver_root=tmp_path / "silver-fallback", reuse_from=reuse_from
        )
        assert _universe_fingerprint(rebuilt) == expected

    for details in ({}, {"partitions": [{"session": "2024-01-02"}]}):
        variant = tmp_path / "silver-variant"
        if variant.exists():
            shutil.rmtree(variant)
        shutil.copytree(first, variant / first.name)
        manifest_path = variant / first.name / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {**json.loads(manifest_path.read_text(encoding="utf-8")), "details": details},
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rebuilt = materialize_ordinary_universe_from_catalog(
            catalog=catalog, sessions=days, silver_root=tmp_path / "silver-fallback",
            reuse_from=variant / first.name,
        )
        assert _universe_fingerprint(rebuilt) == expected
    partial = tmp_path / "silver-partial"
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(first, partial / first.name)
    partial_manifest = partial / first.name / "manifest.json"
    raw_manifest = json.loads(partial_manifest.read_text(encoding="utf-8"))
    raw_manifest["partitions"] = [
        item for item in raw_manifest["partitions"] if "2024-01-03" not in str(item.get("path"))
    ]
    raw_manifest["rows"] = sum(int(item["rows"]) for item in raw_manifest["partitions"])
    partial_manifest.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rebuilt = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-fallback",
        reuse_from=partial / first.name,
    )
    assert _universe_fingerprint(rebuilt) == expected
    assert load_manifest(first).kind == "ordinary_universe"
    assert REVISION == "krx-ordinary-equity-v1"


def test_unreadable_prior_partition_falls_back_to_payload(tmp_path: Path, monkeypatch) -> None:
    import polars as pl

    from src.data.ordinary_universe import materialize_ordinary_universe_from_catalog
    from src.data.receipt_catalog import ReceiptCatalog

    bronze = tmp_path / "bronze"
    days = (date(2024, 1, 2), date(2024, 1, 3))
    for day in days:
        _publish_master(bronze, day, retrieved_at=datetime(2024, 1, 5, tzinfo=UTC))
    catalog = ReceiptCatalog(bronze / "catalog")
    first = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-a"
    )
    real_read_parquet = pl.read_parquet

    def _flaky(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path).endswith("session=2024-01-02/part.parquet"):
            raise OSError("transient read race")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", _flaky)
    rebuilt = materialize_ordinary_universe_from_catalog(
        catalog=catalog, sessions=days, silver_root=tmp_path / "silver-reuse", reuse_from=first
    )
    monkeypatch.undo()
    assert _universe_fingerprint(rebuilt) == _universe_fingerprint(
        materialize_ordinary_universe_from_catalog(
            catalog=catalog, sessions=days, silver_root=tmp_path / "silver-full"
        )
    )
