"""Catalog-driven industry snapshot tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import EvidenceKind
from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

T1 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
T2 = datetime(2024, 2, 5, 9, 0, tzinfo=UTC)


def _seed(catalog: ReceiptCatalog, bronze_root: Path, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze_root / "industry" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    catalog.publish(
        [],
        blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INDUSTRY, source="kis_industry",
                          usable=True, unusable_reason=None, retrieved_at=T1,
                          payload_path=target / "payload.json")],
    )
    return digest


def _quote(symbol: str, collected_at: datetime, industry: str = "전기·전자", market: str = "KOSPI") -> dict:
    return {
        "provider": "KIS", "endpoint": "inquire-price", "symbol": symbol,
        "collected_at": collected_at.isoformat(),
        "records": [{"industry_name": industry, "market_name": market}],
    }


def _stock(symbol: str, collected_at: datetime, code: str = "032604") -> dict:
    return {
        "provider": "KIS", "endpoint": "search-stock-info", "symbol": symbol,
        "collected_at": collected_at.isoformat(),
        "records": [{"ksic_code": code, "ksic_name": "통신장비", "delisted_on": ""}],
    }


def _materialize(bronze_root: Path, catalog: ReceiptCatalog, silver_root: Path, symbols=None):
    from src.data.industry_silver import materialize_industry_classification_silver

    return materialize_industry_classification_silver(
        catalog=catalog, silver_root=silver_root, symbols=symbols
    )


def _frame(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path / "part.parquet")


def test_builds_from_catalog_blobs(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed(catalog, bronze, _quote("005930", T1))
    _seed(catalog, bronze, _quote("005930", T2, industry="반도체"))
    result = _materialize(bronze, catalog, silver)
    assert result.rows == 1
    assert _frame(result.dataset_path)["industry_name"].to_list() == ["반도체"]


def test_preview_reads_no_payload(tmp_path: Path) -> None:
    """Preview reads no payload: blob_digest works with payload files unreadable."""
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed(catalog, bronze, _quote("005930", T1))
    for path in bronze.rglob("payload.json"):
        path.chmod(0o000)
    try:
        assert catalog.blob_digest(source="kis_industry") != catalog.blob_digest(source="missing")
    finally:
        for path in bronze.rglob("payload.json"):
            path.chmod(0o644)


def test_uncatalogued_files_ignored(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed(catalog, bronze, _quote("005930", T1))
    extra = _quote("999999", T1)
    raw = json.dumps(extra, sort_keys=True, ensure_ascii=False).encode("utf-8")
    target = bronze / "industry" / hashlib.sha256(raw).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(raw)
    result = _materialize(bronze, catalog, silver)
    assert "999999" not in _frame(result.dataset_path)["ticker"].to_list()


def test_unusable_blobs_hidden(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    catalog = ReceiptCatalog(bronze / "catalog")
    _seed(catalog, bronze, _quote("005930", T1))
    raw = json.dumps(_quote("000660", T1), sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    target = bronze / "industry" / digest
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.json").write_bytes(b"corrupt-bytes-not-matching")
    assert list(catalog.blobs(source="kis_industry", usable=True)).__len__() == 1
    result = _materialize(bronze, catalog, silver)
    assert result.rows == 1


def test_rejects_missing_evidence_and_bad_filter(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    catalog = ReceiptCatalog(tmp_path / "bronze" / "catalog")
    with pytest.raises(PITDataError, match="no certified"):
        _materialize(tmp_path / "bronze", catalog, tmp_path / "silver")
    bronze = tmp_path / "bronze2"
    catalog2 = ReceiptCatalog(bronze / "catalog")
    _seed(catalog2, bronze, _quote("005930", T1))
    with pytest.raises(PITDataError, match="non-empty"):
        _materialize(bronze, catalog2, tmp_path / "silver2", symbols=frozenset())


def test_fail_closed_on_unreadable_blob(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    catalog = ReceiptCatalog(bronze / "catalog")
    digest = _seed(catalog, bronze, _quote("005930", T1))
    path = bronze / "industry" / digest / "payload.json"
    path.chmod(0o000)
    try:
        with pytest.raises(PITDataError):
            _materialize(bronze, catalog, silver)
    finally:
        path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PITDataError):
        _materialize(bronze, catalog, tmp_path / "silver-tampered")


def test_fail_closed_on_malformed_blobs(tmp_path: Path) -> None:
    from src.core.pit import PITDataError

    cases = [
        b"not json",
        json.dumps({"provider": "KIS", "endpoint": "other", "symbol": "005930",
                    "collected_at": T1.isoformat(), "records": [{"x": 1}]}).encode(),
    ]
    for index, raw in enumerate(cases):
        bronze = tmp_path / f"bronze-bad-{index}"
        catalog = ReceiptCatalog(bronze / "catalog")
        digest = hashlib.sha256(raw).hexdigest()
        target = bronze / "industry" / digest
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.json").write_bytes(raw)
        catalog.publish(
            [],
            blobs=[BlobEntry(content_hash=digest, kind=EvidenceKind.INDUSTRY, source="kis_industry",
                              usable=True, unusable_reason=None, retrieved_at=T1,
                              payload_path=target / "payload.json")],
        )
        with pytest.raises(PITDataError):
            _materialize(bronze, catalog, tmp_path / f"silver-bad-{index}")
