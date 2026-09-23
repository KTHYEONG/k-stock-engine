"""Invariant guards for the certified industry-classification Silver snapshot."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

T1 = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)
T2 = datetime(2024, 2, 5, 9, 0, tzinfo=UTC)


def _write_receipt(
    bronze_root: Path,
    symbol: str,
    collected_at: datetime,
    industry: str = "전기·전자",
    market: str = "KOSPI",
) -> None:
    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind

    payload = {
        "provider": "KIS",
        "endpoint": "inquire-price",
        "symbol": symbol,
        "collected_at": collected_at.isoformat(),
        "output": {"bstp_kor_isnm": industry, "rprs_mrkt_kor_name": market},
        "records": [{"ticker": symbol, "industry_name": industry, "market_name": market}],
    }
    BronzeStore(bronze_root).import_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=collected_at,
        source_label=f"KIS:inquire-price:{symbol}:{collected_at.date().isoformat()}",
    )


def _write_raw_receipt(bronze_root: Path, raw: bytes) -> None:
    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind

    BronzeStore(bronze_root).import_bytes(
        raw,
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=T1,
        source_label="KIS:inquire-price:raw",
    )


def _materialize(bronze_root: Path, silver_root: Path, symbols=None):
    from src.data.industry_silver import materialize_industry_classification_silver

    return materialize_industry_classification_silver(
        bronze_root=bronze_root, silver_root=silver_root, symbols=symbols
    )


def _output_frame(dataset_path: Path) -> pl.DataFrame:
    return pl.read_parquet(dataset_path / "part.parquet")


def test_materialize_keeps_latest_collection_per_ticker(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_receipt(bronze, "005930", T2, industry="반도체")

    result = _materialize(bronze, silver)

    assert result.rows == 1
    frame = _output_frame(result.dataset_path)
    assert frame["ticker"].to_list() == ["005930"]
    assert frame["industry_name"].to_list() == ["반도체"]
    assert frame["available_at"].to_list() == [T2]
    assert frame["instrument_id"].to_list() == ["KRX:005930"]


def test_materialize_restricts_to_symbol_filter(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)
    _write_receipt(bronze, "000660", T1)

    result = _materialize(bronze, silver, symbols=frozenset({"005930"}))

    assert result.rows == 1
    assert _output_frame(result.dataset_path)["ticker"].to_list() == ["005930"]


def test_materialize_accepts_naive_collected_at_as_utc(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    naive = datetime(2024, 1, 3, 9, 0)
    _write_raw_receipt(
        bronze,
        json.dumps({
            "provider": "KIS",
            "endpoint": "inquire-price",
            "symbol": "005930",
            "collected_at": naive.isoformat(),
            "output": {"bstp_kor_isnm": "전기·전자", "rprs_mrkt_kor_name": "KOSPI"},
            "records": [{"ticker": "005930", "industry_name": "전기·전자", "market_name": "KOSPI"}],
        }).encode("utf-8"),
    )

    result = _materialize(bronze, silver)

    assert _output_frame(result.dataset_path)["available_at"].to_list() == [
        naive.replace(tzinfo=UTC)
    ]


def test_materialize_is_deterministic_across_runs(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)
    _write_receipt(bronze, "000660", T2)

    first = _materialize(bronze, silver)
    manifest_before = (first.dataset_path / "manifest.json").read_bytes()
    second = _materialize(bronze, silver)

    assert second.dataset_id == first.dataset_id
    assert second.dataset_path == first.dataset_path
    assert (second.dataset_path / "manifest.json").read_bytes() == manifest_before


def test_materialize_rejects_tampered_bronze_receipt(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)
    part = next((bronze / "industry").rglob("payload.json"))
    part.write_bytes(part.read_bytes() + b"tampered")

    with pytest.raises(PITDataError):
        _materialize(bronze, silver)

    assert not silver.exists()


def test_materialize_rejects_malformed_payloads(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    cases = [
        b"not json{",
        b"[1, 2]",
        json.dumps({"collected_at": T1.isoformat(), "records": []}).encode(),
        json.dumps({"symbol": "005930", "collected_at": "not-a-date", "records": []}).encode(),
        json.dumps({"symbol": "005930", "collected_at": T1.isoformat(), "records": []}).encode(),
        json.dumps({"symbol": "005930", "collected_at": T1.isoformat(), "records": ["nope"]}).encode(),
        json.dumps({
            "symbol": "005930",
            "collected_at": T1.isoformat(),
            "records": [{"ticker": "005930", "industry_name": "  "}],
        }).encode(),
    ]
    for index, raw in enumerate(cases):
        bronze, silver = tmp_path / f"bronze-{index}", tmp_path / f"silver-{index}"
        _write_raw_receipt(bronze, raw)
        with pytest.raises(PITDataError):
            _materialize(bronze, silver)


def test_materialize_rejects_empty_symbol_filter(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="non-empty"):
        _materialize(tmp_path / "bronze", tmp_path / "silver", symbols=frozenset())


def test_materialize_rejects_missing_evidence(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="no certified"):
        _materialize(tmp_path / "bronze", tmp_path / "silver")


def test_materialize_rejects_unmatched_symbol_filter(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze = tmp_path / "bronze"
    _write_receipt(bronze, "005930", T1)

    with pytest.raises(PITDataError, match="matches the symbol filter"):
        _materialize(bronze, tmp_path / "silver", symbols=frozenset({"000660"}))


def test_materialize_rejects_filtered_empty_bronze(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    with pytest.raises(PITDataError, match="no certified"):
        _materialize(tmp_path / "bronze", tmp_path / "silver", symbols=frozenset({"005930"}))


def test_materialize_rejects_differing_existing_dataset(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)

    result = _materialize(bronze, silver)
    manifest_path = result.dataset_path / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    tampered = manifest_path.read_bytes()

    with pytest.raises(PITDataError, match="differs"):
        _materialize(bronze, silver)

    assert manifest_path.read_bytes() == tampered


def test_materialize_rejects_unreadable_existing_manifest(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)

    result = _materialize(bronze, silver)
    (result.dataset_path / "manifest.json").unlink()

    with pytest.raises(PITDataError, match="unreadable"):
        _materialize(bronze, silver)


def test_materialize_distinguishes_close_collections(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_receipt(bronze, "005930", T1 + timedelta(microseconds=1), industry="반도체")

    result = _materialize(bronze, silver)

    assert _output_frame(result.dataset_path)["industry_name"].to_list() == ["반도체"]
