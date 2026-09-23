"""Invariant guards for the certified industry-classification Silver snapshot."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
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


def _write_stock_receipt(
    bronze_root: Path,
    symbol: str,
    collected_at: datetime,
    ksic_code: str = "032604",
    ksic_name: str = "통신 및 방송 장비 제조업",
    delisted_on: str = "",
) -> None:
    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind

    payload = {
        "provider": "KIS",
        "endpoint": "search-stock-info",
        "symbol": symbol,
        "collected_at": collected_at.isoformat(),
        "output": {
            "std_idst_clsf_cd": ksic_code,
            "std_idst_clsf_cd_name": ksic_name,
            "lstg_abol_dt": delisted_on.replace("-", "") if delisted_on else "",
        },
        "records": [{"ticker": symbol, "ksic_code": ksic_code, "ksic_name": ksic_name, "delisted_on": delisted_on}],
    }
    BronzeStore(bronze_root).import_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=collected_at,
        source_label=f"KIS:search-stock-info:{symbol}:{collected_at.date().isoformat()}",
    )


def _write_raw_payload(bronze_root: Path, payload: dict, source_label: str = "KIS:test:raw") -> None:
    from src.data.bronze import BronzeStore
    from src.data.schemas import EvidenceKind

    BronzeStore(bronze_root).import_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        kind=EvidenceKind.INDUSTRY,
        retrieved_at=T1,
        source_label=source_label,
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


def test_quote_only_evidence_yields_observed_rows(tmp_path: Path) -> None:
    from src.data.industry_silver import POLICY_VERSION

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자", market="KOSPI")
    _write_receipt(bronze, "000660", T2, industry="반도체", market="KOSPI")

    result = _materialize(bronze, silver)

    assert result.rows == 2
    assert (result.observed_rows, result.inferred_rows, result.unmapped_rows) == (2, 0, 0)
    frame = _output_frame(result.dataset_path).sort("ticker")
    assert frame["industry_basis"].to_list() == ["observed_quote", "observed_quote"]
    assert frame["industry_name"].to_list() == ["반도체", "전기·전자"]
    assert frame["market_name"].to_list() == ["KOSPI", "KOSPI"]
    assert frame["ksic_code"].to_list() == [None, None]
    assert frame["ksic_name"].to_list() == [None, None]
    assert frame["ksic_support"].to_list() == [None, None]
    assert frame["ksic_source_hash"].to_list() == [None, None]
    assert frame["policy_version"].to_list() == [POLICY_VERSION, POLICY_VERSION]
    assert POLICY_VERSION == "kis-industry-classification-v2"


def test_output_schema_matches_v2_contract(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)
    _write_stock_receipt(bronze, "005930", T1)

    result = _materialize(bronze, silver)

    frame = _output_frame(result.dataset_path)
    assert frame.columns == [
        "ticker", "instrument_id", "industry_name", "industry_basis", "market_name",
        "ksic_code", "ksic_name", "ksic_support", "delisted_on", "available_at",
        "source_hash", "ksic_source_hash", "policy_version",
    ]
    assert result.rows == 1


def test_observed_value_wins_and_disagreement_is_counted(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="화학")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_receipt(bronze, "000660", T1, industry="화학")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="032604")
    _write_receipt(bronze, "035420", T1, industry="제약")
    _write_stock_receipt(bronze, "035420", T1, ksic_code="032604")

    result = _materialize(bronze, silver)

    assert result.observed_rows == 3
    assert result.mapping_disagreements == 1
    frame = _output_frame(result.dataset_path).sort("ticker")
    assert frame.filter(pl.col("ticker") == "035420")["industry_name"].to_list() == ["제약"]
    assert frame.filter(pl.col("ticker") == "035420")["industry_basis"].to_list() == ["observed_quote"]
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["conflicting_ksic"] == ["032604"]


def test_delisted_ticker_with_only_ksic_is_inferred(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_receipt(bronze, "000660", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T2, ksic_code="032604", delisted_on="2026-01-27")

    result = _materialize(bronze, silver)

    assert (result.observed_rows, result.inferred_rows, result.unmapped_rows) == (2, 1, 0)
    frame = _output_frame(result.dataset_path).sort("ticker")
    inferred = frame.filter(pl.col("ticker") == "000001")
    assert inferred["industry_basis"].to_list() == ["ksic_inferred"]
    assert inferred["industry_name"].to_list() == ["전기·전자"]
    assert inferred["ksic_support"].to_list() == [2]
    assert inferred["market_name"].to_list() == [None]
    assert inferred["ksic_code"].to_list() == ["032604"]
    assert inferred["delisted_on"].to_list() == [date(2026, 1, 27)]


def test_unmapped_code_stays_null_with_ksic_populated(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="011101", ksic_name="작물 재배업")

    result = _materialize(bronze, silver)

    assert (result.observed_rows, result.inferred_rows, result.unmapped_rows) == (1, 0, 1)
    frame = _output_frame(result.dataset_path).sort("ticker")
    unmapped = frame.filter(pl.col("ticker") == "000001")
    assert unmapped["industry_basis"].to_list() == ["unmapped"]
    assert unmapped["industry_name"].to_list() == [None]
    assert unmapped["ksic_code"].to_list() == ["011101"]
    assert unmapped["ksic_name"].to_list() == ["작물 재배업"]


def test_conflicting_code_leaves_ksic_only_ticker_unmapped(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_receipt(bronze, "000660", T1, industry="화학")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="032604")

    result = _materialize(bronze, silver)

    frame = _output_frame(result.dataset_path).sort("ticker")
    assert frame.filter(pl.col("ticker") == "000001")["industry_basis"].to_list() == ["unmapped"]
    assert result.mapping_disagreements == 2
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["conflicting_ksic"] == ["032604"]


def test_mapping_uses_only_observed_pairs(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="011101")

    result = _materialize(bronze, silver)

    frame = _output_frame(result.dataset_path).sort("ticker")
    assert frame.filter(pl.col("ticker") == "000660")["ksic_support"].to_list() == [1]
    assert frame.filter(pl.col("ticker") == "000001")["industry_basis"].to_list() == ["unmapped"]


def test_available_at_is_max_of_contributing_receipts(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T2, ksic_code="032604")

    result = _materialize(bronze, silver)

    assert _output_frame(result.dataset_path)["available_at"].to_list() == [T2]


def test_latest_stock_receipt_per_ticker_wins(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="011101")
    _write_stock_receipt(bronze, "000001", T2, ksic_code="032604")

    result = _materialize(bronze, silver)

    frame = _output_frame(result.dataset_path).sort("ticker")
    inferred = frame.filter(pl.col("ticker") == "000001")
    assert inferred["ksic_code"].to_list() == ["032604"]
    assert inferred["industry_basis"].to_list() == ["ksic_inferred"]
    assert inferred["ksic_source_hash"].to_list() == inferred["source_hash"].to_list()


def test_unknown_endpoint_fails_closed_without_dataset(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1)
    _write_raw_payload(bronze, {
        "provider": "KIS",
        "endpoint": "other",
        "symbol": "000660",
        "collected_at": T1.isoformat(),
        "output": {},
        "records": [{"ticker": "000660"}],
    })

    with pytest.raises(PITDataError, match="endpoint"):
        _materialize(bronze, silver)

    assert not silver.exists() or not list(silver.iterdir())


def test_malformed_stock_info_payload_fails_closed(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_raw_payload(bronze, {
        "provider": "KIS",
        "endpoint": "search-stock-info",
        "symbol": "000001",
        "collected_at": T1.isoformat(),
        "output": {"std_idst_clsf_cd": "32604"},
        "records": [{"ticker": "000001", "ksic_code": "32604", "ksic_name": "", "delisted_on": ""}],
    })

    with pytest.raises(PITDataError):
        _materialize(bronze, silver)


def test_malformed_delisted_on_fails_closed(tmp_path: Path) -> None:
    from src.data.schemas import PITDataError

    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_raw_payload(bronze, {
        "provider": "KIS",
        "endpoint": "search-stock-info",
        "symbol": "000001",
        "collected_at": T1.isoformat(),
        "output": {"std_idst_clsf_cd": "032604"},
        "records": [{"ticker": "000001", "ksic_code": "032604", "ksic_name": "", "delisted_on": "27-01-2026"}],
    })

    with pytest.raises(PITDataError):
        _materialize(bronze, silver)


def test_counts_reconcile_and_manifest_records_them(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_receipt(bronze, "000660", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="032604")
    _write_stock_receipt(bronze, "000002", T1, ksic_code="011101")

    result = _materialize(bronze, silver)

    assert result.rows == result.observed_rows + result.inferred_rows + result.unmapped_rows
    assert (result.observed_rows, result.inferred_rows, result.unmapped_rows) == (2, 1, 1)
    manifest = json.loads((result.dataset_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observed_rows"] == 2
    assert manifest["inferred_rows"] == 1
    assert manifest["unmapped_rows"] == 1
    assert manifest["mapping_disagreements"] == 0
    assert manifest["conflicting_ksic"] == []


def test_rebuild_is_deterministic_regardless_of_receipt_order(tmp_path: Path) -> None:
    bronze_a, bronze_b = tmp_path / "bronze-a", tmp_path / "bronze-b"
    _write_receipt(bronze_a, "005930", T1)
    _write_stock_receipt(bronze_a, "005930", T2, ksic_code="032604")
    _write_stock_receipt(bronze_b, "005930", T2, ksic_code="032604")
    _write_receipt(bronze_b, "005930", T1)

    first = _materialize(bronze_a, tmp_path / "silver-a")
    second = _materialize(bronze_b, tmp_path / "silver-b")

    assert second.dataset_id == first.dataset_id
    repeat = _materialize(bronze_a, tmp_path / "silver-a")
    assert repeat.dataset_id == first.dataset_id
    assert repeat.dataset_path == first.dataset_path


def test_symbol_filter_restricts_both_streams_and_mapping(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_receipt(bronze, "005930", T1, industry="전기·전자")
    _write_stock_receipt(bronze, "005930", T1, ksic_code="032604")
    _write_receipt(bronze, "000660", T1, industry="화학")
    _write_stock_receipt(bronze, "000660", T1, ksic_code="011101")
    _write_stock_receipt(bronze, "000001", T1, ksic_code="011101")

    filtered = _materialize(bronze, silver, symbols=frozenset({"005930", "000001"}))

    assert filtered.rows == 2
    frame = _output_frame(filtered.dataset_path).sort("ticker")
    assert frame["ticker"].to_list() == ["000001", "005930"]
    assert frame.filter(pl.col("ticker") == "000001")["industry_basis"].to_list() == ["unmapped"]

    unfiltered = _materialize(bronze, tmp_path / "silver-all")

    assert unfiltered.rows == 3
    assert unfiltered.dataset_id != filtered.dataset_id
