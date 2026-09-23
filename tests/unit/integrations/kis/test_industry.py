from datetime import UTC, datetime

import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.integrations.kis.industry import KisIndustryCollector, KisStockClassificationCollector

RETRIEVED_AT = datetime(2024, 1, 3, 9, 0, tzinfo=UTC)


def _output(industry="전기·전자", market="KOSPI", **extra):
    return {"bstp_kor_isnm": industry, "rprs_mrkt_kor_name": market, **extra}


class _StubClient:
    def __init__(self, outputs) -> None:
        self.outputs = dict(outputs)
        self.calls: list = []

    def inquire_price(self, symbol):
        self.calls.append(symbol)
        outcome = self.outputs[symbol]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _receipts(bronze_root):
    import json

    found = []
    for receipt_path in sorted((bronze_root / "industry").rglob("receipt.json")):
        meta = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload = json.loads((receipt_path.parent / "payload.json").read_text(encoding="utf-8"))
        found.append((meta, payload))
    return found


def test_fetch_maps_industry_name_verbatim(tmp_path) -> None:
    outputs = {"005930": _output("전기·전자", "KOSPI", iscd_stat_cls_code="11", extra_field="kept")}

    pages = tuple(
        KisIndustryCollector(("005930",), client=_StubClient(outputs)).fetch_industry_classification(
            bronze_root=tmp_path / "bronze", retrieved_at=RETRIEVED_AT
        )
    )

    assert pages[0]["records"] == [
        {"ticker": "005930", "industry_name": "전기·전자", "market_name": "KOSPI"}
    ]


def test_fetch_rejects_missing_industry_field(tmp_path) -> None:
    bronze_root = tmp_path / "bronze"

    with pytest.raises(PITDataError, match="bstp_kor_isnm"):
        tuple(
            KisIndustryCollector(("000001",), client=_StubClient({"000001": {"rprs_mrkt_kor_name": "KOSPI"}})).fetch_industry_classification(
                bronze_root=bronze_root, retrieved_at=RETRIEVED_AT
            )
        )

    assert not (bronze_root / "industry").exists()


def test_fetch_rejects_empty_industry_field(tmp_path) -> None:
    bronze_root = tmp_path / "bronze"

    with pytest.raises(PITDataError, match="bstp_kor_isnm"):
        tuple(
            KisIndustryCollector(("000001",), client=_StubClient({"000001": _output("   ")})).fetch_industry_classification(
                bronze_root=bronze_root, retrieved_at=RETRIEVED_AT
            )
        )

    assert not (bronze_root / "industry").exists()


def test_fetch_writes_one_industry_receipt_per_symbol(tmp_path) -> None:
    outputs = {
        "005930": _output("전기·전자", "KOSPI"),
        "000660": _output("전기·전자", "KOSPI"),
        "035420": _output("서비스업", "KOSPI"),
    }
    bronze_root = tmp_path / "bronze"

    pages = tuple(
        KisIndustryCollector(tuple(outputs), client=_StubClient(outputs)).fetch_industry_classification(
            bronze_root=bronze_root, retrieved_at=RETRIEVED_AT
        )
    )

    assert [page["symbol"] for page in pages] == ["005930", "000660", "035420"]
    found = _receipts(bronze_root)
    assert len(found) == 3
    for meta, payload in found:
        assert meta["kind"] == EvidenceKind.INDUSTRY.value
        assert meta["source_path"].startswith(f"KIS:inquire-price:{payload['symbol']}:2024-01-03")


def test_fetch_persists_raw_output_verbatim(tmp_path) -> None:
    raw = _output("전기·전자", "KOSPI", iscd_stat_cls_code="11", another_raw_field="raw-value")
    bronze_root = tmp_path / "bronze"

    tuple(
        KisIndustryCollector(("005930",), client=_StubClient({"005930": raw})).fetch_industry_classification(
            bronze_root=bronze_root, retrieved_at=RETRIEVED_AT
        )
    )

    (meta, payload) = _receipts(bronze_root)[0]
    assert payload["output"] == raw


def test_collector_rejects_empty_universe() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        KisIndustryCollector(())
    with pytest.raises(ValueError, match="at least one symbol"):
        KisIndustryCollector(("   ",))


def test_fetch_propagates_transient_error(tmp_path) -> None:
    with pytest.raises(PITDataError, match="collection failed"):
        tuple(
            KisIndustryCollector(
                ("005930",), client=_StubClient({"005930": RuntimeError("boom")})
            ).fetch_industry_classification(bronze_root=tmp_path / "bronze")
        )


def _stock_output(code="032604", name="통신 및 방송 장비 제조업", abolished="", **extra):
    return {"std_idst_clsf_cd": code, "std_idst_clsf_cd_name": name, "lstg_abol_dt": abolished, **extra}


class _StockStubClient:
    def __init__(self, outputs) -> None:
        self.outputs = dict(outputs)
        self.calls: list = []

    def search_stock_info(self, symbol):
        self.calls.append(symbol)
        outcome = self.outputs[symbol]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_fetch_stock_classification_maps_code_and_name_verbatim(tmp_path) -> None:
    outputs = {"005930": _stock_output("032604", "통신 및 방송 장비 제조업", "")}

    pages = tuple(
        KisStockClassificationCollector(("005930",), client=_StockStubClient(outputs)).fetch_stock_classification(
            bronze_root=tmp_path / "bronze", retrieved_at=RETRIEVED_AT
        )
    )

    assert pages[0]["records"] == [
        {"ticker": "005930", "ksic_code": "032604", "ksic_name": "통신 및 방송 장비 제조업", "delisted_on": ""}
    ]


def test_fetch_stock_classification_normalizes_abolition_date(tmp_path) -> None:
    outputs = {"000001": _stock_output("011101", "작물 재배업", "20260127")}

    pages = tuple(
        KisStockClassificationCollector(("000001",), client=_StockStubClient(outputs)).fetch_stock_classification(
            bronze_root=tmp_path / "bronze", retrieved_at=RETRIEVED_AT
        )
    )

    assert pages[0]["records"][0]["delisted_on"] == "2026-01-27"


def test_fetch_stock_classification_rejects_malformed_code(tmp_path) -> None:
    for bad in ("", "32604", "03260a", "0326041"):
        bronze_root = tmp_path / f"bronze-{bad or 'empty'}"
        with pytest.raises(PITDataError, match="000001"):
            tuple(
                KisStockClassificationCollector(
                    ("000001",), client=_StockStubClient({"000001": _stock_output(bad)})
                ).fetch_stock_classification(bronze_root=bronze_root, retrieved_at=RETRIEVED_AT)
            )
        assert not (bronze_root / "industry").exists()


def test_fetch_stock_classification_rejects_malformed_abolition_date(tmp_path) -> None:
    for bad in ("2026-01-27", "26", "20260230"):
        bronze_root = tmp_path / f"bronze-{bad.replace('-', '')}"
        with pytest.raises(PITDataError, match="000001"):
            tuple(
                KisStockClassificationCollector(
                    ("000001",), client=_StockStubClient({"000001": _stock_output("032604", "통신 및 방송 장비 제조업", bad)})
                ).fetch_stock_classification(bronze_root=bronze_root, retrieved_at=RETRIEVED_AT)
            )
        assert not (bronze_root / "industry").exists()


def test_fetch_stock_classification_persists_discriminated_evidence(tmp_path) -> None:
    raw = _stock_output("032604", "통신 및 방송 장비 제조업", "", idx_bztp_scls_cd_name="old-sector")
    bronze_root = tmp_path / "bronze"

    tuple(
        KisStockClassificationCollector(("005930",), client=_StockStubClient({"005930": raw})).fetch_stock_classification(
            bronze_root=bronze_root, retrieved_at=RETRIEVED_AT
        )
    )

    (meta, payload) = _receipts(bronze_root)[0]
    assert payload["endpoint"] == "search-stock-info"
    assert payload["output"] == raw
    assert meta["kind"] == EvidenceKind.INDUSTRY.value
    assert meta["source_path"].startswith("KIS:search-stock-info:005930:2024-01-03")


def test_fetch_stock_classification_wraps_transport_failure(tmp_path) -> None:
    with pytest.raises(PITDataError, match="005930"):
        tuple(
            KisStockClassificationCollector(
                ("005930",), client=_StockStubClient({"005930": RuntimeError("boom")})
            ).fetch_stock_classification(bronze_root=tmp_path / "bronze", retrieved_at=RETRIEVED_AT)
        )


def test_stock_collector_rejects_empty_universe() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        KisStockClassificationCollector(())
    with pytest.raises(ValueError, match="at least one symbol"):
        KisStockClassificationCollector(("   ",))
