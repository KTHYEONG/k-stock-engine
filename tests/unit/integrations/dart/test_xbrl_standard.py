"""OpenDART standard quarterly financial facts."""
from __future__ import annotations

from typing import Any


def test_opendart_standard_facts_parsed_with_values_and_fiscal_period() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    mock_raw = {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "thstrm_amount": "47,117,896,000,000",
            },
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_OperatingProfit",
                "account_nm": "영업이익",
                "fs_div": "CFS",
                "thstrm_amount": "5,979,343,000,000",
            },
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "stock_code": "005930",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Assets",
                "account_nm": "자산총계",
                "fs_div": "CFS",
                "thstrm_amount": "233,401,659,000,000",
            },
        ],
    }
    collector = DartXbrlCollector(
        api_key="fixture-key",
        request_json=lambda _endpoint, _params: mock_raw,
    )
    identity = {
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "biz_year": "2015",
        "reprt_code": "11013",
        "fs_div": "CFS",
        "published_at": "2015-05-15",
    }
    pages = list(collector.fetch_financial_fact_sources((identity,)))
    assert len(pages) == 1
    page = pages[0]
    assert page["source_kind"] == "opendart_standard"
    assert page["status"] == "000"
    records = page["records"]
    assert len(records) == 3
    sales_rec = next(r for r in records if r["fact"] == "sales")
    assert sales_rec["value"] == 47117896000000.0
    assert sales_rec["fiscal_period"] == "2015Q1"
    assert sales_rec["unit"] == "KRW"
    assert sales_rec["company_id"] == "00126380"
    assert sales_rec["consolidated"] is True


def test_normalize_dart_financial_facts_accepts_opendart_standard_records() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    page = {
        "source_kind": "opendart_standard",
        "status": "000",
        "mapping_version": "dart-fact-map-v1",
        "raw_document_hash": None,
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "fiscal_period": "2015Q1",
        "published_at": "2015-05-15T09:00:00+09:00",
        "records": [
            {
                "company_id": "00126380",
                "fiscal_period": "2015Q1",
                "filing_id": "20150515001111",
                "fact": "sales",
                "value": 47117896000000.0,
                "unit": "KRW",
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "opendart_standard",
                "mapping_version": "dart-fact-map-v1",
                "raw_document_hash": None,
            },
            {
                "company_id": "00126380",
                "fiscal_period": "2015Q1",
                "filing_id": "20150515001111",
                "fact": "operating_profit",
                "value": 5979343000000.0,
                "unit": "KRW",
                "consolidated": True,
                "restatement_id": "r0",
                "source_kind": "opendart_standard",
                "mapping_version": "dart-fact-map-v1",
                "raw_document_hash": None,
            },
        ],
    }
    decision_time = datetime(2016, 1, 1, 9, 0, tzinfo=UTC)
    calendar = SessionCalendar((datetime(2015, 5, 18, 9, 0, tzinfo=UTC),))
    df = normalize_dart_financial_facts(
        pages=[page],
        disclosure_rows=(),
        source_hash="a" * 64,
        calendar=calendar,
        decision_time=decision_time,
        ticker_by_corp_code={"00126380": "005930"},
        bridge_receipt_hash="b" * 64,
    )
    assert df.height == 2
    assert set(df["fact"].to_list()) == {"sales", "operating_profit"}
    assert df["company_id"].to_list() == ["005930", "005930"]
    assert df["ticker"].to_list() == ["005930", "005930"]
    assert df["dart_corp_code"].to_list() == ["00126380", "00126380"]
    assert df["fiscal_period"].to_list() == ["2015Q1", "2015Q1"]
    assert df["value"].to_list() == [47117896000000.0, 5979343000000.0]


def test_collect_dart_disclosures_persists_bronze_receipt(tmp_path: Any) -> None:
    from datetime import UTC, date, datetime
    from pathlib import Path
    from src.data.collection import collect_dart_disclosures
    from src.data.schemas import EvidenceKind

    class DummyDartCollector:
        def fetch_disclosures(self, start: date, end: date) -> list[dict[str, Any]]:
            return [
                {
                    "records": [
                        {
                            "rcept_no": "20150515001111",
                            "corp_code": "00126380",
                            "report_nm": "분기보고서 (2015.03)",
                            "rcept_dt": "20150515",
                        }
                    ]
                }
            ]

    bronze_root = Path(tmp_path) / "bronze"
    retrieved_at = datetime(2016, 1, 1, 9, 0, tzinfo=UTC)
    artifact = collect_dart_disclosures(
        dart=DummyDartCollector(),
        start=date(2015, 1, 1),
        end=date(2015, 12, 31),
        bronze_root=bronze_root,
        retrieved_at=retrieved_at,
    )
    assert EvidenceKind.DISCLOSURES in artifact.receipts
    receipt = artifact.receipts[EvidenceKind.DISCLOSURES]
    assert receipt.payload_path.exists()
    assert artifact.report_path.exists()


def test_filing_identities_from_bronze_multi_receipt(tmp_path: Any) -> None:
    import json
    from datetime import date
    from pathlib import Path
    from src.integrations.dart.xbrl import DartXbrlCollector

    bronze_root = Path(tmp_path) / "bronze"
    receipt1_dir = bronze_root / "disclosures" / "receipt1"
    receipt2_dir = bronze_root / "disclosures" / "receipt2"
    receipt1_dir.mkdir(parents=True, exist_ok=True)
    receipt2_dir.mkdir(parents=True, exist_ok=True)

    payload1 = {
        "records": [
            {
                "rcept_no": "20150515001111",
                "corp_code": "00126380",
                "report_nm": "분기보고서 (2015.03)",
                "rcept_dt": "20150515",
            }
        ]
    }
    payload2 = {
        "records": [
            {
                "rcept_no": "20150817002222",
                "corp_code": "00126380",
                "report_nm": "반기보고서 (2015.06)",
                "rcept_dt": "20150817",
            }
        ]
    }
    (receipt1_dir / "payload.json").write_text(json.dumps(payload1), encoding="utf-8")
    (receipt2_dir / "payload.json").write_text(json.dumps(payload2), encoding="utf-8")

    identities = DartXbrlCollector.filing_identities_from_bronze(
        bronze_root, start=date(2015, 1, 1), end=date(2015, 12, 31)
    )
    assert len(identities) == 2
    fids = {item["filing_id"] for item in identities}
    assert fids == {"20150515001111", "20150817002222"}
    reprt_codes = {item["reprt_code"] for item in identities}
    assert reprt_codes == {"11013", "11012"}


def test_filing_identities_attach_frozen_ticker_and_required_period_only(tmp_path) -> None:
    import json
    from datetime import date
    from src.integrations.dart.xbrl import DartXbrlCollector

    path = tmp_path / "bronze" / "disclosures" / "r1"
    path.mkdir(parents=True)
    path.joinpath("payload.json").write_text(json.dumps({"records": [{"rcept_no": "20150515000001", "corp_code": "00126380", "report_nm": "분기보고서 (2015.03)", "rcept_dt": "20150515"}, {"rcept_no": "20151115000002", "corp_code": "00126380", "report_nm": "분기보고서 (2015.09)", "rcept_dt": "20151115"}]}), encoding="utf-8")

    rows = DartXbrlCollector.filing_identities_from_bronze(tmp_path / "bronze", start=date(2015, 1, 1), end=date(2015, 12, 31), ticker_by_corp_code={"00126380": "005930"}, required_periods=frozenset({"2015Q1"}))

    assert len(rows) == 1
    assert rows[0]["ticker"] == "005930"
    assert rows[0]["reprt_code"] == "11013"

def test_fetch_one_financial_fact_source_matches_original_per_identity_behavior() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: the same fixture as the pre-existing CFS-success test.
    mock_raw = {
        "status": "000",
        "list": [
            {
                "rcept_no": "20150515001111",
                "bsns_year": "2015",
                "corp_code": "00126380",
                "reprt_code": "11013",
                "account_id": "ifrs-full_Revenue",
                "account_nm": "매출액",
                "fs_div": "CFS",
                "thstrm_amount": "47,117,896,000,000",
            }
        ],
    }
    collector = DartXbrlCollector(api_key="fixture-key", request_json=lambda _e, _p: mock_raw)
    identity = {
        "corp_code": "00126380",
        "filing_id": "20150515001111",
        "rcept_no": "20150515001111",
        "biz_year": "2015",
        "reprt_code": "11013",
        "fs_div": "CFS",
        "published_at": "2015-05-15",
        "ticker": "",
    }

    # When
    page = collector._fetch_one_financial_fact_source(identity)

    # Then
    assert page["source_kind"] == "opendart_standard"
    assert page["status"] == "000"
    sales_rec = next(r for r in page["records"] if r["fact"] == "sales")
    assert sales_rec["value"] == 47117896000000.0
    assert sales_rec["consolidated"] is True


def test_fetch_one_financial_fact_source_client_transport_success_and_error_wrapping() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: client transport succeeds (status 013, no request_json set).
    class FakeClientOK:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_ok = DartXbrlCollector(api_key="k", client=FakeClientOK())
    page = collector_ok._fetch_one_financial_fact_source(identity)
    assert page["source_kind"] == "unavailable"

    # Given/When/Then: client raises a DART-specific exception -> wrapped.
    class FakeClientDartError:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            from src.integrations.dart.client import DartApiError

            raise DartApiError("boom")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_dart_err = DartXbrlCollector(api_key="k", client=FakeClientDartError())
    with pytest.raises(PITDataError, match="F1"):
        collector_dart_err._fetch_one_financial_fact_source(identity)

    # Given/When/Then: client raises a generic exception -> also wrapped.
    class FakeClientGeneric:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            raise RuntimeError("network blip")

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_generic = DartXbrlCollector(api_key="k", client=FakeClientGeneric())
    with pytest.raises(PITDataError, match="F1"):
        collector_generic._fetch_one_financial_fact_source(identity)


def test_fetch_one_financial_fact_source_raises_when_no_transport_configured() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given: a constructed collector with both request transports cleared post-construction.
    collector_no_request = DartXbrlCollector(api_key="k")
    collector_no_request._client = None
    collector_no_request._request_json = None

    # When/Then: the request stage fails closed.
    with pytest.raises(PITDataError, match="not configured"):
        collector_no_request._fetch_one_financial_fact_source(identity)

    # Given: a collector that can request but cannot fetch the archive fallback.
    collector_no_archive = DartXbrlCollector(
        api_key="k", request_json=lambda _e, _p: {"status": "013", "list": []}
    )
    collector_no_archive._client = None
    collector_no_archive._request_bytes = None

    # When/Then: the archive stage fails closed too.
    with pytest.raises(PITDataError, match="not configured"):
        collector_no_archive._fetch_one_financial_fact_source(identity)

    # Given/When/Then: an empty (falsy) raw response also fails closed.
    collector_empty_raw = DartXbrlCollector(api_key="k", request_json=lambda _e, _p: {})
    with pytest.raises(PITDataError, match="F1"):
        collector_empty_raw._fetch_one_financial_fact_source(identity)


def test_fetch_one_financial_fact_source_records_every_row_diagnostic_kind() -> None:
    from src.integrations.dart.xbrl import DartXbrlCollector

    # Given: one row per malformed shape, plus a valid row with no fiscal_period source.
    rows: list[object] = [
        "not-a-dict",
        {"account_id": "unknown_xyz", "account_nm": "???", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "abc", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "inf", "bsns_year": "2020"},
        {"account_id": "ifrs-full_Revenue", "account_nm": "매출액", "thstrm_amount": "100"},
    ]
    collector = DartXbrlCollector(
        api_key="k", request_json=lambda _e, _p: {"status": "000", "list": rows}
    )
    # identity.biz_year is empty so the last row (no row-level bsns_year) cannot resolve a period.
    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # When
    page = collector._fetch_one_financial_fact_source(identity)

    # Then: every malformed row produced its own diagnostic and was skipped.
    diagnostics = page["diagnostics"]
    assert any(d.startswith("unknown_account") for d in diagnostics)
    assert any(d.startswith("missing_amount") for d in diagnostics)
    assert any(d.startswith("non_finite") for d in diagnostics)
    assert any(d.startswith("missing_fiscal_period") for d in diagnostics)
    assert page["records"] == []

    # Given/When/Then: a status-000 response with an empty facts list falls through
    # to the archive fallback rather than raising.
    empty_collector = DartXbrlCollector(
        api_key="k",
        request_json=lambda _e, _p: {"status": "000", "list": []},
        request_bytes=lambda _e, _p: b"",
    )
    ofs_identity = {**identity, "biz_year": "2020", "fs_div": "OFS"}
    empty_page = empty_collector._fetch_one_financial_fact_source(ofs_identity)
    assert empty_page["source_kind"] == "unavailable"


def test_fetch_one_financial_fact_source_archive_fetch_error_handling() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.integrations.dart.xbrl import DartXbrlCollector

    identity = {
        "corp_code": "00000001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: the client-based archive fetch path is used and succeeds.
    class FakeClientArchiveOK:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            return b""

    collector_ok = DartXbrlCollector(api_key="k", client=FakeClientArchiveOK())
    page = collector_ok._fetch_one_financial_fact_source(identity)
    assert page["source_kind"] == "unavailable"

    # Given/When/Then: a PITDataError from the client archive fetch propagates as-is.
    class FakeClientArchiveRaisesPIT:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            raise PITDataError("archive boom")

    collector_pit = DartXbrlCollector(api_key="k", client=FakeClientArchiveRaisesPIT())
    with pytest.raises(PITDataError, match="archive boom"):
        collector_pit._fetch_one_financial_fact_source(identity)

    # Given/When/Then: a generic exception from the client archive fetch is wrapped.
    class FakeClientArchiveRaisesGeneric:
        def _request_validated(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"status": "013", "list": []}

        def fetch_document_archive(self, rcept_no: str) -> bytes:
            raise RuntimeError("archive network blip")

    collector_generic = DartXbrlCollector(api_key="k", client=FakeClientArchiveRaisesGeneric())
    with pytest.raises(PITDataError, match="F1"):
        collector_generic._fetch_one_financial_fact_source(identity)


def test_fetch_one_financial_fact_source_rejects_invalid_and_parses_valid_legacy_archive() -> None:
    import io
    import zipfile

    from src.integrations.dart.xbrl import DartXbrlCollector

    def make_legacy_archive(files: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content.encode("utf-8"))
        return buf.getvalue()

    identity = {
        "corp_code": "001",
        "filing_id": "F1",
        "rcept_no": "F1",
        "biz_year": "2020",
        "reprt_code": "11011",
        "fs_div": "CFS",
    }

    # Given/When/Then: a non-empty, non-zip archive is rejected explicitly.
    collector_bad_zip = DartXbrlCollector(
        api_key="k",
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: b"not-a-zip-archive",
    )
    bad_zip_page = collector_bad_zip._fetch_one_financial_fact_source(identity)
    assert bad_zip_page["diagnostics"] == ("invalid_document_archive",)

    # Given/When/Then: a valid zip with a well-formed two-account statement parses successfully.
    good_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>100</amount><unit>KRW</unit></account>"
        "<account><account_nm>\uc790\uc0b0\ucd1d\uacc4</account_nm><amount>1000</amount><unit>KRW</unit></account>"
        "</document>"
    )
    good_archive = make_legacy_archive({"F1.xml": good_xml})
    collector_good = DartXbrlCollector(
        api_key="k",
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: good_archive,
    )
    good_page = collector_good._fetch_one_financial_fact_source(identity)
    assert good_page["source_kind"] == "legacy_document"
    assert len(good_page["records"]) >= 1

    # Given/When/Then: a valid zip with an ambiguous (duplicate) statement fails extraction.
    ambiguous_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<document>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>100</amount><unit>KRW</unit></account>"
        "<account><account_nm>\ub9e4\ucd9c\uc561</account_nm><amount>200</amount><unit>KRW</unit></account>"
        "</document>"
    )
    ambiguous_archive = make_legacy_archive({"F1.xml": ambiguous_xml})
    collector_ambiguous = DartXbrlCollector(
        api_key="k",
        request_json=lambda _e, _p: {"status": "013", "list": []},
        request_bytes=lambda _e, _p: ambiguous_archive,
    )
    ambiguous_page = collector_ambiguous._fetch_one_financial_fact_source(identity)
    assert ambiguous_page["status"] == "extraction_failed"
    assert ambiguous_page["records"] == []

