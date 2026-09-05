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
    )
    assert df.height == 2
    assert set(df["fact"].to_list()) == {"sales", "operating_profit"}
    assert df["company_id"].to_list() == ["00126380", "00126380"]
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
