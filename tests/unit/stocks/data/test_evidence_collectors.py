"""Tests for KRX and OpenDART evidence collectors."""
from __future__ import annotations

from datetime import UTC, date, datetime

from src.stocks.data.evidence_collectors import KRXEvidenceCollector, OpenDartEvidenceCollector


def test_krx_master_classification_uses_explicit_security_type() -> None:
    def request(endpoint: str, _params: dict[str, str]) -> dict:
        if endpoint.endswith("stk_isu_base_info"):
            return {
                "OutBlock_1": [
                    {
                        "ISU_SRT_CD": "005930",
                        "ISU_CD": "KR7005930003",
                        "ISU_ABBRV": "삼성전자",
                        "KIND_STKCERT_TP_NM": "보통주",
                        "LIST_DD": "19750611",
                    },
                    {
                        "ISU_SRT_CD": "0001A0",
                        "ISU_CD": "KR70001A0000",
                        "ISU_ABBRV": "특수상품",
                        "KIND_STKCERT_TP_NM": "신주인수권증서",
                        "LIST_DD": "20260305",
                    },
                ]
            }
        return {"OutBlock_1": []}

    collector = KRXEvidenceCollector(
        request_json=request,
        generated_time=datetime(2026, 3, 10, tzinfo=UTC),
    )
    snapshot = collector.collect_master_snapshot(date(2026, 3, 10))

    records = {record.source_identifier: record for record in snapshot.records}
    assert records["005930"].is_common_stock is True
    assert records["0001A0"].is_common_stock is False
    assert records["0001A0"].asset_type == "krx:신주인수권증서"


def test_krx_calendar_uses_only_dates_with_market_records() -> None:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        if endpoint.endswith("_bydd_trd"):
            return {"OutBlock_1": [{"ISU_CD": "005930"}]} if params["basDd"] == "20240102" else {"OutBlock_1": []}
        return {"OutBlock_1": []}

    collector = KRXEvidenceCollector(request_json=request, generated_time=datetime(2026, 1, 1, tzinfo=UTC))
    calendar = collector.collect_session_calendar(date(2024, 1, 1), date(2024, 1, 3))
    assert calendar.sessions == (date(2024, 1, 2),)


def test_dart_disclosures_preserve_receipt_identity_and_candidates() -> None:
    def request(endpoint: str, params: dict[str, str]) -> dict:
        assert endpoint == "list.json"
        if params["page_no"] == "1":
            return {
                "status": "000",
                "total_page": 1,
                "list": [
                    {
                        "rcept_no": "20240102000001",
                        "rcept_dt": "20240102",
                        "corp_code": "00126380",
                        "corp_name": "테스트",
                        "report_nm": "현금배당결정",
                        "rm": "",
                    }
                ],
            }
        return {"status": "000", "total_page": 1, "list": []}

    collector = OpenDartEvidenceCollector(
        api_key="fixture-key",
        request_json=request,
        generated_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    records = collector.collect_disclosures(date(2024, 1, 1), date(2024, 1, 3))
    candidates = collector.collect_corporate_action_candidates(date(2024, 1, 1), date(2024, 1, 3))
    assert records[0]["rcept_no"] == "20240102000001"
    assert records[0]["rcept_dt"] == "20240102"
    assert candidates[0]["verification_status"] == "candidate_only"
