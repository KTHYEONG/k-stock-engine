def test_dart_client_rejects_non_success_api_status() -> None:
    from datetime import date

    import pytest

    from src.integrations.dart.client import DartApiClient, DartApiError

    client = DartApiClient(
        api_key='key',
        request_json=lambda endpoint, params: {'status': '020', 'message': 'blocked'},
    )

    with pytest.raises(DartApiError, match='020'):
        client.list_disclosures(start=date(2026, 1, 1), end=date(2026, 1, 2))


def test_list_disclosures_collects_all_pages_deduplicates_and_orders() -> None:
    from datetime import date
    from src.integrations.dart.client import DartApiClient

    pages = {"1": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000002", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}, "2": {"status": "000", "total_page": "2", "list": [{"rcept_no": "20150515000001", "rcept_dt": "20150515", "corp_code": "001", "corp_name": "A", "report_nm": "분기보고서 (2015.03)", "rm": ""}]}}
    client = DartApiClient(api_key="key", request_json=lambda _endpoint, p: pages[p["page_no"]])

    rows = client.list_disclosures(date(2015, 1, 1), date(2015, 12, 31))

    assert [row["rcept_no"] for row in rows] == ["20150515000001", "20150515000002"]


# test_dart_client_preserves_success_and_no_data_pages
def test_fetch_corporate_action_decisions_preserves_success_and_no_data_pages(monkeypatch) -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key='test-key')
    responses = iter([{'status': '000', 'list': [{'rcept_no': '20240102000001'}]}, {'status': '013', 'list': []}] * 3)
    monkeypatch.setattr(client, '_request_validated', lambda endpoint, params: next(responses))

    pages = client.fetch_corporate_action_decisions(corp_codes=['00123456'], start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert len(pages) == 5
    assert {page.status for page in pages} == {'000', '013'}
    assert pages[0].records == ({'rcept_no': '20240102000001'},)


def test_fetch_corporate_action_decisions_rejects_invalid_arguments() -> None:
    from datetime import date
    import pytest
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key")
    with pytest.raises(ValueError, match="start"):
        client.fetch_corporate_action_decisions(corp_codes=("1",), start=date(2024, 2, 1), end=date(2024, 1, 1))
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_corporate_action_decisions(corp_codes=(), start=date(2024, 1, 1), end=date(2024, 1, 2))


def test_fetch_dividend_disclosures_queries_every_report_code(monkeypatch) -> None:
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key")
    seen: list[dict[str, str]] = []

    def fake_request_validated(endpoint, params):
        seen.append(dict(params))
        if params["reprt_code"] == "11011":
            return {"status": "000", "list": [{"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"}]}
        return {"status": "013", "list": []}

    monkeypatch.setattr(client, "_request_validated", fake_request_validated)

    pages = client.fetch_dividend_disclosures(corp_codes=["00126380"], bsns_years=["2022"])

    assert len(pages) == 4
    assert {page.reprt_code for page in pages} == {"11011", "11012", "11013", "11014"}
    assert all(page.corp_code == "00126380" and page.bsns_year == "2022" for page in pages)
    annual = next(page for page in pages if page.reprt_code == "11011")
    assert annual.status == "000"
    assert annual.records == ({"se": "주당 현금배당금(원)", "stock_knd": "보통주", "thstrm": "1,444"},)
    assert {call["reprt_code"] for call in seen} == {"11011", "11012", "11013", "11014"}


def test_fetch_dividend_disclosures_rejects_invalid_arguments() -> None:
    import pytest
    from src.integrations.dart.client import DartApiClient

    client = DartApiClient(api_key="test-key")
    with pytest.raises(ValueError, match="corp_codes"):
        client.fetch_dividend_disclosures(corp_codes=(), bsns_years=("2022",))
    with pytest.raises(ValueError, match="bsns_years"):
        client.fetch_dividend_disclosures(corp_codes=("00126380",), bsns_years=())


from datetime import date, datetime

from src.core.time import KRX_TZ, SessionCalendar
from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind
from src.integrations.dart.lifecycle import DartLifecycleCollector

class FakeDart:
    def load_corp_code_records(self):
        return ()

def test_dart_lifecycle_collector_persists_unresolved_mapping_gap():
    session = datetime(2016, 1, 21, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate('KRX:003945', '003945', datetime(2016, 1, 20, 9, tzinfo=KRX_TZ), session, ('master-hash',))
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((candidate.last_tradable_session, session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.UNRESOLVED
    assert evidence.evidence_reason == 'missing_corp_code'
    assert evidence.cash_settlement_per_share is None


def test_dart_lifecycle_collector_resolves_cleanup_cash_settlement():
    from datetime import date, datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate
    from src.integrations.dart.lifecycle import DartLifecycleCollector
    from src.data.lifecycle import LifecycleResolutionKind
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("master-hash",))
    archive = "정리매매 기간은 2016.05.10.부터 2016.05.18.까지이며 2016.05.19.자로 상장폐지. 주당 10,200원.".encode()
    class FakeDart:
        def load_corp_code_records(self):
            from src.integrations.dart.client import DartCorpCodeRecord
            return (DartCorpCodeRecord(ticker="008020", corp_code="00101336", corp_name="경남에너지"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            assert corp_code == "00101336"
            return [{"rcept_no": "20160502800560", "rcept_dt": "20160502", "corp_code": corp_code, "corp_name": "경남에너지", "report_nm": "기타경영사항(자율공시)(상장폐지결정 및 정리매매)", "rm": "유"}]
        def fetch_document_archive(self, rcept_no):
            assert rcept_no == "20160502800560"
            return archive
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar(sessions), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.CASH_SETTLEMENT
    assert evidence.cash_settlement_per_share == 10200.0
    assert evidence.document_receipt_no == "20160502800560"


def test_dart_lifecycle_collector_unresolved_when_no_lifecycle_filings():
    from datetime import date, datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind
    from src.integrations.dart.lifecycle import DartLifecycleCollector
    session = datetime(2016, 4, 15, 9, tzinfo=KRX_TZ)
    candidate = LifecycleCandidate("KRX:051310", "051310", datetime(2016, 4, 14, 9, tzinfo=KRX_TZ), session, ("h",))
    class FakeDart:
        def load_corp_code_records(self):
            from src.integrations.dart.client import DartCorpCodeRecord
            return (DartCorpCodeRecord(ticker="051310", corp_code="00181934", corp_name="플랜텍"),)
        def list_disclosures(self, start, end, *, corp_code=None):
            return [{"rcept_no": "20160128800671", "rcept_dt": "20160128", "corp_code": corp_code, "corp_name": "플랜텍", "report_nm": "분기보고서", "rm": "유"}]
    evidence = DartLifecycleCollector(dart=FakeDart(), calendar=SessionCalendar((datetime(2016, 4, 14, 9, tzinfo=KRX_TZ), session)), coverage_start=date(2015, 1, 1)).collect(candidate)
    assert evidence.resolution_kind is LifecycleResolutionKind.UNRESOLVED
    assert evidence.evidence_reason == "missing_terms"
