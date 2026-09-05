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
