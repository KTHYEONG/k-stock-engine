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
