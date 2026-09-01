def test_dart_client_transport_module_is_importable() -> None:
    from datetime import date

    from src.integrations.dart.client import DartApiClient, DartApiError

    calls: list[tuple[str, dict[str, str]]] = []

    def request(endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append((endpoint, params))
        return {
            "status": "000",
            "list": [{"rcept_no": "202601020001", "rcept_dt": "20260102"}],
        }

    client = DartApiClient(api_key="key", request_json=request)
    records = client.list_disclosures(start=date(2026, 1, 2), end=date(2026, 1, 2))

    assert records == [
        {
            "rcept_no": "202601020001",
            "rcept_dt": "20260102",
            "corp_code": "",
            "corp_name": "",
            "report_nm": "",
            "rm": "",
        }
    ]
    assert calls[0][0] == "list.json"
    assert calls[0][1]["crtfc_key"] == "key"
    assert issubclass(DartApiError, RuntimeError)
