def test_krx_client_returns_only_object_records_from_injected_transport() -> None:
    from datetime import date

    from src.integrations.krx.client import KrxApiClient, KrxMarket

    calls: list[tuple[str, dict[str, str]]] = []

    def request(endpoint: str, params: dict[str, str]) -> dict[str, object]:
        calls.append((endpoint, params))
        return {'OutBlock_1': [{'ISU_SRT_CD': '005930'}, 'invalid']}

    client = KrxApiClient(api_key='key', request_json=request)
    records = client.fetch_master_records(as_of=date(2026, 1, 2), market=KrxMarket.KOSPI)

    assert records == [{'ISU_SRT_CD': '005930'}]
    assert calls == [('sto/stk_isu_base_info', {'basDd': '20260102'})]
