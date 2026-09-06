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


def test_krx_client_normalizes_quoted_key() -> None:
    from src.integrations.krx.client import KrxApiClient

    client = KrxApiClient(api_key='"key"', request_json=lambda *_: {})
    assert client.api_key == 'key'


def test_krx_client_backoffs_on_rate_limit(monkeypatch) -> None:
    from types import SimpleNamespace
    from src.integrations.krx.client import KrxApiClient

    responses = iter((SimpleNamespace(status_code=429, headers={'Retry-After': '2'}), SimpleNamespace(status_code=200, headers={}, json=lambda: {'ok': True})))
    sleeps: list[float] = []
    client = KrxApiClient(api_key='key', request_json=lambda *_: {})
    client._session = SimpleNamespace(get=lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr('src.integrations.krx.client.time.sleep', lambda seconds: sleeps.append(seconds))

    assert client._request('sto/stk_bydd_trd', {'basDd': '20240102'}) == {'ok': True}
    assert 2.0 in sleeps

    responses = iter((SimpleNamespace(status_code=429, headers={'Retry-After': 'invalid'}), SimpleNamespace(status_code=200, headers={}, json=lambda: {'ok': True})))
    client._session = SimpleNamespace(get=lambda *_args, **_kwargs: next(responses))
    assert client._request('sto/stk_bydd_trd', {'basDd': '20240102'}) == {'ok': True}


def test_krx_429_blocks_endpoint_without_extra_transport(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace
    import pytest
    from src.integrations.krx.client import KrxApiClient, KrxApiError
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    calls: list[object] = []
    client = KrxApiClient(api_key='key', request_json=None, quota_store=ProviderQuotaStateStore(tmp_path), now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    client._session = SimpleNamespace(get=lambda *_args, **_kwargs: calls.append(object()) or SimpleNamespace(status_code=429, headers={}, json=lambda: {}))
    monkeypatch.setattr('src.integrations.krx.client.time.sleep', lambda _seconds: None)
    with pytest.raises(KrxApiError, match='429'):
        client._request('sto/stk_bydd_trd', {'basDd': '20240102'})
    with pytest.raises(ProviderQuotaBlocked):
        client._request('sto/stk_bydd_trd', {'basDd': '20240103'})
    assert len(calls) == 1
