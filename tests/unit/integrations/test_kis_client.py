def test_kis_client_rejects_invalid_order_without_network() -> None:
    import pytest

    from src.integrations.kis.client import KisClient, KisCredentials

    credentials = KisCredentials(
        app_key='key',
        app_secret='secret',
        account_no='12345678',
        account_product_code='01',
        env='demo',
    )
    client = KisClient(credentials)

    with pytest.raises(ValueError, match='qty'):
        client.place_order(symbol='005930', side='buy', qty=0)


def test_kis_token_cache_rejects_insecure_file(tmp_path) -> None:
    from datetime import datetime, timedelta

    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(KisCredentials('key', 'secret', '12345678', '01', env='demo'))
    client._token_cache_path = tmp_path / 'token.json'
    client._token_cache_path.write_text(
        '{"access_token": "cached", "expire_at": "'
        + (datetime.now() + timedelta(hours=1)).isoformat()
        + '"}',
        encoding='utf-8',
    )
    client._token_cache_path.chmod(0o644)

    client._load_cached_token()

    assert client._access_token is None


def _search_client(monkeypatch=None, response=None):
    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(KisCredentials('key', 'secret', '12345678', '01', env='demo'))
    calls: list[dict] = []
    payload = {'output': response} if response is not None else {'output': {'std_idst_clsf_cd': '032604'}}

    def fake_call(*, method, path, tr_id=None, params=None, body=None, include_auth=True, use_hashkey=False):
        calls.append({'method': method, 'path': path, 'tr_id': tr_id, 'params': dict(params or {})})
        return dict(payload)

    client._call = fake_call  # type: ignore[method-assign]
    return client, calls


def test_search_stock_info_sends_documented_request() -> None:
    raw = {'std_idst_clsf_cd': '032604', 'std_idst_clsf_cd_name': '통신 및 방송 장비 제조업'}
    client, calls = _search_client(response=raw)

    assert client.search_stock_info('005930') == raw
    assert len(calls) == 1
    assert calls[0]['method'] == 'GET'
    assert calls[0]['path'] == '/uapi/domestic-stock/v1/quotations/search-stock-info'
    assert calls[0]['tr_id'] == 'CTPF1002R'
    assert calls[0]['params'] == {'PRDT_TYPE_CD': '300', 'PDNO': '005930'}


def test_search_stock_info_rejects_blank_symbol() -> None:
    import pytest

    client, calls = _search_client(response={})
    with pytest.raises(ValueError, match='symbol is required'):
        client.search_stock_info('   ')
    assert calls == []


def test_search_stock_info_rejects_malformed_output() -> None:
    import pytest

    client, _ = _search_client(response=['not-a-dict'])
    with pytest.raises(RuntimeError, match='search-stock-info'):
        client.search_stock_info('005930')
