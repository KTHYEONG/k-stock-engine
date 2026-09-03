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
