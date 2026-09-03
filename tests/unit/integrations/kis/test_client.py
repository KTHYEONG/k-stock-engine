def test_kis_client_rejects_invalid_order_without_network() -> None:
    import pytest

    from src.integrations.kis.client import KisClient, KisCredentials

    client = KisClient(
        KisCredentials(
            app_key="key",
            app_secret="secret",
            account_no="12345678",
            account_product_code="01",
            env="demo",
        )
    )

    with pytest.raises(ValueError, match="qty"):
        client.place_order(symbol="005930", side="buy", qty=0)
