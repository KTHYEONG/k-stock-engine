"""KisMockBroker invariant tests."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.core.market_rules import KrxMarket, load_krx_market_rules
from src.config.runtime import load_runtime_config
from src.core.instruments import AssetKind
from src.execution.adapters.kis_broker import KisMockBroker, LiveTradingForbiddenError
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState
from src.integrations.kis.client import KisClient, KisCredentials


def _rules():  # type: ignore[no-untyped-def]
    try:
        return load_krx_market_rules(load_runtime_config().market_rules)
    except Exception:
        return load_krx_market_rules(Path("config/market/krx_market_rules.toml"))


def _client(env: str, transport: object | None = None) -> KisClient:
    creds = KisCredentials(
        app_key="mock-key",
        app_secret="mock-secret",
        account_no="12345678",
        account_product_code="01",
        env=env,
    )
    client = KisClient(creds)
    if transport is not None:
        client._call = transport  # type: ignore[attr-defined]
    return client


def _request(**overrides: object) -> OrderRequest:
    from datetime import UTC, datetime

    values: dict[str, object] = {
        "order_id": "order:2026-10-02:KRX:005930",
        "asset_kind": AssetKind.STOCK,
        "instrument_id": "KRX:005930",
        "side": OrderSide.BUY,
        "quantity": 10.0,
        "price": None,
        "request_time": datetime(2026, 10, 2, 0, 0, tzinfo=UTC),
        "idempotency_key": "k",
        "intent_id": "2026-10-02:KRX:005930",
    }
    values.update(overrides)
    return OrderRequest(**values)  # type: ignore[arg-type]


def test_live_credentials_refused() -> None:
    client = _client("real")
    with pytest.raises(LiveTradingForbiddenError):
        KisMockBroker(client, rules=_rules(), markets={"KRX:005930": KrxMarket.KOSPI}, session=date(2026, 10, 2))


@pytest.mark.parametrize("env", ["VPS", "vps"])
def test_live_endpoint_cannot_be_disguised_as_mock(env: str) -> None:
    client = _client(env)
    client.base_url = KisClient.REAL_BASE_URL
    with pytest.raises(LiveTradingForbiddenError):
        KisMockBroker(client, rules=_rules(), markets={}, session=date(2026, 10, 2))


def test_missing_success_status_propagates() -> None:
    client = _client("vps", transport=lambda **kwargs: {})
    broker = KisMockBroker(client, rules=_rules(), markets={"KRX:005930": KrxMarket.KOSPI}, session=date(2026, 10, 2))
    with pytest.raises(ValueError, match="malformed"):
        broker.submit(_request())


def test_mock_tr_ids_used() -> None:
    seen: list[dict[str, Any]] = []

    def fake_call(method: str, path: str, tr_id: str | None = None, **kwargs: object) -> dict[str, Any]:
        seen.append({"method": method, "path": path, "tr_id": tr_id, "kwargs": kwargs})
        return {"rt_cd": "0", "output": {"ODNO": "12345"}}

    client = _client("vps", transport=fake_call)
    broker = KisMockBroker(
        client, rules=_rules(), markets={"KRX:005930": KrxMarket.KOSPI, "KRX:000660": KrxMarket.KOSPI},
        session=date(2026, 10, 2),
    )
    broker.submit(_request(side=OrderSide.BUY))
    broker.submit(_request(order_id="order:2", instrument_id="KRX:000660", side=OrderSide.SELL, intent_id="i2"))
    tr_ids = [call["tr_id"] for call in seen]
    assert "VTTC0012U" in tr_ids
    assert "VTTC0011U" in tr_ids
    assert not any(str(tr).startswith("TTTC") for tr in tr_ids)
    bodies = [call["kwargs"]["body"] for call in seen]
    assert all(body["ORD_DVSN"] == "01" for body in bodies)


def test_broker_rejection_becomes_rejected_record() -> None:
    def fake_call(method: str, path: str, tr_id: str | None = None, **kwargs: object) -> dict[str, Any]:
        return {"rt_cd": "1", "msg1": "halted"}

    from src.integrations.errors import ProviderTerminalError

    def raising_call(method: str, path: str, tr_id: str | None = None, **kwargs: object) -> dict[str, Any]:
        raise ProviderTerminalError("KIS API error (1): halted", provider="KIS", endpoint="order-cash")

    client = _client("vps", transport=raising_call)
    broker = KisMockBroker(
        client, rules=_rules(), markets={"KRX:005930": KrxMarket.KOSPI}, session=date(2026, 10, 2)
    )
    record = broker.submit(_request())
    assert record.state is OrderState.REJECTED
    assert "halted" in record.reason
