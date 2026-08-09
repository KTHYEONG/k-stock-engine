"""Broker port contract tests (protocol shape + paper fill semantics)."""
from __future__ import annotations

from datetime import datetime, UTC

from src.core.instruments import AssetKind
from src.execution.adapters.paper_broker import PaperBroker
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState
from src.execution.ports.broker import BrokerPort


def _request(quantity: float = 10.0) -> OrderRequest:
    return OrderRequest(
        order_id="order:1",
        asset_kind=AssetKind.STOCK,
        instrument_id="KRX:005930",
        side=OrderSide.BUY,
        quantity=quantity,
        price=5000.0,
        request_time=datetime(2024, 6, 3, 1, 0, tzinfo=UTC),
        idempotency_key="key",
        intent_id="intent-a",
    )


class TestBrokerPort:
    def test_paper_broker_conforms_to_port(self) -> None:
        assert isinstance(PaperBroker(), BrokerPort)

    def test_paper_broker_fills_at_requested_price(self) -> None:
        broker = PaperBroker()
        record = broker.submit(_request(quantity=5.0))
        assert record.state is OrderState.FILLED
        assert record.filled_quantity == 5.0
        assert record.filled_price == 5000.0

    def test_reconcile_returns_own_account_only(self) -> None:
        broker = PaperBroker()
        broker.submit(_request())
        assert len(broker.reconcile("paper")) == 1
        assert broker.reconcile("other") == []
