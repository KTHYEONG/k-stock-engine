"""Order state contract tests."""
from __future__ import annotations

from datetime import datetime, UTC

import pytest

from src.core.instruments import AssetKind
from src.execution.domain.orders import OrderRequest, OrderSide, OrderState, OrderStateRecord


class TestOrderRequest:
    def test_rejects_non_positive_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            _request(quantity=0.0)

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(ValueError, match="price"):
            _request(price=0.0)


class TestOrderStateRecord:
    def test_record_defaults_to_new_with_no_fill(self) -> None:
        record = _record()
        assert record.state is OrderState.NEW
        assert record.filled_quantity == 0.0
        assert record.filled_price is None


def _request(**overrides: object) -> OrderRequest:
    values: dict[str, object] = {
        "order_id": "order:1",
        "asset_kind": AssetKind.STOCK,
        "instrument_id": "KRX:005930",
        "side": OrderSide.BUY,
        "quantity": 10.0,
        "price": 5000.0,
        "request_time": datetime(2024, 6, 3, 1, 0, tzinfo=UTC),
        "idempotency_key": "key",
        "intent_id": "intent-a",
    }
    values.update(overrides)
    return OrderRequest(**values)


def _record() -> OrderStateRecord:
    return OrderStateRecord(
        order_id="order:1",
        intent_id="intent-a",
        instrument_id="KRX:005930",
        asset_kind=AssetKind.STOCK,
        side=OrderSide.BUY,
        state=OrderState.NEW,
        submitted_quantity=10.0,
    )
