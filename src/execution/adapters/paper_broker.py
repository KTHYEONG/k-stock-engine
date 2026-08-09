"""Paper broker and paper price provider adapters."""
from __future__ import annotations

from src.execution.domain.orders import OrderRequest, OrderState, OrderStateRecord


class PaperBroker:
    """Deterministic paper broker: fills at the requested price (1.0 if none)."""

    def __init__(self, account: str = "paper"):
        self._account = account
        self._submitted: list[OrderStateRecord] = []

    @property
    def account(self) -> str:
        return self._account

    def submit(self, request: OrderRequest) -> OrderStateRecord:
        record = OrderStateRecord(
            order_id=request.order_id,
            intent_id=request.intent_id,
            instrument_id=request.instrument_id,
            asset_kind=request.asset_kind,
            side=request.side,
            state=OrderState.FILLED,
            submitted_quantity=request.quantity,
            filled_quantity=request.quantity,
            filled_price=request.price if request.price is not None else 1.0,
            reason="paper-fill",
        )
        self._submitted.append(record)
        return record

    def reconcile(self, account: str) -> list[OrderStateRecord]:
        if account != self._account:
            return []
        return list(self._submitted)


class ConstantPriceProvider:
    """Reference price provider returning a fixed price per instrument."""

    def __init__(self, price: float = 1.0):
        if price <= 0:
            raise ValueError("price must be positive")
        self._price = price

    def price_of(self, instrument_id: str) -> float:
        return self._price
