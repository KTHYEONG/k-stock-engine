"""Paper broker and in-memory state store implementations."""
from __future__ import annotations

from src.execution.domain.order import OrderRequest, OrderState, OrderStateRecord


class PaperBroker:
    """Deterministic paper broker: fills at requested price (or 1.0 if none)."""

    def __init__(self, account: str = "paper"):
        self.account = account
        self._submitted: list[OrderStateRecord] = []

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
        if account != self.account:
            return []
        return list(self._submitted)


class InMemoryStateStore:
    """In-memory ``StateStorePort``; never touches a production SQLite file."""

    def __init__(self) -> None:
        self._records: dict[str, OrderStateRecord] = {}
        self._idempotency_keys: set[str] = set()

    def save(self, record: OrderStateRecord) -> None:
        self._records[record.order_id] = record

    def get(self, order_id: str) -> OrderStateRecord | None:
        return self._records.get(order_id)

    def mark_intent(self, idempotency_key: str) -> None:
        self._idempotency_keys.add(idempotency_key)

    def is_duplicate_intent(self, idempotency_key: str) -> bool:
        return idempotency_key in self._idempotency_keys


class BrokerSessionValidator:
    """Reject orders for instruments of a different asset kind or bad session."""

    def __init__(self, session_open: str = "09:00", session_close: str = "15:30"):
        self.session_open = session_open
        self.session_close = session_close
