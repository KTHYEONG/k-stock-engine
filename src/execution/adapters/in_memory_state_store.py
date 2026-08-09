"""In-memory state store adapter; never touches a production SQLite file."""
from __future__ import annotations

from src.execution.domain.orders import OrderStateRecord


class InMemoryStateStore:
    """In-memory ``StateStorePort`` for tests and paper mode."""

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
