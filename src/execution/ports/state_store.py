"""State-store port for the execution boundary."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.execution.domain.orders import OrderStateRecord


@runtime_checkable
class StateStorePort(Protocol):
    """Persistent order/state tracking without inventing price history."""

    def save(self, record: OrderStateRecord) -> None:
        ...

    def get(self, order_id: str) -> OrderStateRecord | None:
        ...

    def is_duplicate_intent(self, idempotency_key: str) -> bool:
        ...

    def mark_intent(self, idempotency_key: str) -> None:
        ...
