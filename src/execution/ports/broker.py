"""Broker and state-store ports for the execution boundary."""
from __future__ import annotations

from typing import Protocol

from src.execution.domain.order import OrderRequest, OrderStateRecord


class BrokerPort(Protocol):
    """Abstract broker adapter. Paper mode is the default implementation."""

    def submit(self, request: OrderRequest) -> OrderStateRecord:
        """Submit a validated order and return its state."""
        ...

    def reconcile(self, account: str) -> list[OrderStateRecord]:
        """Return broker-confirmed order states for an account."""
        ...


class StateStorePort(Protocol):
    """Persistent order/state tracking without inventing price history."""

    def save(self, record: OrderStateRecord) -> None:
        ...

    def get(self, order_id: str) -> OrderStateRecord | None:
        ...

    def is_duplicate_intent(self, idempotency_key: str) -> bool:
        ...
