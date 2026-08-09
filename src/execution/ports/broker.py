"""Broker port for the execution boundary."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.execution.domain.orders import OrderRequest, OrderStateRecord


@runtime_checkable
class BrokerPort(Protocol):
    """Abstract broker adapter. Paper mode is the default implementation."""

    @property
    def account(self) -> str: ...

    def submit(self, request: OrderRequest) -> OrderStateRecord:
        """Submit a validated order and return its state."""
        ...

    def reconcile(self, account: str) -> list[OrderStateRecord]:
        """Return broker-confirmed order states for an account."""
        ...


class PriceProvider(Protocol):
    """Quotes a reference price used to size intents into share quantities."""

    def price_of(self, instrument_id: str) -> float:
        ...
