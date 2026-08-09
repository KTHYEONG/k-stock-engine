"""State-store port contract tests."""
from __future__ import annotations

from src.execution.adapters.in_memory_state_store import InMemoryStateStore
from src.execution.ports.state_store import StateStorePort
from tests.unit.execution.test_orders import _record


class TestStateStorePort:
    def test_in_memory_store_conforms_to_port(self) -> None:
        assert isinstance(InMemoryStateStore(), StateStorePort)

    def test_save_and_get_round_trip(self) -> None:
        store = InMemoryStateStore()
        record = _record()
        store.save(record)
        assert store.get("order:1") == record
        assert store.get("missing") is None

    def test_idempotency_tracking(self) -> None:
        store = InMemoryStateStore()
        assert not store.is_duplicate_intent("key")
        store.mark_intent("key")
        assert store.is_duplicate_intent("key")
