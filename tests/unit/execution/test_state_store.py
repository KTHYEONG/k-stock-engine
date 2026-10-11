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


class TestSqliteStateStore:
    def test_conforms_to_port(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.execution.adapters.sqlite_state_store import SqliteStateStore

        store = SqliteStateStore(tmp_path / "state.db")
        try:
            assert isinstance(store, StateStorePort)
        finally:
            store.close()

    def test_idempotency_survives_restart(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.execution.adapters.sqlite_state_store import SqliteStateStore

        path = tmp_path / "state.db"
        first = SqliteStateStore(path)
        first.mark_intent("k1")
        first.close()
        second = SqliteStateStore(path)
        try:
            assert second.is_duplicate_intent("k1")
        finally:
            second.close()

    def test_illegal_transition_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import pytest
        from dataclasses import replace

        from src.execution.adapters.sqlite_state_store import SqliteStateStore
        from src.execution.domain.orders import OrderState

        store = SqliteStateStore(tmp_path / "state.db")
        try:
            filled = replace(_record(), state=OrderState.SUBMITTED, order_id="order:9")
            store.save(filled)
            bad = replace(filled, state=OrderState.FILLED)
            store.save(bad)
            with pytest.raises(ValueError, match="illegal"):
                store.save(replace(bad, state=OrderState.SUBMITTED))
            assert store.get("order:9") == bad
        finally:
            store.close()

    def test_history_is_append_only(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from dataclasses import replace

        from src.execution.adapters.sqlite_state_store import SqliteStateStore
        from src.execution.domain.orders import OrderState

        store = SqliteStateStore(tmp_path / "state.db")
        try:
            base = replace(_record(), order_id="order:2026-10-02:KRX:005930")
            store.save(base)
            store.save(replace(base, state=OrderState.VALIDATED))
            store.save(replace(base, state=OrderState.SUBMITTED))
            assert len(store.history(base.order_id)) == 2
            ordered = store.history(base.order_id)
            assert [r.state for r in ordered] == [OrderState.NEW, OrderState.VALIDATED]
        finally:
            store.close()
