"""submit_intents application contract tests."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.execution.adapters.in_memory_state_store import InMemoryStateStore
from src.execution.adapters.paper_broker import ConstantPriceProvider, PaperBroker
from src.execution.application.readiness import SubmissionGate
from src.execution.application.submit_intents import ExecutionContext, submit_intents
from src.execution.domain.orders import OrderState
from src.execution.settings import DEFAULT_EXECUTION
from tests.unit.execution.test_intents import make_intent


def paper_context(*, store: InMemoryStateStore | None = None) -> ExecutionContext:
    return ExecutionContext(
        settings=DEFAULT_EXECUTION,
        gate=SubmissionGate(),
        broker=PaperBroker(),
        state_store=store or InMemoryStateStore(),
        price_provider=ConstantPriceProvider(5_000.0),
    )


class TestSubmitIntents:
    def test_submit_intents_fills_in_paper_mode(self) -> None:
        store = InMemoryStateStore()
        context = paper_context(store=store)
        intents = [make_intent("x"), replace(make_intent("y"), intent_id="intent-y", idempotency_key="key-y")]
        records = submit_intents(intents, context)
        assert len(records) == 2
        assert all(r.state is OrderState.FILLED for r in records)

    def test_order_quantity_is_value_over_price_not_raw_value(self) -> None:
        context = paper_context()
        records = submit_intents([make_intent("q")], context)
        assert records[0].submitted_quantity == 1_000_000.0 / 5_000.0

    def test_duplicate_intent_rejected(self) -> None:
        store = InMemoryStateStore()
        context = paper_context(store=store)
        intent = make_intent()
        submit_intents([intent], context)
        with pytest.raises(ValueError, match="duplicate"):
            submit_intents([intent], context)
