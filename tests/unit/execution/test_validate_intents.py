"""Intent validation contract tests."""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

import pytest

from src.execution.adapters.in_memory_state_store import InMemoryStateStore
from src.execution.application.validate_intents import IntentValidator
from tests.unit.execution.test_intents import make_intent


class TestIntentValidator:
    def test_valid_intent_in_session_passes(self) -> None:
        store = InMemoryStateStore()
        validator = IntentValidator(store)
        intent = make_intent()  # 10:00 KST, inside KRX_DAILY
        validator.validate(intent, datetime(2024, 6, 3, 1, 0, tzinfo=UTC))

    def test_duplicate_intent_rejected(self) -> None:
        store = InMemoryStateStore()
        validator = IntentValidator(store)
        intent = make_intent()
        store.mark_intent(intent.idempotency_key)
        with pytest.raises(ValueError, match="duplicate"):
            validator.validate(intent, datetime(2024, 6, 3, 1, 0, tzinfo=UTC))

    def test_execution_outside_session_rejected(self) -> None:
        store = InMemoryStateStore()
        validator = IntentValidator(store)
        decision = datetime(2024, 6, 3, 23, 0, tzinfo=UTC)  # 08:00 KST next day
        intent = make_intent(decision_time=decision, execution_time=decision + timedelta(minutes=15))
        with pytest.raises(ValueError, match="session"):
            validator.validate(intent, datetime(2024, 6, 3, 23, 0, tzinfo=UTC))
