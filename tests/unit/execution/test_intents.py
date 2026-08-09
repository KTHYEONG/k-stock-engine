"""TradeIntent contract tests."""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

import pytest

from src.core.instruments import AssetKind
from src.execution.domain.intents import TradeIntent


def make_intent(suffix: str = "a", **overrides: object) -> TradeIntent:
    decision = datetime(2024, 6, 3, 1, 0, tzinfo=UTC)
    values: dict[str, object] = {
        "intent_id": f"intent-{suffix}",
        "asset_kind": AssetKind.STOCK,
        "instrument_id": "KRX:005930",
        "target_value": 1_000_000.0,
        "decision_time": decision,
        "execution_time": decision + timedelta(minutes=15),
        "strategy_id": "stock_alpha_v1",
        "reason": "score-rank-policy",
        "idempotency_key": f"stock_alpha_v1:005930:2024-06-03:{suffix}",
    }
    values.update(overrides)
    return TradeIntent(**values)


class TestTradeIntent:
    def test_module_of_record_is_intents(self) -> None:
        assert TradeIntent.__module__ == "src.execution.domain.intents"

    def test_rejects_non_positive_target_value(self) -> None:
        with pytest.raises(ValueError, match="target_value"):
            make_intent(target_value=0.0)
        with pytest.raises(ValueError, match="target_value"):
            make_intent(target_value=-5.0)

    def test_rejects_decision_after_execution(self) -> None:
        decision = datetime(2024, 6, 3, 1, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="decision_time"):
            make_intent(decision_time=decision, execution_time=decision - timedelta(minutes=1))
