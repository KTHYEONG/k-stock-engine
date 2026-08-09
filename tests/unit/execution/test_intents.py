"""TradeIntent contract tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.core.instruments import AssetKind
from src.execution.domain.intents import TradeIntent, read_v1_intent


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
        "account_snapshot_id": "account-a",
    }
    values.update(overrides)
    return TradeIntent(**values)


class TestTradeIntent:
    def test_module_of_record_is_intents(self) -> None:
        assert TradeIntent.__module__ == "src.execution.domain.intents"

    def test_zero_target_value_represents_full_exit(self) -> None:
        intent = make_intent(target_value=0.0)
        assert intent.target_value == 0.0

    def test_rejects_negative_target_value(self) -> None:
        with pytest.raises(ValueError, match="target_value"):
            make_intent(target_value=-5.0)

    def test_rejects_non_finite_target_value(self) -> None:
        with pytest.raises(ValueError, match="target_value"):
            make_intent(target_value=float("nan"))

    def test_rejects_empty_account_snapshot_id(self) -> None:
        with pytest.raises(ValueError, match="account_snapshot_id"):
            make_intent(account_snapshot_id="")

    def test_rejects_negative_price_guard(self) -> None:
        with pytest.raises(ValueError, match="reference_price_guard_bps"):
            make_intent(reference_price_guard_bps=-1.0)

    def test_rejects_decision_after_execution(self) -> None:
        decision = datetime(2024, 6, 3, 1, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="decision_time"):
            make_intent(decision_time=decision, execution_time=decision - timedelta(minutes=1))


class TestReadV1Intent:
    def test_migrates_positive_target_intent(self) -> None:
        intent = read_v1_intent(
            {
                "intent_id": "v1-intent",
                "asset_kind": "STOCK",
                "instrument_id": "KRX:005930",
                "target_value": 500_000.0,
                "decision_time": "2024-06-03T01:00:00+00:00",
                "execution_time": "2024-06-03T01:15:00+00:00",
                "strategy_id": "stock_alpha_v1",
                "reason": "score-rank-policy",
                "idempotency_key": "key-v1",
                "account_snapshot_id": "account-a",
            }
        )
        assert intent.target_value == 500_000.0
        assert intent.account_snapshot_id == "account-a"

    def test_migrated_exit_requires_account_snapshot(self) -> None:
        with pytest.raises(ValueError, match="account_snapshot_id"):
            read_v1_intent(
                {
                    "intent_id": "v1-exit",
                    "asset_kind": "STOCK",
                    "instrument_id": "KRX:005930",
                    "target_value": 0.0,
                    "decision_time": "2024-06-03T01:00:00+00:00",
                    "execution_time": "2024-06-03T01:15:00+00:00",
                    "strategy_id": "stock_alpha_v1",
                    "idempotency_key": "key-exit",
                }
            )
