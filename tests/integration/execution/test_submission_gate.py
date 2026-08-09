"""PLAN-06-LIVE-SUBMISSION-GATE: paper default, live rejected until gate passes."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, UTC

import pytest

from src.core.instruments import AssetKind
from src.execution.application.run_cycle import run_cycle
from src.execution.application.submission_gate import (
    LiveExecutionNotReadyError,
    ReadinessEvidence,
    SubmissionGate,
    SubmissionRequest,
)
from src.execution.domain.order import OrderSide, OrderState, TradeIntent
from src.execution.infrastructure.paper import InMemoryStateStore, PaperBroker


def make_intent(suffix: str = "a") -> TradeIntent:
    decision = datetime(2024, 6, 1, 8, 50, tzinfo=UTC)
    return TradeIntent(
        intent_id=f"intent-{suffix}",
        asset_kind=AssetKind.STOCK,
        instrument_id="KRX:005930",
        target_value=1_000_000.0,
        decision_time=decision,
        execution_time=decision + timedelta(minutes=15),
        strategy_id="stock_alpha_v1",
        reason="score-rank-policy",
        idempotency_key=f"stock_alpha_v1:005930:{decision.date().isoformat()}",
    )


class TestSubmissionGate:
    def test_paper_mode_requires_complete_intent_and_succeeds(self) -> None:
        gate = SubmissionGate(ReadinessEvidence())  # live evidence NOT satisfied
        gate.authorize(SubmissionRequest(intent=make_intent(), mode="paper"))
        # a second identical paper intent is rejected as duplicate
        with pytest.raises(ValueError, match="duplicate"):
            gate.authorize(SubmissionRequest(intent=make_intent(), mode="paper"))

    def test_live_submission_rejected_until_readiness_gate(self) -> None:
        gate = SubmissionGate(ReadinessEvidence())
        with pytest.raises(LiveExecutionNotReadyError):
            gate.authorize(SubmissionRequest(intent=make_intent("live"), mode="live"))

    def test_live_submission_allowed_after_complete_readiness(self) -> None:
        evidence = ReadinessEvidence(
            broker_reconciliation=True,
            order_state_transitions=True,
            paper_acceptance=True,
            idempotency_verified=True,
        )
        gate = SubmissionGate(evidence)
        gate.authorize(SubmissionRequest(intent=make_intent("live-ok"), mode="live"))

    def test_invalid_mode_rejected(self) -> None:
        gate = SubmissionGate()
        with pytest.raises(ValueError, match="mode"):
            gate.authorize(SubmissionRequest(intent=make_intent("bad"), mode="nope"))


class TestRunCycle:
    def test_run_cycle_emits_intents_in_paper_mode(self) -> None:
        gate = SubmissionGate(ReadinessEvidence())
        broker = PaperBroker()
        store = InMemoryStateStore()
        intents = [make_intent("x"), make_intent("y")]
        # unique idempotency keys required for the second intent
        intents[1] = replace(
            intents[0], intent_id="intent-y", idempotency_key="key-y"
        )
        records = run_cycle(intents, submission_gate=gate, broker=broker, state_store=store)
        assert len(records) == 2
        assert all(r.state is OrderState.FILLED for r in records)
        assert all(r.side is OrderSide.BUY for r in records)

    def test_run_cycle_rejects_duplicate_intent(self) -> None:
        gate = SubmissionGate()
        broker = PaperBroker()
        store = InMemoryStateStore()
        intent = make_intent()
        run_cycle([intent], submission_gate=gate, broker=broker, state_store=store)
        with pytest.raises(ValueError, match="duplicate"):
            run_cycle([intent], submission_gate=gate, broker=broker, state_store=store)

    def test_live_broker_not_enabled_in_run_cycle(self) -> None:
        # run_cycle hard-codes paper mode; no live path exists yet
        from src.execution.application.run_cycle import make_default_cycle

        gate, broker, store = make_default_cycle()
        assert isinstance(broker, PaperBroker)

    def test_validator_rejects_negative_value(self) -> None:
        # fail-closed: a non-positive target value cannot even be constructed
        with pytest.raises(ValueError, match="target_value"):
            replace(make_intent(), target_value=-5.0)
