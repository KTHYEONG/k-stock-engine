"""PLAN-06-LIVE-SUBMISSION-GATE: paper default, live rejected until gate passes."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, UTC

import pytest

from src.core.instruments import AssetKind
from src.execution.adapters.in_memory_state_store import InMemoryStateStore
from src.execution.adapters.paper_broker import ConstantPriceProvider, PaperBroker
from src.execution.application.readiness import (
    LiveExecutionNotReadyError,
    ReadinessEvidence,
    SubmissionGate,
    SubmissionRequest,
)
from src.execution.application.submit_intents import ExecutionContext, submit_intents
from src.execution.domain.intents import TradeIntent
from src.execution.domain.orders import OrderState
from src.execution.settings import DEFAULT_EXECUTION


def make_intent(suffix: str = "a") -> TradeIntent:
    # 10:00 KST == 01:00 UTC, safely inside the KRX_DAILY session
    decision = datetime(2024, 6, 3, 1, 0, tzinfo=UTC)
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
        account_snapshot_id="account-a",
    )


def paper_context(
    *, gate: SubmissionGate | None = None, store: InMemoryStateStore | None = None
) -> ExecutionContext:
    return ExecutionContext(
        settings=DEFAULT_EXECUTION,
        gate=gate or SubmissionGate(ReadinessEvidence()),
        broker=PaperBroker(),
        state_store=store or InMemoryStateStore(),
        price_provider=ConstantPriceProvider(5_000.0),
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


class TestSubmitIntents:
    def test_submit_intents_fills_in_paper_mode(self) -> None:
        store = InMemoryStateStore()
        gate = SubmissionGate(ReadinessEvidence())
        context = paper_context(gate=gate, store=store)
        intents = [make_intent("x"), replace(make_intent("y"), intent_id="intent-y", idempotency_key="key-y")]
        records = submit_intents(intents, context)
        assert len(records) == 2
        assert all(r.state is OrderState.FILLED for r in records)

    def test_order_quantity_is_value_over_price_not_raw_value(self) -> None:
        context = paper_context()
        intent = make_intent("q")
        records = submit_intents([intent], context)
        assert records[0].submitted_quantity == 1_000_000.0 / 5_000.0

    def test_submit_intents_rejects_duplicate_intent(self) -> None:
        store = InMemoryStateStore()
        context = paper_context(store=store)
        intent = make_intent()
        submit_intents([intent], context)
        with pytest.raises(ValueError, match="duplicate"):
            submit_intents([intent], context)

    def test_paper_is_the_only_default_mode(self) -> None:
        assert DEFAULT_EXECUTION.default_mode == "paper"

    def test_validator_rejects_negative_value(self) -> None:
        # fail-closed: a non-positive target value cannot even be constructed
        with pytest.raises(ValueError, match="target_value"):
            replace(make_intent(), target_value=-5.0)
