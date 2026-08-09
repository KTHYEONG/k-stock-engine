"""Readiness gate contract tests."""
from __future__ import annotations

import pytest

from src.execution.application.readiness import (
    LiveExecutionNotReadyError,
    ReadinessEvidence,
    SubmissionGate,
    SubmissionRequest,
)
from tests.unit.execution.test_intents import make_intent


class TestReadinessGate:
    def test_paper_mode_succeeds_without_live_evidence(self) -> None:
        gate = SubmissionGate(ReadinessEvidence())
        gate.authorize(SubmissionRequest(intent=make_intent(), mode="paper"))

    def test_live_submission_rejected_until_readiness_complete(self) -> None:
        gate = SubmissionGate(ReadinessEvidence())
        with pytest.raises(LiveExecutionNotReadyError):
            gate.authorize(SubmissionRequest(intent=make_intent("live"), mode="live"))

    def test_duplicate_idempotency_key_rejected(self) -> None:
        gate = SubmissionGate()
        gate.authorize(SubmissionRequest(intent=make_intent(), mode="paper"))
        with pytest.raises(ValueError, match="duplicate"):
            gate.authorize(SubmissionRequest(intent=make_intent(), mode="paper"))

    def test_invalid_mode_rejected(self) -> None:
        gate = SubmissionGate()
        with pytest.raises(ValueError, match="mode"):
            gate.authorize(SubmissionRequest(intent=make_intent(), mode="nope"))
