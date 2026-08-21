"""Tests for trading decomposition and allocation.

Scenarios:
- ALLOCATION_06: AllocationDecision exposes candidate/ranked/selected counts
  and one typed cash reason without mutating StockRiskPolicy.
"""
from __future__ import annotations

from src.stocks.trading.policy import ExecutionUtility, SizingMethod, StockRiskPolicy
from src.stocks.trading.transitions import TransitionEvidence
from src.stocks.trading.allocation import AllocationDecision
from src.stocks.trading.portfolio_constructor import stock_risk_policy_fingerprint


class TestExecutionUtility:
    """ExecutionUtility is a semantic enum."""

    def test_all_values(self) -> None:
        assert ExecutionUtility.LEGACY_TARGET_INTERPOLATION == "legacy_target_interpolation_v1"
        assert ExecutionUtility.DELTA_COST_AWARE == "delta_cost_aware_v1"
        assert ExecutionUtility.SPARSE_HOLD_REPLACE == "sparse_hold_replace_v2"

    def test_member_count(self) -> None:
        assert len(ExecutionUtility) == 3


class TestSizingMethod:
    """SizingMethod is a semantic enum."""

    def test_all_values(self) -> None:
        assert SizingMethod.ALPHA_VOL_SQUARED == "alpha_vol_squared_v1"
        assert SizingMethod.RISK_BALANCED_WATERFILL == "risk_balanced_waterfill_v2"
        assert SizingMethod.CONFIDENCE_MEAN_VARIANCE == "confidence_mean_variance_v1"

    def test_member_count(self) -> None:
        assert len(SizingMethod) == 3


class TestStockRiskPolicyReExport:
    """StockRiskPolicy is re-exported from policy module."""

    def test_import_from_policy(self) -> None:
        from src.stocks.trading.policy import StockRiskPolicy as policy_type

        assert policy_type is StockRiskPolicy

    def test_default_policy(self) -> None:
        policy = StockRiskPolicy()
        assert policy.top_k == 20
        assert policy.gross_cap == 0.90
        assert policy.single_name_cap == 0.08


class TestTransitionEvidence:
    """TransitionEvidence captures transition counts."""

    def test_default_evidence(self) -> None:
        ev = TransitionEvidence()
        assert ev.retained_count == 0
        assert ev.entry_count == 0
        assert ev.exit_count == 0

    def test_evidence_with_values(self) -> None:
        ev = TransitionEvidence(
            retained_count=10,
            entry_count=5,
            exit_count=3,
            turnover_bps=150.0,
        )
        assert ev.retained_count == 10
        assert ev.turnover_bps == 150.0


class TestAllocationDecision:
    """AllocationDecision captures allocation output."""

    def test_default_decision(self) -> None:
        decision = AllocationDecision()
        assert decision.candidate_count == 0
        assert decision.selected_count == 0
        assert decision.cash_reason == ""

    def test_decision_with_values(self) -> None:
        decision = AllocationDecision(
            candidate_count=20,
            ranked_count=15,
            selected_count=10,
            cash_reason="target_met",
        )
        assert decision.candidate_count == 20
        assert decision.selected_count == 10
        assert decision.cash_reason == "target_met"


class TestPolicyFingerprint:
    """Policy fingerprint is deterministic."""

    def test_fingerprint_deterministic(self) -> None:
        policy = StockRiskPolicy()
        fp1 = stock_risk_policy_fingerprint(policy)
        fp2 = stock_risk_policy_fingerprint(policy)
        assert fp1 == fp2
        assert len(fp1) == 64
