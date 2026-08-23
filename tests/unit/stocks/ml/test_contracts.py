"""Net-alpha ML policy-profile contract tests."""
from __future__ import annotations

import pytest

from src.stocks.ml.contracts import (
    DEFAULT_POLICY_PROFILES,
    LOWER_BOUND_ONLY_PROFILE_ID,
    NetAlphaTrainingRequest,
    PolicyProfile,
    policy_portfolio_fingerprint,
    validate_policy_profiles,
)


def test_default_policy_profiles_are_the_three_pre_registered() -> None:
    assert tuple(p.profile_id for p in DEFAULT_POLICY_PROFILES) == (
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
    )
    assert tuple(p.no_trade_band_bps for p in DEFAULT_POLICY_PROFILES) == (5.0, 0.0, 0.0)
    assert tuple(p.growth_risk_aversion for p in DEFAULT_POLICY_PROFILES) == (1.0, 1.0, 2.0)


def test_default_policy_profiles_pin_sparse_frontier() -> None:
    """SPARSE_GROWTH_01_DEFAULT_FRONTIER: first two profiles pin sparse modes, third uses confidence sizing."""
    assert tuple(p.execution_utility_mode for p in DEFAULT_POLICY_PROFILES) == (
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
    )
    assert tuple(p.sizing_mode for p in DEFAULT_POLICY_PROFILES) == (
        "risk_balanced_waterfill_v2",
        "risk_balanced_waterfill_v2",
        "confidence_mean_variance_v1",
    )
    validated = validate_policy_profiles(DEFAULT_POLICY_PROFILES)
    assert tuple(p.execution_utility_mode for p in validated) == (
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
    )
    assert tuple(p.sizing_mode for p in validated) == (
        "risk_balanced_waterfill_v2",
        "risk_balanced_waterfill_v2",
        "confidence_mean_variance_v1",
    )


def test_policy_profile_validates_input_range() -> None:
    with pytest.raises(ValueError, match="profile_id must be non-empty"):
        PolicyProfile(profile_id="", no_trade_band_bps=0.0)
    with pytest.raises(ValueError, match="no_trade_band_bps must be a finite non-negative"):
        PolicyProfile(profile_id="x", no_trade_band_bps=-1.0)
    with pytest.raises(ValueError, match="no_trade_band_bps must be a finite non-negative"):
        PolicyProfile(profile_id="x", no_trade_band_bps=float("nan"))
    profile = PolicyProfile(profile_id="lower_bound_only", no_trade_band_bps=0.0)
    assert profile.profile_id == LOWER_BOUND_ONLY_PROFILE_ID


def test_validate_policy_profiles_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="profile ids must be unique"):
        validate_policy_profiles(
            (
                PolicyProfile("legacy_overlay_5bps", 5.0),
                PolicyProfile("legacy_overlay_5bps", 5.0),
            )
        )


def test_validate_policy_profiles_rejects_missing_default() -> None:
    with pytest.raises(ValueError, match="default policy profile"):
        validate_policy_profiles(
            (PolicyProfile("legacy_overlay_5bps", 5.0), PolicyProfile("lower_bound_only", 0.0))
        )


def test_validate_policy_profiles_rejects_non_positive_aversion() -> None:
    with pytest.raises(ValueError, match="growth_risk_aversion"):
        PolicyProfile(profile_id="x", no_trade_band_bps=0.0, growth_risk_aversion=0.0)
    with pytest.raises(ValueError, match="growth_risk_aversion"):
        PolicyProfile(profile_id="x", no_trade_band_bps=0.0, growth_risk_aversion=-1.0)
    with pytest.raises(ValueError, match="growth_risk_aversion"):
        PolicyProfile(profile_id="x", no_trade_band_bps=0.0, growth_risk_aversion=float("nan"))


def test_validate_policy_profiles_rejects_empty_and_extra() -> None:
    with pytest.raises(ValueError, match="at least one profile"):
        validate_policy_profiles(())
    with pytest.raises(ValueError, match="exactly the three default profiles"):
        validate_policy_profiles((*DEFAULT_POLICY_PROFILES, PolicyProfile("extra", 1.0)))


def test_training_request_defaults_to_pre_registered_frontier() -> None:
    request = NetAlphaTrainingRequest(artifact_id="v1")
    assert tuple(p.profile_id for p in request.policy_profiles) == tuple(
        p.profile_id for p in DEFAULT_POLICY_PROFILES
    )


def test_training_request_rejects_divergent_frontier() -> None:
    with pytest.raises(ValueError, match="default policy profile"):
        NetAlphaTrainingRequest(
            artifact_id="v1",
            policy_profiles=(PolicyProfile("custom", 0.0),),
        )
    with pytest.raises(ValueError, match="profile ids must be unique"):
        NetAlphaTrainingRequest(
            artifact_id="v1",
            policy_profiles=(
                PolicyProfile("legacy_overlay_5bps", 5.0),
                PolicyProfile("legacy_overlay_5bps", 5.0),
            ),
        )


def test_policy_portfolio_fingerprint_is_deterministic_and_sensitive() -> None:
    base = policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005)
    assert base == policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005)
    assert base != policy_portfolio_fingerprint(5, 0.08, 0.9, 0.005)
    assert base != policy_portfolio_fingerprint(20, 0.2, 0.9, 0.005)
    assert base != policy_portfolio_fingerprint(20, 0.08, 1.0, 0.005)
    assert base != policy_portfolio_fingerprint(20, 0.08, 0.9, 0.01)

def test_outcome_status_vocabulary_is_fixed_and_validated() -> None:
    from src.stocks.ml.contracts import (
        OUTCOME_PARTIAL_TAIL,
        OUTCOME_REALIZED,
        OUTCOME_STATUS_VOCABULARY,
        RESOLVED_OUTCOME_STATUSES,
        validate_outcome_status,
    )

    assert OUTCOME_REALIZED in OUTCOME_STATUS_VOCABULARY
    assert OUTCOME_PARTIAL_TAIL in OUTCOME_STATUS_VOCABULARY
    assert len(OUTCOME_STATUS_VOCABULARY) >= 9
    assert RESOLVED_OUTCOME_STATUSES == (OUTCOME_REALIZED,)
    assert validate_outcome_status(OUTCOME_REALIZED) == OUTCOME_REALIZED
    assert validate_outcome_status(OUTCOME_PARTIAL_TAIL) == OUTCOME_PARTIAL_TAIL
    with pytest.raises(ValueError, match="unknown outcome status"):
        validate_outcome_status("FABRICATED_STATE")
    with pytest.raises(ValueError, match="non-empty string"):
        validate_outcome_status("")


def test_outcome_status_counts_bounded_record() -> None:
    from src.stocks.ml.contracts import (
        OUTCOME_MISSING_EXIT_PRICE,
        OUTCOME_PARTIAL_TAIL,
        OUTCOME_REALIZED,
        OutcomeStatusCounts,
    )

    counts = OutcomeStatusCounts.from_mapping(
        {OUTCOME_REALIZED: 5, OUTCOME_MISSING_EXIT_PRICE: 3, OUTCOME_PARTIAL_TAIL: 1}
    )
    assert counts.realized == 5
    assert counts.partial_tail == 1
    assert counts.unresolved == 3
    assert counts.count(OUTCOME_REALIZED) == 5
    assert counts.to_json() == {
        OUTCOME_MISSING_EXIT_PRICE: 3,
        OUTCOME_PARTIAL_TAIL: 1,
        OUTCOME_REALIZED: 5,
    }
    with pytest.raises(ValueError, match="unknown outcome status"):
        OutcomeStatusCounts.from_mapping({"FABRICATED_STATE": 1})
    with pytest.raises(ValueError, match="non-negative int"):
        OutcomeStatusCounts.from_mapping({OUTCOME_REALIZED: -1})


def test_horizon_join_evidence_records_decision_realized_status() -> None:
    from src.stocks.ml.contracts import (
        OUTCOME_REALIZED,
        HorizonJoinEvidence,
        OutcomeStatusCounts,
    )

    evidence = HorizonJoinEvidence(
        horizon_sessions=5,
        feature_rows=100,
        label_rows=80,
        joined_rows=80,
        decision_rows=100,
        realized_rows=80,
        status_counts=OutcomeStatusCounts.from_mapping({OUTCOME_REALIZED: 80}),
    )
    assert evidence.decision_rows == 100
    assert evidence.realized_rows == 80
    assert evidence.status_counts is not None
    assert evidence.status_counts.realized == 80


def test_three_profile_frontier() -> None:
    """CGRA-04-three-profile-frontier"""
    from src.stocks.ml.contracts import (
        DEFAULT_POLICY_PROFILES,
        LOWER_BOUND_HALF_KELLY_PROFILE_ID,
        validate_policy_profiles,
    )

    assert LOWER_BOUND_HALF_KELLY_PROFILE_ID == "lower_bound_half_kelly"
    assert len(DEFAULT_POLICY_PROFILES) == 3
    ids = [p.profile_id for p in DEFAULT_POLICY_PROFILES]
    bands = [p.no_trade_band_bps for p in DEFAULT_POLICY_PROFILES]
    aversions = [p.growth_risk_aversion for p in DEFAULT_POLICY_PROFILES]
    assert ids == [
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
    ]
    assert bands == [5.0, 0.0, 0.0]
    assert aversions == [1.0, 1.0, 2.0]
    validated = validate_policy_profiles(DEFAULT_POLICY_PROFILES)
    assert len(validated) == 3


def test_default_policy_profiles_confidence_frontier() -> None:
    """ML_CONFIDENCE_FRONTIER_01_DEFAULT_PROFILE_COMPOSITION."""
    from src.stocks.ml.contracts import (
        DEFAULT_POLICY_PROFILES,
        validate_policy_profiles,
    )

    ids = [p.profile_id for p in DEFAULT_POLICY_PROFILES]
    assert ids == [
        "legacy_overlay_5bps",
        "lower_bound_only",
        "lower_bound_half_kelly",
    ]
    assert [p.execution_utility_mode for p in DEFAULT_POLICY_PROFILES] == [
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
        "sparse_hold_replace_v2",
    ]
    assert [p.sizing_mode for p in DEFAULT_POLICY_PROFILES] == [
        "risk_balanced_waterfill_v2",
        "risk_balanced_waterfill_v2",
        "confidence_mean_variance_v1",
    ]
    half_kelly = DEFAULT_POLICY_PROFILES[2]
    assert half_kelly.no_trade_band_bps == 0.0
    assert half_kelly.growth_risk_aversion == 2.0
    validated = validate_policy_profiles(DEFAULT_POLICY_PROFILES)
    assert validated[2].sizing_mode == "confidence_mean_variance_v1"


def test_discovery_model_family_must_be_declared() -> None:
    """SCENARIO_DISCOVERY_FAMILY_VALIDATION."""
    from src.stocks.ml.contracts import ELASTIC_NET_FAMILY

    with pytest.raises(ValueError, match="discovery_model_family"):
        NetAlphaTrainingRequest(artifact_id="v1", discovery_model_family="not-a-family")
    request = NetAlphaTrainingRequest(artifact_id="v1")
    assert request.discovery_model_family == ELASTIC_NET_FAMILY
