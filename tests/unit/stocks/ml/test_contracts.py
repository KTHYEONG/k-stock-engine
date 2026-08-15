"""Net-alpha ML policy-profile contract tests."""
from __future__ import annotations

import pytest

from src.stocks.ml.contracts import (
    DEFAULT_POLICY_PROFILE_IDS,
    DEFAULT_POLICY_PROFILES,
    LOWER_BOUND_ONLY_PROFILE_ID,
    NetAlphaTrainingRequest,
    PolicyProfile,
    policy_portfolio_fingerprint,
    validate_policy_profiles,
)


def test_default_policy_profiles_are_the_two_pre_registered() -> None:
    assert tuple(p.profile_id for p in DEFAULT_POLICY_PROFILES) == (
        "legacy_overlay_5bps",
        "lower_bound_only",
    )
    assert tuple(p.no_trade_band_bps for p in DEFAULT_POLICY_PROFILES) == (5.0, 0.0)


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
            (PolicyProfile("legacy_overlay_5bps", 5.0), PolicyProfile("custom", 0.0))
        )


def test_validate_policy_profiles_rejects_empty_and_extra() -> None:
    with pytest.raises(ValueError, match="at least one profile"):
        validate_policy_profiles(())
    with pytest.raises(ValueError, match="exactly the two default profiles"):
        validate_policy_profiles((*DEFAULT_POLICY_PROFILES, PolicyProfile("extra", 1.0)))


def test_training_request_defaults_to_pre_registered_frontier() -> None:
    request = NetAlphaTrainingRequest(artifact_id="v1")
    assert tuple(p.profile_id for p in request.policy_profiles) == DEFAULT_POLICY_PROFILE_IDS


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
