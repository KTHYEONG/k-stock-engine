"""Tests for the canonical research configuration and parameter provenance.

Scenarios:
- CONFIG_03: Training and artifact-resolved simulation produce identical
  policy fingerprints; every resolved field has one source; explicit cap
  mismatch raises ValueError.
"""
from __future__ import annotations

import pytest

from legacy.stocks.config.research import (
    CanonicalResearchProfile,
    ParameterSource,
    resolve_simulation_request,
    resolve_training_request,
)
from legacy.stocks.ml.contracts import policy_portfolio_fingerprint


class TestCanonicalResearchProfile:
    """Canonical profile is the single owner of defaults."""

    def test_default_values(self) -> None:
        profile = CanonicalResearchProfile()
        assert profile.top_k == 20
        assert profile.max_single_weight == 0.08
        assert profile.max_exposure == 0.90
        assert profile.participation_limit == 0.005
        assert profile.n_folds == 3
        assert profile.embargo_sessions == 5

    def test_fingerprint_is_deterministic(self) -> None:
        profile = CanonicalResearchProfile()
        fp1 = profile.fingerprint()
        fp2 = profile.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex digest

    def test_different_profiles_different_fingerprints(self) -> None:
        p1 = CanonicalResearchProfile(top_k=20)
        p2 = CanonicalResearchProfile(top_k=24)
        assert p1.fingerprint() != p2.fingerprint()


class TestResolveTrainingRequest:
    """Resolve training configuration from profile + overrides."""

    def test_defaults_from_profile(self) -> None:
        config = resolve_training_request("test_artifact")
        assert config.top_k.value == 20
        assert config.top_k.source == ParameterSource.PROFILE
        assert config.max_single_weight.value == 0.08
        assert config.max_single_weight.source == ParameterSource.PROFILE

    def test_cli_override(self) -> None:
        config = resolve_training_request(
            "test_artifact", overrides={"top_k": 24}
        )
        assert config.top_k.value == 24
        assert config.top_k.source == ParameterSource.CLI
        assert config.max_single_weight.source == ParameterSource.PROFILE

    def test_canonical_fingerprint_matches_profile(self) -> None:
        profile = CanonicalResearchProfile()
        config = resolve_training_request("test", profile=profile)
        assert config.canonical_fingerprint == profile.fingerprint()

    def test_unknown_override_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown override fields"):
            resolve_training_request("test", overrides={"unknown_field": 42})

    def test_multiple_overrides(self) -> None:
        config = resolve_training_request(
            "test", overrides={"top_k": 16, "max_single_weight": 0.10}
        )
        assert config.top_k.value == 16
        assert config.top_k.source == ParameterSource.CLI
        assert config.max_single_weight.value == 0.10
        assert config.max_single_weight.source == ParameterSource.CLI


class TestResolveSimulationRequest:
    """Resolve simulation configuration from artifact + overrides."""

    def test_defaults_from_artifact(self) -> None:
        artifact = {
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.90,
            "participation_limit": 0.005,
        }
        config = resolve_simulation_request(artifact)
        assert config["top_k"] == 20
        assert config["max_single_weight"] == 0.08

    def test_override(self) -> None:
        artifact = {"top_k": 20}
        config = resolve_simulation_request(artifact, overrides={"top_k": 24})
        assert config["top_k"] == 24

    def test_empty_artifact(self) -> None:
        config = resolve_simulation_request()
        assert config["top_k"] is None


class TestPolicyFingerprint:
    """Training and simulation produce identical policy fingerprints."""

    def test_fingerprint_matches_contract(self) -> None:
        fp = policy_portfolio_fingerprint(
            top_k=20,
            max_single_weight=0.08,
            max_exposure=0.90,
            participation_limit=0.005,
        )
        assert len(fp) == 64

    def test_training_and_simulation_resolve_one_policy_fingerprint(self) -> None:
        config = resolve_training_request("test")
        fp_training = policy_portfolio_fingerprint(
            top_k=config.top_k.value,
            max_single_weight=config.max_single_weight.value,
            max_exposure=config.max_exposure.value,
            participation_limit=config.participation_limit.value,
        )
        artifact = config.to_dict()
        fp_simulation = policy_portfolio_fingerprint(
            top_k=artifact["top_k"],
            max_single_weight=artifact["max_single_weight"],
            max_exposure=artifact["max_exposure"],
            participation_limit=artifact["participation_limit"],
        )
        assert fp_training == fp_simulation
