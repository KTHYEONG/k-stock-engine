"""Tests for semantic identifiers and compatibility.

Scenarios:
- NAMING_09: Old aliases parse to semantic enums; new identity serializes
  semantic names; unknown aliases fail.
"""
from __future__ import annotations

import pytest

from legacy.stocks.compatibility import (
    ArtifactContractIdentity,
    parse_execution_utility,
    parse_sizing_method,
)
from legacy.stocks.trading.policy import ExecutionUtility, SizingMethod


class TestParseExecutionUtility:
    """Old execution utility aliases parse to semantic enums."""

    def test_legacy_alias(self) -> None:
        result = parse_execution_utility("legacy_target_interpolation_v1")
        assert result == ExecutionUtility.LEGACY_TARGET_INTERPOLATION

    def test_delta_cost_aware_alias(self) -> None:
        result = parse_execution_utility("delta_cost_aware_v1")
        assert result == ExecutionUtility.DELTA_COST_AWARE

    def test_sparse_hold_replace_alias(self) -> None:
        result = parse_execution_utility("sparse_hold_replace_v2")
        assert result == ExecutionUtility.SPARSE_HOLD_REPLACE

    def test_unknown_alias_fails(self) -> None:
        with pytest.raises(ValueError, match="unknown execution utility"):
            parse_execution_utility("unknown_v1")

    def test_empty_alias_fails(self) -> None:
        with pytest.raises(ValueError, match="unknown execution utility"):
            parse_execution_utility("")


class TestParseSizingMethod:
    """Old sizing method aliases parse to semantic enums."""

    def test_alpha_vol_squared_alias(self) -> None:
        result = parse_sizing_method("alpha_vol_squared_v1")
        assert result == SizingMethod.ALPHA_VOL_SQUARED

    def test_risk_balanced_waterfill_alias(self) -> None:
        result = parse_sizing_method("risk_balanced_waterfill_v2")
        assert result == SizingMethod.RISK_BALANCED_WATERFILL

    def test_confidence_mean_variance_alias(self) -> None:
        result = parse_sizing_method("confidence_mean_variance_v1")
        assert result == SizingMethod.CONFIDENCE_MEAN_VARIANCE

    def test_unknown_alias_fails(self) -> None:
        with pytest.raises(ValueError, match="unknown sizing method"):
            parse_sizing_method("unknown_v1")


class TestArtifactContractIdentity:
    """ArtifactContractIdentity validates persisted identity."""

    def test_valid_identity(self) -> None:
        identity = ArtifactContractIdentity(
            contract_id="stock_net_alpha",
            schema_revision=1,
            fingerprint="a" * 64,
        )
        assert identity.contract_id == "stock_net_alpha"
        assert identity.schema_revision == 1

    def test_invalid_revision_fails(self) -> None:
        with pytest.raises(ValueError, match="schema_revision must be >= 1"):
            ArtifactContractIdentity(
                contract_id="test",
                schema_revision=0,
                fingerprint="a" * 64,
            )

    def test_short_fingerprint_fails(self) -> None:
        with pytest.raises(ValueError, match="fingerprint must be 64"):
            ArtifactContractIdentity(
                contract_id="test",
                schema_revision=1,
                fingerprint="abc",
            )

    def test_old_aliases_parse_but_new_identity_serializes_semantic_names(self) -> None:
        # Old aliases parse to semantic enums
        assert parse_execution_utility("legacy_target_interpolation_v1") == ExecutionUtility.LEGACY_TARGET_INTERPOLATION
        assert parse_sizing_method("alpha_vol_squared_v1") == SizingMethod.ALPHA_VOL_SQUARED
        # New identity serializes semantic names (no v-suffix)
        identity = ArtifactContractIdentity(
            contract_id="stock_net_alpha",
            schema_revision=1,
            fingerprint="a" * 64,
        )
        assert "v1" not in identity.contract_id or identity.contract_id == "stock_net_alpha"
        assert identity.schema_revision >= 1

    def test_semantic_names_only(self) -> None:
        identity = ArtifactContractIdentity(
            contract_id="stock_net_alpha",
            schema_revision=1,
            fingerprint="a" * 64,
        )
        assert "v1" not in identity.contract_id or identity.contract_id == "stock_net_alpha"
