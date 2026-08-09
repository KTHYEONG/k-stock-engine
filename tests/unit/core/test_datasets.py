"""Dataset manifest and provenance contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.datasets import (
    DatasetManifest,
    make_manifest,
    schema_hash,
    validate_dataset_manifest,
)
from src.core.instruments import AssetKind

FEATURE_SET = "stock_alpha_v1"
DECISION = datetime(2024, 3, 15, 8, 50, tzinfo=UTC)


def manifest() -> DatasetManifest:
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=["session_index", "date", "instrument_id", "close"],
        feature_set=FEATURE_SET,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 1, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="v1",
        row_count=10,
    )


class TestMakeManifest:
    def test_requires_explicit_provenance(self) -> None:
        with pytest.raises(TypeError):
            make_manifest(
                asset_kind=AssetKind.STOCK,
                columns=["a"],
                feature_set=FEATURE_SET,
                label_definition="x",
                label_horizon_sessions=1,
                time_start=datetime(2024, 1, 1, tzinfo=UTC),
                time_end=datetime(2024, 3, 1, tzinfo=UTC),
                row_count=1,
            )

    def test_hashes_schema_and_policy_version(self) -> None:
        m = manifest()
        assert m.schema_hash == schema_hash(["session_index", "date", "instrument_id", "close"])
        assert m.universe_policy_hash == schema_hash(["v1"])

    def test_rejects_non_asset_kind(self) -> None:
        with pytest.raises(ValueError, match="asset_kind"):
            DatasetManifest(
                asset_kind="OTHER",  # type: ignore[arg-type]
                schema_version="v1",
                schema_hash="h",
                provider_version="p",
                universe_policy_version="v1",
                universe_policy_hash="h",
                feature_set=FEATURE_SET,
                feature_set_hash="h",
                label_definition="x",
                label_horizon_sessions=1,
                time_start=datetime(2024, 1, 1, tzinfo=UTC),
                time_end=datetime(2024, 3, 1, tzinfo=UTC),
                generated_time=datetime(2024, 3, 2, tzinfo=UTC),
                row_count=1,
            )


class TestValidateDatasetManifest:
    def test_valid_manifest_passes(self) -> None:
        assert validate_dataset_manifest(manifest(), AssetKind.STOCK, FEATURE_SET, DECISION) is None

    def test_kind_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_kind"):
            validate_dataset_manifest(manifest(), AssetKind.ETF, FEATURE_SET, DECISION)

    def test_feature_set_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature_set"):
            validate_dataset_manifest(manifest(), AssetKind.STOCK, "etf_switch_v1", DECISION)

    def test_unavailable_rejected(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            validate_dataset_manifest(
                manifest(), AssetKind.STOCK, FEATURE_SET, datetime(2024, 2, 1, 8, 0, tzinfo=UTC)
            )
