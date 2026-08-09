"""Stock ML point-in-time dataset contracts.

The ``DatasetManifest`` type itself is a low-level shared primitive and lives in
``core``; this module owns the stock-side validation contract.
"""
from __future__ import annotations

from datetime import datetime

from src.core.instruments import AssetKind
from src.core.manifest import DatasetManifest, make_manifest, schema_hash

__all__ = ["DatasetManifest", "make_manifest", "schema_hash", "validate_dataset_manifest"]


def validate_dataset_manifest(
    manifest: DatasetManifest,
    expected_kind: AssetKind,
    expected_feature_set: str,
    decision_time: datetime,
) -> None:
    """Validate a manifest against the expectations of a consumer.

    Raises ``ValueError`` on any mismatch without materializing the underlying
    Parquet data.
    """
    if manifest.asset_kind is not expected_kind:
        raise ValueError(
            f"asset_kind mismatch: manifest has {manifest.asset_kind.value}, "
            f"expected {expected_kind.value}"
        )
    if manifest.feature_set != expected_feature_set:
        raise ValueError(
            f"feature_set mismatch: manifest has {manifest.feature_set!r}, "
            f"expected {expected_feature_set!r}"
        )
    if manifest.time_end > decision_time:
        raise ValueError(
            f"dataset not available at decision_time: dataset ends "
            f"{manifest.time_end.isoformat()}, decision at {decision_time.isoformat()}"
        )
    if not manifest.schema_hash:
        raise ValueError("manifest schema_hash must not be empty")
    if not manifest.universe_policy_hash:
        raise ValueError("manifest universe_policy_hash must not be empty")
    if manifest.label_horizon_sessions <= 0:
        raise ValueError("label_horizon_sessions must be positive")
