"""Shared dataset manifest and provenance contracts.

``core`` is intentionally small and cannot import ``stocks``, ``etfs``, or a
provider library, so the manifest lives here and is reused by both asset
subsystems. The generic storage adapter validates manifests with the same
contract before touching Parquet.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from src.core.instruments import AssetKind


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Metadata contract written alongside a curated Parquet dataset.

    Consumers validate the manifest before reading Parquet. Mixing stock and
    ETF rows into one artifact, feature set, or model request is invalid.
    """

    asset_kind: AssetKind
    schema_version: str
    schema_hash: str
    provider_version: str
    universe_policy_version: str
    universe_policy_hash: str
    feature_set: str
    feature_set_hash: str
    label_definition: str
    label_horizon_sessions: int
    time_start: datetime
    time_end: datetime
    generated_time: datetime
    row_count: int

    def __post_init__(self) -> None:
        if self.asset_kind not in (AssetKind.STOCK, AssetKind.ETF):
            raise ValueError(f"asset_kind must be STOCK or ETF, got {self.asset_kind}")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")


def schema_hash(columns: list[str]) -> str:
    """Deterministic fingerprint of a column list, stable across sessions."""
    return sha256("\n".join(columns).encode("utf-8")).hexdigest()


def make_manifest(
    *,
    asset_kind: AssetKind,
    columns: list[str],
    feature_set: str,
    label_definition: str,
    label_horizon_sessions: int,
    time_start: datetime,
    time_end: datetime,
    provider_version: str,
    universe_policy_version: str,
    row_count: int,
    generated_time: datetime | None = None,
) -> DatasetManifest:
    """Build a manifest from a concrete column list (hashing the schema).

    Provenance is explicit: provider and universe policy versions must be
    supplied by the caller. No fixture defaults are baked into the production
    factory, so production provenance can never be silently fabricated.
    """
    return DatasetManifest(
        asset_kind=asset_kind,
        schema_version="v1",
        schema_hash=schema_hash(columns),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        universe_policy_hash=schema_hash([universe_policy_version]),
        feature_set=feature_set,
        feature_set_hash=schema_hash([feature_set]),
        label_definition=label_definition,
        label_horizon_sessions=label_horizon_sessions,
        time_start=time_start,
        time_end=time_end,
        generated_time=generated_time or datetime.now(UTC),
        row_count=row_count,
    )


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
