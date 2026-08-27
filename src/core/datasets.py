"""Shared dataset manifest and provenance contracts.

``core`` is intentionally small and cannot import ``stocks``, ``etfs``, or a
provider library, so the manifest lives here and is reused by both asset
subsystems. The generic storage adapter validates manifests with the same
contract before touching Parquet.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

from src.core.instruments import AssetKind


class DatasetCertification(StrEnum):
    """Explicit dataset certification tier.

    ``PROVISIONAL`` data may support research diagnostics only; ``RESEARCH``
    supports research and paper workflows; ``PRODUCTION`` is the only tier that
    may feed promoted artifacts or live trading. Certification is never
    inferred from a filename or a source provider.
    """

    PROVISIONAL = "provisional"
    RESEARCH = "research"
    PRODUCTION = "production"


# Declared physical layout of curated datasets. ``storage_layout`` values are
# part of the manifest contract: an unknown layout fails validation instead of
# being guessed at read time.
HIVE_PARTITION_LAYOUT = "hive:partitions/year(session)/month(session)"


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
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    calendar_hash: str = ""
    corporate_action_hash: str = ""
    cost_source_hash: str = ""
    master_hash: str = ""
    quality_report_hash: str = ""
    content_hash: str = ""
    storage_layout: str = ""
    reference_notional: float | None = None

    def __post_init__(self) -> None:
        if self.asset_kind not in (AssetKind.STOCK, AssetKind.ETF):
            raise ValueError(f"asset_kind must be STOCK or ETF, got {self.asset_kind}")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.schema_version not in ("v1", "v2"):
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.storage_layout not in ("", HIVE_PARTITION_LAYOUT):
            raise ValueError(f"unknown storage_layout {self.storage_layout!r}")
        if self.reference_notional is not None:
            import math as _math

            if not _math.isfinite(self.reference_notional) or self.reference_notional <= 0:
                raise ValueError("reference_notional must be positive finite when supplied")


def validate_production_manifest(manifest: DatasetManifest) -> None:
    """Fail closed unless the manifest is production-certified and complete.

    Production requires explicit calendar, corporate-action, and cost-source
    coverage hashes in addition to the certification tier. A missing hash is
    treated as missing coverage, never as an implicit acceptable default.
    """
    if manifest.certification is not DatasetCertification.PRODUCTION:
        raise ValueError(
            f"production requires PRODUCTION certification, got {manifest.certification.value}"
        )
    if not manifest.calendar_hash:
        raise ValueError("production manifest must carry a calendar_hash")
    if not manifest.corporate_action_hash:
        raise ValueError("production manifest must carry a corporate_action_hash")
    if not manifest.cost_source_hash:
        raise ValueError("production manifest must carry a cost_source_hash")


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
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    calendar_hash: str = "",
    corporate_action_hash: str = "",
    cost_source_hash: str = "",
    master_hash: str = "",
    quality_report_hash: str = "",
    schema_version: str = "v1",
    content_hash: str = "",
    storage_layout: str = "",
    reference_notional: float | None = None,
) -> DatasetManifest:
    """Build a manifest from a concrete column list (hashing the schema).

    Provenance is explicit: provider and universe policy versions must be
    supplied by the caller. No fixture defaults are baked into the production
    factory, so production provenance can never be silently fabricated. New
    curated datasets advance ``schema_version`` to ``"v2"`` and must carry the
    immutable ``content_hash`` and declared ``storage_layout``.
    """
    return DatasetManifest(
        asset_kind=asset_kind,
        schema_version=schema_version,
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
        certification=certification,
        calendar_hash=calendar_hash,
        corporate_action_hash=corporate_action_hash,
        cost_source_hash=cost_source_hash,
        master_hash=master_hash,
        quality_report_hash=quality_report_hash,
        content_hash=content_hash,
        storage_layout=storage_layout,
        reference_notional=reference_notional,
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
    if manifest.schema_version == "v2" and not manifest.content_hash:
        raise ValueError("v2 manifest must carry a content_hash")
    if manifest.schema_version == "v2" and not manifest.storage_layout:
        raise ValueError("v2 manifest must carry a storage_layout")
    if not manifest.schema_hash:
        raise ValueError("manifest schema_hash must not be empty")
    if not manifest.universe_policy_hash:
        raise ValueError("manifest universe_policy_hash must not be empty")
    if manifest.label_horizon_sessions <= 0:
        raise ValueError("label_horizon_sessions must be positive")
