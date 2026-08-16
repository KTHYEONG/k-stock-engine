"""Atomic stock_alpha_v2 research snapshot materialization.

Materializes an immutable v2 feature panel and calendar-correct residual label
dataset from one source snapshot's base panel and calendar, then publishes a
replacement snapshot manifest that differs from the source only in its
``FEATURES`` and ``LABELS`` references. The source base panel, evidence, and all
v1 artifacts are never modified; the new feature/label datasets are written,
re-read and hash-validated before any catalog append, and the snapshot manifest
is written atomically (fsync + ``os.replace``).

Failures before the catalog append leave only unregistered output directories
behind; failures after the append leave immutable catalog entries and no usable
snapshot manifest. Nothing is ever deleted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Union

import polars as pl

from src.core.costs import CostPoint, CostSchedule
from src.core.datasets import DatasetCertification, DatasetManifest
from src.core.instruments import AssetKind
from src.stocks.data.catalog import (
    SNAPSHOT_MANIFEST_NAME,
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    ResearchDataSnapshot,
    SnapshotManifest,
    SnapshotResolver,
    build_snapshot_manifest,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows
from src.stocks.data.costs import krx_market_for_code, load_cost_evidence
from src.stocks.data.curation import (
    BASE_PANEL_FEATURE_SET,
    FeaturePanelRequest,
    FeaturePanelResult,
    build_feature_panel,
    build_stock_alpha_v2_feature_panel,
)
from src.stocks.data.evidence import load_krx_calendar_snapshot
from src.stocks.data.labels import (
    LABEL_FEATURE_SET,
    MULTI_HORIZON_RESIDUAL_DEFINITION,
    RESIDUAL_O2O_PREFIX,
    SUPPORTED_RESIDUAL_HORIZONS,
    build_multi_horizon_residual_label_dataset,
    build_net_alpha_label_dataset,
    build_residual_o2o_label_dataset,
    publish_multi_horizon_residual_label_dataset,
    publish_net_alpha_label_dataset,
    publish_residual_o2o_label_dataset,
)
from src.stocks.data.outcome_evidence import (
    OUTCOME_EVIDENCE_DATASET_SUFFIX,
    build_partitioned_outcome_evidence,
    publish_outcome_evidence_dataset,
)
from src.stocks.data.outcome_open_bars import load_outcome_open_bar_evidence
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.data.repositories import ResearchDataRepository
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.stocks.ml.contracts import DEFAULT_CANDIDATE_HORIZON_SESSIONS
from src.stocks.ml.data import assess_outcome_readiness
from src.stocks.research.features import stock_alpha_v2_allowlist
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.research_v2")

MaterializationRequest = Union[
    "StockAlphaV2MaterializationRequest", "NetAlphaMaterializationRequest"
]

STOCK_ALPHA_V2_FEATURE_SET = "stock_alpha_v2"
RESIDUAL_LABEL_DEFINITION = f"{RESIDUAL_O2O_PREFIX}5d"
RESIDUAL_HORIZON_SESSIONS = 5
MULTI_HORIZON_LABEL_DEFINITION = MULTI_HORIZON_RESIDUAL_DEFINITION
MULTI_HORIZON_HORIZONS = SUPPORTED_RESIDUAL_HORIZONS

_CERTIFICATION_EVIDENCE = (
    (CatalogKind.INSTRUMENT_MASTER, "master"),
    (CatalogKind.CORPORATE_ACTIONS, "corporate_actions"),
    (CatalogKind.COSTS, "costs"),
)


@dataclass(frozen=True, slots=True)
class StockAlphaV2MaterializationRequest:
    """Explicit, non-empty inputs for one v2 snapshot materialization.

    Every id is supplied by the caller; there is no implicit ``latest``
    selection. ``windows`` must be strictly ordered (enforced by
    ``ResearchWindows``) and contained by the source base panel and calendar
    coverage. ``certification`` defaults to ``PROVISIONAL``; higher tiers
    require complete source evidence.
    """

    source_snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    snapshot_id: str
    catalog_root: Path
    base_root: Path
    feature_root: Path
    label_root: Path
    generated_time: datetime
    windows: ResearchWindows
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    min_coverage: float = 0.75
    calendar_path: Path | None = None
    net_alpha_horizons: tuple[int, ...] = ()
    net_alpha_reference_notional: float | None = None

    def __post_init__(self) -> None:
        for field in (
            "source_snapshot_id",
            "feature_dataset_id",
            "label_dataset_id",
            "snapshot_id",
        ):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be within (0, 1]")
        if self.net_alpha_horizons:
            if tuple(self.net_alpha_horizons) != tuple(
                sorted(set(self.net_alpha_horizons))
            ):
                raise ValueError("net_alpha_horizons must be strictly ascending and unique")
            if any(h < 1 for h in self.net_alpha_horizons):
                raise ValueError("net_alpha_horizons must be positive sessions")
            if self.net_alpha_reference_notional is None or self.net_alpha_reference_notional <= 0:
                raise ValueError("net_alpha_reference_notional must be positive when net_alpha_horizons set")


@dataclass(frozen=True, slots=True)
class StockAlphaV2MaterializationResult:
    """Immutable outcome of one v2 snapshot materialization."""

    snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    feature_content_hash: str
    label_content_hash: str
    feature_row_count: int
    label_row_count: int
    min_coverage: float
    certification: DatasetCertification


def materialize_stock_alpha_v2_snapshot(
    request: StockAlphaV2MaterializationRequest,
) -> StockAlphaV2MaterializationResult:
    """Materialize a v2 snapshot end-to-end with fail-closed preflight and re-reads."""
    catalog = CatalogStore(request.catalog_root)
    source = SnapshotResolver(catalog).resolve(request.source_snapshot_id)
    if source.base_panel is None:
        raise ValueError("source snapshot has no base-panel reference")
    if source.calendar is None:
        raise ValueError("source snapshot has no calendar reference")
    base_entry = source.base_panel
    calendar_entry = source.calendar
    base_store = ParquetDatasetStore(request.base_root)
    base_manifest = base_store.read_manifest(base_entry.name)
    calendar = _load_calendar(request, source, calendar_entry)
    _preflight(request, catalog, source, base_manifest, calendar)

    allowlist = stock_alpha_v2_allowlist()
    feature_result = build_stock_alpha_v2_feature_panel(
        request.base_root,
        request.feature_root,
        FeaturePanelRequest(
            dataset_id=request.feature_dataset_id,
            base_panel_id=base_entry.name,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            generated_time=request.generated_time,
            certification=request.certification,
        ),
        min_coverage=request.min_coverage,
    )
    feature_manifest = _re_read_features(request, allowlist, feature_result)

    base_frame = base_store.read(
        base_entry.name, AssetKind.STOCK, BASE_PANEL_FEATURE_SET,
        request.generated_time,
    )
    labels_frame = build_residual_o2o_label_dataset(
        base_frame, calendar, horizon_sessions=RESIDUAL_HORIZON_SESSIONS
    )
    label_result = publish_residual_o2o_label_dataset(
        labels_frame,
        destination_root=request.label_root,
        dataset_id=request.label_dataset_id,
        base_panel_hash=base_manifest.content_hash,
        calendar_hash=calendar.content_hash,
        horizon_sessions=RESIDUAL_HORIZON_SESSIONS,
        certification=request.certification,
        generated_time=request.generated_time,
    )
    label_manifest = _re_read_labels(request, calendar)

    feature_entry = CatalogEntry(
        kind=CatalogKind.FEATURES,
        name=request.feature_dataset_id,
        content_hash=feature_manifest.content_hash,
        schema_hash=feature_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=feature_manifest.time_start.date(),
            end=feature_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.feature_root / request.feature_dataset_id),
        references=(
            (CatalogKind.BASE_PANEL.value, base_entry.name),
            (CatalogKind.CALENDAR.value, calendar_entry.name),
        ),
        row_count=feature_result.row_count,
    )
    label_entry = CatalogEntry(
        kind=CatalogKind.LABELS,
        name=request.label_dataset_id,
        content_hash=label_manifest.content_hash,
        schema_hash=label_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=label_manifest.time_start.date(),
            end=label_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.label_root / request.label_dataset_id),
        references=(
            (CatalogKind.BASE_PANEL.value, base_entry.name),
            (CatalogKind.CALENDAR.value, calendar_entry.name),
        ),
        row_count=label_result.row_count,
    )
    catalog.register(feature_entry)
    catalog.register(label_entry)

    manifest = build_snapshot_manifest(
        snapshot_id=request.snapshot_id,
        certification=request.certification,
        timing_convention=source.manifest.timing_convention,
        windows=request.windows,
        references=_replacement_references(source, feature_entry, label_entry),
    )
    _write_manifest_atomic(request.catalog_root, manifest)

    resolved = SnapshotResolver(catalog).resolve(request.snapshot_id)
    repository = ResearchDataRepository(
        base_root=request.base_root,
        feature_root=request.feature_root,
        label_root=request.label_root,
    )
    composed = repository.compose_labeled_training_snapshot(
        resolved,
        feature_set=STOCK_ALPHA_V2_FEATURE_SET,
        decision_time=request.generated_time,
    )
    if composed.frame.is_empty():
        raise ValueError("v2 snapshot composition produced no rows")

    logger.info(
        "materialized v2 snapshot %s: %s features, %s labels from source %s",
        request.snapshot_id,
        request.feature_dataset_id,
        request.label_dataset_id,
        request.source_snapshot_id,
    )
    return StockAlphaV2MaterializationResult(
        snapshot_id=request.snapshot_id,
        feature_dataset_id=request.feature_dataset_id,
        label_dataset_id=request.label_dataset_id,
        feature_content_hash=feature_manifest.content_hash,
        label_content_hash=label_manifest.content_hash,
        feature_row_count=feature_result.row_count,
        label_row_count=label_result.row_count,
        min_coverage=request.min_coverage,
        certification=request.certification,
    )

def materialize_stock_alpha_v3_snapshot(
    request: StockAlphaV2MaterializationRequest,
) -> StockAlphaV2MaterializationResult:
    """Materialize a v3 snapshot with the immutable multi-horizon label panel.

    Mirrors the v2 path (same feature panel, preflight, evidence, and atomic
    manifest write) but builds and publishes the key-aligned
    ``residual_o2o_multi_5_10_15d`` label dataset so training can select a
    (5/10/15)-session cost-amortizing route. The manifest declares the
    multi-horizon label definition with a five-day control horizon; the source
    base panel, features, and every existing v2 artifact are never modified.
    """
    catalog = CatalogStore(request.catalog_root)
    source = SnapshotResolver(catalog).resolve(request.source_snapshot_id)
    if source.base_panel is None:
        raise ValueError("source snapshot has no base-panel reference")
    if source.calendar is None:
        raise ValueError("source snapshot has no calendar reference")
    base_entry = source.base_panel
    calendar_entry = source.calendar
    base_store = ParquetDatasetStore(request.base_root)
    base_manifest = base_store.read_manifest(base_entry.name)
    calendar = _load_calendar(request, source, calendar_entry)
    _preflight(request, catalog, source, base_manifest, calendar)

    allowlist = stock_alpha_v2_allowlist()
    feature_result = build_stock_alpha_v2_feature_panel(
        request.base_root,
        request.feature_root,
        FeaturePanelRequest(
            dataset_id=request.feature_dataset_id,
            base_panel_id=base_entry.name,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            generated_time=request.generated_time,
            certification=request.certification,
        ),
        min_coverage=request.min_coverage,
    )
    feature_manifest = _re_read_features(request, allowlist, feature_result)

    base_frame = base_store.read(
        base_entry.name, AssetKind.STOCK, BASE_PANEL_FEATURE_SET,
        request.generated_time,
    )
    labels_frame = build_multi_horizon_residual_label_dataset(base_frame, calendar)
    label_result = publish_multi_horizon_residual_label_dataset(
        labels_frame,
        destination_root=request.label_root,
        dataset_id=request.label_dataset_id,
        base_panel_hash=base_manifest.content_hash,
        calendar_hash=calendar.content_hash,
        horizons=MULTI_HORIZON_HORIZONS,
        certification=request.certification,
        generated_time=request.generated_time,
    )
    label_manifest = _re_read_labels_multi_horizon(request, calendar)

    if request.net_alpha_horizons:
        _publish_net_alpha_labels(
            request,
            base_frame,
            calendar,
            base_manifest,
            catalog,
            base_entry,
            calendar_entry,
        )

    feature_entry = CatalogEntry(
        kind=CatalogKind.FEATURES,
        name=request.feature_dataset_id,
        content_hash=feature_manifest.content_hash,
        schema_hash=feature_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=feature_manifest.time_start.date(),
            end=feature_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.feature_root / request.feature_dataset_id),
        references=(
            (CatalogKind.BASE_PANEL.value, base_entry.name),
            (CatalogKind.CALENDAR.value, calendar_entry.name),
        ),
        row_count=feature_result.row_count,
    )
    label_entry = CatalogEntry(
        kind=CatalogKind.LABELS,
        name=request.label_dataset_id,
        content_hash=label_manifest.content_hash,
        schema_hash=label_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=label_manifest.time_start.date(),
            end=label_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.label_root / request.label_dataset_id),
        references=(
            (CatalogKind.BASE_PANEL.value, base_entry.name),
            (CatalogKind.CALENDAR.value, calendar_entry.name),
        ),
        row_count=label_result.row_count,
    )
    catalog.register(feature_entry)
    catalog.register(label_entry)

    manifest = build_snapshot_manifest(
        snapshot_id=request.snapshot_id,
        certification=request.certification,
        timing_convention=source.manifest.timing_convention,
        windows=request.windows,
        references=_replacement_references(source, feature_entry, label_entry),
    )
    _write_manifest_atomic(request.catalog_root, manifest)

    resolved = SnapshotResolver(catalog).resolve(request.snapshot_id)
    repository = ResearchDataRepository(
        base_root=request.base_root,
        feature_root=request.feature_root,
        label_root=request.label_root,
    )
    composed = repository.compose_labeled_training_snapshot(
        resolved,
        feature_set=STOCK_ALPHA_V2_FEATURE_SET,
        decision_time=request.generated_time,
    )
    if composed.frame.is_empty():
        raise ValueError("v3 snapshot composition produced no rows")

    logger.info(
        "materialized v3 snapshot %s: %s features, multi-horizon labels %s from source %s",
        request.snapshot_id,
        request.feature_dataset_id,
        list(MULTI_HORIZON_HORIZONS),
        request.source_snapshot_id,
    )
    return StockAlphaV2MaterializationResult(
        snapshot_id=request.snapshot_id,
        feature_dataset_id=request.feature_dataset_id,
        label_dataset_id=request.label_dataset_id,
        feature_content_hash=feature_manifest.content_hash,
        label_content_hash=label_manifest.content_hash,
        feature_row_count=feature_result.row_count,
        label_row_count=label_result.row_count,
        min_coverage=request.min_coverage,
        certification=request.certification,
    )


def _publish_net_alpha_labels(
    request: StockAlphaV2MaterializationRequest,
    base_frame: pl.DataFrame,
    calendar: KRXSessionCalendar,
    base_manifest: DatasetManifest,
    catalog: CatalogStore,
    base_entry: CatalogEntry,
    calendar_entry: CatalogEntry,
) -> None:
    """Publish continuous net-alpha label horizons per pre-registered discovery horizon.

    Each horizon is built and published independently with the same policy
    kernel as the exact replay: no common-universe inner join is performed, so
    the training side records per-horizon label universes rather than shrinking
    the sample to the intersection of all horizons. The reference notional is
    derived from the replay minimum order unit and portfolio value, never an
    arbitrary participation constant.
    """
    # Evidence entries are versioned by semantic name (for example
    # ``kis_lifetime_preferential_counterfactual_v1``); ``costs`` is the
    # catalog kind, not a valid entry name.  Prefer the source snapshot's
    # exact reference and fall back to the sole registered cost entry for
    # backwards-compatible callers.
    cost_entry = catalog.get(CatalogKind.COSTS, "costs")
    if cost_entry is None:
        entries = catalog.list(CatalogKind.COSTS)
        cost_entry = entries[0] if len(entries) == 1 else None
    if cost_entry is None:
        raise ValueError("net-alpha label publication requires a costs catalog entry")
    cost_evidence = load_cost_evidence(
        Path(cost_entry.path), CoverageRange(start=base_manifest.time_start.date(), end=base_manifest.time_end.date())
    )
    cost_schedule = CostSchedule(
        name="net-alpha-reference",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=cost_evidence.commission_for(
                    request.generated_time
                ).buy_rate,
                tax_rate=cost_evidence.sell_tax_for(
                    krx_market_for_code(base_frame["instrument_id"][0]),
                    request.generated_time,
                ).sell_tax_rate,
                slippage_bps=cost_evidence.liquidity_model.impact_coefficient * 10_000.0,
                settlement_days=cost_evidence.settlement_days,
            ),
        ),
    )
    liquidity_model = cost_evidence.base_liquidity_model
    assert request.net_alpha_reference_notional is not None
    for horizon in request.net_alpha_horizons:
        labels_frame = build_net_alpha_label_dataset(
            base_frame,
            calendar,
            cost_schedule,
            liquidity_model,
            horizon_sessions=horizon,
            reference_notional=request.net_alpha_reference_notional,
        )
        label_result = publish_net_alpha_label_dataset(
            labels_frame,
            destination_root=request.label_root,
            dataset_id=f"{request.label_dataset_id}_net_alpha_{horizon}d",
            base_panel_hash=base_manifest.content_hash,
            calendar_hash=calendar.content_hash,
            horizon_sessions=horizon,
            certification=request.certification,
            generated_time=request.generated_time,
        )
        label_entry = CatalogEntry(
            kind=CatalogKind.LABELS,
            name=label_result.dataset_id,
            content_hash=label_result.manifest.content_hash,
            schema_hash=label_result.manifest.schema_hash,
            registered_at=request.generated_time,
            coverage=CoverageRange(
                start=label_result.manifest.time_start.date(),
                end=label_result.manifest.time_end.date(),
            ),
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(request.label_root / label_result.dataset_id),
            references=(
                (CatalogKind.BASE_PANEL.value, base_entry.name),
                (CatalogKind.CALENDAR.value, calendar_entry.name),
            ),
            row_count=label_result.row_count,
        )
        catalog.register(label_entry)


def _preflight(
    request: MaterializationRequest,
    catalog: CatalogStore,
    source: ResearchDataSnapshot,
    base_manifest: DatasetManifest,
    calendar: KRXSessionCalendar,
) -> None:
    """Fail closed before any write: ids, evidence, and window containment."""
    if source.base_panel is None:
        raise ValueError("source snapshot has no base-panel reference")
    if source.calendar is None:
        raise ValueError("source snapshot has no calendar reference")

    for root, dataset_id in (
        (request.feature_root, request.feature_dataset_id),
        (request.label_root, request.label_dataset_id),
    ):
        if (Path(root) / dataset_id).exists():
            raise ValueError(
                f"output id already exists as a directory: {dataset_id}"
            )
    if catalog.get(CatalogKind.FEATURES, request.feature_dataset_id) is not None:
        raise ValueError(
            f"catalog already has FEATURES {request.feature_dataset_id}"
        )
    if catalog.get(CatalogKind.LABELS, request.label_dataset_id) is not None:
        raise ValueError(f"catalog already has LABELS {request.label_dataset_id}")
    if (
        request.catalog_root / "snapshots" / request.snapshot_id / SNAPSHOT_MANIFEST_NAME
    ).exists():
        raise ValueError(f"snapshot manifest already exists: {request.snapshot_id}")

    if request.certification is not DatasetCertification.PROVISIONAL:
        missing = [
            kind.value
            for kind, name in _CERTIFICATION_EVIDENCE
            if getattr(source, name) is None
            or getattr(source, name).completeness is not EvidenceCompleteness.COMPLETE
        ]
        if missing:
            raise ValueError(
                f"{request.certification.value} certification requires complete "
                f"source evidence, missing/incomplete {missing}"
            )

    research = request.windows.research_range
    if base_manifest.time_start.date() > research.start or base_manifest.time_end.date() < research.end:
        raise ValueError(
            "requested research windows exceed base panel coverage "
            f"{base_manifest.time_start.date()}..{base_manifest.time_end.date()}"
        )
    if calendar.sessions[0] > research.start or calendar.sessions[-1] < research.end:
        raise ValueError(
            "requested research windows exceed calendar coverage "
            f"{calendar.sessions[0]}..{calendar.sessions[-1]}"
        )


def _load_calendar(
    request: MaterializationRequest,
    source: ResearchDataSnapshot,
    calendar_entry: CatalogEntry,
) -> KRXSessionCalendar:
    path = request.calendar_path
    if path is None:
        if not calendar_entry.path:
            raise ValueError("source snapshot has no calendar path")
        path = Path(calendar_entry.path)
    if not path.is_file():
        raise FileNotFoundError(f"calendar evidence not found: {path}")
    return load_krx_calendar_snapshot(path)


def _re_read_features(
    request: StockAlphaV2MaterializationRequest,
    allowlist: tuple[str, ...],
    feature_result: FeaturePanelResult,
) -> DatasetManifest:
    """Re-read the written v2 feature panel and verify its contract."""
    store = ParquetDatasetStore(request.feature_root)
    manifest = store.read_manifest(request.feature_dataset_id)
    if manifest.content_hash != feature_result.manifest.content_hash:
        raise ValueError(
            "v2 feature re-read content hash mismatch "
            f"{manifest.content_hash} vs built {feature_result.manifest.content_hash}"
        )
    if manifest.feature_set != STOCK_ALPHA_V2_FEATURE_SET:
        raise ValueError(
            f"v2 feature re-read manifest feature_set {manifest.feature_set!r}"
        )
    expected_columns = ["instrument_id", "session"] + [
        f"feature__{name}" for name in allowlist
    ]
    if store.content_columns(request.feature_dataset_id) != expected_columns:
        raise ValueError("v2 feature re-read column order mismatch")
    research = request.windows.research_range
    frame = store.read_bounded(
        request.feature_dataset_id,
        AssetKind.STOCK,
        STOCK_ALPHA_V2_FEATURE_SET,
        request.generated_time,
        session_start=research.start,
        session_end=research.end,
        columns=expected_columns,
    )
    if frame.is_empty():
        raise ValueError("v2 feature re-read produced no rows in the research range")
    if frame.columns != expected_columns:
        raise ValueError("v2 feature re-read produced unexpected columns")
    height = frame.height
    for column in expected_columns[2:]:
        coverage = (height - int(frame[column].null_count())) / height
        if coverage < request.min_coverage:
            raise ValueError(
                f"v2 feature {column} coverage {coverage:.6f} below "
                f"{request.min_coverage} in the research range"
            )
    return manifest


def _re_read_net_alpha_features(
    request: NetAlphaMaterializationRequest,
    allowlist: tuple[tuple[str, str], ...],
    feature_result: FeaturePanelResult,
) -> DatasetManifest:
    """Re-read the written net-alpha feature panel and verify its contract."""
    from src.stocks.ml.contracts import CANONICAL_FEATURE_SET

    store = ParquetDatasetStore(request.feature_root)
    manifest = store.read_manifest(request.feature_dataset_id)
    if manifest.content_hash != feature_result.manifest.content_hash:
        raise ValueError(
            "net-alpha feature re-read content hash mismatch "
            f"{manifest.content_hash} vs built {feature_result.manifest.content_hash}"
        )
    if manifest.feature_set != CANONICAL_FEATURE_SET:
        raise ValueError(
            f"net-alpha feature re-read manifest feature_set "
            f"{manifest.feature_set!r}, expected {CANONICAL_FEATURE_SET!r}"
        )
    expected_columns = ["instrument_id", "session"] + [
        f"feature__{source}" for source, _role in allowlist
    ]
    if store.content_columns(request.feature_dataset_id) != expected_columns:
        raise ValueError("net-alpha feature re-read column order mismatch")
    research = request.windows.research_range
    frame = store.read_bounded(
        request.feature_dataset_id,
        AssetKind.STOCK,
        CANONICAL_FEATURE_SET,
        request.generated_time,
        session_start=research.start,
        session_end=research.end,
        columns=expected_columns,
    )
    if frame.is_empty():
        raise ValueError(
            "net-alpha feature re-read produced no rows in the research range"
        )
    if frame.columns != expected_columns:
        raise ValueError("net-alpha feature re-read produced unexpected columns")
    height = frame.height
    for column in expected_columns[2:]:
        coverage = (height - int(frame[column].null_count())) / height
        if coverage < request.min_coverage:
            raise ValueError(
                f"net-alpha feature {column} coverage {coverage:.6f} below "
                f"{request.min_coverage} in the research range"
            )
    return manifest


def _re_read_labels(
    request: StockAlphaV2MaterializationRequest,
    calendar: KRXSessionCalendar,
) -> DatasetManifest:
    """Re-read the written residual label dataset and verify its contract."""
    store = ParquetDatasetStore(request.label_root)
    manifest = store.read_manifest(request.label_dataset_id)
    if manifest.label_definition != RESIDUAL_LABEL_DEFINITION:
        raise ValueError(
            "v2 label re-read manifest definition "
            f"{manifest.label_definition!r}, expected {RESIDUAL_LABEL_DEFINITION!r}"
        )
    if manifest.label_horizon_sessions != RESIDUAL_HORIZON_SESSIONS:
        raise ValueError(
            "v2 label re-read horizon "
            f"{manifest.label_horizon_sessions}, expected {RESIDUAL_HORIZON_SESSIONS}"
        )
    expected_columns = [
        "instrument_id",
        "session",
        RESIDUAL_LABEL_DEFINITION,
        "relevance",
        "label_available_time",
    ]
    if store.content_columns(request.label_dataset_id) != expected_columns:
        raise ValueError("v2 label re-read schema mismatch")
    research = request.windows.research_range
    frame = store.read_bounded(
        request.label_dataset_id,
        AssetKind.STOCK,
        LABEL_FEATURE_SET,
        request.generated_time,
        session_start=research.start,
        session_end=research.end,
        columns=expected_columns,
    )
    if frame.is_empty():
        raise ValueError("v2 label re-read produced no rows in the research range")
    if frame.columns != expected_columns:
        raise ValueError("v2 label re-read produced unexpected columns")
    if frame.filter(
        pl.col(RESIDUAL_LABEL_DEFINITION).is_not_null()
        & ~pl.col(RESIDUAL_LABEL_DEFINITION).is_finite()
    ).height:
        raise ValueError("v2 label re-read contains non-finite residuals")
    relevance = frame["relevance"]
    if relevance.null_count():
        raise ValueError("v2 label re-read contains null relevance")
    relevance_values = [float(value) for value in relevance.to_list() if value is not None]
    if relevance_values and (
        min(relevance_values) < 0 or max(relevance_values) > 4
    ):
        raise ValueError("v2 label re-read relevance outside integer 0..4")
    _assert_label_availability(frame, calendar)
    return manifest


def _assert_label_availability(
    frame: pl.DataFrame,
    calendar: KRXSessionCalendar,
) -> None:
    """Every label must be available at or after its exit-session open (UTC)."""
    sessions = list(calendar.sessions)
    cal = pl.DataFrame(
        {
            "_cal_pos": list(range(len(sessions))),
            "_session_date": sessions,
        }
    )
    exits = pl.DataFrame(
        {
            "_cal_pos": list(range(len(sessions))),
            "_exit_date": [
                sessions[p + 1 + RESIDUAL_HORIZON_SESSIONS]
                if p + 1 + RESIDUAL_HORIZON_SESSIONS < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    exit_open = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(time.min))
        .dt.replace_time_zone("UTC")
    )
    joined = (
        frame.with_columns(pl.col("session").cast(pl.Date).alias("_session_date"))
        .join(cal, on="_session_date", how="left")
        .join(exits, on="_cal_pos", how="left")
    )
    unknown = joined.filter(pl.col("_exit_date").is_null())
    if not unknown.is_empty():
        raise ValueError("v2 label re-read decision session outside calendar horizon")
    too_early = joined.filter(pl.col("label_available_time") < exit_open)
    if not too_early.is_empty():
        raise ValueError("v2 label re-read available before exit-session open")

def _re_read_labels_multi_horizon(
    request: StockAlphaV2MaterializationRequest,
    calendar: KRXSessionCalendar,
) -> DatasetManifest:
    """Re-read the written multi-horizon label dataset and verify its contract."""
    store = ParquetDatasetStore(request.label_root)
    manifest = store.read_manifest(request.label_dataset_id)
    if manifest.label_definition != MULTI_HORIZON_LABEL_DEFINITION:
        raise ValueError(
            "v3 label re-read manifest definition "
            f"{manifest.label_definition!r}, expected {MULTI_HORIZON_LABEL_DEFINITION!r}"
        )
    if manifest.label_horizon_sessions != RESIDUAL_HORIZON_SESSIONS:
        raise ValueError(
            "v3 label re-read control horizon "
            f"{manifest.label_horizon_sessions}, expected {RESIDUAL_HORIZON_SESSIONS}"
        )
    expected_columns = ["instrument_id", "session"]
    for h in MULTI_HORIZON_HORIZONS:
        expected_columns += [
            f"{RESIDUAL_O2O_PREFIX}{h}d",
            f"relevance_{h}d",
            f"label_available_time_{h}d",
        ]
    if store.content_columns(request.label_dataset_id) != expected_columns:
        raise ValueError("v3 label re-read schema mismatch")
    research = request.windows.research_range
    frame = store.read_bounded(
        request.label_dataset_id,
        AssetKind.STOCK,
        LABEL_FEATURE_SET,
        request.generated_time,
        session_start=research.start,
        session_end=research.end,
        columns=expected_columns,
    )
    if frame.is_empty():
        raise ValueError("v3 label re-read produced no rows in the research range")
    if frame.columns != expected_columns:
        raise ValueError("v3 label re-read produced unexpected columns")
    for h in MULTI_HORIZON_HORIZONS:
        residual = f"{RESIDUAL_O2O_PREFIX}{h}d"
        relevance = f"relevance_{h}d"
        if frame.filter(
            pl.col(residual).is_not_null() & ~pl.col(residual).is_finite()
        ).height:
            raise ValueError(f"v3 label re-read contains non-finite residuals in {residual}")
        relevance_values = [
            float(value) for value in frame[relevance].to_list() if value is not None
        ]
        if relevance_values and (
            min(relevance_values) < 0 or max(relevance_values) > 4
        ):
            raise ValueError(f"v3 label re-read relevance {relevance} outside 0..4")
    _assert_label_availability_multi_horizon(frame, calendar)
    return manifest


def _assert_label_availability_multi_horizon(
    frame: pl.DataFrame,
    calendar: KRXSessionCalendar,
) -> None:
    """Every horizon's label must be available at or after its exit-session open."""
    sessions = list(calendar.sessions)
    cal = pl.DataFrame(
        {
            "_cal_pos": list(range(len(sessions))),
            "_session_date": sessions,
        }
    )
    base = frame.with_columns(pl.col("session").cast(pl.Date).alias("_session_date")).join(
        cal, on="_session_date", how="left"
    )
    for h in MULTI_HORIZON_HORIZONS:
        exits = pl.DataFrame(
            {
                "_cal_pos": list(range(len(sessions))),
                "_exit_date": [
                    sessions[p + 1 + h] if p + 1 + h < len(sessions) else None
                    for p in range(len(sessions))
                ],
            }
        )
        exit_open = pl.col("_exit_date").dt.combine(pl.lit(time.min)).dt.replace_time_zone("UTC")
        joined = base.join(exits, on="_cal_pos", how="left")
        unknown = joined.filter(pl.col("_exit_date").is_null())
        if not unknown.is_empty():
            raise ValueError(
                f"v3 label re-read decision session outside calendar horizon {h}d"
            )
        available = f"label_available_time_{h}d"
        too_early = joined.filter(pl.col(available) < exit_open)
        if not too_early.is_empty():
            raise ValueError(
                f"v3 label re-read {available} available before exit-session open"
            )


def _replacement_references(
    source: ResearchDataSnapshot,
    feature_entry: CatalogEntry,
    label_entry: CatalogEntry,
    status_entry: CatalogEntry | None = None,
    raw_bar_entry: CatalogEntry | None = None,
    outcome_open_entry: CatalogEntry | None = None,
    evidence_entry: CatalogEntry | None = None,
) -> tuple[CatalogEntry, ...]:
    """Source references with derived artifacts replaced (or appended)."""
    replaced_features = False
    replaced_labels = False
    replaced_status = False
    replaced_raw_bars = False
    replaced_outcome_open = False
    replaced_evidence = False
    refs: list[CatalogEntry] = []
    for entry in source.manifest.references:
        if entry.kind is CatalogKind.FEATURES:
            refs.append(feature_entry)
            replaced_features = True
        elif entry.kind is CatalogKind.LABELS:
            refs.append(label_entry)
            replaced_labels = True
        elif entry.kind is CatalogKind.OUTCOME_STATUS:
            if status_entry is not None:
                refs.append(status_entry)
                replaced_status = True
        elif entry.kind is CatalogKind.RAW_BARS:
            refs.append(raw_bar_entry or entry)
            replaced_raw_bars = raw_bar_entry is not None
        elif entry.kind is CatalogKind.OUTCOME_OPEN_BARS:
            if outcome_open_entry is not None:
                refs.append(outcome_open_entry)
                replaced_outcome_open = True
        elif entry.kind is CatalogKind.OUTCOME_EVIDENCE:
            if evidence_entry is not None:
                refs.append(evidence_entry)
                replaced_evidence = True
        else:
            refs.append(entry)
    if not replaced_features:
        refs.append(feature_entry)
    if not replaced_labels:
        refs.append(label_entry)
    if status_entry is not None and not replaced_status:
        refs.append(status_entry)
    if raw_bar_entry is not None and not replaced_raw_bars:
        refs.append(raw_bar_entry)
    if outcome_open_entry is not None and not replaced_outcome_open:
        refs.append(outcome_open_entry)
    if evidence_entry is not None and not replaced_evidence:
        refs.append(evidence_entry)
    return tuple(refs)


def _load_raw_bar_evidence(
    catalog: CatalogStore,
    dataset_id: str | None,
) -> tuple[CatalogEntry | None, pl.DataFrame | None]:
    """Load one hash-verified immutable RAW_BARS artifact without network I/O."""
    if dataset_id is None:
        return None, None
    entry = catalog.get(CatalogKind.RAW_BARS, dataset_id)
    if entry is None:
        raise ValueError(f"raw bar dataset not found: {dataset_id!r}")
    if entry.completeness is not EvidenceCompleteness.COMPLETE:
        raise ValueError(f"raw bar dataset is not complete: {dataset_id!r}")
    path = Path(entry.path)
    if not path.is_file():
        raise ValueError(f"raw bar dataset path is not a file: {path}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry.content_hash:
        raise ValueError(f"raw bar dataset hash mismatch: {dataset_id!r}")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"raw bar dataset is invalid JSON: {path}") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("raw bar dataset records must be a list")
    if payload.get("record_count") != len(records):
        raise ValueError("raw bar dataset record_count mismatch")
    required = {"instrument_id", "price_date", "open"}
    if any(not isinstance(record, dict) or not required.issubset(record) for record in records):
        raise ValueError("raw bar dataset contains an invalid record")
    if not records:
        return entry, pl.DataFrame(
            schema={"instrument_id": pl.Utf8, "price_date": pl.Date, "open": pl.Float64}
        )
    frame = pl.DataFrame(records).select(
        pl.col("instrument_id").cast(pl.Utf8),
        pl.col("price_date").cast(pl.Date),
        pl.col("open").cast(pl.Float64),
    )
    return entry, frame


def _write_manifest_atomic(catalog_root: Path, manifest: SnapshotManifest) -> None:
    """Atomically persist a snapshot manifest (temp file, fsync, replace)."""
    target = (
        catalog_root / "snapshots" / manifest.snapshot_id / SNAPSHOT_MANIFEST_NAME
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{manifest.snapshot_id}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(manifest.to_json(), sort_keys=True, indent=2)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


@dataclass(frozen=True, slots=True)
class NetAlphaMaterializationRequest:
    """Explicit, non-empty inputs for one net-alpha snapshot materialization.

    Materializes the canonical ``stock_net_alpha_v1`` feature panel and the
    long, ``horizon_sessions``-partitioned net-alpha label dataset from one
    source snapshot's base panel and calendar, then publishes a replacement
    snapshot manifest.
    """

    source_snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    snapshot_id: str
    catalog_root: Path
    base_root: Path
    feature_root: Path
    label_root: Path
    generated_time: datetime
    windows: ResearchWindows
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    min_coverage: float = 0.75
    calendar_path: Path | None = None
    candidate_horizon_sessions: tuple[int, ...] = DEFAULT_CANDIDATE_HORIZON_SESSIONS
    reference_notional: float = 100_000_000.0
    policy: ExecutionOutcomePolicy | None = None
    raw_bar_dataset_id: str | None = None
    outcome_open_bar_dataset_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "source_snapshot_id",
            "feature_dataset_id",
            "label_dataset_id",
            "snapshot_id",
        ):
            if not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be within (0, 1]")
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if tuple(self.candidate_horizon_sessions) != tuple(
            sorted(set(self.candidate_horizon_sessions))
        ):
            raise ValueError(
                "candidate_horizon_sessions must be strictly ascending and unique"
            )
        if any(h < 1 for h in self.candidate_horizon_sessions):
            raise ValueError("candidate_horizon_sessions must be positive sessions")
        if self.reference_notional <= 0:
            raise ValueError("reference_notional must be positive")
        if self.raw_bar_dataset_id is not None and not self.raw_bar_dataset_id:
            raise ValueError("raw_bar_dataset_id must be non-empty when supplied")
        if self.outcome_open_bar_dataset_id is not None and not self.outcome_open_bar_dataset_id:
            raise ValueError("outcome_open_bar_dataset_id must be non-empty when supplied")
        if self.raw_bar_dataset_id is not None and self.outcome_open_bar_dataset_id is not None:
            raise ValueError("supply either raw_bar_dataset_id or outcome_open_bar_dataset_id")


@dataclass(frozen=True, slots=True)
class NetAlphaMaterializationResult:
    """Immutable outcome of one net-alpha snapshot materialization."""

    snapshot_id: str
    feature_dataset_id: str
    label_dataset_id: str
    feature_content_hash: str
    label_content_hash: str
    feature_row_count: int
    label_row_count: int
    min_coverage: float
    certification: DatasetCertification
    policy_id: str = "scheduled_open_v1"


def materialize_net_alpha_snapshot(
    request: NetAlphaMaterializationRequest,
) -> NetAlphaMaterializationResult:
    """Materialize a canonical ``stock_net_alpha_v1`` research snapshot.

    Builds and publishes the ``stock_net_alpha_v1`` feature panel (semantic
    roles) and the long, ``horizon_sessions``-partitioned net-alpha label
    dataset, then registers catalog entries and writes the replacement snapshot
    manifest atomically. Every horizon keeps its own universe; no common
    universe inner join is performed.
    """
    from src.stocks.ml.features import (
        STOCK_NET_ALPHA_V1_FEATURE_SET,
        stock_net_alpha_v1_contract_book,
        stock_net_alpha_v1_role_allowlist,
    )
    from src.stocks.ml.labels import (
        OUTCOME_STATUS_DATASET_SUFFIX,
        build_partitioned_net_alpha_labels_with_status,
        publish_outcome_status_sidecar,
        publish_partitioned_net_alpha_label_dataset,
    )

    policy = request.policy
    if policy is None:
        from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1

        policy = SCHEDULED_OPEN_V1

    catalog = CatalogStore(request.catalog_root)
    source = SnapshotResolver(catalog).resolve(request.source_snapshot_id)
    if source.base_panel is None:
        raise ValueError("source snapshot has no base-panel reference")
    if source.calendar is None:
        raise ValueError("source snapshot has no calendar reference")
    base_entry = source.base_panel
    calendar_entry = source.calendar
    base_store = ParquetDatasetStore(request.base_root)
    base_manifest = base_store.read_manifest(base_entry.name)
    raw_bar_entry, raw_bar_evidence = _load_raw_bar_evidence(catalog, request.raw_bar_dataset_id)
    outcome_open_entry, outcome_open_evidence = load_outcome_open_bar_evidence(
        catalog, request.outcome_open_bar_dataset_id, request.generated_time
    )
    if outcome_open_evidence is not None:
        raw_bar_evidence = outcome_open_evidence
    calendar = _load_calendar(request, source, calendar_entry)
    _preflight(request, catalog, source, base_manifest, calendar)

    feature_result = build_feature_panel(
        request.base_root,
        request.feature_root,
        FeaturePanelRequest(
            dataset_id=request.feature_dataset_id,
            base_panel_id=base_entry.name,
            feature_set=STOCK_NET_ALPHA_V1_FEATURE_SET,
            feature_contract_book=stock_net_alpha_v1_contract_book(),
            generated_time=request.generated_time,
            certification=request.certification,
        ),
    )
    feature_manifest = _re_read_net_alpha_features(
        request, stock_net_alpha_v1_role_allowlist(), feature_result
    )

    base_frame = base_store.read(
        base_entry.name, AssetKind.STOCK, BASE_PANEL_FEATURE_SET,
        request.generated_time,
    )
    cost_entry = source.costs
    if cost_entry is None:
        entries = catalog.list(CatalogKind.COSTS)
        cost_entry = entries[0] if len(entries) == 1 else None
    if cost_entry is None:
        raise ValueError("net-alpha label publication requires a costs catalog entry")
    cost_coverage = cost_entry.coverage
    if cost_coverage is None:
        raise ValueError(
            "net-alpha label publication requires costs coverage metadata"
        )
    cost_evidence = load_cost_evidence(
        Path(cost_entry.path),
        CoverageRange(
            start=base_manifest.time_start.date(),
            end=min(base_manifest.time_end.date(), cost_coverage.end),
        ),
    )
    # Do not manufacture labels beyond the verified cost-evidence horizon.
    # The base panel can be refreshed ahead of the cost catalog, so trim only
    # the label source rows while retaining the full feature dataset.
    if cost_coverage.end < base_manifest.time_end.date():
        base_frame = base_frame.filter(
            pl.col("session") <= cost_coverage.end
        )
    # The canonical base panel stores derived controls under explicit
    # ``raw__`` names.  Normalize them at the label boundary rather than
    # requiring a second, duplicate feature panel.  Beta is unavailable in
    # this source vintage; a neutral zero beta keeps the risk projection
    # deterministic and makes the limitation visible in the manifest.
    aliases = {
        "adtv": "raw__adtv_20d",
        "volatility": "raw__volatility_20d",
    }
    for target, source_column in aliases.items():
        if target not in base_frame.columns and source_column in base_frame.columns:
            base_frame = base_frame.with_columns(pl.col(source_column).alias(target))
    if "beta" not in base_frame.columns:
        base_frame = base_frame.with_columns(pl.lit(0.0).alias("beta"))
    cost_schedule = CostSchedule(
        name="net-alpha-reference",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=cost_evidence.commission_for(
                    request.generated_time
                ).buy_rate,
                tax_rate=cost_evidence.sell_tax_for(
                    krx_market_for_code(base_frame["instrument_id"][0]),
                    request.generated_time,
                ).sell_tax_rate,
                slippage_bps=(
                    cost_evidence.liquidity_model.impact_coefficient * 10_000.0
                ),
                settlement_days=cost_evidence.settlement_days,
            ),
        ),
    )
    liquidity_model = cost_evidence.base_liquidity_model

    labels_frame, status_frame = build_partitioned_net_alpha_labels_with_status(
        base_frame,
        calendar,
        cost_schedule,
        liquidity_model,
        horizon_sessions=request.candidate_horizon_sessions,
        reference_notional=request.reference_notional,
        policy=policy,
        bar_evidence=raw_bar_evidence,
    )
    readiness = assess_outcome_readiness(
        base_frame.select("instrument_id", "session"),
        status_frame,
        request.candidate_horizon_sessions,
    )
    logger.info(
        "net-alpha snapshot %s outcome readiness passed=%s horizons=%s",
        request.snapshot_id,
        readiness.passed,
        [result.horizon_sessions for result in readiness.horizon_results],
    )
    if not readiness.passed and request.certification is not DatasetCertification.PROVISIONAL:
        raise ValueError(
            "net-alpha snapshot outcome readiness failed: "
            f"{json.dumps(readiness.to_json(), sort_keys=True)}"
        )
    label_result = publish_partitioned_net_alpha_label_dataset(
        labels_frame,
        destination_root=request.label_root,
        dataset_id=request.label_dataset_id,
        base_panel_hash=base_manifest.content_hash,
        calendar_hash=calendar.content_hash,
        horizon_sessions=request.candidate_horizon_sessions,
        certification=request.certification,
        generated_time=request.generated_time,
    )
    label_manifest = label_result.manifest
    status_result = publish_outcome_status_sidecar(
        status_frame,
        destination_root=request.label_root,
        dataset_id=f"{request.label_dataset_id}{OUTCOME_STATUS_DATASET_SUFFIX}",
        base_panel_hash=base_manifest.content_hash,
        calendar_hash=calendar.content_hash,
        horizon_sessions=request.candidate_horizon_sessions,
        certification=request.certification,
        generated_time=request.generated_time,
        policy=policy,
    )
    evidence_frame = build_partitioned_outcome_evidence(
        base_frame,
        calendar,
        horizon_sessions=request.candidate_horizon_sessions,
        policy=policy,
        bar_evidence=raw_bar_evidence,
    )
    evidence_result = publish_outcome_evidence_dataset(
        evidence_frame,
        destination_root=request.label_root,
        dataset_id=f"{request.label_dataset_id}{OUTCOME_EVIDENCE_DATASET_SUFFIX}",
        base_panel_hash=base_manifest.content_hash,
        calendar_hash=calendar.content_hash,
        horizon_sessions=request.candidate_horizon_sessions,
        certification=request.certification,
        generated_time=request.generated_time,
        policy=policy,
    )

    feature_entry = CatalogEntry(
        kind=CatalogKind.FEATURES,
        name=request.feature_dataset_id,
        content_hash=feature_manifest.content_hash,
        schema_hash=feature_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=feature_manifest.time_start.date(),
            end=feature_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.feature_root / request.feature_dataset_id),
        references=(
            (CatalogKind.BASE_PANEL.value, base_entry.name),
            (CatalogKind.CALENDAR.value, calendar_entry.name),
        ),
        row_count=feature_result.row_count,
    )
    label_references: list[tuple[str, str]] = [
        (CatalogKind.BASE_PANEL.value, base_entry.name),
        (CatalogKind.CALENDAR.value, calendar_entry.name),
    ]
    if raw_bar_entry is not None:
        label_references.append((CatalogKind.RAW_BARS.value, raw_bar_entry.name))
    label_entry = CatalogEntry(
        kind=CatalogKind.LABELS,
        name=request.label_dataset_id,
        content_hash=label_manifest.content_hash,
        schema_hash=label_manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=label_manifest.time_start.date(),
            end=label_manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.label_root / request.label_dataset_id),
        references=tuple(label_references),
        row_count=label_result.row_count,
    )
    status_references: list[tuple[str, str]] = [
        (CatalogKind.BASE_PANEL.value, base_entry.name),
        (CatalogKind.CALENDAR.value, calendar_entry.name),
        (CatalogKind.LABELS.value, request.label_dataset_id),
    ]
    if source.master is not None:
        status_references.append(
            (CatalogKind.INSTRUMENT_MASTER.value, source.master.name)
        )
    if source.corporate_actions is not None:
        status_references.append(
            (CatalogKind.CORPORATE_ACTIONS.value, source.corporate_actions.name)
        )
    status_references.append((CatalogKind.COSTS.value, cost_entry.name))
    if raw_bar_entry is not None:
        status_references.append((CatalogKind.RAW_BARS.value, raw_bar_entry.name))
    status_entry = CatalogEntry(
        kind=CatalogKind.OUTCOME_STATUS,
        name=status_result.dataset_id,
        content_hash=status_result.manifest.content_hash,
        schema_hash=status_result.manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=status_result.manifest.time_start.date(),
            end=status_result.manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.label_root / status_result.dataset_id),
        references=tuple(status_references),
        row_count=status_result.row_count,
    )
    evidence_references: list[tuple[str, str]] = [
        (CatalogKind.BASE_PANEL.value, base_entry.name),
        (CatalogKind.CALENDAR.value, calendar_entry.name),
        (CatalogKind.LABELS.value, request.label_dataset_id),
        (CatalogKind.OUTCOME_STATUS.value, status_result.dataset_id),
    ]
    if raw_bar_entry is not None:
        evidence_references.append((CatalogKind.RAW_BARS.value, raw_bar_entry.name))
    evidence_entry = CatalogEntry(
        kind=CatalogKind.OUTCOME_EVIDENCE,
        name=evidence_result.dataset_id,
        content_hash=evidence_result.manifest.content_hash,
        schema_hash=evidence_result.manifest.schema_hash,
        registered_at=request.generated_time,
        coverage=CoverageRange(
            start=evidence_result.manifest.time_start.date(),
            end=evidence_result.manifest.time_end.date(),
        ),
        completeness=EvidenceCompleteness.COMPLETE,
        path=str(request.label_root / evidence_result.dataset_id),
        references=tuple(evidence_references),
        row_count=evidence_result.row_count,
    )
    catalog.register(feature_entry)
    catalog.register(label_entry)
    catalog.register(status_entry)
    catalog.register(evidence_entry)

    manifest = build_snapshot_manifest(
        snapshot_id=request.snapshot_id,
        certification=request.certification,
        timing_convention=source.manifest.timing_convention,
        windows=request.windows,
        references=_replacement_references(
            source, feature_entry, label_entry, status_entry, raw_bar_entry,
            outcome_open_entry, evidence_entry,
        ),
    )
    _write_manifest_atomic(request.catalog_root, manifest)

    logger.info(
        "materialized net-alpha snapshot %s: features %s, labels %s, status %s, "
        "evidence %s, policy %s horizons %s",
        request.snapshot_id,
        request.feature_dataset_id,
        request.label_dataset_id,
        status_result.dataset_id,
        evidence_result.dataset_id,
        policy.policy_id,
        list(request.candidate_horizon_sessions),
    )
    return NetAlphaMaterializationResult(
        snapshot_id=request.snapshot_id,
        feature_dataset_id=request.feature_dataset_id,
        label_dataset_id=request.label_dataset_id,
        feature_content_hash=feature_manifest.content_hash,
        label_content_hash=label_manifest.content_hash,
        feature_row_count=feature_result.row_count,
        label_row_count=label_result.row_count,
        min_coverage=request.min_coverage,
        certification=request.certification,
        policy_id=policy.policy_id,
    )
