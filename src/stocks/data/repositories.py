"""Stock dataset repository: translates domain requests into the generic store.

The repository enforces ``AssetKind.STOCK`` at its boundary and reuses the
asset-neutral ``ParquetDatasetStore`` for persistence; it is not a duplicate
store implementation. A deterministic migration adapter certifies the legacy
``*_feat.parquet`` files as a provisional, non-promotable point-in-time panel.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetCertification, DatasetManifest, make_manifest
from src.core.instruments import AssetKind
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogStore,
    ResearchDataSnapshot,
    SnapshotResolver,
)
from src.stocks.data.contracts import CoverageRange, DatasetSnapshot
from src.stocks.data.curation import BASE_PANEL_FEATURE_SET
from src.stocks.data.feature_contracts import feature_contract_book_from_allowlist
from src.stocks.data.labels import LABEL_FEATURE_SET
from src.stocks.ml.contracts import CANONICAL_FEATURE_SET, OUTCOME_STATUS_COLUMN
from src.stocks.ml.labels import HORIZON_COLUMN, OUTCOME_STATUS_DATASET_SUFFIX
from src.stocks.research.datasets import (
    ELIGIBLE_STATUS,
    QUALITY_REASON_COLUMN,
    QUALITY_STATUS_COLUMN,
    QUARANTINED_STATUS,
    research_eligible_frame,
)
from src.stocks.research.features import stock_alpha_v2_allowlist
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.repositories")

PROVISIONAL_PROVIDER_VERSION = "provisional-legacy"
_IDENTITY_COLUMNS = ("date", "ticker")
_OHLC_COLUMNS = ("open", "high", "low", "close")
_TARGET_PREFIXES = ("target_", "label_")

STOCK_ALPHA_V2_FEATURE_SET = "stock_alpha_v2"
V2_LABEL_DEFINITION = "residual_o2o_5d"
V2_LABEL_HORIZON = 5
V2_REQUIRED_LABEL_COLUMNS = ("residual_o2o_5d", "relevance", "label_available_time")
MULTI_HORIZON_LABEL_DEFINITION = "residual_o2o_multi_5_10_15d"
MULTI_HORIZON_HORIZONS = (5, 10, 15)
MULTI_HORIZON_LABEL_COLUMNS = tuple(
    column
    for h in MULTI_HORIZON_HORIZONS
    for column in (
        f"residual_o2o_{h}d",
        f"relevance_{h}d",
        f"label_available_time_{h}d",
    )
)
OUTCOME_EVIDENCE_FEATURE_SET = "outcome_evidence"
_OUTCOME_EVIDENCE_PROJECTION = (
    "instrument_id",
    "session",
    HORIZON_COLUMN,
    "policy_hash",
    "resolution_kind",
    OUTCOME_STATUS_COLUMN,
    "scheduled_entry_session",
    "scheduled_exit_session",
    "entry_disposition",
    "exit_disposition",
)


def read_provisional_legacy_panel(
    root: Path,
    start_date: date,
    end_date: date,
    allowed_features: tuple[str, ...],
) -> DatasetSnapshot:
    """Certify the legacy ``year=*/*_feat.parquet`` files as a provisional panel.

    Reads exactly the feature files in deterministic sorted order, rejects
    schema variants, duplicate ``(session, instrument_id)`` rows, missing
    lineage, and unknown date semantics, converts NaN/Inf to null, and drops
    target/label columns from the predictor view. The result is marked
    ``provisional-legacy`` and must never be promoted until security-master,
    corporate-action, and tradability coverage pass.
    """
    files = sorted(Path(root).glob("year=*/*_feat.parquet"))
    if not files:
        raise FileNotFoundError(f"no *_feat.parquet under {root}")

    frames: list[pl.DataFrame] = []
    expected_schema: dict[str, pl.DataType] | None = None
    for parquet_file in files:
        frame = pl.read_parquet(parquet_file)
        missing = [c for c in _IDENTITY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{parquet_file}: missing lineage columns {missing}")
        date_dtype = frame["date"].dtype
        if not isinstance(date_dtype, (pl.Date, pl.Datetime)):
            raise ValueError(f"{parquet_file}: unknown date semantics {date_dtype}")
        if expected_schema is None:
            expected_schema = frame.schema
        elif frame.schema != expected_schema:
            raise ValueError(f"{parquet_file}: schema variant vs first file")
        frames.append(frame)

    panel = pl.concat(frames)
    duplicates = (
        panel.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError(f"{duplicates.height} duplicate (date, ticker) rows")

    panel = _null_nan_inf(panel)
    quality_checks = [
        pl.col(column).is_null() | (pl.col(column) <= 0)
        for column in _OHLC_COLUMNS
        if column in panel.columns
    ]
    invalid_ohlc = pl.any_horizontal(quality_checks) if quality_checks else pl.lit(True)
    panel = panel.with_columns(
        pl.when(invalid_ohlc)
        .then(pl.lit(QUARANTINED_STATUS))
        .otherwise(pl.lit(ELIGIBLE_STATUS))
        .alias(QUALITY_STATUS_COLUMN),
        pl.when(invalid_ohlc)
        .then(pl.lit("non_positive_or_missing_ohlc"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias(QUALITY_REASON_COLUMN),
    )

    in_window = panel.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    if in_window.is_empty():
        raise ValueError(
            f"no legacy rows in {start_date.isoformat()}..{end_date.isoformat()}"
        )

    keep = [c for c in _IDENTITY_COLUMNS + _OHLC_COLUMNS if c in panel.columns]
    keep += [
        c
        for c in allowed_features
        if c in panel.columns and not c.startswith(_TARGET_PREFIXES)
    ]
    keep += [c for c in ("volume", "trading_value", "market_cap", "sector") if c in panel.columns]
    keep += [QUALITY_STATUS_COLUMN, QUALITY_REASON_COLUMN]
    predictor = in_window.select(keep).unique(subset=["date", "ticker"], keep="first")

    predictor = predictor.with_columns(
        (pl.lit("KRX:") + pl.col("ticker").cast(pl.Utf8)).alias("instrument_id"),
        pl.col("date").cast(pl.Date).alias("session"),
    )
    session_dt = (
        pl.col("date")
        .dt.combine(pl.lit(time(15, 30)))
        .dt.replace_time_zone("UTC")
    )
    predictor = predictor.with_columns(
        session_dt.alias("observation_time"),
        session_dt.alias("available_time"),
    )
    predictor = predictor.select(
        ["instrument_id", "session", "observation_time", "available_time"]
        + [c for c in predictor.columns if c not in ("instrument_id", "session", "observation_time", "available_time", "date", "ticker")]
    )

    columns = predictor.columns
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=columns,
        feature_set="provisional-legacy",
        label_definition="provisional-legacy",
        label_horizon_sessions=5,
        time_start=datetime.combine(start_date, time.min, tzinfo=UTC),
        time_end=datetime.combine(end_date, time.min, tzinfo=UTC),
        provider_version=PROVISIONAL_PROVIDER_VERSION,
        universe_policy_version=PROVISIONAL_PROVIDER_VERSION,
        row_count=predictor.height,
        generated_time=datetime.now(UTC),
    )
    logger.info(
        "certified provisional legacy panel: %s rows, %s columns, %s files",
        predictor.height,
        len(columns),
        len(files),
    )
    return DatasetSnapshot(manifest=manifest, frame=predictor)


def _null_nan_inf(frame: pl.DataFrame) -> pl.DataFrame:
    float_columns = [c for c, dtype in frame.schema.items() if dtype.is_float()]
    if not float_columns:
        return frame
    return frame.with_columns(
        [
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
            for c in float_columns
        ]
    )


class StockDatasetRepository:
    """Stock-only facade over ``ParquetDatasetStore``."""

    def __init__(self, store: ParquetDatasetStore):
        self.store = store
        self.asset_kind = AssetKind.STOCK

    def write(
        self,
        snapshot: DatasetSnapshot,
        *,
        dataset_id: str,
        feature_set: str,
        decision_time: datetime,
    ) -> str:
        """Persist a validated stock dataset, rejecting non-stock manifests."""
        self._assert_stock(snapshot.manifest)
        self.store.write(
            snapshot.frame,
            dataset_id=dataset_id,
            manifest=snapshot.manifest,
            expected_feature_set=feature_set,
            decision_time=decision_time,
        )
        return dataset_id

    def read(
        self,
        dataset_id: str,
        feature_set: str,
        decision_time: datetime,
    ) -> DatasetSnapshot:
        """Read a validated stock dataset snapshot (stock kind enforced).

        Enforces the manifest's requested certification tier and exposes only
        ``eligible`` rows to modern workflows. A research/production dataset
        must carry validated master, calendar, action, and quality-report
        evidence hashes; otherwise the read fails closed.
        """
        manifest = self.store.read_manifest(dataset_id)
        self._assert_stock(manifest)
        frame = self.store.read(
            dataset_id, AssetKind.STOCK, feature_set, decision_time
        )
        self._assert_tier_evidence(manifest)
        frame = research_eligible_frame(frame)
        return DatasetSnapshot(manifest=manifest, frame=frame)

    def _assert_tier_evidence(self, manifest: DatasetManifest) -> None:
        if manifest.certification is DatasetCertification.PROVISIONAL:
            return
        evidence = {
            "calendar_hash": manifest.calendar_hash,
            "corporate_action_hash": manifest.corporate_action_hash,
            "master_hash": manifest.master_hash,
            "quality_report_hash": manifest.quality_report_hash,
        }
        missing = [name for name, value in evidence.items() if not value]
        if missing:
            raise ValueError(
                f"{manifest.certification.value} read requires evidence hashes, "
                f"missing {missing}"
            )

    def _assert_stock(self, manifest: DatasetManifest) -> None:
        if manifest.asset_kind is not AssetKind.STOCK:
            raise ValueError(
                f"stock repository rejects {manifest.asset_kind.value} manifest"
            )


_BASE_TRAINING_COLUMNS = (
    "instrument_id",
    "session",
    "observation_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "sector",
    "action_interval_covered",
    QUALITY_STATUS_COLUMN,
    QUALITY_REASON_COLUMN,
)


def resolve_snapshot_for_mode(
    catalog_root: Path,
    snapshot_id: str,
    *,
    mode: str,
) -> ResearchDataSnapshot:
    """Resolve a snapshot and reject provisional evidence for paper/live modes."""
    if not snapshot_id:
        raise ValueError("a snapshot-id is required; no implicit newest selection")
    store = CatalogStore(catalog_root)
    resolver = SnapshotResolver(store)
    snapshot = (
        resolver.resolve_execution(snapshot_id)
        if mode == "research"
        else resolver.resolve(snapshot_id)
    )
    if mode in ("paper", "live") and (
        snapshot.manifest.certification is DatasetCertification.PROVISIONAL
    ):
        raise ValueError(
            f"snapshot {snapshot_id} is provisional and cannot drive {mode} mode"
        )
    return snapshot


class ResearchDataRepository:
    """Snapshot-aware composition of base panels, feature panels, and labels.

    A :class:`ResearchDataSnapshot` selects exact immutable dataset versions;
    this repository reads them through the verified bounded lazy read plan and
    composes the training/backtest frame. It never copies rows across versions
    and never falls back to an implicit newest dataset.
    """

    def __init__(
        self,
        *,
        base_root: Path,
        feature_root: Path,
        label_root: Path,
    ):
        self.base_store = ParquetDatasetStore(base_root)
        self.feature_store = ParquetDatasetStore(feature_root)
        self.label_store = ParquetDatasetStore(label_root)

    def read_base_bounded(
        self,
        entry: CatalogEntry,
        decision_time: datetime,
        *,
        research_range: CoverageRange,
        columns: tuple[str, ...] = _BASE_TRAINING_COLUMNS,
    ) -> pl.DataFrame:
        return self.base_store.read_bounded(
            entry.name,
            AssetKind.STOCK,
            BASE_PANEL_FEATURE_SET,
            decision_time,
            session_start=research_range.start,
            session_end=research_range.end,
            columns=list(columns),
        )

    def read_features_bounded(
        self,
        entry: CatalogEntry,
        feature_set: str,
        decision_time: datetime,
        *,
        research_range: CoverageRange,
    ) -> pl.DataFrame:
        feature_columns = [
            c for c in self.feature_store.content_columns(entry.name) if c.startswith("feature__")
        ]
        if not feature_columns:
            raise ValueError(f"feature panel {entry.name} exposes no feature__ columns")
        return self.feature_store.read_bounded(
            entry.name,
            AssetKind.STOCK,
            feature_set,
            decision_time,
            session_start=research_range.start,
            session_end=research_range.end,
            columns=["instrument_id", "session", *feature_columns],
        )

    def read_labels_bounded(
        self,
        entry: CatalogEntry,
        decision_time: datetime,
        *,
        research_range: CoverageRange,
        columns: tuple[str, ...] = ("instrument_id", "session"),
    ) -> pl.DataFrame:
        return self.label_store.read_bounded(
            entry.name,
            AssetKind.STOCK,
            LABEL_FEATURE_SET,
            decision_time,
            session_start=research_range.start,
            session_end=research_range.end,
            columns=list(columns),
        )

    def _read_base_and_features(
        self,
        snapshot: ResearchDataSnapshot,
        *,
        feature_set: str,
        decision_time: datetime,
        research_range: CoverageRange,
    ) -> tuple[pl.DataFrame, pl.DataFrame, DatasetManifest]:
        if snapshot.base_panel is None:
            raise ValueError("snapshot has no base-panel reference")
        if snapshot.features is None:
            raise ValueError("snapshot has no feature-panel reference")
        base = self.read_base_bounded(
            snapshot.base_panel, decision_time, research_range=research_range
        )
        features = self.read_features_bounded(
            snapshot.features, feature_set, decision_time, research_range=research_range
        )
        feature_manifest = self.feature_store.read_manifest(snapshot.features.name)
        if feature_manifest.feature_set != feature_set:
            raise ValueError(
                f"snapshot feature panel {snapshot.features.name} has feature_set "
                f"{feature_manifest.feature_set!r}, requested {feature_set!r}"
            )
        return base, features, feature_manifest

    def compose_labeled_training_snapshot(
        self,
        snapshot: ResearchDataSnapshot,
        *,
        feature_set: str,
        decision_time: datetime,
        research_range: CoverageRange | None = None,
    ) -> DatasetSnapshot:
        """Hash-bound inner join of base, feature, and canonical label rows.

        Labels are canonical rows from the label dataset referenced by the
        snapshot; only rows whose ``label_available_time`` is at or before
        ``decision_time`` are joined, so a trainer never observes a label that
        could not have been known at decision time. Every referenced content
        hash is verified against the snapshot's pinned catalog entry before the
        frame is returned.
        """
        if snapshot.labels is None:
            raise ValueError("snapshot has no label-panel reference")
        range_ = research_range or snapshot.execution_range

        base, features, feature_manifest = self._read_base_and_features(
            snapshot, feature_set=feature_set, decision_time=decision_time,
            research_range=range_,
        )
        if snapshot.base_panel is None:
            raise ValueError("snapshot has no base-panel reference")
        if snapshot.features is None:
            raise ValueError("snapshot has no feature-panel reference")
        base_manifest = self.base_store.read_manifest(snapshot.base_panel.name)
        _assert_content_hash_matches(base_manifest, snapshot.base_panel)
        _assert_content_hash_matches(feature_manifest, snapshot.features)

        label_columns = [
            c
            for c in self.label_store.content_columns(snapshot.labels.name)
            if c not in ("instrument_id", "session")
        ]
        if not label_columns:
            raise ValueError(f"label panel {snapshot.labels.name} exposes no label columns")
        labels = self.read_labels_bounded(
            snapshot.labels,
            decision_time,
            research_range=range_,
            columns=("instrument_id", "session", *label_columns),
        )

        if feature_set == CANONICAL_FEATURE_SET:
            return self._compose_net_alpha_snapshot(
                snapshot,
                decision_time=decision_time,
                research_range=range_,
                base=base,
                features=features,
                labels=labels,
            )

        available_column = (
            "label_available_time_5d"
            if "label_available_time_5d" in labels.columns
            else ("label_available_time" if "label_available_time" in labels.columns else None)
        )
        if available_column is not None:
            labels = labels.filter(
                pl.col(available_column).is_not_null()
                & (pl.col(available_column) <= decision_time)
            )

        composed = (
            base.join(features, on=["instrument_id", "session"], how="inner")
            .join(labels, on=["instrument_id", "session"], how="inner")
            .sort(["instrument_id", "session"])
        )
        if composed.is_empty():
            raise ValueError("labeled snapshot composition produced no rows")

        label_manifest = self.label_store.read_manifest(snapshot.labels.name)
        _assert_content_hash_matches(label_manifest, snapshot.labels)
        if feature_set == STOCK_ALPHA_V2_FEATURE_SET:
            self._assert_v2_composition(
                snapshot, composed, feature_manifest, label_manifest, decision_time
            )
        from src.core.datasets import DatasetManifest as _DatasetManifest

        merged_manifest = _DatasetManifest(
            asset_kind=feature_manifest.asset_kind,
            schema_version=feature_manifest.schema_version,
            schema_hash=_composed_schema_hash(feature_set, feature_manifest),
            provider_version=feature_manifest.provider_version,
            universe_policy_version=feature_manifest.universe_policy_version,
            universe_policy_hash=feature_manifest.universe_policy_hash,
            feature_set=feature_manifest.feature_set,
            feature_set_hash=feature_manifest.feature_set_hash,
            label_definition=label_manifest.label_definition,
            label_horizon_sessions=label_manifest.label_horizon_sessions,
            time_start=feature_manifest.time_start,
            time_end=feature_manifest.time_end,
            generated_time=feature_manifest.generated_time,
            row_count=composed.height,
            certification=feature_manifest.certification,
            calendar_hash=feature_manifest.calendar_hash,
            corporate_action_hash=feature_manifest.corporate_action_hash,
            cost_source_hash=feature_manifest.cost_source_hash,
            master_hash=feature_manifest.master_hash,
            quality_report_hash=feature_manifest.quality_report_hash,
            content_hash=feature_manifest.content_hash,
            storage_layout=feature_manifest.storage_layout,
            reference_notional=label_manifest.reference_notional,
        )
        return DatasetSnapshot(manifest=merged_manifest, frame=composed)

    def _compose_net_alpha_snapshot(
        self,
        snapshot: ResearchDataSnapshot,
        *,
        decision_time: datetime,
        research_range: CoverageRange,
        base: pl.DataFrame,
        features: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> DatasetSnapshot:
        """Net-alpha composition that preserves the full decision score universe.

        The base-plus-feature keys are retained with a LEFT join to the long
        label/status data: a feature key is never dropped because it lacks an
        outcome, so the score universe cannot be silently narrowed by the
        result. When the hash-bound ``outcome_status`` sidecar exists it is
        joined first as the decision spine, giving exactly one typed status row
        per ``(instrument_id, session, horizon_sessions)``; otherwise labels
        are left-joined and ``compose_net_alpha_training_data`` derives
        statuses from label availability.
        """
        if snapshot.labels is None:
            raise ValueError("snapshot has no label-panel reference")
        if snapshot.features is None:
            raise ValueError("snapshot has no feature-panel reference")
        decision_universe = base.join(
            features, on=["instrument_id", "session"], how="inner"
        ).sort(["instrument_id", "session"])
        if decision_universe.is_empty():
            raise ValueError("net-alpha composition produced no decision rows")

        label_manifest = self.label_store.read_manifest(snapshot.labels.name)
        _assert_content_hash_matches(label_manifest, snapshot.labels)

        status = self._read_outcome_status_sidecar(
            snapshot.outcome_status, decision_time, research_range
        )
        if status is not None and not status.is_empty():
            self._validate_status_spine(
                status, labels, certification=snapshot.manifest.certification
            )
            composed = decision_universe.join(
                status, on=["instrument_id", "session"], how="left"
            )
            join_keys = (
                ["instrument_id", "session", HORIZON_COLUMN]
                if HORIZON_COLUMN in labels.columns
                else ["instrument_id", "session"]
            )
            composed = composed.join(
                labels, on=join_keys, how="left"
            ).sort(["instrument_id", "session"])
            evidence = self._read_outcome_evidence(
                snapshot.outcome_evidence, decision_time, research_range
            )
            if evidence is not None and not evidence.is_empty():
                if HORIZON_COLUMN in labels.columns:
                    composed = composed.join(
                        evidence,
                        on=["instrument_id", "session", HORIZON_COLUMN],
                        how="left",
                    ).sort(["instrument_id", "session"])
                else:
                    logger.warning(
                        "snapshot %s pins outcome evidence but its label panel is "
                        "not long-format; evidence cannot be bound per horizon",
                        snapshot.manifest.snapshot_id,
                    )
        else:
            legacy_status_id = f"{snapshot.labels.name}{OUTCOME_STATUS_DATASET_SUFFIX}"
            logger.info(
                "net-alpha composition uses legacy-inferred statuses: no pinned "
                "OUTCOME_STATUS reference (legacy sibling %s)",
                legacy_status_id,
            )
            composed = decision_universe.join(
                labels, on=["instrument_id", "session"], how="left"
            ).sort(["instrument_id", "session"])
        if composed.is_empty():
            raise ValueError("net-alpha composition produced no rows")

        feature_manifest = self.feature_store.read_manifest(snapshot.features.name)
        from src.core.datasets import DatasetManifest as _DatasetManifest

        merged_manifest = _DatasetManifest(
            asset_kind=feature_manifest.asset_kind,
            schema_version=feature_manifest.schema_version,
            schema_hash=_composed_schema_hash(
                CANONICAL_FEATURE_SET, feature_manifest
            ),
            provider_version=feature_manifest.provider_version,
            universe_policy_version=feature_manifest.universe_policy_version,
            universe_policy_hash=feature_manifest.universe_policy_hash,
            feature_set=feature_manifest.feature_set,
            feature_set_hash=feature_manifest.feature_set_hash,
            label_definition=label_manifest.label_definition,
            label_horizon_sessions=label_manifest.label_horizon_sessions,
            time_start=feature_manifest.time_start,
            time_end=feature_manifest.time_end,
            generated_time=feature_manifest.generated_time,
            row_count=composed.height,
            certification=feature_manifest.certification,
            calendar_hash=feature_manifest.calendar_hash,
            corporate_action_hash=feature_manifest.corporate_action_hash,
            cost_source_hash=feature_manifest.cost_source_hash,
            master_hash=feature_manifest.master_hash,
            quality_report_hash=feature_manifest.quality_report_hash,
            content_hash=feature_manifest.content_hash,
            storage_layout=feature_manifest.storage_layout,
            reference_notional=label_manifest.reference_notional,
        )
        return DatasetSnapshot(manifest=merged_manifest, frame=composed)

    def _read_outcome_status_sidecar(
        self,
        status_entry: CatalogEntry | None,
        decision_time: datetime,
        research_range: CoverageRange,
    ) -> pl.DataFrame | None:
        """Read the declared, hash-checked outcome-status sidecar, if pinned.

        Reads exactly the OUTCOME_STATUS reference the snapshot manifest
        declares and verifies its content hash before returning rows. Returns
        ``None`` when the snapshot has no pinned reference (a legacy snapshot),
        so the caller falls back to legacy-inferred provenance that is
        diagnostic-only and never promoted.
        """
        if status_entry is None:
            return None
        manifest = self.label_store.read_manifest(status_entry.name)
        _assert_content_hash_matches(manifest, status_entry)
        return self.label_store.read_bounded(
            status_entry.name,
            AssetKind.STOCK,
            LABEL_FEATURE_SET,
            decision_time,
            session_start=research_range.start,
            session_end=research_range.end,
            columns=["instrument_id", "session", HORIZON_COLUMN, OUTCOME_STATUS_COLUMN],
        )

    def _read_outcome_evidence(
        self,
        evidence_entry: CatalogEntry | None,
        decision_time: datetime,
        research_range: CoverageRange,
    ) -> pl.DataFrame | None:
        """Read the declared, hash-checked outcome-evidence artifact, if pinned.

        Reads exactly the OUTCOME_EVIDENCE reference the snapshot manifest
        declares and verifies its content hash before returning the compact
        per-key ``(resolution_kind, policy_hash)`` projection. Returns ``None``
        when no evidence is pinned, in which case confirmed no-bars cannot be
        distinguished from collection gaps and the snapshot cannot certify.
        """
        if evidence_entry is None:
            return None
        manifest = self.label_store.read_manifest(evidence_entry.name)
        _assert_content_hash_matches(manifest, evidence_entry)
        frame = self.label_store.read_bounded(
            evidence_entry.name,
            AssetKind.STOCK,
            OUTCOME_EVIDENCE_FEATURE_SET,
            decision_time,
            session_start=research_range.start,
            session_end=research_range.end,
            columns=list(_OUTCOME_EVIDENCE_PROJECTION),
        )
        if frame.is_empty():
            return frame
        return frame.with_columns(
            pl.col("session").cast(pl.Datetime("us", "UTC"))
        )

    def _validate_status_spine(
        self,
        status: pl.DataFrame,
        labels: pl.DataFrame,
        *,
        certification: DatasetCertification,
    ) -> None:
        """Fail closed unless the pinned status spine is complete and duplicate-free.

        The sidecar must emit exactly one row per ``(instrument_id, session,
        horizon_sessions)`` decision key, carry only vocabulary statuses, and
        cover every horizon the label panel declares. Certified execution is
        rejected when any decision key, horizon coverage, or status value is
        missing or inconsistent.
        """
        from src.stocks.ml.contracts import OUTCOME_STATUS_VOCABULARY

        identity = ["instrument_id", "session", HORIZON_COLUMN]
        duplicates = status.group_by(identity).len().filter(pl.col("len") > 1)
        if not duplicates.is_empty():
            raise ValueError(
                "outcome-status sidecar must emit exactly one row per "
                f"(instrument_id, session, horizon_sessions); "
                f"{duplicates.height} keys are duplicated"
            )
        unknown = status.filter(
            ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(OUTCOME_STATUS_VOCABULARY))
        )
        if not unknown.is_empty():
            raise ValueError("outcome-status sidecar contains unknown states")
        if certification is DatasetCertification.PROVISIONAL:
            return
        if HORIZON_COLUMN in labels.columns and HORIZON_COLUMN in status.columns:
            label_horizons = set(labels[HORIZON_COLUMN].unique().to_list())
            status_horizons = set(status[HORIZON_COLUMN].unique().to_list())
            if label_horizons and not label_horizons <= status_horizons:
                raise ValueError(
                    f"outcome-status sidecar horizon coverage {sorted(status_horizons)} "
                    f"is missing label horizons {sorted(label_horizons - status_horizons)}"
                )

    def _assert_v2_composition(
        self,
        snapshot: ResearchDataSnapshot,
        composed: pl.DataFrame,
        feature_manifest: DatasetManifest,
        label_manifest: DatasetManifest,
        decision_time: datetime,
    ) -> None:
        """Fail closed unless the v2/v3 composition satisfies the frozen contract.

        The feature manifest must declare ``stock_alpha_v2`` with a contract
        hash equal to an allowlist-built contract book; exactly the 34 ordered
        ``feature__`` columns must be declared and joined. The label manifest
        must be either the single-horizon ``residual_o2o_5d`` (horizon 5) or the
        multi-horizon ``residual_o2o_multi_5_10_15d`` panel whose ordered
        per-horizon residual/relevance/availability columns all pass the label
        contract. The composed frame must have no empty result, no duplicate
        ``(instrument_id, session)``, and no label unavailable at
        ``decision_time``.
        """
        if feature_manifest.feature_set != STOCK_ALPHA_V2_FEATURE_SET:
            raise ValueError(
                f"v2 composition requires feature manifest stock_alpha_v2, got "
                f"{feature_manifest.feature_set!r}"
            )
        if snapshot.features is None:
            raise ValueError("v2 composition has no feature-panel reference")
        contract_path = (
            self.feature_store.root / snapshot.features.name / "feature_contract.json"
        )
        if not contract_path.is_file():
            raise ValueError(f"v2 feature panel has no feature_contract.json: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        book = feature_contract_book_from_allowlist(
            STOCK_ALPHA_V2_FEATURE_SET, stock_alpha_v2_allowlist()
        )
        expected_hash = book.schema_hash
        stored_hash = str(contract.get("feature_set_hash", ""))
        if stored_hash != expected_hash:
            raise ValueError(
                f"v2 feature contract hash mismatch: stored {stored_hash}, "
                f"expected {expected_hash}"
            )
        expected_feature_columns = ["instrument_id", "session"] + [
            f"feature__{name}" for name in stock_alpha_v2_allowlist()
        ]
        feature_columns = [
            c for c in composed.columns if c.startswith("feature__")
        ]
        if feature_columns != expected_feature_columns[2:]:
            raise ValueError(
                f"v2 composition feature columns do not match the ordered allowlist: "
                f"got {feature_columns}"
            )
        if label_manifest.label_definition == V2_LABEL_DEFINITION:
            self._assert_single_horizon_labels(
                composed, label_manifest, decision_time,
            )
        elif label_manifest.label_definition == MULTI_HORIZON_LABEL_DEFINITION:
            self._assert_multi_horizon_labels(
                composed, label_manifest, decision_time,
            )
        else:
            raise ValueError(
                f"v2 composition unsupported label definition "
                f"{label_manifest.label_definition!r}"
            )

    def _assert_single_horizon_labels(
        self,
        composed: pl.DataFrame,
        label_manifest: DatasetManifest,
        decision_time: datetime,
    ) -> None:
        """Fail closed unless the v2 single-horizon label contract holds."""
        if label_manifest.label_horizon_sessions != V2_LABEL_HORIZON:
            raise ValueError(
                f"v2 composition requires label horizon {V2_LABEL_HORIZON}, "
                f"got {label_manifest.label_horizon_sessions}"
            )
        missing_label_columns = [
            c for c in V2_REQUIRED_LABEL_COLUMNS if c not in composed.columns
        ]
        if missing_label_columns:
            raise ValueError(
                f"v2 composition missing required label columns {missing_label_columns}"
            )
        duplicates = (
            composed.group_by(["instrument_id", "session"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError("v2 composition has duplicate (instrument_id, session) rows")
        non_finite = composed.filter(
            pl.col(V2_LABEL_DEFINITION).is_not_null()
            & ~pl.col(V2_LABEL_DEFINITION).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError("v2 composition contains non-finite residual labels")
        relevance = composed[V2_REQUIRED_LABEL_COLUMNS[1]]
        if relevance.null_count():
            raise ValueError("v2 composition relevance must be integer within 0..4")
        relevance_values = [
            float(value) for value in relevance.to_list() if value is not None
        ]
        if relevance_values and (
            min(relevance_values) < 0 or max(relevance_values) > 4
        ):
            raise ValueError("v2 composition relevance must be integer within 0..4")
        if composed[V2_REQUIRED_LABEL_COLUMNS[2]].null_count():
            raise ValueError("v2 composition joined a row without label_available_time")
        unavailable = composed.filter(
            pl.col(V2_REQUIRED_LABEL_COLUMNS[2]) > decision_time
        )
        if not unavailable.is_empty():
            raise ValueError("v2 composition joined an unavailable label row")

    def _assert_multi_horizon_labels(
        self,
        composed: pl.DataFrame,
        label_manifest: DatasetManifest,
        decision_time: datetime,
    ) -> None:
        """Fail closed unless the v3 multi-horizon label contract holds.

        Exactly the ordered ``(residual, relevance, availability)`` column
        triples for each supported horizon must be present; every residual must
        be finite, every relevance must be integer ``0..4``, every availability
        column must be non-null, and no row may carry an availability time
        after ``decision_time``. Rows without any label value are excluded by
        the inner join before this assertion runs.
        """
        if label_manifest.label_horizon_sessions != V2_LABEL_HORIZON:
            raise ValueError(
                "v3 composition requires control label horizon "
                f"{V2_LABEL_HORIZON}, got {label_manifest.label_horizon_sessions}"
            )
        present = [c for c in MULTI_HORIZON_LABEL_COLUMNS if c in composed.columns]
        if present != list(MULTI_HORIZON_LABEL_COLUMNS):
            raise ValueError(
                "v3 composition requires the ordered multi-horizon label columns "
                f"{list(MULTI_HORIZON_LABEL_COLUMNS)}, missing/extra {present}"
            )
        duplicates = (
            composed.group_by(["instrument_id", "session"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError("v3 composition has duplicate (instrument_id, session) rows")
        for h in MULTI_HORIZON_HORIZONS:
            residual = f"residual_o2o_{h}d"
            relevance = f"relevance_{h}d"
            available = f"label_available_time_{h}d"
            non_finite = composed.filter(
                pl.col(residual).is_not_null() & ~pl.col(residual).is_finite()
            )
            if not non_finite.is_empty():
                raise ValueError(
                    f"v3 composition contains non-finite residual labels in {residual}"
                )
            relevance_values = [
                float(value)
                for value in composed[relevance].to_list()
                if value is not None
            ]
            if relevance_values and (
                min(relevance_values) < 0 or max(relevance_values) > 4
            ):
                raise ValueError(
                    f"v3 composition relevance {relevance} must be integer 0..4"
                )
            if composed[available].null_count():
                raise ValueError(
                    f"v3 composition joined a row without {available}"
                )
        unavailable = composed.filter(
            pl.any_horizontal(
                pl.col(f"label_available_time_{h}d") > decision_time
                for h in MULTI_HORIZON_HORIZONS
            )
        )
        if not unavailable.is_empty():
            raise ValueError(
                "v3 composition joined a row with an unavailable horizon label"
            )

    def compose_training_snapshot(
        self,
        snapshot: ResearchDataSnapshot,
        *,
        feature_set: str,
        decision_time: datetime,
        research_range: CoverageRange | None = None,
    ) -> DatasetSnapshot:
        """Compose a training frame from the snapshot's base + feature versions.

        Execution prices and OHLCV come from the referenced base panel; the
        reusable feature columns come from the referenced feature panel. The
        result manifest is the feature panel's manifest so evidence hashes and
        the feature set are preserved.
        """
        if snapshot.base_panel is None:
            raise ValueError("snapshot has no base-panel reference")
        if snapshot.features is None:
            raise ValueError("snapshot has no feature-panel reference")
        range_ = research_range or snapshot.execution_range

        base = self.read_base_bounded(
            snapshot.base_panel, decision_time, research_range=range_
        )
        features = self.read_features_bounded(
            snapshot.features, feature_set, decision_time, research_range=range_
        )
        feature_manifest = self.feature_store.read_manifest(snapshot.features.name)
        if feature_manifest.feature_set != feature_set:
            raise ValueError(
                f"snapshot feature panel {snapshot.features.name} has feature_set "
                f"{feature_manifest.feature_set!r}, requested {feature_set!r}"
            )

        composed = base.join(
            features, on=["instrument_id", "session"], how="inner"
        ).sort(["instrument_id", "session"])
        if composed.is_empty():
            raise ValueError("snapshot composition produced no rows")
        return DatasetSnapshot(manifest=feature_manifest, frame=composed)

    def compose_labeled_training_data(
        self,
        lineage: object,
        feature_set: str,
        decision_time: datetime,
    ) -> object:
        """Compose labeled training data from resolved lineage (snapshotless path).

        Reads base, feature, and label panels from the catalog entries in
        ``lineage``, applies point-in-time filtering, and returns a
        ``ResearchDataBundle`` with the composed frame, lineage, and outcome
        coverage. This is the new direct-resolution path that replaces
        snapshot-based composition.
        """
        from src.stocks.data.lineage import ResearchDataBundle, ResolvedDataLineage

        if not isinstance(lineage, ResolvedDataLineage):
            raise TypeError("lineage must be a ResolvedDataLineage")
        entries = lineage.entries
        base_entry = entries.get("base_panel")
        feature_entry = entries.get("features")
        label_entry = entries.get("labels")
        if base_entry is None:
            raise ValueError("resolved lineage has no base_panel entry")
        if feature_entry is None:
            raise ValueError("resolved lineage has no features entry")
        if label_entry is None:
            raise ValueError("resolved lineage has no labels entry")

        base = self.read_base_bounded(
            base_entry, decision_time, research_range=lineage.research_range
        )
        features = self.read_features_bounded(
            feature_entry, feature_set, decision_time,
            research_range=lineage.research_range,
        )
        label_columns = [
            c
            for c in self.label_store.content_columns(label_entry.name)
            if c not in ("instrument_id", "session")
        ]
        labels = self.read_labels_bounded(
            label_entry,
            decision_time,
            research_range=lineage.research_range,
            columns=("instrument_id", "session", *label_columns),
        )

        available_column = (
            "label_available_time_5d"
            if "label_available_time_5d" in labels.columns
            else ("label_available_time" if "label_available_time" in labels.columns else None)
        )
        if available_column is not None:
            labels = labels.filter(
                pl.col(available_column).is_not_null()
                & (pl.col(available_column) <= decision_time)
            )

        composed = (
            base.join(features, on=["instrument_id", "session"], how="inner")
            .join(labels, on=["instrument_id", "session"], how="inner")
            .sort(["instrument_id", "session"])
        )
        if composed.is_empty():
            raise ValueError("labeled training data composition produced no rows")

        return ResearchDataBundle(
            frame=composed,
            manifest=self.feature_store.read_manifest(feature_entry.name),
            lineage=lineage,
            outcome_coverage=lineage.outcome_coverage,
        )


def _assert_content_hash_matches(
    manifest: DatasetManifest,
    entry: CatalogEntry,
) -> None:
    """Fail closed unless the stored dataset hash equals the catalog pin."""
    if not manifest.content_hash:
        raise ValueError(f"stored dataset {entry.name} has no content_hash")
    if manifest.content_hash != entry.content_hash:
        raise ValueError(
            f"stored dataset {entry.name} content hash {manifest.content_hash} "
            f"does not match catalog pin {entry.content_hash}"
        )


def _composed_schema_hash(
    feature_set: str,
    feature_manifest: DatasetManifest,
) -> str:
    """Bind the composed snapshot schema to the frozen v2 feature contract.

    The v2 predictor contract is the ordered stock_alpha_v2 allowlist; its
    contract-hash is what scoring requests must reproduce, so the composed
    manifest's ``schema_hash`` carries it instead of a column-list hash.
    """
    if feature_set != "stock_alpha_v2":
        return feature_manifest.schema_hash
    book = feature_contract_book_from_allowlist(
        feature_set, stock_alpha_v2_allowlist()
    )
    return book.schema_hash
