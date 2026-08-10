"""Stock dataset repository: translates domain requests into the generic store.

The repository enforces ``AssetKind.STOCK`` at its boundary and reuses the
asset-neutral ``ParquetDatasetStore`` for persistence; it is not a duplicate
store implementation. A deterministic migration adapter certifies the legacy
``*_feat.parquet`` files as a provisional, non-promotable point-in-time panel.
"""
from __future__ import annotations

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
from src.stocks.data.labels import LABEL_FEATURE_SET
from src.stocks.research.datasets import (
    ELIGIBLE_STATUS,
    QUALITY_REASON_COLUMN,
    QUALITY_STATUS_COLUMN,
    QUARANTINED_STATUS,
    research_eligible_frame,
)
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.repositories")

PROVISIONAL_PROVIDER_VERSION = "provisional-legacy"
_IDENTITY_COLUMNS = ("date", "ticker")
_OHLC_COLUMNS = ("open", "high", "low", "close")
_TARGET_PREFIXES = ("target_", "label_")


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
    snapshot = SnapshotResolver(store).resolve(snapshot_id)
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
        range_ = research_range or snapshot.research_range

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
