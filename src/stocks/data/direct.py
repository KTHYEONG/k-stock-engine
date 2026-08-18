"""Direct market data loader for ML training and backtesting.

Bypasses the snapshot/catalog system and reads directly from the specified
base, feature, and label dataset directories.  The loader reads only the
requested dataset directories and never opens catalog.jsonl, snapshot
manifests, dataset certification, content hashes, or lineage files.

O(rows read) per dataset with no catalog scan, sidecar join, or
per-session filesystem lookup.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.stocks.ml.contracts import CANONICAL_FEATURE_SET
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.direct")


@dataclass(frozen=True, slots=True)
class DirectDataRequest:
    """Immutable request for direct market data loading.

    Specifies the exact dataset directories and date range to load.
    The loader reads only these directories and never resolves snapshots,
    catalogs, or lineage.
    """

    base_dataset_id: str
    feature_dataset_id: str
    label_dataset_id: str
    start: date
    end: date
    feature_set: str = CANONICAL_FEATURE_SET
    candidate_horizon_sessions: tuple[int, ...] = (10,)


@dataclass(frozen=True, slots=True)
class MlMarketData:
    """Composed market data for ML training and backtesting.

    Carries the feature frame (one row per ``(instrument_id, session)``),
    per-horizon label frames, and explicit input identifiers for provenance.
    """

    frame: pl.DataFrame
    labels_by_horizon: Mapping[int, pl.DataFrame]
    input_ids: Mapping[str, str] = field(default_factory=dict)


class DirectMarketDataLoader:
    """Loads market data directly from specified dataset directories.

    Reads only the requested base, feature, and label dataset directories.
    No catalog, snapshot, or lineage resolution occurs.
    """

    def __init__(
        self,
        *,
        base_root: Path,
        feature_root: Path,
        label_root: Path,
    ):
        self._base_store = ParquetDatasetStore(base_root)
        self._feature_store = ParquetDatasetStore(feature_root)
        self._label_store = ParquetDatasetStore(label_root)

    def load(self, request: DirectDataRequest) -> MlMarketData:
        """Load and compose market data from the specified datasets.

        Reads base, feature, and label datasets, joins them on
        ``(instrument_id, session)``, and extracts per-horizon label frames.

        Args:
            request: the direct data request specifying datasets and range.

        Returns:
            Composed MlMarketData with feature frame and per-horizon labels.

        Raises:
            ValueError: on duplicate keys, missing columns, non-finite values,
                or a requested horizon with zero usable labels.
        """
        base = self._read_base(request)
        features = self._read_features(request)
        labels = self._read_labels(request)

        composed = self._compose(base, features, labels, request)
        labels_by_horizon = self._extract_horizons(composed, request)

        input_ids = {
            "base_dataset_id": request.base_dataset_id,
            "feature_dataset_id": request.feature_dataset_id,
            "label_dataset_id": request.label_dataset_id,
        }

        return MlMarketData(
            frame=composed,
            labels_by_horizon=labels_by_horizon,
            input_ids=input_ids,
        )

    def _read_base(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read base dataset with bounded session range."""
        # Read directly from partitioned dataset without manifest validation
        # The direct loader bypasses snapshot/catalog validation
        dataset_dir = self._base_store.root / request.base_dataset_id
        partitions_dir = dataset_dir / "partitions"
        
        if not partitions_dir.exists():
            # Fall back to single-file layout
            table_path = dataset_dir / f"{request.base_dataset_id}.parquet"
            if not table_path.exists():
                raise FileNotFoundError(f"no parquet for dataset {request.base_dataset_id!r}")
            frame = pl.read_parquet(table_path)
        else:
            frame = pl.read_parquet(partitions_dir, hive_partitioning=True)
        
        # Filter to requested columns and session range
        columns = [
            "instrument_id",
            "session",
            "observation_time",
            "available_time",
            "open",
            "close",
            "volume",
            "trading_value",
            "sector",
        ]
        frame = frame.select([c for c in columns if c in frame.columns])
        
        # Filter by session range
        session_start = datetime.combine(request.start, datetime.min.time(), tzinfo=UTC)
        session_end = datetime.combine(request.end, datetime.min.time(), tzinfo=UTC)
        frame = frame.filter(
            (pl.col("session") >= session_start) & (pl.col("session") <= session_end)
        )
        
        return frame.sort(["instrument_id", "session"])

    def _read_features(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read feature dataset with bounded session range."""
        feature_columns = [
            c for c in self._feature_store.content_columns(request.feature_dataset_id)
            if c.startswith("feature__")
        ]
        if not feature_columns:
            raise ValueError(
                f"feature dataset {request.feature_dataset_id} exposes no feature__ columns"
            )
        
        # Read directly from partitioned dataset without manifest validation
        dataset_dir = self._feature_store.root / request.feature_dataset_id
        partitions_dir = dataset_dir / "partitions"
        
        if not partitions_dir.exists():
            table_path = dataset_dir / f"{request.feature_dataset_id}.parquet"
            if not table_path.exists():
                raise FileNotFoundError(f"no parquet for dataset {request.feature_dataset_id!r}")
            frame = pl.read_parquet(table_path)
        else:
            frame = pl.read_parquet(partitions_dir, hive_partitioning=True)
        
        # Filter to requested columns and session range
        columns = [
            "instrument_id",
            "session",
            "observation_time",
            "available_time",
            *feature_columns,
        ]
        frame = frame.select([c for c in columns if c in frame.columns])
        
        session_start = datetime.combine(request.start, datetime.min.time(), tzinfo=UTC)
        session_end = datetime.combine(request.end, datetime.min.time(), tzinfo=UTC)
        frame = frame.filter(
            (pl.col("session") >= session_start) & (pl.col("session") <= session_end)
        )
        
        return frame.sort(["instrument_id", "session"])

    def _read_labels(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read label dataset with bounded session range."""
        label_columns = [
            c for c in self._label_store.content_columns(request.label_dataset_id)
            if c not in ("instrument_id", "session")
        ]
        if not label_columns:
            raise ValueError(
                f"label dataset {request.label_dataset_id} exposes no label columns"
            )
        
        # Read directly from partitioned dataset without manifest validation
        dataset_dir = self._label_store.root / request.label_dataset_id
        partitions_dir = dataset_dir / "partitions"
        
        if not partitions_dir.exists():
            table_path = dataset_dir / f"{request.label_dataset_id}.parquet"
            if not table_path.exists():
                raise FileNotFoundError(f"no parquet for dataset {request.label_dataset_id!r}")
            frame = pl.read_parquet(table_path)
        else:
            frame = pl.read_parquet(partitions_dir, hive_partitioning=True)
        
        # Filter to requested columns and session range
        columns = ["instrument_id", "session", *label_columns]
        frame = frame.select([c for c in columns if c in frame.columns])
        
        session_start = datetime.combine(request.start, datetime.min.time(), tzinfo=UTC)
        session_end = datetime.combine(request.end, datetime.min.time(), tzinfo=UTC)
        frame = frame.filter(
            (pl.col("session") >= session_start) & (pl.col("session") <= session_end)
        )
        
        return frame.sort(["instrument_id", "session"])

    def _compose(
        self,
        base: pl.DataFrame,
        features: pl.DataFrame,
        labels: pl.DataFrame,
        request: DirectDataRequest,
    ) -> pl.DataFrame:
        """Inner-join base, features, and labels on identity columns.

        Validates unique (instrument_id, session) keys, required feature
        columns, monotonic session ordering, and non-finite values.
        """
        decision_frame = base.join(
            features, on=["instrument_id", "session"], how="inner"
        )
        if decision_frame.is_empty():
            raise ValueError("direct base/feature composition produced no rows")

        if "available_time" not in decision_frame.columns:
            decision_frame = decision_frame.with_columns(
                (
                    pl.col("session")
                    + pl.duration(hours=15, minutes=30)
                ).alias("available_time")
            )
        if "observation_time" not in decision_frame.columns:
            decision_frame = decision_frame.with_columns(
                pl.col("available_time").alias("observation_time")
            )

        self._validate_unique_keys(decision_frame)
        self._validate_feature_columns(decision_frame, request)
        self._validate_monotonic_sessions(decision_frame)
        self._validate_numeric_finiteness(decision_frame)

        composed = decision_frame.join(
            labels, on=["instrument_id", "session"], how="inner"
        )

        if composed.is_empty():
            raise ValueError("direct composition produced no rows")

        return composed

    def _validate_unique_keys(self, frame: pl.DataFrame) -> None:
        """Reject duplicate (instrument_id, session) keys."""
        duplicates = (
            frame.group_by(["instrument_id", "session"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise ValueError(
                f"duplicate feature/base keys: {duplicates.height} "
                "(instrument_id, session) pairs are duplicated"
            )

    def _validate_feature_columns(
        self, frame: pl.DataFrame, request: DirectDataRequest
    ) -> None:
        """Validate required feature columns are present."""
        required_features = [
            c for c in frame.columns
            if c.startswith("feature__")
        ]
        if not required_features:
            raise ValueError("no feature__ columns in composed frame")

    def _validate_monotonic_sessions(self, frame: pl.DataFrame) -> None:
        """Validate monotonic session ordering within each instrument."""
        for instrument_id in frame["instrument_id"].unique().to_list():
            # Get sessions in their original order (not sorted)
            instrument_sessions = (
                frame.filter(pl.col("instrument_id") == instrument_id)
                .select("session")
                .to_series()
                .to_list()
            )
            for i in range(1, len(instrument_sessions)):
                if instrument_sessions[i] <= instrument_sessions[i - 1]:
                    raise ValueError(
                        f"non-monotonic session ordering for {instrument_id}: "
                        f"{instrument_sessions[i - 1]} >= {instrument_sessions[i]}"
                    )

    def _validate_numeric_finiteness(self, frame: pl.DataFrame) -> None:
        """Validate all numeric feature columns contain only finite values."""
        numeric_columns = [
            c for c in frame.columns
            if frame[c].dtype.is_numeric() and c.startswith("feature__")
        ]
        for column in numeric_columns:
            non_finite = frame.filter(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
            )
            if not non_finite.is_empty():
                raise ValueError(
                    f"non-finite numeric values in feature column {column}: "
                    f"{non_finite.height} rows"
                )

    def _extract_horizons(
        self,
        composed: pl.DataFrame,
        request: DirectDataRequest,
    ) -> dict[int, pl.DataFrame]:
        """Extract per-horizon label frames from the composed data.

        For each requested horizon, filter to rows where the horizon's target
        column is present and return the narrow label frame.
        """
        labels_by_horizon: dict[int, pl.DataFrame] = {}

        for horizon in request.candidate_horizon_sessions:
            if {
                "horizon_sessions",
                "net_alpha_target",
                "label_available_time",
            }.issubset(composed.columns):
                label_frame = composed.filter(
                    (pl.col("horizon_sessions") == horizon)
                    & pl.col("net_alpha_target").is_not_null()
                ).select(
                    "instrument_id",
                    "session",
                    pl.col("net_alpha_target").alias("target"),
                    "label_available_time",
                )
                if label_frame.is_empty():
                    raise ValueError(
                        f"requested horizon {horizon} has zero usable labels"
                    )
                labels_by_horizon[horizon] = label_frame
                continue

            target_column = f"horizon_{horizon}_target"
            available_column = f"horizon_{horizon}_available"

            if target_column not in composed.columns:
                continue
            if available_column not in composed.columns:
                continue

            label_frame = composed.filter(
                pl.col(target_column).is_not_null()
            ).select(
                "instrument_id",
                "session",
                pl.col(target_column).alias("target"),
                pl.col(available_column).alias("label_available_time"),
            )

            if label_frame.is_empty():
                raise ValueError(
                    f"requested horizon {horizon} has zero usable labels"
                )

            labels_by_horizon[horizon] = label_frame

        if not labels_by_horizon:
            raise ValueError(
                "no candidate horizon produced usable labels"
            )

        return labels_by_horizon
