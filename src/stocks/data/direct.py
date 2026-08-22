"""Direct market data loader for ML training and backtesting.

Bypasses the snapshot/catalog system and reads directly from the specified
base, feature, and label dataset directories using manifest-verified bounded
partitions with lazy projection and predicate pushdown before collect.

The composed contract is physically separated:

- ``MlMarketData.frame`` carries exactly one sorted row per
  ``(instrument_id, session)`` of base/feature columns only.
- ``MlMarketData.labels_by_horizon`` owns independent narrow per-horizon label
  frames; no label row ever duplicates a feature row across horizons.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.stocks.ml.contracts import CANONICAL_FEATURE_SET
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.direct")

_BASE_COLUMNS = (
    "instrument_id",
    "session",
    "observation_time",
    "available_time",
    "open",
    "close",
    "volume",
    "trading_value",
    "sector",
)


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

    Carries the feature frame (exactly one row per ``(instrument_id,
    session)``), independent per-horizon narrow label frames, and explicit
    input identifiers for provenance. ``feature_manifest`` is the immutable
    schema/identity of the selected feature dataset so a training manifest can
    bind its ``feature_schema_hash`` to the exact feature dataset that produced
    the model matrix, and ``input_content_hashes`` preserves the raw
    feature/content identity of every input dataset for certified replay and
    fail-closed schema/lineage checks.
    """

    frame: pl.DataFrame
    labels_by_horizon: Mapping[int, pl.DataFrame]
    input_ids: Mapping[str, str] = field(default_factory=dict)
    feature_manifest: DatasetManifest | None = None
    input_content_hashes: Mapping[str, str] = field(default_factory=dict)


def _decision_boundary(end: date) -> datetime:
    """Far-future coverage boundary: bounded reads accept any dataset end."""
    return datetime.max.replace(tzinfo=UTC)


class DirectMarketDataLoader:
    """Loads market data directly from specified dataset directories.

    Reads only the requested base, feature, and label dataset directories
    through manifest-verified bounded partition scans. No catalog, snapshot,
    or lineage resolution occurs.
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
        self._active_feature_set: str = CANONICAL_FEATURE_SET

    def load(self, request: DirectDataRequest) -> MlMarketData:
        """Load and compose separated market data from the specified datasets.

        Reads bounded projected partitions of the base, feature, and label
        datasets, validates identity/finiteness on the composed decision
        frame, and returns the one-row-per-key feature frame plus independent
        per-horizon label frames.

        Args:
            request: the direct data request specifying datasets and range.

        Returns:
            Composed MlMarketData with the separated feature frame and
            per-horizon labels.

        Raises:
            ValueError: on duplicate keys, missing columns, non-finite values,
                source-order violations, or a requested horizon with zero
                usable labels.
        """
        base = self._read_base(request)
        features = self._read_features(request)
        labels = self._read_labels(request)

        composed = self._compose(base, features, request)
        labels_by_horizon = self._extract_horizons(
            composed, labels, request.candidate_horizon_sessions
        )

        input_ids = {
            "base_dataset_id": request.base_dataset_id,
            "feature_dataset_id": request.feature_dataset_id,
            "label_dataset_id": request.label_dataset_id,
        }
        feature_manifest = self._read_feature_manifest(request)
        input_content_hashes = {
            "base_dataset_id": request.base_dataset_id,
            "feature_dataset_id": request.feature_dataset_id,
            "label_dataset_id": request.label_dataset_id,
        }
        if feature_manifest is not None:
            input_content_hashes["feature_content_hash"] = (
                feature_manifest.content_hash or feature_manifest.schema_hash
            )
            input_content_hashes["feature_schema_hash"] = feature_manifest.schema_hash

        return MlMarketData(
            frame=composed,
            labels_by_horizon=labels_by_horizon,
            input_ids=input_ids,
            feature_manifest=feature_manifest,
            input_content_hashes=input_content_hashes,
        )

    def _read_feature_manifest(
        self, request: DirectDataRequest
    ) -> DatasetManifest | None:
        """Read the immutable feature dataset manifest, or ``None`` if absent.

        A missing feature manifest is not a loader failure: the training CLI
        treats it as a fail-closed schema-parity gap rather than fabricating a
        hash.
        """
        try:
            return self._feature_store.read_manifest(request.feature_dataset_id)
        except FileNotFoundError:
            return None

    def _read_bounded_projection(
        self,
        dataset_id: str,
        columns: tuple[str, ...],
        start: date,
        end: date,
        predicates: tuple[pl.Expr, ...] = (),
        store: ParquetDatasetStore | None = None,
        expected_feature_set: str | None = None,
    ) -> pl.DataFrame:
        """Read only partitions intersecting ``[start, end]`` with pushdown.

        Delegates to :meth:`ParquetDatasetStore.read_bounded`, which verifies
        the selected partitions against the dataset content metadata, scans
        lazily with projection and session-range pushdown, and collects before
        any extra predicate is applied to the already-bounded result.
        """
        store_for_id = store or (
            self._base_store
            if (self._base_store.root / dataset_id).exists()
            else (
                self._feature_store
                if (self._feature_store.root / dataset_id).exists()
                else self._label_store
            )
        )
        frame = store_for_id.read_bounded(
            dataset_id,
            AssetKind.STOCK,
            expected_feature_set or self._active_feature_set,
            _decision_boundary(end),
            session_start=start,
            session_end=end,
            columns=list(columns),
        )
        for predicate in predicates:
            frame = frame.filter(predicate)
        key_columns = [c for c in ("instrument_id", "session") if c in columns]
        return frame.sort(key_columns or list(columns[:1]))

    def _available_columns(self, store: ParquetDatasetStore, dataset_id: str, wanted: tuple[str, ...]) -> tuple[str, ...]:
        present = set(store.content_columns(dataset_id))
        return tuple(c for c in wanted if c in present)

    def _read_base(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read base dataset with bounded session range."""
        self._active_feature_set = request.feature_set
        columns = self._available_columns(
            self._base_store, request.base_dataset_id, _BASE_COLUMNS
        )
        required = ("instrument_id", "session", "open", "close", "volume", "trading_value")
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(
                f"base dataset {request.base_dataset_id} is missing required "
                f"execution columns {missing}"
            )
        return self._read_bounded_projection(
            request.base_dataset_id, columns, request.start, request.end,
            store=self._base_store,
            expected_feature_set="base_panel",
        )

    def _read_features(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read feature dataset with bounded session range."""
        self._active_feature_set = request.feature_set
        feature_columns = tuple(
            c
            for c in self._feature_store.content_columns(request.feature_dataset_id)
            if c.startswith("feature__")
        )
        if not feature_columns:
            raise ValueError(
                f"feature dataset {request.feature_dataset_id} exposes no feature__ columns"
            )
        identity = self._available_columns(
            self._feature_store,
            request.feature_dataset_id,
            ("instrument_id", "session", "observation_time", "available_time"),
        )
        columns = (*identity, *feature_columns)
        return self._read_bounded_projection(
            request.feature_dataset_id, columns, request.start, request.end,
            store=self._feature_store,
            expected_feature_set=request.feature_set,
        )

    def _read_labels(self, request: DirectDataRequest) -> pl.DataFrame:
        """Read label dataset with bounded range and horizon pushdown."""
        self._active_feature_set = request.feature_set
        all_columns = self._label_store.content_columns(request.label_dataset_id)
        label_columns = tuple(c for c in all_columns if c not in ("instrument_id", "session"))
        if not label_columns:
            raise ValueError(
                f"label dataset {request.label_dataset_id} exposes no label columns"
            )
        columns = ("instrument_id", "session", *label_columns)
        predicates: tuple[pl.Expr, ...] = ()
        if "horizon_sessions" in columns:
            # Predicate applied to the bounded collected scan; only the
            # requested candidates ever reach composition.
            predicates = (
                pl.col("horizon_sessions").is_in(
                    list(request.candidate_horizon_sessions)
                ),
            )
        return self._read_bounded_projection(
            request.label_dataset_id, columns, request.start, request.end,
            predicates=predicates,
            store=self._label_store,
            expected_feature_set="labels",
        )

    def _compose(
        self,
        base: pl.DataFrame,
        features: pl.DataFrame,
        request: DirectDataRequest,
    ) -> pl.DataFrame:
        """Inner-join base and features on identity keys, then validate.

        The composed frame stays label-free: exactly one sorted row per
        ``(instrument_id, session)``. Duplicate keys, non-finite feature
        values, and source-order monotonicity violations fail closed through
        one bounded aggregate/window plan.
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

        numeric_columns = tuple(
            c
            for c in decision_frame.columns
            if c.startswith("feature__")
            and decision_frame.schema[c].is_numeric()
        )
        _validate_direct_frame(decision_frame, numeric_columns)

        return decision_frame.sort(["instrument_id", "session"])

    def _extract_horizons(
        self,
        composed: pl.DataFrame,
        labels: pl.DataFrame,
        candidate_horizons: tuple[int, ...],
    ) -> dict[int, pl.DataFrame]:
        """Extract independent narrow per-horizon label frames.

        Each horizon's rows are filtered straight from the label table (never
        by re-joining duplicated feature rows) and semi-joined to the composed
        decision keys so label universes stay aligned with the feature panel.
        """
        labels_by_horizon: dict[int, pl.DataFrame] = {}
        has_long_format = {"horizon_sessions", "net_alpha_target"}.issubset(
            labels.columns
        )

        for horizon in candidate_horizons:
            if has_long_format and "label_available_time" in labels.columns:
                label_columns = [
                    "instrument_id",
                    "session",
                    pl.col("net_alpha_target").alias("target"),
                    "label_available_time",
                ]
                label_columns.extend(
                    optional_column
                    for optional_column in (
                        "gross_return",
                        "risk_residual",
                        "reference_cost",
                    )
                    if optional_column in labels.columns
                )
                horizon_rows = labels.filter(
                    (pl.col("horizon_sessions") == horizon)
                    & pl.col("net_alpha_target").is_not_null()
                ).select(label_columns)
            else:
                target_column = f"horizon_{horizon}_target"
                available_column = f"horizon_{horizon}_available"
                if target_column not in labels.columns:
                    continue
                if available_column not in labels.columns:
                    continue
                horizon_rows = labels.filter(
                    pl.col(target_column).is_not_null()
                ).select(
                    "instrument_id",
                    "session",
                    pl.col(target_column).alias("target"),
                    pl.col(available_column).alias("label_available_time"),
                )

            if horizon_rows.is_empty():
                raise ValueError(
                    f"requested horizon {horizon} has zero usable labels"
                )
            duplicates = (
                horizon_rows.group_by(["instrument_id", "session"])
                .len()
                .filter(pl.col("len") > 1)
            )
            if not duplicates.is_empty():
                raise ValueError(
                    f"duplicate label keys at horizon {horizon}: "
                    f"{duplicates.height} (instrument_id, session) pairs"
                )
            joined = horizon_rows.join(
                composed.select("instrument_id", "session"),
                on=["instrument_id", "session"],
                how="semi",
            ).sort(["instrument_id", "session"])
            if joined.is_empty():
                raise ValueError(
                    f"requested horizon {horizon} has zero usable labels"
                )
            labels_by_horizon[horizon] = joined

        if not labels_by_horizon:
            raise ValueError(
                "no candidate horizon produced usable labels"
            )

        return labels_by_horizon


def _validate_direct_frame(
    frame: pl.DataFrame, numeric_columns: tuple[str, ...]
) -> None:
    """Fail closed on duplicate keys, disorder, and non-finite features.

    Uses exactly one aggregate projection for uniqueness/finiteness and one
    window expression for source-order monotonicity, so validation cost is
    independent of the instrument count.
    """
    unique_keys = int(frame.select(pl.struct(["instrument_id", "session"]).n_unique()).item())
    if unique_keys != frame.height:
        raise ValueError(
            f"duplicate feature/base keys: {frame.height - unique_keys} "
            "(instrument_id, session) pairs are duplicated"
        )

    non_monotonic = frame.filter(
        pl.col("session").diff().over("instrument_id") <= timedelta(0)
    )
    if not non_monotonic.is_empty():
        offending = non_monotonic.row(0, named=True)
        raise ValueError(
            f"non-monotonic session ordering for {offending['instrument_id']}: "
            f"{offending['session']}"
        )

    if numeric_columns:
        counts = frame.select(
            [
                (
                    pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                )
                .sum()
                .alias(column)
                for column in numeric_columns
            ]
        ).row(0, named=True)
        for column, count in counts.items():
            if int(count) > 0:
                raise ValueError(
                    f"non-finite numeric values in feature column {column}: "
                    f"{int(count)} rows"
                )
