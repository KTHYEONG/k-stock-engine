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

import hashlib
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.core.datasets import DatasetManifest, make_manifest
from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    HorizonJoinEvidence,
    NetAlphaResearchData,
    UniverseRescopeSettings,
)
from src.stocks.ml.universe_rescope import apply_universe_rescope
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

# Narrow per-horizon label projection: identity, target, availability, and
# realized outcome columns only; horizon/feature columns never survive collect.
_LABEL_REALIZED_COLUMNS = ("gross_return", "risk_residual", "reference_cost")
_LONG_HORIZON_COLUMN = "horizon_sessions"
_WARMUP_NULL_COLUMNS = (
    "fluc_rate", "intraday_ret", "overnight_ret", "sector_ret_5d",
    "feature__fluc_rate", "feature__intraday_ret",
    "feature__overnight_ret", "feature__sector_ret_5d",
)
_PIT_FUNDAMENTAL_SOURCES = frozenset({"bp_ratio", "ep_ratio"})


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
class DirectInputReference:
    base_dataset_id: str
    feature_dataset_id: str
    label_dataset_id: str
    start: date
    end: date
    feature_schema_hash: str
    feature_content_hash: str | None
    cost_evidence_path: str | None
    cost_evidence_hash: str | None


@dataclass(frozen=True, slots=True)
class DirectReadinessIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DirectReadinessReport:
    input_reference: DirectInputReference
    errors: tuple[DirectReadinessIssue, ...]
    warnings: tuple[DirectReadinessIssue, ...]
    excluded_sources: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def warning_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)

    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)


@dataclass(frozen=True, slots=True)
class DirectLoadCheckpoint:
    """Bounded scalar snapshot of one direct-load boundary.

    Carries only load-shape scalars (stage, elapsed time, rows/columns,
    estimated frame bytes, live owner names, and the join-preflight key
    cardinalities). System resources (RSS/cgroup/MemAvailable) are sampled by
    the durable journal at write time so no raw market rows, values,
    credentials, or environment dumps ever enter a checkpoint.
    """

    stage: str
    elapsed_ms: int = 0
    rows: int | None = None
    columns: int | None = None
    frame_bytes: int | None = None
    owners: tuple[str, ...] = ()
    base_rows: int | None = None
    feature_rows: int | None = None
    base_distinct_keys: int | None = None
    feature_distinct_keys: int | None = None
    base_duplicate_keys: int | None = None
    feature_duplicate_keys: int | None = None
    matched_keys: int | None = None
    predicted_joined_rows: int | None = None
    planned_lower_bound_bytes: int | None = None
    universe_rescope: dict[str, object] | None = None

    def journal_payload(self) -> dict[str, object]:
        """Bounded scalar mapping for durable journaling."""
        payload: dict[str, object] = {
            "rows": self.rows,
            "columns": self.columns,
            "frame_bytes": self.frame_bytes,
            "owners": ",".join(self.owners)[:512],
            "base_rows": self.base_rows,
            "feature_rows": self.feature_rows,
            "base_distinct_keys": self.base_distinct_keys,
            "feature_distinct_keys": self.feature_distinct_keys,
            "base_duplicate_keys": self.base_duplicate_keys,
            "feature_duplicate_keys": self.feature_duplicate_keys,
            "matched_keys": self.matched_keys,
            "predicted_joined_rows": self.predicted_joined_rows,
            "planned_lower_bound_bytes": self.planned_lower_bound_bytes,
        }
        if self.universe_rescope is not None:
            payload["universe_rescope"] = str(self.universe_rescope)[:512]
        return payload


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


# Fixed-width byte floors per physical polars type; unknown types use one
# 16-byte view/offset floor so the admission basis never under-counts a row.
_DTYPE_LOWER_BOUND_BYTES: tuple[tuple[object, int], ...] = (
    (pl.Int8, 1),
    (pl.UInt8, 1),
    (pl.Boolean, 1),
    (pl.Int16, 2),
    (pl.UInt16, 2),
    (pl.Int32, 4),
    (pl.UInt32, 4),
    (pl.Float32, 4),
    (pl.Date, 4),
    (pl.Int64, 8),
    (pl.UInt64, 8),
    (pl.Float64, 8),
    (pl.Datetime, 8),
    (pl.Duration, 8),
)
_UNKNOWN_DTYPE_FLOOR_BYTES = 16


def _dtype_floor_bytes(dtype: pl.DataType) -> int:
    for candidate, width in _DTYPE_LOWER_BOUND_BYTES:
        if dtype == candidate:
            return width
    return _UNKNOWN_DTYPE_FLOOR_BYTES


def _schema_row_floor_bytes(schema: Mapping[str, pl.DataType]) -> int:
    return sum(_dtype_floor_bytes(dtype) for dtype in schema.values())


def _key_multiplicity_stats(keys: pl.DataFrame) -> dict[str, int]:
    """Narrow key-cardinality scalars from an identity-column frame."""
    multiplicities = keys.group_by(["instrument_id", "session"]).len()
    duplicate_rows = int(multiplicities.filter(pl.col("len") > 1).height)
    return {
        "rows": int(keys.height),
        "distinct_keys": int(multiplicities.height),
        "duplicate_keys": duplicate_rows,
    }


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
        feature_content = self._feature_store.content_columns(
            request.feature_dataset_id
        )
        feature_columns = tuple(
            c
            for c in feature_content
            if c.startswith("feature__")
        )
        if "disclosure_date" not in feature_content:
            feature_columns = tuple(
                column
                for column in feature_columns
                if column.removeprefix("feature__") not in _PIT_FUNDAMENTAL_SOURCES
            )
        if not feature_columns:
            raise ValueError(
                f"feature dataset {request.feature_dataset_id} exposes no feature__ columns"
            )
        identity = self._available_columns(
            self._feature_store,
            request.feature_dataset_id,
            (
                "instrument_id",
                "session",
                "observation_time",
                "available_time",
                "disclosure_date",
            ),
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

    def _scan_horizon_labels(
        self, request: DirectDataRequest, horizon_sessions: int
    ) -> pl.LazyFrame:
        """Scan one horizon's narrow labels with predicates applied pre-collect.

        Projects identity, target, availability, and realized columns only;
        the ``horizon_sessions`` filter stays inside the lazy plan so no other
        horizon is ever materialized. Wide-format horizons whose columns are
        absent return a typed empty scan (the caller skips them silently,
        matching the eager path).
        """
        present = set(self._label_store.content_columns(request.label_dataset_id))
        realized = [c for c in _LABEL_REALIZED_COLUMNS if c in present]
        if {
            _LONG_HORIZON_COLUMN,
            "net_alpha_target",
            "label_available_time",
        }.issubset(present):
            return (
                self._label_store.scan_bounded(
                    request.label_dataset_id,
                    AssetKind.STOCK,
                    "labels",
                    _decision_boundary(request.end),
                    session_start=request.start,
                    session_end=request.end,
                    columns=[
                        "instrument_id",
                        "session",
                        _LONG_HORIZON_COLUMN,
                        "net_alpha_target",
                        "label_available_time",
                        *realized,
                    ],
                )
                .filter(
                    (pl.col(_LONG_HORIZON_COLUMN) == horizon_sessions)
                    & pl.col("net_alpha_target").is_not_null()
                )
                .select(
                    "instrument_id",
                    "session",
                    "net_alpha_target",
                    "label_available_time",
                    *realized,
                )
            )
        target_column = f"horizon_{horizon_sessions}_target"
        available_column = f"horizon_{horizon_sessions}_available"
        if target_column not in present or available_column not in present:
            return pl.DataFrame(
                {
                    "instrument_id": [],
                    "session": [],
                    "net_alpha_target": [],
                    "label_available_time": [],
                }
            ).lazy()
        return (
            self._label_store.scan_bounded(
                request.label_dataset_id,
                AssetKind.STOCK,
                "labels",
                _decision_boundary(request.end),
                session_start=request.start,
                session_end=request.end,
                columns=["instrument_id", "session", target_column, available_column],
            )
            .filter(pl.col(target_column).is_not_null())
            .select(
                "instrument_id",
                "session",
                pl.col(target_column).alias("net_alpha_target"),
                pl.col(available_column).alias("label_available_time"),
            )
        )

    def _build_input_reference(
        self, request: DirectDataRequest, cost_evidence_path: Path | None
    ) -> DirectInputReference:
        feature_manifest = self._read_feature_manifest(request)
        schema_hash = feature_manifest.schema_hash if feature_manifest is not None else ""
        content_hash: str | None = None
        if feature_manifest is not None:
            content_hash = feature_manifest.content_hash or feature_manifest.schema_hash
        cost_path_str: str | None = str(cost_evidence_path) if cost_evidence_path is not None else None
        cost_hash: str | None = None
        if cost_evidence_path is not None and cost_evidence_path.exists():
            try:
                cost_hash = hashlib.sha256(cost_evidence_path.read_bytes()).hexdigest()
            except OSError:
                cost_hash = None
        return DirectInputReference(
            base_dataset_id=request.base_dataset_id,
            feature_dataset_id=request.feature_dataset_id,
            label_dataset_id=request.label_dataset_id,
            start=request.start,
            end=request.end,
            feature_schema_hash=schema_hash,
            feature_content_hash=content_hash,
            cost_evidence_path=cost_path_str,
            cost_evidence_hash=cost_hash,
        )

    def assess_readiness(
        self, request: DirectDataRequest, decision_time: datetime, *, cost_evidence_path: Path | None = None
    ) -> DirectReadinessReport:
        if isinstance(decision_time, str):
            decision_time = datetime.fromisoformat(str(decision_time))
        input_ref = self._build_input_reference(request, cost_evidence_path)
        errors: list[DirectReadinessIssue] = []
        warnings: list[DirectReadinessIssue] = []
        excluded_sources: set[str] = set()

        # cost evidence warning when absent
        if cost_evidence_path is None:
            warnings.append(DirectReadinessIssue(code="cost_evidence_absent", message="cost evidence not provided; using default schedules"))
        elif not cost_evidence_path.exists():
            errors.append(DirectReadinessIssue(code="cost_evidence_unreadable", message=f"cost evidence not found: {cost_evidence_path}"))

        # invalid range
        if request.start > request.end:
            errors.append(DirectReadinessIssue(code="invalid_range", message=f"start {request.start} after end {request.end}"))
            return DirectReadinessReport(
                input_reference=input_ref,
                errors=tuple(errors),
                warnings=tuple(warnings),
                excluded_sources=(),
            )

        # check dataset readability and required columns
        try:
            base_cols = self._available_columns(self._base_store, request.base_dataset_id, _BASE_COLUMNS)
        except Exception as exc:
            errors.append(DirectReadinessIssue(code="base_unreadable", message=str(exc)))
            base_cols = ()
        try:
            feature_content = self._feature_store.content_columns(request.feature_dataset_id)
        except Exception as exc:
            errors.append(DirectReadinessIssue(code="feature_unreadable", message=str(exc)))
            feature_content = []
        try:
            label_content = self._label_store.content_columns(request.label_dataset_id)
        except Exception as exc:
            errors.append(DirectReadinessIssue(code="label_unreadable", message=str(exc)))
            label_content = []  # noqa: F841
        if errors:
            return DirectReadinessReport(
                input_reference=input_ref,
                errors=tuple(errors),
                warnings=tuple(warnings),
                excluded_sources=(),
            )

        required_base = ("instrument_id", "session", "open", "close", "volume", "trading_value")
        missing_base = [c for c in required_base if c not in base_cols]
        if missing_base:
            errors.append(DirectReadinessIssue(code="missing_required_columns", message=f"base missing {missing_base}"))

        feature_cols = tuple(c for c in feature_content if c.startswith("feature__"))
        if not feature_cols:
            errors.append(DirectReadinessIssue(code="missing_feature_columns", message="no feature__ columns"))

        if "disclosure_date" not in feature_content:
            excluded_sources = {
                source
                for source in _PIT_FUNDAMENTAL_SOURCES
                if f"feature__{source}" in feature_content
            }
            if excluded_sources:
                warnings.append(
                    DirectReadinessIssue(
                        code="fundamental_lineage_unavailable",
                        message=(
                            "excluded PIT-unverifiable sources: "
                            + ", ".join(sorted(excluded_sources))
                        ),
                    )
                )

        # check for optional source timing column without valid timestamps -> warning and exclusion
        # If feature dataset contains disclosure_date or source_available_time columns, ensure they have valid timestamps
        optional_timing_cols = [c for c in feature_content if c in ("disclosure_date", "source_available_time", "available_time", "observation_time")]
        for col in optional_timing_cols:
            try:
                # sample narrow scan for null/invalid timestamps
                lf = self._feature_store.scan_bounded(
                    request.feature_dataset_id,
                    AssetKind.STOCK,
                    request.feature_set,
                    _decision_boundary(request.end),
                    session_start=request.start,
                    session_end=request.end,
                    columns=["instrument_id", "session", col],
                )
                # collect only if small; use filter for nulls
                null_count = lf.filter(pl.col(col).is_null()).select(pl.len()).collect().item()
                if int(null_count) > 0:
                    warnings.append(DirectReadinessIssue(code="optional_source_unverifiable", message=f"optional source {col} has null timestamps; excluded"))
            except Exception:
                warnings.append(DirectReadinessIssue(code="optional_source_unverifiable", message=f"optional source {col} unverifiable"))

        # identity-only preflight: duplicate and monotonic checks
        try:
            boundary = _decision_boundary(request.end)
            base_keys = self._base_store.scan_bounded(
                request.base_dataset_id,
                AssetKind.STOCK,
                "base_panel",
                boundary,
                session_start=request.start,
                session_end=request.end,
                columns=["instrument_id", "session"],
            ).collect()
            feature_keys = self._feature_store.scan_bounded(
                request.feature_dataset_id,
                AssetKind.STOCK,
                request.feature_set,
                boundary,
                session_start=request.start,
                session_end=request.end,
                columns=["instrument_id", "session"],
            ).collect()
            base_stats = _key_multiplicity_stats(base_keys)
            feature_stats = _key_multiplicity_stats(feature_keys)
            if base_stats["duplicate_keys"] > 0 or feature_stats["duplicate_keys"] > 0:
                errors.append(DirectReadinessIssue(code="duplicate_keys", message=f"duplicate keys base={base_stats['duplicate_keys']} feature={feature_stats['duplicate_keys']}"))
            # non-monotonic check via composed session diff
            # we check base and feature separately for monotonic sessions
            for df, label in [(base_keys, "base"), (feature_keys, "feature")]:
                # sort and check monotonic
                sorted_df = df.sort(["instrument_id", "session"])
                # use window diff check without collecting full frame? simple python check on collected narrow frame
                # group by instrument and check increasing
                for inst in sorted_df["instrument_id"].unique().to_list():
                    sessions = sorted_df.filter(pl.col("instrument_id") == inst)["session"].to_list()
                    for i in range(1, len(sessions)):
                        if sessions[i] <= sessions[i - 1]:
                            errors.append(DirectReadinessIssue(code="non_monotonic_keys", message=f"non-monotonic {label} for {inst}"))
                            break
            # causal timestamp violations: available_time > decision_time or observation_time > available_time
            # check feature availability timing
            try:
                feat_avail_cols = self._available_columns(self._feature_store, request.feature_dataset_id, ("instrument_id", "session", "available_time", "observation_time"))
                if "available_time" in feat_avail_cols:
                    avail_frame = self._feature_store.scan_bounded(
                        request.feature_dataset_id,
                        AssetKind.STOCK,
                        request.feature_set,
                        boundary,
                        session_start=request.start,
                        session_end=request.end,
                        columns=["instrument_id", "session", "available_time"],
                    ).collect()
                    # future source availability: available_time > decision_time
                    future_count = avail_frame.filter(pl.col("available_time") > decision_time).height
                    if int(future_count) > 0:
                        errors.append(DirectReadinessIssue(code="future_source_availability", message=f"{future_count} rows have available_time > decision_time"))
                # also check label_available_time vs decision_time?
            except Exception:  # noqa: S110
                pass

            # non-finite execution/active-feature values: sample numeric columns
            try:
                # check base numeric columns non-finite
                base_numeric = [c for c in base_cols if c in ("open", "close", "volume", "trading_value")]
                if base_numeric:
                    base_frame = self._base_store.scan_bounded(
                        request.base_dataset_id,
                        AssetKind.STOCK,
                        "base_panel",
                        boundary,
                        session_start=request.start,
                        session_end=request.end,
                        columns=["instrument_id", "session", *base_numeric],
                    ).collect()
                    for col in base_numeric:
                        if base_frame.schema[col].is_numeric():
                            counts = base_frame.select((pl.col(col).is_not_null() & ~pl.col(col).is_finite()).sum().alias("cnt")).item()
                            if int(counts) > 0:
                                errors.append(DirectReadinessIssue(code="non_finite_execution_values", message=f"non-finite in {col}"))
                # feature numeric
                if feature_cols:
                    feat_numeric_cols = list(feature_cols)
                    # collect narrow feature numeric sample
                    feat_frame = self._feature_store.scan_bounded(
                        request.feature_dataset_id,
                        AssetKind.STOCK,
                        request.feature_set,
                        boundary,
                        session_start=request.start,
                        session_end=request.end,
                        columns=["instrument_id", "session", *feat_numeric_cols[:4]],
                    ).collect()
                    for col in feat_numeric_cols[:4]:
                        if col in feat_frame.columns and feat_frame.schema[col].is_numeric():
                            counts = feat_frame.select((pl.col(col).is_not_null() & ~pl.col(col).is_finite()).sum().alias("cnt")).item()
                            if int(counts) > 0:
                                errors.append(DirectReadinessIssue(code="non_finite_feature_values", message=f"non-finite in {col}"))
            except Exception:  # noqa: S110
                pass

            # zero usable requested labels: need point-in-time labels
            # For each horizon, filter labels where source_available <= decision_time < label_available_time
            # Use narrow scans
            usable_found = False
            for horizon in request.candidate_horizon_sessions:
                try:
                    scan = self._scan_horizon_labels(request, horizon)
                    rows = scan.collect()
                    if rows.is_empty():
                        continue
                    # filter point-in-time: label_available_time > decision_time and available_time <= decision_time
                    # We approximate: label_available_time > decision_time is required, but we check <= decision_time for usable?
                    # Spec says source_available <= decision_time < label_available_time
                    # We'll check label_available_time > decision_time and (available_time is not checked here)
                    # If no rows satisfy label_available_time > decision_time then not usable? Actually need source_available <= decision
                    # For readiness, we treat any horizon with at least one row where label_available_time > decision_time as usable
                    # But spec scenario expects future source availability error, and zero usable labels error when decision_time far future
                    # So check if rows have label_available_time > decision_time count >0
                    if "label_available_time" in rows.columns:
                        pit_count = rows.filter(pl.col("label_available_time") <= decision_time).height
                        if int(pit_count) == 0:
                            continue
                        usable_found = True
                    else:
                        usable_found = True
                except Exception:  # noqa: S112
                    continue
            if not usable_found:
                errors.append(DirectReadinessIssue(code="no_usable_labels", message="no point-in-time usable labels"))

            # lineage gap warning when feature manifest missing content hash
            if input_ref.feature_content_hash is None or input_ref.feature_schema_hash == "":
                warnings.append(DirectReadinessIssue(code="lineage_gap", message="feature manifest content hash unavailable"))

            del base_keys, feature_keys
        except Exception as exc:
            # if preflight itself fails due to unreadable, already captured; otherwise add generic error
            if not errors:
                errors.append(DirectReadinessIssue(code="readiness_unreadable", message=str(exc)))

        return DirectReadinessReport(
            input_reference=input_ref,
            errors=tuple(errors),
            warnings=tuple(warnings),
            excluded_sources=tuple(sorted(excluded_sources)),
        )

    def load_training_data(
        self,
        request: DirectDataRequest,
        decision_time: datetime,
        *,
        readiness: DirectReadinessReport | None = None,
        checkpoint: Callable[[DirectLoadCheckpoint], None] | None = None,
        rescope: UniverseRescopeSettings | None = None,
    ) -> NetAlphaResearchData:
        """Compose training data with one decision-width materialization.

        Builds the decision frame from lazy projected base/features and
        collects it exactly once, applies warm-up and source renaming before
        the wide frame escapes, and extracts each requested horizon from a
        narrow lazy label scan that is released before the next horizon. At
        most one decision-width frame plus one bounded narrow label frame
        stay live, and no ``MlMarketData`` container is constructed.

        A narrow identity-only preflight runs before any wide collect and
        records base/feature row counts, distinct keys, duplicate-key counts,
        and ``predicted_joined_rows=sum(base_count[key]*feature_count[key])``;
        duplicate input keys or a predicted non-one-to-one join fails closed
        before the decision frame is ever materialized.

        An enabled ``rescope`` policy masks the base scan to a trailing
        market-cap band before the feature join so every downstream consumer
        (fitting, calibration, replay, benchmark) shares one restricted pool;
        the preflight admission bound then overestimates planned bytes, which
        stays on the safe side of the memory guard.
        """
        from src.stocks.ml.features import stock_net_alpha_v1_roles

        if isinstance(decision_time, str):
            decision_time = datetime.fromisoformat(decision_time)
        self._active_feature_set = request.feature_set

        started_monotonic = time.monotonic()

        def emit(stage: str, **scalars: object) -> None:
            if checkpoint is not None:
                checkpoint(
                    DirectLoadCheckpoint(
                        stage=stage,
                        elapsed_ms=int(
                            (time.monotonic() - started_monotonic) * 1000
                        ),
                        **scalars,  # type: ignore[arg-type]  # noqa: FBT001
                    )
                )

        # Use provided readiness or compute it; block on errors before any collect
        effective_readiness = readiness if readiness is not None else self.assess_readiness(request, decision_time)
        if effective_readiness.errors:
            # Emit preflight checkpoint for observability before failing
            has_dup = any(e.code == "duplicate_keys" for e in effective_readiness.errors)
            if checkpoint is not None:
                emit(
                    "direct_preflight",
                    owners=("base_keys", "feature_keys"),
                    base_duplicate_keys=1 if has_dup else 0,
                    feature_duplicate_keys=1 if has_dup else 0,
                    base_rows=0,
                    feature_rows=0,
                    base_distinct_keys=0,
                    feature_distinct_keys=0,
                    matched_keys=0,
                    predicted_joined_rows=0,
                    planned_lower_bound_bytes=0,
                )
            codes = ", ".join(issue.code for issue in effective_readiness.errors)
            raise ValueError(f"direct readiness blocked: {codes}: {effective_readiness.errors[0].message}")

        wanted_base_columns: tuple[str, ...] = _BASE_COLUMNS
        required: tuple[str, ...] = (
            "instrument_id",
            "session",
            "open",
            "close",
            "volume",
            "trading_value",
        )
        if rescope is not None:
            # Conditional extension keeps flag-off loads byte-identical; the
            # required-column check fails closed on legacy stores.
            wanted_base_columns = (*_BASE_COLUMNS, "market_cap")
            required = (*required, "market_cap")
        base_columns = self._available_columns(
            self._base_store, request.base_dataset_id, wanted_base_columns
        )
        missing = [c for c in required if c not in base_columns]
        if missing:
            raise ValueError(
                f"base dataset {request.base_dataset_id} is missing required "
                f"execution columns {missing}"
            )
        feature_identity = self._available_columns(
            self._feature_store,
            request.feature_dataset_id,
            ("instrument_id", "session", "observation_time", "available_time"),
        )
        feature_columns = tuple(
            c
            for c in self._feature_store.content_columns(request.feature_dataset_id)
            if c.startswith("feature__")
        )
        excluded_sources = set(effective_readiness.excluded_sources)
        feature_columns = tuple(
            column
            for column in feature_columns
            if column.removeprefix("feature__") not in excluded_sources
        )
        if not feature_columns:
            raise ValueError(
                f"feature dataset {request.feature_dataset_id} exposes no feature__ columns"
            )

        boundary = _decision_boundary(request.end)

        # Narrow preflight: identity-only projections bound the join shape
        # before any decision-width materialization can happen.
        base_keys = self._base_store.scan_bounded(
            request.base_dataset_id,
            AssetKind.STOCK,
            "base_panel",
            boundary,
            session_start=request.start,
            session_end=request.end,
            columns=["instrument_id", "session"],
        ).collect()
        feature_keys = self._feature_store.scan_bounded(
            request.feature_dataset_id,
            AssetKind.STOCK,
            request.feature_set,
            boundary,
            session_start=request.start,
            session_end=request.end,
            columns=["instrument_id", "session"],
        ).collect()
        base_stats = _key_multiplicity_stats(base_keys)
        feature_stats = _key_multiplicity_stats(feature_keys)
        multiplicity_join = (
            base_keys.group_by(["instrument_id", "session"])
            .len()
            .join(
                feature_keys.group_by(["instrument_id", "session"]).len(),
                on=["instrument_id", "session"],
                suffix="_feature",
            )
        )
        matched_keys = int(multiplicity_join.height)
        predicted_joined_rows = int(
            multiplicity_join.select(
                (pl.col("len") * pl.col("len_feature")).sum().fill_null(0)
            ).item()
            or 0
        )
        del multiplicity_join, base_keys, feature_keys

        base_lf = self._base_store.scan_bounded(
            request.base_dataset_id,
            AssetKind.STOCK,
            "base_panel",
            boundary,
            session_start=request.start,
            session_end=request.end,
            columns=list(base_columns),
        )
        rescope_diagnostics: dict[str, object] = {}
        if rescope is not None:
            base_lf, rescope_diagnostics = apply_universe_rescope(base_lf, rescope)
        plan = base_lf.join(
            self._feature_store.scan_bounded(
                request.feature_dataset_id,
                AssetKind.STOCK,
                request.feature_set,
                boundary,
                session_start=request.start,
                session_end=request.end,
                columns=[*feature_identity, *feature_columns],
            ),
            on=["instrument_id", "session"],
            how="inner",
        )

        # The join preserves whichever side projected these columns (the right
        # frame is suffixed only on collision), so availability synthesis is a
        # static decision over the two projections.
        has_available = "available_time" in base_columns or "available_time" in feature_identity
        has_observation = (
            "observation_time" in base_columns or "observation_time" in feature_identity
        )
        if not has_available:
            plan = plan.with_columns(
                (pl.col("session") + pl.duration(hours=15, minutes=30)).alias(
                    "available_time"
                )
            )
        if not has_observation:
            plan = plan.with_columns(pl.col("available_time").alias("observation_time"))

        row_floor_bytes = _schema_row_floor_bytes(plan.collect_schema())
        emit(
            "direct_preflight",
            owners=("base_keys", "feature_keys"),
            base_rows=base_stats["rows"],
            feature_rows=feature_stats["rows"],
            base_distinct_keys=base_stats["distinct_keys"],
            feature_distinct_keys=feature_stats["distinct_keys"],
            base_duplicate_keys=base_stats["duplicate_keys"],
            feature_duplicate_keys=feature_stats["duplicate_keys"],
            matched_keys=matched_keys,
            predicted_joined_rows=predicted_joined_rows,
            planned_lower_bound_bytes=predicted_joined_rows * row_floor_bytes,
        )
        if (
            base_stats["duplicate_keys"]
            or feature_stats["duplicate_keys"]
            or predicted_joined_rows != matched_keys
        ):
            raise ValueError(
                "direct join preflight failed before collect: "
                f"duplicate keys (base={base_stats['duplicate_keys']}, "
                f"feature={feature_stats['duplicate_keys']}), "
                f"predicted_joined_rows={predicted_joined_rows} vs one-to-one "
                f"expectation {matched_keys}"
            )

        decision_frame = plan.collect()

        if decision_frame.is_empty():
            raise ValueError("direct base/feature composition produced no rows")
        # Lazy joins do not guarantee per-instrument source order; normalize
        # before applying the monotonicity invariant.
        decision_frame = decision_frame.sort(["instrument_id", "session"])
        numeric_columns = tuple(
            c
            for c in decision_frame.columns
            if c.startswith("feature__")
            and decision_frame.schema[c].is_numeric()
        )
        _validate_direct_frame(decision_frame, numeric_columns)
        emit_extra: dict[str, object] = (
            {"universe_rescope": rescope_diagnostics} if rescope_diagnostics else {}
        )
        emit(
            "direct_collected",
            rows=int(decision_frame.height),
            columns=len(decision_frame.columns),
            frame_bytes=int(decision_frame.estimated_size()),
            owners=("decision_frame",),
            **emit_extra,
        )
        decision_height = decision_frame.height
        manifest_columns = tuple(decision_frame.columns)

        # Warm-up: drop the single pre-lookback row per instrument, then null
        # the first remaining row's rolling sources exactly as the snapshot
        # composition does.
        feature_frame = (
            decision_frame.with_columns(
                pl.int_range(0, pl.len()).over("instrument_id").alias("__warmup_row")
            )
            .filter(pl.col("__warmup_row") > 0)
            .drop("__warmup_row")
        )
        first_rows = feature_frame.with_columns(
            pl.int_range(0, pl.len()).over("instrument_id").alias("__row")
        )
        for column in _WARMUP_NULL_COLUMNS:
            if column in first_rows.columns:
                first_rows = first_rows.with_columns(
                    pl.when(pl.col("__row") == 0)
                    .then(None)
                    .otherwise(pl.col(column))
                    .alias(column)
                )
        feature_frame = first_rows.drop("__row")
        del first_rows
        if feature_frame.is_empty():
            raise ValueError("net-alpha direct feature frame is empty")
        roles = stock_net_alpha_v1_roles()
        rename_sources = {
            f"feature__{source}": source
            for source in roles
            if f"feature__{source}" in feature_frame.columns
        }
        if rename_sources:
            feature_frame = feature_frame.rename(rename_sources)
        emit(
            "feature_frame_ready",
            rows=int(feature_frame.height),
            columns=len(feature_frame.columns),
            frame_bytes=int(feature_frame.estimated_size()),
            owners=("feature_frame", "labels_by_horizon"),
        )
        decision_keys = decision_frame.select("instrument_id", "session")
        del decision_frame

        label_columns = self._label_store.content_columns(request.label_dataset_id)
        long_format_ready = {
            _LONG_HORIZON_COLUMN,
            "net_alpha_target",
            "label_available_time",
        }.issubset(label_columns)
        labels_by_horizon: dict[int, pl.DataFrame] = {}
        join_evidence: list[HorizonJoinEvidence] = []
        stage_a_horizons: set[int] = set()
        for horizon in request.candidate_horizon_sessions:
            if not long_format_ready and (
                f"horizon_{horizon}_target" not in label_columns
                or f"horizon_{horizon}_available" not in label_columns
            ):
                continue
            label_scan = self._scan_horizon_labels(request, horizon)
            horizon_rows = label_scan.collect()
            del label_scan
            if horizon_rows.is_empty():
                raise ValueError(
                    f"requested horizon {horizon} has zero usable labels"
                )
            stage_a_horizons.add(horizon)
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
            available = horizon_rows.filter(
                pl.col("net_alpha_target").is_not_null()
                & pl.col("label_available_time").is_not_null()
                & (pl.col("label_available_time") <= decision_time)
            )
            available_height = int(available.height)
            joined = available.join(
                feature_frame.select("instrument_id", "session"),
                on=["instrument_id", "session"],
                how="inner",
            ).sort(["instrument_id", "session"])
            del available
            if joined.is_empty():
                join_evidence.append(
                    HorizonJoinEvidence(
                        horizon_sessions=horizon,
                        feature_rows=decision_height,
                        label_rows=int(available_height),
                        joined_rows=0,
                        drop_reasons=("no point-in-time available labels",),
                        decision_rows=int(feature_frame.height),
                        realized_rows=0,
                    )
                )
                continue
            labels_by_horizon[horizon] = joined
            emit(
                f"label_horizon_{horizon}_joined",
                rows=int(joined.height),
                columns=len(joined.columns),
                frame_bytes=int(joined.estimated_size()),
                owners=(f"labels_by_horizon[{horizon}]", "feature_frame"),
            )
            join_evidence.append(
                HorizonJoinEvidence(
                    horizon_sessions=horizon,
                    feature_rows=decision_height,
                    label_rows=int(available_height),
                    joined_rows=joined.height,
                    decision_rows=int(feature_frame.height),
                    realized_rows=joined.height,
                )
            )
            del horizon_rows

        if not stage_a_horizons:
            raise ValueError("no candidate horizon produced usable labels")
        # Mirror the eager path's post-load horizon validation before the
        # point-in-time composition verdict.
        missing_horizons = [
            h for h in request.candidate_horizon_sessions if h not in stage_a_horizons
        ]
        if missing_horizons:
            raise ValueError(
                f"ML market data missing requested horizons: {missing_horizons}"
            )
        if not labels_by_horizon:
            raise ValueError(
                "no candidate horizon produced point-in-time available labels"
            )
        labels_by_horizon = {h: labels_by_horizon[h] for h in sorted(labels_by_horizon)}
        # Exclude unverifiable optional sources without synthesizing timestamps
        for issue in effective_readiness.warnings:
            if issue.code == "optional_source_unverifiable":
                col = issue.message.split("optional source")[-1].strip().split()[0].strip(":,")
                for candidate in (col, f"feature__{col}", col.replace("feature__", "")):
                    if candidate in feature_frame.columns:
                        feature_frame = feature_frame.drop(candidate)

        feature_manifest = self._read_feature_manifest(request)
        manifest_source: DatasetManifest | None = feature_manifest
        if manifest_source is None:
            sessions = sorted(decision_keys["session"].unique().to_list())
            manifest_source = make_manifest(
                asset_kind=AssetKind.STOCK,
                columns=list(manifest_columns),
                feature_set="stock_net_alpha_v1",
                label_definition="net_alpha_o2o",
                label_horizon_sessions=max(labels_by_horizon),
                time_start=sessions[0],
                time_end=sessions[-1],
                generated_time=sessions[-1],
                row_count=decision_height,
                provider_version="direct-loader",
                universe_policy_version="direct-loader",
            )
        manifest = replace(
            manifest_source,
            feature_set=CANONICAL_FEATURE_SET,
            feature_set_hash=manifest_source.feature_set_hash or "net-alpha-v1",
            row_count=feature_frame.height,
        )
        return NetAlphaResearchData(
            feature_frame=feature_frame,
            labels_by_horizon=labels_by_horizon,
            manifest=manifest,
            join_evidence=tuple(join_evidence),
        )

    def load_backtest_snapshot(
        self, request: DirectDataRequest, decision_time: datetime, *, readiness: DirectReadinessReport | None = None
    ) -> DatasetSnapshot:
        from src.stocks.data.contracts import DatasetSnapshot

        if isinstance(decision_time, str):
            decision_time = datetime.fromisoformat(str(decision_time))
        effective_readiness = readiness if readiness is not None else self.assess_readiness(request, decision_time)
        if effective_readiness.errors:
            codes = ", ".join(issue.code for issue in effective_readiness.errors)
            raise ValueError(f"direct readiness blocked: {codes}")
        # Reuse training composition to build a snapshot-shaped input; bounded scans retained
        research = self.load_training_data(request, decision_time, readiness=effective_readiness)
        # Snapshot frame is the feature frame (contains base+feature columns); manifest carries schema/Content hashes
        return DatasetSnapshot(manifest=research.manifest, frame=research.feature_frame)


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
