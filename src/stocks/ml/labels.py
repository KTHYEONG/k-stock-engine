"""Partitioned net-alpha label dataset: continuous, cost/risk-aware, long format.

The label dataset is published as a single long-format dataset partitioned by
``horizon_sessions``. Each horizon keeps its own universe; horizons are never
inner-joined into a common universe, so the trainer records per-horizon label
universes instead of shrinking the sample. The target is the session-robust
continuous net residual:

``net_alpha_target(i,t,h) = robust_zscore_session((adjusted_open(i,t+h) /
adjusted_open(i,t+1)) - 1 - point_in_time_risk_projection(i,t,h) -
reference_round_trip_cost(i,t,h))``

The label frame schema is ``instrument_id, decision_session, horizon_sessions,
net_alpha_target, label_available_time, gross_return, risk_residual,
reference_cost``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    DatasetManifest,
    make_manifest,
)
from src.core.instruments import AssetKind
from src.stocks.data.labels import (
    COST_AWARE_GROSS_PREFIX,
    COST_AWARE_REFERENCE_COST_PREFIX,
    COST_AWARE_RISK_RESIDUAL_PREFIX,
    LABEL_FEATURE_SET,
    NET_ALPHA_ALGORITHM_VERSION,
    NET_ALPHA_CONTINUOUS_SUFFIX,
    NET_ALPHA_DEFINITION,
    NET_ALPHA_TARGET_PREFIX,
    build_net_alpha_label_dataset_with_status,
)
from src.stocks.data.quality import CorporateActionSnapshot, KRXSessionCalendar
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.stocks.ml.contracts import (
    OUTCOME_STATUS_COLUMN,
    OUTCOME_STATUS_VOCABULARY,
)
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
HORIZON_COLUMN = "horizon_sessions"
TARGET_COLUMN = "net_alpha_target"
AVAILABLE_COLUMN = "label_available_time"
GROSS_COLUMN = "gross_return"
RISK_RESIDUAL_COLUMN = "risk_residual"
REFERENCE_COST_COLUMN = "reference_cost"
REALIZED_RETURN_COLUMN = "realized_net_return"


def _horizon_suffix(horizon_sessions: int) -> str:
    return f"{horizon_sessions}d"


def build_partitioned_net_alpha_labels(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    horizon_sessions: tuple[int, ...],
    reference_notional: float,
    policy: ExecutionOutcomePolicy | None = None,
    bar_evidence: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build one long, partitioned net-alpha label dataset for all candidate horizons.

    Each horizon is built independently with the same policy kernel as the
    exact replay: no common-universe inner join is performed. The result is a
    long frame keyed by ``(instrument_id, decision_session, horizon_sessions)``
    with the continuous net-alpha target, the exact exit-open
    ``label_available_time``, the gross open-to-open return, the risk residual,
    and the reference round-trip cost.

    Args:
        base_panel: adjusted base panel carrying identity, OHLC, sector, ADTV,
            market_cap, beta, and volatility columns.
        calendar: the KRX session calendar used for entry/exit alignment.
        cost_schedule: the reference cost schedule.
        liquidity_model: the point-in-time liquidity slippage model.
        horizon_sessions: ascending candidate horizons, each strictly positive.
        reference_notional: the reference notional for round-trip cost.
        policy: the immutable execution policy applied to every horizon
            (default ``scheduled_open_v1``).

    Returns:
        A long ``pl.DataFrame`` with ``horizon_sessions`` partition column whose
        unique values equal the requested horizon set.

    Raises:
        ValueError: for an empty/unsorted/duplicated horizon set, a non-positive
            reference notional, or an unsupported horizon.
    """
    labels, _status = build_partitioned_net_alpha_labels_with_status(
        base_panel,
        calendar,
        cost_schedule,
        liquidity_model,
        horizon_sessions=horizon_sessions,
        reference_notional=reference_notional,
        policy=policy,
        bar_evidence=bar_evidence,
    )
    return labels


def build_partitioned_net_alpha_labels_with_status(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    cost_schedule: CostSchedule,
    liquidity_model: LiquiditySlippageModel,
    horizon_sessions: tuple[int, ...],
    reference_notional: float,
    policy: ExecutionOutcomePolicy | None = None,
    bar_evidence: pl.DataFrame | None = None,
    tradability_events: pl.DataFrame | None = None,
    corporate_actions: CorporateActionSnapshot | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the long net-alpha label frame and its hash-bound status sidecar.

    Every candidate horizon is built independently. The status sidecar emits
    exactly one row per ``(instrument_id, decision_session, horizon_sessions)``
    in the decision universe with a typed outcome state, so the trainer never
    has to infer why a key lacks a realised label. ``policy`` is the immutable
    execution policy shared with the replay and backtester. When
    ``corporate_actions`` is supplied, paths crossing an action or uncovered
    interval emit ``UNSUPPORTED_CORPORATE_ACTION`` and never a realised label;
    labels are simple decimal rates (``exit_open / entry_open - 1``).
    """
    if not horizon_sessions:
        raise ValueError("horizon_sessions must be non-empty")
    if tuple(horizon_sessions) != tuple(sorted(set(horizon_sessions))):
        raise ValueError("horizon_sessions must be strictly ascending and unique")
    if any(h < 1 for h in horizon_sessions):
        raise ValueError("horizon_sessions must be positive sessions")
    if reference_notional <= 0:
        raise ValueError("reference_notional must be positive")

    frames: list[pl.DataFrame] = []
    status_frames: list[pl.DataFrame] = []
    for horizon in horizon_sessions:
        suffix = _horizon_suffix(horizon)
        wide, status = build_net_alpha_label_dataset_with_status(
            base_panel,
            calendar,
            cost_schedule,
            liquidity_model,
            horizon_sessions=horizon,
            reference_notional=reference_notional,
            policy=policy,
            bar_evidence=bar_evidence,
            tradability_events=tradability_events,
            corporate_actions=corporate_actions,
        )
        status_frames.append(
            status.select(
                pl.col(ID_COLUMN),
                pl.col(SESSION_COLUMN),
                pl.lit(horizon, dtype=pl.Int64).alias(HORIZON_COLUMN),
                pl.col(OUTCOME_STATUS_COLUMN),
            )
        )
        if wide.is_empty():
            continue
        long = wide.select(
            pl.col(ID_COLUMN),
            pl.col("session").alias(SESSION_COLUMN),
            pl.lit(horizon, dtype=pl.Int64).alias(HORIZON_COLUMN),
            pl.col(f"{NET_ALPHA_TARGET_PREFIX}{suffix}{NET_ALPHA_CONTINUOUS_SUFFIX}")
            .alias(TARGET_COLUMN),
            pl.col(f"label_available_time_{suffix}").alias(AVAILABLE_COLUMN),
            pl.col(f"{COST_AWARE_GROSS_PREFIX}{suffix}").alias(GROSS_COLUMN),
            pl.col(f"{COST_AWARE_RISK_RESIDUAL_PREFIX}{suffix}").alias(
                RISK_RESIDUAL_COLUMN
            ),
            pl.col(f"{COST_AWARE_REFERENCE_COST_PREFIX}{suffix}").alias(
                REFERENCE_COST_COLUMN
            ),
        )
        frames.append(long)

    if not frames:
        labels = pl.DataFrame(
            schema={
                ID_COLUMN: pl.Utf8,
                SESSION_COLUMN: pl.Datetime("us", "UTC"),
                HORIZON_COLUMN: pl.Int64,
                TARGET_COLUMN: pl.Float64,
                AVAILABLE_COLUMN: pl.Datetime("us", "UTC"),
                GROSS_COLUMN: pl.Float64,
                RISK_RESIDUAL_COLUMN: pl.Float64,
                REFERENCE_COST_COLUMN: pl.Float64,
            }
        )
    else:
        labels = pl.concat(frames).sort([ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN])
        present = sorted(labels[HORIZON_COLUMN].unique().to_list())
        if set(present) != set(horizon_sessions):
            missing = sorted(set(horizon_sessions) - set(present))
            raise ValueError(
                f"partitioned net-alpha labels missing horizons {missing}; "
                "no session cleared the risk/cost/MAD gates"
            )
    status_out = pl.concat(status_frames).sort(
        [ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN]
    )
    unknown = status_out.filter(
        ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(OUTCOME_STATUS_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("partitioned net-alpha status sidecar contains unknown states")
    return labels, status_out


OUTCOME_STATUS_DATASET_SUFFIX = "_outcome_status"


def publish_outcome_status_sidecar(
    status_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: tuple[int, ...],
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
    policy: ExecutionOutcomePolicy | None = None,
    corporate_actions_hash: str | None = None,
) -> NetAlphaLabelDatasetResult:
    """Publish the hash-bound per-key outcome-status sidecar dataset.

    The sidecar carries exactly one typed ``outcome_status`` row per
    ``(instrument_id, decision_session, horizon_sessions)`` in the decision
    universe, partitioned like the label dataset. Its manifest binds the exact
    column order and values through the schema/content hashes and, when
    ``policy`` is supplied, pins the immutable execution policy id and hash so
    no consumer can silently classify outcomes under a different policy. When
    ``corporate_actions_hash`` is supplied it binds the exact action snapshot
    that gated every realised label.
    """
    if status_frame.is_empty():
        raise ValueError("cannot publish an empty outcome-status sidecar")
    expected_columns = [
        ID_COLUMN,
        SESSION_COLUMN,
        HORIZON_COLUMN,
        OUTCOME_STATUS_COLUMN,
    ]
    if status_frame.columns != expected_columns:
        raise ValueError(
            "outcome-status sidecar must carry exactly "
            f"{expected_columns}, got {status_frame.columns}"
        )
    unknown = status_frame.filter(
        ~pl.col(OUTCOME_STATUS_COLUMN).is_in(list(OUTCOME_STATUS_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("outcome-status sidecar contains unknown states")
    duplicates = status_frame.group_by(
        [ID_COLUMN, SESSION_COLUMN, HORIZON_COLUMN]
    ).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError(
            "outcome-status sidecar must emit exactly one row per "
            "(instrument_id, session, horizon_sessions); "
            f"{duplicates.height} keys are duplicated"
        )
    present = sorted(status_frame[HORIZON_COLUMN].unique().to_list())
    if set(present) != set(horizon_sessions):
        raise ValueError(
            f"outcome-status sidecar horizon partitions {present}, "
            f"expected {sorted(horizon_sessions)}"
        )

    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(status_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=NET_ALPHA_DEFINITION,
        label_horizon_sessions=horizon_sessions[0],
        time_start=_as_utc_datetime(status_frame[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(status_frame[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=status_frame.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(status_frame, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": NET_ALPHA_DEFINITION,
        "horizon_sessions": list(horizon_sessions),
        "outcome_status_vocabulary": list(OUTCOME_STATUS_VOCABULARY),
        "generated_time": generated_time.isoformat(),
    }
    if policy is not None:
        content_manifest["policy_id"] = policy.policy_id
        content_manifest["policy_hash"] = policy.canonical_hash
    if corporate_actions_hash is not None:
        content_manifest["corporate_actions_hash"] = corporate_actions_hash
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        status_frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=LABEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    return NetAlphaLabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=status_frame.height,
        base_panel_hash=base_panel_hash,
    )


def partition_labels_by_horizon(
    labels: pl.DataFrame,
    horizon_sessions: tuple[int, ...],
) -> dict[int, pl.DataFrame]:
    """Split a long partitioned label frame into per-horizon independent frames."""
    if labels.is_empty():
        raise ValueError("cannot partition an empty label frame")
    if HORIZON_COLUMN not in labels.columns:
        raise ValueError(f"label frame missing {HORIZON_COLUMN!r} partition column")
    missing = [h for h in horizon_sessions if h not in labels[HORIZON_COLUMN]]
    if missing:
        raise ValueError(f"label frame has no rows for horizons {missing}")
    return {
        horizon: labels.filter(pl.col(HORIZON_COLUMN) == horizon).drop(HORIZON_COLUMN)
        for horizon in horizon_sessions
    }


@dataclass(frozen=True, slots=True)
class NetAlphaLabelDatasetResult:
    """Immutable outcome of one partitioned net-alpha label publication."""

    dataset_id: str
    manifest: DatasetManifest
    partition_paths: tuple[Path, ...]
    row_count: int
    base_panel_hash: str


def publish_partitioned_net_alpha_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: tuple[int, ...],
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
    corporate_actions_hash: str | None = None,
) -> NetAlphaLabelDatasetResult:
    """Publish the long, ``horizon_sessions``-partitioned net-alpha label dataset.

    The manifest declares ``label_definition="net_alpha_o2o"`` with the control
    horizon (first candidate) and records the horizon set and net-alpha
    algorithm version in the content manifest. The schema/content hashes bind
    the exact column order and values, and ``corporate_actions_hash`` binds the
    action snapshot that gated every realised row.
    """
    if not horizon_sessions:
        raise ValueError("horizon_sessions must be non-empty")
    if tuple(horizon_sessions) != tuple(sorted(set(horizon_sessions))):
        raise ValueError("horizon_sessions must be strictly ascending and unique")
    expected_columns = [
        ID_COLUMN,
        SESSION_COLUMN,
        HORIZON_COLUMN,
        TARGET_COLUMN,
        AVAILABLE_COLUMN,
        GROSS_COLUMN,
        RISK_RESIDUAL_COLUMN,
        REFERENCE_COST_COLUMN,
    ]
    if labels_frame.columns != expected_columns:
        raise ValueError(
            "partitioned net-alpha label dataset must carry exactly "
            f"{expected_columns}, got {labels_frame.columns}"
        )
    if labels_frame.is_empty():
        raise ValueError("cannot publish an empty partitioned net-alpha label dataset")
    if labels_frame.filter(pl.col(AVAILABLE_COLUMN).is_null()).height:
        raise ValueError(
            "partitioned net-alpha label dataset contains rows without label_available_time"
        )
    present = sorted(labels_frame[HORIZON_COLUMN].unique().to_list())
    if set(present) != set(horizon_sessions):
        raise ValueError(
            f"partitioned net-alpha label horizon partitions {present}, "
            f"expected {sorted(horizon_sessions)}"
        )

    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(labels_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=NET_ALPHA_DEFINITION,
        label_horizon_sessions=horizon_sessions[0],
        time_start=_as_utc_datetime(labels_frame[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(labels_frame[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=labels_frame.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(labels_frame, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": NET_ALPHA_DEFINITION,
        "label_horizon_sessions": list(horizon_sessions),
        "horizon_sessions": list(horizon_sessions),
        "entry_field": "open",
        "exit_field": "open",
        "generated_time": generated_time.isoformat(),
        "label_algorithm_version": NET_ALPHA_ALGORITHM_VERSION,
    }
    if corporate_actions_hash is not None:
        content_manifest["corporate_actions_hash"] = corporate_actions_hash
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        labels_frame,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set=LABEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    return NetAlphaLabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=labels_frame.height,
        base_panel_hash=base_panel_hash,
    )


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    from datetime import date as _date

    if isinstance(value, _date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    raise ValueError(f"expected a date or datetime timestamp, got {value!r}")
