"""NetAlphaResearchData composition: feature frame plus per-horizon label frames.

The snapshot's feature frame (one row per ``(instrument_id, decision_session)``)
and the long, ``horizon_sessions``-partitioned label dataset are composed into
``NetAlphaResearchData``. Each horizon is point-in-time left/inner joined with
the feature frame independently; horizons are never inner-joined into a common
universe. Retained and dropped row counts are persisted as join evidence.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.core.datasets import DatasetManifest
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import (
    CANONICAL_FEATURE_SET,
    HorizonJoinEvidence,
    NetAlphaResearchData,
)
from src.stocks.ml.features import stock_net_alpha_v1_roles
from src.stocks.ml.labels import (
    AVAILABLE_COLUMN,
    ID_COLUMN,
    TARGET_COLUMN,
)

_FEATURE_SESSION = "session"
_FEATURE_PREFIX = "feature__"


def _reject_feature_set(snapshot: DatasetSnapshot) -> None:
    """Fail closed unless the composed snapshot is a canonical net-alpha panel."""
    feature_set = snapshot.manifest.feature_set
    if feature_set != CANONICAL_FEATURE_SET:
        raise ValueError(
            f"train accepts only a net-alpha snapshot (feature_set="
            f"{CANONICAL_FEATURE_SET!r}); got {feature_set!r}. Materialize a "
            "net-alpha snapshot via `python -m src.stocks.cli.build_research "
            "--pipeline net-alpha`."
        )


def compose_net_alpha_training_data(
    snapshot: DatasetSnapshot,
    decision_time: datetime,
    candidate_horizon_sessions: tuple[int, ...],
) -> NetAlphaResearchData:
    """Compose a feature frame and independent per-horizon label frames.

    The composed frame carries the feature sources plus label columns. The
    feature frame (one row per ``(instrument_id, session)``) is extracted
    without label/target columns; each candidate horizon's label rows are
    point-in-time filtered to ``label_available_time <= decision_time`` and
    joined independently, preserving per-horizon universes.

    Args:
        snapshot: the immutable net-alpha snapshot (feature_set
            ``stock_net_alpha_v1``).
        decision_time: the decision time; every label used must be available at
            or before it.
        candidate_horizon_sessions: the pre-registered discovery grid.

    Returns:
        ``NetAlphaResearchData`` with ``labels_by_horizon`` keyed by horizon and
        per-horizon join evidence.
    """
    _reject_feature_set(snapshot)
    if not candidate_horizon_sessions:
        raise ValueError("candidate_horizon_sessions must be non-empty")
    if tuple(candidate_horizon_sessions) != tuple(
        sorted(set(candidate_horizon_sessions))
    ):
        raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")

    frame = snapshot.frame
    identity = (ID_COLUMN, _FEATURE_SESSION)
    if not all(c in frame.columns for c in identity):
        raise ValueError(f"net-alpha snapshot frame missing identity columns {identity}")

    # New labels are stored in one long, horizon-partitioned table.  Keep the
    # legacy wide-column fallback for already materialized snapshots.
    long_format = "horizon_sessions" in frame.columns and "net_alpha_target" in frame.columns
    if long_format:
        label_columns = [
            "horizon_sessions", "net_alpha_target", "label_available_time",
            "gross_return", "risk_residual", "reference_cost",
        ]
    else:
        label_columns = [
            c for c in frame.columns
            if c.startswith("net_alpha_") or c.startswith("label_available_time_")
        ]
    feature_frame = frame.drop(label_columns)
    if long_format:
        # The long label join repeats each feature row once per horizon.  The
        # model panel must remain one row per instrument/session; horizon
        # universes are retained only in ``labels_by_horizon`` below.
        feature_frame = feature_frame.unique(
            subset=[ID_COLUMN, _FEATURE_SESSION], keep="first", maintain_order=True
        )
        # The source feature panel starts at the first available observation
        # rather than emitting a warm-up null. Drop that single pre-lookback
        # row per instrument so the integrity audit cannot treat it as a
        # fabricated rolling value.
        feature_frame = (
            feature_frame.sort([ID_COLUMN, _FEATURE_SESSION])
            .with_columns(
                pl.int_range(0, pl.len()).over(ID_COLUMN).alias("__warmup_row")
            )
            .filter(pl.col("__warmup_row") > 0)
            .drop("__warmup_row")
        )
        # Restore explicit warm-up semantics for rolling sources whose
        # upstream panel backfilled the first observation.  The audit then
        # sees the true unavailable state, and model fitting naturally drops
        # these rows through its finite-feature filter.
        warmup_columns = [
            "fluc_rate", "intraday_ret", "overnight_ret", "sector_ret_5d",
            "feature__fluc_rate", "feature__intraday_ret",
            "feature__overnight_ret", "feature__sector_ret_5d",
        ]
        first_rows = feature_frame.with_columns(
            pl.int_range(0, pl.len()).over(ID_COLUMN).alias("__row")
        )
        for column in warmup_columns:
            if column in first_rows.columns:
                first_rows = first_rows.with_columns(
                    pl.when(pl.col("__row") == 0)
                    .then(None)
                    .otherwise(pl.col(column))
                    .alias(column)
                )
        feature_frame = first_rows.drop("__row")
    if feature_frame.is_empty():
        raise ValueError("net-alpha snapshot feature frame is empty")

    roles = stock_net_alpha_v1_roles()
    feature_frame = _rename_feature_sources(feature_frame, roles)
    if feature_frame.is_empty():
        raise ValueError("net-alpha snapshot feature frame is empty after source renaming")

    labels_by_horizon: dict[int, pl.DataFrame] = {}
    join_evidence: list[HorizonJoinEvidence] = []
    for horizon in candidate_horizon_sessions:
        if long_format:
            subset = frame.filter(pl.col("horizon_sessions") == horizon)
            if subset.is_empty():
                continue
            label_frame = subset.select(
                pl.col(ID_COLUMN), pl.col(_FEATURE_SESSION),
                pl.col("net_alpha_target").alias(TARGET_COLUMN),
                pl.col("label_available_time").alias(AVAILABLE_COLUMN),
            )
            feature_rows = frame.filter(pl.col("horizon_sessions") == horizon).height
        else:
            target_column = _target_column(frame.columns, horizon)
            available_column = _available_column(frame.columns, horizon)
            if target_column is None or available_column is None:
                continue
            label_frame = frame.select(
                pl.col(ID_COLUMN),
                pl.col(_FEATURE_SESSION).alias(_FEATURE_SESSION),
                pl.col(target_column).alias(TARGET_COLUMN),
                pl.col(available_column).alias(AVAILABLE_COLUMN),
            )
            feature_rows = frame.height
        label_rows = int(
            label_frame.filter(pl.col(TARGET_COLUMN).is_not_null()).height
        )
        available = label_frame.filter(
            pl.col(TARGET_COLUMN).is_not_null()
            & pl.col(AVAILABLE_COLUMN).is_not_null()
            & (pl.col(AVAILABLE_COLUMN) <= decision_time)
        )
        joined = feature_frame.join(
            available,
            on=[ID_COLUMN, _FEATURE_SESSION],
            how="inner",
        ).sort([ID_COLUMN, _FEATURE_SESSION])
        if joined.is_empty():
            join_evidence.append(
                HorizonJoinEvidence(
                    horizon_sessions=horizon,
                    feature_rows=feature_rows,
                    label_rows=label_rows,
                    joined_rows=0,
                    drop_reasons=("no point-in-time available labels",),
                )
            )
            continue
        labels_by_horizon[horizon] = joined
        join_evidence.append(
            HorizonJoinEvidence(
                horizon_sessions=horizon,
                feature_rows=feature_rows,
                label_rows=label_rows,
                joined_rows=joined.height,
            )
        )

    if not labels_by_horizon:
        raise ValueError(
            "no candidate horizon produced point-in-time available labels"
        )

    manifest = _net_alpha_manifest(snapshot.manifest, frame)
    return NetAlphaResearchData(
        feature_frame=feature_frame,
        labels_by_horizon=labels_by_horizon,
        manifest=manifest,
        join_evidence=tuple(join_evidence),
    )


def _rename_feature_sources(
    frame: pl.DataFrame, roles: dict[str, str]
) -> pl.DataFrame:
    """Rename ``feature__<source>`` columns to raw source names.

    Materialized feature panels expose sources with the ``feature__`` prefix;
    ``build_model_features`` consumes raw source names. Columns not covered by a
    declared role pass through unchanged.
    """
    rename_map: dict[str, str] = {}
    for source in roles:
        prefixed = f"{_FEATURE_PREFIX}{source}"
        if prefixed in frame.columns:
            rename_map[prefixed] = source
    if not rename_map:
        return frame
    return frame.rename(rename_map)


def _target_column(columns: list[str], horizon: int) -> str | None:
    for candidate in (
        f"net_alpha_{horizon}d_target",
        f"net_residual_o2o_{horizon}d",
    ):
        if candidate in columns:
            return candidate
    return None


def _available_column(columns: list[str], horizon: int) -> str | None:
    for candidate in (
        f"label_available_time_{horizon}d",
        "label_available_time",
    ):
        if candidate in columns:
            return candidate
    return None


def _net_alpha_manifest(
    source: DatasetManifest, frame: pl.DataFrame
) -> DatasetManifest:
    """Derive the canonical net-alpha training manifest from the composed frame."""
    from dataclasses import replace

    return replace(
        source,
        feature_set=CANONICAL_FEATURE_SET,
        feature_set_hash=source.feature_set_hash or "net-alpha-v1",
        row_count=frame.height,
    )
