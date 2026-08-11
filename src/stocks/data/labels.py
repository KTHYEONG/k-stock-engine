"""Calendar-aware forward-label builder and immutable label publication.

Labels are never predictors and never reside in a feature panel. A label
dataset is built per ``(base_panel_hash, calendar_hash, label_definition,
horizon, price convention)`` and keyed by ``(instrument_id, session)``.

Timing is strictly KRX-session based: the decision session's entry is the next
KRX session open and its exit is the close of ``horizon`` sessions later. Rows
whose future horizon is incomplete (missing calendar session or missing price)
are absent, never guessed, and every emitted label carries a terminal
``label_available_time`` at or after the terminal horizon session (spec
acceptance criterion 3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    DatasetManifest,
    make_manifest,
)
from src.core.instruments import AssetKind
from src.stocks.data.quality import KRXSessionCalendar
from src.stocks.research.labels import RELEVANCE_COLUMN, LabelDefinition
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

logger = logging.getLogger("stocks.data.labels")

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
LABEL_AVAILABLE_COLUMN = "label_available_time"
LABEL_FEATURE_SET = "labels"
_KRX_AVAILABLE_TIME = time(15, 31)
_KRX_TZ = ZoneInfo("Asia/Seoul")

RESIDUAL_O2O_PREFIX = "residual_o2o_"
MIN_RESIDUAL_GROUP = 20
RESIDUAL_ALGORITHM_VERSION = "calendar-residual-o2o-v1"


def build_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    definition: LabelDefinition,
) -> pl.DataFrame:
    """Build calendar-aware forward labels keyed by ``(instrument_id, session)``.

    Only rows with a complete ``horizon``-session future horizon are emitted;
    each such row has a terminal ``label_available_time``. The base panel must
    carry ``instrument_id``, ``session``, and the definition's entry/exit
    fields (``open`` / ``close`` by convention).

    Raises:
        ValueError: if the base panel lacks identity or price columns, or if a
            base-panel session is not a KRX calendar session.
    """
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN) if c not in base_panel.columns]
    if missing:
        raise ValueError(f"base panel missing {', '.join(missing)}")
    price_missing = [c for c in (definition.entry_field, definition.exit_field) if c not in base_panel.columns]
    if price_missing:
        raise ValueError(f"label {definition.name} requires price columns {price_missing}")

    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
            "_entry_date": [sessions[p + 1] if p + 1 < len(sessions) else None for p in range(len(sessions))],
            "_exit_date": [
                sessions[p + definition.horizon_sessions]
                if p + definition.horizon_sessions < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(
        pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null()
    )
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    prices = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_price_date"),
        pl.col(definition.entry_field),
        pl.col(definition.exit_field),
    )
    entries = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_entry_date"),
        pl.col(definition.entry_field).alias("_entry_price"),
    )
    exits = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_exit_date"),
        pl.col(definition.exit_field).alias("_exit_price"),
    )
    panel = (
        panel.join(entries, on=[ID_COLUMN, "_entry_date"], how="left")
        .join(exits, on=[ID_COLUMN, "_exit_date"], how="left")
        .filter(pl.col("_entry_price").is_not_null() & pl.col("_exit_price").is_not_null())
    )

    label = pl.col("_exit_price").log() - pl.col("_entry_price").log()
    label_available = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )
    out = panel.with_columns(
        pl.col("_session_date").alias(SESSION_COLUMN),
        label.alias(definition.name),
        label_available.alias(LABEL_AVAILABLE_COLUMN),
    ).select(
        ID_COLUMN,
        SESSION_COLUMN,
        definition.name,
        LABEL_AVAILABLE_COLUMN,
    ).sort([ID_COLUMN, SESSION_COLUMN])

    incomplete = out.filter(pl.col(LABEL_AVAILABLE_COLUMN).is_null() | pl.col(definition.name).is_null())
    if not incomplete.is_empty():
        raise ValueError("label builder emitted an incomplete horizon row")
    logger.info(
        "built label %s: %s rows over %s sessions",
        definition.name,
        out.height,
        len(sessions),
    )
    return out


def build_residual_o2o_label_dataset(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    *,
    horizon_sessions: int = 5,
) -> pl.DataFrame:
    """Build calendar-correct residual open-to-open labels with LambdaRank relevance.

    For a decision session at calendar position ``p``, a label is computed only
    when the same instrument has a valid open at both exact calendar sessions:

    ``entry = calendar[p + 1]`` and ``exit = calendar[p + 1 + horizon]``.
    The gross return is ``log(open(exit) / open(entry))`` and the residual is
    the gross minus the equal-weight mean gross within the decision session.
    The per-session 1st/99th percentile-clipped residual rank maps to
    ``relevance = floor(percentile * 5)`` clipped to integer ``0..4``.

    Sessions with fewer than ``MIN_RESIDUAL_GROUP`` finite residuals are dropped
    completely and no fabricated label is ever retained. The result schema is:

    ``instrument_id, session, residual_o2o_<horizon>d, relevance, label_available_time``

    ``label_available_time`` is set to the exact exit-session open timestamp in
    UTC (never the decision session), so a decision row is only usable after the
    terminal horizon open is observable.
    """
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN, "open") if c not in base_panel.columns]
    if missing:
        raise ValueError(f"residual label requires base panel columns {missing}")

    label_column = f"{RESIDUAL_O2O_PREFIX}{horizon_sessions}d"
    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
            "_entry_date": [
                sessions[p + 1] if p + 1 < len(sessions) else None for p in range(len(sessions))
            ],
            "_exit_date": [
                sessions[p + 1 + horizon_sessions]
                if p + 1 + horizon_sessions < len(sessions)
                else None
                for p in range(len(sessions))
            ],
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null())
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    prices = panel.select(
        ID_COLUMN,
        pl.col("_session_date").alias("_price_date"),
        pl.col("open"),
    )
    entries = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_entry_date"),
        pl.col("open").alias("_entry_open"),
    )
    exits = prices.select(
        ID_COLUMN,
        pl.col("_price_date").alias("_exit_date"),
        pl.col("open").alias("_exit_open"),
    )
    panel = (
        panel.join(entries, on=[ID_COLUMN, "_entry_date"], how="left")
        .join(exits, on=[ID_COLUMN, "_exit_date"], how="left")
        .filter(pl.col("_entry_open").is_not_null() & pl.col("_exit_open").is_not_null())
    )

    gross = pl.col("_exit_open").log() - pl.col("_entry_open").log()
    residual = gross - gross.mean().over(SESSION_COLUMN)
    label_available = (
        pl.col("_exit_date")
        .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
        .dt.replace_time_zone("Asia/Seoul")
        .dt.convert_time_zone("UTC")
    )
    session_utc = (
        pl.col("_session_date")
        .cast(pl.Datetime)
        .dt.replace_time_zone("UTC")
    )

    winsor = (
        panel.with_columns(residual.alias("_residual"))
        .group_by(SESSION_COLUMN)
        .agg(
            pl.col("_residual").quantile(0.01).alias("__lo"),
            pl.col("_residual").quantile(0.99).alias("__hi"),
        )
    )
    clipped = (
        panel.with_columns(residual.alias("_residual"))
        .join(winsor, on=SESSION_COLUMN, how="left")
        .with_columns(
            pl.col("_residual").clip(pl.col("__lo"), pl.col("__hi")).alias("__clipped")
        )
    )
    count = pl.col("_residual").count().over(SESSION_COLUMN)
    pct_rank = (
        (pl.col("__clipped").rank("average").over(SESSION_COLUMN) - 1.0) / (count - 1.0)
    ).fill_null(0.5)
    relevance = (
        pl.when(pct_rank.is_not_null())
        .then((pct_rank * 5.0).floor().cast(pl.Int8).clip(0, 4))
        .otherwise(None)
    )

    out = clipped.with_columns(
        session_utc.alias(SESSION_COLUMN),
        pl.col("_residual").alias(label_column),
        relevance.alias(RELEVANCE_COLUMN),
        label_available.alias(LABEL_AVAILABLE_COLUMN),
    ).select(
        ID_COLUMN,
        SESSION_COLUMN,
        label_column,
        RELEVANCE_COLUMN,
        LABEL_AVAILABLE_COLUMN,
    )

    eligible_sessions = (
        out.filter(pl.col(label_column).is_not_null())
        .group_by(SESSION_COLUMN)
        .len()
        .filter(pl.col("len") >= MIN_RESIDUAL_GROUP)[SESSION_COLUMN]
    )
    if eligible_sessions.is_empty():
        logger.info(
            "residual label %s: no session has at least %s finite residuals",
            label_column,
            MIN_RESIDUAL_GROUP,
        )
        return out.head(0)
    out = out.filter(
        pl.col(SESSION_COLUMN).is_in(eligible_sessions.to_list())
    ).sort([ID_COLUMN, SESSION_COLUMN])
    incomplete = out.filter(
        pl.col(LABEL_AVAILABLE_COLUMN).is_null() | pl.col(label_column).is_null()
    )
    if not incomplete.is_empty():
        raise ValueError("residual label builder emitted an incomplete row")
    logger.info(
        "built residual label %s: %s rows over %s eligible sessions",
        label_column,
        out.height,
        len(eligible_sessions),
    )
    return out


def label_available_time(exit_session: date) -> datetime:
    """Availability timestamp for a terminal horizon session (after close)."""
    return datetime.combine(exit_session, _KRX_AVAILABLE_TIME, tzinfo=_KRX_TZ).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class LabelDatasetResult:
    """Immutable outcome of a label-dataset publication."""

    dataset_id: str
    manifest: DatasetManifest
    partition_paths: tuple[Path, ...]
    row_count: int
    base_panel_hash: str


def publish_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    definition: LabelDefinition,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
    algorithm_version: str = "",
) -> LabelDatasetResult:
    """Publish an immutable, session-keyed label dataset.

    The label dataset never lives in a feature panel; it is a separate
    ``canonical/stocks/labels/<label-id>`` version keyed by
    ``(instrument_id, session)`` and bound to the base-panel and calendar
    hashes it was built from.
    """
    required = (ID_COLUMN, SESSION_COLUMN, definition.name, LABEL_AVAILABLE_COLUMN)
    missing = [c for c in required if c not in labels_frame.columns]
    if missing:
        raise ValueError(f"label dataset missing columns {missing}")
    if labels_frame.is_empty():
        raise ValueError("cannot publish an empty label dataset")
    if labels_frame.filter(pl.col(LABEL_AVAILABLE_COLUMN).is_null()).height:
        raise ValueError("label dataset contains rows without label_available_time")

    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(labels_frame.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=LABEL_FEATURE_SET,
        label_definition=definition.name,
        label_horizon_sessions=definition.horizon_sessions,
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
        "label_definition": definition.name,
        "label_horizon_sessions": definition.horizon_sessions,
        "entry_field": definition.entry_field,
        "exit_field": definition.exit_field,
        "generated_time": generated_time.isoformat(),
    }
    if algorithm_version:
        content_manifest["label_algorithm_version"] = algorithm_version
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
    logger.info(
        "published label dataset %s: %s rows, horizon %s",
        dataset_id,
        labels_frame.height,
        definition.horizon_sessions,
    )
    return LabelDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=labels_frame.height,
        base_panel_hash=base_panel_hash,
    )


def publish_residual_o2o_label_dataset(
    labels_frame: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: int = 5,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
) -> LabelDatasetResult:
    """Publish a calendar-correct residual open-to-open label dataset.

    A dedicated publisher so the generic ``fwd_ret_*`` path stays unchanged for
    v1 consumers. The label column is ``residual_o2o_<horizon>d`` and the
    content manifest records the residual-label algorithm version.
    """
    label_column = f"{RESIDUAL_O2O_PREFIX}{horizon_sessions}d"
    definition = LabelDefinition(
        name=label_column,
        entry_field="open",
        exit_field="open",
        horizon_sessions=horizon_sessions,
    )
    return publish_label_dataset(
        labels_frame,
        destination_root=destination_root,
        dataset_id=dataset_id,
        base_panel_hash=base_panel_hash,
        calendar_hash=calendar_hash,
        definition=definition,
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        certification=certification,
        generated_time=generated_time,
        algorithm_version=RESIDUAL_ALGORITHM_VERSION,
    )


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    raise ValueError(f"expected a date or datetime timestamp, got {value!r}")
