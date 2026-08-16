"""Outcome-evidence materialization under a declared execution policy.

The outcome-evidence artifact is a derived, non-feature record: one row per
``(instrument_id, decision_session, horizon_sessions, policy_id)`` carrying the
scheduled and actual entry/exit sessions, delay counts, dispositions, typed
resolution kind, the pinned policy hash, raw-bar partition provenance, and the
actual label availability. It never enters model features or scores.

Resolution is strictly bar-evidence based and fail-closed: entry and exit are
resolved only from immutable, source-date daily bars keyed by
``(instrument_id, price_date)`` with a finite, strictly positive ``open``. A
missing open is never forward-filled, zero-filled, interpolated, or replaced by
close/high/low. ``scheduled_open_v1`` requires both scheduled opens;
``first_tradable_open_v1`` may use the first valid open no later than its
explicit non-negative delay bounds, never a backwards or unverified bar.
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
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

logger = logging.getLogger("stocks.data.outcome_evidence")

ID_COLUMN = "instrument_id"
SESSION_COLUMN = "session"
PRICE_DATE_COLUMN = "price_date"
OPEN_COLUMN = "open"

RESOLUTION_SCHEDULED_OPEN = "SCHEDULED_OPEN"
RESOLUTION_DEFERRED_OPEN = "DEFERRED_OPEN"
RESOLUTION_CONFIRMED_NO_BAR = "CONFIRMED_NO_BAR"
RESOLUTION_SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
RESOLUTION_VERIFIED_CORPORATE_SETTLEMENT = "VERIFIED_CORPORATE_SETTLEMENT"
RESOLUTION_UNSUPPORTED_CORPORATE_ACTION = "UNSUPPORTED_CORPORATE_ACTION"
RESOLUTION_PARTIAL_TAIL = "PARTIAL_TAIL"

RESOLUTION_KIND_VOCABULARY = (
    RESOLUTION_SCHEDULED_OPEN,
    RESOLUTION_DEFERRED_OPEN,
    RESOLUTION_CONFIRMED_NO_BAR,
    RESOLUTION_SOURCE_UNAVAILABLE,
    RESOLUTION_VERIFIED_CORPORATE_SETTLEMENT,
    RESOLUTION_UNSUPPORTED_CORPORATE_ACTION,
    RESOLUTION_PARTIAL_TAIL,
)

DISPOSITION_FILLED = "FILLED"
DISPOSITION_NO_BAR = "NO_BAR"
DISPOSITION_SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
DISPOSITION_TAIL = "TAIL"
DISPOSITION_UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"

_KRX_AVAILABLE_TIME = time(15, 31)
_KRX_TZ = ZoneInfo("Asia/Seoul")

OUTCOME_EVIDENCE_COLUMNS = (
    ID_COLUMN,
    SESSION_COLUMN,
    "horizon_sessions",
    "policy_id",
    "policy_hash",
    "scheduled_entry_session",
    "scheduled_exit_session",
    "actual_entry_session",
    "actual_exit_session",
    "entry_delay_sessions",
    "exit_delay_sessions",
    "entry_disposition",
    "exit_disposition",
    "resolution_kind",
    "outcome_status",
    "entry_partition_hash",
    "exit_partition_hash",
    "corporate_action_event_id",
    "label_available_time",
)

OUTCOME_EVIDENCE_DATASET_SUFFIX = "_outcome_evidence"


def merge_open_bar_evidence(
    base_panel: pl.DataFrame,
    bar_evidence: pl.DataFrame | None,
) -> pl.DataFrame:
    """Overlay verified raw opens on matching base-panel keys.

    Raw evidence can repair or correct only an identical
    ``(instrument_id, price_date)`` key.  The base panel remains the source for
    every key not present in the immutable backfill artifact, so a partial
    backfill cannot erase unrelated historical opens.
    """
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN, OPEN_COLUMN) if c not in base_panel.columns]
    if missing:
        raise ValueError(f"base panel missing {', '.join(missing)}")
    base = base_panel.select(
        pl.col(ID_COLUMN),
        pl.col(SESSION_COLUMN).cast(pl.Date).alias(PRICE_DATE_COLUMN),
        pl.col(OPEN_COLUMN),
        pl.lit(0, dtype=pl.Int8).alias("_source_priority"),
    )
    if bar_evidence is None:
        return base.drop("_source_priority")
    evidence_missing = [
        c for c in (ID_COLUMN, PRICE_DATE_COLUMN, OPEN_COLUMN) if c not in bar_evidence.columns
    ]
    if evidence_missing:
        raise ValueError(f"bar evidence missing {', '.join(evidence_missing)}")
    raw = bar_evidence.select(
        pl.col(ID_COLUMN),
        pl.col(PRICE_DATE_COLUMN).cast(pl.Date),
        pl.col(OPEN_COLUMN),
        pl.lit(1, dtype=pl.Int8).alias("_source_priority"),
    )
    invalid = raw.filter(
        pl.col(OPEN_COLUMN).is_null()
        | ~pl.col(OPEN_COLUMN).is_finite()
        | (pl.col(OPEN_COLUMN) <= 0)
        | pl.col(PRICE_DATE_COLUMN).is_null()
    )
    if not invalid.is_empty():
        raise ValueError("bar evidence contains non-positive or non-finite opens")
    duplicate = raw.group_by([ID_COLUMN, PRICE_DATE_COLUMN]).len().filter(pl.col("len") > 1)
    if not duplicate.is_empty():
        raise ValueError("bar evidence contains duplicate (instrument_id, price_date) keys")
    return (
        pl.concat([base, raw])
        .sort([ID_COLUMN, PRICE_DATE_COLUMN, "_source_priority"])
        .unique(subset=[ID_COLUMN, PRICE_DATE_COLUMN], keep="last", maintain_order=True)
        .drop("_source_priority")
    )


def build_partitioned_outcome_evidence(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    *,
    horizon_sessions: tuple[int, ...],
    policy: ExecutionOutcomePolicy,
    bar_evidence: pl.DataFrame | None = None,
    unavailable_sessions: set[date] | None = None,
    corporate_action_keys: set[tuple[str, date]] | None = None,
    corporate_action_event_id: str | None = None,
) -> pl.DataFrame:
    """Build one long, ``horizon_sessions``-partitioned outcome-evidence frame.

    Each horizon is resolved independently under the same policy; the result is
    a long frame keyed by ``(instrument_id, session, horizon_sessions)`` with
    the full ``OUTCOME_EVIDENCE_COLUMNS`` schema.
    """
    if not horizon_sessions:
        raise ValueError("horizon_sessions must be non-empty")
    if tuple(horizon_sessions) != tuple(sorted(set(horizon_sessions))):
        raise ValueError("horizon_sessions must be strictly ascending and unique")
    frames = [
        resolve_policy_outcome(
            base_panel,
            calendar,
            horizon_sessions=horizon,
            policy=policy,
            bar_evidence=bar_evidence,
            unavailable_sessions=unavailable_sessions,
            corporate_action_keys=corporate_action_keys,
            corporate_action_event_id=corporate_action_event_id,
        )
        for horizon in horizon_sessions
    ]
    return pl.concat(frames).sort([ID_COLUMN, SESSION_COLUMN, "horizon_sessions"])


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    raise ValueError(f"expected a date or datetime timestamp, got {value!r}")


def label_available_time(exit_session: date) -> datetime:
    """Availability timestamp for a terminal horizon session (after close)."""
    return datetime.combine(exit_session, _KRX_AVAILABLE_TIME, tzinfo=_KRX_TZ).astimezone(UTC)


def _action_key_expr(action_keys: set[tuple[str, date]]) -> pl.Expr:
    """Boolean expression marking decision keys with an unsupported action."""
    keys = sorted(
        (str(key[0]), key[1].isoformat()) for key in action_keys
    )
    if not keys:
        return pl.lit(False)
    return pl.concat_str(
        pl.col(ID_COLUMN).cast(pl.Utf8),
        pl.lit("|"),
        pl.col("_session_date").cast(pl.Utf8),
        separator="",
    ).is_in([f"{instrument}|{day}" for instrument, day in keys])


def resolve_policy_outcome(
    base_panel: pl.DataFrame,
    calendar: KRXSessionCalendar,
    *,
    horizon_sessions: int,
    policy: ExecutionOutcomePolicy,
    bar_evidence: pl.DataFrame | None = None,
    unavailable_sessions: set[date] | None = None,
    entry_partition_hash: str | None = None,
    exit_partition_hash: str | None = None,
    corporate_action_keys: set[tuple[str, date]] | None = None,
    corporate_action_event_id: str | None = None,
) -> pl.DataFrame:
    """Resolve one policy-aware outcome row per decision key.

    ``base_panel`` must carry ``instrument_id``, ``session`` (KRX calendar
    sessions), and ``open``; every decision session must be a calendar session.
    ``bar_evidence`` is the immutable verified daily-bar source keyed by
    ``(instrument_id, price_date)`` with a finite, strictly positive ``open``;
    when omitted the base panel's own ``open`` is used as the verified source.
    ``unavailable_sessions`` marks request dates whose source response was
    unavailable or invalid (``SOURCE_UNAVAILABLE``); a calendar session with no
    verified bar is ``CONFIRMED_NO_BAR``. ``corporate_action_keys`` marks keys
    whose exit can only be an unsupported corporate-action settlement.

    Returns the outcome-evidence frame with ``OUTCOME_EVIDENCE_COLUMNS``.
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    missing = [c for c in (ID_COLUMN, SESSION_COLUMN, OPEN_COLUMN) if c not in base_panel.columns]
    if missing:
        raise ValueError(f"base panel missing {', '.join(missing)}")
    sessions = list(calendar.sessions)
    by_date = {session: index for index, session in enumerate(sessions)}
    if len(by_date) != len(sessions):
        raise ValueError("calendar contains duplicate sessions")

    panel = base_panel.with_columns(pl.col(SESSION_COLUMN).cast(pl.Date).alias("_session_date"))
    calendar_frame = pl.DataFrame(
        {
            "_session_date": sessions,
            "_cal_pos": list(range(len(sessions))),
        }
    )
    panel = panel.join(calendar_frame, on="_session_date", how="left")
    unknown = panel.filter(pl.col("_cal_pos").is_null() | pl.col("_session_date").is_null())
    if not unknown.is_empty():
        raise ValueError("base panel contains non-calendar sessions")

    entry_offset = policy.entry_offset_sessions
    panel = panel.with_columns(
        (pl.col("_cal_pos") + entry_offset).alias("_scheduled_entry_pos"),
        (pl.col("_cal_pos") + entry_offset + horizon_sessions).alias("_scheduled_exit_pos"),
    )
    pos_lookup = pl.DataFrame(
        {
            "_pos": list(range(len(sessions))),
            "_pos_date": sessions,
        }
    )
    panel = (
        panel.join(pos_lookup, left_on="_scheduled_entry_pos", right_on="_pos", how="left")
        .rename({"_pos_date": "scheduled_entry_session"})
        .join(pos_lookup, left_on="_scheduled_exit_pos", right_on="_pos", how="left")
        .rename({"_pos_date": "scheduled_exit_session"})
    )

    source = merge_open_bar_evidence(base_panel, bar_evidence)

    panel = _resolve_open_leg(
        panel,
        source,
        calendar=calendar,
        scheduled_pos_col="_scheduled_entry_pos",
        max_delay=policy.max_entry_delay_sessions,
        prefix="entry",
    )
    panel = _resolve_open_leg(
        panel,
        source,
        calendar=calendar,
        scheduled_pos_col="_scheduled_exit_pos",
        max_delay=policy.max_exit_delay_sessions,
        prefix="exit",
    )

    tail = pl.col("scheduled_entry_session").is_null() | pl.col("scheduled_exit_session").is_null()
    unavailable_dates = sorted(unavailable_sessions or ())
    unavailable = (
        pl.col("scheduled_entry_session").is_in(unavailable_dates)
        | pl.col("scheduled_exit_session").is_in(unavailable_dates)
    )
    action_key = _action_key_expr(corporate_action_keys or set())
    entry_open = pl.col("_entry_open")
    exit_open = pl.col("_exit_open")

    resolution_kind = (
        pl.when(tail)
        .then(pl.lit(RESOLUTION_PARTIAL_TAIL))
        .when(action_key)
        .then(pl.lit(RESOLUTION_UNSUPPORTED_CORPORATE_ACTION))
        .when(entry_open.is_not_null() & exit_open.is_not_null())
        .then(
            pl.when(
                pl.col("entry_delay_sessions").fill_null(0).eq(0)
                & pl.col("exit_delay_sessions").fill_null(0).eq(0)
            )
            .then(pl.lit(RESOLUTION_SCHEDULED_OPEN))
            .otherwise(pl.lit(RESOLUTION_DEFERRED_OPEN))
        )
        .when(unavailable)
        .then(pl.lit(RESOLUTION_SOURCE_UNAVAILABLE))
        .otherwise(pl.lit(RESOLUTION_CONFIRMED_NO_BAR))
    )
    status = (
        pl.when(tail)
        .then(pl.lit("PARTIAL_TAIL"))
        .when(action_key)
        .then(pl.lit("UNSUPPORTED_CORPORATE_ACTION"))
        .when(entry_open.is_not_null() & exit_open.is_not_null())
        .then(pl.lit("REALIZED"))
        .when(entry_open.is_null())
        .then(pl.lit("MISSING_ENTRY_PRICE"))
        .otherwise(pl.lit("MISSING_EXIT_PRICE"))
    )
    entry_disposition = (
        pl.when(tail)
        .then(pl.lit(DISPOSITION_TAIL))
        .when(entry_open.is_not_null())
        .then(pl.lit(DISPOSITION_FILLED))
        .when(unavailable)
        .then(pl.lit(DISPOSITION_SOURCE_UNAVAILABLE))
        .otherwise(pl.lit(DISPOSITION_NO_BAR))
    )
    exit_disposition = (
        pl.when(tail)
        .then(pl.lit(DISPOSITION_TAIL))
        .when(exit_open.is_not_null())
        .then(pl.lit(DISPOSITION_FILLED))
        .when(action_key)
        .then(pl.lit(DISPOSITION_UNSUPPORTED_ACTION))
        .when(unavailable)
        .then(pl.lit(DISPOSITION_SOURCE_UNAVAILABLE))
        .otherwise(pl.lit(DISPOSITION_NO_BAR))
    )

    actual_entry = pl.when(tail | action_key).then(None).otherwise(pl.col("_entry_actual"))
    actual_exit = pl.when(tail | action_key).then(None).otherwise(pl.col("_exit_actual"))
    label_available = (
        pl.when(actual_exit.is_not_null())
        .then(
            actual_exit
            .dt.combine(pl.lit(_KRX_AVAILABLE_TIME))
            .dt.replace_time_zone("Asia/Seoul")
            .dt.convert_time_zone("UTC")
        )
        .otherwise(None)
    )

    out = panel.with_columns(
        pl.col("_session_date").cast(pl.Date).alias(SESSION_COLUMN),
        pl.lit(horizon_sessions, dtype=pl.Int64).alias("horizon_sessions"),
        pl.lit(policy.policy_id).alias("policy_id"),
        pl.lit(policy.canonical_hash).alias("policy_hash"),
        actual_entry.alias("actual_entry_session"),
        actual_exit.alias("actual_exit_session"),
        entry_disposition.alias("entry_disposition"),
        exit_disposition.alias("exit_disposition"),
        resolution_kind.alias("resolution_kind"),
        status.alias("outcome_status"),
        pl.lit(entry_partition_hash).alias("entry_partition_hash"),
        pl.lit(exit_partition_hash).alias("exit_partition_hash"),
        pl.lit(corporate_action_event_id).alias("corporate_action_event_id"),
        label_available.alias("label_available_time"),
    ).select(*OUTCOME_EVIDENCE_COLUMNS)

    unknown = out.filter(
        ~pl.col("resolution_kind").is_in(list(RESOLUTION_KIND_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("outcome evidence emitted an unknown resolution kind")
    if out.filter(pl.col("outcome_status").is_null()).height:
        raise ValueError("outcome evidence emitted an untagged decision key")
    return out.sort([ID_COLUMN, SESSION_COLUMN, "horizon_sessions"])


def _resolve_open_leg(
    panel: pl.DataFrame,
    source: pl.DataFrame,
    *,
    calendar: KRXSessionCalendar,
    scheduled_pos_col: str,
    max_delay: int,
    prefix: str,
) -> pl.DataFrame:
    """Resolve one entry/exit leg under the policy's forward delay bound.

    Joins the verified bar source at each candidate position from the scheduled
    position up to ``max_delay`` sessions forward, then coalesces to the first
    valid (finite, strictly positive) open. The actual session is the candidate
    date of that first fill and the delay is the forward session count. A null
    open for every candidate leaves the leg unresolved (never zero-filled).
    """
    sessions = list(calendar.sessions)
    pos_lookup = pl.DataFrame(
        {"_pos": list(range(len(sessions))), "_pos_date": sessions}
    )
    candidate = source.select(
        pl.col(ID_COLUMN),
        pl.col(PRICE_DATE_COLUMN).cast(pl.Date).alias("_cand_date"),
        pl.col(OPEN_COLUMN).alias("_cand_open"),
    )
    open_exprs: list[pl.Expr] = []
    delay_exprs: list[pl.Expr] = []
    date_exprs: list[pl.Expr] = []
    for delay in range(max_delay + 1):
        panel = panel.with_columns(
            (pl.col(scheduled_pos_col) + delay).alias("__cand_pos")
        )
        panel = panel.join(pos_lookup, left_on="__cand_pos", right_on="_pos", how="left")
        panel = panel.rename({"_pos_date": f"_{prefix}_cand_{delay}_date"}).drop("__cand_pos")
        panel = panel.join(
            candidate.rename({"_cand_open": f"_{prefix}_open_{delay}"}),
            left_on=[ID_COLUMN, f"_{prefix}_cand_{delay}_date"],
            right_on=[ID_COLUMN, "_cand_date"],
            how="left",
        )
        valid = (
            pl.col(f"_{prefix}_open_{delay}").is_not_null()
            & pl.col(f"_{prefix}_open_{delay}").is_finite()
            & (pl.col(f"_{prefix}_open_{delay}") > 0)
        )
        open_exprs.append(pl.col(f"_{prefix}_open_{delay}"))
        delay_exprs.append(pl.when(valid).then(pl.lit(delay)))
        date_exprs.append(pl.when(valid).then(pl.col(f"_{prefix}_cand_{delay}_date")))
    return panel.with_columns(
        pl.coalesce(*open_exprs).alias(f"_{prefix}_open"),
        pl.coalesce(*delay_exprs).alias(f"{prefix}_delay_sessions"),
        pl.coalesce(*date_exprs).alias(f"_{prefix}_actual"),
    )


def build_outcome_status_sidecar(evidence: pl.DataFrame) -> pl.DataFrame:
    """Reduce one policy-aware evidence frame to the compact status sidecar.

    The sidecar keeps exactly one ``outcome_status`` row per decision key
    ``(instrument_id, session, horizon_sessions)`` from the fixed vocabulary.
    """
    required = [ID_COLUMN, SESSION_COLUMN, "horizon_sessions", "outcome_status"]
    missing = [c for c in required if c not in evidence.columns]
    if missing:
        raise ValueError(f"evidence frame missing {', '.join(missing)}")
    out = evidence.select(*required)
    duplicate = out.group_by([ID_COLUMN, SESSION_COLUMN, "horizon_sessions"]).len().filter(
        pl.col("len") > 1
    )
    if not duplicate.is_empty():
        raise ValueError(
            "outcome evidence must emit exactly one status row per "
            "(instrument_id, session, horizon_sessions)"
        )
    return out.sort([ID_COLUMN, SESSION_COLUMN, "horizon_sessions"])


@dataclass(frozen=True, slots=True)
class OutcomeEvidenceDatasetResult:
    """Immutable outcome of one outcome-evidence dataset publication."""

    dataset_id: str
    manifest: DatasetManifest
    partition_paths: tuple[Path, ...]
    row_count: int
    base_panel_hash: str


_UNRESOLVED_STATUSES = ("MISSING_ENTRY_PRICE", "MISSING_EXIT_PRICE", "UNSUPPORTED_CORPORATE_ACTION")
_DISPOSITION_VOCABULARY = (
    DISPOSITION_FILLED,
    DISPOSITION_NO_BAR,
    DISPOSITION_SOURCE_UNAVAILABLE,
    DISPOSITION_TAIL,
    DISPOSITION_UNSUPPORTED_ACTION,
)


@dataclass(frozen=True, slots=True)
class OutcomeRecoveryReport:
    """Vectorized bounded recovery classification of unresolved outcome rows.

    Groups every unresolved evidence row by ``(horizon_sessions,
    resolution_kind, instrument_id, scheduled entry/exit date,
    outcome_status)`` and partitions them into a recoverable backfill queue
    (``SOURCE_UNAVAILABLE``), verified structural no-bars
    (``CONFIRMED_NO_BAR``), and corporate-action rows awaiting a verified
    settlement (``UNSUPPORTED_CORPORATE_ACTION``). Source hashes, policy hash,
    and dispositions are preserved per group. No recovery path substitutes
    another OHLC field, interpolates a price, or mutates the evidence.
    """

    horizon_sessions: tuple[int, ...]
    rows: pl.DataFrame
    recoverable_backfill_count: int
    confirmed_no_bar_count: int
    unsupported_corporate_action_count: int

    def __post_init__(self) -> None:
        if not self.horizon_sessions:
            raise ValueError("recovery report requires at least one horizon")
        for name in (
            "recoverable_backfill_count",
            "confirmed_no_bar_count",
            "unsupported_corporate_action_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_json(self) -> dict[str, object]:
        return {
            "horizon_sessions": list(self.horizon_sessions),
            "recoverable_backfill_count": int(self.recoverable_backfill_count),
            "confirmed_no_bar_count": int(self.confirmed_no_bar_count),
            "unsupported_corporate_action_count": int(
                self.unsupported_corporate_action_count
            ),
            "grouped_rows": int(self.rows.height),
        }


def build_outcome_recovery_report(
    evidence: pl.DataFrame,
    candidate_horizon_sessions: tuple[int, ...],
) -> OutcomeRecoveryReport:
    """Classify unresolved evidence rows into a bounded recovery report.

    Only rows whose ``outcome_status`` is an unresolved state
    (``MISSING_ENTRY_PRICE``, ``MISSING_EXIT_PRICE``,
    ``UNSUPPORTED_CORPORATE_ACTION``) are recovery candidates; resolved and
    terminal-tail rows are not emitted. Every candidate is grouped with Polars
    group-bys (never a Python row loop) by horizon, resolution kind,
    instrument, scheduled entry/exit date, and status, preserving the pinned
    policy hash, source partition hashes, and dispositions. An unknown
    resolution kind, a horizon partition mismatch, or missing required columns
    raises ``ValueError``.
    """
    if not candidate_horizon_sessions:
        raise ValueError("candidate_horizon_sessions must be non-empty")
    if tuple(candidate_horizon_sessions) != tuple(
        sorted(set(candidate_horizon_sessions))
    ):
        raise ValueError("candidate_horizon_sessions must be strictly ascending and unique")
    required = (
        ID_COLUMN,
        SESSION_COLUMN,
        "horizon_sessions",
        "policy_hash",
        "resolution_kind",
        "scheduled_entry_session",
        "scheduled_exit_session",
        "entry_disposition",
        "exit_disposition",
        "entry_partition_hash",
        "exit_partition_hash",
        "outcome_status",
    )
    missing = [c for c in required if c not in evidence.columns]
    if missing:
        raise ValueError(f"outcome evidence missing recovery columns {missing}")
    present = sorted(evidence["horizon_sessions"].unique().to_list())
    if set(present) != set(candidate_horizon_sessions):
        raise ValueError(
            f"outcome evidence horizon partitions {present}, "
            f"expected {sorted(candidate_horizon_sessions)}"
        )
    unknown = evidence.filter(
        ~pl.col("resolution_kind").is_in(list(RESOLUTION_KIND_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("outcome evidence contains unknown resolution kinds")
    unknown_disposition = evidence.filter(
        pl.col("entry_disposition").is_not_null()
        & ~pl.col("entry_disposition").is_in(list(_DISPOSITION_VOCABULARY))
        | pl.col("exit_disposition").is_not_null()
        & ~pl.col("exit_disposition").is_in(list(_DISPOSITION_VOCABULARY))
    )
    if not unknown_disposition.is_empty():
        raise ValueError("outcome evidence contains unknown entry/exit dispositions")

    unresolved = evidence.filter(
        pl.col("outcome_status").is_in(list(_UNRESOLVED_STATUSES))
    )
    grouped = (
        unresolved.group_by(
            [
                "horizon_sessions",
                "resolution_kind",
                ID_COLUMN,
                "scheduled_entry_session",
                "scheduled_exit_session",
                "outcome_status",
            ]
        )
        .agg(
            pl.len().alias("count"),
            pl.col("policy_hash").first().alias("policy_hash"),
            pl.col("entry_partition_hash").first().alias("entry_partition_hash"),
            pl.col("exit_partition_hash").first().alias("exit_partition_hash"),
            pl.col("entry_disposition").first().alias("entry_disposition"),
            pl.col("exit_disposition").first().alias("exit_disposition"),
        )
        .sort(
            [
                "horizon_sessions",
                "resolution_kind",
                ID_COLUMN,
                "scheduled_entry_session",
                "scheduled_exit_session",
                "outcome_status",
            ]
        )
    )

    def _count(kind: str) -> int:
        return int(unresolved.filter(pl.col("resolution_kind") == kind).height)

    return OutcomeRecoveryReport(
        horizon_sessions=tuple(candidate_horizon_sessions),
        rows=grouped,
        recoverable_backfill_count=_count(RESOLUTION_SOURCE_UNAVAILABLE),
        confirmed_no_bar_count=_count(RESOLUTION_CONFIRMED_NO_BAR),
        unsupported_corporate_action_count=_count(RESOLUTION_UNSUPPORTED_CORPORATE_ACTION),
    )


def publish_outcome_evidence_dataset(
    evidence: pl.DataFrame,
    *,
    destination_root: Path,
    dataset_id: str,
    base_panel_hash: str,
    calendar_hash: str,
    horizon_sessions: tuple[int, ...],
    policy: ExecutionOutcomePolicy,
    provider_version: str = "base-panel-labels",
    universe_policy_version: str = "provisional-legacy",
    certification: DatasetCertification = DatasetCertification.PROVISIONAL,
    generated_time: datetime | None = None,
) -> OutcomeEvidenceDatasetResult:
    """Publish the hash-bound per-key outcome-evidence artifact.

    The manifest and content manifest pin the exact policy id and canonical
    policy hash, so any consumer can verify that the evidence was resolved
    under the same immutable policy as the label/status artifacts.
    """
    if evidence.is_empty():
        raise ValueError("cannot publish an empty outcome-evidence dataset")
    if list(evidence.columns) != list(OUTCOME_EVIDENCE_COLUMNS):
        raise ValueError(
            "outcome-evidence dataset must carry exactly "
            f"{list(OUTCOME_EVIDENCE_COLUMNS)}, got {evidence.columns}"
        )
    present = sorted(evidence["horizon_sessions"].unique().to_list())
    if set(present) != set(horizon_sessions):
        raise ValueError(
            f"outcome-evidence horizon partitions {present}, "
            f"expected {sorted(horizon_sessions)}"
        )
    wrong_policy = evidence.filter(
        (pl.col("policy_id") != policy.policy_id)
        | (pl.col("policy_hash") != policy.canonical_hash)
    )
    if not wrong_policy.is_empty():
        raise ValueError("outcome-evidence dataset carries a foreign policy hash")

    generated_time = generated_time or datetime.now(UTC)
    ordered_columns = list(evidence.columns)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set="outcome_evidence",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=horizon_sessions[0],
        time_start=_as_utc_datetime(evidence[SESSION_COLUMN].min()),
        time_end=_as_utc_datetime(evidence[SESSION_COLUMN].max()),
        provider_version=provider_version,
        universe_policy_version=universe_policy_version,
        row_count=evidence.height,
        generated_time=generated_time,
        certification=certification,
        calendar_hash=calendar_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(evidence, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    content_manifest: dict[str, object] = {
        "base_panel_hash": base_panel_hash,
        "calendar_hash": calendar_hash,
        "label_definition": "net_alpha_o2o",
        "horizon_sessions": list(horizon_sessions),
        "policy_id": policy.policy_id,
        "policy_hash": policy.canonical_hash,
        "resolution_kind_vocabulary": list(RESOLUTION_KIND_VOCABULARY),
        "generated_time": generated_time.isoformat(),
    }
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        evidence,
        dataset_id=dataset_id,
        manifest=manifest,
        expected_feature_set="outcome_evidence",
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    return OutcomeEvidenceDatasetResult(
        dataset_id=dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=evidence.height,
        base_panel_hash=base_panel_hash,
    )
