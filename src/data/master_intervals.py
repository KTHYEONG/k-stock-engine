"""SCD2 interval compaction for security-master snapshots."""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.core.time import KRX_TZ
from src.data.schemas import PITDataError

MASTER_INTERVAL_KEY_COLUMNS: tuple[str, ...] = ("instrument_id", "valid_from", "valid_to", "available_at")
MASTER_INTERVAL_LINEAGE_COLUMNS: frozenset[str] = frozenset({"source_hash"})
MASTER_SENTINEL_VALUE = "__UNKNOWN__"
MASTER_SENTINEL_COLUMNS: tuple[str, ...] = ("market", "status", "sector")


def resolve_sentinel_master_conflicts(master: pl.DataFrame) -> pl.DataFrame:
    """Drop placeholder master rows dominated by real evidence on the same key.

    The bootstrap normalization wrote both a fully sentinel placeholder and the
    provider-backed row for the same ``(instrument_id, valid_from)``, so the
    certified table violates its own primary key. A row whose sentinel count is
    strictly higher than a sibling's carries no fact the sibling lacks, so
    discarding it removes a duplicate rather than imputing a value. Ties are
    left intact for the caller's primary-key gate to reject: equal evidence
    density is genuine ambiguity and must never be resolved by row order.

    Args:
        master: Security-master rows, possibly with duplicate primary keys.

    Returns:
        Frame with the same columns, keeping only minimal-sentinel rows per key.
    """
    present = [name for name in MASTER_SENTINEL_COLUMNS if name in master.columns]
    if not present or master.height == 0:
        return master
    sentinel_count = pl.lit(0, dtype=pl.Int32)
    for name in present:
        sentinel_count = sentinel_count + (pl.col(name) == MASTER_SENTINEL_VALUE).fill_null(False).cast(pl.Int32)
    counted = master.with_columns(sentinel_count.alias("_sentinel_count"))
    minimal = pl.col("_sentinel_count").min().over(["instrument_id", "valid_from"])
    return counted.filter(pl.col("_sentinel_count") == minimal).select(master.columns)


def compact_security_master_intervals(master: pl.DataFrame, *, sessions: tuple[datetime, ...]) -> pl.DataFrame:
    """Fold consecutive daily snapshots into minimal SCD2 intervals.

    Two session-adjacent rows of one instrument collapse into a single
    interval only when all merge conditions hold: identical attribute tuples,
    exactly +1 session-rank adjacency, and single-session PIT stamps (both
    rows carry available_at and valid_to on their own valid_from date), so a
    late receipt never backdates and a pre-declared range never re-folds.

    Placeholder rows dominated by real evidence on the same primary key are
    dropped first (see :func:`resolve_sentinel_master_conflicts`); any residual
    duplicate is genuine ambiguity and fails closed.

    Args:
        master: Daily security-master snapshots.
        sessions: Certified calendar sessions in strictly increasing order.

    Returns:
        Interval frame with identical column names, order, and dtypes,
        sorted by (instrument_id, valid_from).

    Raises:
        PITDataError: If sessions are empty or unordered, a key column is
            missing, a timestamp is naive, a valid_from falls outside the
            calendar, or an (instrument_id, valid_from) key repeats.
    """
    if len(sessions) == 0 or any(sessions[index] >= sessions[index + 1] for index in range(len(sessions) - 1)):
        message = "sessions must be non-empty" if len(sessions) == 0 else "sessions must be strictly increasing"
        raise PITDataError(message)
    for name in MASTER_INTERVAL_KEY_COLUMNS:
        if name not in master.columns:
            raise PITDataError(f"security master missing required column: {name}")
    if master.height == 0:
        return master.clear()
    for name in ("valid_from", "valid_to", "available_at"):
        if getattr(master.schema[name], "time_zone", None) is None:
            raise PITDataError("security master timestamps must be timezone-aware")
    # 증거 우세 행만 남긴 뒤에도 남는 PK 중복은 진짜 모호성이므로 아래에서 차단된다.
    resolved = resolve_sentinel_master_conflicts(master)
    # 세션 인접 판정은 KRX 일자 기준 (00:00 캘린더 앵커와 09:00 마스터 앵커 혼재).
    session_dates = [session.astimezone(KRX_TZ).date() for session in sessions]
    rank_by_date = {day: rank for rank, day in enumerate(dict.fromkeys(session_dates))}
    ordered = resolved.sort(["instrument_id", "valid_from"])
    dated = ordered.with_columns(
        pl.col("valid_from").dt.convert_time_zone(str(KRX_TZ)).dt.date().alias("_valid_date"),
        pl.col("available_at").dt.convert_time_zone(str(KRX_TZ)).dt.date().alias("_available_date"),
        pl.col("valid_to").dt.convert_time_zone(str(KRX_TZ)).dt.date().alias("_valid_to_date"),
    )
    if not dated.select(pl.col("_valid_date").is_in(session_dates).all()).item(0, 0):
        raise PITDataError("security master valid_from is outside the certified calendar")
    if ordered.select(["instrument_id", "valid_from"]).is_duplicated().any():
        raise PITDataError("duplicate security master primary key")
    rank_frame = pl.DataFrame(
        {
            "_valid_date": list(rank_by_date),
            "_session_rank": list(rank_by_date.values()),
        }
    )
    ranked = dated.join(rank_frame, on="_valid_date", how="left")
    # lineage 컬럼은 병합 키에서 제외하고 구간 첫 값을 취한다.
    attribute_columns = [
        column
        for column in master.columns
        if column not in MASTER_INTERVAL_KEY_COLUMNS and column not in MASTER_INTERVAL_LINEAGE_COLUMNS
    ]
    keyed = ranked.with_columns(
        (pl.struct(attribute_columns) if attribute_columns else pl.lit(0)).alias("_attribute_key"),
    )
    # 병합 조건: 동일 속성 + 세션 순위 +1 인접 + 양쪽 행 단일 세션 PIT 스탬프.
    flagged = keyed.with_columns(
        (
            (pl.col("_attribute_key") == pl.col("_attribute_key").shift(1).over("instrument_id"))
            & ((pl.col("_session_rank") - pl.col("_session_rank").shift(1).over("instrument_id")) == 1)
            & (pl.col("_available_date") == pl.col("_valid_date"))
            & (
                pl.col("_available_date").shift(1).over("instrument_id")
                == pl.col("_valid_date").shift(1).over("instrument_id")
            )
            & (pl.col("_valid_to_date") == pl.col("_valid_date"))
            & (
                pl.col("_valid_to_date").shift(1).over("instrument_id")
                == pl.col("_valid_date").shift(1).over("instrument_id")
            )
        )
        .fill_null(False)
        .alias("_continues"),
    )
    grouped = flagged.with_columns(
        pl.col("_continues").not_().cum_sum().over("instrument_id").alias("_interval_group"),
    )
    aggregations = [
        pl.col(name).min().alias(name)
        if name in ("valid_from", "available_at")
        else pl.col(name).max().alias(name)
        if name == "valid_to"
        else pl.col(name).first().alias(name)
        for name in master.columns
        if name != "instrument_id"
    ]
    return (
        grouped.group_by(["instrument_id", "_interval_group"], maintain_order=True)
        .agg(aggregations)
        .sort(["instrument_id", "valid_from"])
        .select(master.columns)
    )
