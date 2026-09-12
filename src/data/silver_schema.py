"""Canonical Silver session-key normalization and time-semantics observation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.core.time import KRX_TZ
from src.data.schemas import PITDataError, SilverTable

CANONICAL_SESSION_TZ = "Asia/Seoul"
CANONICAL_SESSION_HOUR = 9

SESSION_KEY_COLUMNS: frozenset[str] = frozenset(
    {
        "session",
        "decision_session",
        "valid_from",
        "valid_to",
        "effective_date",
        "cleanup_start",
        "cleanup_end",
        "last_tradable_session",
    }
)

AVAILABILITY_COLUMNS: frozenset[str] = frozenset({"available_at", "published_at", "ingested_at"})

SILVER_SESSION_KEY_COLUMN: Mapping[SilverTable, str | None] = {
    SilverTable.CALENDAR: "session",
    SilverTable.SECURITY_MASTER: "valid_from",
    SilverTable.DAILY_MARKET: "session",
    SilverTable.INVESTOR_FLOW: "session",
    SilverTable.FINANCIAL_FACTS: None,
    SilverTable.CORPORATE_ACTIONS: "effective_date",
    SilverTable.DISCLOSURES: None,
    SilverTable.HISTORICAL_COSTS: "effective_date",
    SilverTable.LIFECYCLE_EVENTS: "last_tradable_session",
}


@dataclass(frozen=True, slots=True)
class TimeSemanticsObservation:
    """Observed physical encoding of one temporal column.

    Attributes:
        column: Column name that was observed.
        time_zone: Storage time zone label, or None for tz-naive columns.
        hour_anchors: Distinct wall-clock hours in ascending order.
        canonical: Whether the encoding matches the canonical contract.
    """

    column: str
    time_zone: str | None
    hour_anchors: tuple[int, ...]
    canonical: bool


def canonicalize_session_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """Re-anchor session-key columns to the canonical KRX session label.

    Only ``SESSION_KEY_COLUMNS`` members with a Datetime dtype are rewritten:
    each value is converted to ``KRX_TZ`` and then re-anchored to
    ``CANONICAL_SESSION_HOUR:00:00`` on the same KRX date. Availability
    instants, non-Datetime columns, and absent columns pass through untouched.

    Args:
        frame: Silver frame whose session keys use heterogeneous encodings.

    Returns:
        Frame with canonical session keys and unchanged column order.

    Raises:
        PITDataError: If a session-key column is tz-naive.
    """
    expressions: list[pl.Expr] = []
    for name in frame.columns:
        # 가용시각은 instant 이므로 재앵커링 대상에서 제외한다.
        if name not in SESSION_KEY_COLUMNS:
            continue
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime):
            continue
        if dtype.time_zone is None:
            raise PITDataError("session key must be timezone-aware")
        expressions.append(
            pl.col(name)
            .dt.convert_time_zone(str(KRX_TZ))
            .dt.truncate("1d")
            .dt.offset_by(f"{CANONICAL_SESSION_HOUR}h")
            .alias(name)
        )
    if not expressions:
        return frame
    # KRX 날짜를 보존한 채 정규 시각으로 재라벨링한다 (멱등).
    return frame.with_columns(expressions)


def observe_time_semantics(frame: pl.DataFrame) -> tuple[TimeSemanticsObservation, ...]:
    """Observe the physical time encoding of each temporal column.

    Args:
        frame: Silver frame to inspect.

    Returns:
        Per-column observations in ascending column order, or an empty tuple
        when the frame carries no temporal column.
    """
    watched = SESSION_KEY_COLUMNS | AVAILABILITY_COLUMNS
    observations: list[TimeSemanticsObservation] = []
    for name in sorted(watched):
        if name not in frame.columns:
            continue
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime):
            continue
        time_zone = dtype.time_zone
        # 0행 프레임은 빈 앵커로 보고하고 예외를 던지지 않는다.
        hour_anchors = tuple(frame.select(pl.col(name).dt.hour().unique().sort()).to_series().to_list())
        if name in SESSION_KEY_COLUMNS:
            canonical = time_zone == CANONICAL_SESSION_TZ and hour_anchors == (CANONICAL_SESSION_HOUR,)
        else:
            canonical = time_zone is not None
        observations.append(
            TimeSemanticsObservation(
                column=name,
                time_zone=time_zone,
                hour_anchors=hour_anchors,
                canonical=canonical,
            )
        )
    return tuple(observations)
