"""Timestamped tradability-status classification for pre-entry exclusion.

Classifies immutable official KRX-status or OpenDART disclosure evidence into a
canonical tradability state per ``(instrument_id, published_at)``. An event
published after a historical decision cutoff is never used to reject that
decision. Only the hard-exclusion states (``ACTIVE_HALT``,
``DELISTING_OR_SETTLEMENT``, ``CORPORATE_CONTINUITY_BREAK``) exclude an entry;
candidate-only keyword matches are ``WATCH_ONLY`` and never hard-exclude.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
)

ID_COLUMN = "instrument_id"
PUBLISHED_AT_COLUMN = "published_at"
SOURCE_COLUMN = "source"
SOURCE_ID_COLUMN = "source_id"
EVENT_KIND_COLUMN = "event_kind"
EFFECTIVE_SESSION_COLUMN = "effective_session"
RAW_RESPONSE_HASH_COLUMN = "raw_response_hash"
TRADABILITY_STATE_COLUMN = "tradability_state"

TRADABILITY_STATE_ACTIVE_HALT = "ACTIVE_HALT"
TRADABILITY_STATE_DELISTING_OR_SETTLEMENT = "DELISTING_OR_SETTLEMENT"
TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK = "CORPORATE_CONTINUITY_BREAK"
TRADABILITY_STATE_WATCH_ONLY = "WATCH_ONLY"

TRADABILITY_STATE_VOCABULARY = (
    TRADABILITY_STATE_ACTIVE_HALT,
    TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
    TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK,
    TRADABILITY_STATE_WATCH_ONLY,
)

HARD_EXCLUSION_STATES = (
    TRADABILITY_STATE_ACTIVE_HALT,
    TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
    TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK,
)

# Official sources with authoritative tradability status. Browser search is an
# investigative fallback only and is never a production input.
OFFICIAL_SOURCES = ("KRX", "OPENDART")

_EVENT_KIND_ACTIVE_HALT = "TRADING_HALT"
_EVENT_KIND_DELISTING = "DELISTING"
_EVENT_KIND_SETTLEMENT = "SETTLEMENT"
_EVENT_KIND_CONTINUITY_BREAK = "CORPORATE_CONTINUITY_BREAK"

_EVENT_KIND_TO_STATE = {
    _EVENT_KIND_ACTIVE_HALT: TRADABILITY_STATE_ACTIVE_HALT,
    _EVENT_KIND_DELISTING: TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
    _EVENT_KIND_SETTLEMENT: TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
    _EVENT_KIND_CONTINUITY_BREAK: TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK,
}

TRADABILITY_EVENTS_COLUMNS = (
    ID_COLUMN,
    PUBLISHED_AT_COLUMN,
    SOURCE_COLUMN,
    SOURCE_ID_COLUMN,
    EVENT_KIND_COLUMN,
    EFFECTIVE_SESSION_COLUMN,
    RAW_RESPONSE_HASH_COLUMN,
    TRADABILITY_STATE_COLUMN,
)


def load_tradability_event_evidence(
    catalog: CatalogStore,
    dataset_id: str | None,
    decision_time: datetime,
) -> tuple[CatalogEntry | None, pl.DataFrame | None]:
    """Load hash-bound official event evidence from a catalog JSON artifact."""
    if dataset_id is None:
        return None, None
    entry = catalog.get(CatalogKind.CORPORATE_ACTIONS, dataset_id)
    if entry is None or entry.completeness is not EvidenceCompleteness.COMPLETE:
        raise ValueError(f"tradability event dataset is not complete: {dataset_id!r}")
    path = Path(entry.path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry.content_hash:
        raise ValueError(f"tradability event dataset hash mismatch: {dataset_id!r}")
    payload = json.loads(content)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("tradability event records must be a list")
    if not records:
        return entry, pl.DataFrame(
            schema=dict.fromkeys(TRADABILITY_EVENTS_COLUMNS, pl.Utf8)
        )
    frame = pl.DataFrame(records).select(
        pl.col(ID_COLUMN).cast(pl.Utf8),
        pl.col(PUBLISHED_AT_COLUMN).str.to_datetime(time_zone="UTC"),
        pl.col(SOURCE_COLUMN).cast(pl.Utf8),
        pl.col(SOURCE_ID_COLUMN).cast(pl.Utf8),
        pl.col(EVENT_KIND_COLUMN).cast(pl.Utf8),
        pl.col(EFFECTIVE_SESSION_COLUMN).str.to_date(),
        pl.col(RAW_RESPONSE_HASH_COLUMN).cast(pl.Utf8),
    )
    return entry, classify_tradability_events(frame, decision_cutoff=decision_time)


def classify_tradability_events(
    events: pl.DataFrame,
    *,
    decision_cutoff: datetime,
) -> pl.DataFrame:
    """Classify immutable official tradability events as-of a decision cutoff.

    ``events`` must carry ``instrument_id``, ``published_at`` (UTC datetime),
    ``source`` (an official ``KRX`` or ``OPENDART`` value), ``source_id``,
    ``event_kind``, ``effective_session`` (date), and ``raw_response_hash``.
    Only events published at or before ``decision_cutoff`` are retained, so a
    disclosure published after a historical decision never modifies it. Each
    event emits one canonical ``tradability_state`` from
    ``TRADABILITY_STATE_VOCABULARY``; an event kind without a versioned rule
    maps to ``WATCH_ONLY`` and never hard-excludes an entry.

    Raises ``ValueError`` for a missing column, a non-official source, a null
    ``published_at``/``effective_session``, an empty ``source_id``/hash, or a
    duplicate ``(instrument_id, published_at)`` key.
    """
    required = (
        ID_COLUMN,
        PUBLISHED_AT_COLUMN,
        SOURCE_COLUMN,
        SOURCE_ID_COLUMN,
        EVENT_KIND_COLUMN,
        EFFECTIVE_SESSION_COLUMN,
        RAW_RESPONSE_HASH_COLUMN,
    )
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"tradability events missing columns {missing}")
    if not isinstance(decision_cutoff, datetime) or decision_cutoff.tzinfo is None:
        raise ValueError("decision_cutoff must be a timezone-aware datetime")
    cutoff = decision_cutoff.astimezone(UTC)

    invalid = events.filter(
        pl.col(PUBLISHED_AT_COLUMN).is_null()
        | pl.col(EFFECTIVE_SESSION_COLUMN).is_null()
        | (pl.col(SOURCE_ID_COLUMN).cast(pl.Utf8).str.strip_chars() == "")
        | (pl.col(RAW_RESPONSE_HASH_COLUMN).cast(pl.Utf8).str.strip_chars() == "")
        | ~pl.col(SOURCE_COLUMN).is_in(list(OFFICIAL_SOURCES))
    )
    if not invalid.is_empty():
        raise ValueError(
            "tradability events contain a null timestamp/session, an empty "
            "source identifier/hash, or a non-official source"
        )
    duplicate = (
        events.group_by([ID_COLUMN, PUBLISHED_AT_COLUMN])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate.is_empty():
        raise ValueError(
            "tradability events contain duplicate (instrument_id, published_at) keys"
        )

    out = (
        events.filter(
            pl.col(PUBLISHED_AT_COLUMN).dt.convert_time_zone("UTC") <= cutoff
        )
        .with_columns(
            pl.col(EVENT_KIND_COLUMN)
            .replace_strict(
                _EVENT_KIND_TO_STATE,
                default=TRADABILITY_STATE_WATCH_ONLY,
                return_dtype=pl.Utf8,
            )
            .alias(TRADABILITY_STATE_COLUMN)
        )
        .select(*TRADABILITY_EVENTS_COLUMNS)
        .sort([ID_COLUMN, PUBLISHED_AT_COLUMN])
    )
    unknown = out.filter(
        ~pl.col(TRADABILITY_STATE_COLUMN).is_in(list(TRADABILITY_STATE_VOCABULARY))
    )
    if not unknown.is_empty():
        raise ValueError("tradability events emitted an unknown canonical state")
    return out
