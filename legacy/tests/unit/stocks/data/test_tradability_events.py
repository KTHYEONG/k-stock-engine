"""Timestamped tradability-status classification tests."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from legacy.stocks.data.tradability_events import (
    HARD_EXCLUSION_STATES,
    TRADABILITY_EVENTS_COLUMNS,
    TRADABILITY_STATE_ACTIVE_HALT,
    TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK,
    TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
    TRADABILITY_STATE_WATCH_ONLY,
    classify_tradability_events,
)


def _events() -> pl.DataFrame:
    published = datetime(2024, 1, 10, tzinfo=UTC)
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002", "KRX:00003", "KRX:00004"],
            "published_at": [
                published,
                published,
                published + timedelta(days=1),
                published + timedelta(days=2),
            ],
            "source": ["KRX", "OPENDART", "KRX", "OPENDART"],
            "source_id": ["krx-1", "dart-1", "krx-2", "dart-2"],
            "event_kind": ["TRADING_HALT", "DELISTING", "SETTLEMENT", "KEYWORD_MATCH"],
            "effective_session": [
                date(2024, 1, 11),
                date(2024, 1, 12),
                date(2024, 1, 13),
                date(2024, 1, 14),
            ],
            "raw_response_hash": ["h-1", "h-2", "h-3", "h-4"],
        }
    )


def test_classify_emits_canonical_states_and_cutoff_boundary() -> None:
    cutoff = datetime(2024, 1, 11, tzinfo=UTC)
    classified = classify_tradability_events(_events(), decision_cutoff=cutoff)
    assert list(classified.columns) == list(TRADABILITY_EVENTS_COLUMNS)
    states = {
        row["instrument_id"]: row["tradability_state"]
        for row in classified.iter_rows(named=True)
    }
    assert states["KRX:00001"] == TRADABILITY_STATE_ACTIVE_HALT
    assert states["KRX:00002"] == TRADABILITY_STATE_DELISTING_OR_SETTLEMENT
    # KRX:00003 published exactly at the cutoff is retained; KRX:00004
    # published one day after the cutoff must never be used.
    assert states["KRX:00003"] == TRADABILITY_STATE_DELISTING_OR_SETTLEMENT
    assert "KRX:00004" not in states


def test_classify_unknown_kind_is_watch_only() -> None:
    classified = classify_tradability_events(
        _events().head(4), decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC)
    )
    watch = classified.filter(
        pl.col("instrument_id") == "KRX:00004"
    ).row(0, named=True)
    assert watch["tradability_state"] == TRADABILITY_STATE_WATCH_ONLY


def test_classify_hard_exclusion_states_are_exactly_three() -> None:
    assert set(HARD_EXCLUSION_STATES) == {
        TRADABILITY_STATE_ACTIVE_HALT,
        TRADABILITY_STATE_DELISTING_OR_SETTLEMENT,
        TRADABILITY_STATE_CORPORATE_CONTINUITY_BREAK,
    }
    assert TRADABILITY_STATE_WATCH_ONLY not in HARD_EXCLUSION_STATES


def test_classify_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        classify_tradability_events(
            _events().drop("source_id"),
            decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC),
        )


def test_classify_rejects_non_official_source() -> None:
    events = _events().with_columns(pl.lit("BROWSER").alias("source"))
    with pytest.raises(ValueError, match="non-official source"):
        classify_tradability_events(
            events, decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC)
        )


def test_classify_rejects_null_or_empty_provenance() -> None:
    events = _events().with_columns(
        pl.when(pl.col("instrument_id") == "KRX:00001")
        .then(None)
        .otherwise(pl.col("published_at"))
        .alias("published_at")
    )
    with pytest.raises(ValueError, match="null timestamp/session"):
        classify_tradability_events(
            events, decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC)
        )
    empty_id = _events().with_columns(
        pl.when(pl.col("instrument_id") == "KRX:00001")
        .then(pl.lit(""))
        .otherwise(pl.col("source_id"))
        .alias("source_id")
    )
    with pytest.raises(ValueError, match="empty source identifier"):
        classify_tradability_events(
            empty_id, decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC)
        )


def test_classify_rejects_duplicate_instrument_timestamp() -> None:
    events = pl.concat([_events(), _events()])
    with pytest.raises(ValueError, match="duplicate"):
        classify_tradability_events(
            events, decision_cutoff=datetime(2024, 1, 20, tzinfo=UTC)
        )


def test_classify_rejects_naive_cutoff() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        classify_tradability_events(_events(), decision_cutoff=datetime(2024, 1, 20))
