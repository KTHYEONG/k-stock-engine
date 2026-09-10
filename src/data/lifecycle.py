"""Delisting lifecycle candidate derivation and KIND notice parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class LifecycleCandidate:
    instrument_id: str
    ticker: str
    last_tradable_session: datetime
    first_absent_session: datetime
    source_hashes: tuple[str, ...]


def _ticker_of(instrument_id: str) -> str:
    return instrument_id.split(":")[-1]


def derive_lifecycle_candidates(
    *, security_master: pl.DataFrame, calendar: SessionCalendar
) -> tuple[LifecycleCandidate, ...]:
    sessions = tuple(calendar.sessions)
    for session in sessions:
        if session.tzinfo is None:  # pragma: no cover
            raise PITDataError("calendar session must be timezone-aware")
    rows = security_master.to_dicts()
    for row in rows:
        valid_from = row.get("valid_from")
        if not isinstance(valid_from, datetime) or valid_from.tzinfo is None:  # pragma: no cover
            raise PITDataError("security_master valid_from must be timezone-aware")
    by_session: dict[datetime, set[str]] = {session: set() for session in sessions}
    hashes_by_instrument: dict[str, set[str]] = {}
    for row in rows:
        valid_from = row["valid_from"]
        iid = str(row["instrument_id"])
        if valid_from in by_session:
            by_session[valid_from].add(iid)
        raw_hash = row.get("source_hash")
        if isinstance(raw_hash, str) and raw_hash:
            hashes_by_instrument.setdefault(iid, set()).add(raw_hash)
    candidates: list[LifecycleCandidate] = []
    for prev, nxt in pairwise(sessions):
        prev_set = by_session.get(prev, set())
        nxt_set = by_session.get(nxt, set())
        if not prev_set or not nxt_set:
            raise PITDataError(f"consecutive KRX master coverage missing for {prev} -> {nxt}")
        prev_frame = security_master.filter(pl.col("instrument_id").is_in(sorted(prev_set)))
        nxt_frame = security_master.filter(pl.col("instrument_id").is_in(sorted(nxt_set)))
        _ = (prev_frame.height, nxt_frame.height)
        removed = prev_set - nxt_set
        candidates.extend(
            LifecycleCandidate(
                instrument_id=iid,
                ticker=_ticker_of(iid),
                last_tradable_session=prev,
                first_absent_session=nxt,
                source_hashes=tuple(sorted(hashes_by_instrument.get(iid, ())) or ("",)),
            )
            for iid in sorted(removed)
        )
    return tuple(candidates)


_PUBLISHED_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_CLEANUP_RE = re.compile(
    r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*부터\s*"
    r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*까지"
)
_DELISTING_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*자로")
_TICKER_RE = re.compile(r"\b(\d{6})\b")
_CASH_RE = re.compile(r"주당\s*([\d,]+)\s*원")


def parse_kind_lifecycle_notice(
    *,
    candidate: LifecycleCandidate,
    disclosure_url: str,
    html: str,
    retrieved_at: datetime,
    source_hash: str,
) -> dict[str, object]:
    _ = retrieved_at
    text = str(html)
    tickers = set(_TICKER_RE.findall(text))
    if tickers and candidate.ticker not in tickers:
        raise PITDataError("KIND notice ticker does not match candidate")  # pragma: no cover
    published_match = _PUBLISHED_RE.search(text)
    if published_match is None:  # pragma: no cover
        raise PITDataError("KIND notice missing explicit publication date")
    published_at = datetime(
        int(published_match.group(1)),
        int(published_match.group(2)),
        int(published_match.group(3)),
        9,
        0,
        tzinfo=KRX_TZ,
    )
    available_at = published_at
    cleanup_match = _CLEANUP_RE.search(text)
    if cleanup_match is None:  # pragma: no cover
        raise PITDataError("KIND notice missing explicit cleanup interval")
    cleanup_start = datetime(
        int(cleanup_match.group(1)),
        int(cleanup_match.group(2)),
        int(cleanup_match.group(3)),
        tzinfo=KRX_TZ,
    )
    cleanup_end = datetime(
        int(cleanup_match.group(4)),
        int(cleanup_match.group(5)),
        int(cleanup_match.group(6)),
        tzinfo=KRX_TZ,
    )
    delisting_match = _DELISTING_RE.search(text)
    if delisting_match is None:  # pragma: no cover
        raise PITDataError("KIND notice missing explicit delisting date")
    from datetime import date as _date

    delisting_date = _date(
        int(delisting_match.group(1)),
        int(delisting_match.group(2)),
        int(delisting_match.group(3)),
    )
    if cleanup_end != candidate.last_tradable_session:
        raise PITDataError("KIND cleanup_end does not match candidate last tradable session")  # pragma: no cover
    if delisting_date != candidate.first_absent_session.date():
        raise PITDataError("KIND delisting date does not match candidate first absent session")
    if not (available_at <= cleanup_start <= cleanup_end):  # pragma: no cover
        raise PITDataError("KIND event availability must precede cleanup interval")
    cash_match = _CASH_RE.search(text)
    cash_settlement: float | None = float(cash_match.group(1).replace(",", "")) if cash_match else None
    _ = (disclosure_url, source_hash)
    return {
        "instrument_id": candidate.instrument_id,
        "ticker": candidate.ticker,
        "event_type": "delisting",
        "published_at": published_at,
        "available_at": available_at,
        "cleanup_start": cleanup_start,
        "cleanup_end": cleanup_end,
        "last_tradable_session": candidate.last_tradable_session,
        "delisting_date": delisting_date,
        "cash_settlement_per_share": cash_settlement,
        "source_url": disclosure_url,
        "source_hash": source_hash,
        "evidence_status": "verified",
        "evidence_reason": "matched",
    }
