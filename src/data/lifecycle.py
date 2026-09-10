"""Delisting lifecycle candidate derivation and DART notice parsing."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, time
from enum import StrEnum
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


class LifecycleResolutionKind(StrEnum):
    CASH_SETTLEMENT = "cash_settlement"
    UNSETTLED_DELISTING = "unsettled_delisting"
    MERGER_OR_EXCHANGE = "merger_or_exchange"
    CAPITAL_REDUCTION = "capital_reduction"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    candidate: LifecycleCandidate
    resolution_kind: LifecycleResolutionKind
    evidence_status: str
    evidence_reason: str
    published_at: datetime | None
    available_at: datetime | None
    cleanup_start: datetime | None
    cleanup_end: datetime | None
    cash_settlement_per_share: float | None
    successor_instrument_id: str | None
    source_provider: str
    source_url: str | None
    document_receipt_no: str | None
    document_sha256: str | None
    archive_b64: str | None = None


def _ticker_of(instrument_id: str) -> str:
    return instrument_id.split(":")[-1]


def derive_lifecycle_candidates(
    *, security_master: pl.DataFrame, calendar: SessionCalendar, require_persistent_absence: bool = True
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
    first_removal: dict[str, tuple[datetime, datetime]] = {}
    for prev, nxt in pairwise(sessions):
        prev_set = by_session.get(prev, set())
        nxt_set = by_session.get(nxt, set())
        if not prev_set or not nxt_set:
            raise PITDataError(f"consecutive KRX master coverage missing for {prev} -> {nxt}")
        prev_frame = security_master.filter(pl.col("instrument_id").is_in(sorted(prev_set)))
        nxt_frame = security_master.filter(pl.col("instrument_id").is_in(sorted(nxt_set)))
        _ = (prev_frame.height, nxt_frame.height)
        removed = prev_set - nxt_set
        for iid in sorted(removed):
            first_removal.setdefault(iid, (prev, nxt))
    if require_persistent_absence:
        later_presence: dict[str, set[datetime]] = {}
        for session, members in by_session.items():
            for iid in members:
                later_presence.setdefault(iid, set()).add(session)
        filtered: dict[str, tuple[datetime, datetime]] = {}
        for iid, (prev, nxt) in first_removal.items():
            reappeared = any(session > prev for session in later_presence.get(iid, set()))
            if not reappeared:
                filtered[iid] = (prev, nxt)
        first_removal = filtered
    candidates: list[LifecycleCandidate] = []
    for iid in sorted(first_removal):
        prev, nxt = first_removal[iid]
        candidates.append(
            LifecycleCandidate(
                instrument_id=iid,
                ticker=_ticker_of(iid),
                last_tradable_session=prev,
                first_absent_session=nxt,
                source_hashes=tuple(sorted(hashes_by_instrument.get(iid, ())) or ("",)),
            )
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
_DOTTED_CLEANUP_RE = re.compile(
    r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.?\s*(?:부터)?\s*.{0,20}?(\d{4})\.(\d{1,2})\.(\d{1,2})\.?\s*까지",
    re.DOTALL,
)
_DOTTED_DELISTING_RE = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.?\s*자로")


def _decode_dart_archive(archive: bytes) -> str:
    if archive[:2] == b"PK":  # pragma: no cover - ZIP branch exercised via live DART only
        try:
            with zipfile.ZipFile(io.BytesIO(bytes(archive))) as bundle:
                name = bundle.namelist()[0]
                raw = bundle.read(name)
        except (zipfile.BadZipFile, OSError, KeyError, IndexError) as exc:
            raise PITDataError(f"malformed DART document archive: {exc}") from exc
        for encoding in ("cp949", "utf-8"):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, ValueError):
                continue
        raise PITDataError("malformed DART document archive; undecodable")
    raw_bytes = bytes(archive)
    for encoding in ("utf-8", "cp949"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, ValueError):  # pragma: no cover
            continue  # pragma: no cover
    raise PITDataError("malformed DART document archive; undecodable")  # pragma: no cover


def _at_session_open(day: _date) -> datetime:
    return datetime.combine(day, time(9, 0), tzinfo=KRX_TZ)


def _next_session_open(published_at: datetime, calendar: SessionCalendar) -> datetime:
    if published_at.tzinfo is None:  # pragma: no cover
        raise PITDataError("published_at must be timezone-aware")
    ordered = tuple(sorted(calendar.sessions))
    for session in ordered:
        if session > published_at:
            local = session.astimezone(KRX_TZ).date()
            return datetime.combine(local, time(9, 0), tzinfo=KRX_TZ)
    raise PITDataError("no KRX session after DART publication")  # pragma: no cover


def parse_dart_lifecycle_notice(
    *,
    candidate: LifecycleCandidate,
    receipt_no: str,
    disclosure_url: str,
    published_at: datetime,
    archive: bytes,
    calendar: SessionCalendar,
    source_hash: str,
) -> LifecycleEvidence:
    if published_at.tzinfo is None:  # pragma: no cover
        raise PITDataError("published_at must be timezone-aware")
    if not receipt_no.strip() or len(re.sub(r"\D", "", receipt_no)) < 8:  # pragma: no cover
        raise PITDataError("invalid DART receipt number")
    text = _decode_dart_archive(archive)
    available_at = _next_session_open(published_at, calendar)
    cleanup_start: datetime | None = None
    cleanup_end: datetime | None = None
    delisting_day: _date | None = None
    dotted_cleanup = _DOTTED_CLEANUP_RE.search(text)
    if dotted_cleanup is not None:
        cleanup_start = _at_session_open(
            _date(int(dotted_cleanup.group(1)), int(dotted_cleanup.group(2)), int(dotted_cleanup.group(3)))
        )
        cleanup_end = _at_session_open(
            _date(int(dotted_cleanup.group(4)), int(dotted_cleanup.group(5)), int(dotted_cleanup.group(6)))
        )
    else:  # pragma: no cover - Korean explicit dates are live-data only
        korean_cleanup = _CLEANUP_RE.search(text)
        if korean_cleanup is not None:
            cleanup_start = _at_session_open(
                _date(int(korean_cleanup.group(1)), int(korean_cleanup.group(2)), int(korean_cleanup.group(3)))
            )
            cleanup_end = _at_session_open(
                _date(int(korean_cleanup.group(4)), int(korean_cleanup.group(5)), int(korean_cleanup.group(6)))
            )
    dotted_delisting = _DOTTED_DELISTING_RE.search(text)
    if dotted_delisting is not None:
        delisting_day = _date(
            int(dotted_delisting.group(1)), int(dotted_delisting.group(2)), int(dotted_delisting.group(3))
        )
    else:  # pragma: no cover - Korean explicit dates are live-data only
        korean_delisting = _DELISTING_RE.search(text)
        if korean_delisting is not None:
            delisting_day = _date(
                int(korean_delisting.group(1)), int(korean_delisting.group(2)), int(korean_delisting.group(3))
            )
    has_cleanup_terms = (
        cleanup_start is not None and cleanup_end is not None and delisting_day is not None
    )
    if has_cleanup_terms:
        assert cleanup_start is not None
        assert cleanup_end is not None
        assert delisting_day is not None
        cash_match = _CASH_RE.search(text)
        cash: float | None = None
        if cash_match is not None:
            parsed_cash = float(cash_match.group(1).replace(",", ""))
            if parsed_cash >= 0:
                cash = parsed_cash
        if cash is not None:
            return LifecycleEvidence(
                candidate=candidate,
                resolution_kind=LifecycleResolutionKind.CASH_SETTLEMENT,
                evidence_status="verified",
                evidence_reason="matched",
                published_at=published_at,
                available_at=available_at,
                cleanup_start=cleanup_start,
                cleanup_end=cleanup_end,
                cash_settlement_per_share=cash,
                successor_instrument_id=None,
                source_provider="opendart",
                source_url=disclosure_url,
                document_receipt_no=receipt_no,
                document_sha256=source_hash,
            )
        return LifecycleEvidence(
            candidate=candidate,
            resolution_kind=LifecycleResolutionKind.UNSETTLED_DELISTING,
            evidence_status="verified",
            evidence_reason="matched",
            published_at=published_at,
            available_at=available_at,
            cleanup_start=cleanup_start,
            cleanup_end=cleanup_end,
            cash_settlement_per_share=None,
            successor_instrument_id=None,
            source_provider="opendart",
            source_url=disclosure_url,
            document_receipt_no=receipt_no,
            document_sha256=source_hash,
        )
    if "주식교환" in text or "주식이전" in text or "합병" in text:  # pragma: no cover - merger path is live-data only
        return LifecycleEvidence(
            candidate=candidate,
            resolution_kind=LifecycleResolutionKind.UNRESOLVED,
            evidence_status="unresolved",
            evidence_reason="missing_terms",
            published_at=published_at,
            available_at=available_at,
            cleanup_start=None,
            cleanup_end=None,
            cash_settlement_per_share=None,
            successor_instrument_id=None,
            source_provider="opendart",
            source_url=disclosure_url,
            document_receipt_no=receipt_no,
            document_sha256=source_hash,
        )
    if "감자" in text or "자본감소" in text:  # pragma: no cover - capital-reduction path is live-data only
        return LifecycleEvidence(
            candidate=candidate,
            resolution_kind=LifecycleResolutionKind.UNRESOLVED,
            evidence_status="unresolved",
            evidence_reason="missing_terms",
            published_at=published_at,
            available_at=available_at,
            cleanup_start=None,
            cleanup_end=None,
            cash_settlement_per_share=None,
            successor_instrument_id=None,
            source_provider="opendart",
            source_url=disclosure_url,
            document_receipt_no=receipt_no,
            document_sha256=source_hash,
        )
    return LifecycleEvidence(  # pragma: no cover - unresolved fallback is collector-covered
        candidate=candidate,
        resolution_kind=LifecycleResolutionKind.UNRESOLVED,
        evidence_status="unresolved",
        evidence_reason="missing_terms",
        published_at=published_at,
        available_at=available_at,
        cleanup_start=None,
        cleanup_end=None,
        cash_settlement_per_share=None,
        successor_instrument_id=None,
        source_provider="opendart",
        source_url=disclosure_url,
        document_receipt_no=receipt_no,
        document_sha256=source_hash,
    )


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
