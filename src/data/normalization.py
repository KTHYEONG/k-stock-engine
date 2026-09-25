"""PIT normalization from Bronze receipts to certified Silver tables."""
from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any, Final

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.schemas import PITDataError

_DART_MAPPING_VERSION = "dart-fact-map-v1"


TRUSTED_FACT_SOURCE_KINDS: Final[frozenset[str]] = frozenset({"opendart_standard", "legacy_document_verified"})
ATTESTED_FACT_SOURCE_KINDS: Final[frozenset[str]] = frozenset({"legacy_document_verified"})


def _has_valid_attestation(record: Mapping[str, Any]) -> bool:
    """Return True only when the record carries a complete attestation mapping."""
    verification = record.get("verification")
    if not isinstance(verification, Mapping):
        return False
    benchmark_id = verification.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        return False
    fact_class = verification.get("fact_class")
    if not isinstance(fact_class, str) or not fact_class.strip():
        return False
    checks = verification.get("checks")
    return isinstance(checks, list) and bool(checks)


@dataclass(frozen=True, slots=True)
class QuarantinedFiling:
    """A filing whose facts were withheld because their source kind is untrusted."""

    company_id: str
    dart_corp_code: str
    fiscal_period: str
    filing_id: str
    source_kind: str
    published_at: datetime  # UTC, midnight KST of the effective receipt date
    available_at: datetime  # UTC, next KRX session open after the effective receipt date


def _flatten_dart_fact_pages(page_list: list[Any]) -> list[Mapping[str, Any]]:
    """Inline page identity into per-record mappings (page ``source_kind`` inherited)."""
    flat: list[Mapping[str, Any]] = []
    for page in page_list:
        if isinstance(page, Mapping) and "records" in page and isinstance(page["records"], list):
            page_kind = str(page.get("source_kind") or "opendart_standard")
            page_version = str(page.get("mapping_version") or _DART_MAPPING_VERSION)
            page_hash = page.get("raw_document_hash")
            raw_identity = page.get("identity")
            page_identity: Mapping[str, Any] = raw_identity if isinstance(raw_identity, Mapping) else {}
            raw_verification = page.get("verification")
            page_verification: Mapping[str, Any] | None = (
                raw_verification if isinstance(raw_verification, Mapping) else None
            )
            for rec in page["records"]:
                if isinstance(rec, Mapping):
                    merged: dict[str, Any] = dict(rec)
                    merged.setdefault("source_kind", page_kind)
                    merged.setdefault("mapping_version", page_version)
                    if page_verification is not None:
                        merged.setdefault("verification", page_verification)
                    if "raw_document_hash" not in merged:
                        merged["raw_document_hash"] = page_hash
                    for k in ("company_id", "filing_id", "fiscal_period", "published_at"):
                        if (not merged.get(k)) and page.get(k) is not None:
                            merged[k] = page[k]
                    page_ticker = str(page.get("ticker") or "").strip()
                    if page_ticker and not merged.get("ticker"):
                        merged["ticker"] = page_ticker
                    if page_ticker and not merged.get("corp_code") and page.get("corp_code"):
                        merged["corp_code"] = page["corp_code"]
                    if (not merged.get("filing_id")) and page_identity.get("filing_id"):
                        merged["filing_id"] = page_identity["filing_id"]
                    if (not merged.get("company_id")) and page_identity.get("corp_code"):
                        merged["company_id"] = page_identity["corp_code"]
                    flat.append(merged)
        elif isinstance(page, Mapping):
            flat.append(page)
    return flat


def _resolve_dart_company(
    rec: Mapping[str, Any],
    ticker_by_corp_code: Mapping[str, str] | None,
    bridge_receipt_hash: str | None,
) -> tuple[str, str, str] | None:
    """Resolve (company_id, dart_corp_code, ticker) with the frozen bridge rules."""
    import re as _re

    raw_company = str(rec.get("company_id") or "").strip()
    raw_ticker = str(rec.get("ticker") or "").strip()
    raw_corp = str(rec.get("corp_code") or rec.get("dart_corp_code") or "").strip()
    ticker_ok = bool(_re.match(r"^\d{6}$", raw_ticker)) if raw_ticker else False
    if raw_ticker and not ticker_ok:
        return None
    if raw_ticker:
        if not raw_corp:
            return None
        return (raw_ticker, raw_corp, raw_ticker)
    if raw_company:
        if _re.match(r"^\d{8}$", raw_company) and not ticker_ok:
            corp_candidate = raw_corp or raw_company
            bridged = ticker_by_corp_code.get(corp_candidate) if ticker_by_corp_code else None
            if bridged is not None and _re.match(r"^\d{6}$", bridged) and bridge_receipt_hash:
                return (bridged, corp_candidate, bridged)
            return None
        if raw_corp:
            bridged = ticker_by_corp_code.get(raw_corp) if ticker_by_corp_code else None
            if bridged is not None and _re.match(r"^\d{6}$", bridged):
                return (bridged, raw_corp, bridged)
            return None
        if _re.match(r"^\d{6}$", raw_company):
            return (raw_company, "", raw_company)
        return (raw_company, raw_corp, str(rec.get("ticker") or "").strip())
    if raw_corp and not raw_company:
        return None
    return None


def _fact_availability(
    *,
    rec: Mapping[str, Any],
    filing_id: str,
    disc_published: Mapping[str, Any],
    session_dates: list[date],
    session_opens: list[datetime],
    decision_time: datetime,
) -> tuple[datetime, datetime] | None:
    """Shared effective-receipt-date and next-session-open rule for rows and quarantine.

    Returns ``(published_at, available_at)`` or ``None`` when the filing is not
    yet observable at ``decision_time`` or its timestamps are unusable.
    Raises ``PITDataError`` when the calendar has no session after an effective
    receipt date that is not after ``decision_time``.
    """
    try:
        raw_published = rec.get("published_at")
        if raw_published is None and filing_id in disc_published:
            raw_published = disc_published[filing_id]
        identity_date = _as_aware(raw_published, decision_time).astimezone(KRX_TZ).date()
        receipt_date = _effective_receipt_date(rec, filing_id)
    except (PITDataError, ValueError, TypeError):
        return None
    effective_date = identity_date if receipt_date is None else max(identity_date, receipt_date)
    published = datetime.combine(effective_date, time(0, 0), tzinfo=KRX_TZ).astimezone(UTC)
    if published > decision_time:
        return None
    avail = _next_session_open_after(effective_date, session_dates, session_opens)
    if avail > decision_time:
        return None
    return (published, avail)


def _empty_dart_fact_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_id": pl.Series([], dtype=pl.String),
            "dart_corp_code": pl.Series([], dtype=pl.String),
            "ticker": pl.Series([], dtype=pl.String),
            "fiscal_period": pl.Series([], dtype=pl.String),
            "filing_id": pl.Series([], dtype=pl.String),
            "fact": pl.Series([], dtype=pl.String),
            "published_at": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "available_at": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "value": pl.Series([], dtype=pl.Float64),
            "unit": pl.Series([], dtype=pl.String),
            "consolidated": pl.Series([], dtype=pl.Boolean),
            "restatement_id": pl.Series([], dtype=pl.String),
            "source_hash": pl.Series([], dtype=pl.String),
            "source_kind": pl.Series([], dtype=pl.String),
            "mapping_version": pl.Series([], dtype=pl.String),
            "raw_document_hash": pl.Series([], dtype=pl.String),
        }
    )


def normalize_dart_financial_facts_with_quarantine(
    *,
    pages: Any,
    disclosure_rows: Any,
    source_hash: str,
    calendar: SessionCalendar,
    decision_time: datetime,
    ticker_by_corp_code: Mapping[str, str] | None = None,
    bridge_receipt_hash: str | None = None,
    trusted_source_kinds: frozenset[str] = TRUSTED_FACT_SOURCE_KINDS,
) -> tuple[pl.DataFrame, tuple[QuarantinedFiling, ...]]:
    """Normalize trusted DART fact records and list the filings withheld as untrusted.

    A page whose ``source_kind`` is not in ``trusted_source_kinds`` (for example a
    value scraped from an HTML filing) contributes no fact rows. It still proves
    that a periodic filing existed and when it became observable, so it is
    returned as a quarantine record with the same effective-receipt-date and
    next-session availability rules as fact rows. Untrusted values are never
    repaired, rescaled, or cross-filled from other filings.

    Args:
        pages: Bronze fact payloads.
        disclosure_rows: Disclosure rows used only to fill a missing filing date.
        source_hash: Lineage hash stamped on every fact row.
        calendar: KRX sessions covering every effective receipt date up to ``decision_time``.
        decision_time: Rows and quarantine records not yet available at this instant are excluded.
        ticker_by_corp_code: Frozen corp-code bridge for pages lacking a ticker.
        bridge_receipt_hash: Receipt hash of the bridge.
        trusted_source_kinds: Source kinds whose values may enter Silver.

    Returns:
        The fact frame (same schema as ``normalize_dart_financial_facts``) and the
        quarantined filings, one per (company, fiscal period, filing), sorted by
        (available_at, company_id, fiscal_period, filing_id).

    Raises:
        PITDataError: same conditions as ``normalize_dart_financial_facts``.
    """
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if not calendar.sessions:
        raise PITDataError("calendar must contain sessions")
    try:
        page_list = list(pages)
    except TypeError as exc:
        raise PITDataError("pages must be iterable") from exc
    try:
        disc_list = list(disclosure_rows) if disclosure_rows is not None else []
    except TypeError:
        disc_list = []
    disc_published: dict[str, Any] = {}
    for row in disc_list:
        if isinstance(row, Mapping):
            raw_available = row.get("available_at")
            if raw_available is not None:
                try:
                    available_at = (
                        raw_available
                        if isinstance(raw_available, datetime)
                        else datetime.fromisoformat(str(raw_available).replace("Z", "+00:00"))
                    )
                except ValueError:
                    continue
                if available_at.tzinfo is None or available_at > decision_time:
                    continue
            fid = str(row.get("filing_id") or row.get("rcept_no") or "").strip()
            if fid and row.get("published_at") is not None:
                disc_published[fid] = row.get("published_at")
    flat = _flatten_dart_fact_pages(page_list)
    ordered_sessions = sorted(calendar.sessions)
    for session in ordered_sessions:
        if not isinstance(session, datetime) or session.tzinfo is None:
            raise PITDataError("calendar session must be timezone-aware")
    session_dates = [s.astimezone(KRX_TZ).date() for s in ordered_sessions]
    session_opens = [s.astimezone(UTC) for s in ordered_sessions]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, bool]] = set()
    trusted_keys: set[tuple[str, str, str]] = set()
    quarantine_best: dict[tuple[str, str, str], QuarantinedFiling] = {}
    for rec in flat:
        fiscal_period = str(rec.get("fiscal_period") or "").strip()
        filing_id = str(rec.get("filing_id") or rec.get("rcept_no") or "").strip()
        if not fiscal_period or not filing_id:
            continue
        resolved = _resolve_dart_company(rec, ticker_by_corp_code, bridge_receipt_hash)
        if resolved is None:
            continue
        company_id, dart_corp_code, ticker = resolved
        source_kind = str(rec.get("source_kind") or "opendart_standard")
        if source_kind in ATTESTED_FACT_SOURCE_KINDS and not _has_valid_attestation(rec):
            trusted = False
        else:
            trusted = source_kind in trusted_source_kinds
        if not trusted:
            timing = _fact_availability(
                rec=rec,
                filing_id=filing_id,
                disc_published=disc_published,
                session_dates=session_dates,
                session_opens=session_opens,
                decision_time=decision_time,
            )
            if timing is None:
                continue
            published, avail = timing
            filing_key = (company_id, fiscal_period, filing_id)
            candidate = QuarantinedFiling(
                company_id=company_id,
                dart_corp_code=dart_corp_code,
                fiscal_period=fiscal_period,
                filing_id=filing_id,
                source_kind=source_kind,
                published_at=published,
                available_at=avail,
            )
            previous = quarantine_best.get(filing_key)
            if previous is None or (
                candidate.available_at,
                candidate.published_at,
                candidate.source_kind,
            ) > (
                previous.available_at,
                previous.published_at,
                previous.source_kind,
            ):
                quarantine_best[filing_key] = candidate
            continue
        try:
            fact = str(rec.get("fact") or rec.get("account") or "").strip()
            if not fact:
                continue
            restatement_id = str(rec.get("restatement_id") or rec.get("restatement") or "r0").strip() or "r0"
            consolidated = rec.get("consolidated")
            if consolidated is None:
                consolidated = True
            consolidated = bool(consolidated)
            key = (company_id, fiscal_period, filing_id, fact, restatement_id, consolidated)
            if key in seen:
                continue
            seen.add(key)
            mapping_version = str(rec.get("mapping_version") or _DART_MAPPING_VERSION)
            if dart_corp_code and ticker_by_corp_code and bridge_receipt_hash and ticker_by_corp_code.get(dart_corp_code) == ticker:
                mapping_version = f"{mapping_version}+bridge:{bridge_receipt_hash}"
            raw_hash = rec.get("raw_document_hash")
        except (PITDataError, ValueError, TypeError):
            continue
        timing = _fact_availability(
            rec=rec,
            filing_id=filing_id,
            disc_published=disc_published,
            session_dates=session_dates,
            session_opens=session_opens,
            decision_time=decision_time,
        )
        if timing is None:
            continue
        published, avail = timing
        pending: dict[str, Any] = {
            "company_id": company_id,
            "dart_corp_code": dart_corp_code,
            "ticker": ticker,
            "fiscal_period": fiscal_period,
            "filing_id": filing_id,
            "fact": fact,
            "published_at": published,
            "consolidated": consolidated,
            "restatement_id": restatement_id,
            "source_hash": source_hash,
            "source_kind": source_kind,
            "mapping_version": mapping_version,
            "raw_document_hash": raw_hash,
        }
        try:
            raw_value = rec.get("value")
            if raw_value is None:
                continue
            value = float(raw_value)
            import math as _math

            if not _math.isfinite(value):
                continue
            unit = str(rec.get("unit") or "").strip()
            if not unit:
                continue
            rows.append({**pending, "available_at": avail, "value": value, "unit": unit})
            trusted_keys.add((company_id, fiscal_period, filing_id))
        except (ValueError, TypeError):
            continue
    quarantined = tuple(
        record
        for filing_key, record in sorted(
            quarantine_best.items(),
            key=lambda item: (
                item[1].available_at,
                item[1].company_id,
                item[1].fiscal_period,
                item[1].filing_id,
            ),
        )
        if filing_key not in trusted_keys
    )
    if not rows:
        return _empty_dart_fact_frame(), quarantined
    return pl.DataFrame(rows), quarantined


def _as_aware(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date) and not isinstance(value, datetime):
        dt = datetime.combine(value, time(9, 0), tzinfo=KRX_TZ)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = parsed if isinstance(parsed, datetime) else datetime.combine(parsed, time(9, 0), tzinfo=KRX_TZ)
        except ValueError:
            try:
                d = date.fromisoformat(text[:10])
                dt = datetime.combine(d, time(9, 0), tzinfo=KRX_TZ)
            except ValueError:
                raise PITDataError(f"invalid provider timestamp: {value!r}") from None
    else:
        raise PITDataError(f"missing provider timestamp: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KRX_TZ)
    return dt


def _receipt_source_date(value: str) -> date:
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise PITDataError(f"invalid receipt date in {value!r}") from exc


def _effective_receipt_date(record: Mapping[str, Any], filing_id: str) -> date | None:
    raw = record.get("rcept_no")
    if raw is not None and str(raw).strip() != "":
        candidate = str(raw).strip()
        if re.fullmatch(r"\d{14}", candidate) is None:
            raise PITDataError(f"malformed receipt number: {candidate!r}")
        return _receipt_source_date(candidate)
    candidate = str(filing_id).strip()
    if re.fullmatch(r"\d{14}", candidate) is None:
        return None
    return _receipt_source_date(candidate)


def _next_session_open_after(
    effective: date, session_dates: list[date], session_opens: list[datetime]
) -> datetime:
    idx = bisect_right(session_dates, effective)
    if idx >= len(session_opens):
        raise PITDataError(f"no next KRX session after {effective}")
    return session_opens[idx]


def _required_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    raise PITDataError(f"missing required provider field: {'/'.join(keys)}")


def normalize_corporate_action_records(*, action_records: Sequence[Mapping[str, Any]], calendar_sessions: tuple[datetime, ...], corporate_action_available_at: datetime, corporate_action_source_hash: str) -> pl.DataFrame:
    fallback_session = calendar_sessions[0]
    rows: list[dict[str, Any]] = []
    for record in action_records:
        rec = dict(record)
        atype = str(rec.get("type") or rec.get("action_type") or rec.get("action_code") or "no_action").strip()
        if (atype == "bonus_issue" and (rec.get("share_listing_date") is None or rec.get("share_delta") is None)) or atype not in {"no_action", "split", "dividend", "reverse_split", "merger", "spin_off", "rights_issue", "bonus_issue", "unresolved"}:
            raise PITDataError(f"unsupported legacy corporate action {atype!r} requires rebuild from raw OpenDART Bronze; certification blocked")
        if atype == "no_action" or "evidence_status" not in rec or "evidence_reason" not in rec:
            raise PITDataError(f"legacy {atype!r} corporate-action evidence status missing; requires rebuild from raw OpenDART Bronze")
        status = str(rec.get("evidence_status") or "").strip()
        reason = rec.get("evidence_reason")
        if status not in {"verified", "unresolved"}:
            raise PITDataError(f"invalid corporate-action evidence status {status!r}; requires rebuild from raw OpenDART Bronze")
        if status == "verified":
            if reason not in (None, ""):
                raise PITDataError(f"verified corporate-action evidence reason must be null for {rec.get('action_id')!r}; certification blocked")
        elif not isinstance(reason, str) or not reason.strip():
            raise PITDataError(f"unresolved corporate-action evidence reason missing for {rec.get('action_id')!r}; requires rebuild from raw OpenDART Bronze")
        effective = _as_aware(
            rec.get("effective_date") or rec.get("effective_session") or rec.get("session") or fallback_session,
            fallback_session,
        ).astimezone(KRX_TZ)
        raw_listing = rec.get("share_listing_date")
        listing = (
            _as_aware(raw_listing, fallback_session).astimezone(KRX_TZ)
            if raw_listing is not None
            else None
        )
        rows.append({"instrument_id": str(_required_value(rec, "instrument_id")), "effective_date": effective, "coverage_end": _as_aware(rec.get("coverage_end") or effective, fallback_session).astimezone(KRX_TZ), "action_id": str(rec.get("action_id") or rec.get("actionId") or "no_action"), "type": atype, "factor": float(rec.get("factor") or rec.get("adjustment_factor") or 1.0), "cash_amount": float(rec.get("cash_amount") or 0.0), "source": str(rec.get("source") or "KRX"), "share_listing_date": listing, "share_delta": rec.get("share_delta"), "available_at": _as_aware(rec.get("available_at") or corporate_action_available_at, fallback_session).astimezone(UTC), "source_hash": corporate_action_source_hash, "evidence_status": status, "evidence_reason": reason})
    # Corporate-action cache rows may first carry null share deltas and only
    # expose integer values after the default inference sample.  Infer from
    # the full bounded action payload so a verified cache remains replayable.
    return pl.DataFrame(rows, infer_schema_length=None).unique(
        subset=["instrument_id", "effective_date", "action_id"], maintain_order=True
    )
