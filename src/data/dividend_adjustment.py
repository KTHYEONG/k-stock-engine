"""Cash-dividend corporate-action evidence: DART alotMatter parsing and factor derivation."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ
from src.data.schemas import BronzeReceipt, PITDataError

CASH_DIVIDEND_PER_SHARE_LABEL = "주당 현금배당금(원)"
COMMON_STOCK_LABEL = "보통주"


@dataclass(frozen=True, slots=True)
class CashDividendRecord:
    corp_code: str
    stock_kind: str
    cash_per_share: float
    stlm_dt: date
    rcept_no: str


def parse_cash_dividend_records(
    raw_records: Sequence[Mapping[str, object]], *, corp_code: str
) -> tuple[CashDividendRecord, ...]:
    """Extract common-stock cash-dividend-per-share facts from raw alotMatter records.

    Args:
        raw_records: Raw DART ``alotMatter.json`` ``list`` entries for one filing.
        corp_code: The 8-digit DART corp code these records belong to.

    Returns:
        One record per common-stock cash-dividend row carrying a positive
        value; empty when no such row is present.

    Raises:
        PITDataError: If a matching row's thstrm/stlm_dt/rcept_no is malformed.
    """
    results: list[CashDividendRecord] = []
    for raw in raw_records:
        if str(raw.get("se") or "").strip() != CASH_DIVIDEND_PER_SHARE_LABEL:
            continue
        if str(raw.get("stock_knd") or "").strip() != COMMON_STOCK_LABEL:
            continue
        thstrm = str(raw.get("thstrm") or "").strip()
        if thstrm in ("", "-"):
            continue
        try:
            cash_per_share = float(thstrm.replace(",", ""))
        except ValueError as exc:
            raise PITDataError(f"malformed alotMatter thstrm value: {thstrm!r}") from exc
        if cash_per_share <= 0:
            continue
        stlm_dt_raw = str(raw.get("stlm_dt") or "").strip()
        try:
            stlm_dt = date.fromisoformat(stlm_dt_raw)
        except ValueError as exc:
            raise PITDataError(f"malformed alotMatter stlm_dt value: {stlm_dt_raw!r}") from exc
        rcept_no = str(raw.get("rcept_no") or "").strip()
        if not re.fullmatch(r"\d{14}", rcept_no):
            raise PITDataError(f"malformed alotMatter rcept_no value: {rcept_no!r}")
        results.append(
            CashDividendRecord(
                corp_code=corp_code,
                stock_kind=COMMON_STOCK_LABEL,
                cash_per_share=cash_per_share,
                stlm_dt=stlm_dt,
                rcept_no=rcept_no,
            )
        )
    return tuple(results)


def filing_date_from_rcept_no(rcept_no: str) -> date:
    """Return the DART filing date encoded in the first 8 digits of rcept_no.

    Raises:
        PITDataError: If rcept_no does not carry a valid 8-digit date prefix.
    """
    digits = re.sub(r"\D", "", str(rcept_no))
    if len(digits) < 8:
        raise PITDataError(f"malformed rcept_no value: {rcept_no!r}")
    try:
        return date(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError as exc:
        raise PITDataError(f"malformed rcept_no value: {rcept_no!r}") from exc


def resolve_ex_dividend_session(stlm_dt: date, *, sessions: tuple[datetime, ...]) -> datetime:
    """Resolve a fiscal settlement date to the KRX ex-dividend trading session.

    The Korean cash-dividend record date is the fiscal period end; the price
    adjustment (ex-dividend) applies on the last certified trading session on
    or before that date.

    Raises:
        PITDataError: If no certified session exists on or before stlm_dt.
    """
    candidates = [session for session in sessions if session.astimezone(KRX_TZ).date() <= stlm_dt]
    if not candidates:
        raise PITDataError(f"no certified session on or before stlm_dt {stlm_dt.isoformat()}")
    return max(candidates)


def compute_cash_dividend_factor(*, close_before_ex: float, cash_per_share: float) -> float:
    """Compute the multiplicative back-adjustment factor for a cash dividend.

    Raises:
        PITDataError: If close_before_ex is non-positive, cash_per_share is
            negative, or the dividend meets/exceeds the pre-ex close (a
            factor <= 0 has no valid economic interpretation).
    """
    if close_before_ex <= 0:
        raise PITDataError(f"close_before_ex must be positive, got {close_before_ex}")
    if cash_per_share < 0:
        raise PITDataError(f"cash_per_share must be non-negative, got {cash_per_share}")
    factor = (close_before_ex - cash_per_share) / close_before_ex
    if factor <= 0:
        raise PITDataError(
            f"cash dividend {cash_per_share} meets/exceeds pre-ex close {close_before_ex}; certification blocked"
        )
    return factor


def build_cash_dividend_corporate_action_records(
    *,
    raw_records: Sequence[Mapping[str, object]],
    corp_code: str,
    instrument_id: str,
    sessions: tuple[datetime, ...],
    daily_market: pl.DataFrame,
) -> list[dict[str, object]]:
    """Build generic corporate-action records ready for normalize_corporate_action_records.

    Args:
        raw_records: Raw DART alotMatter list entries for one corp_code/filing.
        corp_code: The 8-digit DART corp code.
        instrument_id: The certified KRX instrument id (e.g. "KRX:005930").
        sessions: Certified calendar sessions (any order).
        daily_market: Certified daily bars containing at least "session",
            "instrument_id", and "close" for instrument_id.

    Returns:
        One record per parsed dividend, each carrying type="dividend",
        evidence_status="verified", a positive factor, and available_at
        derived strictly from the filing's rcept_no (end of the filing day,
        never the fiscal record date, so no future information leaks).

    Raises:
        PITDataError: If the resolved ex-dividend session has no prior
            certified session, or no daily close exists for instrument_id on
            the session immediately before it.
    """
    parsed = parse_cash_dividend_records(raw_records, corp_code=corp_code)
    if not parsed:
        return []
    ordered_sessions = tuple(sorted(sessions))
    own_bars = (
        daily_market.filter(pl.col("instrument_id") == instrument_id)
        .select("session", "close")
        .sort("session")
    )
    close_by_date = {
        row["session"].astimezone(KRX_TZ).date(): float(row["close"]) for row in own_bars.iter_rows(named=True)
    }
    records: list[dict[str, object]] = []
    for item in parsed:
        ex_session = resolve_ex_dividend_session(item.stlm_dt, sessions=ordered_sessions)
        ex_index = ordered_sessions.index(ex_session)
        if ex_index == 0:
            raise PITDataError(f"no prior certified session before ex-dividend session for {instrument_id!r}")
        prior_date = ordered_sessions[ex_index - 1].astimezone(KRX_TZ).date()
        close_before_ex = close_by_date.get(prior_date)
        if close_before_ex is None:
            raise PITDataError(
                f"missing daily close for {instrument_id!r} on {prior_date.isoformat()}; certification blocked"
            )
        factor = compute_cash_dividend_factor(close_before_ex=close_before_ex, cash_per_share=item.cash_per_share)
        filing_date = filing_date_from_rcept_no(item.rcept_no)
        available_at = datetime.combine(filing_date, time(23, 59, 59), tzinfo=KRX_TZ)
        records.append(
            {
                "instrument_id": instrument_id,
                "effective_date": ex_session,
                "coverage_end": ex_session,
                "action_id": f"dividend:{instrument_id}:{item.stlm_dt.isoformat()}:{item.rcept_no}",
                "type": "dividend",
                "factor": factor,
                "cash_amount": item.cash_per_share,
                "source": "opendart",
                "available_at": available_at,
                "evidence_status": "verified",
                "evidence_reason": None,
            }
        )
    return records


def load_dividend_corporate_action_pages(*, action_receipts: tuple[BronzeReceipt, ...]) -> list[dict[str, Any]]:
    """Read persisted alotMatter dividend-disclosure Bronze payloads.

    Raises:
        PITDataError: If a matching receipt's payload is unreadable or lacks
            a corp_code envelope.
    """
    pages: list[dict[str, Any]] = []
    for receipt in action_receipts:
        if not str(receipt.source_path).startswith("opendart_dividend:"):
            continue
        try:
            payload = json.loads(Path(receipt.payload_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PITDataError("missing dividend-disclosure payload; certification blocked") from exc
        if not isinstance(payload, dict) or "corp_code" not in payload:
            raise PITDataError("invalid dividend-disclosure payload; certification blocked")
        pages.append(payload)
    return pages


def resolve_dividend_corporate_action_records(
    *, pages: Sequence[Mapping[str, Any]], daily_market: pl.DataFrame, sessions: tuple[datetime, ...]
) -> list[dict[str, object]]:
    """Resolve verified cash-dividend corporate-action records from dividend-disclosure pages.

    Args:
        pages: Bronze payloads from load_dividend_corporate_action_pages.
        daily_market: Certified daily bars covering every page's instrument.
        sessions: Certified calendar sessions (any order).

    Raises:
        PITDataError: If a page has no directly-mapped instrument and
            daily_market does not resolve to exactly one instrument.
    """
    records: list[dict[str, object]] = []
    for page in pages:
        corp_code = str(page.get("corp_code") or "")
        if not corp_code:
            continue
        requested = page.get("requested_instrument_id")
        provenance = page.get("instrument_mapping_provenance")
        if isinstance(requested, str) and requested.strip() and provenance == "opendart_corp_code_direct":
            instrument_id = requested.strip()
        else:
            instruments = sorted({str(value) for value in daily_market["instrument_id"].to_list()})
            if len(instruments) != 1:
                raise PITDataError(f"missing OpenDART corp_code mapping for {corp_code!r}")
            instrument_id = instruments[0]
        raw_records = page.get("records", [])
        records.extend(
            build_cash_dividend_corporate_action_records(
                raw_records=raw_records if isinstance(raw_records, list) else [],
                corp_code=corp_code,
                instrument_id=instrument_id,
                sessions=sessions,
                daily_market=daily_market,
            )
        )
    return records
