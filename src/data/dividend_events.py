"""Silver cash-dividend event input built from dated decision filings."""
from __future__ import annotations

import base64
import hashlib
import json
from calendar import monthrange as calendar_monthrange
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import polars as pl

from src.config.providers import DividendPlausibilityPolicy
from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_digest,
    dataset_reference,
    publish_dataset,
    read_dataset,
)
from src.data.evidence_sources import DIVIDEND_DECISION_SOURCE
from src.data.parse_cache import ParseCache, source_fingerprint
from src.data.payload_store import read_payload
from src.data.receipt_catalog import BlobEntry, ReceiptCatalog
from src.data.session_index import SessionIndex
from src.integrations.dart.dividend_decision import (
    DividendDecision,
    UndecidedRecordDateError,
    parse_dividend_decision,
)

REVISION = "dividend-events-v3"
_QUARANTINE_FILENAME = "quarantine.json"
_SCHEMA: dict[str, Any] = {
    "instrument_id": pl.String,
    "ticker": pl.String,
    "record_date": pl.Date,
    "ex_session": pl.Date,
    "pay_session": pl.Date,
    "dps_krw": pl.Int64,
    "pay_date_source": pl.String,
    "rcept_no": pl.String,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "policy_version": pl.String,
}


def _load_corp_bridge(catalog: ReceiptCatalog) -> dict[str, str]:
    from src.data.jobs.universe import read_corp_code_bridge

    try:
        mapping, _ = read_corp_code_bridge(catalog)
    except PITDataError as exc:
        raise PITDataError(f"invalid dart corp-code bridge: {exc}") from exc
    return mapping


def _iter_decision_envelopes(catalog: ReceiptCatalog) -> list[dict[str, Any]]:
    envelopes: list[dict[str, Any]] = []
    for blob in catalog.blobs(source=DIVIDEND_DECISION_SOURCE, usable=True):
        payload = _read_decision_envelope(blob)
        if payload is not None:
            envelopes.append(payload)
    return envelopes


def _read_decision_envelope(blob: BlobEntry) -> dict[str, Any] | None:
    path = Path(blob.payload_path)
    try:
        raw = read_payload(path)
    except (OSError, PITDataError) as exc:
        raise PITDataError(f"dividend-decision Bronze payload is unreadable: {path}") from exc
    if hashlib.sha256(raw).hexdigest() != blob.content_hash:
        raise PITDataError(f"dividend-decision Bronze hash mismatch: {path}")
    if raw.lstrip()[:2] == b"PK":
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if "archive_b64" not in payload or "rcept_no" not in payload:
        return None
    try:
        archive = base64.b64decode(str(payload["archive_b64"]), validate=True)
    except (ValueError, TypeError):
        return None
    if b"<status>014</status>" in archive[:600]:
        return None
    return payload


def _decision_from_envelope(payload: dict[str, Any]) -> DividendDecision:
    try:
        rcept_no = str(payload["rcept_no"]).strip()
        corp_code = str(payload["corp_code"]).strip()
        received_on = date.fromisoformat(str(payload["received_on"])[:10])
        archive = base64.b64decode(str(payload["archive_b64"]), validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise PITDataError("invalid dividend-decision Bronze envelope; certification blocked") from exc
    return parse_dividend_decision(
        archive_bytes=archive,
        rcept_no=rcept_no,
        corp_code=corp_code,
        received_on=received_on,
        report_nm=str(payload.get("report_nm", "") or ""),
    )


def _envelope_digest(envelope: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


_PARSE_FINGERPRINT: str | None = None


def parse_fingerprint() -> str:
    """Fingerprint of dividend parsing sources for parse-cache invalidation."""
    global _PARSE_FINGERPRINT
    if _PARSE_FINGERPRINT is None:
        import src.data.dividend_events as _events
        import src.integrations.dart.dividend_decision as _decision_module
        import src.integrations.dart.html_tables as _tables_module

        _PARSE_FINGERPRINT = source_fingerprint(_events, _decision_module, _tables_module)
    return _PARSE_FINGERPRINT


def _encode_decision(decision: DividendDecision) -> dict[str, Any]:
    """Encode one decision into JSON-native form."""
    return {
        "rcept_no": decision.rcept_no,
        "corp_code": decision.corp_code,
        "received_on": decision.received_on.isoformat(),
        "record_date": decision.record_date.isoformat(),
        "pay_date": decision.pay_date.isoformat() if decision.pay_date is not None else None,
        "dps_common_krw": decision.dps_common_krw,
        "is_correction": decision.is_correction,
        "agm_date": decision.agm_date.isoformat() if decision.agm_date is not None else None,
        "dividend_kind": decision.dividend_kind,
        "total_krw": decision.total_krw,
        "market_yield_pct": str(decision.market_yield_pct) if decision.market_yield_pct is not None else None,
    }


def _decode_decision(value: Any) -> DividendDecision:
    """Rebuild a decision from cached JSON."""
    if not isinstance(value, dict):
        raise TypeError("cached decision must be an object")
    rcept_no = value.get("rcept_no")
    corp_code = value.get("corp_code")
    if not isinstance(rcept_no, str) or not isinstance(corp_code, str):
        raise TypeError("cached decision identity is malformed")
    try:
        received_on = date.fromisoformat(str(value.get("received_on")))
        record_date = date.fromisoformat(str(value.get("record_date")))
    except (ValueError, TypeError) as exc:
        raise TypeError("cached decision dates are malformed") from exc
    pay_raw = value.get("pay_date")
    agm_raw = value.get("agm_date")
    try:
        pay_date = None if pay_raw is None else date.fromisoformat(str(pay_raw))
        agm_date = None if agm_raw is None else date.fromisoformat(str(agm_raw))
    except (ValueError, TypeError) as exc:
        raise TypeError("cached decision optional dates are malformed") from exc
    dps = value.get("dps_common_krw")
    if not isinstance(dps, int) or isinstance(dps, bool):
        raise TypeError("cached decision dps is malformed")
    is_correction = value.get("is_correction")
    if not isinstance(is_correction, bool):
        raise TypeError("cached decision correction flag is malformed")
    kind = value.get("dividend_kind")
    if kind is not None and not isinstance(kind, str):
        raise TypeError("cached decision kind is malformed")
    total = value.get("total_krw")
    if total is not None and (not isinstance(total, int) or isinstance(total, bool)):
        raise TypeError("cached decision total is malformed")
    yield_raw = value.get("market_yield_pct")
    market_yield: Decimal | None = None
    if yield_raw is not None:
        if not isinstance(yield_raw, str):
            raise TypeError("cached decision yield is malformed")
        try:
            market_yield = Decimal(yield_raw)
        except InvalidOperation as exc:
            raise TypeError("cached decision yield is malformed") from exc
    return DividendDecision(
        rcept_no=rcept_no,
        corp_code=corp_code,
        received_on=received_on,
        record_date=record_date,
        pay_date=pay_date,
        dps_common_krw=dps,
        is_correction=is_correction,
        agm_date=agm_date,
        dividend_kind=kind,
        total_krw=total,
        market_yield_pct=market_yield,
    )


def _decode_cached_blob(value: Any) -> tuple[str, str | None, DividendDecision | None]:
    """Decode one cached blob entry into (kind, digest, decision)."""
    if not isinstance(value, dict):
        raise TypeError("cached blob must be an object")
    if value.get("skip") is True:
        return ("skip", None, None)
    digest = value.get("digest")
    if not isinstance(digest, str) or not digest:
        raise TypeError("cached blob digest is malformed")
    if value.get("undated") is True:
        return ("undated", digest, None)
    if "decision" not in value:
        raise TypeError("cached blob shape is malformed")
    return ("decision", digest, _decode_decision(value.get("decision")))


def _parse_decision_blob(blob: BlobEntry) -> dict[str, Any]:
    payload = _read_decision_envelope(blob)
    if payload is None:
        return {"skip": True}
    digest = _envelope_digest(payload)
    try:
        decision = _decision_from_envelope(payload)
    except UndecidedRecordDateError:
        return {"digest": digest, "undated": True}
    return {"digest": digest, "decision": _encode_decision(decision)}


def _resolve_decision_blobs(
    catalog: ReceiptCatalog, *, cache: ParseCache | None
) -> tuple[list[DividendDecision], list[str], int]:
    """Per Bronze blob: decision (or undated flag), envelope digest; returns (decisions, envelope_digests, undated_count)."""
    blobs = list(catalog.blobs(source=DIVIDEND_DECISION_SOURCE, usable=True))
    decisions: list[DividendDecision] = []
    digests: list[str] = []
    undated = 0
    cached_raw: dict[str, Any] = {}
    if cache is not None:
        cached_raw = cache.get_many([blob.content_hash for blob in blobs])
    fresh: dict[str, Any] = {}
    for blob in blobs:
        raw_value = cached_raw.get(blob.content_hash) if cache is not None else None
        if raw_value is not None:
            try:
                kind, digest, decision = _decode_cached_blob(raw_value)
            except Exception:
                raw_value = None
            else:
                if kind == "skip":
                    continue
                if kind == "undated" and digest is not None:
                    undated += 1
                    digests.append(digest)
                    continue
                if kind == "decision" and digest is not None and decision is not None:
                    decisions.append(decision)
                    digests.append(digest)
                    continue
        encoded = _parse_decision_blob(blob)
        kind, digest, decision = _decode_cached_blob(encoded)
        if cache is not None:
            fresh[blob.content_hash] = encoded
        if digest is not None:
            digests.append(digest)
        if kind == "undated":
            undated += 1
        if decision is not None:
            decisions.append(decision)
    if cache is not None and fresh:
        cache.put_many(fresh)
    return (decisions, digests, undated)


@dataclass(frozen=True, slots=True)
class DividendQuarantine:
    """A decision withheld from the event table, with the reason it cannot be paid."""

    rcept_no: str
    corp_code: str
    record_date: date
    reason: str


def resolve_dividend_decisions(
    decisions: Sequence[DividendDecision],
    *,
    correction_window_days: int,
) -> tuple[tuple[DividendDecision, ...], tuple[DividendDecision, ...]]:
    """Collapse original and corrected filings into one decision per dividend.

    A correction replaces the latest earlier-received decision of the same company and dividend kind
    whose record date lies within ``correction_window_days`` of the correction's record date, because a
    correction may move the record date itself. Among decisions sharing a record date the latest
    receipt wins, as before.

    Args:
        decisions: Every parsed decision, any order.
        correction_window_days: Maximum record-date distance for a correction to replace a decision.

    Returns:
        ``(kept, superseded)`` ordered by company and record date.
    """
    ordered = sorted(
        decisions, key=lambda item: (item.received_on, item.rcept_no, item.corp_code, item.record_date)
    )
    kept: list[DividendDecision] = []
    superseded: list[DividendDecision] = []
    for decision in ordered:
        if decision.is_correction:
            match: int | None = None
            for index, candidate in enumerate(kept):
                if candidate.corp_code != decision.corp_code:
                    continue
                if (candidate.dividend_kind or "") != (decision.dividend_kind or ""):
                    continue
                if (candidate.received_on, candidate.rcept_no) >= (decision.received_on, decision.rcept_no):
                    continue
                if abs((candidate.record_date - decision.record_date).days) > correction_window_days:
                    continue
                if match is None or (candidate.received_on, candidate.rcept_no) > (
                    kept[match].received_on,
                    kept[match].rcept_no,
                ):
                    match = index
            if match is not None:
                superseded.append(kept.pop(match))
        same = next(
            (
                index
                for index, candidate in enumerate(kept)
                if candidate.corp_code == decision.corp_code and candidate.record_date == decision.record_date
            ),
            None,
        )
        if same is not None:
            superseded.append(kept.pop(same))
        kept.append(decision)
    order = lambda item: (item.corp_code, item.record_date, item.received_on, item.rcept_no)  # noqa: E731
    return tuple(sorted(kept, key=order)), tuple(sorted(superseded, key=order))


def check_dividend_plausibility(
    decision: DividendDecision,
    *,
    close_before_ex: int | None,
    listed_shares: int | None,
    policy: DividendPlausibilityPolicy,
) -> str | None:
    """Return a quarantine reason for a decision that cannot be a real per-share dividend.

    The filing's own numbers are the primary check: the implied yield must stay within
    ``max_yield_ratio`` of the printed market yield; only when no yield is printed must the printed
    total pay at least ``min_paid_share_fraction`` of the listed common shares at the parsed DPS. A hard ceiling on DPS over the
    pre-ex close catches filings that print neither.

    Returns:
        None when plausible, else one of ``dps_exceeds_total``, ``yield_mismatch``,
        ``yield_above_ceiling``, ``no_reference_price``.
    """
    if close_before_ex is None or close_before_ex <= 0:
        return "no_reference_price"
    dps = decision.dps_common_krw
    implied_yield_pct = dps / close_before_ex * 100.0
    printed = decision.market_yield_pct
    if printed is not None and printed > 0:
        # 공시 자체의 시가배당율이 DPS를 확인해 주면 총액 검사는 하지 않는다(차등배당·자사주로 총액이 작을 수 있다).
        ratio = implied_yield_pct / float(printed)
        if ratio > policy.max_yield_ratio or ratio < 1.0 / policy.max_yield_ratio:
            return "yield_mismatch"
    elif (
        decision.total_krw is not None
        and listed_shares is not None
        and listed_shares > 0
        and decision.total_krw < dps * listed_shares * policy.min_paid_share_fraction
    ):
        # 총액을 주당금액으로 읽으면 총액이 지급하는 주식 수가 비정상적으로 적어진다.
        return "dps_exceeds_total"
    if implied_yield_pct > policy.max_yield * 100.0:
        return "yield_above_ceiling"
    return None


def dividend_policy_params(policy: DividendPlausibilityPolicy) -> dict[str, float | int]:
    """Policy values recorded in the dataset identity so retunes mark the dataset stale."""
    return {
        "max_yield": policy.max_yield,
        "max_yield_ratio": policy.max_yield_ratio,
        "min_paid_share_fraction": policy.min_paid_share_fraction,
        "correction_window_days": policy.correction_window_days,
    }


def dividend_dataset_inputs(
    *, bronze_dividend_decisions: str, corp_code_bridge: str, daily_market_id: str | None
) -> dict[str, str]:
    """Dataset inputs for the dividend-events identity, including the daily-market lineage."""
    inputs = {
        "bronze_dividend_decisions": bronze_dividend_decisions,
        "corp_code_bridge": corp_code_bridge,
    }
    if daily_market_id is not None:
        inputs["daily_market"] = dataset_reference(daily_market_id, kind="daily_market")
    return inputs


def _load_market_reference(
    daily_market_path: Path | None,
) -> dict[tuple[str, date], tuple[int | None, int | None]] | None:
    """Map ``(instrument_id, session)`` to ``(close, listed_shares)``.

    Returns None when no daily-market dataset is supplied, in which case the
    plausibility gate is skipped (legacy/test path only; the pipeline always
    supplies the dataset).
    """
    if daily_market_path is None:
        return None
    frame = read_dataset(
        Path(daily_market_path), columns=["session", "instrument_id", "close", "listed_shares"]
    ).collect()
    reference: dict[tuple[str, date], tuple[int | None, int | None]] = {}
    try:
        for row in frame.iter_rows(named=True):
            session = row["session"]
            instrument_id = row["instrument_id"]
            if not isinstance(session, date) or not isinstance(instrument_id, str):
                raise PITDataError(f"invalid daily market reference row for dividend events: {row!r}")
            close = None if row["close"] is None else int(row["close"])
            shares = None if row["listed_shares"] is None else int(row["listed_shares"])
            reference[(instrument_id, session)] = (close, shares)
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid daily market reference for dividend events: {exc}") from exc
    return reference


def _resolve_ex_session(record_date: date, *, index: SessionIndex) -> date:
    n = index.count_on_or_before_date(record_date)
    if n == 0:
        raise PITDataError(f"no certified session on or before record date {record_date.isoformat()}")
    record_session = index.dates[n - 1]
    if n == 1:
        raise PITDataError(f"no prior certified session before record session {record_session.isoformat()}")
    return index.dates[n - 2]


def _add_months(day: date, months: int) -> date:
    year, month_index = divmod(day.year * 12 + day.month - 1 + months, 12)
    month = month_index + 1
    last = calendar_monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def _estimate_pay_date(decision: DividendDecision) -> tuple[date, str] | None:
    if decision.agm_date is not None:
        return _add_months(decision.agm_date, 1), "agm_plus_1m"
    kind = decision.dividend_kind or ""
    if "결산" in kind:
        return _add_months(decision.record_date, 4), "record_plus_4m"
    if "중간" in kind or "분기" in kind:
        return _add_months(max(decision.received_on, decision.record_date), 1), "decision_plus_1m"
    return None


def _resolve_pay_session(pay_date: date, *, index: SessionIndex) -> date:
    n = index.count_on_or_before_date(pay_date)
    if n == 0:
        raise ValueError("max() arg is an empty sequence")
    return index.dates[n - 1]


def _resolve_available_at(received_on: date, *, index: SessionIndex) -> datetime:
    position = index.count_on_or_before_date(received_on)
    if position >= len(index.sessions):
        raise PITDataError(f"no certified session open after receipt date {received_on.isoformat()}")
    return index.sessions[position]


def materialize_dividend_events(
    *,
    catalog: ReceiptCatalog,
    silver_root: Path,
    calendar: SessionCalendar,
    daily_market_path: Path | None = None,
    policy: DividendPlausibilityPolicy | None = None,
    parse_cache: ParseCache | None = None,
) -> Path:
    """Build and publish ``dividend_events_<hash16>`` from decision filings.

    Original and corrected filings collapse to one decision per dividend; each kept decision is
    checked against the Silver daily-market close of the session before ``ex_session`` and the
    listed shares of that session. Implausible decisions are withheld in ``quarantine.json`` next
    to the manifest and never paid. When ``daily_market_path`` is None the plausibility gate is
    skipped (legacy/test path only; the pipeline always supplies the dataset).
    """

    if len(calendar.sessions) < 2:
        raise PITDataError("dividend events require at least two certified sessions")
    active_policy = policy if policy is not None else DividendPlausibilityPolicy()
    bridge = _load_corp_bridge(catalog)
    try:
        from src.data.jobs.universe import read_corp_code_bridge as _read_bridge

        _, bridge_receipt_hash = _read_bridge(catalog)
    except PITDataError:  # pragma: no cover - missing bridge preview fallback
        bridge_receipt_hash = ""
    parsed, envelope_hashes, undated_record_rows = _resolve_decision_blobs(catalog, cache=parse_cache)
    kept, superseded = resolve_dividend_decisions(
        parsed, correction_window_days=active_policy.correction_window_days
    )
    market = _load_market_reference(daily_market_path)
    index = SessionIndex.from_calendar(calendar)

    rows: list[dict[str, Any]] = []
    quarantine: list[DividendQuarantine] = []
    unpaid_date_rows = 0
    estimated_pay_rows = 0
    invalid_pay_rows = 0
    unmapped_rows = 0
    for decision in kept:
        ticker = bridge.get(decision.corp_code)
        if ticker is None:
            unmapped_rows += 1
            continue
        if decision.pay_date is not None:
            pay_date, pay_source = decision.pay_date, "declared"
        else:
            estimate = _estimate_pay_date(decision)
            if estimate is None:
                unpaid_date_rows += 1
                continue
            pay_date, pay_source = estimate
        if pay_date < decision.record_date:
            invalid_pay_rows += 1
            continue
        if pay_source != "declared":
            estimated_pay_rows += 1
        ex_session = _resolve_ex_session(decision.record_date, index=index)
        reason: str | None = None
        if market is not None:
            position = index.position_of_date(ex_session)
            reference_session = index.dates[position - 1] if position > 0 else None
            if reference_session is None:
                close, shares = None, None
            else:
                close, shares = market.get((f"KRX:{ticker}", reference_session), (None, None))
            reason = check_dividend_plausibility(
                decision, close_before_ex=close, listed_shares=shares, policy=active_policy
            )
        if reason is not None:
            quarantine.append(
                DividendQuarantine(
                    rcept_no=decision.rcept_no,
                    corp_code=decision.corp_code,
                    record_date=decision.record_date,
                    reason=reason,
                )
            )
            continue
        rows.append(
            {
                "instrument_id": f"KRX:{ticker}",
                "ticker": ticker,
                "record_date": decision.record_date,
                "ex_session": ex_session,
                "pay_session": _resolve_pay_session(pay_date, index=index),
                "dps_krw": decision.dps_common_krw,
                "pay_date_source": pay_source,
                "rcept_no": decision.rcept_no,
                "available_at": _resolve_available_at(decision.received_on, index=index),
                "policy_version": REVISION,
            }
        )
    frame = (
        pl.DataFrame(rows, schema=_SCHEMA).sort(["ticker", "record_date"])
        if rows
        else pl.DataFrame([], schema=_SCHEMA)
    )
    daily_market_id = Path(daily_market_path).name if daily_market_path is not None else None
    identity = DatasetIdentity(
        kind="dividend_events",
        layer=DatasetLayer.SILVER,
        policy_version=REVISION,
        inputs=dividend_dataset_inputs(
            bronze_dividend_decisions=dataset_digest(envelope_hashes),
            corp_code_bridge=dataset_digest([bridge_receipt_hash]),
            daily_market_id=daily_market_id,
        ),
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions).encode(
                    "utf-8"
                )
            ).hexdigest(),
            **dividend_policy_params(active_policy),
        },
    )
    by_reason: dict[str, int] = {}
    for entry in quarantine:
        by_reason[entry.reason] = by_reason.get(entry.reason, 0) + 1
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions={"part-00000.parquet": frame},
        details={
            "decisions": len(kept),
            "superseded_decisions": len(superseded),
            "estimated_pay_rows": estimated_pay_rows,
            "undated_record_rows": undated_record_rows,
            "invalid_pay_rows": invalid_pay_rows,
            "unmapped_rows": unmapped_rows,
            "unpaid_date_rows": unpaid_date_rows,
            "quarantined_rows": len(quarantine),
            "quarantined_by_reason": dict(sorted(by_reason.items())),
            "corp_code_bridge": bridge_receipt_hash,
        },
    )
    quarantine_payload = [
        {
            "rcept_no": entry.rcept_no,
            "corp_code": entry.corp_code,
            "record_date": entry.record_date.isoformat(),
            "reason": entry.reason,
        }
        for entry in sorted(quarantine, key=lambda item: (item.corp_code, item.record_date, item.rcept_no))
    ]
    (published.path / _QUARANTINE_FILENAME).write_text(
        json.dumps(quarantine_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return published.path
