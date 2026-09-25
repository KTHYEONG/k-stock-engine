"""Silver cash-dividend event input built from dated decision filings."""
from __future__ import annotations

import base64
import hashlib
import json
from calendar import monthrange as calendar_monthrange
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import DatasetIdentity, DatasetLayer, dataset_digest, publish_dataset
from src.integrations.dart.dividend_decision import (
    DividendDecision,
    UndecidedRecordDateError,
    parse_dividend_decision,
)

POLICY_VERSION = "dividend-events-v2"
DIVIDEND_DECISION_SOURCE = "opendart:dividend_decision"
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


def _load_corp_bridge(bronze_root: Path) -> dict[str, str]:
    paths = sorted((bronze_root / "dart_corp_codes").glob("*/payload.json"))
    if not paths:
        raise PITDataError("dart corp-code bridge is missing; certification blocked")
    try:
        raw = paths[-1].read_bytes()
    except OSError as exc:
        raise PITDataError("invalid dart corp-code bridge payload; certification blocked") from exc
    if hashlib.sha256(raw).hexdigest() != paths[-1].parent.name:
        raise PITDataError("dart corp-code bridge hash mismatch; certification blocked")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError("invalid dart corp-code bridge payload; certification blocked") from exc
    if not isinstance(payload, list):
        raise PITDataError("invalid dart corp-code bridge payload; certification blocked")
    mapping: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip()
        corp = str(row.get("corp_code") or "").strip()
        if not ticker or not corp:
            continue
        previous = mapping.get(corp)
        if previous is not None and previous != ticker:
            raise PITDataError(f"corp code {corp} maps to multiple tickers; certification blocked")
        mapping[corp] = ticker
    if not mapping:
        raise PITDataError("dart corp-code bridge is empty; certification blocked")
    return mapping


def _iter_decision_envelopes(bronze_root: Path) -> list[dict[str, Any]]:
    envelopes: list[dict[str, Any]] = []
    for path in sorted((bronze_root / "corporate_actions").glob("*/payload.json")):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PITDataError(f"dividend-decision Bronze payload is unreadable: {path}") from exc
        expected_hash = path.parent.name
        if (
            len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or hashlib.sha256(raw).hexdigest() != expected_hash
        ):
            raise PITDataError(f"dividend-decision Bronze hash mismatch: {path}")
        if raw.lstrip()[:2] == b"PK":
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if "archive_b64" not in payload or "rcept_no" not in payload:
            continue
        try:
            archive = base64.b64decode(str(payload["archive_b64"]), validate=True)
        except (ValueError, TypeError):
            continue
        if b"<status>014</status>" in archive[:600]:
            continue
        envelopes.append(payload)
    return envelopes


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
    )


def _session_dates(calendar: SessionCalendar) -> list[date]:
    return [session.astimezone(KRX_TZ).date() for session in calendar.sessions]


def _resolve_ex_session(record_date: date, *, calendar: SessionCalendar) -> date:
    dates = _session_dates(calendar)
    at_or_before = [day for day in dates if day <= record_date]
    if not at_or_before:
        raise PITDataError(f"no certified session on or before record date {record_date.isoformat()}")
    record_session = max(at_or_before)
    index = dates.index(record_session)
    if index == 0:
        raise PITDataError(f"no prior certified session before record session {record_session.isoformat()}")
    return dates[index - 1]


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


def _resolve_pay_session(pay_date: date, *, calendar: SessionCalendar) -> date:
    return max(day for day in _session_dates(calendar) if day <= pay_date)


def _resolve_available_at(received_on: date, *, calendar: SessionCalendar) -> datetime:
    for session in calendar.sessions:
        if session.astimezone(KRX_TZ).date() > received_on:
            return session
    raise PITDataError(f"no certified session open after receipt date {received_on.isoformat()}")


def materialize_dividend_events(
    *, bronze_root: Path, universe_root: Path, silver_root: Path, calendar: SessionCalendar
) -> Path:
    """Build and publish ``dividend_events_<hash16>`` from decision filings."""

    _ = Path(universe_root)
    if len(calendar.sessions) < 2:
        raise PITDataError("dividend events require at least two certified sessions")
    bridge_root = Path(bronze_root)
    bridge = _load_corp_bridge(bridge_root)
    bridge_paths = sorted((bridge_root / "dart_corp_codes").glob("*/payload.json"))
    bridge_receipt_hash = bridge_paths[-1].parent.name if bridge_paths else ""
    envelopes = _iter_decision_envelopes(bridge_root)
    envelope_hashes = [
        hashlib.sha256(
            json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        for envelope in envelopes
    ]
    latest: dict[tuple[str, date], DividendDecision] = {}
    undated_record_rows = 0
    for envelope in envelopes:
        try:
            decision = _decision_from_envelope(envelope)
        except UndecidedRecordDateError:
            undated_record_rows += 1
            continue
        key = (decision.corp_code, decision.record_date)
        current = latest.get(key)
        if current is None or (decision.received_on, decision.rcept_no) > (
            current.received_on,
            current.rcept_no,
        ):
            latest[key] = decision

    rows: list[dict[str, Any]] = []
    unpaid_date_rows = 0
    estimated_pay_rows = 0
    invalid_pay_rows = 0
    unmapped_rows = 0
    for (corp_code, _record_date), decision in sorted(
        latest.items(), key=lambda item: (item[0][0], item[0][1].isoformat())
    ):
        ticker = bridge.get(corp_code)
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
        rows.append(
            {
                "instrument_id": f"KRX:{ticker}",
                "ticker": ticker,
                "record_date": decision.record_date,
                "ex_session": _resolve_ex_session(decision.record_date, calendar=calendar),
                "pay_session": _resolve_pay_session(pay_date, calendar=calendar),
                "dps_krw": decision.dps_common_krw,
                "pay_date_source": pay_source,
                "rcept_no": decision.rcept_no,
                "available_at": _resolve_available_at(decision.received_on, calendar=calendar),
                "policy_version": POLICY_VERSION,
            }
        )
    frame = (
        pl.DataFrame(rows, schema=_SCHEMA).sort(["ticker", "record_date"])
        if rows
        else pl.DataFrame([], schema=_SCHEMA)
    )
    identity = DatasetIdentity(
        kind="dividend_events",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "bronze_dividend_decisions": dataset_digest(envelope_hashes),
            "corp_code_bridge": dataset_digest([bridge_receipt_hash]),
        },
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions).encode(
                    "utf-8"
                )
            ).hexdigest()
        },
    )
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions={"part-00000.parquet": frame},
        details={
            "decisions": len(latest),
            "estimated_pay_rows": estimated_pay_rows,
            "undated_record_rows": undated_record_rows,
            "invalid_pay_rows": invalid_pay_rows,
            "unmapped_rows": unmapped_rows,
            "unpaid_date_rows": unpaid_date_rows,
            "corp_code_bridge": bridge_receipt_hash,
        },
    )
    return published.path
