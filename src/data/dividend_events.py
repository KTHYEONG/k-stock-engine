"""Silver cash-dividend event input built from dated decision filings."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from calendar import monthrange as calendar_monthrange
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.integrations.dart.dividend_decision import (
    DividendDecision,
    UndecidedRecordDateError,
    parse_dividend_decision,
)

POLICY_VERSION: str = "dividend-events-v2"

DIVIDEND_DECISION_SOURCE: str = "opendart:dividend_decision"

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
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
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
        prev = mapping.get(corp)
        if prev is not None and prev != ticker:
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
        except OSError:  # pragma: no cover - globbed path vanished mid-scan
            continue
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
        if b"<status>014</status>" in base64.b64decode(str(payload["archive_b64"]))[:600]:
            continue  # DART's "file does not exist" body is an absence, not an unreadable filing
        envelopes.append(payload)
    return envelopes


def _decision_from_envelope(payload: dict[str, Any]) -> DividendDecision:
    try:
        rcept_no = str(payload["rcept_no"]).strip()
        corp_code = str(payload["corp_code"]).strip()
        received_on = date.fromisoformat(str(payload["received_on"])[:10])
        archive = base64.b64decode(str(payload["archive_b64"]))
    except (KeyError, ValueError) as exc:
        raise PITDataError("invalid dividend-decision Bronze envelope; certification blocked") from exc
    return parse_dividend_decision(
        archive_bytes=archive, rcept_no=rcept_no, corp_code=corp_code, received_on=received_on
    )


def _session_dates(calendar: SessionCalendar) -> list[date]:
    return [s.astimezone(KRX_TZ).date() for s in calendar.sessions]


def _resolve_ex_session(record_date: date, *, calendar: SessionCalendar) -> date:
    dates = _session_dates(calendar)
    at_or_before = [d for d in dates if d <= record_date]
    if not at_or_before:
        raise PITDataError(f"no certified session on or before record date {record_date.isoformat()}")
    record_session = max(at_or_before)
    idx = dates.index(record_session)
    if idx == 0:
        raise PITDataError(f"no prior certified session before record session {record_session.isoformat()}")
    return dates[idx - 1]


def _add_months(day: date, months: int) -> date:
    year, month_index = divmod(day.year * 12 + day.month - 1 + months, 12)
    month = month_index + 1
    last = calendar_monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def _estimate_pay_date(decision: DividendDecision) -> tuple[date, str] | None:
    """Latest legal payment date when the filing leaves the payment date undecided.

    결산배당은 총회에서 확정되고 상법상 총회일로부터 1개월 이내에 지급해야 한다. 총회일도 미정이면
    정기총회가 결산기 후 3개월 이내에 열린다는 점에 맞춰 기준일 + 4개월을 시한으로 본다.
    중간/분기배당은 이사회 결의일로부터 1개월 이내다. 모두 실제보다 늦은 보수적 시점이며 결정 공시에
    적힌 사실만 쓰므로 미래 정보가 아니다.
    """
    if decision.agm_date is not None:
        return _add_months(decision.agm_date, 1), "agm_plus_1m"
    kind = decision.dividend_kind or ""
    if "결산" in kind:
        return _add_months(decision.record_date, 4), "record_plus_4m"
    if "중간" in kind or "분기" in kind:
        return _add_months(max(decision.received_on, decision.record_date), 1), "decision_plus_1m"
    return None


def _resolve_pay_session(pay_date: date, *, calendar: SessionCalendar) -> date:
    # 호출 전에 기준일 이하 세션이 있음이 확인되고 지급일은 기준일 이후이므로 후보는 항상 존재한다.
    return max(d for d in _session_dates(calendar) if d <= pay_date)


def _resolve_available_at(received_on: date, *, calendar: SessionCalendar) -> datetime:
    for session in calendar.sessions:
        if session.astimezone(KRX_TZ).date() > received_on:
            return session
    raise PITDataError(f"no certified session open after receipt date {received_on.isoformat()}")


def materialize_dividend_events(
    *, bronze_root: Path, universe_root: Path, silver_root: Path, calendar: SessionCalendar
) -> Path:
    """Build ``dividend_events_<hash16>`` from parsed dividend-decision Bronze pages.

    With T+2 settlement, let R be the last session on or before the record
    date; the last cum-dividend session is two sessions before R, so the
    ex-session is the session immediately before R.
    For each (corp, record_date) the latest correction wins; ``available_at`` is
    the next session open after the latest filing's receipt date.

    Args:
        bronze_root: Scope Bronze root holding ``corporate_actions`` decision
            envelopes and the frozen ``dart_corp_codes`` bridge.
        universe_root: Scope Silver root locating the certified universe (kept
            for caller symmetry; session math uses ``calendar``).
        silver_root: Scope Silver root receiving ``dividend_events_<hash16>/``.
        calendar: Certified trading sessions in strictly increasing order.

    Returns:
        The published dataset directory.

    Raises:
        PITDataError: missing bridge, unreadable envelope, unresolvable
            session math, or an existing dataset with different content.
    """
    _ = Path(universe_root)
    bronze_root = Path(bronze_root)
    silver_root = Path(silver_root)
    if len(calendar.sessions) < 2:
        raise PITDataError("dividend events require at least two certified sessions")
    bridge = _load_corp_bridge(bronze_root)
    envelopes = _iter_decision_envelopes(bronze_root)
    latest: dict[tuple[str, date], DividendDecision] = {}
    undated_record_rows = 0
    for envelope in envelopes:
        try:
            decision = _decision_from_envelope(envelope)
        except UndecidedRecordDateError:
            undated_record_rows += 1  # 기준일 미정 공시는 이벤트가 아니라 예고이며, 확정 시 정정/재공시로 온다
            continue
        key = (decision.corp_code, decision.record_date)
        current = latest.get(key)
        if current is None or (decision.received_on, decision.rcept_no) > (current.received_on, current.rcept_no):
            latest[key] = decision
    rows: list[dict[str, Any]] = []
    unpaid_date_rows = 0
    estimated_pay_rows = 0
    invalid_pay_rows = 0
    unmapped_rows = 0
    for (corp_code, _record_date), decision in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1].isoformat())):
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
            # 공시의 연도 오기(전년도 날짜 복사 등)로 지급일이 기준일보다 앞서면 현금 인식 시점을 알 수 없다.
            invalid_pay_rows += 1
            continue
        estimated_pay_rows += pay_source != "declared"
        ex_session = _resolve_ex_session(decision.record_date, calendar=calendar)
        pay_session = _resolve_pay_session(pay_date, calendar=calendar)
        available_at = _resolve_available_at(decision.received_on, calendar=calendar)
        rows.append({
            "instrument_id": f"KRX:{ticker}",
            "ticker": ticker,
            "record_date": decision.record_date,
            "ex_session": ex_session,
            "pay_session": pay_session,
            "dps_krw": decision.dps_common_krw,
            "pay_date_source": pay_source,
            "rcept_no": decision.rcept_no,
            "available_at": available_at,
            "policy_version": POLICY_VERSION,
        })
    frame = (
        pl.DataFrame(rows, schema=_SCHEMA).sort(["ticker", "record_date"])
        if rows
        else pl.DataFrame([], schema=_SCHEMA)
    )
    fingerprint = "\n".join((
        POLICY_VERSION,
        *(
            f"{row['ticker']}|{row['record_date']}|{row['ex_session']}|{row['pay_session']}|"
            f"{row['dps_krw']}|{row['pay_date_source']}|{row['rcept_no']}|{row['available_at'].isoformat()}"
            for row in frame.iter_rows(named=True)
        ),
    ))
    dataset_id = "dividend_events_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    silver_root.mkdir(parents=True, exist_ok=True)
    target = silver_root / dataset_id
    staging = Path(tempfile.mkdtemp(prefix=".dividend-events-", dir=silver_root))
    try:
        out_path = staging / "part-00000.parquet"
        frame.write_parquet(out_path)
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "rows": frame.height,
            "unpaid_date_rows": unpaid_date_rows,
            "estimated_pay_rows": estimated_pay_rows,
            "undated_record_rows": undated_record_rows,
            "invalid_pay_rows": invalid_pay_rows,
            "unmapped_rows": unmapped_rows,
            "decisions": len(latest),
            "partitions": [
                {
                    "path": "part-00000.parquet",
                    "row_count": frame.height,
                    "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                }
            ],
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target.exists():
            try:
                existing = (target / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(f"existing dividend-events dataset is unreadable: {target}") from exc
            if existing != encoded:
                raise PITDataError(f"existing dividend-events dataset differs: {target}")
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target)
    except BaseException:  # pragma: no cover - staging cleanup for unexpected failures
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
