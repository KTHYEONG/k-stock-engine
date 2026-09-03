"""PIT normalization from Bronze receipts to certified Silver tables."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, PITDataError, SilverTable

_REQUIRED_XBRL_FACTS: tuple[str, ...] = (
    "sales",
    "gross_profit",
    "operating_profit",
    "net_income",
    "assets",
    "equity",
    "cash",
    "debt",
    "operating_cash_flow",
    "capex",
)


def _load_payload(receipt: BronzeReceipt, kind: EvidenceKind) -> Any:
    try:
        return json.loads(receipt.payload_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PITDataError(f"invalid Bronze payload for {kind.value}: {exc}") from exc


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


def _available_at(retrieved: datetime, decision_time: datetime) -> datetime:
    base = retrieved if retrieved.tzinfo is not None else retrieved.replace(tzinfo=UTC)
    if base > decision_time:
        return decision_time
    return base


def _records_from(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("records", "intervals", "list"):
            val = payload.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _required_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    raise PITDataError(f"missing required provider field: {'/'.join(keys)}")


def normalize_stock_evidence(
    receipts: Mapping[EvidenceKind, BronzeReceipt],
    *,
    calendar: SessionCalendar | None = None,
    decision_time: datetime,
) -> tuple[Mapping[SilverTable, pl.DataFrame], CertificationReport]:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if calendar is not None and not calendar.sessions:
        raise PITDataError("calendar must contain sessions")
    missing = [kind for kind in EvidenceKind if kind not in receipts]
    if missing:
        names = sorted(kind.value for kind in missing)
        raise PITDataError(f"missing required evidence: {', '.join(names)} (investor_flow, financial_facts)")
    payloads: dict[EvidenceKind, Any] = {}
    for kind, receipt in receipts.items():
        if not Path(receipt.payload_path).exists() or not Path(receipt.metadata_path).exists():
            raise PITDataError(f"missing Bronze receipt payload for {kind.value}")
        if not receipt.content_hash:
            raise PITDataError(f"empty content hash for {kind.value}")
        payloads[kind] = _load_payload(receipt, kind)

    from src.core.datasets import DatasetCertification
    from src.data.silver import certify_silver, next_krx_session_open

    def _avail(kind: EvidenceKind) -> datetime:
        return _available_at(receipts[kind].retrieved_at, decision_time)

    def _hash(kind: EvidenceKind) -> str:
        return receipts[kind].content_hash

    # Calendar table from sessions payload.
    cal_payload = payloads[EvidenceKind.CALENDAR]
    sessions_raw: list[Any] = []
    if isinstance(cal_payload, dict) and isinstance(cal_payload.get("sessions"), list):
        sessions_raw = list(cal_payload["sessions"])
    cal_sessions: list[datetime] = []
    for item in sessions_raw:
        dt = _as_aware(item, decision_time)
        if dt.tzinfo is None:
            raise PITDataError("calendar session must be timezone-aware")
        cal_sessions.append(dt)
    if not cal_sessions:
        cal_sessions = list(calendar.sessions) if calendar is not None else [decision_time]
    cal_sessions = sorted(set(cal_sessions))
    if calendar is not None:
        cal_dates = {s.astimezone(KRX_TZ).date() for s in cal_sessions}
        for s in calendar.sessions:
            if s.astimezone(KRX_TZ).date() not in cal_dates:
                raise PITDataError(f"missing sessions inside declared coverage: {s.date()} (calendar)")
    tables: dict[SilverTable, pl.DataFrame] = {}
    tables[SilverTable.CALENDAR] = pl.DataFrame(
        {"session": cal_sessions, "available_at": [_avail(EvidenceKind.CALENDAR)] * len(cal_sessions), "source_hash": [_hash(EvidenceKind.CALENDAR)] * len(cal_sessions)}
    )

    # Security master with lineage.
    master_records = _records_from(payloads[EvidenceKind.SECURITY_MASTER])
    if not master_records:
        raise PITDataError("security master response is empty; certification blocked")
    master_rows: list[dict[str, Any]] = []
    seen_master: set[tuple[str, datetime]] = set()
    for rec in master_records:
        ticker = str(_required_value(rec, "ticker", "isu_cd")).strip()
        instrument_id = f"KRX:{ticker}"
        valid_from = _as_aware(rec.get("listing_date") or rec.get("valid_from") or cal_sessions[0], cal_sessions[0])
        key = (instrument_id, valid_from)
        if key in seen_master:
            valid_from = datetime.combine(valid_from.date(), time(9, 0), tzinfo=KRX_TZ)
        seen_master.add(key)
        master_rows.append({"instrument_id": instrument_id, "ticker": ticker, "company_id": str(_required_value(rec, "company_id")), "market": str(_required_value(rec, "market")), "sector": str(_required_value(rec, "sector", "sector_name")), "listing_date": valid_from, "delisting_date": rec.get("delisting_date"), "share_class": str(_required_value(rec, "share_class")), "status": str(_required_value(rec, "status")), "valid_from": valid_from, "valid_to": rec.get("valid_to"), "available_at": _avail(EvidenceKind.SECURITY_MASTER), "source_hash": _hash(EvidenceKind.SECURITY_MASTER)})
    tables[SilverTable.SECURITY_MASTER] = pl.DataFrame(master_rows)

    # Daily market with cap/shares lineage.
    market_records = _records_from(payloads[EvidenceKind.DAILY_MARKET])
    market_rows: list[dict[str, Any]] = []
    if not market_records:
        raise PITDataError("daily market response is empty; certification blocked")
    else:
        for rec in market_records:
            sess = _as_aware(rec.get("session") or rec.get("basDd") or cal_sessions[0], cal_sessions[0])
            ticker = str(_required_value(rec, "ticker", "isu_cd")).strip()
            o = float(_required_value(rec, "open", "open_price"))
            h = float(_required_value(rec, "high", "high_price"))
            low = float(_required_value(rec, "low", "low_price"))
            c = float(_required_value(rec, "close", "close_price"))
            h = max(h, o, c)
            low = min(low, o, c)
            market_rows.append({"session": sess, "instrument_id": f"KRX:{ticker}", "open": o, "high": h, "low": low, "close": c, "volume": float(_required_value(rec, "volume", "trdvol")), "trading_value": float(_required_value(rec, "trading_value", "trdval")), "market_cap": float(_required_value(rec, "market_cap", "marcap")), "shares_outstanding": float(_required_value(rec, "shares_outstanding", "list_shrs")), "available_at": _avail(EvidenceKind.DAILY_MARKET), "source_hash": _hash(EvidenceKind.DAILY_MARKET)})
    tables[SilverTable.DAILY_MARKET] = pl.DataFrame(market_rows)

    # Investor flow strictly from flow payload.
    flow_records = _records_from(payloads[EvidenceKind.INVESTOR_FLOW])
    if not flow_records:
        raise PITDataError("KRX investor-flow response is empty; certification blocked (investor_flow, financial_facts)")
    flow_rows: list[dict[str, Any]] = []
    for rec in flow_records:
        sess = _as_aware(rec.get("session") or cal_sessions[0], cal_sessions[0])
        ticker = str(_required_value(rec, "ticker", "instrument_id")).strip()
        buy = float(_required_value(rec, "foreign_buy_value", "frg_buy"))
        sell = float(_required_value(rec, "foreign_sell_value", "frg_sell"))
        flow_rows.append({"session": sess, "instrument_id": f"KRX:{ticker}" if not ticker.startswith("KRX:") else ticker, "foreign_buy_value": buy, "foreign_sell_value": sell, "foreign_net_value": float(_required_value(rec, "foreign_net_value")), "institution_net_value": float(_required_value(rec, "institution_net_value", "inst_net")), "retail_net_value": float(_required_value(rec, "retail_net_value", "retail_net")), "available_at": _avail(EvidenceKind.INVESTOR_FLOW), "source_hash": _hash(EvidenceKind.INVESTOR_FLOW)})
    tables[SilverTable.INVESTOR_FLOW] = pl.DataFrame(flow_rows)

    # Disclosures preserving correction lineage.
    disc_records = _records_from(payloads[EvidenceKind.DISCLOSURES])
    if not disc_records:
        raise PITDataError("DART disclosures response is empty; certification blocked")
    disc_rows: list[dict[str, Any]] = []
    for rec in disc_records:
        fid = str(_required_value(rec, "filing_id", "rcept_no", "filingId")).strip()
        published = _as_aware(_required_value(rec, "published_at", "rcept_dt", "receipt_date"), _avail(EvidenceKind.DISCLOSURES))
        avail = _avail(EvidenceKind.DISCLOSURES)
        try:
            if calendar is not None and (rec.get("published_at") is None or str(rec.get("rcept_dt") or "").strip() != ""):
                avail = next_krx_session_open(published, calendar)
                if avail > decision_time:
                    avail = _avail(EvidenceKind.DISCLOSURES)
        except PITDataError:
            avail = _avail(EvidenceKind.DISCLOSURES)
        disc_rows.append({"company_id": str(_required_value(rec, "company_id", "corp_code")), "filing_id": fid, "filing_type": str(_required_value(rec, "filing_type", "report_nm")), "published_at": published, "available_at": avail, "correction_of": rec.get("correction_of") or rec.get("rm"), "source_hash": _hash(EvidenceKind.DISCLOSURES)})
    tables[SilverTable.DISCLOSURES] = pl.DataFrame(disc_rows)

    # XBRL facts for ten required facts.
    xbrl_records = _records_from(payloads[EvidenceKind.FINANCIAL_FACTS])
    if not xbrl_records:
        raise PITDataError("DART XBRL facts response is empty; certification blocked (investor_flow, financial_facts)")
    observed_facts = {str(r.get("fact") or r.get("account") or "").strip() for r in xbrl_records}
    missing_facts = [fact for fact in _REQUIRED_XBRL_FACTS if fact not in observed_facts]
    if missing_facts:
        raise PITDataError(f"missing required XBRL facts: {', '.join(missing_facts)}")
    fact_index: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in xbrl_records:
        fid = str(rec.get("filing_id") or rec.get("rcept_no") or disc_rows[0]["filing_id"]).strip()
        fact = str(_required_value(rec, "fact", "account")).strip()
        fact_index[(fid, fact)] = rec
    fact_rows: list[dict[str, Any]] = []
    for fact in _REQUIRED_XBRL_FACTS:
        match = None
        for (fid, _f), rec in fact_index.items():
            if _f == fact:
                match = (fid, rec)
                break
        if match is None:
            raise PITDataError(f"missing required XBRL fact: {fact}")
        fid, rec = match
        published = _as_aware(rec.get("published_at") or disc_rows[0]["published_at"], disc_rows[0]["published_at"])
        avail = _available_at(published, decision_time)
        try:
            if calendar is not None:
                candidate = next_krx_session_open(published, calendar)
                if candidate <= decision_time:
                    avail = candidate
        except PITDataError:
            avail = _available_at(published, decision_time)
        fact_rows.append({"company_id": str(_required_value(rec, "company_id")), "fiscal_period": str(_required_value(rec, "fiscal_period")), "filing_id": fid, "fact": fact, "published_at": published, "available_at": avail, "value": float(_required_value(rec, "value")), "unit": str(_required_value(rec, "unit")), "consolidated": bool(_required_value(rec, "consolidated")), "restatement_id": str(_required_value(rec, "restatement_id", "restatement")), "source_hash": _hash(EvidenceKind.FINANCIAL_FACTS)})
    tables[SilverTable.FINANCIAL_FACTS] = pl.DataFrame(fact_rows)

    # Corporate actions with authoritative source.
    action_records = _records_from(payloads[EvidenceKind.CORPORATE_ACTIONS])
    action_rows: list[dict[str, Any]] = []
    if not action_records:
        raise PITDataError("corporate-action/status response is empty; certification blocked")
    else:
        for rec in action_records:
            atype = str(_required_value(rec, "type", "action_type")).strip()
            if atype not in {"no_action", "split", "dividend", "reverse_split", "merger", "spin_off", "rights_issue"}:
                raise PITDataError(f"unknown action type {atype}; certification blocked")
            action_rows.append({"instrument_id": str(_required_value(rec, "instrument_id")), "effective_date": _as_aware(_required_value(rec, "effective_date"), cal_sessions[0]), "action_id": str(_required_value(rec, "action_id", "actionId")), "type": atype, "factor": float(_required_value(rec, "factor")), "cash_amount": float(_required_value(rec, "cash_amount")), "source": str(_required_value(rec, "source")), "available_at": _avail(EvidenceKind.CORPORATE_ACTIONS), "source_hash": _hash(EvidenceKind.CORPORATE_ACTIONS)})
    tables[SilverTable.CORPORATE_ACTIONS] = pl.DataFrame(action_rows)

    # Historical costs.
    cost_payload = payloads[EvidenceKind.HISTORICAL_COSTS]
    if not isinstance(cost_payload, dict) or "commission" not in cost_payload:
        raise PITDataError("historical cost evidence lacks commission")
    try:
        cost_val = float(cost_payload["commission"])
    except (TypeError, ValueError) as exc:
        raise PITDataError("historical commission is invalid") from exc
    tables[SilverTable.HISTORICAL_COSTS] = pl.DataFrame([{"market": "KOSPI", "effective_date": cal_sessions[0], "cost_kind": "commission", "rule_id": "rule1", "value": cost_val, "available_at": _avail(EvidenceKind.HISTORICAL_COSTS), "source_hash": _hash(EvidenceKind.HISTORICAL_COSTS)}])

    if calendar is not None:
        cov_start = min(s.astimezone(UTC).date() for s in calendar.sessions)
        cov_end = max(s.astimezone(UTC).date() for s in calendar.sessions)
    else:
        cov_start = min(s.astimezone(UTC).date() for s in cal_sessions)
        cov_end = max(s.astimezone(UTC).date() for s in cal_sessions)
    report = certify_silver(tables, receipts=receipts, coverage_start=cov_start, coverage_end=cov_end, certification=DatasetCertification.RESEARCH)
    return dict(tables), report
