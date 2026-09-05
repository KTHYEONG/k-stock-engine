"""PIT normalization from Bronze receipts to certified Silver tables."""
from __future__ import annotations

import hashlib
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

_DART_MAPPING_VERSION = "dart-fact-map-v1"


def normalize_dart_financial_facts(
    *,
    pages: Any,
    disclosure_rows: Any,
    source_hash: str,
    calendar: SessionCalendar,
    decision_time: datetime,
) -> pl.DataFrame:
    """Normalize every canonical DART record; partial coverage retained."""
    from src.data.silver import next_krx_session_open

    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
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
            fid = str(row.get("filing_id") or row.get("rcept_no") or "").strip()
            if fid and row.get("published_at") is not None:
                disc_published[fid] = row.get("published_at")
    flat: list[Mapping[str, Any]] = []
    for page in page_list:
        if isinstance(page, Mapping) and "records" in page and isinstance(page["records"], list):
            page_kind = str(page.get("source_kind") or "opendart_standard")
            page_version = str(page.get("mapping_version") or _DART_MAPPING_VERSION)
            page_hash = page.get("raw_document_hash")
            raw_identity = page.get("identity")
            page_identity: Mapping[str, Any] = raw_identity if isinstance(raw_identity, Mapping) else {}
            for rec in page["records"]:
                if isinstance(rec, Mapping):
                    merged: dict[str, Any] = dict(rec)
                    merged.setdefault("source_kind", page_kind)
                    merged.setdefault("mapping_version", page_version)
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
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for rec in flat:
        try:
            import re as _re

            raw_company = str(rec.get("company_id") or "").strip()
            raw_ticker = str(rec.get("ticker") or "").strip()
            raw_corp = str(rec.get("corp_code") or rec.get("dart_corp_code") or "").strip()
            fiscal_period = str(rec.get("fiscal_period") or "").strip()
            filing_id = str(rec.get("filing_id") or rec.get("rcept_no") or "").strip()
            fact = str(rec.get("fact") or rec.get("account") or "").strip()
            if not fiscal_period or not filing_id or not fact:
                continue
            ticker_ok = bool(_re.match(r"^\d{6}$", raw_ticker)) if raw_ticker else False
            if raw_ticker and not ticker_ok:
                continue
            if raw_ticker:
                if not raw_corp:
                    continue
                company_id = raw_ticker
                dart_corp_code = raw_corp
                ticker = raw_ticker
            elif raw_company:
                if raw_corp:
                    # A corp_code without the frozen ticker bridge is not joinable PIT evidence.
                    if not _re.match(r"^\d{6}$", raw_company):
                        continue
                    continue
                elif _re.match(r"^\d{6}$", raw_company):
                    company_id = raw_company
                    dart_corp_code = ""
                    ticker = raw_company
                else:
                    company_id = raw_company
                    dart_corp_code = raw_corp
                    ticker = str(rec.get("ticker") or "").strip()
            else:
                continue
            # Fail-closed: ticker-bridged records require six-digit ticker plus corp code.
            if raw_corp and raw_ticker == "" and not raw_company:
                continue
            if not company_id:
                continue
            restatement_id = str(rec.get("restatement_id") or rec.get("restatement") or "r0").strip() or "r0"
            key = (company_id, fiscal_period, filing_id, fact, restatement_id)
            if key in seen:
                continue
            seen.add(key)
            raw_published = rec.get("published_at")
            if raw_published is None and filing_id in disc_published:
                raw_published = disc_published[filing_id]
            published = _as_aware(raw_published, decision_time).astimezone(UTC)
            if published > decision_time:
                continue
            avail = _available_at(published, decision_time).astimezone(UTC)
            try:
                candidate = next_krx_session_open(published, calendar)
                if candidate <= decision_time:
                    avail = candidate.astimezone(UTC)
            except PITDataError:
                avail = _available_at(published, decision_time).astimezone(UTC)
            if avail > decision_time:
                continue
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
            consolidated = rec.get("consolidated")
            if consolidated is None:
                consolidated = True
            consolidated = bool(consolidated)
            source_kind = str(rec.get("source_kind") or "opendart_standard")
            mapping_version = str(rec.get("mapping_version") or _DART_MAPPING_VERSION)
            raw_hash = rec.get("raw_document_hash")
            rows.append(
                {
                    "company_id": company_id,
                    "dart_corp_code": dart_corp_code,
                    "ticker": ticker,
                    "fiscal_period": fiscal_period,
                    "filing_id": filing_id,
                    "fact": fact,
                    "published_at": published,
                    "available_at": avail,
                    "value": value,
                    "unit": unit,
                    "consolidated": consolidated,
                    "restatement_id": restatement_id,
                    "source_hash": source_hash,
                    "source_kind": source_kind,
                    "mapping_version": mapping_version,
                    "raw_document_hash": raw_hash,
                }
            )
        except (PITDataError, ValueError, TypeError):
            continue
    if not rows:
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
    return pl.DataFrame(rows)


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
    streamed_tables: frozenset[SilverTable] = frozenset(),
    streamed_corporate_actions: list[dict[str, Any]] | None = None,
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
        if SilverTable(kind.value) not in streamed_tables:
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
    if SilverTable.SECURITY_MASTER in streamed_tables:
        tables[SilverTable.SECURITY_MASTER] = pl.DataFrame(
            {column: pl.Series([], dtype=pl.String) for column in (
                "instrument_id", "ticker", "company_id", "market", "sector", "listing_date",
                "delisting_date", "share_class", "status", "valid_from", "valid_to", "available_at", "source_hash"
            )}
        )
        master_records = []
    else:
        master_records = _records_from(payloads[EvidenceKind.SECURITY_MASTER])
    if not master_records and SilverTable.SECURITY_MASTER not in streamed_tables:
        raise PITDataError("security master response is empty; certification blocked")
    master_rows: list[dict[str, Any]] = []
    seen_master: set[tuple[str, datetime]] = set()
    for rec in master_records:
        ticker = str(_required_value(rec, "ticker", "isu_cd", "ISU_SRT_CD", "source_identifier")).strip()
        instrument_id = f"KRX:{ticker}"
        valid_from = _as_aware(rec.get("listing_date") or rec.get("listed_from") or rec.get("LIST_DD") or rec.get("valid_from") or cal_sessions[0], cal_sessions[0])
        key = (instrument_id, valid_from)
        if key in seen_master:
            valid_from = datetime.combine(valid_from.date(), time(9, 0), tzinfo=KRX_TZ)
        seen_master.add(key)
        kind_name = str(rec.get("KIND_STKCERT_TP_NM") or "")
        master_rows.append({"instrument_id": instrument_id, "ticker": ticker, "company_id": str(rec.get("company_id") or rec.get("corp_code") or ticker), "market": str(_required_value(rec, "market", "MKT_TP_NM")), "sector": str(rec.get("sector") or rec.get("sector_name") or "__GLOBAL__"), "listing_date": valid_from, "delisting_date": rec.get("delisting_date") or rec.get("delisted_on"), "share_class": str(rec.get("share_class") or ("common" if kind_name == "보통주" or bool(rec.get("is_common_stock")) else "other")), "status": str(rec.get("status") or "listed"), "valid_from": valid_from, "valid_to": rec.get("valid_to"), "available_at": _avail(EvidenceKind.SECURITY_MASTER), "source_hash": _hash(EvidenceKind.SECURITY_MASTER)})
    if SilverTable.SECURITY_MASTER not in streamed_tables:
        tables[SilverTable.SECURITY_MASTER] = pl.DataFrame(master_rows)

    # Daily market with cap/shares lineage.
    if SilverTable.DAILY_MARKET in streamed_tables:
        tables[SilverTable.DAILY_MARKET] = pl.DataFrame(
            {column: pl.Series([], dtype=pl.String) for column in (
                "session", "instrument_id", "open", "high", "low", "close", "volume",
                "trading_value", "market_cap", "shares_outstanding", "available_at", "source_hash"
            )}
        )
        market_records = []
    else:
        market_records = _records_from(payloads[EvidenceKind.DAILY_MARKET])
    market_rows: list[dict[str, Any]] = []
    if not market_records and SilverTable.DAILY_MARKET not in streamed_tables:
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
    if SilverTable.DAILY_MARKET not in streamed_tables:
        tables[SilverTable.DAILY_MARKET] = pl.DataFrame(market_rows)

    # Investor flow strictly from flow payload.
    flow_records = _records_from(payloads[EvidenceKind.INVESTOR_FLOW])
    if not flow_records:
        raise PITDataError("KRX investor-flow response is empty; certification blocked (investor_flow, financial_facts)")
    flow_rows: list[dict[str, Any]] = []
    flow_fingerprints: dict[tuple[datetime, str], str] = {}
    for rec in flow_records:
        sess = _as_aware(rec.get("session") or cal_sessions[0], cal_sessions[0])
        ticker = str(_required_value(rec, "ticker", "instrument_id")).strip()
        buy = float(_required_value(rec, "foreign_buy_value", "frg_buy"))
        sell = float(_required_value(rec, "foreign_sell_value", "frg_sell"))
        instrument_id = f"KRX:{ticker}" if not ticker.startswith("KRX:") else ticker
        row = {
            "session": sess,
            "instrument_id": instrument_id,
            "foreign_buy_value": buy,
            "foreign_sell_value": sell,
            "foreign_net_value": float(_required_value(rec, "foreign_net_value")),
            "institution_net_value": float(_required_value(rec, "institution_net_value", "inst_net")),
            "retail_net_value": float(_required_value(rec, "retail_net_value", "retail_net")),
            "available_at": _avail(EvidenceKind.INVESTOR_FLOW),
            "source_hash": _hash(EvidenceKind.INVESTOR_FLOW),
        }
        flow_key = (sess, instrument_id)
        fingerprint = hashlib.sha256(
            json.dumps({k: v for k, v in row.items() if k not in {"available_at", "source_hash"}}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        previous = flow_fingerprints.get(flow_key)
        if previous == fingerprint:
            continue
        if previous is not None:
            raise PITDataError(f"conflicting investor_flow primary key {flow_key!r}; certification blocked")
        flow_fingerprints[flow_key] = fingerprint
        flow_rows.append(row)
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
    effective_calendar = calendar
    if effective_calendar is None:
        effective_calendar = SessionCalendar(tuple(sorted(set(cal_sessions))))
    tables[SilverTable.FINANCIAL_FACTS] = normalize_dart_financial_facts(
        pages=xbrl_records,
        disclosure_rows=disc_rows,
        source_hash=_hash(EvidenceKind.FINANCIAL_FACTS),
        calendar=effective_calendar,
        decision_time=decision_time,
    )
    if tables[SilverTable.FINANCIAL_FACTS].height == 0:
        raise PITDataError("DART XBRL facts response is empty; certification blocked (investor_flow, financial_facts)")

    # Corporate actions with authoritative source.
    action_records = (
        list(streamed_corporate_actions)
        if streamed_corporate_actions is not None
        else _records_from(payloads[EvidenceKind.CORPORATE_ACTIONS])
    )
    action_rows: list[dict[str, Any]] = []
    if not action_records:
        raise PITDataError("corporate-action/status response is empty; certification blocked")
    else:
        for rec in action_records:
            atype = str(
                rec.get("type") or rec.get("action_type") or rec.get("action_code") or "no_action"
            ).strip()
            if atype == "no_action":
                effective = rec.get("effective_date") or rec.get("session") or cal_sessions[0]
                action_rows.append(
                    {
                        "instrument_id": str(rec.get("instrument_id") or "KRX:__NO_ACTION__"),
                        "effective_date": _as_aware(effective, cal_sessions[0]),
                        "action_id": str(rec.get("action_id") or rec.get("actionId") or "no_action"),
                        "type": atype,
                        "factor": float(rec.get("factor") or rec.get("adjustment_factor") or 1.0),
                        "cash_amount": float(rec.get("cash_amount") or 0.0),
                        "source": str(rec.get("source") or "KRX"),
                        "available_at": _avail(EvidenceKind.CORPORATE_ACTIONS),
                        "source_hash": _hash(EvidenceKind.CORPORATE_ACTIONS),
                    }
                )
                continue
            if atype not in {"no_action", "split", "dividend", "reverse_split", "merger", "spin_off", "rights_issue"}:
                raise PITDataError(f"unknown action type {atype}; certification blocked")
            action_rows.append({"instrument_id": str(_required_value(rec, "instrument_id")), "effective_date": _as_aware(_required_value(rec, "effective_date"), cal_sessions[0]), "action_id": str(_required_value(rec, "action_id", "actionId")), "type": atype, "factor": float(_required_value(rec, "factor")), "cash_amount": float(_required_value(rec, "cash_amount")), "source": str(_required_value(rec, "source")), "available_at": _avail(EvidenceKind.CORPORATE_ACTIONS), "source_hash": _hash(EvidenceKind.CORPORATE_ACTIONS)})
    tables[SilverTable.CORPORATE_ACTIONS] = pl.DataFrame(action_rows)

    # Historical costs.
    cost_payload = payloads[EvidenceKind.HISTORICAL_COSTS]
    if not isinstance(cost_payload, dict) or "commission" not in cost_payload:
        raise PITDataError("historical cost evidence lacks commission")
    raw_commission = cost_payload["commission"]
    if isinstance(raw_commission, list):
        candidates = [item for item in raw_commission if isinstance(item, Mapping)]
        if not candidates:
            raise PITDataError("historical commission is invalid")
        raw_commission = candidates[0].get("buy_rate", candidates[0].get("rate"))
    try:
        cost_val = float(raw_commission)
    except (TypeError, ValueError) as exc:
        raise PITDataError("historical commission is invalid") from exc
    tables[SilverTable.HISTORICAL_COSTS] = pl.DataFrame([{"market": "KOSPI", "effective_date": cal_sessions[0], "cost_kind": "commission", "rule_id": "rule1", "value": cost_val, "available_at": _avail(EvidenceKind.HISTORICAL_COSTS), "source_hash": _hash(EvidenceKind.HISTORICAL_COSTS)}])

    if calendar is not None:
        cov_start = min(s.astimezone(UTC).date() for s in calendar.sessions)
        cov_end = max(s.astimezone(UTC).date() for s in calendar.sessions)
    else:
        cov_start = min(s.astimezone(UTC).date() for s in cal_sessions)
        cov_end = max(s.astimezone(UTC).date() for s in cal_sessions)
    report = certify_silver(
        tables,
        receipts=receipts,
        coverage_start=cov_start,
        coverage_end=cov_end,
        certification=DatasetCertification.RESEARCH,
        decision_time=decision_time,
    )
    return dict(tables), report
