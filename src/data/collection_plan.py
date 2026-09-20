"""Historical collection plan, checkpoints, and readiness gates."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from src.data.schemas import EvidenceKind, PITDataError
from src.strategy.universe import UniverseDecision

PLAN_ARTIFACT_DIR = Path("data/artifacts/collection-plans")
CHECKPOINT_ARTIFACT_DIR = Path("data/artifacts/collection-checkpoints")

LS_MAX_SESSIONS_PER_REQUEST: Final[int] = 700


def _redact_message(message: str) -> str:
    lowered = message.lower()
    for token in ("app_key", "appkey", "app_secret", "appsecret", "token", "authorization", "bearer", "crtfc_key"):
        if token in lowered:
            raise PITDataError("collection failed; see provider status")
    return message


@dataclass(frozen=True, slots=True)
class PlanChunk:
    chunk_id: str
    symbol: str
    sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class HistoricalCollectionPlan:
    plan_id: str
    coverage_start: date = date(2026, 3, 1)
    coverage_end: date = date(2026, 3, 6)
    chunk_size: int = 1
    chunks: tuple[PlanChunk, ...] = ()
    content_hash: str = ""
    dataset_name: str = ""
    created_at: datetime | None = None


CollectionChunk = PlanChunk


def _canonical_universe(universe: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in universe or ():
        if not isinstance(entry, dict):
            raise PITDataError("universe entry must be a mapping")
        symbol = str(entry.get("symbol") or "").strip()
        if not symbol:
            raise PITDataError("universe entry missing symbol")
        tradable_from = entry.get("tradable_from")
        tradable_to = entry.get("tradable_to")
        items.append(
            {
                "symbol": symbol,
                "is_common_stock": bool(entry.get("is_common_stock", True)),
                "tradable_from": tradable_from.isoformat() if isinstance(tradable_from, date) else None,
                "tradable_to": tradable_to.isoformat() if isinstance(tradable_to, date) else None,
            }
        )
    return sorted(items, key=lambda e: e["symbol"])


def build_historical_collection_plan(
    sessions: Any = (),
    universe: Any = (),
    start: date | None = None,
    end: date | None = None,
    chunk_size: int = 30,
    *,
    artifact_root: Path | str | None = None,
    input_receipt_digest: str | None = None,
    coverage: tuple[date, date] | None = None,
    symbols: Any | None = None,
) -> HistoricalCollectionPlan:
    if coverage is not None:
        start, end = coverage[0], coverage[1]
    if symbols is not None:
        universe = symbols
    if start is None or end is None:
        raise PITDataError("coverage start and end are required")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    raw_sessions = list(sessions or ())
    for session in raw_sessions:
        if not isinstance(session, date):
            raise PITDataError("sessions must contain dates only")
    in_range = sorted(s for s in raw_sessions if start <= s <= end)
    if not in_range:
        raise PITDataError("no sessions inside declared coverage")
    if universe is None or (isinstance(universe, (list, tuple)) and not universe):
        raise PITDataError("universe must list at least one symbol")
    canonical = _canonical_universe(universe)
    common = [entry for entry in universe if isinstance(entry, dict) and entry.get("is_common_stock", True)]
    if not common:
        raise PITDataError("universe must include at least one common stock")
    digest = hashlib.sha256()
    digest.update(str(input_receipt_digest or "legacy-validated").encode("utf-8"))
    digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(canonical, sort_keys=True).encode("utf-8"))
    plan_id = f"plan-{digest.hexdigest()[:16]}"
    content_hash = digest.hexdigest()
    chunks: list[PlanChunk] = []
    for entry in sorted(common, key=lambda e: str(e.get("symbol"))):
        symbol = str(entry.get("symbol")).strip()
        tradable_from = entry.get("tradable_from")
        tradable_to = entry.get("tradable_to")
        eligible = tuple(
            s for s in in_range if (tradable_from is None or s >= tradable_from) and (tradable_to is None or s <= tradable_to)
        )
        for index in range(0, len(eligible), chunk_size):
            window = eligible[index : index + chunk_size]
            if not window:
                continue
            chunk_id = f"{plan_id}:{symbol}:{index // chunk_size:04d}"
            chunks.append(PlanChunk(chunk_id=chunk_id, symbol=symbol, sessions=tuple(window)))
    plan = HistoricalCollectionPlan(
        plan_id=plan_id,
        coverage_start=start,
        coverage_end=end,
        chunk_size=chunk_size,
        chunks=tuple(chunks),
        content_hash=content_hash,
    )
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        receipt_path = root / f"{plan_id}.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "chunk_size": chunk_size,
                    "input_receipt_digest": input_receipt_digest or "legacy-validated",
                    "universe": canonical,
                    "chunks": [
                        {"chunk_id": c.chunk_id, "symbol": c.symbol, "sessions": [s.isoformat() for s in c.sessions]}
                        for c in chunks
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PITDataError(_redact_message(f"plan receipt write failed: {type(exc).__name__}")) from exc
    return plan


def _parse_plan_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError:
            return None
    return None


def _normalize_master_record(record: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(
        record.get("ISU_SRT_CD")
        or record.get("isu_cd")
        or record.get("ISU_CD")
        or record.get("source_identifier")
        or record.get("symbol")
        or record.get("ticker")
        or ""
    ).strip()
    if not symbol:
        return None
    kind_name = record.get("KIND_STKCERT_TP_NM")
    if kind_name is not None:
        is_common = str(kind_name).strip() == "보통주"
    elif "is_common_stock" in record:
        is_common = bool(record.get("is_common_stock"))
    elif record.get("share_class") is not None:
        is_common = str(record.get("share_class")).strip() == "common"
    else:
        is_common = True
    tradable_from = _parse_plan_date(
        record.get("LIST_DD")
        or record.get("tradable_from")
        or record.get("listing_date")
        or record.get("listed_from")
    )
    tradable_to = _parse_plan_date(
        record.get("delisting_date") or record.get("delisted_on") or record.get("tradable_to")
    )
    return {"symbol": symbol, "is_common_stock": is_common, "tradable_from": tradable_from, "tradable_to": tradable_to}


CLASSIFICATION_POLICY_VERSION: Final[str] = "ordinary-share-v1"
_ALLOWED_MASTER_MARKETS: Final[frozenset[str]] = frozenset({"KOSPI", "KOSDAQ"})


def _fast_payload_date(raw: bytes, keys: tuple[str, ...] = ("as_of", "session")) -> date | None:
    """Extract a top-level date field without parsing multi-MB record arrays."""
    import re as _re

    try:
        text = raw[:65536].decode("utf-8", errors="ignore") if len(raw) > 65536 else raw.decode("utf-8", errors="ignore")
    except (OSError, ValueError):
        return None
    for key in keys:
        match = _re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        if match:
            parsed = _parse_plan_date(match.group(1))
            if parsed is not None:
                return parsed
    if len(raw) > 65536:
        try:
            tail = raw[-4096:].decode("utf-8", errors="ignore")
        except (OSError, ValueError):
            return None
        for key in keys:
            match = _re.search(rf'"{key}"\s*:\s*"([^"]+)"', tail)
            if match:
                parsed = _parse_plan_date(match.group(1))
                if parsed is not None:
                    return parsed
    return None


def _master_snapshot_date(payload: dict[str, Any]) -> date | None:
    for key in ("as_of", "session", "BAS_DD", "bas_dd"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        parsed = _parse_plan_date(raw)
        if parsed is not None:
            return parsed
    return None


def _daily_page_session(payload: dict[str, Any]) -> date | None:
    for key in ("session", "BAS_DD", "bas_dd", "as_of"):
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        parsed = _parse_plan_date(raw)
        if parsed is not None:
            return parsed
    records = payload.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("BAS_DD", "bas_dd", "session"):
                raw = record.get(key)
                if raw in (None, ""):
                    continue
                parsed = _parse_plan_date(raw)
                if parsed is not None:
                    return parsed
    return None


def _master_ticker(record: dict[str, Any]) -> str:
    for key in ("ISU_SRT_CD", "isu_srt_cd"):
        raw = str(record.get(key) or "").strip()
        if raw:
            return raw
    for key in ("ISU_CD", "isu_cd"):
        raw = str(record.get(key) or "").strip()
        if raw and not (len(raw) == 12 and raw.upper().startswith("KR")):
            return raw
    for key in ("source_identifier", "symbol", "ticker", "instrument_id"):
        raw = str(record.get(key) or "").strip().removeprefix("KRX:")
        if raw:
            return raw
    return ""


def _daily_ticker(record: dict[str, Any]) -> str:
    for key in ("ISU_SRT_CD", "ticker", "instrument_id"):
        raw = str(record.get(key) or "").strip().removeprefix("KRX:")
        if raw:
            return raw
    raw = str(record.get("ISU_CD") or record.get("isu_cd") or "").strip()
    if raw and not (len(raw) == 12 and raw.upper().startswith("KR")):
        return raw
    for key in ("source_identifier", "symbol"):
        alt = str(record.get(key) or "").strip().removeprefix("KRX:")
        if alt:
            return alt
    return ""


def _is_spac_record(record: dict[str, Any]) -> bool:
    sect = str(record.get("SECT_TP_NM") or record.get("sect_tp_nm") or "")
    if "SPAC" in sect.upper():
        return True
    names = " ".join(
        str(record.get(key) or "")
        for key in ("ISU_NM", "ISU_ABBRV", "ISU_ENG_NM", "isu_nm", "isu_abbrv")
    )
    if "스팩" in names or "기업인수목적" in names:
        return True
    upper = names.upper()
    return "SPECIAL PURPOSE ACQUISITION" in upper or " SPAC" in f" {upper}"


def _classify_master_record_strict(record: dict[str, Any]) -> tuple[str, bool, str]:
    """Classify one authoritative KRX master row.

    Returns (ticker, eligible, reason). Only exact ordinary operating-company
    shares are eligible; every other type carries an explicit reason.
    """
    ticker = _master_ticker(record)
    if not ticker:
        return "", False, "missing-ticker"
    kind = record.get("KIND_STKCERT_TP_NM")
    if kind is None or str(kind).strip() == "":
        return ticker, False, "unknown-kind"
    if str(kind).strip() != "보통주":
        return ticker, False, f"non-ordinary-kind:{str(kind).strip()}"
    secugrp = record.get("SECUGRP_NM")
    if secugrp is None or str(secugrp).strip() == "":
        return ticker, False, "unknown-secugrp"
    if str(secugrp).strip() != "주권":
        return ticker, False, f"non-equity-secugrp:{str(secugrp).strip()}"
    market = str(record.get("MKT_TP_NM") or record.get("market") or "").strip()
    if market == "":
        return ticker, False, "unknown-market"
    if market not in _ALLOWED_MASTER_MARKETS:
        return ticker, False, f"excluded-market:{market}"
    if _is_spac_record(record):
        return ticker, False, "spac"
    return ticker, True, "eligible"


def _parse_daily_volume(record: dict[str, Any]) -> float:
    for key in ("ACC_TRDVOL", "volume", "trdvol", "TRD_QTY", "acc_trdvol"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            raise PITDataError("invalid daily-market volume; certification blocked")
        try:
            parsed = float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError) as exc:
            raise PITDataError("invalid daily-market volume; certification blocked") from exc
        import math as _math

        if not _math.isfinite(parsed) or parsed < 0:
            raise PITDataError("invalid daily-market volume; certification blocked")
        return parsed
    raise PITDataError("daily-market row is missing volume; certification blocked")


def _write_historical_flow_plan_receipt(
    *,
    plan_id: str,
    content_hash: str,
    start: date,
    end: date,
    chunk_size: int,
    input_digest: str,
    input_hashes: list[str],
    chunks: list[PlanChunk],
    requested_symbol_sessions: int,
    non_trading_symbol_sessions: int,
    exclusion_reasons: dict[str, int],
    artifact_root: Path | str | None,
) -> None:
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "chunk_size": chunk_size,
                    "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
                    "input_receipt_digest": input_digest,
                    "input_hashes": sorted(input_hashes),
                    "requested_symbol_sessions": requested_symbol_sessions,
                    "non_trading_symbol_sessions": non_trading_symbol_sessions,
                    "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
                    "chunks": [
                        {"chunk_id": c.chunk_id, "symbol": c.symbol, "sessions": [s.isoformat() for s in c.sessions]}
                        for c in chunks
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PITDataError(_redact_message(f"plan receipt write failed: {type(exc).__name__}")) from exc


def build_historical_collection_plan_from_bronze(  # pragma: no cover
    *,
    bronze_root: Path | str,
    start: date,
    end: date,
    chunk_size: int = LS_MAX_SESSIONS_PER_REQUEST,
    symbols: tuple[str, ...] | None = None,
    artifact_root: Path | str | None = None,
) -> HistoricalCollectionPlan:
    """Build a plan only from the retained calendar and historical master evidence."""
    from src.data.bronze_aggregation import discover_verified_bronze_receipts

    root = Path(bronze_root)
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    wanted = frozenset(symbols or ())
    grouped = discover_verified_bronze_receipts(bronze_root=root)
    calendars: list[tuple[date, Any]] = []
    masters: list[tuple[Any, list[dict[str, Any]]]] = []
    input_hashes: list[str] = []
    if grouped.get(EvidenceKind.CALENDAR) or grouped.get(EvidenceKind.SECURITY_MASTER):
        from src.data.schemas import EvidenceKind as _Kind

        for receipt in grouped.get(_Kind.CALENDAR, ()):
            try:
                payload = json.loads(receipt.payload_path.read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            raw = payload.get("sessions") if isinstance(payload, dict) else None
            if not isinstance(raw, list):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            for value in raw:
                parsed = _parse_plan_date(value)
                if parsed is None:
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                calendars.append((parsed, receipt))
            input_hashes.append(f"calendar:{receipt.content_hash}")
        for receipt in grouped.get(_Kind.SECURITY_MASTER, ()):
            try:
                payload = json.loads(receipt.payload_path.read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            raw_records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(raw_records, list):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            normalized: list[dict[str, Any]] = []
            for record in raw_records:
                if not isinstance(record, dict):
                    continue
                entry = _normalize_master_record(record)
                if entry is not None:
                    normalized.append(entry)
            masters.append((receipt, normalized))
            input_hashes.append(f"security_master:{receipt.content_hash}")
    else:
        try:
            calendar_paths = sorted((root / "calendar").glob("*/payload.json"))
            master_paths = sorted((root / "security_master").glob("*/payload.json"))
            if not calendar_paths or not master_paths:
                raise PITDataError("missing retained calendar and security master Bronze receipts")
            for path in calendar_paths:
                calendar = json.loads(path.read_bytes())
                raw_sessions = calendar.get("sessions") if isinstance(calendar, dict) else None
                if not isinstance(raw_sessions, list):
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                for value in raw_sessions:
                    parsed = _parse_plan_date(value)
                    if parsed is None:
                        raise PITDataError("retained Bronze plan inputs have invalid schema")
                    calendars.append((parsed, None))
            for path in master_paths:
                master = json.loads(path.read_bytes())
                raw_records = master.get("records") if isinstance(master, dict) else None
                if not isinstance(raw_records, list):
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                normalized2: list[dict[str, Any]] = []
                for record in raw_records:
                    if not isinstance(record, dict):
                        continue
                    entry = _normalize_master_record(record)
                    if entry is not None:
                        normalized2.append(entry)
                masters.append((None, normalized2))
            digest_legacy = hashlib.sha256()
            for path in [*calendar_paths, *master_paths]:
                digest_legacy.update(path.read_bytes())
                digest_legacy.update(b"\x00")
            legacy_digest = digest_legacy.hexdigest()
        except (OSError, ValueError) as exc:
            if isinstance(exc, PITDataError):
                raise
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        if not calendars or not masters:
            raise PITDataError("missing retained calendar and security master Bronze receipts")
        all_sessions = sorted({day for day, _ in calendars if start <= day <= end})
        if not all_sessions:
            raise PITDataError("no sessions inside declared coverage")
        universe_legacy: dict[str, dict[str, Any]] = {}
        for _, entries in masters:
            for entry in entries:
                if not entry["is_common_stock"]:
                    continue
                symbol = entry["symbol"]
                if wanted and symbol not in wanted:
                    continue
                universe_legacy[symbol] = entry
        if wanted and set(universe_legacy) != wanted:
            missing = sorted(wanted - set(universe_legacy))
            raise PITDataError(f"requested symbols absent from retained security master: {','.join(missing)}")
        if not universe_legacy:
            raise PITDataError("PIT universe has no eligible symbols")
        return build_historical_collection_plan(
            sessions=tuple(all_sessions),
            universe=tuple(universe_legacy.values()),
            start=start,
            end=end,
            chunk_size=chunk_size,
            artifact_root=artifact_root,
            input_receipt_digest=legacy_digest,
        )
    if not calendars or not masters:
        raise PITDataError("missing retained calendar and security master Bronze receipts")
    # Verified strict path: dated snapshots establish membership; retrieved_at
    # is provenance only and never shortens a historical session.
    from src.data.schemas import EvidenceKind as _Kind

    requested_days = sorted({day for day, _ in calendars if start <= day <= end})
    if not requested_days:
        raise PITDataError("no sessions inside declared coverage")
    calendar_hashes = sorted({f"calendar:{r.content_hash}" for _, r in calendars if r is not None})
    master_receipts = list(grouped.get(_Kind.SECURITY_MASTER, ()))
    daily_receipts = list(grouped.get(_Kind.DAILY_MARKET, ()))
    if not master_receipts or not daily_receipts:
        raise PITDataError("missing retained calendar and security master Bronze receipts")
    # Index master pages once by stated snapshot date; working memory holds
    # one page at a time plus the resulting eligibility index.
    master_by_day: dict[date, Any] = {}
    master_hash_by_day: dict[date, str] = {}
    master_eligible: dict[date, set[str]] = {}
    exclusion_reasons: dict[str, int] = {}
    selected_hashes: list[str] = list(calendar_hashes)
    for receipt in master_receipts:
        try:
            raw = receipt.payload_path.read_bytes()
        except OSError as exc:
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        snap = _fast_payload_date(raw, ("as_of", "session"))
        if snap is None:
            try:
                payload = json.loads(raw)
            except ValueError as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            if not isinstance(payload, dict):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            snap = _master_snapshot_date(payload)
            if snap is None:
                # Undated legacy aggregates cannot supply daily membership alone.
                continue
        if snap < start or snap > end:
            continue
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        if not isinstance(payload, dict):
            raise PITDataError("retained Bronze plan inputs have invalid schema")
        if snap in master_by_day:
            raise PITDataError(f"ambiguous master snapshots for {snap.isoformat()}; certification blocked")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise PITDataError("retained Bronze plan inputs have invalid schema")
        seen: set[str] = set()
        eligible: set[str] = set()
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            ticker, ok, reason = _classify_master_record_strict(record)
            if not ticker:
                raise PITDataError(f"master record is missing instrument identity for {snap.isoformat()}")
            if ticker in seen:
                raise PITDataError(f"duplicate master rows for {(snap.isoformat(), ticker)}")
            seen.add(ticker)
            listed_on = _parse_plan_date(record.get("LIST_DD"))
            if ok and listed_on is not None and snap < listed_on:
                ok, reason = False, "not-listed-yet"
            if ok:
                eligible.add(ticker)
            else:
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        master_by_day[snap] = receipt
        master_hash_by_day[snap] = receipt.content_hash
        master_eligible[snap] = eligible
        selected_hashes.append(f"security_master:{receipt.content_hash}")
    missing_master = [d for d in requested_days if d not in master_by_day]
    if missing_master:
        raise PITDataError(f"missing verified master snapshot for {missing_master[0].isoformat()}")
    # Index daily pages once by session; positive volume gates flow requests.
    daily_by_day: dict[date, Any] = {}
    for receipt in daily_receipts:
        try:
            raw_daily = receipt.payload_path.read_bytes()
        except OSError as exc:
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        sess = _fast_payload_date(raw_daily, ("session", "as_of"))
        if sess is None:
            try:
                payload_daily = json.loads(raw_daily)
            except ValueError as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            if not isinstance(payload_daily, dict):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            sess = _daily_page_session(payload_daily)
        if sess is None:
            raise PITDataError("daily-market page is missing its trading session")
        if sess < start or sess > end:
            continue
        if sess in daily_by_day:
            raise PITDataError(f"duplicate daily-market pages for {sess.isoformat()}")
        daily_by_day[sess] = receipt
    missing_daily = [d for d in requested_days if d not in daily_by_day]
    if missing_daily:
        raise PITDataError(f"missing verified daily-market snapshot for {missing_daily[0].isoformat()}")
    selected_hashes.extend(
        f"daily_market:{daily_by_day[sess].content_hash}" for sess in sorted(daily_by_day)
    )
    eligible_by_symbol: dict[str, list[date]] = {}
    listed_by_symbol: dict[str, list[date]] = {}
    non_trading_cells = 0
    for day in requested_days:
        try:
            payload = json.loads(daily_by_day[day].payload_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise PITDataError(f"daily-market snapshot is empty for {day.isoformat()}")
        seen_daily: set[str] = set()
        volume_by_ticker: dict[str, float] = {}
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            ticker = _daily_ticker(record)
            if not ticker:
                raise PITDataError(f"daily-market record is missing instrument identity for {day.isoformat()}")
            if ticker in seen_daily:
                raise PITDataError(f"duplicate daily-market rows for {(day.isoformat(), ticker)}")
            seen_daily.add(ticker)
            for key in ("BAS_DD", "bas_dd"):
                raw = record.get(key)
                if raw in (None, ""):
                    continue
                parsed = _parse_plan_date(raw)
                if parsed is not None and parsed != day:
                    raise PITDataError(f"conflicting price/master identity for {(day.isoformat(), ticker)}")
            volume_by_ticker[ticker] = _parse_daily_volume(record)
        for ticker in sorted(master_eligible[day]):
            if wanted and ticker not in wanted:
                continue
            if ticker not in volume_by_ticker:
                raise PITDataError(f"conflicting price/master identity for {(day.isoformat(), ticker)}")
            listed_by_symbol.setdefault(ticker, []).append(day)
            if volume_by_ticker[ticker] > 0:
                eligible_by_symbol.setdefault(ticker, []).append(day)
            else:
                non_trading_cells += 1
    if wanted:
        missing_wanted = sorted(s for s in wanted if not listed_by_symbol.get(s))
        if missing_wanted:
            raise PITDataError(f"requested symbols absent from retained security master: {','.join(missing_wanted)}")
    if not eligible_by_symbol:
        if non_trading_cells > 0 or any(
            s in listed_by_symbol for s in wanted
        ):
            digest = hashlib.sha256()
            for token in sorted(set(selected_hashes)):
                digest.update(token.encode("utf-8"))
                digest.update(b"\x00")
            digest.update(start.isoformat().encode("utf-8"))
            digest.update(b"\x00")
            digest.update(end.isoformat().encode("utf-8"))
            digest.update(b"\x00")
            digest.update(CLASSIFICATION_POLICY_VERSION.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(chunk_size).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(",".join(sorted(wanted)).encode("utf-8"))
            content_hash = digest.hexdigest()
            plan_id = f"plan-{content_hash[:16]}"
            _write_historical_flow_plan_receipt(
                plan_id=plan_id,
                content_hash=content_hash,
                start=start,
                end=end,
                chunk_size=chunk_size,
                input_digest=content_hash,
                input_hashes=selected_hashes,
                chunks=[],
                requested_symbol_sessions=0,
                non_trading_symbol_sessions=non_trading_cells,
                exclusion_reasons=exclusion_reasons,
                artifact_root=artifact_root,
            )
            return HistoricalCollectionPlan(
                plan_id=plan_id,
                coverage_start=start,
                coverage_end=end,
                chunk_size=chunk_size,
                chunks=(),
                content_hash=content_hash,
            )
        raise PITDataError("PIT universe has no eligible symbols")
    digest = hashlib.sha256()
    for token in sorted(set(selected_hashes)):
        digest.update(token.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(CLASSIFICATION_POLICY_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(",".join(sorted(wanted)).encode("utf-8"))
    for symbol in sorted(eligible_by_symbol):
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(",".join(d.isoformat() for d in eligible_by_symbol[symbol]).encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    plan_id = f"plan-{content_hash[:16]}"
    chunks: list[PlanChunk] = []
    for symbol in sorted(eligible_by_symbol):
        days = eligible_by_symbol[symbol]
        for index in range(0, len(days), chunk_size):
            window = tuple(days[index : index + chunk_size])
            chunks.append(PlanChunk(chunk_id=f"{plan_id}:{symbol}:{index // chunk_size:04d}", symbol=symbol, sessions=window))
    requested_symbol_sessions = sum(len(v) for v in eligible_by_symbol.values())
    _write_historical_flow_plan_receipt(
        plan_id=plan_id,
        content_hash=content_hash,
        start=start,
        end=end,
        chunk_size=chunk_size,
        input_digest=content_hash,
        input_hashes=selected_hashes,
        chunks=chunks,
        requested_symbol_sessions=requested_symbol_sessions,
        non_trading_symbol_sessions=non_trading_cells,
        exclusion_reasons=exclusion_reasons,
        artifact_root=artifact_root,
    )
    return HistoricalCollectionPlan(
        plan_id=plan_id,
        coverage_start=start,
        coverage_end=end,
        chunk_size=chunk_size,
        chunks=tuple(chunks),
        content_hash=content_hash,
    )


def build_historical_collection_plan_from_universe_decisions(
    *,
    sessions: tuple[date, ...],
    decisions: tuple[UniverseDecision, ...],
    start: date,
    end: date,
    chunk_size: int = 20,
    warmup_sessions: int = 20,
    artifact_root: Path | str | None = None,
    input_receipt_digest: str = "pit-universe",
) -> HistoricalCollectionPlan:
    """Plan KIS requests from PIT eligibility, including only feature warm-up dates."""
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    if not isinstance(warmup_sessions, int) or isinstance(warmup_sessions, bool) or warmup_sessions < 0:
        raise PITDataError("warmup_sessions must be a non-negative integer")
    in_range = tuple(sorted({session for session in sessions if start <= session <= end}))
    if not in_range:
        raise PITDataError("no sessions inside declared coverage")
    index_by_session = {session: index for index, session in enumerate(in_range)}
    eligible_by_symbol: dict[str, list[int]] = {}
    seen: set[tuple[date, str]] = set()
    for decision in decisions:
        decision_date = decision.decision_session.date()
        if decision_date not in index_by_session:
            continue
        key = (decision_date, decision.instrument_id)
        if key in seen:
            raise PITDataError(f"duplicate PIT universe decision: {decision.instrument_id}:{decision_date.isoformat()}")
        seen.add(key)
        if not decision.eligible:
            continue
        eligible_by_symbol.setdefault(decision.instrument_id, []).append(index_by_session[decision_date])
    if not eligible_by_symbol:
        raise PITDataError("PIT universe has no eligible symbols")

    chunks: list[PlanChunk] = []
    digest = hashlib.sha256()
    digest.update(input_receipt_digest.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(warmup_sessions).encode("utf-8"))
    for symbol in sorted(eligible_by_symbol):
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\x00")
        runs: list[list[int]] = []
        for index in sorted(eligible_by_symbol[symbol]):
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        for run in runs:
            required = list(range(max(0, run[0] - warmup_sessions), run[-1] + 1))
            for offset in range(0, len(required), chunk_size):
                members = required[offset : offset + chunk_size]
                chunk_sessions = tuple(in_range[index] for index in members)
                digest.update("|".join(value.isoformat() for value in chunk_sessions).encode("utf-8"))
                digest.update(b"\x00")
                chunks.append(PlanChunk(chunk_id="", symbol=symbol, sessions=chunk_sessions))
    content_hash = digest.hexdigest()
    plan_id = f"plan-{content_hash[:16]}"
    numbered_chunks = tuple(
        PlanChunk(chunk_id=f"{plan_id}:{chunk.symbol}:{index:04d}", symbol=chunk.symbol, sessions=chunk.sessions)
        for index, chunk in enumerate(chunks)
    )
    plan = HistoricalCollectionPlan(plan_id, start, end, chunk_size, numbered_chunks, content_hash)
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "plan_id": plan.plan_id,
                    "content_hash": plan.content_hash,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "chunk_size": chunk_size,
                    "warmup_sessions": warmup_sessions,
                    "input_receipt_digest": input_receipt_digest,
                    "chunks": [
                        {"chunk_id": chunk.chunk_id, "symbol": chunk.symbol, "sessions": [value.isoformat() for value in chunk.sessions]}
                        for chunk in numbered_chunks
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PITDataError(_redact_message(f"plan receipt write failed: {type(exc).__name__}")) from exc
    return plan


class CollectionCheckpointStore:
    """Checkpoint of completed chunks keyed by receipt digest."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _chunk_path(self, plan_id: str, chunk_id: str) -> Path:
        safe_plan = str(plan_id).strip().replace("/", "_") or "plan"
        safe_chunk = str(chunk_id).strip().replace("/", "_") or "chunk"
        return self._root / safe_plan / f"{safe_chunk}.json"

    def mark_complete(
        self,
        *,
        plan_id: str,
        chunk_id: str,
        receipt_digest: str,
        plan_digest: str | None = None,
        receipt_hashes: tuple[str, ...] = (),
    ) -> Path:
        if not str(plan_id).strip() or not str(chunk_id).strip() or not str(receipt_digest).strip():
            raise PITDataError("plan_id, chunk_id, and receipt_digest are required")
        path = self._chunk_path(plan_id, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "chunk_id": chunk_id,
                    "receipt_digest": receipt_digest,
                    "receipt_hashes": list(receipt_hashes),
                    "plan_digest": plan_digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def is_pending(
        self,
        *,
        plan_id: str,
        chunk_id: str,
        receipt_digest: str,
        plan_digest: str | None = None,
        expected_plan_digest: str | None = None,
    ) -> bool:
        path = self._chunk_path(plan_id, chunk_id)
        if not path.exists():
            return True
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        if not isinstance(stored, dict):
            return True
        if stored.get("receipt_digest") != receipt_digest:
            return True
        want_plan = expected_plan_digest if expected_plan_digest is not None else plan_digest
        # Stored checkpoint without plan digest cannot prove same plan.
        return want_plan is not None and stored.get("plan_digest") != want_plan  # noqa: SIM103

    def pending_chunks(
        self,
        plan: HistoricalCollectionPlan,
        receipt_digests: Any,
    ) -> tuple[PlanChunk, ...]:
        digests = dict(receipt_digests or {})
        pending: list[PlanChunk] = []
        for chunk in plan.chunks:
            current = digests.get(chunk.chunk_id)
            if current is None or self.is_pending(
                plan_id=plan.plan_id, chunk_id=chunk.chunk_id, receipt_digest=str(current), plan_digest=plan.content_hash,
                expected_plan_digest=plan.content_hash,
            ):
                pending.append(chunk)
        return tuple(pending)

    def has_verified_receipt(
        self, *, plan: HistoricalCollectionPlan, chunk: PlanChunk, bronze_root: Path | str
    ) -> bool:
        path = self._chunk_path(plan.plan_id, chunk.chunk_id)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            digests = tuple(str(value) for value in stored.get("receipt_hashes", ()) if str(value))
            if not digests:
                digests = (str(stored.get("receipt_digest") or ""),)
            return (
                stored.get("plan_digest") == plan.content_hash
                and bool(digests)
                and all(
                    (payload := Path(bronze_root) / "investor_flow" / digest / "payload.json").exists()
                    and hashlib.sha256(payload.read_bytes()).hexdigest() == digest
                    for digest in digests
                )
            )
        except (OSError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class CollectionReadinessReport:
    certifiable: bool
    unresolved_reasons: tuple[str, ...] = ()
    coverage_gaps: tuple[str, ...] = ()
    pit_lineage_ok: bool = True
    action_provenance_ok: bool = True
    status_provenance_ok: bool = True
    corporate_status_reason: str = ""
    corporate_action_reason: str = ""

    @classmethod
    def incomplete(
        cls,
        corporate_status_reason: str = "",
        corporate_action_reason: str = "",
        coverage_gaps: tuple[str, ...] = (),
        unresolved_reasons: tuple[str, ...] = (),
    ) -> CollectionReadinessReport:
        reasons = list(unresolved_reasons)
        if corporate_status_reason and corporate_status_reason not in reasons:
            reasons.append(corporate_status_reason)
        if corporate_action_reason and corporate_action_reason not in reasons:
            reasons.append(corporate_action_reason)
        for gap in coverage_gaps:
            if gap not in reasons:
                reasons.append(gap)
        if not reasons:
            reasons.append("incomplete collection evidence")
        return cls(
            certifiable=False,
            unresolved_reasons=tuple(reasons),
            coverage_gaps=tuple(coverage_gaps),
            pit_lineage_ok=False,
            action_provenance_ok=not bool(corporate_action_reason),
            status_provenance_ok=not bool(corporate_status_reason),
            corporate_status_reason=corporate_status_reason,
            corporate_action_reason=corporate_action_reason,
        )

    @classmethod
    def certifiable_report(cls) -> CollectionReadinessReport:
        return cls(certifiable=True, unresolved_reasons=())

    def require_certifiable(self) -> CollectionReadinessReport:
        if self.certifiable and not self.unresolved_reasons:
            return self
        detail = "; ".join(self.unresolved_reasons) if self.unresolved_reasons else "unresolved collection gaps"
        raise PITDataError(_redact_message(f"Silver certification blocked: {detail}"))


@dataclass(frozen=True, slots=True)
class CollectionPlanReceipt:
    plan: HistoricalCollectionPlan
    receipt_path: Path


def load_collection_plan(plan_id: str, *, artifact_root: Path | str | None = None) -> HistoricalCollectionPlan:
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    path = root / f"{plan_id}.json"
    if not path.exists():
        raise PITDataError(f"unknown collection plan: {plan_id}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError("collection plan receipt is unreadable") from exc
    chunks = tuple(
        PlanChunk(
            chunk_id=str(item.get("chunk_id")),
            symbol=str(item.get("symbol")),
            sessions=tuple(date.fromisoformat(s) for s in item.get("sessions", ())),
        )
        for item in raw.get("chunks", ())
    )
    return HistoricalCollectionPlan(
        plan_id=str(raw.get("plan_id")),
        coverage_start=date.fromisoformat(str(raw.get("coverage_start"))),
        coverage_end=date.fromisoformat(str(raw.get("coverage_end"))),
        chunk_size=int(raw.get("chunk_size", 0) or 0),
        chunks=chunks,
        content_hash=str(raw.get("content_hash", "")),
    )


__all__ = [
    "CollectionCheckpointStore",
    "CollectionPlanReceipt",
    "CollectionReadinessReport",
    "EvidenceCoverage",
    "HistoricalCollectionPlan",
    "HistoricalCollectionWindow",
    "PlanChunk",
    "audit_historical_readiness",
    "build_historical_collection_plan",
    "build_historical_collection_plan_from_bronze",
    "build_historical_collection_plan_from_universe_decisions",
    "derive_historical_collection_window",
    "load_collection_plan",
]


@dataclass(frozen=True, slots=True)
class HistoricalCollectionWindow:
    history_start: date
    validation_start: date
    validation_end: date
    execution_end: date
    sessions: tuple[date, ...]


def derive_historical_collection_window(
    *,
    sessions: Iterable[date],
    validation_start: date,
    validation_end: date,
    warmup_sessions: int,
) -> HistoricalCollectionWindow:
    ordered = tuple(sessions)
    if not ordered or any(not isinstance(s, date) for s in ordered):
        raise PITDataError("sessions must be a non-empty date tuple")
    if tuple(sorted(ordered)) != ordered:
        raise PITDataError("sessions must be strictly increasing")
    if len(set(ordered)) != len(ordered):
        raise PITDataError("sessions must be strictly increasing")
    if not isinstance(warmup_sessions, int) or isinstance(warmup_sessions, bool) or warmup_sessions < 0:
        raise PITDataError("warmup_sessions must be a non-negative integer")
    if validation_start > validation_end:
        raise PITDataError("validation_start must not be after validation_end")
    try:
        start_idx = ordered.index(validation_start)
        end_idx = ordered.index(validation_end)
    except ValueError as exc:
        raise PITDataError("validation window must be within sessions") from exc
    if start_idx - warmup_sessions < 0:
        raise PITDataError("insufficient warmup sessions before validation_start")
    if end_idx + 1 >= len(ordered):
        raise PITDataError("missing next execution session after validation_end")
    # Never shorten validation; history/execution derive from calendar.
    return HistoricalCollectionWindow(
        history_start=ordered[start_idx - warmup_sessions],
        validation_start=validation_start,
        validation_end=validation_end,
        execution_end=ordered[end_idx + 1],
        sessions=ordered,
    )


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    kind: EvidenceKind
    instrument_id: str | None
    session: date | None
    state: Literal["complete", "source_unavailable", "retryable_failure", "invalid"]
    receipt_hash: str | None
    reason: str


def audit_historical_readiness(
    *,
    plan: HistoricalCollectionPlan,
    coverage: Iterable[EvidenceCoverage],
    usable_feature_count_by_session: Mapping[date, int],
    minimum_cohort: int,
) -> CollectionReadinessReport:
    if not isinstance(minimum_cohort, int) or isinstance(minimum_cohort, bool) or minimum_cohort < 1:
        raise PITDataError("minimum_cohort must be a positive integer")
    items = tuple(coverage)
    for entry in items:
        if entry.state not in ("complete", "source_unavailable", "retryable_failure", "invalid"):
            raise PITDataError(f"unknown coverage state {entry.state!r}")
    gaps: list[str] = []
    unresolved: list[str] = []
    for entry in items:
        if entry.state == "complete":
            continue
        label = f"{entry.kind.value}:{entry.instrument_id or '*'}:{entry.session.isoformat() if entry.session else '*'}:{entry.state}:{entry.reason}"
        gaps.append(label)
        if entry.state in ("retryable_failure", "invalid"):
            unresolved.append(f"global {entry.state} blocks certification: {label}")
    # Global hash/schema/time-order failures block; local gaps need cohort.
    plan_sessions: set[date] = set()
    for chunk in plan.chunks:
        plan_sessions.update(chunk.sessions)
    # The feature map is authoritative for validation sessions.  During
    # warmup, zero rows are expected and must not make an otherwise complete
    # validation cohort fail; an empty map still fails closed against plan.
    check_sessions = set(usable_feature_count_by_session.keys()) or plan_sessions
    for entry in items:
        if entry.session is not None:
            check_sessions.add(entry.session)
    for session in sorted(check_sessions):
        count = int(usable_feature_count_by_session.get(session, 0))
        # Zero-row Gold fails; sub-threshold session fails.
        if count < minimum_cohort:
            unresolved.append(
                f"usable cohort {count} below minimum {minimum_cohort} on {session.isoformat()}"
            )
    if unresolved:
        return CollectionReadinessReport(
            certifiable=False,
            unresolved_reasons=tuple(unresolved),
            coverage_gaps=tuple(gaps),
            pit_lineage_ok=False,
            action_provenance_ok=True,
            status_provenance_ok=True,
        )
    return CollectionReadinessReport(
        certifiable=True,
        unresolved_reasons=(),
        coverage_gaps=tuple(gaps),
        pit_lineage_ok=True,
        action_provenance_ok=True,
        status_provenance_ok=True,
    )
