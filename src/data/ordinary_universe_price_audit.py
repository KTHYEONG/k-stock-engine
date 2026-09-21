"""Point-in-time raw-price availability audit for the ordinary-share universe."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.data.schemas import PITDataError

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrdinaryUniversePriceAudit:
    """Certified raw-price availability counts for eligible ordinary-share cells."""

    dataset_id: str
    universe_dataset_id: str
    sessions: int
    universe_rows: int
    eligible_rows: int
    price_rows: int
    tradable_rows: int
    missing_price_rows: int
    invalid_price_rows: int
    zero_volume_rows: int
    report_hash: str


def _parse_date(value: Any) -> date:
    raw = str(value).strip()
    try:
        return date.fromisoformat(raw[:10]) if "-" in raw else datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError as exc:
        raise PITDataError(f"invalid daily-market session: {raw!r}") from exc


def _only_universe_dataset(root: Path) -> Path:
    datasets = sorted(path for path in Path(root).glob("ordinary_universe_*") if path.is_dir())
    if len(datasets) != 1:
        raise PITDataError("ordinary universe requires exactly one published dataset")
    return datasets[0]


def _load_universe_manifest(dataset: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        parts = manifest["partitions"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError("invalid ordinary-universe manifest") from exc
    if manifest.get("dataset_id") != dataset.name or not isinstance(parts, list) or not parts:
        raise PITDataError("invalid ordinary-universe manifest")
    previous: date | None = None
    for part in parts:
        if not isinstance(part, dict):
            raise PITDataError("invalid ordinary-universe partition")
        session = _parse_date(part.get("session"))
        if previous is not None and session <= previous:
            raise PITDataError("ordinary-universe partitions are not strictly ordered")
        previous = session
    return parts


def _daily_payloads_by_session(bronze_root: Path, sessions: set[date]) -> dict[date, list[Path]]:
    found: dict[date, list[Path]] = defaultdict(list)
    for receipt_path in (Path(bronze_root) / "daily_market").glob("*/receipt.json"):
        payload_path = receipt_path.parent / "payload.json"
        try:
            raw = payload_path.read_bytes()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise PITDataError(f"invalid daily-market Bronze receipt: {receipt_path}") from exc
        if hashlib.sha256(raw).hexdigest() != receipt.get("content_hash"):
            raise PITDataError(f"daily-market Bronze hash mismatch: {receipt_path}")
        if not isinstance(payload, dict):
            raise PITDataError(f"invalid daily-market payload: {payload_path}")
        raw_session = payload.get("session")
        records = payload.get("records")
        if raw_session in (None, "") and isinstance(records, list) and records:
            first = payload["records"][0]
            raw_session = first.get("BAS_DD") if isinstance(first, dict) else None
        if raw_session in (None, ""):
            continue
        session = _parse_date(raw_session)
        if session in sessions:
            if not isinstance(records, list):
                raise PITDataError(f"invalid daily-market payload: {payload_path}")
            found[session].append(payload_path)
    return found


def _ticker(record: dict[str, Any]) -> str:
    for key in ("ISU_CD", "ISU_SRT_CD", "ticker", "instrument_id"):
        raw = record.get(key)
        if raw not in (None, ""):
            return str(raw).removeprefix("KRX:").strip()
    raise PITDataError("daily-market record lacks ticker")


def _finite_number(record: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if record.get(key) not in (None, ""):
            try:
                number = float(str(record[key]).replace(",", ""))
            except (TypeError, ValueError) as exc:
                raise PITDataError(f"invalid daily-market numeric field: {key}") from exc
            if not math.isfinite(number):
                raise PITDataError(f"invalid daily-market numeric field: {key}")
            return number
    raise PITDataError(f"missing daily-market numeric field: {keys[0]}")


def _price_state(record: dict[str, Any]) -> str:
    prices = tuple(_finite_number(record, *keys) for keys in (
        ("TDD_OPNPRC", "open"), ("TDD_HGPRC", "high"), ("TDD_LWPRC", "low"), ("TDD_CLSPRC", "close"),
    ))
    volume = _finite_number(record, "ACC_TRDVOL", "volume")
    trading_value = _finite_number(record, "ACC_TRDVAL", "trading_value")
    if min(prices) <= 0 or trading_value < 0:
        return "invalid"
    return "tradable" if volume > 0 else "zero_volume"


def audit_ordinary_universe_price_availability(
    *, universe_root: Path, bronze_root: Path, artifact_root: Path
) -> OrdinaryUniversePriceAudit:
    """Certify raw daily-price availability for every eligible ordinary-share cell.

    The audit joins one verified daily-market Bronze page to each dated
    ordinary-universe Silver partition. It records observation state only and
    never fills prices, suspensions, or strategy liquidity thresholds.
    """
    dataset = _only_universe_dataset(Path(universe_root))
    parts = _load_universe_manifest(dataset)
    sessions = {_parse_date(part["session"]) for part in parts}
    daily_pages = _daily_payloads_by_session(Path(bronze_root), sessions)
    counts = {
        "universe_rows": 0, "eligible_rows": 0, "price_rows": 0,
        "tradable_rows": 0, "missing_price_rows": 0,
        "invalid_price_rows": 0, "zero_volume_rows": 0,
    }
    provenance: list[str] = []
    for position, part in enumerate(parts, start=1):
        session = _parse_date(part["session"])
        path = dataset / str(part["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != part.get("parquet_sha256"):
            raise PITDataError(f"ordinary-universe partition hash mismatch: {session}")
        frame = pl.read_parquet(path)
        if frame.height != part.get("row_count") or frame.filter(pl.col("eligible")).height != part.get("eligible_count"):
            raise PITDataError(f"ordinary-universe partition count mismatch: {session}")
        counts["universe_rows"] += frame.height
        eligible = frame.filter(pl.col("eligible"))
        counts["eligible_rows"] += eligible.height
        pages = daily_pages.get(session, [])
        if len(pages) > 1:
            raise PITDataError(f"ambiguous daily-market page: {session}")
        records_by_ticker: dict[str, dict[str, Any]] = {}
        if pages:
            payload = json.loads(pages[0].read_text(encoding="utf-8"))
            for record in payload["records"]:
                if not isinstance(record, dict):
                    raise PITDataError(f"invalid daily-market record: {session}")
                ticker = _ticker(record)
                if ticker in records_by_ticker:
                    raise PITDataError(f"duplicate daily-market ticker: {(session, ticker)!r}")
                bar_day = record.get("BAS_DD", record.get("session"))
                if bar_day not in (None, "") and _parse_date(bar_day) != session:
                    raise PITDataError(f"conflicting daily-market session: {(session, ticker)!r}")
                records_by_ticker[ticker] = record
            provenance.append(f"{session}:{hashlib.sha256(pages[0].read_bytes()).hexdigest()}")
        for ticker in eligible["ticker"].to_list():
            record = records_by_ticker.get(ticker)
            if record is None:
                counts["missing_price_rows"] += 1
                continue
            counts["price_rows"] += 1
            state = _price_state(record)
            counts[{"tradable": "tradable_rows", "invalid": "invalid_price_rows", "zero_volume": "zero_volume_rows"}[state]] += 1
        if position % 100 == 0:  # pragma: no cover - operational heartbeat
            _LOG.info("ordinary-universe-price-audit sessions=%s/%s", position, len(parts))
    identity = "\n".join([dataset.name, *provenance, *[f"{key}={counts[key]}" for key in sorted(counts)]])
    report_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    audit = OrdinaryUniversePriceAudit(dataset_id=f"ordinary-universe-price-audit-{report_hash[:16]}", universe_dataset_id=dataset.name, sessions=len(parts), report_hash=report_hash, **counts)
    target = Path(artifact_root) / "ordinary-universe-price-audit"
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / f"{audit.dataset_id}.json"
    encoded = json.dumps(asdict(audit), sort_keys=True, indent=2) + "\n"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != encoded:
        raise PITDataError(f"ordinary-universe price audit collision: {report_path}")
    report_path.write_text(encoded, encoding="utf-8")
    return audit
