"""Silver KRX cash-series bars: KODEX 단기채권 close."""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_reference,
    publish_dataset,
    resolve_bronze_digest,
    universe_sessions,
)
from src.data.evidence_sources import KRX_CASH_SERIES_SOURCE
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

POLICY_VERSION: Final = "krx-cash-series-v1"

_LOG = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "cash_close": pl.Int64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


class CashSeriesConfig(BaseModel):
    """Collection and Silver policy of the cash-equivalent ETF series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_start: date
    ticker: str
    available_time: time

    @field_validator("ticker")
    @classmethod
    def _six_digit_ticker(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) != 6 or not value.isdigit():
            raise ValueError("ticker must be a 6-digit string")
        return value


def load_cash_series_config(path: Path) -> CashSeriesConfig:
    """Parse the TOML; raises ValueError on unknown keys, missing keys or invalid values."""
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ValueError(f"cash-series config is unreadable: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"cash-series config is invalid TOML: {path}") from exc
    try:
        return CashSeriesConfig.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"cash-series config is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CashSeriesResult:
    dataset_path: Path
    dataset_id: str
    sessions: int


def _resolve_universe(universe_root: Path, dataset_id: str | None) -> tuple[str, tuple[date, ...]]:
    root = Path(universe_root)
    if dataset_id is None:
        candidates = sorted(
            path for path in root.glob("ordinary_universe_*") if path.is_dir() and not path.is_symlink()
        )
        if len(candidates) != 1:
            raise PITDataError("ordinary universe requires exactly one published dataset")
        dataset_id = candidates[0].name
    return universe_sessions(root, dataset_id, allow_legacy=False)


def _load_session_payload(entry: ReceiptIndexEntry, session: date) -> list[Any]:
    try:
        raw = Path(entry.payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError(f"KRX cash series payload is unreadable for {session}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"KRX cash series hash mismatch for {session}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid KRX cash series JSON for {session}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid KRX cash series root for {session}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PITDataError(f"KRX cash series records must be a list for {session}")
    return records


def _parse_int(record: dict[str, Any], field: str, *, session: date) -> int:
    raw = record.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError(f"KRX cash series record is missing {field} for {session}")
    if isinstance(raw, bool):
        raise PITDataError(f"KRX cash series record has invalid {field} for {session}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise PITDataError(f"KRX cash series record has non-integral {field} for {session}")
        return int(raw)
    try:
        parsed = Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation as exc:
        raise PITDataError(f"KRX cash series record has invalid {field} for {session}") from exc
    if parsed != parsed.to_integral_value():
        raise PITDataError(f"KRX cash series record has non-integral {field} for {session}")
    return int(parsed)


def _parse_bas_dd(value: Any, *, session: date) -> date:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise PITDataError(f"KRX cash series record has invalid BAS_DD {value!r} for {session}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise PITDataError(f"KRX cash series record has invalid BAS_DD {value!r} for {session}") from exc


def materialize_cash_series_silver(
    *,
    catalog: ReceiptCatalog,
    universe_root: Path,
    silver_root: Path,
    config: CashSeriesConfig,
    universe_dataset_id: str | None = None,
) -> CashSeriesResult:
    """Publish Silver kind ``cash_series``: one row per certified session ≥ ``collection_start``.

    Schema (partition ``session=YYYY-MM-DD/part.parquet``): session Date; cash_close Int64;
    available_at Datetime(us, Asia/Seoul); source_hash String; policy_version "krx-cash-series-v1".

    Raises:
        PITDataError: a certified session lacks a SUCCESS receipt, hash mismatch, ``BAS_DD`` conflict,
            the ticker row missing on any session, or a non-positive close.
    """
    universe_dataset_id, calendar = _resolve_universe(Path(universe_root), universe_dataset_id)
    sessions = tuple(day for day in calendar if day >= config.collection_start)
    if not sessions:
        raise PITDataError("KRX cash series has no certified sessions at or after collection_start")
    entries = catalog.latest(
        source=KRX_CASH_SERIES_SOURCE,
        natural_keys={session.isoformat() for session in sessions},
    )
    for session in sessions:
        entry = entries.get(session.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            raise PITDataError(f"KRX cash series page is missing for {session}")
        if entry.as_of != session:
            raise PITDataError(f"KRX cash series catalog date conflicts for {session}")

    source_hashes = [entries[session.isoformat()].content_hash for session in sessions]
    identity = DatasetIdentity(
        kind="cash_series",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "universe": dataset_reference(universe_dataset_id, kind="ordinary_universe"),
            "bronze_cash": resolve_bronze_digest(
                None, source_hashes, label="cash-series Bronze source"
            ),
        },
        params={
            "collection_start": config.collection_start.isoformat(),
            "ticker": config.ticker,
            "available_time": config.available_time.isoformat(),
        },
    )
    partitions: dict[str, pl.DataFrame] = {}
    partition_details: list[dict[str, object]] = []
    rows: list[dict[str, Any]] = []
    for position, session in enumerate(sessions, start=1):
        entry = entries[session.isoformat()]
        raw_records = _load_session_payload(entry, session)
        etf_rows = [
            r
            for r in raw_records
            if isinstance(r, dict)
            and r.get("_endpoint") == "etf"
            and str(r.get("ISU_CD") or "").strip() == config.ticker
        ]
        if len(etf_rows) != 1:
            if len(etf_rows) > 1:
                raise PITDataError(f"KRX cash series ETF row is duplicated for {session}")
            raise PITDataError(f"KRX cash series ETF row is missing for {session}")
        etf_record = etf_rows[0]
        if _parse_bas_dd(etf_record.get("BAS_DD"), session=session) != session:
            raise PITDataError(f"KRX cash series page date conflicts for {session}")
        cash_close = _parse_int(etf_record, "TDD_CLSPRC", session=session)
        if cash_close <= 0:
            raise PITDataError(f"KRX cash series cash close is invalid for {session}")
        available_at = datetime.combine(session, config.available_time, tzinfo=KRX_TZ)
        rows.append(
            {
                "session": session,
                "cash_close": cash_close,
                "available_at": available_at,
                "source_hash": entry.content_hash,
                "policy_version": POLICY_VERSION,
            }
        )
        frame = pl.DataFrame([rows[-1]], schema=_SCHEMA)
        relative_path = f"session={session.isoformat()}/part.parquet"
        partitions[relative_path] = frame
        partition_details.append(
            {
                "session": session.isoformat(),
                "source_hash": entry.content_hash,
                "rows": 1,
            }
        )
        if position % 100 == 0 or position == len(sessions):
            _LOG.info("[DATA] stage=cash_series sessions=%d/%d", position, len(sessions))
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions=partitions,
        details={
            "sessions": len(sessions),
            "partitions": partition_details,
        },
    )
    return CashSeriesResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        sessions=len(sessions),
    )
