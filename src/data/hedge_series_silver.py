"""Silver KRX hedge-series bars: KOSDAQ150 index level and inverse-ETF close."""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
from src.data.evidence_sources import KRX_HEDGE_SERIES_SOURCE
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

POLICY_VERSION: Final = "krx-hedge-series-v1"

_LOG = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "index_level": pl.Float64,
    "inverse_close": pl.Int64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


class HedgeSeriesConfig(BaseModel):
    """Collection and Silver policy for the KRX hedge series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_start: date
    index_name: str
    inverse_ticker: str
    available_time: time

    @field_validator("index_name")
    @classmethod
    def _non_empty_index(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("index_name must be a non-empty string")
        return value

    @field_validator("inverse_ticker")
    @classmethod
    def _six_digit_ticker(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) != 6 or not value.isdigit():
            raise ValueError("inverse_ticker must be a 6-digit string")
        return value


def load_hedge_series_config(path: Path) -> HedgeSeriesConfig:
    """Parse the TOML; raises ValueError on unknown keys, missing keys or invalid values."""
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ValueError(f"hedge-series config is unreadable: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"hedge-series config is invalid TOML: {path}") from exc
    try:
        return HedgeSeriesConfig.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"hedge-series config is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HedgeSeriesResult:
    dataset_path: Path
    dataset_id: str
    sessions: int
    inverse_listing_session: date


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
        raise PITDataError(f"KRX hedge series payload is unreadable for {session}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"KRX hedge series hash mismatch for {session}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid KRX hedge series JSON for {session}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid KRX hedge series root for {session}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PITDataError(f"KRX hedge series records must be a list for {session}")
    return records


def _parse_int(record: dict[str, Any], field: str, *, session: date) -> int:
    raw = record.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError(f"KRX hedge series record is missing {field} for {session}")
    if isinstance(raw, bool):
        raise PITDataError(f"KRX hedge series record has invalid {field} for {session}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise PITDataError(f"KRX hedge series record has non-integral {field} for {session}")
        return int(raw)
    try:
        parsed = Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation as exc:
        raise PITDataError(f"KRX hedge series record has invalid {field} for {session}") from exc
    if parsed != parsed.to_integral_value():
        raise PITDataError(f"KRX hedge series record has non-integral {field} for {session}")
    return int(parsed)


def _parse_bas_dd(value: Any, *, session: date) -> date:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise PITDataError(f"KRX hedge series record has invalid BAS_DD {value!r} for {session}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise PITDataError(f"KRX hedge series record has invalid BAS_DD {value!r} for {session}") from exc


def materialize_hedge_series_silver(
    *,
    catalog: ReceiptCatalog,
    universe_root: Path,
    silver_root: Path,
    config: HedgeSeriesConfig,
    universe_dataset_id: str | None = None,
) -> HedgeSeriesResult:
    """Normalize one hash-verified KRX hedge page per certified session ≥ ``config.collection_start``
    and publish Silver kind ``hedge_series``.

    Why: the hedge overlay needs the futures underlying (index level) and the real inverse-ETF price path;
    both are exchange-certified daily values, stored without any adjustment or fill.

    Schema (one partition ``session=YYYY-MM-DD/part.parquet`` per session, one row per session):
    session: Date; index_level: Float64; inverse_close: Int64 (null before the inverse ETF's first listed
    session); available_at: Datetime("us","Asia/Seoul"); source_hash: String; policy_version: String.

    Raises:
        PITDataError: a session in the certified calendar ≥ collection_start has no SUCCESS receipt of source
            ``krx_hedge_series``, a payload hash mismatch, a page whose ``BAS_DD`` differs from the session, a
            duplicate or missing index row, a non-positive or non-finite value, or an inverse close that is
            missing after its first listed session.
    """
    universe_dataset_id, calendar = _resolve_universe(Path(universe_root), universe_dataset_id)
    sessions = tuple(day for day in calendar if day >= config.collection_start)
    if not sessions:
        raise PITDataError("KRX hedge series has no certified sessions at or after collection_start")
    entries = catalog.latest(
        source=KRX_HEDGE_SERIES_SOURCE,
        natural_keys={session.isoformat() for session in sessions},
    )
    for session in sessions:
        entry = entries.get(session.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            raise PITDataError(f"KRX hedge series page is missing for {session}")
        if entry.as_of != session:
            raise PITDataError(f"KRX hedge series catalog date conflicts for {session}")

    source_hashes = [entries[session.isoformat()].content_hash for session in sessions]
    identity = DatasetIdentity(
        kind="hedge_series",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs={
            "universe": dataset_reference(universe_dataset_id, kind="ordinary_universe"),
            "bronze_hedge": resolve_bronze_digest(
                None, source_hashes, label="hedge-series Bronze source"
            ),
        },
        params={
            "collection_start": config.collection_start.isoformat(),
            "index_name": config.index_name,
            "inverse_ticker": config.inverse_ticker,
            "available_time": config.available_time.isoformat(),
        },
    )
    partitions: dict[str, pl.DataFrame] = {}
    partition_details: list[dict[str, object]] = []
    listing_session: date | None = None
    rows: list[dict[str, Any]] = []
    for position, session in enumerate(sessions, start=1):
        entry = entries[session.isoformat()]
        raw_records = _load_session_payload(entry, session)
        index_rows = [r for r in raw_records if isinstance(r, dict) and r.get("_endpoint") == "index"]
        etf_rows = [r for r in raw_records if isinstance(r, dict) and r.get("_endpoint") == "etf"]
        if len(index_rows) != 1:
            raise PITDataError(f"KRX hedge series index row is missing or duplicated for {session}")
        if len(etf_rows) > 1:
            raise PITDataError(f"KRX hedge series ETF row is duplicated for {session}")
        index_record = index_rows[0]
        etf_record = etf_rows[0] if etf_rows else None
        if _parse_bas_dd(index_record.get("BAS_DD"), session=session) != session:
            raise PITDataError(f"KRX hedge series page date conflicts for {session}")
        try:
            index_level = float(str(index_record.get("CLSPRC_IDX")).replace(",", "").strip())
        except (TypeError, ValueError) as exc:
            raise PITDataError(f"KRX hedge series index level is invalid for {session}") from exc
        if not math.isfinite(index_level) or index_level <= 0:
            raise PITDataError(f"KRX hedge series index level is invalid for {session}")
        inverse_close: int | None = None
        if etf_record is not None:
            if _parse_bas_dd(etf_record.get("BAS_DD"), session=session) != session:
                raise PITDataError(f"KRX hedge series page date conflicts for {session}")
            inverse_close = _parse_int(etf_record, "TDD_CLSPRC", session=session)
            if inverse_close <= 0:
                raise PITDataError(f"KRX hedge series inverse close is invalid for {session}")
            if listing_session is None:
                listing_session = session
        elif listing_session is not None:
            raise PITDataError(f"KRX hedge series ETF row is missing for {session}")
        available_at = datetime.combine(session, config.available_time, tzinfo=KRX_TZ)
        rows.append(
            {
                "session": session,
                "index_level": index_level,
                "inverse_close": inverse_close,
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
            _LOG.info("[DATA] stage=hedge_series sessions=%d/%d", position, len(sessions))
    if listing_session is None:
        raise PITDataError("KRX hedge series inverse ETF never listed in the certified calendar")
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions=partitions,
        details={
            "sessions": len(sessions),
            "partitions": partition_details,
            "inverse_listing_session": listing_session.isoformat(),
        },
    )
    return HedgeSeriesResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        sessions=len(sessions),
        inverse_listing_session=listing_session,
    )
