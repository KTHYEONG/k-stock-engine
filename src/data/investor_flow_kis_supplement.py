"""Provider-tagged KIS Silver supplement covering exactly the LS coverage gap."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_partition_paths,
    dataset_reference,
    publish_dataset,
    resolve_bronze_digest,
)
from src.data.flow_targets import FlowTargets
from src.data.investor_flow_gap import compute_missing_investor_flow_cells
from src.data.parse_cache import ParseCache, source_fingerprint
from src.data.payload_store import read_payload
from src.data.receipt_catalog import BlobEntry, ReceiptCatalog

REVISION = "kis-investor-trade-net-shares-supplement-v2"

_LOG = logging.getLogger(__name__)

_PAGE_BATCH = 1000

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
    "provider": pl.String,
    "individual_net_shares": pl.Int64,
    "foreign_net_shares": pl.Int64,
    "institution_net_shares": pl.Int64,
    "other_net_shares": pl.Int64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


@dataclass(frozen=True, slots=True)
class InvestorFlowKisSupplementPolicy:
    """Availability contract for the KIS supplement, matching P0-3's LS policy shape."""

    available_session_lag: int = 1
    available_time: time = time(8, 0)


@dataclass(frozen=True, slots=True)
class InvestorFlowKisSupplementResult:
    dataset_path: Path
    dataset_id: str
    ls_dataset_id: str
    target_cells: int
    filled_cells: int
    still_missing_cells: int
    identity_violation_cells: int
    rows: int


def _parse_qty(row: dict[str, Any], field: str) -> int:
    raw = row.get(field)
    if raw is None or isinstance(raw, bool) or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError(f"KIS investor flow row is missing {field}")
    try:
        parsed = Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation as exc:
        raise PITDataError(f"KIS investor flow row has invalid {field}") from exc
    if parsed != parsed.to_integral_value():
        raise PITDataError(f"KIS investor flow row has non-integral {field}")
    return int(parsed)


def _parse_session(value: Any) -> date:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise PITDataError(f"KIS investor flow row has invalid date {value!r}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise PITDataError(f"KIS investor flow row has invalid date {value!r}") from exc


def _parse_kis_blob_full(
    entry: BlobEntry, sessions: frozenset[date]
) -> tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], list[date]]:
    try:
        raw = read_payload(Path(entry.payload_path))
    except (OSError, PITDataError) as exc:
        raise PITDataError(f"investor-flow Bronze payload is unreadable: {entry.payload_path}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"investor-flow Bronze hash mismatch: {entry.payload_path}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid investor-flow Bronze JSON: {entry.payload_path}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid investor-flow Bronze root: {entry.payload_path}")
    provider = payload.get("provider")
    if not isinstance(provider, str) or provider.strip() != "KIS":
        raise PITDataError(f"investor-flow Bronze page is not KIS: {entry.payload_path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PITDataError(f"KIS investor flow page carries no rows: {entry.payload_path}")
    query = payload.get("query")
    query_map = query if isinstance(query, dict) else {}
    symbol = str(query_map.get("symbol") or payload.get("symbol") or "").strip()
    if not symbol:
        raise PITDataError(f"KIS investor flow raw page lacks query symbol: {entry.payload_path}")
    parsed: list[tuple[date, tuple[int, int, int, int]]] = []
    out_of_calendar: list[date] = []
    for row in rows:
        if not isinstance(row, dict):
            raise PITDataError(f"KIS investor flow row must be an object: {entry.payload_path}")
        session = _parse_session(row.get("stck_bsop_date"))
        if session not in sessions:
            out_of_calendar.append(session)
            continue
        quantities = (
            _parse_qty(row, "prsn_ntby_qty"),
            _parse_qty(row, "frgn_ntby_qty"),
            _parse_qty(row, "orgn_ntby_qty"),
            _parse_qty(row, "etc_ntby_qty"),
        )
        parsed.append((session, quantities))
    return (entry.content_hash, symbol, parsed, out_of_calendar)


def _parse_kis_blob(
    entry: BlobEntry, sessions: frozenset[date]
) -> tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], int]:
    page_hash, symbol, parsed, out_of_calendar = _parse_kis_blob_full(entry, sessions)
    return (page_hash, symbol, parsed, len(out_of_calendar))


_PageRows = list[tuple[date, tuple[int, int, int, int]]]
_PageOutOfCalendar = list[date]

_PARSE_FINGERPRINT: str | None = None


def parse_fingerprint() -> str:
    """Fingerprint of this module's source for parse-cache invalidation."""
    global _PARSE_FINGERPRINT
    if _PARSE_FINGERPRINT is None:
        _PARSE_FINGERPRINT = source_fingerprint(sys.modules[__name__])
    return _PARSE_FINGERPRINT


def _encode_page(symbol: str, rows: _PageRows, out_of_calendar: _PageOutOfCalendar) -> dict[str, Any]:
    """Encode one parsed page into JSON-native form."""
    return {
        "symbol": symbol,
        "rows": [[session.toordinal(), list(quantities)] for session, quantities in rows],
        "oc": [session.toordinal() for session in out_of_calendar],
    }


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decode_page(
    value: Any, sessions: frozenset[date]
) -> tuple[str, _PageRows, _PageOutOfCalendar]:
    """Rebuild parse tuples from cached JSON; calendar drift is a miss."""
    if not isinstance(value, dict):
        raise TypeError("cached page must be an object")
    symbol = value.get("symbol")
    raw_rows = value.get("rows")
    raw_oc = value.get("oc")
    if not isinstance(symbol, str) or not symbol:
        raise TypeError("cached page symbol is malformed")
    if not isinstance(raw_rows, list) or not isinstance(raw_oc, list):
        raise TypeError("cached page shape is malformed")
    rows: _PageRows = []
    for item in raw_rows:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("cached cell is malformed")
        ordinal, quantities = item
        if not _is_plain_int(ordinal):
            raise TypeError("cached cell session is malformed")
        if not isinstance(quantities, list) or len(quantities) != 4:
            raise TypeError("cached cell quantities are malformed")
        if any(not _is_plain_int(item) for item in quantities):
            raise TypeError("cached cell quantities are malformed")
        try:
            session = date.fromordinal(int(ordinal))
        except ValueError as exc:
            raise TypeError("cached cell session is malformed") from exc
        rows.append((session, (int(quantities[0]), int(quantities[1]), int(quantities[2]), int(quantities[3]))))
    out_of_calendar: _PageOutOfCalendar = []
    for ordinal in raw_oc:
        if not _is_plain_int(ordinal):
            raise TypeError("cached oc session is malformed")
        try:
            out_of_calendar.append(date.fromordinal(int(ordinal)))
        except ValueError as exc:
            raise TypeError("cached oc session is malformed") from exc
    for session, _ in rows:
        if session not in sessions:
            raise ValueError("cached page is stale for the current calendar")
    for session in out_of_calendar:
        if session in sessions:
            raise ValueError("cached page is stale for the current calendar")
    return (symbol, rows, out_of_calendar)


def _load_daily_calendar(daily_market_path: Path) -> tuple[date, ...]:
    sessions: set[date] = set()
    for path in dataset_partition_paths(Path(daily_market_path), allow_legacy=False):
        try:
            frame = pl.read_parquet(path, columns=["session"])
        except pl.exceptions.ColumnNotFoundError:
            continue
        sessions.update(value for value in frame["session"].to_list() if isinstance(value, date))
    if not sessions:
        raise PITDataError(f"daily market has no session partitions: {daily_market_path}")
    return tuple(sorted(sessions))


def materialize_investor_flow_kis_supplement(
    *,
    catalog: ReceiptCatalog,
    targets: FlowTargets,
    ls_flow_silver_path: Path,
    silver_root: Path,
    policy: InvestorFlowKisSupplementPolicy = InvestorFlowKisSupplementPolicy(),  # noqa: B008
    bronze_kis_digest: str | None = None,
    parse_cache: ParseCache | None = None,
) -> InvestorFlowKisSupplementResult:
    """Publish the KIS supplement for exactly the LS coverage gap."""

    if (
        isinstance(policy.available_session_lag, bool)
        or not isinstance(policy.available_session_lag, int)
        or policy.available_session_lag < 1
    ):
        raise PITDataError("KIS supplement availability lag must be positive")
    if not isinstance(policy.available_time, time):
        raise PITDataError("KIS supplement available_time must be a time")
    gap = compute_missing_investor_flow_cells(
        targets=targets, ls_flow_silver_path=Path(ls_flow_silver_path)
    )
    target = {(session, symbol.ticker) for symbol in gap.symbols for session in symbol.sessions}
    daily_market_path = Path(silver_root) / targets.daily_market_dataset_id
    calendar = _load_daily_calendar(daily_market_path)
    session_set = frozenset(calendar)
    blobs = list(catalog.blobs(source="kis_investor_flow", usable=True))
    options: dict[tuple[date, str], dict[tuple[int, int, int, int], str]] = {}
    violation_keys: set[tuple[date, str]] = set()
    kis_hashes: list[str] = []
    parsed_pages = 0
    out_of_calendar_rows = 0
    for offset in range(0, len(blobs), _PAGE_BATCH):
        batch = blobs[offset : offset + _PAGE_BATCH]
        ordered: list[tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], int]] = []
        if parse_cache is None:
            ordered.extend(_parse_kis_blob(entry, session_set) for entry in batch)
        else:
            cached_raw = parse_cache.get_many([blob.content_hash for blob in batch])
            hits: dict[str, tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], int]] = {}
            miss_unique: list[BlobEntry] = []
            miss_seen: set[str] = set()
            for blob in batch:
                raw_value = cached_raw.get(blob.content_hash)
                if raw_value is None:
                    if blob.content_hash not in miss_seen:
                        miss_seen.add(blob.content_hash)
                        miss_unique.append(blob)
                    continue
                try:
                    symbol, rows, oc = _decode_page(raw_value, session_set)
                except Exception:
                    if blob.content_hash not in miss_seen:
                        miss_seen.add(blob.content_hash)
                        miss_unique.append(blob)
                    continue
                hits[blob.content_hash] = (blob.content_hash, symbol, rows, len(oc))
            fresh: dict[str, dict[str, Any]] = {}
            miss_by_hash: dict[str, tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], int]] = {}
            for entry in miss_unique:
                page_hash, symbol, rows, oc_sessions = _parse_kis_blob_full(entry, session_set)
                miss_by_hash[page_hash] = (page_hash, symbol, rows, len(oc_sessions))
                fresh[page_hash] = _encode_page(symbol, rows, oc_sessions)
            for blob in batch:
                if blob.content_hash in hits:
                    ordered.append(hits[blob.content_hash])
                else:
                    ordered.append(miss_by_hash[blob.content_hash])
            if fresh:
                parse_cache.put_many(fresh)
        for page_hash, symbol, parsed_rows, page_out_of_calendar in ordered:
            parsed_pages += 1
            out_of_calendar_rows += page_out_of_calendar
            kis_hashes.append(page_hash)
            for session, values in parsed_rows:
                key = (session, symbol)
                if key not in target:
                    continue
                if values[0] + values[1] + values[2] + values[3] != 0:
                    violation_keys.add(key)
                    continue
                known = options.setdefault(key, {}).get(values)
                if known is None or page_hash < known:
                    options[key][values] = page_hash
        _LOG.info(
            "[DATA] stage=investor_flow_kis pages=%d/%d filled=%d out_of_calendar_rows=%d",
            parsed_pages,
            len(blobs),
            len(options),
            out_of_calendar_rows,
        )

    lag = policy.available_session_lag
    agreed: dict[tuple[date, str], tuple[tuple[int, int, int, int], str]] = {}
    for key, hashes in options.items():
        if len(hashes) == 1:
            values = next(iter(hashes))
            agreed[key] = (values, hashes[values])
    resolved = [
        (session, ticker, values[0], values[1], values[2], values[3], source_hash)
        for (session, ticker), (values, source_hash) in agreed.items()
    ]
    frame = (
        pl.DataFrame(
            {
                "session": [item[0] for item in resolved],
                "ticker": [item[1] for item in resolved],
                "individual_net_shares": [item[2] for item in resolved],
                "foreign_net_shares": [item[3] for item in resolved],
                "institution_net_shares": [item[4] for item in resolved],
                "other_net_shares": [item[5] for item in resolved],
                "source_hash": [item[6] for item in resolved],
            },
            schema={
                "session": pl.Date,
                "ticker": pl.String,
                "individual_net_shares": pl.Int64,
                "foreign_net_shares": pl.Int64,
                "institution_net_shares": pl.Int64,
                "other_net_shares": pl.Int64,
                "source_hash": pl.String,
            },
        )
        .with_columns(
            pl.concat_str(pl.lit("KRX:"), pl.col("ticker")).alias("instrument_id"),
            pl.lit("KIS").alias("provider"),
        )
        .join(
            pl.DataFrame(
                {
                    "session": list(calendar),
                    "available_session": [
                        calendar[index + lag] if index + lag < len(calendar) else None
                        for index in range(len(calendar))
                    ],
                },
                schema={"session": pl.Date, "available_session": pl.Date},
            ),
            on="session",
            how="left",
        )
        .filter(pl.col("available_session").is_not_null())
        .select(
            "session",
            "instrument_id",
            "ticker",
            "provider",
            "individual_net_shares",
            "foreign_net_shares",
            "institution_net_shares",
            "other_net_shares",
            pl.col("available_session")
            .dt.combine(policy.available_time)
            .dt.replace_time_zone(str(KRX_TZ))
            .alias("available_at"),
            "source_hash",
            pl.lit(REVISION).alias("policy_version"),
        )
        .cast(pl.Schema(_SCHEMA))
        .sort(["session", "ticker"])
    )
    partitions: dict[str, pl.DataFrame] = {}
    partition_details: list[dict[str, object]] = []
    for year in sorted(frame["session"].dt.year().unique().to_list()) if frame.height else []:
        part = frame.filter(pl.col("session").dt.year() == year)
        relative_path = f"year={year}/part.parquet"
        partitions[relative_path] = part
        partition_details.append({"year": year, "rows": part.height})

    source_digest = resolve_bronze_digest(
        bronze_kis_digest,
        kis_hashes,
        label="KIS investor-flow Bronze source",
    )
    identity = DatasetIdentity(
        kind="investor_flow_kis_supplement",
        layer=DatasetLayer.SILVER,
        policy_version=REVISION,
        inputs={
            "universe": dataset_reference(targets.universe_dataset_id, kind="ordinary_universe"),
            "daily_market": dataset_reference(targets.daily_market_dataset_id, kind="daily_market"),
            "ls": dataset_reference(gap.ls_dataset_id, kind="investor_flow_ls"),
            "bronze_kis": source_digest,
        },
        params={
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
        },
    )
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions=partitions,
        details={
            "universe_dataset_id": targets.universe_dataset_id,
            "daily_market_dataset_id": targets.daily_market_dataset_id,
            "ls_dataset_id": gap.ls_dataset_id,
            "partitions": partition_details,
            "target_cells": len(target),
            "filled_cells": frame.height,
            "still_missing_cells": len(target) - frame.height,
            "identity_violation_cells": len(violation_keys),
            "out_of_calendar_rows": out_of_calendar_rows,
            "parsed_pages": parsed_pages,
        },
    )
    return InvestorFlowKisSupplementResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        ls_dataset_id=gap.ls_dataset_id,
        target_cells=len(target),
        filled_cells=frame.height,
        still_missing_cells=len(target) - frame.height,
        identity_violation_cells=len(violation_keys),
        rows=frame.height,
    )
