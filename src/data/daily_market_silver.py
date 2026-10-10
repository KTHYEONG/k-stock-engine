"""Silver KRX daily-market bars with exchange-certified base prices."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import polars as pl

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_reference,
    file_sha256,
    load_manifest,
    publish_dataset,
    resolve_bronze_digest,
    universe_sessions,
)
from src.data.payload_store import read_payload
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

REVISION = "krx-daily-market-v1"
_LOG = logging.getLogger(__name__)
_MARKETS: frozenset[str] = frozenset({"KOSPI", "KOSDAQ"})

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
    "market": pl.String,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "change": pl.Int64,
    "base_price": pl.Int64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
    "market_cap": pl.Int64,
    "listed_shares": pl.Int64,
    "fluc_rate": pl.Float64,
    "price_state": pl.String,
    "invalid_reason": pl.String,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


class PriceState(StrEnum):
    TRADABLE = "tradable"
    ZERO_VOLUME = "zero_volume"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class DailyMarketSilverPolicy:
    """Validation and availability contract for KRX daily bars."""

    available_time: time = time(18, 0)
    fluc_tolerance_pct: float = 0.01


@dataclass(frozen=True, slots=True)
class DailyMarketSilverResult:
    dataset_path: Path
    dataset_id: str
    sessions: int
    rows: int
    tradable_rows: int
    zero_volume_rows: int
    invalid_rows: int
    base_adjusted_rows: int


def _parse_int(record: dict[str, Any], field: str) -> int:
    raw = record.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError(f"KRX daily market record is missing {field}")
    if isinstance(raw, bool):
        raise PITDataError(f"KRX daily market record has invalid {field}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise PITDataError(f"KRX daily market record has non-integral {field}")
        return int(raw)
    try:
        parsed = Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation as exc:
        raise PITDataError(f"KRX daily market record has invalid {field}") from exc
    if parsed != parsed.to_integral_value():
        raise PITDataError(f"KRX daily market record has non-integral {field}")
    return int(parsed)


def _parse_fluc(record: dict[str, Any]) -> float:
    raw = record.get("FLUC_RT")
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError("KRX daily market record is missing FLUC_RT")
    try:
        value = float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise PITDataError("KRX daily market record has invalid FLUC_RT") from exc
    if not math.isfinite(value):
        raise PITDataError("KRX daily market record has invalid FLUC_RT")
    return value


def _parse_bas_dd(value: Any) -> date:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise PITDataError(f"KRX daily market record has invalid BAS_DD {value!r}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise PITDataError(f"KRX daily market record has invalid BAS_DD {value!r}") from exc


def _classify(
    *,
    open_price: int,
    high_price: int,
    low_price: int,
    close: int,
    change: int,
    base_price: int,
    volume: int,
    fluc_rate: float,
    tolerance: float,
) -> tuple[PriceState, str | None]:
    if close <= 0 or base_price <= 0:
        return (PriceState.INVALID, "non_positive_close_or_base")
    if abs(change / base_price * 100.0 - fluc_rate) > tolerance:
        return (PriceState.INVALID, "fluc_mismatch")
    if volume == 0:
        return (PriceState.ZERO_VOLUME, None)
    if (
        open_price <= 0
        or high_price <= 0
        or low_price <= 0
        or low_price > min(open_price, close)
        or high_price < max(open_price, close)
    ):
        return (PriceState.INVALID, "ohlc_inconsistent")
    return (PriceState.TRADABLE, None)


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
        raw = read_payload(Path(entry.payload_path))
    except (OSError, PITDataError) as exc:
        raise PITDataError(f"KRX daily market payload is unreadable for {session}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"KRX daily market hash mismatch for {session}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid KRX daily market JSON for {session}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid KRX daily market root for {session}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PITDataError(f"KRX daily market records must be a list for {session}")
    return records


def _universe_partition_shas(universe_root: Path, universe_dataset_id: str) -> dict[str, str]:
    """Manifest partition digests of the resolved universe keyed by session isoformat."""
    manifest = load_manifest(Path(universe_root) / universe_dataset_id)
    shas: dict[str, str] = {}
    for partition in manifest.partitions:
        marker = "session="
        if marker not in partition.path:
            continue
        shas[partition.path.split(marker, 1)[1].split("/", 1)[0]] = partition.sha256
    return shas


def _reusable_daily_frames(
    reuse_from: Path | None,
    current: dict[str, str],
    universe_shas: dict[str, str],
    expected_params: dict[str, object],
) -> dict[date, pl.DataFrame]:
    """Prior-session frames whose manifest entry verifies and whose inputs are unchanged."""
    if reuse_from is None:
        return {}
    try:
        manifest = load_manifest(Path(reuse_from))
    except PITDataError:
        return {}
    if manifest.kind != "daily_market" or manifest.policy_version != REVISION:
        return {}
    if dict(manifest.params) != dict(expected_params):
        return {}
    raw_partitions = manifest.details.get("partitions")
    if not isinstance(raw_partitions, list):
        return {}
    recorded: dict[str, tuple[str, str | None]] = {}
    for item in raw_partitions:
        if not isinstance(item, dict) or not isinstance(item.get("session"), str) or not isinstance(item.get("source_hash"), str):
            return {}
        universe_sha = item.get("universe_sha")
        if universe_sha is not None and not isinstance(universe_sha, str):
            return {}
        recorded[str(item["session"])] = (str(item["source_hash"]), str(universe_sha) if universe_sha is not None else None)
    expected_sha = {partition.path: partition.sha256 for partition in manifest.partitions}
    reused: dict[date, pl.DataFrame] = {}
    for session_iso, source_hash in current.items():
        entry = recorded.get(session_iso)
        if entry is None or entry[0] != source_hash:
            continue
        if entry[1] is None or entry[1] != universe_shas.get(session_iso):
            continue
        relative_path = f"session={session_iso}/part.parquet"
        if expected_sha.get(relative_path) is None:
            continue
        path = Path(reuse_from) / relative_path
        try:
            if file_sha256(path) != expected_sha[relative_path]:
                continue
            reused[date.fromisoformat(session_iso)] = pl.read_parquet(path)
        except Exception:  # noqa: S112 - an unverifiable prior partition is rebuilt from payloads
            continue
    return reused


def materialize_daily_market_silver(
    *,
    catalog: ReceiptCatalog,
    universe_root: Path,
    silver_root: Path,
    policy: DailyMarketSilverPolicy = DailyMarketSilverPolicy(),  # noqa: B008
    universe_dataset_id: str | None = None,
    bronze_daily_digest: str | None = None,
    reuse_from: Path | None = None,
) -> DailyMarketSilverResult:
    """Publish the daily-market Silver; unchanged sessions may be copied from the dataset at reuse_from."""

    if not isinstance(policy.available_time, time):
        raise PITDataError("daily market available_time must be a time")
    if (
        isinstance(policy.fluc_tolerance_pct, bool)
        or not isinstance(policy.fluc_tolerance_pct, (int, float))
        or not math.isfinite(float(policy.fluc_tolerance_pct))
        or policy.fluc_tolerance_pct < 0
    ):
        raise PITDataError("daily market fluctuation tolerance must be finite and non-negative")
    universe_dataset_id, calendar = _resolve_universe(Path(universe_root), universe_dataset_id)
    entries = catalog.latest(
        source="krx_daily_market",
        natural_keys={session.isoformat() for session in calendar},
    )
    for session in calendar:
        entry = entries.get(session.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            raise PITDataError(f"KRX daily market page is missing for {session}")
        if entry.as_of != session:
            raise PITDataError(f"KRX daily market catalog date conflicts for {session}")

    source_hashes = [entries[session.isoformat()].content_hash for session in calendar]
    identity = DatasetIdentity(
        kind="daily_market",
        layer=DatasetLayer.SILVER,
        policy_version=REVISION,
        inputs={
            "universe": dataset_reference(universe_dataset_id, kind="ordinary_universe"),
            "bronze_daily": resolve_bronze_digest(
                bronze_daily_digest,
                source_hashes,
                label="daily-market Bronze source",
            ),
        },
        params={
            "available_time": policy.available_time.isoformat(),
            "fluc_tolerance_pct": policy.fluc_tolerance_pct,
        },
    )
    partitions: dict[str, pl.DataFrame] = {}
    partition_details: list[dict[str, object]] = []
    tradable_rows = 0
    zero_volume_rows = 0
    invalid_rows = 0
    base_adjusted_rows = 0
    universe_shas = _universe_partition_shas(Path(universe_root), universe_dataset_id)
    current_hashes = {session.isoformat(): entries[session.isoformat()].content_hash for session in calendar}
    reusable = _reusable_daily_frames(
        reuse_from, current_hashes, universe_shas, dict(identity.params)
    )
    previous_closes: dict[str, int] = {}
    for position, session in enumerate(calendar, start=1):
        entry = entries[session.isoformat()]
        cached = reusable.get(session)
        if cached is not None:
            frame = cached
        else:
            records = _load_session_payload(entry, session)
            rows: list[dict[str, Any]] = []
            seen: set[str] = set()
            seen_tickers: set[str] = set()
            available_at = datetime.combine(session, policy.available_time, tzinfo=KRX_TZ)
            for record in records:
                if not isinstance(record, dict):
                    raise PITDataError(f"KRX daily market record must be an object for {session}")
                market = str(record.get("MKT_NM") or "").strip()
                if market not in _MARKETS:
                    continue
                if _parse_bas_dd(record.get("BAS_DD")) != session:
                    raise PITDataError(f"KRX daily market page date conflicts for {session}")
                isu = str(record.get("ISU_CD") or "").strip()
                if not isu:
                    raise PITDataError(f"KRX daily market record is missing instrument identity for {session}")
                if isu in seen:
                    raise PITDataError(f"KRX daily market has duplicate rows for {session}")
                seen.add(isu)
                open_price = _parse_int(record, "TDD_OPNPRC")
                high_price = _parse_int(record, "TDD_HGPRC")
                low_price = _parse_int(record, "TDD_LWPRC")
                close = _parse_int(record, "TDD_CLSPRC")
                change = _parse_int(record, "CMPPREVDD_PRC")
                volume = _parse_int(record, "ACC_TRDVOL")
                trading_value = _parse_int(record, "ACC_TRDVAL")
                market_cap = _parse_int(record, "MKTCAP")
                listed_shares = _parse_int(record, "LIST_SHRS")
                fluc_rate = _parse_fluc(record)
                base_price = close - change
                state, reason = _classify(
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close=close,
                    change=change,
                    base_price=base_price,
                    volume=volume,
                    fluc_rate=fluc_rate,
                    tolerance=policy.fluc_tolerance_pct,
                )
                ticker = str(record.get("ISU_SRT_CD") or isu).strip()
                if ticker in seen_tickers:
                    raise PITDataError(f"KRX daily market has duplicate ticker for {session}: {ticker}")
                seen_tickers.add(ticker)
                rows.append(
                    {
                        "session": session,
                        # The universe and all downstream Silver/Gold tables use the
                        # exchange ticker as their stable instrument key.  ISU_CD is
                        # retained as the source-page identity only; using it here
                        # would silently turn every universe join into an absent row.
                        "instrument_id": f"KRX:{ticker}",
                        "ticker": ticker,
                        "market": market,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close,
                        "change": change,
                        "base_price": base_price,
                        "volume": volume,
                        "trading_value": trading_value,
                        "market_cap": market_cap,
                        "listed_shares": listed_shares,
                        "fluc_rate": fluc_rate,
                        "price_state": state.value,
                        "invalid_reason": reason,
                        "available_at": available_at,
                        "source_hash": entry.content_hash,
                        "policy_version": REVISION,
                    }
                )
            frame = (
                pl.DataFrame(rows, schema=_SCHEMA).sort(["session", "ticker"])
                if rows
                else pl.DataFrame([], schema=_SCHEMA)
            )
        if frame.height:
            tickers = frame.get_column("ticker").to_list()
            closes = frame.get_column("close").to_list()
            bases = frame.get_column("base_price").to_list()
            for ticker, base_price in zip(tickers, bases, strict=True):
                if ticker in previous_closes and base_price != previous_closes[ticker]:
                    base_adjusted_rows += 1
            previous_closes = dict(zip(tickers, closes, strict=True))
            tradable_rows += int(frame.filter(pl.col("price_state") == "tradable").height)
            zero_volume_rows += int(frame.filter(pl.col("price_state") == "zero_volume").height)
            invalid_rows += int(frame.filter(pl.col("price_state") == "invalid").height)
        else:
            previous_closes = {}
        relative_path = f"session={session.isoformat()}/part.parquet"
        partitions[relative_path] = frame
        partition_details.append(
            {
                "session": session.isoformat(),
                "source_hash": entry.content_hash,
                "universe_sha": universe_shas.get(session.isoformat()),
                "rows": frame.height,
                "tradable_rows": int(frame.filter(pl.col("price_state") == "tradable").height) if frame.height else 0,
                "zero_volume_rows": int(frame.filter(pl.col("price_state") == "zero_volume").height) if frame.height else 0,
                "invalid_rows": int(frame.filter(pl.col("price_state") == "invalid").height) if frame.height else 0,
            }
        )
        if position % 100 == 0 or position == len(calendar):
            _LOG.info(
                "[DATA] stage=daily_market sessions=%d/%d rows=%d",
                position,
                len(calendar),
                sum(cast(int, item["rows"]) for item in partition_details),
            )

    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions=partitions,
        details={
            "sessions": len(calendar),
            "partitions": partition_details,
            "tradable_rows": tradable_rows,
            "zero_volume_rows": zero_volume_rows,
            "invalid_rows": invalid_rows,
            "base_adjusted_rows": base_adjusted_rows,
        },
    )
    return DailyMarketSilverResult(
        dataset_path=published.path,
        dataset_id=published.dataset_id,
        sessions=len(calendar),
        rows=sum(cast(int, item["rows"]) for item in partition_details),
        tradable_rows=tradable_rows,
        zero_volume_rows=zero_volume_rows,
        invalid_rows=invalid_rows,
        base_adjusted_rows=base_adjusted_rows,
    )
