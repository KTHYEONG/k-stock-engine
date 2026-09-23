"""Silver investor-flow dataset built from hash-verified raw LS t1702 rows."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ
from src.data.schemas import PITDataError

POLICY_VERSION = "ls-t1702-net-shares-v1"

_LOG = logging.getLogger(__name__)

_GROUP_CODES: tuple[str, ...] = (
    *(f"tjj{i:04d}" for i in range(12)),
    "tjj0016",
    "tjj0017",
    "tjj0018",
)
_SUBGROUP_COLUMNS: tuple[str, ...] = (
    *(f"tjj{i:04d}_net_shares" for i in range(8)),
    "tjj0009_net_shares",
    "tjj0010_net_shares",
    "tjj0011_net_shares",
)
_NEGATIVE_STATUSES: frozenset[str] = frozenset({"provider_error", "missing_sessions"})
_PAGE_BATCH = 1000
_SHARD_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "ticker": pl.String,
    **dict.fromkeys(_GROUP_CODES, pl.Int64),
    "ls_close": pl.Int64,
    "ls_volume": pl.Int64,
    "ls_value_mkrw": pl.Int64,
    "source_hash": pl.String,
}

_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
    "provider": pl.String,
    "individual_net_shares": pl.Int64,
    "foreign_net_shares": pl.Int64,
    "institution_net_shares": pl.Int64,
    "other_net_shares": pl.Int64,
    **dict.fromkeys(_SUBGROUP_COLUMNS, pl.Int64),
    "ls_close": pl.Int64,
    "ls_volume": pl.Int64,
    "ls_value_mkrw": pl.Int64,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


@dataclass(frozen=True, slots=True)
class InvestorFlowSilverPolicy:
    """Availability contract for per-investor daily flows.

    Attributes:
        available_session_lag: Sessions after the flow session before the row
            may be consumed (1 = next session).
        available_time: KST wall-clock time on the lagged session.
    """

    available_session_lag: int = 1
    available_time: time = time(8, 0)


@dataclass(frozen=True, slots=True)
class InvestorFlowSilverResult:
    dataset_path: Path
    dataset_id: str
    rows: int
    tickers: int
    raw_pages: int
    ignored_records_only_pages: int
    negative_cells: int
    conflict_cells: int
    identity_violation_cells: int
    unavailable_tail_cells: int
    retail_exceeds_volume_rows: int
    dateless_rows: int


def _parse_share_int(row: dict[str, Any], field: str) -> int:
    raw = row.get(field)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise PITDataError(f"LS investor flow row is missing {field}")
    if isinstance(raw, bool):
        raise PITDataError(f"LS investor flow row has invalid {field}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise PITDataError(f"LS investor flow row has non-integral {field}")
        return int(raw)
    try:
        parsed = Decimal(str(raw).replace(",", "").strip())
    except InvalidOperation as exc:
        raise PITDataError(f"LS investor flow row has invalid {field}") from exc
    if parsed != parsed.to_integral_value():
        raise PITDataError(f"LS investor flow row has non-integral {field}")
    return int(parsed)


def _parse_row_date(value: Any) -> date:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise PITDataError(f"LS investor flow row has invalid date {value!r}")
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise PITDataError(f"LS investor flow row has invalid date {value!r}") from exc


def _parse_query_bound(value: Any, *, label: str) -> date:
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise PITDataError(f"LS investor flow page has invalid query {label}") from exc


def _check_identities(groups: dict[str, int]) -> None:
    if groups["tjj0018"] != sum(groups[f"tjj{i:04d}"] for i in range(7)):
        raise PITDataError("LS investor flow violates institution aggregate identity")
    if groups["tjj0016"] != groups["tjj0009"] + groups["tjj0010"]:
        raise PITDataError("LS investor flow violates foreign aggregate identity")
    if groups["tjj0017"] != groups["tjj0007"] + groups["tjj0011"]:
        raise PITDataError("LS investor flow violates other aggregate identity")
    if groups["tjj0008"] + groups["tjj0016"] + groups["tjj0017"] + groups["tjj0018"] != 0:
        raise PITDataError("LS investor flow violates zero-sum identity")


def _parse_page(path: Path, sessions: frozenset[date]) -> tuple[str, str, str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.parent.name:
        raise PITDataError(f"investor-flow Bronze hash mismatch: {path}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid investor-flow Bronze JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid investor-flow Bronze root: {path}")
    page_hash = path.parent.name
    rows = payload.get("rows")
    query = payload.get("query")
    query_map = query if isinstance(query, dict) else {}
    symbol = str(query_map.get("symbol") or "").strip()
    if isinstance(rows, list) and rows:
        if not symbol:
            raise PITDataError(f"LS investor flow raw page lacks query symbol: {path}")
        start = _parse_query_bound(query_map.get("start"), label="start")
        end = _parse_query_bound(query_map.get("end"), label="end")
        cells: list[tuple[date, tuple[int, ...], int, int, int]] = []
        dateless_rows = 0
        violations: list[date] = []
        for row in rows:
            if not isinstance(row, dict):
                raise PITDataError(f"LS investor flow row must be an object: {path}")
            if str(row.get("date") or "").strip() == "":
                # 공급처가 붙이는 날짜 없는 부가 행: 수급값이 전부 0일 때만 정보 없음으로 건너뛴다.
                if any(_parse_share_int(row, code) != 0 for code in _GROUP_CODES):
                    raise PITDataError(f"LS investor flow dateless row carries flow values: {path}")
                dateless_rows += 1
                continue
            session = _parse_row_date(row.get("date"))
            if session < start or session > end:
                continue
            if session not in sessions:
                raise PITDataError(f"LS investor flow session outside certified calendar: {session}")
            groups = {code: _parse_share_int(row, code) for code in _GROUP_CODES}
            try:
                _check_identities(groups)
            except PITDataError:
                # 공급처 집계 오류 행은 셀 단위로 격리한다(값 추정·보정 금지).
                violations.append(session)
                continue
            cells.append((
                session,
                tuple(groups[code] for code in _GROUP_CODES),
                _parse_share_int(row, "close"),
                _parse_share_int(row, "volume"),
                _parse_share_int(row, "value"),
            ))
        return ("raw", page_hash, symbol, (cells, dateless_rows, violations))
    status = str(payload.get("status") or "").strip()
    if status in _NEGATIVE_STATUSES:
        key = "sessions" if status == "provider_error" else "missing_sessions"
        items = payload.get(key)
        listed = [str(item) for item in items] if isinstance(items, list) else []
        return (status, page_hash, str(payload.get("symbol") or "").strip(), listed)
    return ("ignored", page_hash, "", [])


def _load_universe_sessions(universe_root: Path) -> tuple[str, tuple[date, ...]]:
    datasets = sorted(path for path in Path(universe_root).glob("ordinary_universe_*") if path.is_dir())
    if len(datasets) != 1:
        raise PITDataError("ordinary universe requires exactly one published dataset")
    dataset = datasets[0]
    try:
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        parts = manifest["partitions"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError("invalid ordinary-universe manifest") from exc
    if manifest.get("dataset_id") != dataset.name or not isinstance(parts, list) or not parts:
        raise PITDataError("invalid ordinary-universe manifest")
    sessions: list[date] = []
    for part in parts:
        if not isinstance(part, dict):
            raise PITDataError("invalid ordinary-universe partition")
        try:
            session = date.fromisoformat(str(part.get("session"))[:10])
        except ValueError as exc:
            raise PITDataError("invalid ordinary-universe partition session") from exc
        if sessions and session <= sessions[-1]:
            raise PITDataError("ordinary-universe partitions are not strictly ordered")
        sessions.append(session)
    return (dataset.name, tuple(sessions))


def materialize_investor_flow_silver(
    *,
    bronze_root: Path,
    universe_root: Path,
    silver_root: Path,
    policy: InvestorFlowSilverPolicy = InvestorFlowSilverPolicy(),  # noqa: B008 - spec-mandated immutable default
    workers: int = 4,
) -> InvestorFlowSilverResult:
    """Build an immutable Silver investor-flow dataset from raw LS t1702 rows.

    Values come only from hash-verified raw ``rows``; derived ``records`` are
    ignored because their historical unit labeling is untrustworthy. The
    certified ordinary-universe session list is the calendar that assigns
    ``available_at``; sessions outside it are rejected.

    Args:
        bronze_root: Scope Bronze root containing ``investor_flow/<sha256>/``.
        universe_root: Scope Silver root holding exactly one ordinary-universe
            dataset whose manifest defines certified sessions.
        silver_root: Scope Silver root receiving ``investor_flow_<hash16>/``.
        policy: Availability contract.
        workers: Bounded parallel page parsers.

    Returns:
        Counts describing coverage and every excluded cell class.

    Raises:
        PITDataError: payload hash mismatch, malformed raw row, violated
            aggregate identity, non-integral quantity, unknown session, or an
            existing dataset with different content.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise PITDataError("workers must be a positive integer")
    universe_dataset_id, calendar = _load_universe_sessions(Path(universe_root))
    session_set = frozenset(calendar)
    page_paths = sorted((Path(bronze_root) / "investor_flow").glob("*/payload.json"))
    silver_root = Path(silver_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".investor-flow-silver-", dir=silver_root))
    try:
        shard_dir = staging / "_shards"
        shard_dir.mkdir()
        raw_hashes: list[str] = []
        ignored_records_only_pages = 0
        negative_keys: set[tuple[str, str]] = set()
        violation_keys: set[tuple[str, str]] = set()
        dateless_rows = 0
        parsed_pages = 0
        parsed_rows = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(page_paths), _PAGE_BATCH):
                batch = page_paths[offset : offset + _PAGE_BATCH]
                columns: dict[str, list[Any]] = {name: [] for name in _SHARD_SCHEMA}
                for kind, page_hash, symbol, content in pool.map(_parse_page, batch, [session_set] * len(batch)):
                    parsed_pages += 1
                    if kind == "raw":
                        raw_hashes.append(page_hash)
                        cells, page_dateless, page_violations = content
                        dateless_rows += page_dateless
                        violation_keys.update((day.isoformat(), symbol) for day in page_violations)
                        for session, groups, close, volume, value in cells:
                            columns["session"].append(session)
                            columns["ticker"].append(symbol)
                            for code, amount in zip(_GROUP_CODES, groups, strict=True):
                                columns[code].append(amount)
                            columns["ls_close"].append(close)
                            columns["ls_volume"].append(volume)
                            columns["ls_value_mkrw"].append(value)
                            columns["source_hash"].append(page_hash)
                        parsed_rows += len(cells)
                    elif kind == "ignored":
                        ignored_records_only_pages += 1
                    else:
                        for listed_session in content:
                            negative_keys.add((listed_session, symbol))
                # 페이지 묶음마다 컬럼형 shard로 내려 전체 셀을 Python 객체로 쌓지 않는다.
                if columns["session"]:
                    pl.DataFrame(columns, schema=_SHARD_SCHEMA).write_parquet(shard_dir / f"shard-{offset:07d}.parquet")
                _LOG.info("stage=investor_flow_silver pages=%s/%s rows=%s", parsed_pages, len(page_paths), parsed_rows)
        raw_hashes.sort()
        raw_digest = hashlib.sha256("\n".join(raw_hashes).encode("utf-8")).hexdigest()
        dataset_id = "investor_flow_" + hashlib.sha256(
            "\n".join((
                POLICY_VERSION,
                str(policy.available_session_lag),
                policy.available_time.isoformat(),
                universe_dataset_id,
                *raw_hashes,
            )).encode("utf-8")
        ).hexdigest()[:16]
        target = silver_root / dataset_id
        key_schema = {"session": pl.Date, "ticker": pl.String}
        violations = pl.DataFrame(
            {"session": [date.fromisoformat(day) for day, _ in violation_keys], "ticker": [t for _, t in violation_keys]},
            schema=key_schema,
        )
        negatives = pl.DataFrame(
            {"session": [date.fromisoformat(day) for day, _ in negative_keys], "ticker": [t for _, t in negative_keys]},
            schema=key_schema,
        )
        lag = policy.available_session_lag
        availability = pl.DataFrame(
            {
                "session": list(calendar),
                "available_session": [calendar[i + lag] if i + lag < len(calendar) else None for i in range(len(calendar))],
            },
            schema={"session": pl.Date, "available_session": pl.Date},
        )
        shards = sorted(shard_dir.glob("*.parquet"))
        partitions: list[dict[str, Any]] = []
        conflict_cells = 0
        unavailable_tail_cells = 0
        retail_exceeds_volume_rows = 0
        negative_cells = 0
        total_rows = 0
        tickers: set[str] = set()
        for year in sorted({session.year for session in calendar}):
            year_negatives = negatives.filter(pl.col("session").dt.year() == year)
            if not shards:
                negative_cells += year_negatives.height
                continue
            cells = (
                pl.scan_parquet(shards)
                .filter(pl.col("session").dt.year() == year)
                .collect()
                .join(violations, on=["session", "ticker"], how="anti")
            )
            # 원본 행이 덮은 셀은 음성 기록으로 중복 계상하지 않는다(위반 셀은 덮은 것으로 보지 않음).
            negative_cells += year_negatives.join(cells.select("session", "ticker"), on=["session", "ticker"], how="anti").height
            grouped = (
                cells.sort("source_hash")
                .group_by(["session", "ticker"], maintain_order=True)
                .agg(pl.struct(list(_GROUP_CODES)).n_unique().alias("_distinct"), pl.all().first())
            )
            conflict_cells += grouped.filter(pl.col("_distinct") > 1).height
            resolved = grouped.filter(pl.col("_distinct") == 1).join(availability, on="session", how="left")
            unavailable_tail_cells += resolved.filter(pl.col("available_session").is_null()).height
            resolved = resolved.filter(pl.col("available_session").is_not_null())
            retail_exceeds_volume_rows += resolved.filter(pl.col("tjj0008").abs() > pl.col("ls_volume")).height
            part = resolved.select(
                "session",
                pl.concat_str(pl.lit("KRX:"), pl.col("ticker")).alias("instrument_id"),
                "ticker",
                pl.lit("LS").alias("provider"),
                pl.col("tjj0008").alias("individual_net_shares"),
                pl.col("tjj0016").alias("foreign_net_shares"),
                pl.col("tjj0018").alias("institution_net_shares"),
                pl.col("tjj0017").alias("other_net_shares"),
                *(pl.col(column.removesuffix("_net_shares")).alias(column) for column in _SUBGROUP_COLUMNS),
                "ls_close",
                "ls_volume",
                "ls_value_mkrw",
                pl.col("available_session")
                .dt.combine(policy.available_time)
                .dt.replace_time_zone(str(KRX_TZ))
                .alias("available_at"),
                "source_hash",
                pl.lit(POLICY_VERSION).alias("policy_version"),
            ).cast(pl.Schema(_SCHEMA)).sort(["session", "ticker"])
            if part.height == 0:
                continue
            rel = Path(f"year={year}") / "part.parquet"
            out_path = staging / rel
            out_path.parent.mkdir(parents=True)
            part.write_parquet(out_path)
            partitions.append({
                "path": str(rel),
                "row_count": part.height,
                "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                "year": year,
            })
            total_rows += part.height
            tickers.update(part["ticker"].unique().to_list())
        shutil.rmtree(shard_dir)
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
            "universe_dataset_id": universe_dataset_id,
            "raw_page_hashes_digest": raw_digest,
            "rows": total_rows,
            "tickers": len(tickers),
            "raw_pages": len(raw_hashes),
            "ignored_records_only_pages": ignored_records_only_pages,
            "negative_cells": negative_cells,
            "conflict_cells": conflict_cells,
            "identity_violation_cells": len(violation_keys),
            "identity_violation_keys": sorted(f"{day}:{ticker}" for day, ticker in violation_keys),
            "unavailable_tail_cells": unavailable_tail_cells,
            "retail_exceeds_volume_rows": retail_exceeds_volume_rows,
            "dateless_rows": dateless_rows,
            "partitions": partitions,
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target.exists():
            try:
                current = (target / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(f"existing investor-flow dataset is unreadable: {target}") from exc
            if current != encoded:
                raise PITDataError(f"existing investor-flow dataset differs: {target}")
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return InvestorFlowSilverResult(
        dataset_path=target,
        dataset_id=dataset_id,
        rows=total_rows,
        tickers=len(tickers),
        raw_pages=len(raw_hashes),
        ignored_records_only_pages=ignored_records_only_pages,
        negative_cells=negative_cells,
        conflict_cells=conflict_cells,
        identity_violation_cells=len(violation_keys),
        unavailable_tail_cells=unavailable_tail_cells,
        retail_exceeds_volume_rows=retail_exceeds_volume_rows,
        dateless_rows=dateless_rows,
    )
