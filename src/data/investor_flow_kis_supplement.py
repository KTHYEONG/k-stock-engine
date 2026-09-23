"""Provider-tagged KIS Silver supplement covering exactly the LS coverage gap."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ
from src.data.investor_flow_gap import compute_missing_investor_flow_cells
from src.data.schemas import PITDataError

POLICY_VERSION = "kis-investor-trade-net-shares-supplement-v1"

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


def _parse_kis_page(
    path: Path, sessions: frozenset[date]
) -> tuple[str, str, list[tuple[date, tuple[int, int, int, int]]], int] | None:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.parent.name:
        raise PITDataError(f"investor-flow Bronze hash mismatch: {path}")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"invalid investor-flow Bronze JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid investor-flow Bronze root: {path}")
    if payload.get("provider") != "KIS":
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    query = payload.get("query")
    query_map = query if isinstance(query, dict) else {}
    symbol = str(query_map.get("symbol") or payload.get("symbol") or "").strip()
    if not symbol:
        raise PITDataError(f"KIS investor flow raw page lacks query symbol: {path}")
    parsed: list[tuple[date, tuple[int, int, int, int]]] = []
    out_of_calendar = 0
    for row in rows:
        if not isinstance(row, dict):
            raise PITDataError(f"KIS investor flow row must be an object: {path}")
        session = _parse_session(row.get("stck_bsop_date"))
        if session not in sessions:
            # KIS's anchor-walk returns a fixed 30-row window that can spill
            # past the certified range's edge (e.g. into the prior scope
            # year); this is normal pagination overflow, not a data defect.
            out_of_calendar += 1
            continue
        quantities = (
            _parse_qty(row, "prsn_ntby_qty"),
            _parse_qty(row, "frgn_ntby_qty"),
            _parse_qty(row, "orgn_ntby_qty"),
            _parse_qty(row, "etc_ntby_qty"),
        )
        parsed.append((session, quantities))
    return (path.parent.name, symbol, parsed, out_of_calendar)


def _load_panel_calendar(market_panel_path: Path) -> tuple[date, ...]:
    # The panel was just fully hash-verified inside compute_missing_investor_flow_cells
    # for these same inputs, so only its certified session list is re-read here.
    manifest = json.loads((market_panel_path / "manifest.json").read_text(encoding="utf-8"))
    files = [str(market_panel_path / str(part["path"])) for part in manifest["partitions"]]
    return tuple(
        pl.scan_parquet(files)
        .select(pl.col("session").cast(pl.Date))
        .unique()
        .collect()
        .sort("session")["session"]
        .to_list()
    )


def materialize_investor_flow_kis_supplement(
    *,
    bronze_root: Path,
    market_panel_path: Path,
    ls_flow_silver_path: Path,
    silver_root: Path,
    policy: InvestorFlowKisSupplementPolicy = InvestorFlowKisSupplementPolicy(),  # noqa: B008 - spec-mandated immutable default
) -> InvestorFlowKisSupplementResult:
    """Build a provider-tagged Silver supplement for exactly the LS coverage gap.

    Values come only from hash-verified raw KIS ``rows`` (never the collector's
    derived ``records``), mapped through the four-group net-share identity
    verified for KIS's ``prsn/frgn/orgn/etc`` fields. A row is written only
    when its ``(ticker, session)`` is in the target gap computed from
    :func:`compute_missing_investor_flow_cells`; any KIS Bronze evidence
    outside that set (for example an anchor page's incidental overlap with a
    session LS already covers) is read but never written, so this dataset can
    never conflict with the LS dataset it supplements.

    Args:
        bronze_root: Scope Bronze root containing ``investor_flow/<sha256>/``
            KIS pages (``provider == "KIS"``).
        market_panel_path: Certified Gold market panel (defines the gap).
        ls_flow_silver_path: Certified LS Silver investor-flow dataset (defines
            the gap and is recorded as this dataset's supplemented target).
        silver_root: Scope Silver root receiving
            ``investor_flow_kis_supplement_<hash16>/``.
        policy: Availability contract.

    Returns:
        Coverage counts: how much of the target gap this run's KIS evidence
        actually filled, and how much is still missing.

    Raises:
        PITDataError: payload hash mismatch, malformed raw row, non-integral
            quantity, or an existing dataset with different content. A row
            whose session falls outside the certified calendar is dropped,
            not raised — KIS's fixed anchor-walk window legitimately spills
            past the requested range's edge.
    """
    gap = compute_missing_investor_flow_cells(
        market_panel_path=Path(market_panel_path), ls_flow_silver_path=Path(ls_flow_silver_path)
    )
    target = {(session, symbol.ticker) for symbol in gap.symbols for session in symbol.sessions}
    calendar = _load_panel_calendar(Path(market_panel_path))
    session_set = frozenset(calendar)
    page_paths = sorted((Path(bronze_root) / "investor_flow").glob("*/payload.json"))
    silver_root = Path(silver_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".investor-flow-kis-supplement-", dir=silver_root))
    try:
        options: dict[tuple[date, str], dict[tuple[int, int, int, int], str]] = {}
        violation_keys: set[tuple[date, str]] = set()
        kis_hashes: list[str] = []
        parsed_pages = 0
        out_of_calendar_rows = 0
        for offset in range(0, len(page_paths), _PAGE_BATCH):
            for path in page_paths[offset : offset + _PAGE_BATCH]:
                parsed_pages += 1
                kept = _parse_kis_page(path, session_set)
                if kept is None:
                    continue
                page_hash, symbol, parsed_rows, page_out_of_calendar = kept
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
                "stage=investor_flow_kis_supplement pages=%s/%s filled=%s out_of_calendar_rows=%s",
                parsed_pages,
                len(page_paths),
                len(options),
                out_of_calendar_rows,
            )
        kis_hashes.sort()
        dataset_id = "investor_flow_kis_supplement_" + hashlib.sha256(
            "\n".join((
                POLICY_VERSION,
                str(policy.available_session_lag),
                policy.available_time.isoformat(),
                gap.ls_dataset_id,
                gap.market_panel_dataset_id,
                *kis_hashes,
            )).encode("utf-8")
        ).hexdigest()[:16]
        target_path = silver_root / dataset_id
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
                pl.lit(POLICY_VERSION).alias("policy_version"),
            )
            .cast(pl.Schema(_SCHEMA))
            .sort(["session", "ticker"])
        )
        partitions: list[dict[str, Any]] = []
        total_rows = 0
        for year in sorted(frame["session"].dt.year().unique().to_list()):
            part = frame.filter(pl.col("session").dt.year() == year)
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
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "available_session_lag": policy.available_session_lag,
            "available_time": policy.available_time.isoformat(),
            "ls_dataset_id": gap.ls_dataset_id,
            "market_panel_dataset_id": gap.market_panel_dataset_id,
            "target_cells": len(target),
            "filled_cells": total_rows,
            "still_missing_cells": len(target) - total_rows,
            "identity_violation_cells": len(violation_keys),
            "rows": total_rows,
            "partitions": partitions,
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target_path.exists():
            try:
                current = (target_path / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(
                    f"existing investor-flow KIS supplement is unreadable: {target_path}"
                ) from exc
            if current != encoded:
                raise PITDataError(
                    f"existing investor-flow KIS supplement differs: {target_path}"
                )
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return InvestorFlowKisSupplementResult(
        dataset_path=target_path,
        dataset_id=dataset_id,
        ls_dataset_id=gap.ls_dataset_id,
        target_cells=len(target),
        filled_cells=total_rows,
        still_missing_cells=len(target) - total_rows,
        identity_violation_cells=len(violation_keys),
        rows=total_rows,
    )
