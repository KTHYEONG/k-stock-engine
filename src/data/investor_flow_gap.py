"""Coverage-requirement diff between the Gold market panel and LS Silver flow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from src.data.schemas import PITDataError


@dataclass(frozen=True, slots=True)
class InvestorFlowGapSymbol:
    ticker: str
    sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class MissingInvestorFlowCells:
    """The exact (ticker, session) cells a certified LS Silver dataset lacks.

    Attributes:
        market_panel_dataset_id: Gold panel that defines which cells require
            coverage (eligible, tradable sessions).
        ls_dataset_id: The LS investor-flow Silver dataset this gap is
            computed against. A supplement built from this result is only
            ever valid for exactly this LS dataset id.
        symbols: Per-ticker missing session lists, sorted by ticker.
        total_cells: Sum of every symbol's missing session count.
    """

    market_panel_dataset_id: str
    ls_dataset_id: str
    symbols: tuple[InvestorFlowGapSymbol, ...]
    total_cells: int


def _read_verified_partitions(dataset_dir: Path, *, label: str) -> tuple[str, list[str]]:
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        parts = manifest["partitions"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError(f"invalid {label} manifest: {dataset_dir}") from exc
    if manifest.get("dataset_id") != dataset_dir.name or not isinstance(parts, list) or not parts:
        raise PITDataError(f"invalid {label} manifest: {dataset_dir}")
    files: list[str] = []
    for part in parts:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("path"), str)
            or not isinstance(part.get("parquet_sha256"), str)
        ):
            raise PITDataError(f"invalid {label} partition: {dataset_dir}")
        target = dataset_dir / str(part["path"])
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise PITDataError(f"{label} partition is unreadable: {part['path']}") from exc
        if hashlib.sha256(data).hexdigest() != part["parquet_sha256"]:
            raise PITDataError(f"{label} partition hash mismatch: {part['path']}")
        files.append(str(target))
    return (manifest["dataset_id"], files)


def compute_missing_investor_flow_cells(
    *, market_panel_path: Path, ls_flow_silver_path: Path
) -> MissingInvestorFlowCells:
    """Diff the Gold market panel's coverage requirement against certified LS flow.

    A cell requires investor-flow coverage when its market-panel row is
    ``eligible`` and ``price_state == "tradable"``. The gap is exactly the
    requirement set minus the LS Silver dataset's ``(ticker, session)`` rows,
    computed directly from certified data rather than a historical collection
    plan, so it stays correct regardless of how that plan was built or has
    since drifted.

    Args:
        market_panel_path: Certified Gold ``market_panel_<id>`` directory.
        ls_flow_silver_path: Certified Silver ``investor_flow_<id>`` directory
            (the LS dataset from P0-3).

    Returns:
        The missing-cell set, pinned to both input dataset ids.

    Raises:
        PITDataError: either input's manifest or partition hashes fail
            certification.
    """
    panel_dir = Path(market_panel_path)
    ls_dir = Path(ls_flow_silver_path)
    panel_id, panel_files = _read_verified_partitions(panel_dir, label="market-panel")
    ls_id, ls_files = _read_verified_partitions(ls_dir, label="investor-flow")
    requirement = (
        pl.scan_parquet(panel_files)
        .select(["eligible", "price_state", "session", "instrument_id"])
        .filter(pl.col("eligible") & (pl.col("price_state") == "tradable"))
        .select(
            pl.col("session").cast(pl.Date),
            pl.col("instrument_id").cast(pl.String).str.split(":").list.last().alias("ticker"),
        )
        .unique()
        .collect()
    )
    coverage = (
        pl.scan_parquet(ls_files)
        .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String))
        .unique()
        .collect()
    )
    missing = requirement.join(coverage, on=["session", "ticker"], how="anti").sort(["ticker", "session"])
    grouped = missing.group_by("ticker", maintain_order=True).agg(pl.col("session").sort().alias("sessions"))
    symbols = tuple(
        InvestorFlowGapSymbol(ticker=str(ticker), sessions=tuple(sessions))
        for ticker, sessions in zip(
            grouped["ticker"].to_list(), grouped["sessions"].to_list(), strict=True
        )
    )
    return MissingInvestorFlowCells(
        market_panel_dataset_id=panel_id,
        ls_dataset_id=ls_id,
        symbols=symbols,
        total_cells=sum(len(symbol.sessions) for symbol in symbols),
    )
