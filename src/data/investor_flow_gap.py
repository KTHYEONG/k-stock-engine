"""Coverage-requirement diff between the Gold market panel and LS Silver flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from src.data.datasets import dataset_partition_paths, load_manifest
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


def _read_verified_partitions(
    dataset_dir: Path, *, label: str, expected_kind: str
) -> tuple[str, list[str]]:
    """Read either contract at the migration boundary and verify every hash."""

    try:
        paths = dataset_partition_paths(dataset_dir, allow_legacy=False)
    except PITDataError as exc:
        raise PITDataError(f"invalid {label} manifest: {dataset_dir}: {exc}") from exc
    if not paths:
        raise PITDataError(f"invalid {label} manifest: {dataset_dir}")
    try:
        manifest = load_manifest(dataset_dir)
    except PITDataError:
        manifest = None
    if manifest is not None and manifest.kind != expected_kind:
        raise PITDataError(f"invalid {label} kind: {dataset_dir}")
    return dataset_dir.name, [str(path) for path in paths]


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
    panel_id, panel_files = _read_verified_partitions(
        panel_dir, label="market-panel", expected_kind="market_panel"
    )
    ls_id, ls_files = _read_verified_partitions(
        ls_dir, label="investor-flow", expected_kind="investor_flow_ls"
    )
    panel_frames: list[pl.DataFrame] = []
    for file_path in panel_files:
        candidate = pl.read_parquet(file_path)
        if {"eligible", "price_state", "session", "instrument_id"}.issubset(candidate.columns):
            panel_frames.append(candidate)
    if not panel_frames:
        raise PITDataError("market panel has no dense partitions")
    requirement = (
        pl.concat(panel_frames, how="vertical_relaxed")
        .select(["eligible", "price_state", "session", "instrument_id"])
        .filter(pl.col("eligible") & (pl.col("price_state") == "tradable"))
        .select(
            pl.col("session").cast(pl.Date),
            pl.col("instrument_id").cast(pl.String).str.split(":").list.last().alias("ticker"),
        )
        .unique()
    )
    ls_frames: list[pl.DataFrame] = []
    for file_path in ls_files:
        candidate = pl.read_parquet(file_path)
        if {"session", "ticker"}.issubset(candidate.columns):
            ls_frames.append(candidate)
    if not ls_frames:
        raise PITDataError("LS flow has no dense partitions")
    coverage = (
        pl.concat(ls_frames, how="vertical_relaxed")
        .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String))
        .unique()
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
