"""Investor-flow target cells derived only from Silver, plus coverage diffing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import dataset_partition_paths
from src.data.receipt_catalog import CoverageRange, EvidenceStatus
from src.data.runtime import DataRuntime

__all__ = ["FlowTargets", "investor_flow_targets", "investor_flow_targets_from_ids", "pending_flow_cells"]

_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY})


@dataclass(frozen=True, slots=True)
class FlowTargets:
    """Investor-flow cells the research scope requires, pinned to their Silver inputs."""

    universe_dataset_id: str
    daily_market_dataset_id: str
    cells: pl.DataFrame  # columns: ticker (String), session (Date); unique, sorted by (ticker, session)


def _verified_partitions(dataset_dir: Path, *, label: str) -> list[str]:
    """Partition paths of one verified Silver dataset, failing closed otherwise."""
    from src.data.datasets import load_manifest

    try:
        load_manifest(dataset_dir)
    except PITDataError as exc:
        raise PITDataError(f"invalid {label} manifest: {dataset_dir}: {exc}") from exc
    try:
        paths = dataset_partition_paths(dataset_dir, allow_legacy=False)
    except PITDataError as exc:
        raise PITDataError(f"invalid {label} dataset: {dataset_dir}: {exc}") from exc
    if not paths:
        raise PITDataError(f"invalid {label} dataset: {dataset_dir}")
    return [str(path) for path in paths]


def investor_flow_targets_from_ids(
    silver_root: Path, universe_id: str, daily_id: str, *, since: date | None = None
) -> FlowTargets:
    """Cells for explicit Silver inputs, so phased builds use resolved ids."""
    silver = Path(silver_root)
    universe_files = _verified_partitions(silver / universe_id, label="ordinary-universe")
    daily_files = _verified_partitions(silver / daily_id, label="daily-market")
    try:
        eligible = (
            pl.scan_parquet(universe_files)
            .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String), pl.col("eligible"))
            .filter(pl.col("eligible"))
            .select("session", "ticker")
            .unique()
            .collect()
        )
        tradable = (
            pl.scan_parquet(daily_files)
            .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String), pl.col("price_state"))
            .filter(pl.col("price_state") == "tradable")
            .select("session", "ticker")
            .unique()
            .collect()
        )
    except Exception as exc:
        raise PITDataError(f"investor-flow Silver inputs are unreadable: {exc}") from exc
    cells = eligible.join(tradable, on=["session", "ticker"], how="inner").unique().sort(["ticker", "session"])
    if since is not None:
        cells = cells.filter(pl.col("session") >= since)
    return FlowTargets(universe_dataset_id=universe_id, daily_market_dataset_id=daily_id, cells=cells)


def investor_flow_targets(runtime: DataRuntime, *, since: date | None = None) -> FlowTargets:
    """Cells that need investor flow: eligible ordinary shares on sessions they were tradable.

    Derived only from Silver (the registry's current ``ordinary_universe`` and
    ``daily_market``), so collection never depends on a Gold dataset or on a
    hand-made requirements file.

    Raises:
        PITDataError: a registry dataset is missing or fails verification.
    """
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = registry.require("ordinary_universe")
    daily_id = registry.require("daily_market")
    return investor_flow_targets_from_ids(
        Path(runtime.workspace.silver_root), universe_id, daily_id, since=since
    )


def _merged_intervals(ranges: list[CoverageRange]) -> dict[str, tuple[list[date], list[date]]]:
    """Merge overlapping answered ranges per subject into disjoint ``(starts, ends)``."""
    by_subject: dict[str, list[tuple[date, date]]] = {}
    for item in ranges:
        if item.status not in _ANSWERED:
            continue
        by_subject.setdefault(item.subject, []).append((item.start, item.end))
    merged: dict[str, tuple[list[date], list[date]]] = {}
    for subject, spans in by_subject.items():
        spans.sort()
        starts: list[date] = []
        ends: list[date] = []
        for start, end in spans:
            if starts and start <= ends[-1]:
                if end > ends[-1]:
                    ends[-1] = end
            else:
                starts.append(start)
                ends.append(end)
        merged[subject] = (starts, ends)
    return merged


def pending_flow_cells(targets: pl.DataFrame, answered: Iterable[CoverageRange]) -> pl.DataFrame:
    """Target cells not inside any answered range of the same subject (ticker)."""
    import numpy as np

    frame = targets.select(pl.col("ticker").cast(pl.String), pl.col("session").cast(pl.Date)).unique().sort(["ticker", "session"])
    if frame.height == 0:
        return frame
    merged = _merged_intervals(list(answered))
    if not merged:
        return frame
    kept: list[pl.DataFrame] = []
    for part in frame.partition_by("ticker", maintain_order=True, as_dict=False):
        ticker = str(part["ticker"][0])
        spans = merged.get(ticker)
        if spans is None:
            kept.append(part)
            continue
        starts, ends = spans
        sessions = part["session"].to_list()
        bounds = np.array([day.toordinal() for day in starts])
        closes = np.array([day.toordinal() for day in ends])
        ordinals = np.array([day.toordinal() for day in sessions])
        position = np.searchsorted(bounds, ordinals, side="right") - 1
        covered = (position >= 0) & (ordinals <= closes[np.clip(position, 0, len(closes) - 1)])
        kept.append(part.filter(~covered))
    return pl.concat(kept, how="vertical").sort(["ticker", "session"])
