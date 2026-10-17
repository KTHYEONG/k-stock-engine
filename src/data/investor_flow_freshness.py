"""Fail-closed freshness gate for previous-session investor flow at seal time."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl

from src.config.live_gate import LiveGatePolicy
from src.core.pit import PITDataError
from src.data.datasets import read_dataset
from src.data.evidence_sources import KIS_FLOW_SOURCE, LS_FLOW_SOURCE
from src.data.flow_targets import investor_flow_targets
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime

__all__ = ["FlowFreshness", "StaleInvestorFlowError", "check_investor_flow_freshness", "investor_flow_freshness"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FlowFreshness:
    """Coverage of one flow source session against its required cells at seal time."""

    source_session: date
    required_cells: int
    present_cells: int
    late_cells: int
    missing_cells: int
    missing_fraction: float


class StaleInvestorFlowError(PITDataError):
    """Required previous-session investor flow is missing or landed after the seal time."""

    reason: ClassVar[str] = "stale_investor_flow"

    def __init__(self, freshness: Sequence[FlowFreshness]) -> None:
        self.freshness: tuple[FlowFreshness, ...] = tuple(freshness)
        failing = ", ".join(
            f"{item.source_session.isoformat()}:{item.missing_fraction:.6f}"
            for item in self.freshness
            if item.missing_cells > 0
        )
        super().__init__(f"stale investor flow for {failing or 'no session'}")


def investor_flow_freshness(
    *,
    targets: pl.DataFrame,
    flow: pl.LazyFrame,
    retrieved_at: Mapping[str, datetime],
    source_sessions: Sequence[date],
    seal_time: datetime,
) -> tuple[FlowFreshness, ...]:
    """Per source session, required target cells vs union rows whose Bronze page was retrieved at or before ``seal_time``."""
    if seal_time.tzinfo is None:
        raise ValueError("seal_time must be timezone-aware")
    wanted = list(source_sessions)
    required = (
        targets.lazy()
        .filter(pl.col("session").is_in(wanted))
        .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String))
        .unique()
        .collect()
    )
    landed = (
        flow.filter(pl.col("session").is_in(wanted))
        .select(
            pl.col("session").cast(pl.Date),
            pl.col("ticker").cast(pl.String),
            pl.col("source_hash").cast(pl.String, strict=False),
        )
        .group_by(["session", "ticker"])
        .agg(pl.col("source_hash").unique().alias("hashes"))
        .collect()
    )
    on_time: dict[tuple[date, str], bool] = {}
    has_row: set[tuple[date, str]] = set()
    for session, ticker, hashes in landed.select("session", "ticker", "hashes").iter_rows():
        key = (session, ticker)
        has_row.add(key)
        on_time[key] = any(
            (seen := retrieved_at.get(str(item))) is not None
            and (seen if seen.tzinfo is not None else seen.replace(tzinfo=UTC)) <= seal_time
            for item in (hashes or [])
        )
    out: list[FlowFreshness] = []
    for session in wanted:
        tickers = required.filter(pl.col("session") == session)["ticker"].to_list()
        need = len(tickers)
        present = sum(1 for ticker in tickers if on_time.get((session, ticker), False))
        late = sum(1 for ticker in tickers if (session, ticker) in has_row and not on_time.get((session, ticker), False))
        missing = need - present
        out.append(
            FlowFreshness(
                source_session=session,
                required_cells=need,
                present_cells=present,
                late_cells=late,
                missing_cells=missing,
                missing_fraction=(missing / need) if need else 0.0,
            )
        )
    return tuple(out)


def check_investor_flow_freshness(
    *,
    runtime: DataRuntime,
    flow_dataset_dir: Path,
    source_sessions: Sequence[date],
    seal_time: datetime,
    policy: LiveGatePolicy,
) -> tuple[FlowFreshness, ...]:
    """Fail closed when any source session's missing fraction exceeds ``policy.max_missing_flow_fraction``."""
    wanted = list(source_sessions)
    flow = read_dataset(Path(flow_dataset_dir), columns=("session", "ticker", "source_hash"))
    needed = {
        str(item)
        for item in flow.filter(pl.col("session").is_in(wanted))
        .select(pl.col("source_hash").cast(pl.String, strict=False))
        .unique()
        .collect()["source_hash"]
        .to_list()
        if item is not None
    }
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    retrieved_at: dict[str, datetime] = {}
    for source in (LS_FLOW_SOURCE, KIS_FLOW_SOURCE):
        for blob in catalog.blobs(source=source):
            if blob.content_hash not in needed:
                continue
            seen = retrieved_at.get(blob.content_hash)
            if seen is None or blob.retrieved_at < seen:
                retrieved_at[blob.content_hash] = blob.retrieved_at
    freshness = investor_flow_freshness(
        targets=investor_flow_targets(runtime).cells,
        flow=flow,
        retrieved_at=retrieved_at,
        source_sessions=wanted,
        seal_time=seal_time,
    )
    for item in freshness:
        _LOG.info(
            "[DATA] gate=investor_flow source_session=%s required=%d present=%d late=%d missing_fraction=%.6f",
            item.source_session.isoformat(),
            item.required_cells,
            item.present_cells,
            item.late_cells,
            item.missing_fraction,
        )
    if any(item.missing_fraction > policy.max_missing_flow_fraction for item in freshness):
        raise StaleInvestorFlowError(freshness)
    return freshness
