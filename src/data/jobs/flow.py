"""Range-planned investor-flow collection jobs over Silver-derived targets."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config.providers import ProviderPolicy
from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.datasets import DatasetManifest, dataset_partition_paths, load_manifest
from src.data.evidence_sources import KIS_FLOW_SOURCE, LS_FLOW_SOURCE, raw_rows_envelope, source_contract
from src.data.flow_targets import investor_flow_targets, pending_flow_cells
from src.data.investor_flow_silver import flow_tail_sessions
from src.data.jobs.runner import JobContext, JobUnit
from src.data.receipt_catalog import CoverageRange, EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRangePayload
from src.integrations.kis.investor_flow import KisInvestorFlowCollector
from src.integrations.ls.investor_flow import LsInvestorFlowCollector
from src.integrations.quota import ProviderQuotaStateStore

_LOG = logging.getLogger(__name__)

__all__ = [
    "KIS_CHUNK_SIZE",
    "LS_CHUNK_SIZE",
    "KisInvestorFlowJob",
    "LsInvestorFlowJob",
    "build_kis_job_context",
    "build_ls_job_context",
]

LS_CHUNK_SIZE = 10
KIS_CHUNK_SIZE = 5

_KIS_ENDPOINT = "investor-trade-by-stock-daily"
_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY})


def _universe_calendar(ctx: JobContext) -> tuple[date, ...]:
    """Ordered scope sessions backing calendar-position windows."""
    from src.data.datasets import universe_sessions

    registry = DatasetRegistry(ctx.runtime.workspace.state_root)
    universe_id = registry.require("ordinary_universe")
    _, calendar = universe_sessions(ctx.runtime.workspace.silver_root, universe_id, allow_legacy=False)
    if not calendar:
        raise PITDataError("ordinary universe has no sessions")
    return calendar


def _calendar_span_groups(
    sessions: Sequence[date], *, position: Mapping[date, int], span: int
) -> tuple[tuple[date, date], ...]:
    """Greedy cover of sorted sessions by windows whose calendar-position span is below ``span``."""
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise PITDataError("calendar span must be a positive integer")
    if not sessions:
        return ()
    groups: list[tuple[date, date]] = []
    group_first = sessions[0]
    try:
        first_index = position[group_first]
    except KeyError:
        raise PITDataError(
            f"investor-flow target session is outside the universe calendar: {group_first}"
        ) from None
    group_last = group_first
    for session in sessions[1:]:
        try:
            index = position[session]
        except KeyError:
            raise PITDataError(
                f"investor-flow target session is outside the universe calendar: {session}"
            ) from None
        if index - first_index < span:
            group_last = session
        else:
            groups.append((group_first, group_last))
            group_first = session
            group_last = session
            first_index = index
    groups.append((group_first, group_last))
    return tuple(groups)


def _ls_units(
    *, pending: pl.DataFrame, calendar: tuple[date, ...], limit: int
) -> tuple[JobUnit, ...]:
    """One unit per ticker per pending-session window spanning at most ``limit`` calendar sessions."""
    position = {session: index for index, session in enumerate(calendar)}
    units: list[JobUnit] = []
    for part in pending.partition_by("ticker", maintain_order=True, as_dict=False):
        ticker = str(part["ticker"][0])
        sessions = sorted(part["session"].to_list())
        for start, end in _calendar_span_groups(sessions, position=position, span=limit):
            units.append(
                JobUnit(
                    source=LS_FLOW_SOURCE,
                    natural_key=f"{ticker}:{start.isoformat()}:{end.isoformat()}",
                    payload={"symbol": ticker, "start": start.isoformat(), "end": end.isoformat()},
                    max_requests=1,
                )
            )
    return tuple(units)


class LsInvestorFlowJob:
    """Request LS t1702 windows for target cells not yet answered by LS.

    One unit is one (ticker, window) whose bounds are pending sessions and whose
    span is at most ``ls.max_sessions_per_request`` calendar sessions.
    The natural key is ``<ticker>:<start>:<end>``.
    """

    name = "ls_investor_flow"
    fail_fast = True

    def __init__(self, *, since: date | None = None) -> None:
        self.since = since

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        targets = investor_flow_targets(ctx.runtime, since=self.since)
        answered = ctx.catalog.ranges(source=LS_FLOW_SOURCE)
        pending = pending_flow_cells(targets.cells, answered)
        if pending.height == 0:
            return ()
        return _ls_units(
            pending=pending,
            calendar=_universe_calendar(ctx),
            limit=ctx.provider.ls.max_sessions_per_request,
        )

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRangePayload]:
        contract = source_contract(LS_FLOW_SOURCE)
        retrieved_at = ctx.now()
        out: list[ScopedRangePayload] = []
        for unit in units:
            symbol = unit.payload["symbol"]
            start = date.fromisoformat(unit.payload["start"])
            end = date.fromisoformat(unit.payload["end"])
            response = ctx.collector.fetch(symbol, start, end)
            if response.rows:
                envelope = raw_rows_envelope(
                    contract, query=dict(response.query), rows=[dict(row) for row in response.rows]
                )
                ranges = (
                    CoverageRange(
                        source=LS_FLOW_SOURCE, subject=symbol, start=start, end=end,
                        status=EvidenceStatus.SUCCESS, content_hash=None, retrieved_at=retrieved_at,
                    ),
                )
                out.append(
                    ScopedRangePayload(
                        source=LS_FLOW_SOURCE, payload=envelope, ranges=ranges,
                        retrieved_at=retrieved_at,
                        source_label=f"LS:{contract.endpoint_label()}:{symbol}:{end.isoformat()}",
                    )
                )
            else:
                ranges = (
                    CoverageRange(
                        source=LS_FLOW_SOURCE, subject=symbol, start=start, end=end,
                        status=EvidenceStatus.EMPTY, content_hash=None, retrieved_at=retrieved_at,
                    ),
                )
                out.append(
                    ScopedRangePayload(
                        source=LS_FLOW_SOURCE, payload=None, ranges=ranges,
                        retrieved_at=retrieved_at,
                        source_label=f"LS:{contract.endpoint_label()}:{symbol}:{end.isoformat()}",
                    )
                )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _kis_session(value: Any) -> date | None:
    """Parse one KIS row session, returning ``None`` for undecipherable values."""
    text = str(value).strip().replace("/", "-")
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _read_ls_valued_cells(*, silver_root: Path, dataset_id: str) -> pl.DataFrame:
    """Cells the current LS Silver dataset already values."""
    dataset_dir = Path(silver_root) / dataset_id
    try:
        paths = dataset_partition_paths(dataset_dir, allow_legacy=False)
    except PITDataError as exc:
        raise PITDataError(f"invalid investor_flow_ls dataset: {dataset_dir}: {exc}") from exc
    if not paths:
        raise PITDataError(f"invalid investor_flow_ls dataset: {dataset_dir}")
    try:
        return (
            pl.scan_parquet([str(path) for path in paths])
            .select(pl.col("session").cast(pl.Date), pl.col("ticker").cast(pl.String))
            .unique()
            .collect()
        )
    except Exception as exc:
        raise PITDataError(f"investor_flow_ls dataset is unreadable: {dataset_id}: {exc}") from exc


def _kis_windows(
    *, pending: pl.DataFrame, calendar: tuple[date, ...], rows_per_page: int
) -> dict[str, tuple[tuple[date, date], ...]]:
    """Group each ticker's pending cells into windows spanning few calendar sessions."""
    if (
        isinstance(rows_per_page, bool)
        or not isinstance(rows_per_page, int)
        or rows_per_page < 1
    ):
        raise PITDataError("investor_flow_rows_per_page must be a positive integer")
    position = {session: index for index, session in enumerate(calendar)}
    windows: dict[str, tuple[tuple[date, date], ...]] = {}
    for part in pending.partition_by("ticker", maintain_order=True, as_dict=False):
        ticker = str(part["ticker"][0])
        sessions = sorted(part["session"].to_list())
        windows[ticker] = _calendar_span_groups(sessions, position=position, span=rows_per_page)
    return windows


def _ls_availability_lag(manifest: DatasetManifest) -> int:
    """The ``available_session_lag`` the LS Silver dataset was built with."""
    lag = manifest.params.get("available_session_lag")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise PITDataError("investor_flow_ls manifest lacks a valid available_session_lag")
    return lag


class KisInvestorFlowJob:
    """Request KIS pages for target cells LS left without a value, deferring cells not yet available."""

    name = "kis_investor_flow"
    fail_fast = True

    def __init__(self, *, since: date | None = None) -> None:
        self.since = since
        self.deferred_tail_cells: int = 0

    def _pending_windows(self, ctx: JobContext) -> dict[str, tuple[tuple[date, date], ...]]:
        targets = investor_flow_targets(ctx.runtime, since=self.since)
        registry = DatasetRegistry(ctx.runtime.workspace.state_root)
        ls_id = registry.require("investor_flow_ls")
        try:
            manifest = load_manifest(ctx.runtime.workspace.silver_root / ls_id)
        except PITDataError as exc:
            raise PITDataError(f"invalid investor_flow_ls manifest: {ls_id}: {exc}") from exc
        if manifest.inputs.get("bronze_flow") != ctx.catalog.blob_digest(source=LS_FLOW_SOURCE):
            raise PITDataError("investor_flow_ls is stale")
        valued = _read_ls_valued_cells(silver_root=ctx.runtime.workspace.silver_root, dataset_id=ls_id)
        missing = targets.cells.join(valued, on=["session", "ticker"], how="anti").sort(["ticker", "session"])
        lag = _ls_availability_lag(manifest)
        calendar = _universe_calendar(ctx)
        tail = flow_tail_sessions(calendar, lag=lag)
        deferred = missing.filter(pl.col("session").is_in(sorted(tail)))
        missing = missing.filter(~pl.col("session").is_in(sorted(tail)))
        self.deferred_tail_cells = deferred.height
        _LOG.info(
            "[DATA] job=kis_investor_flow phase=defer deferred_tail_cells=%d tail_sessions=%s",
            self.deferred_tail_cells,
            sorted(day.isoformat() for day in tail),
        )
        pending = pending_flow_cells(missing, ctx.catalog.ranges(source=KIS_FLOW_SOURCE))
        if pending.height == 0:
            return {}
        return _kis_windows(
            pending=pending,
            calendar=calendar,
            rows_per_page=ctx.provider.kis.investor_flow_rows_per_page,
        )

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        windows_by_ticker = self._pending_windows(ctx)
        units: list[JobUnit] = []
        for ticker in sorted(windows_by_ticker):
            windows = windows_by_ticker[ticker]
            units.append(
                JobUnit(
                    source=KIS_FLOW_SOURCE,
                    natural_key=ticker,
                    payload={
                        "symbol": ticker,
                        "windows": json.dumps(
                            [[start.isoformat(), anchor.isoformat()] for start, anchor in windows]
                        ),
                    },
                    max_requests=len(windows),
                )
            )
        return tuple(units)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRangePayload]:
        contract = source_contract(KIS_FLOW_SOURCE)
        retrieved_at = ctx.now()
        out: list[ScopedRangePayload] = []
        for unit in units:
            symbol = unit.payload["symbol"]
            try:
                windows = [
                    (date.fromisoformat(start_raw), date.fromisoformat(anchor_raw))
                    for start_raw, anchor_raw in json.loads(unit.payload["windows"])
                ]
            except (ValueError, TypeError) as exc:
                raise PITDataError(f"invalid KIS flow unit windows for {symbol!r}") from exc
            for window_start, anchor in windows:
                ctx.quota_store.record_attempt(
                    provider=ctx.runner.quota_provider, endpoint=_KIS_ENDPOINT,
                    now=ctx.now(), daily_limit=ctx.runner.daily_budget,
                )
                response = ctx.collector.fetch(symbol, anchor)
                if response.rows:
                    row_sessions = [
                        session
                        for session in (_kis_session(row.get("stck_bsop_date")) for row in response.rows)
                        if session is not None
                    ]
                    earliest = min(row_sessions) if row_sessions else anchor
                    # A KIS page reaches back a fixed row count, so early anchors return pre-scope rows;
                    # the answered range may only claim sessions inside the scope.
                    earliest = max(earliest, ctx.runtime.scope.evidence_start)
                    envelope = raw_rows_envelope(
                        contract, query=dict(response.query), rows=[dict(row) for row in response.rows]
                    )
                    ranges = (
                        CoverageRange(
                            source=KIS_FLOW_SOURCE, subject=symbol, start=earliest, end=anchor,
                            status=EvidenceStatus.SUCCESS, content_hash=None, retrieved_at=retrieved_at,
                        ),
                    )
                    out.append(
                        ScopedRangePayload(
                            source=KIS_FLOW_SOURCE, payload=envelope, ranges=ranges,
                            retrieved_at=retrieved_at,
                            source_label=f"KIS:{contract.endpoint_label()}:{symbol}:{anchor.isoformat()}",
                        )
                    )
                else:
                    ranges = (
                        CoverageRange(
                            source=KIS_FLOW_SOURCE, subject=symbol, start=window_start, end=anchor,
                            status=EvidenceStatus.EMPTY, content_hash=None, retrieved_at=retrieved_at,
                        ),
                    )
                    out.append(
                        ScopedRangePayload(
                            source=KIS_FLOW_SOURCE, payload=None, ranges=ranges,
                            retrieved_at=retrieved_at,
                            source_label=f"KIS:{contract.endpoint_label()}:{symbol}:{anchor.isoformat()}",
                        )
                    )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _flow_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: LsInvestorFlowCollector | KisInvestorFlowCollector | None,
    runner_provider: str,
    key_env: str,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=provider.runner(runner_provider),
        key_env=key_env,
        collector=collector,
        now=now or (lambda: datetime.now(UTC)),
        sleep=sleep or time.sleep,
    )


def build_ls_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: LsInvestorFlowCollector | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one LS job without issuing any provider request."""
    return _flow_job_context(
        runtime=runtime, provider=provider, collector=collector,
        runner_provider="ls", key_env=provider.ls.app_key_env, now=now, sleep=sleep,
    )


def build_kis_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: KisInvestorFlowCollector | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one KIS job without issuing any provider request."""
    return _flow_job_context(
        runtime=runtime, provider=provider, collector=collector,
        runner_provider="kis", key_env=provider.kis.app_key_env, now=now, sleep=sleep,
    )
