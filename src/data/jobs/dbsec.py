"""DB Securities fallback collection jobs: ticker-master snapshot, daily charts, closing bars."""

from __future__ import annotations

import hashlib
import json
import re
import time as _time_module
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final

from src.config.providers import ProviderPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.core.time import KRX_TZ
from src.data.evidence_sources import (
    DBSEC_CLOSING_BAR_SOURCE,
    DBSEC_DAILY_CHART_SOURCE,
    DBSEC_TICKER_MASTER_SOURCE,
    raw_rows_envelope,
    source_contract,
)
from src.data.jobs.runner import JobContext, JobSpec, JobUnit
from src.data.payload_store import read_payload
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.dbsec.client import DbsecClient
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "DBSEC_JOBS",
    "DBSEC_QUOTA_PROVIDER",
    "DbsecClosingBarJob",
    "DbsecDailyChartJob",
    "DbsecTickerMasterJob",
    "build_dbsec_job_context",
    "dbsec_final_sessions",
    "dbsec_master_tickers",
    "resolve_dbsec_job",
]

DBSEC_QUOTA_PROVIDER: Final = "DBSEC"

_ANSWERED: Final = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY})
_TICKER_PATTERN: Final = re.compile(r"^[0-9A-Z]{6}$")
_CLOSING_HOUR: Final = "153000"
_HISTORY_WINDOW_DAYS: Final = 120
_NEXT_SESSION_PROBES: Final = (7, 30, 120)


def _kst_now(now: datetime) -> datetime:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return moment.astimezone(KRX_TZ)


def _parse_hhmm(text: str) -> time:
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", str(text))
    if match is None:
        raise PITDataError(f"invalid KST time {text!r}: expected HH:MM")
    return time(int(match.group(1)), int(match.group(2)))


def _xkrx_sessions(start: date, end: date) -> tuple[date, ...]:
    from src.core.krx_calendar import xkrx_session_calendar

    calendar = xkrx_session_calendar(start=start, end=end)
    return tuple(sorted({instant.astimezone(KRX_TZ).date() for instant in calendar.sessions}))


def dbsec_final_sessions(
    *,
    now: datetime,
    final_time_kst: time,
    lookback: int,
    calendar: Callable[[date, date], tuple[date, ...]] | None = None,
) -> tuple[date, ...]:
    """Last ``lookback`` XKRX sessions whose DB data is final at ``now`` (session day + ``final_time_kst``), ascending."""
    if isinstance(lookback, bool) or int(lookback) < 1:
        raise ValueError(f"invalid lookback {lookback!r}: must be a positive integer")
    moment = _kst_now(now)
    end = moment.date()
    # Calendar span scales with lookback so a large configured lookback is never silently truncated.
    window = timedelta(days=_HISTORY_WINDOW_DAYS + 2 * int(lookback))
    sessions = sorted(set((calendar or _xkrx_sessions)(end - window, end)))
    final = [day for day in sessions if moment >= datetime.combine(day, final_time_kst, KRX_TZ)]
    return tuple(final[-int(lookback) :])


def _next_session(day: date, *, calendar: Callable[[date, date], tuple[date, ...]] | None = None) -> date:
    probe = calendar or _xkrx_sessions
    start = day + timedelta(days=1)
    for span in _NEXT_SESSION_PROBES:
        try:
            sessions = probe(start, day + timedelta(days=span))
        except ValueError:
            continue
        later = sorted({session for session in sessions if session > day})
        if later:
            return later[0]
    raise PITDataError(f"no XKRX session after {day.isoformat()}")


def _master_envelope_bytes(ctx: JobContext, entry: ReceiptIndexEntry) -> bytes:
    try:
        raw = read_payload(Path(entry.payload_path))
    except (OSError, PITDataError) as exc:
        raise PITDataError(f"retained DB master payload is missing for {entry.natural_key}") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError(f"retained DB master payload hash mismatch for {entry.natural_key}")
    return raw


def dbsec_master_tickers(entry_payload: bytes, *, market_names: Collection[str]) -> tuple[str, ...]:
    """Sorted unique 6-character tickers of equity rows in one stored master envelope."""
    try:
        document = json.loads(entry_payload)
    except ValueError as exc:
        raise PITDataError("DB master payload is not valid JSON") from exc
    rows = document.get("rows") if isinstance(document, Mapping) else None
    if not isinstance(rows, list):
        raise PITDataError("DB master payload carries no rows list")
    names = set(market_names)
    tickers = {
        code
        for row in rows
        if isinstance(row, Mapping)
        and _TICKER_PATTERN.fullmatch(code := str(row.get("Iscd", "")).strip())
        and str(row.get("MrktClsName", "")).strip() in names
    }
    return tuple(sorted(tickers))


def _scoped_payload(
    *,
    kind: EvidenceKind,
    source: str,
    natural_key: str,
    session: date,
    query: Mapping[str, str],
    rows: Sequence[Mapping[str, object]],
    retrieved_at: datetime,
) -> ScopedRawPayload:
    contract = source_contract(source)
    payload = raw_rows_envelope(contract, query=query, rows=rows)
    return ScopedRawPayload(
        kind=kind,
        source=source,
        natural_key=natural_key,
        as_of=session,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS if rows else EvidenceStatus.EMPTY,
        payload=payload,
        retrieved_at=retrieved_at,
        source_label=f"dbsec:{source}:{natural_key}",
    )


def _in_scope_sessions(ctx: JobContext) -> tuple[date, ...]:
    policy = ctx.provider.dbsec
    sessions = dbsec_final_sessions(
        now=ctx.now(),
        final_time_kst=_parse_hhmm(policy.session_final_time_kst),
        lookback=int(policy.lookback_sessions),
    )
    return tuple(day for day in sessions if day >= ctx.runtime.scope.evidence_start)


class DbsecTickerMasterJob:
    """JCODES equity snapshot for the newest DB-final session, taken before that session's master window closes."""

    name = "dbsec_ticker_master"
    source = DBSEC_TICKER_MASTER_SOURCE
    kind = EvidenceKind.SECURITY_MASTER

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        sessions = _in_scope_sessions(ctx)
        if not sessions:
            return ()
        newest = sessions[-1]
        entry = ctx.catalog.latest(source=self.source, natural_keys={newest.isoformat()}).get(newest.isoformat())
        if entry is not None and entry.status in _ANSWERED:
            return ()
        valid_until = _parse_hhmm(ctx.provider.dbsec.master_valid_until_kst)
        deadline = datetime.combine(_next_session(newest), valid_until, KRX_TZ)
        if not _kst_now(ctx.now()) < deadline:
            return ()
        return (
            JobUnit(
                source=self.source,
                natural_key=newest.isoformat(),
                payload={"session": newest.isoformat()},
                max_requests=1,
            ),
        )

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            rows = [dict(row) for row in ctx.collector.fetch_ticker_master("J")]
            out.append(
                _scoped_payload(
                    kind=self.kind,
                    source=self.source,
                    natural_key=session.isoformat(),
                    session=session,
                    query={"market_div": "J", "session": session.isoformat()},
                    rows=rows,
                    retrieved_at=retrieved_at,
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


def _pending_bars(*, ctx: JobContext, source: str) -> tuple[JobUnit, ...]:
    sessions = _in_scope_sessions(ctx)
    if not sessions:
        return ()
    masters = ctx.catalog.latest(source=DBSEC_TICKER_MASTER_SOURCE, natural_keys={day.isoformat() for day in sessions})
    candidates: list[JobUnit] = []
    for day in sessions:
        entry = masters.get(day.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            continue
        tickers = dbsec_master_tickers(
            _master_envelope_bytes(ctx, entry), market_names=ctx.provider.dbsec.equity_market_names
        )
        candidates.extend(
            JobUnit(
                source=source,
                natural_key=f"{day.isoformat()}:{ticker}",
                payload={"session": day.isoformat(), "ticker": ticker},
                max_requests=1,
            )
            for ticker in tickers
        )
    if not candidates:
        return ()
    answered = ctx.catalog.latest(source=source, natural_keys={unit.natural_key for unit in candidates})
    return tuple(
        unit
        for unit in candidates
        if (entry := answered.get(unit.natural_key)) is None or entry.status not in _ANSWERED
    )


class DbsecDailyChartJob:
    """Raw daily-chart row per (session, ticker) for sessions that have a DB master snapshot."""

    name = "dbsec_daily_chart"
    source = DBSEC_DAILY_CHART_SOURCE
    kind = EvidenceKind.DAILY_MARKET

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        return _pending_bars(ctx=ctx, source=self.source)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            ticker = unit.payload["ticker"]
            stamp = session.strftime("%Y%m%d")
            rows = [
                dict(row)
                for row in ctx.collector.fetch_daily_chart(ticker, session, session, adjusted=False)
                if isinstance(row, Mapping) and str(row.get("Date", "")) == stamp
            ]
            out.append(
                _scoped_payload(
                    kind=self.kind,
                    source=self.source,
                    natural_key=unit.natural_key,
                    session=session,
                    query={"ticker": ticker, "session": session.isoformat(), "adjusted": "0"},
                    rows=rows,
                    retrieved_at=retrieved_at,
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


class DbsecClosingBarJob:
    """15:30 closing-auction 1-minute bar per (session, ticker) for sessions that have a DB master snapshot."""

    name = "dbsec_closing_bar"
    source = DBSEC_CLOSING_BAR_SOURCE
    kind = EvidenceKind.DAILY_MARKET

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        return _pending_bars(ctx=ctx, source=self.source)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        policy = ctx.provider.dbsec
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            ticker = unit.payload["ticker"]
            stamp = session.strftime("%Y%m%d")
            rows = [
                dict(row)
                for row in ctx.collector.fetch_minute_chart(ticker, session, bar_count=int(policy.closing_bar_count))
                if isinstance(row, Mapping)
                and str(row.get("Date", "")) == stamp
                and str(row.get("Hour", "")) == _CLOSING_HOUR
            ]
            if len(rows) > 1:
                raise PITDataError(f"duplicate 1530 closing bar for {unit.natural_key}")
            out.append(
                _scoped_payload(
                    kind=self.kind,
                    source=self.source,
                    natural_key=unit.natural_key,
                    session=session,
                    query={
                        "ticker": ticker,
                        "session": session.isoformat(),
                        "bar_count": str(int(policy.closing_bar_count)),
                        "row_filter": f"Date={stamp},Hour={_CLOSING_HOUR}",
                    },
                    rows=rows,
                    retrieved_at=retrieved_at,
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


DBSEC_JOBS: Mapping[str, JobSpec] = {
    DbsecTickerMasterJob.name: DbsecTickerMasterJob(),
    DbsecDailyChartJob.name: DbsecDailyChartJob(),
    DbsecClosingBarJob.name: DbsecClosingBarJob(),
}


def resolve_dbsec_job(name: str) -> JobSpec:
    """Return the DB job registered under ``name``; raises PITDataError when unknown."""
    try:
        return DBSEC_JOBS[str(name)]
    except KeyError:
        raise PITDataError(f"unknown DB job {name!r}: expected one of {sorted(DBSEC_JOBS)}") from None


def build_dbsec_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: DbsecClient | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one DB job without issuing any provider request."""
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=provider.runner("dbsec"),
        key_env=provider.dbsec.app_key_env,
        collector=collector,
        now=now or (lambda: datetime.now(UTC)),
        sleep=sleep or _time_module.sleep,
    )
