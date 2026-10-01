"""KRX scoped collection jobs: daily-market pages and security-master snapshots.

Each job covers one unit per completed XKRX session from the scope's evidence
start to the last completed session at run time. A session is complete only
after 18:00 KST of that day, so intraday partial data can never enter Bronze.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from src.config.providers import ProviderPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.core.time import KRX_TZ
from src.data.evidence_sources import (
    KRX_DAILY_MARKET_SOURCE,
    KRX_HEDGE_SERIES_SOURCE,
    KRX_SECURITY_MASTER_SOURCE,
)
from src.data.jobs.runner import JobContext, JobSpec, JobUnit
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.krx.client import KrxApiClient
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "KRX_DAILY_MARKET_SOURCE",
    "KRX_HEDGE_SERIES_SOURCE",
    "KRX_JOBS",
    "KRX_QUOTA_PROVIDER",
    "KRX_SECURITY_MASTER_SOURCE",
    "KrxDailyMarketJob",
    "KrxHedgeSeriesJob",
    "KrxSecurityMasterJob",
    "build_krx_job_context",
    "completed_sessions",
    "krx_daily_market_scoped_payload",
    "krx_hedge_series_scoped_payload",
    "krx_security_master_scoped_payload",
    "last_completed_session_day",
    "resolve_krx_job",
]

KRX_QUOTA_PROVIDER = "KRX"

_SESSION_CLOSE_HOUR_KST = 18
_MAX_REQUESTS_PER_SESSION = 2
KRX_CHUNK_SIZE = 5
_KST = timedelta(hours=9)

_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED})


def _kst_now(now: datetime) -> datetime:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return moment.astimezone(UTC) + _KST


def last_completed_session_day(now: datetime) -> date:
    """Last session whose trading day is complete (18:00 KST cutoff)."""
    kst = _kst_now(now)
    today = kst.date()
    if (kst.hour, kst.minute) >= (_SESSION_CLOSE_HOUR_KST, 0):
        return today
    return today - timedelta(days=1)


def _xkrx_sessions(start: date, end: date) -> tuple[date, ...]:
    from src.core.krx_calendar import xkrx_session_calendar

    calendar = xkrx_session_calendar(start=start, end=end)
    return tuple(sorted({instant.astimezone(KRX_TZ).date() for instant in calendar.sessions}))


def completed_sessions(
    *,
    evidence_start: date,
    now: datetime,
    calendar: Callable[[date, date], tuple[date, ...]] | None = None,
) -> tuple[date, ...]:
    """Completed XKRX sessions from ``evidence_start`` through the last complete day."""
    last = last_completed_session_day(now)
    if last < evidence_start:
        return ()
    sessions = (calendar or _xkrx_sessions)(evidence_start, last)
    return tuple(day for day in sorted(set(sessions)) if day <= last)


def krx_daily_market_scoped_payload(
    *, records: Sequence[Mapping[str, Any]], session: date, retrieved_at: datetime
) -> ScopedRawPayload:
    """Convert one validated KRX daily-market page to a scoped payload."""
    body = json.dumps(
        {"session": session.isoformat(), "records": [dict(record) for record in records]},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return ScopedRawPayload(
        kind=EvidenceKind.DAILY_MARKET,
        source=KRX_DAILY_MARKET_SOURCE,
        natural_key=session.isoformat(),
        as_of=session,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"krx:daily-market:{session.isoformat()}",
    )


def krx_security_master_scoped_payload(
    *, records: Sequence[Mapping[str, Any]], session: date, retrieved_at: datetime
) -> ScopedRawPayload:
    """Convert one validated KRX security-master page to a scoped payload."""
    body = json.dumps(
        {"session": session.isoformat(), "records": [dict(record) for record in records]},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return ScopedRawPayload(
        kind=EvidenceKind.SECURITY_MASTER,
        source=KRX_SECURITY_MASTER_SOURCE,
        natural_key=session.isoformat(),
        as_of=session,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"krx:security-master:{session.isoformat()}",
    )


def krx_hedge_series_scoped_payload(
    *, records: Sequence[Mapping[str, Any]], session: date, retrieved_at: datetime
) -> ScopedRawPayload:
    """Convert one validated KRX hedge-series page to a scoped payload."""
    body = json.dumps(
        {"session": session.isoformat(), "records": [dict(record) for record in records]},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return ScopedRawPayload(
        kind=EvidenceKind.DAILY_MARKET,
        source=KRX_HEDGE_SERIES_SOURCE,
        natural_key=session.isoformat(),
        as_of=session,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"krx:hedge-series:{session.isoformat()}",
    )


def _krx_empty_scoped_payload(
    *, kind: EvidenceKind, source: str, session: date, retrieved_at: datetime
) -> ScopedRawPayload:
    """Record a provider-answered empty session as ``empty``, not as missing."""
    body = json.dumps({"session": session.isoformat(), "records": []}, sort_keys=True).encode("utf-8")
    return ScopedRawPayload(
        kind=kind,
        source=source,
        natural_key=session.isoformat(),
        as_of=session,
        fiscal_period=None,
        status=EvidenceStatus.EMPTY,
        payload=body,
        retrieved_at=retrieved_at,
        source_label=f"krx:{source}:{session.isoformat()}",
    )


def _pending_sessions(*, source: str, ctx: JobContext) -> Sequence[JobUnit]:
    """Completed sessions without an answered catalog receipt under ``source``."""
    sessions = completed_sessions(evidence_start=ctx.runtime.scope.evidence_start, now=ctx.now())
    if not sessions:
        return ()
    answered = ctx.catalog.latest(source=source, natural_keys={day.isoformat() for day in sessions})
    units: list[JobUnit] = []
    for day in sessions:
        entry = answered.get(day.isoformat())
        if entry is not None and entry.status in _ANSWERED:
            continue
        units.append(
            JobUnit(
                source=source,
                natural_key=day.isoformat(),
                payload={"session": day.isoformat()},
                max_requests=_MAX_REQUESTS_PER_SESSION,
            )
        )
    return units


class KrxDailyMarketJob:
    """Daily trade pages for every completed session."""

    name = "krx_daily_market"
    source = KRX_DAILY_MARKET_SOURCE
    kind = EvidenceKind.DAILY_MARKET

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        return _pending_sessions(source=self.source, ctx=ctx)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            records = list(ctx.collector.fetch_daily_records(session))
            if not records:
                out.append(
                    _krx_empty_scoped_payload(
                        kind=self.kind, source=self.source, session=session, retrieved_at=retrieved_at
                    )
                )
                continue
            out.append(
                krx_daily_market_scoped_payload(records=records, session=session, retrieved_at=retrieved_at)
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


class KrxSecurityMasterJob:
    """Security-master snapshots for every completed session."""

    name = "krx_security_master"
    source = KRX_SECURITY_MASTER_SOURCE
    kind = EvidenceKind.SECURITY_MASTER

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        return _pending_sessions(source=self.source, ctx=ctx)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            records = list(ctx.collector.fetch_master_records(session))
            if not records:
                out.append(
                    _krx_empty_scoped_payload(
                        kind=self.kind, source=self.source, session=session, retrieved_at=retrieved_at
                    )
                )
                continue
            out.append(
                krx_security_master_scoped_payload(records=records, session=session, retrieved_at=retrieved_at)
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


class KrxHedgeSeriesJob:
    """KOSDAQ150 index and inverse-ETF pages for every completed session."""

    name = "krx_hedge_series"
    source = KRX_HEDGE_SERIES_SOURCE
    kind = EvidenceKind.DAILY_MARKET

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        from src.config import load_runtime_config
        from src.data.hedge_series_silver import load_hedge_series_config

        hedge = load_hedge_series_config(load_runtime_config().hedge_series)
        sessions = completed_sessions(evidence_start=hedge.collection_start, now=ctx.now())
        if not sessions:
            return ()
        answered = ctx.catalog.latest(source=self.source, natural_keys={day.isoformat() for day in sessions})
        units: list[JobUnit] = []
        for day in sessions:
            entry = answered.get(day.isoformat())
            if entry is not None and entry.status in _ANSWERED:
                continue
            units.append(
                JobUnit(
                    source=self.source,
                    natural_key=day.isoformat(),
                    payload={"session": day.isoformat()},
                    max_requests=_MAX_REQUESTS_PER_SESSION,
                )
            )
        return units

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        from src.config import load_runtime_config
        from src.data.hedge_series_silver import load_hedge_series_config

        hedge = load_hedge_series_config(load_runtime_config().hedge_series)
        retrieved_at = ctx.now()
        out: list[ScopedRawPayload] = []
        for unit in units:
            session = date.fromisoformat(unit.payload["session"])
            records = list(
                ctx.collector.fetch_hedge_records(
                    session, etf_tickers=(hedge.inverse_ticker,), index_name=hedge.index_name
                )
            )
            if not records:
                out.append(
                    _krx_empty_scoped_payload(
                        kind=self.kind, source=self.source, session=session, retrieved_at=retrieved_at
                    )
                )
                continue
            out.append(
                krx_hedge_series_scoped_payload(records=records, session=session, retrieved_at=retrieved_at)
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()


KRX_JOBS: Mapping[str, JobSpec] = {
    KrxDailyMarketJob.name: KrxDailyMarketJob(),
    KrxSecurityMasterJob.name: KrxSecurityMasterJob(),
    KrxHedgeSeriesJob.name: KrxHedgeSeriesJob(),
}


def resolve_krx_job(name: str) -> JobSpec:
    """Return the job spec registered under ``name``.

    Raises:
        PITDataError: no KRX job is registered under ``name``.
    """
    try:
        return KRX_JOBS[str(name)]
    except KeyError:
        raise PITDataError(f"unknown KRX job {name!r}: expected one of {sorted(KRX_JOBS)}") from None


def build_krx_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: KrxApiClient | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one KRX job without issuing any provider request."""
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=provider.runner("krx"),
        key_env=provider.krx.api_key_env,
        collector=collector,
        now=now or (lambda: datetime.now(UTC)),
        sleep=sleep or time.sleep,
    )
