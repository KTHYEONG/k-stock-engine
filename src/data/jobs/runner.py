"""One budgeted, resumable job runner shared by every provider job.

A job describes resumable units keyed by their catalog natural key; the runner
executes them in quota-bounded chunks until done, out of budget, or unsafe to
continue. One process runs a job name at a time, enforced by a lock file.
"""
from __future__ import annotations

import fcntl
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from src.config.providers import ProviderPolicy, RunnerPolicy
from src.config.secrets import read_secret
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRangePayload, ScopedRawPayload
from src.integrations.dart.client import dart_ledger_for_key
from src.integrations.errors import ProviderError as _ProviderError
from src.integrations.errors import ProviderQuotaExhaustedError
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "JobContext",
    "JobReport",
    "JobSpec",
    "JobUnit",
    "ScopedPayload",
    "build_job_context",
    "run_job",
    "seconds_until_window_end",
]

#: One unit of collected evidence: a keyed raw page or an answered session range.
ScopedPayload = ScopedRawPayload | ScopedRangePayload

_LOG = logging.getLogger(__name__)

_KST = timedelta(hours=9)


@dataclass(frozen=True, slots=True)
class JobUnit:
    """One resumable unit of provider work keyed by its catalog natural key."""

    source: str
    natural_key: str
    payload: Mapping[str, str]
    max_requests: int

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.natural_key.strip():
            raise ValueError("job unit requires a source and a natural key")
        if isinstance(self.max_requests, bool) or int(self.max_requests) < 1:
            raise ValueError(f"invalid max_requests {self.max_requests!r}: must be a positive integer")


@dataclass(frozen=True, slots=True)
class JobContext:
    """Immutable execution context for one job run under one key's budget."""

    runtime: DataRuntime
    catalog: ReceiptCatalog
    writer: ScopedBronzeWriter
    provider: ProviderPolicy
    quota_store: ProviderQuotaStateStore
    runner: RunnerPolicy
    key_env: str
    collector: Any
    now: Callable[[], datetime]
    sleep: Callable[[float], None]

    def headroom(self) -> int:
        """Request capacity left for this job after holding back its reserve."""
        remaining = self.quota_store.remaining_daily_attempts(
            provider=self.runner.quota_provider, now=self.now(), daily_limit=self.runner.daily_budget
        )
        return max(0, remaining - int(self.runner.daily_reserve))


class JobSpec(Protocol):
    """One resumable provider job: plan from the catalog, fetch units, check health.

    A spec may set ``fail_fast = True`` to abort the run on the first transport
    failure instead of tolerating it up to the circuit threshold. Failing fast
    keeps chunk retries aligned with per-chunk Bronze checkpoints: a failed
    chunk is never followed by a later chunk in the same run, so the next run
    resumes exactly at the failed chunk.
    """

    name: str

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]: ...
    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedPayload]: ...
    def health_check(self, ctx: JobContext) -> None: ...


@dataclass(frozen=True, slots=True)
class JobReport:
    """Terminal outcome of one job run."""

    status: str
    done: int
    pending_left: int
    requests_used: int


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def seconds_until_window_end(now: datetime, windows: Sequence[Sequence[str]]) -> float:
    """Seconds to wait when ``now`` (KST wall clock) is inside a protected window, else 0.

    Windows mark periods when another job sharing this host's IP is expected to call the provider.
    """
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    kst = moment.astimezone(UTC) + _KST
    minute = kst.hour * 60 + kst.minute
    for start, end in windows:
        if _minutes(start) <= minute < _minutes(end):
            target = kst.replace(hour=_minutes(end) // 60, minute=_minutes(end) % 60, second=0, microsecond=0)
            return max(0.0, (target - kst).total_seconds())
    return 0.0


def build_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    key_env: str | None,
    collector: Any = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> JobContext:
    """Build the execution context for one key without issuing any provider request."""
    import dataclasses

    resolved = key_env or provider.default_key_env
    api_key = read_secret(resolved, default="")
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=dataclasses.replace(
            provider.runner("dart", key_env=resolved),
            quota_provider=dart_ledger_for_key(
                key_env=resolved, primary_key_env=provider.primary_key_env, api_key=api_key
            ),
        ),
        key_env=resolved,
        collector=collector,
        now=now or (lambda: datetime.now(UTC)),
        sleep=sleep or time.sleep,
    )


def run_job(
    spec: JobSpec,
    ctx: JobContext,
    *,
    chunk_size: int,
    max_chunks: int | None,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
) -> JobReport:
    """Run a job in quota-bounded chunks until done, out of budget, or unsafe to continue.

    Before each chunk the headroom is re-read and the chunk is sized for the
    worst case (the sum of ``max_requests``); the shared-IP avoid windows are
    respected by waiting (the only allowed sleep); a quota block stops the run.
    A health check runs once before the first request. Pages of a chunk are
    persisted with one ``ScopedBronzeWriter.persist_many`` call before the next
    chunk starts, so a crash loses at most one chunk. The circuit breaker
    (``RunnerPolicy.circuit_threshold`` consecutive transport failures) aborts the
    chunk without persisting failed units. A spec with ``fail_fast = True``
    aborts the whole run on the first transport failure instead. Every chunk
    emits one progress line carrying ``done``, ``pending``, ``requests_used``,
    ``elapsed_s`` and ``eta_s`` (mean seconds per completed unit).
    """
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError(f"invalid chunk_size {chunk_size!r}: must be a positive integer")
    if max_chunks is not None and (isinstance(max_chunks, bool) or int(max_chunks) < 1):
        raise ValueError(f"invalid max_chunks {max_chunks!r}: must be a positive integer")
    lock_path = ctx.runtime.workspace.state_root / "jobs" / f"{spec.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return JobReport(status="busy", done=0, pending_left=0, requests_used=0)
    try:
        return _run_locked(
            spec, ctx, chunk_size=int(chunk_size), max_chunks=max_chunks, dry_run=dry_run, emit=emit
        )
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _ledger_remaining(ctx: JobContext) -> int:
    return ctx.quota_store.remaining_daily_attempts(
        provider=ctx.runner.quota_provider, now=ctx.now(), daily_limit=ctx.runner.daily_budget
    )


def _emit_done(
    *,
    spec_name: str,
    emit: Callable[[Mapping[str, object]], None],
    status: str,
    done: int,
    pending_left: int,
    requests_used: int,
) -> JobReport:
    emit(
        {
            "phase": "done",
            "job": spec_name,
            "status": status,
            "done": done,
            "pending_left": pending_left,
            "requests_used": requests_used,
        }
    )
    _LOG.info(
        "[DATA] job=%s phase=done status=%s done=%d pending_left=%d requests_used=%d",
        spec_name,
        status,
        done,
        pending_left,
        requests_used,
    )
    return JobReport(status=status, done=done, pending_left=pending_left, requests_used=requests_used)


def _run_locked(
    spec: JobSpec,
    ctx: JobContext,
    *,
    chunk_size: int,
    max_chunks: int | None,
    dry_run: bool,
    emit: Callable[[Mapping[str, object]], None],
) -> JobReport:
    pending = list(spec.pending(ctx))
    headroom = ctx.headroom()
    worst_case = sum(unit.max_requests for unit in pending)
    emit(
        {
            "phase": "plan",
            "job": spec.name,
            "pending": len(pending),
            "worst_case_requests": worst_case,
            "headroom": headroom,
        }
    )
    _LOG.info(
        "[DATA] job=%s phase=plan pending=%d worst_case_requests=%d headroom=%d",
        spec.name,
        len(pending),
        worst_case,
        headroom,
    )
    start_remaining = _ledger_remaining(ctx)
    if dry_run:
        return _emit_done(
            spec_name=spec.name, emit=emit, status="dry_run", done=0,
            pending_left=len(pending), requests_used=0,
        )
    if not pending:
        return _emit_done(
            spec_name=spec.name, emit=emit, status="complete", done=0,
            pending_left=0, requests_used=max(0, start_remaining - _ledger_remaining(ctx)),
        )
    try:
        spec.health_check(ctx)
    except Exception as exc:  # noqa: BLE001 - any failure means this host cannot collect right now
        _LOG.warning("[DATA] job=%s phase=health_check status=failed error=%s", spec.name, exc)
        return _emit_done(
            spec_name=spec.name, emit=emit, status="provider_unreachable", done=0,
            pending_left=len(pending), requests_used=max(0, start_remaining - _ledger_remaining(ctx)),
        )
    threshold = ctx.runner.circuit_threshold
    windows = ctx.runner.avoid_windows_kst
    fail_fast = bool(getattr(spec, "fail_fast", False))
    done = 0
    chunks = 0
    status = "complete"
    position = 0
    started_at = ctx.now()
    while position < len(pending):
        if max_chunks is not None and chunks >= max_chunks:
            status = "chunk_limit"
            break
        wait = seconds_until_window_end(ctx.now(), windows)
        if wait > 0:
            ctx.sleep(wait)
        headroom = ctx.headroom()
        chunk: list[JobUnit] = []
        cumulative = 0
        while (
            position + len(chunk) < len(pending)
            and len(chunk) < chunk_size
            and cumulative + pending[position + len(chunk)].max_requests <= headroom
        ):
            cumulative += pending[position + len(chunk)].max_requests
            chunk.append(pending[position + len(chunk)])
        if not chunk:
            status = "budget_exhausted"
            break
        chunk_payloads: list[ScopedPayload] = []
        chunk_done = 0
        failures = 0
        quota_blocked = False
        for unit in chunk:
            before = len(chunk_payloads)
            try:
                chunk_payloads.extend(spec.fetch(ctx, (unit,)))
            except ProviderQuotaExhaustedError:
                quota_blocked = True
                break
            except _ProviderError:
                failures += 1
                if fail_fast or failures >= threshold:
                    break
                continue
            failures = 0
            if len(chunk_payloads) > before:
                chunk_done += 1
        if quota_blocked:
            status = "quota_blocked"
            break
        if chunk_payloads:
            ctx.writer.persist_many(tuple(chunk_payloads))
        done += chunk_done
        position += len(chunk)
        chunks += 1
        used = max(0, start_remaining - _ledger_remaining(ctx))
        elapsed_s = max(0.0, (ctx.now() - started_at).total_seconds())
        mean_per_unit = elapsed_s / done if done > 0 else 0.0
        eta_s = mean_per_unit * (len(pending) - done)
        emit(
            {
                "phase": "chunk",
                "job": spec.name,
                "done": done,
                "pending": len(pending) - done,
                "requests_used": used,
                "elapsed_s": elapsed_s,
                "eta_s": eta_s,
            }
        )
        _LOG.info(
            "[DATA] job=%s phase=chunk done=%d pending=%d requests_used=%d elapsed_s=%.1f eta_s=%.1f",
            spec.name,
            done,
            len(pending) - done,
            used,
            elapsed_s,
            eta_s,
        )
        if failures >= threshold or (failures > 0 and fail_fast):
            status = "provider_unstable"
            break
    return _emit_done(
        spec_name=spec.name, emit=emit, status=status, done=done,
        pending_left=len(pending) - done,
        requests_used=max(0, start_remaining - _ledger_remaining(ctx)),
    )
