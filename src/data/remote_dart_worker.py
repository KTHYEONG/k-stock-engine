"""Unattended OpenDART fact collection for a remote host with its own IP.

OpenDART blocks a host that bursts requests, independently of the API key, so a
long backfill can be moved to another host and shipped back. The worker owns
its own quota ledger for the key, keeps a small ``done`` list, and writes only
immutable Bronze pages; a local ingest step verifies and registers them.
"""
from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.config.errors import ConfigError
from src.config.secrets import read_secret
from src.core.pit import PITDataError
from src.data.jobs.runner import seconds_until_window_end
from src.integrations.dart.client import dart_ledger_for_key
from src.integrations.quota import ProviderQuotaStateStore

__all__ = ["WorkerResult", "run_named_job", "run_worker", "seconds_until_window_end"]


class _Collector(Protocol):
    aborted: bool

    def health_check(self) -> None: ...


_REQUESTS_PER_IDENTITY = 3  # worst case: CFS, OFS, document archive


@dataclass(frozen=True, slots=True)
class WorkerResult:
    status: str
    done_total: int
    pending_left: int
    requests_today: int


def run_worker(
    *,
    root: Path,
    key_env: str,
    collect: Callable[..., Any],
    build_collector: Callable[[str, ProviderQuotaStateStore, dict[str, Any]], _Collector],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> WorkerResult:
    """Collect pending job identities until done, out of budget, or the provider misbehaves.

    Args:
        root: Worker directory holding ``job.json``; ``out/`` and ``state/`` are created inside it.
        key_env: Environment variable holding the OpenDART key.
        collect: Collector callable returning an artifact with ``report_path`` and
            ``filing_ids``; called as ``collect(dart=..., identities=..., bronze_root=...,
            retrieved_at=...)``.
        build_collector: Factory for the collector given key, own quota ledger, and job policy.
        now: Clock; sleep: Sleep function (both injectable for tests).

    Returns:
        Terminal status: ``complete``, ``budget_exhausted``, ``provider_unreachable``,
        ``provider_unstable``, ``quota_blocked`` or ``busy`` (another worker holds the lock).

    Raises:
        PITDataError: the job file or key is missing or malformed.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "worker.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        return WorkerResult("busy", 0, 0, 0)
    try:
        return _run_locked(root=root, key_env=key_env, collect=collect, build_collector=build_collector, now=now, sleep=sleep)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _run_locked(
    *,
    root: Path,
    key_env: str,
    collect: Callable[..., Any],
    build_collector: Callable[[str, ProviderQuotaStateStore, dict[str, Any]], _Collector],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> WorkerResult:
    from src.integrations.dart.xbrl import DartCircuitOpenError

    job_path = root / "job.json"
    if not job_path.is_file():
        raise PITDataError(f"worker job file is missing: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    policy = dict(job["policy"])
    try:
        api_key = read_secret(key_env)
    except ConfigError as exc:
        raise PITDataError(f"{key_env} is not set") from exc
    if not policy.get("quota_provider"):
        policy["quota_provider"] = dart_ledger_for_key(
            key_env=key_env,
            primary_key_env=str(job.get("primary_key_env", key_env)),
            api_key=api_key,
        )
    out = root / "out"
    out.mkdir(exist_ok=True)
    done_path = out / "done.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    pending = [item for item in job["identities"] if str(item["filing_id"]) not in done]
    store = ProviderQuotaStateStore(root / "state")
    provider = str(policy["quota_provider"])

    def used_today() -> int:
        budget = int(policy["daily_budget"])
        return budget - store.remaining_daily_attempts(provider=provider, now=now(), daily_limit=budget)

    def result(status: str) -> WorkerResult:
        left = len([i for i in job["identities"] if str(i["filing_id"]) not in done])
        (root / "progress.json").write_text(
            json.dumps({"status": status, "done_total": len(done), "pending_left": left, "requests_today": used_today(), "at": now().isoformat()}),
            encoding="utf-8",
        )
        return WorkerResult(status, len(done), left, used_today())

    if not pending:
        (root / "COMPLETE").write_text(now().isoformat(), encoding="utf-8")
        return result("complete")
    if int(policy["daily_budget"]) - int(policy["daily_reserve"]) - used_today() < _REQUESTS_PER_IDENTITY:
        return result("budget_exhausted")  # 예산이 없으면 헬스체크 요청도 아끼기 위해 바로 종료한다.
    collector = build_collector(api_key, store, policy)
    try:
        collector.health_check()
    except Exception:  # noqa: BLE001 - any failure means this host cannot collect right now
        return result("provider_unreachable")
    chunk_size = int(policy["chunk"])
    position = 0
    while position < len(pending):
        wait = seconds_until_window_end(now(), policy.get("avoid_kst_windows", ()))
        if wait > 0:
            sleep(wait)
        headroom = int(policy["daily_budget"]) - int(policy["daily_reserve"]) - used_today()
        allowance = min(chunk_size, len(pending) - position, headroom // _REQUESTS_PER_IDENTITY)
        if allowance < 1:
            return result("budget_exhausted")
        chunk = tuple(pending[position : position + allowance])
        try:
            artifact = collect(dart=collector, identities=chunk, bronze_root=out / "bronze", retrieved_at=now())
        except DartCircuitOpenError:
            return result("provider_unstable")
        report = json.loads(Path(artifact.report_path).read_text(encoding="utf-8"))
        finished = [str(fid) for fid in report["filing_ids"]]
        with done_path.open("a", encoding="utf-8") as handle:
            handle.writelines(f"{fid}\n" for fid in finished)
        done.update(finished)
        position += len(finished)
        if int(report.get("blocked", 0)) > 0:
            return result("quota_blocked")
        if collector.aborted:
            return result("provider_unstable")
    (root / "COMPLETE").write_text(now().isoformat(), encoding="utf-8")
    return result("complete")


def run_named_job(
    *,
    root: Path,
    key_env: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> WorkerResult:
    """Run the DART job named by ``root/job.json`` through the shared budgeted runner.

    The job file names the job and the scope instead of carrying an identity
    list, so the worker plans from catalog state like every local run. The job
    file carries the scope and provider policy dumps, so the collection host
    needs only the code, the key, and a mirrored data root; the quota ledger
    lives in the workspace state and the local pull step folds it exactly once.

    Returns:
        Terminal status: ``complete``, ``budget_exhausted``,
        ``provider_unreachable``, ``provider_unstable``, ``quota_blocked``,
        ``chunk_limit``, ``dry_run`` or ``busy``.

    Raises:
        PITDataError: the job file is missing, malformed, or names an unknown job.
    """
    from src.config.providers import ProviderPolicy
    from src.data.jobs.dart import resolve_dart_job
    from src.data.jobs.runner import build_job_context, run_job
    from src.data.research_scope import ResearchScope
    from src.data.runtime import DataRuntime
    from src.data.workspace import build_workspace
    from src.integrations.dart.xbrl import DartXbrlCollector

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    job_path = root / "job.json"
    if not job_path.is_file():
        raise PITDataError(f"worker job file is missing: {job_path}")
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job_name = str(job["job"])
        scope = ResearchScope.model_validate(job["scope"])
        provider = ProviderPolicy.model_validate(job["provider_policy"])
        data_root = Path(str(job["data_root"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PITDataError(f"worker job file is invalid: {job_path}") from exc
    spec = resolve_dart_job(job_name)
    resolved_key = key_env or str(job.get("key_env") or provider.default_key_env)
    try:
        api_key = read_secret(resolved_key)
    except ConfigError as exc:
        raise PITDataError(f"{resolved_key} is not set") from exc
    runtime = DataRuntime(scope=scope, workspace=build_workspace(data_root=data_root, scope=scope))
    store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    try:
        key_policy = provider.dart_key(resolved_key)
    except ConfigError as exc:
        raise PITDataError(f"worker key {resolved_key!r} has no declared policy") from exc
    collector = DartXbrlCollector(
        api_key,
        quota_store=store,
        quota_provider=dart_ledger_for_key(
            key_env=resolved_key, primary_key_env=str(job.get("primary_key_env", resolved_key)), api_key=api_key
        ),
        max_workers=1,
        min_interval=key_policy.min_interval_seconds,
        daily_request_limit=key_policy.daily_budget,
    )
    ctx = build_job_context(
        runtime=runtime, provider=provider, key_env=resolved_key, collector=collector, now=now, sleep=sleep
    )

    def emit(payload: Mapping[str, object]) -> None:
        (root / "progress.json").write_text(json.dumps(dict(payload), sort_keys=True, default=str), encoding="utf-8")

    max_chunks = job.get("max_chunks")
    chunk = job.get("chunk")
    report = run_job(
        spec,
        ctx,
        chunk_size=int(chunk) if chunk is not None else provider.dart.batch_identities,
        max_chunks=int(max_chunks) if max_chunks is not None else None,
        dry_run=bool(job.get("dry_run", False)),
        emit=emit,
    )
    (root / "progress.json").write_text(
        json.dumps(
            {
                "status": report.status,
                "done_total": report.done,
                "pending_left": report.pending_left,
                "requests_today": report.requests_used,
                "at": now().isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if report.status == "complete":
        (root / "COMPLETE").write_text(now().isoformat(), encoding="utf-8")
    return WorkerResult(report.status, report.done, report.pending_left, report.requests_used)
