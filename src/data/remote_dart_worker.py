"""Unattended OpenDART fact collection for a remote host with its own IP.

OpenDART blocks a host that bursts requests, independently of the API key, so a
long backfill can be moved to another host and shipped back. The worker owns
its own quota ledger for the key, keeps a small ``done`` list, and writes only
immutable Bronze pages; a local ingest step verifies and registers them.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from src.core.pit import PITDataError
from src.integrations.dart.client import dart_quota_provider
from src.integrations.quota import ProviderQuotaStateStore

_KST = timedelta(hours=9)
_REQUESTS_PER_IDENTITY = 3  # worst case: CFS, OFS, document archive


class _Collector(Protocol):
    aborted: bool

    def health_check(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerResult:
    status: str
    done_total: int
    pending_left: int
    requests_today: int


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def seconds_until_window_end(now: datetime, windows: Sequence[Sequence[str]]) -> float:
    """Seconds to wait when ``now`` (KST wall clock) is inside a protected window, else 0.

    Windows mark periods when another job that shares this host's IP is expected to call the provider.
    """
    kst = now.astimezone(UTC) + _KST
    minute = kst.hour * 60 + kst.minute
    for start, end in windows:
        if _minutes(start) <= minute < _minutes(end):
            target = kst.replace(hour=_minutes(end) // 60, minute=_minutes(end) % 60, second=0, microsecond=0)
            return max(0.0, (target - kst).total_seconds())
    return 0.0


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
        collect: ``collect_dart_financial_facts``-compatible callable.
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
    api_key = os.environ.get(key_env)
    if not api_key:
        raise PITDataError(f"{key_env} is not set")
    out = root / "out"
    out.mkdir(exist_ok=True)
    done_path = out / "done.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    pending = [item for item in job["identities"] if str(item["filing_id"]) not in done]
    store = ProviderQuotaStateStore(root / "state")
    provider = dart_quota_provider(api_key)

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
