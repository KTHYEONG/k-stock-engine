"""Scheduled entrypoints: evening collection, morning seal and Drive backup."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final

from src.core.pit import PITDataError
from src.core.time import KRX_TZ
from src.ops.audit import run_daily_audit
from src.ops.guards import (
    DEFAULT_MIN_AVAILABLE_MB,
    DEFAULT_MIN_FREE_DISK_GB,
    GuardRefused,
    require_headroom,
    trading_day,
)
from src.ops.run_outcome import record_run_outcome

if TYPE_CHECKING:
    import polars as pl

    from src.data.runtime import DataRuntime
    from src.ops.alerts import AlertSender

__all__ = [
    "DBSEC_COLLECT_STEPS",
    "EVENING_COLLECT_STEPS",
    "KRX_COLLECT_STEPS",
    "POLL_INTERVAL_S",
    "alert",
    "audit",
    "backup",
    "collect_dbsec",
    "collect_eve",
    "main",
    "morning",
    "weekly_report",
]

_LOG = logging.getLogger(__name__)


def _log_job_failure(job: str, exc: BaseException) -> None:
    """Log the swallowed job exception with its traceback; the job record keeps only the class name."""
    _LOG.error("[OPS] command=%s status=DEGRADED reason=%s", job, type(exc).__name__, exc_info=exc)

EVENING_COLLECT_STEPS: Final[frozenset[str]] = frozenset(
    {
        "dart_corp_codes",
        "dart_disclosures",
        "dart_facts",
        "dividend_decisions",
        "earnings_releases",
        "kind_notice_search",
        "kind_notice_documents",
        "ls_investor_flow",
        "kis_investor_flow",
    }
)
KRX_COLLECT_STEPS: Final[frozenset[str]] = frozenset(
    {
        "krx_daily_market",
        "krx_security_master",
        "krx_hedge_series",
        "krx_trend_series",
        "krx_cash_series",
    }
)
DBSEC_COLLECT_STEPS: Final[frozenset[str]] = frozenset(
    {"dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"}
)
POLL_INTERVAL_S: Final = 60.0

_DEADLINE_HOUR: Final = 8
_DEADLINE_MINUTE: Final = 55
_LATE_SEAL_HOUR: Final = 9
_PRUNE_AFTER_DAYS: Final = 14
_PUSH_SOURCES: Final = ("bronze", "bronze_catalog", "state", "ops")


def _events_root(data_root: Path) -> Path:
    return Path(data_root) / "ops" / "events"


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=KRX_TZ)


def _record(
    data_root: Path,
    job: str,
    outcome: str,
    run_date: date,
    moment: datetime,
    reason: str = "",
    metrics: Mapping[str, object] | None = None,
) -> None:
    record_run_outcome(
        _events_root(data_root), job, outcome, run_date=run_date, now=_aware(moment), reason=reason, metrics=metrics
    )


def _peak_rss_mb() -> float:
    with contextlib.suppress(Exception):
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return 0.0  # pragma: no cover - Linux always provides resource


def _node_peak_metrics(report: object) -> dict[str, float]:
    """Per-node peak RSS in MB, rounded for events; empty when unavailable."""
    try:
        raw = getattr(report, "node_peak_rss_mb", {})
        if not isinstance(raw, Mapping):
            return {}
        return {str(kind): round(float(value), 1) for kind, value in dict(raw).items()}
    except Exception:
        return {}


def _guard_headroom(data_root: Path) -> GuardRefused | None:
    try:
        require_headroom(
            min_available_mb=DEFAULT_MIN_AVAILABLE_MB,
            min_free_disk_gb=DEFAULT_MIN_FREE_DISK_GB,
            data_root=Path(data_root),
        )
    except GuardRefused as exc:
        return exc
    return None


def _previous_session(day: date) -> date:
    from src.core.krx_calendar import xkrx_calendar_through

    ordered = sorted({moment.astimezone(KRX_TZ).date() for moment in xkrx_calendar_through(day).sessions})
    try:
        pos = ordered.index(day)
    except ValueError as exc:
        raise PITDataError(f"{day.isoformat()} is not an XKRX session") from exc
    if pos == 0:
        raise PITDataError(f"no XKRX session before {day.isoformat()}")
    return ordered[pos - 1]


def _run_refresh(
    data_root: Path, *, decision_time: datetime, collect: bool, collect_steps: frozenset[str] | None
) -> object:
    from dataclasses import replace

    from src.data.pipeline_graph import build_refresh_context, refresh_scope
    from src.data.runtime import resolve_data_runtime

    runtime = resolve_data_runtime(data_root=Path(data_root))
    ctx = build_refresh_context(runtime, decision_time=_aware(decision_time))
    if collect_steps == EVENING_COLLECT_STEPS:
        ctx = replace(ctx, deferred_build_kinds=frozenset({
            "ordinary_universe", "daily_market", "hedge_series", "trend_series", "cash_series",
            "market_panel", "reference_benchmarks", "market_actions",
        }))
    return refresh_scope(
        ctx, collect=collect, dry_run=False, emit=lambda payload: _LOG.debug("refresh %s", dict(payload)),
        collect_steps=collect_steps,
    )


def _data_session_present(data_root: Path, session: date) -> bool:
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
    from src.data.runtime import resolve_data_runtime

    try:
        runtime = resolve_data_runtime(data_root=Path(data_root))
        catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
        receipt = catalog.latest(source="krx_daily_market", natural_keys={session.isoformat()}).get(session.isoformat())
        return receipt is not None and receipt.status == EvidenceStatus.SUCCESS
    except PITDataError:
        return False
    return False


def _poll_krx_once(data_root: Path, session: date, decision_time: datetime) -> bool:
    from dataclasses import replace

    from src.data.pipeline_graph import _default_collection_steps, build_refresh_context, run_collection_jobs
    from src.data.runtime import resolve_data_runtime

    ctx = build_refresh_context(resolve_data_runtime(data_root=Path(data_root)), decision_time=_aware(decision_time))
    ctx = replace(ctx, retry_krx_session=session)
    steps = tuple(step for step in _default_collection_steps() if step[0] in KRX_COLLECT_STEPS)
    run_collection_jobs(ctx, dry_run=False, emit=lambda payload: None, steps=steps)
    return _data_session_present(Path(data_root), session)


def _wait_for_krx(
    data_root: Path, session: date, *, now: Callable[[], datetime], sleep: Callable[[float], None],
    deadline: datetime,
) -> bool:
    while True:
        if _aware(now()) > deadline:
            return False
        if _poll_krx_once(Path(data_root), session, _aware(now())):
            return True
        if _aware(now()) >= deadline:
            return False
        sleep(min(POLL_INTERVAL_S, (deadline - _aware(now())).total_seconds()))


def _build_all(data_root: Path, *, decision_time: datetime) -> object:
    return _run_refresh(data_root, decision_time=decision_time, collect=False, collect_steps=None)


def _research_paths(data_root: Path) -> SimpleNamespace:
    from src.config.runtime import load_runtime_config
    from src.data.runtime import resolve_data_runtime

    runtime = resolve_data_runtime(data_root=Path(data_root))
    state_root = runtime.workspace.state_root
    return SimpleNamespace(
        runtime=runtime,
        state_root=state_root,
        spec=load_runtime_config().champion_file,
        state=state_root / "operate" / "state.json",
        decisions=state_root / "decisions",
        shadow=state_root / "shadow",
    )


def _models_base(spec_path: Path, state_root: Path) -> Path:
    from src.research.cli import _load_spec
    from src.research.pipeline import _scores_identity

    identity = _scores_identity(_load_spec(Path(spec_path)))
    return Path(state_root) / "research" / "models" / identity[:16]


def _decide_phase(data_root: Path) -> list[str]:
    from src.research.cli import _run_decide

    paths = _research_paths(Path(data_root))
    args = argparse.Namespace(
        scope_config=None, data_root=Path(data_root), spec=paths.spec, state=paths.state,
        models_dir=_models_base(paths.spec, paths.state_root), decisions_dir=paths.decisions,
        tail_sessions=800, capital=None,
    )
    result = dict(_run_decide(args))
    sealed = result.get("sealed", [])
    items = sealed if isinstance(sealed, list) else []
    return [str(item) for item in items]


def _shadow_phase(data_root: Path) -> None:
    from src.research.cli import _run_shadow
    from src.research.operate import read_chain

    paths = _research_paths(Path(data_root))
    chain = read_chain(paths.decisions)
    if chain and min(item.execution_session for item in chain) > max(item.data_session for item in chain):
        return
    args = argparse.Namespace(
        scope_config=None, data_root=Path(data_root), spec=paths.spec, decisions_dir=paths.decisions,
        shadow_dir=paths.shadow, tail_sessions=800, capital=None,
    )
    _run_shadow(args)


def _sealed_at_for(paths_decisions: Path, execution: date) -> datetime | None:
    from src.research.operate import read_chain

    stamps = [item.sealed_at for item in read_chain(Path(paths_decisions)) if item.execution_session == execution]
    return max(stamps) if stamps else None


def _prune_unreferenced_datasets(root: Path) -> dict[str, object]:
    """Delete unreferenced dataset generations; never raises so a prune fault cannot fail collection."""
    from src.data.dataset_maintenance import prune_datasets
    from src.data.dataset_registry import DatasetRegistry
    from src.data.runtime import resolve_data_runtime

    try:
        runtime = resolve_data_runtime(data_root=root)
        summary = prune_datasets(
            runtime, DatasetRegistry(runtime.workspace.state_root), apply=True, emit=lambda payload: None,
        )
    except Exception as exc:
        return {"prune_error": type(exc).__name__}
    return {
        "pruned": _as_int(summary.get("deleted")),
        "prune_failed": _as_int(summary.get("failed")),
        "prune_unreadable": _as_int(summary.get("unreadable")),
    }


def collect_eve(data_root: Path, *, today: date, now: datetime) -> int:
    """Evening collection and build of everything except the KRX-derived nodes; returns the process exit code."""
    root = Path(data_root)
    moment = _aware(now)
    if not trading_day(today):
        _record(root, "collect_eve", "SKIPPED", today, moment, reason="non_trading_day")
        return 0
    refused = _guard_headroom(root)
    if refused is not None:
        _record(root, "collect_eve", "DEGRADED", today, moment, reason=str(refused)[:300],
                metrics={"guard": "headroom", "available_mb": refused.available_mb, "free_disk_gb": refused.free_disk_gb})
        return 1
    try:
        report = _run_refresh(root, decision_time=moment, collect=True, collect_steps=EVENING_COLLECT_STEPS)
    except Exception as exc:
        _log_job_failure("collect_eve", exc)
        _record(root, "collect_eve", "DEGRADED", today, moment, reason=type(exc).__name__)
        return 1
    blocking = getattr(report, "blocking_job", None)
    degraded = list(getattr(report, "degraded_jobs", ()) or ())
    if blocking is not None:
        _record(
            root, "collect_eve", "DEGRADED", today, moment, reason=f"blocked on {blocking}"[:300],
            metrics={"blocking_job": str(blocking), "degraded_jobs": degraded},
        )
        return 1
    planned = list(getattr(report, "planned", ()))
    built = list(getattr(report, "built", ()))
    node_peaks = _node_peak_metrics(report)
    from src.data.pipeline_graph import build_refresh_context, warm_daily_flag_cache
    from src.data.runtime import resolve_data_runtime

    try:
        warm_ctx = build_refresh_context(resolve_data_runtime(data_root=root), decision_time=moment)
        warm_metrics: dict[str, object] = {"flag_cache_sessions": int(warm_daily_flag_cache(warm_ctx))}
        warm_error: str | None = None
    except Exception as exc:
        warm_metrics = {"flag_cache_error": type(exc).__name__}
        warm_error = type(exc).__name__
    prune_metrics = _prune_unreferenced_datasets(root)
    if degraded:
        _record(
            root, "collect_eve", "DEGRADED", today, moment,
            reason=f"optional:{','.join(degraded)}"[:300],
            metrics={"degraded_jobs": degraded, "planned": len(planned), "built": len(built),
                      "peak_rss_mb": _peak_rss_mb(), "node_peak_rss_mb": node_peaks,
                      **warm_metrics, **prune_metrics},
        )
        return 0
    if warm_error is not None:
        _record(
            root, "collect_eve", "DEGRADED", today, moment, reason="flag_cache_warm_failed",
            metrics={"planned": len(planned), "built": len(built),
                      "peak_rss_mb": _peak_rss_mb(), "node_peak_rss_mb": node_peaks,
                      **warm_metrics, **prune_metrics},
        )
        return 0
    _record(
        root, "collect_eve", "OK", today, moment,
        metrics={"planned": len(planned), "built": len(built), "peak_rss_mb": _peak_rss_mb(),
                 "node_peak_rss_mb": node_peaks, **warm_metrics, **prune_metrics},
    )
    return 0


def _silver_session_frame(silver_root: Path, dataset_id: str, session: date) -> pl.DataFrame | None:
    import polars as pl

    part = Path(silver_root) / dataset_id / f"session={session.isoformat()}" / "part.parquet"
    try:
        if part.is_file() and not part.is_symlink():
            return pl.read_parquet(part)
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise PITDataError(f"unreadable daily-market partition for {session.isoformat()}") from exc
    return None


def _reconcile_dbsec_sessions(
    runtime: DataRuntime, *, now: datetime,
) -> tuple[list[dict[str, object]], bool]:
    from src.config import load_provider_policy, load_runtime_config
    from src.data import dbsec_reconcile as reconcile
    from src.data.dataset_registry import DatasetRegistry
    from src.data.evidence_sources import (
        DBSEC_CLOSING_BAR_SOURCE,
        DBSEC_DAILY_CHART_SOURCE,
        DBSEC_TICKER_MASTER_SOURCE,
    )
    from src.data.jobs import dbsec as dbsec_jobs
    from src.data.payload_store import read_payload
    from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog

    provider = load_provider_policy(load_runtime_config())
    policy = provider.dbsec
    workspace = runtime.workspace
    catalog = ReceiptCatalog(workspace.bronze_root / "catalog")
    sessions = dbsec_jobs.dbsec_final_sessions(
        now=now,
        final_time_kst=datetime.strptime(policy.session_final_time_kst, "%H:%M").time(),
        lookback=int(policy.lookback_sessions),
    )
    report_root = workspace.state_root / "dbsec_reconcile"
    try:
        dataset_id = DatasetRegistry(workspace.state_root).require("daily_market")
    except PITDataError:
        dataset_id = None
    entries: list[dict[str, object]] = []
    for day in sessions:
        if (report_root / f"{day.isoformat()}.json").exists():
            continue
        master = catalog.latest(
            source=DBSEC_TICKER_MASTER_SOURCE, natural_keys={day.isoformat()}
        ).get(day.isoformat())
        if master is None or master.status is not EvidenceStatus.SUCCESS:
            continue
        krx = _silver_session_frame(workspace.silver_root, dataset_id, day) if dataset_id is not None else None
        if krx is None or krx.height == 0:
            continue
        tickers = dbsec_jobs.dbsec_master_tickers(
            read_payload(Path(master.payload_path)),
            market_names=policy.equity_market_names,
        )
        # A report is never rewritten, so a session whose bar collection is unfinished must wait rather
        # than freeze an absent receipt as a permanent FAIL; EMPTY receipts are answers and do not defer.
        keys = {f"{day.isoformat()}:{ticker}" for ticker in tickers}
        answered = (EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY)
        if any(
            {key for key, entry in catalog.latest(source=source, natural_keys=keys).items() if entry.status in answered}
            != keys
            for source in (DBSEC_DAILY_CHART_SOURCE, DBSEC_CLOSING_BAR_SOURCE)
        ):
            continue
        report = reconcile.reconcile_session(
            session=day,
            krx=krx,
            db=reconcile.load_dbsec_session_rows(catalog, day, tickers),
            min_match_rate=float(policy.reconcile_min_match_rate),
            min_coverage=float(policy.reconcile_min_coverage),
            sample_limit=int(policy.reconcile_sample_limit),
        )
        reconcile.write_reconciliation(report, root=report_root)
        entries.append(
            {
                "session": day.isoformat(),
                "verdict": report.verdict.value,
                "coverage": report.coverage,
                "close_match_rate": report.close_match_rate,
                "volume_match_rate": report.volume_match_rate,
            }
        )
        _LOG.info(
            "[DATA] stage=dbsec_reconcile session=%s verdict=%s coverage=%.6f close=%.6f volume=%.6f missing_close=%d",
            day.isoformat(),
            report.verdict.value,
            report.coverage,
            report.close_match_rate,
            report.volume_match_rate,
            report.missing_closing_bar,
        )
    history = reconcile.read_reconciliation_history(report_root)
    return entries, reconcile.fallback_readiness(
        history, ready_sessions=int(policy.reconcile_ready_sessions)
    )


def collect_dbsec(data_root: Path, *, today: date, now: datetime) -> int:
    """Collect DB Securities fallback evidence only; returns the process exit code."""
    from src.data.pipeline_graph import _default_collection_steps, build_refresh_context, run_collection_jobs
    from src.data.runtime import resolve_data_runtime

    root = Path(data_root)
    moment = _aware(now)
    if not trading_day(today):
        _record(root, "collect_dbsec", "SKIPPED", today, moment, reason="non_trading_day")
        return 0
    try:
        runtime = resolve_data_runtime(data_root=root)
        ctx = build_refresh_context(runtime, decision_time=moment)
        steps = tuple(step for step in _default_collection_steps() if step[0] in DBSEC_COLLECT_STEPS)
        report = run_collection_jobs(
            ctx, dry_run=False, emit=lambda payload: _LOG.debug("refresh %s", dict(payload)), steps=steps,
        )
    except Exception as exc:
        _log_job_failure("collect_dbsec", exc)
        _record(root, "collect_dbsec", "DEGRADED", today, moment, reason=type(exc).__name__)
        return 1
    metrics = {
        step.job: {"done": step.done, "pending_left": step.pending_left, "requests_used": step.requests_used}
        for step in report.steps
    }
    failed = [step.job for step in report.steps if step.status == "failed"]
    try:
        reconciled, fallback_ready = _reconcile_dbsec_sessions(runtime, now=moment)
    except Exception as exc:
        _log_job_failure("collect_dbsec", exc)
        _record(
            root, "collect_dbsec", "DEGRADED", today, moment,
            reason=f"reconcile:{type(exc).__name__}"[:300],
            metrics={**metrics, "reconciled": [], "fallback_ready": False},
        )
        return 0
    reconcile_metrics: dict[str, object] = {"reconciled": reconciled, "fallback_ready": fallback_ready}
    if failed:
        _record(
            root, "collect_dbsec", "DEGRADED", today, moment,
            reason=f"failed:{','.join(failed)}"[:300], metrics={**metrics, **reconcile_metrics},
        )
        return 0
    _record(root, "collect_dbsec", "OK", today, moment, metrics={**metrics, **reconcile_metrics})
    return 0


def morning(data_root: Path, *, today: date, now: Callable[[], datetime], sleep: Callable[[float], None]) -> int:
    """Wait for yesterday's KRX bars, refresh the KRX-derived nodes, seal the decision, update the shadow ledger."""
    root = Path(data_root)
    if not trading_day(today):
        _record(root, "morning", "SKIPPED", today, _aware(now()), reason="non_trading_day")
        return 0
    refused = _guard_headroom(root)
    if refused is not None:
        _record(root, "morning", "DEGRADED", today, _aware(now()), reason=str(refused)[:300],
                metrics={"guard": "headroom", "available_mb": refused.available_mb, "free_disk_gb": refused.free_disk_gb,
                         "peak_rss_mb": _peak_rss_mb(), "node_peak_rss_mb": {}})
        return 1
    try:
        data_session = _previous_session(today)
    except PITDataError as exc:
        _record(root, "morning", "DEGRADED", today, _aware(now()), reason=type(exc).__name__)
        return 1
    started = time.monotonic()
    deadline = datetime(today.year, today.month, today.day, _DEADLINE_HOUR, _DEADLINE_MINUTE, tzinfo=KRX_TZ)
    try:
        waited = _wait_for_krx(root, data_session, now=now, sleep=sleep, deadline=deadline)
    except Exception as exc:
        _log_job_failure("morning", exc)
        outcome = "NO_DECISION" if isinstance(exc, PITDataError) else "DEGRADED"
        _record(root, "morning", outcome, today, _aware(now()), reason=type(exc).__name__)
        return 1
    wait_s = time.monotonic() - started
    if not waited:
        _record(
            root, "morning", "NO_DECISION", today, _aware(now()), reason="krx_absent_at_deadline",
            metrics={"data_session": data_session.isoformat(), "wait_s": wait_s, "peak_rss_mb": _peak_rss_mb(),
                     "node_peak_rss_mb": {}},
        )
        return 1
    phase = "build"
    build_report: object = None
    try:
        mark = time.monotonic()
        build_report = _build_all(root, decision_time=_aware(now()))
        build_s = time.monotonic() - mark
        mark = time.monotonic()
        sealed = _decide_phase(root)
        decide_s = time.monotonic() - mark
        sealed_at = _sealed_at_for(_research_paths(root).decisions, today)
        if sealed_at is None:
            raise PITDataError("no sealed decision for today's execution session")
        phase = "shadow"
        mark = time.monotonic()
        _shadow_phase(root)
        shadow_s = time.monotonic() - mark
    except PITDataError as exc:
        outcome = "DEGRADED" if phase == "shadow" else "NO_DECISION"
        _record(root, "morning", outcome, today, _aware(now()), reason=type(exc).__name__,
                metrics={"data_session": data_session.isoformat(), "wait_s": wait_s, "peak_rss_mb": _peak_rss_mb(),
                         "node_peak_rss_mb": _node_peak_metrics(build_report)})
        return 1
    except Exception as exc:
        _log_job_failure("morning", exc)
        _record(root, "morning", "DEGRADED", today, _aware(now()), reason=type(exc).__name__,
                metrics={"data_session": data_session.isoformat(), "wait_s": wait_s, "peak_rss_mb": _peak_rss_mb(),
                         "node_peak_rss_mb": _node_peak_metrics(build_report)})
        return 1
    elapsed_s = time.monotonic() - started
    metrics: dict[str, object] = {
        "data_session": data_session.isoformat(),
        "sealed": sealed,
        "sealed_at": _aware(sealed_at).astimezone(KRX_TZ).isoformat(),
        "elapsed_s": elapsed_s,
        "peak_rss_mb": _peak_rss_mb(),
        "node_peak_rss_mb": _node_peak_metrics(build_report),
        "wait_s": wait_s,
        "build_s": build_s,
        "decide_s": decide_s,
        "shadow_s": shadow_s,
    }
    opened = datetime(today.year, today.month, today.day, _LATE_SEAL_HOUR, tzinfo=KRX_TZ)
    if _aware(sealed_at).astimezone(KRX_TZ) >= opened:
        _record(root, "morning", "DEGRADED", today, _aware(now()), reason="late_seal", metrics=metrics)
        return 1
    _record(root, "morning", "OK", today, _aware(now()), metrics=metrics)
    return 0


def _tool_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tools" / "devops" / "lake_backup.py"


def _backup_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "data" / "lake_backup.toml"


def _confirmed_bronze_paths(*, config_path: Path, work_root: Path) -> set[str]:
    import shutil
    import tomllib

    from src.data import lake_backup as lake

    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    rclone = shutil.which("rclone") or str(Path.home() / ".local" / "bin" / "rclone")
    store = lake.RcloneStore(str(config["remote"]).rstrip("/") + "/bronze", rclone_bin=rclone)
    work = Path(work_root) / "bronze-index"
    work.mkdir(parents=True, exist_ok=True)
    try:
        return {entry.path for entry in lake._load_index(store, work)}  # noqa: SLF001
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _is_older_than(path: Path, cutoff_ts: float) -> bool:
    return path.is_file() and path.stat().st_mtime < cutoff_ts


def _prune_bronze(data_root: Path, *, now: datetime, confirmed: set[str] | None = None) -> dict[str, object]:
    from src.data.payload_store import pack_loose
    from src.data.runtime import resolve_data_runtime

    runtime = resolve_data_runtime(data_root=Path(data_root))
    bronze_root = runtime.workspace.bronze_root
    offsite = set(confirmed) if confirmed is not None else _confirmed_bronze_paths(
        config_path=_backup_config_path(), work_root=Path(data_root) / "state" / "lake_tmp"
    )
    cutoff = _aware(now).timestamp() - float(_PRUNE_AFTER_DAYS * 24 * 3600)
    eligible: set[str] = set()
    for rel in offsite:
        try:
            if _is_older_than(bronze_root / rel, cutoff):
                eligible.add(rel)
        except OSError:
            continue
    kinds = sorted(
        item.name
        for item in bronze_root.iterdir()
        if item.is_dir() and not item.is_symlink() and item.name not in ("catalog", "_packs")
    ) if bronze_root.is_dir() else []
    report = pack_loose(bronze_root, kinds=kinds, offsite_paths=eligible, remove=True, paths=eligible)
    return {
        "packed": report.packed,
        "removed": report.removed,
        "skipped_unverified": report.skipped_unverified,
        "raw_bytes": report.raw_bytes,
        "packed_bytes": report.packed_bytes,
    }


def _push_backup_sources(*, status_path: Path, no_lock: bool) -> dict[str, object]:
    command = [sys.executable, str(_tool_path()), "push", "--status", str(status_path),
               "--data-root", str(status_path.parent.parent)]
    for name in _PUSH_SOURCES:
        command += ["--source", name]
    if no_lock:
        command += ["--no-lock"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3 * 3600, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise PITDataError(f"lake backup push timed out: {exc}") from exc
    if completed.returncode != 0:
        raise PITDataError(f"lake backup push failed rc={completed.returncode}: {completed.stderr.strip()[-300:]}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise PITDataError(f"lake backup push returned no report: {exc}") from exc
    return {str(key): value for key, value in dict(payload).items()}


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _ensure_backup_failure_status(path: Path, moment: datetime) -> None:
    from datetime import UTC

    previous: dict[str, object] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            previous = payload
    except (OSError, ValueError):
        previous = {}
    if previous.get("rc") not in (None, 0):
        return
    last_ok = previous.get("last_ok_at")
    if last_ok is None and previous.get("rc") == 0:
        last_ok = previous.get("finished_at")
    failed = {"finished_at": _aware(moment).astimezone(UTC).isoformat(), "rc": 1,
              "last_ok_at": last_ok, "sources": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".backup-status.", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(failed) + "\n")
        handle.flush()
        temporary = Path(handle.name)
    temporary.replace(path)


def _failure_detail(exc: BaseException) -> str:
    from src.config.secrets import environment_snapshot
    from src.ops.alerts import collect_secret_values, redact

    return redact(str(exc), collect_secret_values(environment_snapshot()))[:300]


def backup(data_root: Path, *, now: datetime, no_lock: bool = False) -> int:
    """Push Bronze, catalog, state and ops to Drive, then pack and prune loose Bronze confirmed offsite."""
    root = Path(data_root)
    moment = _aware(now)
    run_date = moment.astimezone(KRX_TZ).date()
    refused = _guard_headroom(root)
    if refused is not None:
        _record(root, "backup", "DEGRADED", run_date, moment, reason=str(refused)[:300],
                metrics={"guard": "headroom", "available_mb": refused.available_mb, "free_disk_gb": refused.free_disk_gb})
        return 1
    status_path = root / "state" / "lake_backup_status.json"
    try:
        sources = _push_backup_sources(status_path=status_path, no_lock=no_lock)
    except Exception as exc:
        _log_job_failure("backup", exc)
        _ensure_backup_failure_status(status_path, moment)
        _record(root, "backup", "DEGRADED", run_date, moment, reason=type(exc).__name__,
                metrics={"detail": _failure_detail(exc)})
        return 1
    try:
        pruned = _prune_bronze(root, now=moment)
    except Exception as exc:
        _log_job_failure("backup", exc)
        _record(root, "backup", "DEGRADED", run_date, moment, reason=type(exc).__name__,
                metrics={"sources": sorted(sources), "detail": _failure_detail(exc)})
        return 1
    _record(root, "backup", "OK", run_date, moment,
            metrics={"sources": sorted(sources), "removed": _as_int(pruned.get("removed"))})
    return 0


def _alerts_dir(data_root: Path) -> Path:
    return Path(data_root) / "ops" / "alerts"


def _heartbeat_open_issues(data_root: Path) -> list[dict[str, str]]:
    from src.ops.audit import AUDIT_HEARTBEAT_RELPATH

    try:
        payload = json.loads((Path(data_root) / AUDIT_HEARTBEAT_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("open_issues")
    if not isinstance(raw, list):
        return []
    return [
        {"key": str(item.get("key", "")), "text": str(item.get("text", ""))}
        for item in raw if isinstance(item, Mapping)
    ]


def _alert_sender() -> tuple[AlertSender | None, frozenset[str]]:
    from src.config.secrets import environment_snapshot
    from src.ops.alerts import collect_secret_values, sender_from_env

    return sender_from_env(), collect_secret_values(environment_snapshot())


def _drain_best_effort(
    sender: AlertSender | None, alerts_dir: Path, secrets: frozenset[str], now: datetime,
) -> None:
    from src.ops.alerts import drain_outbox

    try:
        drain_outbox(sender=sender, alerts_dir=alerts_dir, secret_values=secrets, now=now)
    except OSError:
        _LOG.debug("[OPS] alert drain best-effort failed")


def alert(data_root: Path, *, unit: str, result: str, exit_status: str, journal_tail: str, now: datetime) -> int:
    """Deliver one unit-failure alert; 0 when sent or suppressed, 1 when it could only be queued."""
    from src.ops.alerts import DispatchOutcome, build_unit_failure, dispatch, redact

    root = Path(data_root)
    moment = _aware(now)
    alerts_dir = _alerts_dir(root)
    issues = _heartbeat_open_issues(root) if unit == "kse-daily-audit.service" else []
    sender, secrets = _alert_sender()
    message = build_unit_failure(
        unit, result=result, exit_status=exit_status, journal_tail=redact(journal_tail, secrets), now=moment,
        open_issues=issues,
    )
    _drain_best_effort(sender, alerts_dir, secrets, moment)
    outcome = dispatch(message, sender=sender, alerts_dir=alerts_dir, now=moment, secret_values=secrets)
    return 0 if outcome in (DispatchOutcome.SENT, DispatchOutcome.SUPPRESSED) else 1


def audit(data_root: Path, *, today: date, now: datetime) -> int:
    """Run the daily audit and deliver its summary; 0 unless an open issue could not be delivered."""
    from src.ops.alerts import DispatchOutcome, build_audit_summary, count_undelivered, dispatch

    root = Path(data_root)
    moment = _aware(now)
    alerts_dir = _alerts_dir(root)
    sender, secrets = _alert_sender()
    _drain_best_effort(sender, alerts_dir, secrets, moment)
    heartbeat = run_daily_audit(root, today=today, now=moment, undelivered_alerts=count_undelivered(alerts_dir))
    message = build_audit_summary(heartbeat)
    if message is None:
        _LOG.info("[OPS] command=audit severity=%s issues=0", str(heartbeat.get("severity", "")))
        return 0
    outcome = dispatch(message, sender=sender, alerts_dir=alerts_dir, now=moment, secret_values=secrets)
    issues = heartbeat.get("open_issues")
    count = len(issues) if isinstance(issues, list) else 0
    _LOG.info("[OPS] command=audit severity=%s issues=%d outcome=%s", str(heartbeat.get("severity", "")), count, outcome.name)
    return 0 if outcome in (DispatchOutcome.SENT, DispatchOutcome.SUPPRESSED) else 1


def weekly_report(data_root: Path, *, today: date, now: datetime) -> int:
    """Aggregate the last seven days of outcomes into a weekly report file."""
    from datetime import timedelta

    from src.ops.run_outcome import load_run_outcomes

    root = Path(data_root)
    moment = _aware(now)
    days = [today - timedelta(days=offset) for offset in range(7)]
    summary = {day.isoformat(): load_run_outcomes(_events_root(root), day) for day in days}
    report = {
        "week_ending": today.isoformat(),
        "generated_at": moment.astimezone(KRX_TZ).isoformat(),
        "days": summary,
    }
    target = root / "ops" / "reports" / f"weekly-{today.isoformat()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(target.parent), prefix=".weekly.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(json.dumps(report, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        temporary = Path(handle.name)
    temporary.replace(target)
    sys.stdout.write(json.dumps(report, sort_keys=True, ensure_ascii=False) + "\n")
    return 0


def _default_data_root() -> Path:
    from src.config.runtime import load_runtime_config

    return Path(load_runtime_config().data_root)


def _read_journal_tail(limit: int = 65536) -> str:
    try:
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        data = stream.read(limit)
    except OSError:
        return ""
    raw = data.encode("utf-8", errors="replace") if isinstance(data, str) else bytes(data)
    if len(raw) > limit:
        raw = raw[:limit]
    return raw.decode("utf-8", errors="replace")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src.ops.jobs")
    parser.add_argument("--data-root", type=Path, required=False, default=None)
    parser.add_argument("--date", type=str, required=False, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect-eve")
    sub.add_parser("collect-dbsec")
    sub.add_parser("morning")
    prune = sub.add_parser("backup")
    prune.add_argument("--no-lock", action="store_true")
    sub.add_parser("audit")
    sub.add_parser("weekly-report")
    failure = sub.add_parser("alert")
    failure.add_argument("--unit", required=True)
    failure.add_argument("--result", default="unknown")
    failure.add_argument("--exit-status", default="unknown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch one scheduled operations command; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv)
    data_root = Path(args.data_root) if args.data_root is not None else _default_data_root()
    today = date.fromisoformat(args.date) if args.date else datetime.now(KRX_TZ).date()
    moment = datetime.now(KRX_TZ)
    if args.command == "collect-eve":
        return collect_eve(data_root, today=today, now=moment)
    if args.command == "collect-dbsec":
        return collect_dbsec(data_root, today=today, now=moment)
    if args.command == "morning":
        return morning(data_root, today=today, now=lambda: datetime.now(KRX_TZ), sleep=time.sleep)
    if args.command == "backup":
        return backup(data_root, now=moment, no_lock=args.no_lock)
    if args.command == "audit":
        return audit(data_root, today=today, now=moment)
    if args.command == "weekly-report":
        return weekly_report(data_root, today=today, now=moment)
    if args.command == "alert":
        return alert(
            data_root, unit=str(args.unit), result=str(args.result),
            exit_status=str(args.exit_status), journal_tail=_read_journal_tail(), now=moment,
        )
    raise SystemExit(f"unknown command {args.command}")  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":  # pragma: no cover - entrypoint glue
    raise SystemExit(main())
