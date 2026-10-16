"""Daily audit over outcome events and artefacts with a dashboard heartbeat."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

from src.core.time import KRX_TZ
from src.data.runtime import resolve_data_runtime
from src.ops.guards import trading_day
from src.ops.run_outcome import load_run_outcomes

__all__ = ["AUDIT_HEARTBEAT_RELPATH", "EXPECTED_JOBS", "run_daily_audit"]

AUDIT_HEARTBEAT_RELPATH: Final = "ops/heartbeat/daily_audit.json"
EXPECTED_JOBS: Final[tuple[str, ...]] = ("collect_eve", "collect_dbsec", "morning", "auction_open", "backup")

_BACKUP_MAX_AGE: Final = timedelta(hours=26)
_LOOSE_BRONZE_MAX_AGE: Final = timedelta(days=14)
_MAX_FOOTPRINT_BYTES: Final = 10 * 1024**3
_PACK_DIRNAME: Final = "_packs"
_CATALOG_DIRNAME: Final = "catalog"
_OPEN_HOUR: Final = 9


def _events_root(data_root: Path) -> Path:
    return Path(data_root) / "ops" / "events"


def _heartbeat_path(data_root: Path) -> Path:
    return Path(data_root) / AUDIT_HEARTBEAT_RELPATH


def _backup_status_path(data_root: Path) -> Path:
    return Path(data_root) / "state" / "lake_backup_status.json"


def _workspace_paths(data_root: Path) -> tuple[Path, Path, Path]:
    """Decisions dir, shadow dir and Bronze root for the configured scope."""
    runtime = resolve_data_runtime(data_root=Path(data_root))
    state_root = runtime.workspace.state_root
    return (state_root / "decisions", state_root / "shadow", runtime.workspace.bronze_root)


def _measure_data_root_bytes(root: Path) -> int:
    total = 0
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def _fresh_divergences(shadow_dir: Path, yesterday: date) -> list[str]:
    found: list[str] = []
    if not shadow_dir.is_dir():
        return found
    for path in sorted(shadow_dir.rglob("divergence-*.json")):
        try:
            if not path.is_file():
                continue
            stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=KRX_TZ).date()
        except OSError:
            continue
        if stamp > yesterday:
            found.append(path.name)
    return found


def _stale_loose_bronze(bronze_root: Path, now: datetime) -> list[str]:
    stale: list[str] = []
    if not bronze_root.is_dir():
        return stale
    cutoff = now.timestamp() - _LOOSE_BRONZE_MAX_AGE.total_seconds()
    for folder, dirs, files in os.walk(bronze_root):
        dirs[:] = sorted(d for d in dirs if d not in (_PACK_DIRNAME, _CATALOG_DIRNAME))
        for name in sorted(files):
            path = Path(folder) / name
            try:
                if path.stat().st_mtime < cutoff:
                    stale.append(path.relative_to(bronze_root).as_posix())
            except OSError:
                continue
    return stale


def _backup_issue(status_path: Path, now: datetime) -> str | None:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "backup status is missing or unreadable"
    try:
        finished = datetime.fromisoformat(str(payload["finished_at"]))
        rc = int(payload["rc"])
    except (KeyError, TypeError, ValueError):
        return "backup status is malformed"
    if finished.tzinfo is None:
        return "backup status finished_at is naive"
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    if rc != 0:
        return f"backup last finished with rc={rc}"
    if finished > moment:
        return "backup timestamp is in the future"
    if moment.astimezone(finished.tzinfo) - finished > _BACKUP_MAX_AGE:
        return "backup is older than 26 h"
    return None


def _decision_issues(decisions_dir: Path, today: date) -> list[tuple[str, str]]:
    from src.research.operate import read_chain

    try:
        chain = read_chain(Path(decisions_dir))
    except ValueError as exc:
        return [("decision", f"decision chain is broken: {type(exc).__name__}")]
    todays = [item for item in chain if item.execution_session == today]
    if not todays:
        return [("decision", "no sealed decision for today")]
    latest = max(todays, key=lambda item: item.sealed_at)
    sealed = latest.sealed_at if latest.sealed_at.tzinfo is not None else latest.sealed_at.replace(tzinfo=KRX_TZ)
    opened = datetime(today.year, today.month, today.day, _OPEN_HOUR, tzinfo=KRX_TZ)
    if sealed.astimezone(KRX_TZ) >= opened:
        return [("late_seal", "latest decision was sealed after the open")]
    return []


def run_daily_audit(data_root: Path, *, today: date, now: datetime, undelivered_alerts: int = 0) -> dict[str, object]:
    """Check the day's outcomes and artefacts, write the heartbeat, and return it."""
    root = Path(data_root)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    queued = undelivered_alerts if isinstance(undelivered_alerts, int) and not isinstance(undelivered_alerts, bool) and undelivered_alerts >= 0 else 0
    open_issues: list[dict[str, str]] = []
    if trading_day(today):
        day_kind = "trading"
        try:
            outcomes = load_run_outcomes(_events_root(root), today)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            outcomes = {}
            open_issues.append({"key": "events", "text": f"outcomes unreadable: {type(exc).__name__}"})
        for job in EXPECTED_JOBS:
            outcome = outcomes.get(job)
            if outcome is None:
                open_issues.append({"key": job, "text": f"no outcome for {job}"})
            elif outcome in ("DEGRADED", "NO_DECISION"):
                open_issues.append({"key": job, "text": f"{job} ended {outcome}"})
        decisions_dir, shadow_dir, bronze_root = _workspace_paths(root)
        open_issues.extend({"key": key, "text": text} for key, text in _decision_issues(decisions_dir, today))
        fresh = _fresh_divergences(shadow_dir, today - timedelta(days=1))
        if fresh:
            open_issues.append({"key": "divergence", "text": f"{len(fresh)} fresh divergence file(s)"})
        problem = _backup_issue(_backup_status_path(root), moment)
        if problem is not None:
            open_issues.append({"key": "backup", "text": problem})
        stale = _stale_loose_bronze(bronze_root, moment)
        if stale:
            open_issues.append({"key": "bronze_loose", "text": f"{len(stale)} loose Bronze file(s) older than 14 days"})
        if _measure_data_root_bytes(root) > _MAX_FOOTPRINT_BYTES:
            open_issues.append({"key": "footprint", "text": "data root is above 10 GiB"})
        severity = "OK" if not open_issues else "WARNING"
    else:
        day_kind = "closed"
        problem = _backup_issue(_backup_status_path(root), moment)
        if problem is not None:
            open_issues.append({"key": "backup", "text": problem})
        severity = "HOLIDAY_SKIP"
    if open_issues:
        subject = f"kse 일일 감사 경고 {len(open_issues)}건: " + ", ".join(item["key"] for item in open_issues)
    elif day_kind == "closed":
        subject = "kse 일일 감사 휴장"
    else:
        subject = "kse 일일 감사 정상"
    heartbeat: dict[str, object] = {
        "schema_version": 1,
        "finished_at": moment.astimezone(KRX_TZ).isoformat(),
        "subject": subject,
        "severity": severity,
        "day_kind": day_kind,
        "snapshot_date": today.isoformat(),
        "undelivered_alerts": queued,
        "open_issues": open_issues,
    }
    target = _heartbeat_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(target.parent), prefix=".daily_audit.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(json.dumps(heartbeat, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    os.chmod(target, 0o644)
    return heartbeat
