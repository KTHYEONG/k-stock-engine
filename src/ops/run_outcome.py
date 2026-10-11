"""Append-only per-day outcome events for scheduled operations jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Final

from src.core.time import KRX_TZ

__all__ = ["OUTCOMES", "load_run_outcomes", "record_run_outcome"]

OUTCOMES: Final = ("OK", "DEGRADED", "NO_DECISION", "SKIPPED")

_MAX_REASON_CHARS: Final = 300


def record_run_outcome(
    events_root: Path,
    job: str,
    outcome: str,
    *,
    run_date: date,
    now: datetime,
    reason: str = "",
    metrics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Append one outcome event to ``events_root/<YYYY-MM>/<run_date>.jsonl`` and return it."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    if not job or not job.strip():
        raise ValueError("job must be a non-empty name")
    if outcome == "SKIPPED" and not reason.strip():
        raise ValueError("SKIPPED outcome requires a reason")
    stamped = now if now.tzinfo is not None else now.replace(tzinfo=KRX_TZ)
    event: dict[str, object] = {
        "ts": stamped.astimezone(KRX_TZ).isoformat(timespec="seconds"),
        "job": job,
        "run_date": run_date.isoformat(),
        "outcome": outcome,
        "reason": reason[:_MAX_REASON_CHARS],
        "metrics": dict(metrics) if metrics is not None else {},
    }
    target = Path(events_root) / run_date.strftime("%Y-%m") / f"{run_date.isoformat()}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return event


def load_run_outcomes(events_root: Path, run_date: date) -> dict[str, str]:
    """Latest outcome per job for the date."""
    target = Path(events_root) / run_date.strftime("%Y-%m") / f"{run_date.isoformat()}.jsonl"
    latest: dict[str, str] = {}
    if not target.is_file():
        return latest
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        latest[str(payload["job"])] = str(payload["outcome"])
    return latest
