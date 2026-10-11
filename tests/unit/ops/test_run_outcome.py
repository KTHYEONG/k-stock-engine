"""Outcome event append and reload invariants."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops.run_outcome import OUTCOMES, load_run_outcomes, record_run_outcome

KRX_TZ = ZoneInfo("Asia/Seoul")
DAY = date(2026, 10, 8)


def _now() -> datetime:
    return datetime(2026, 10, 8, 9, 5, 7, tzinfo=KRX_TZ)


def test_outcome_events_append_and_reload(tmp_path: Path) -> None:
    first = record_run_outcome(tmp_path, "morning", "OK", run_date=DAY, now=_now())
    later = record_run_outcome(
        tmp_path, "morning", "DEGRADED", run_date=DAY, now=_now(), reason="late_seal", metrics={"wait_s": 1.0}
    )
    assert load_run_outcomes(tmp_path, DAY) == {"morning": "DEGRADED"}
    target = tmp_path / "2026-10" / "2026-10-08.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert load_run_outcomes(tmp_path, DAY) == {"morning": "DEGRADED"}
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len([line for line in lines if line.strip()]) == 2
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        assert set(payload) == {"ts", "job", "run_date", "outcome", "reason", "metrics"}
    assert json.loads(lines[0])["outcome"] == "OK"
    assert first["reason"] == ""
    assert later["metrics"] == {"wait_s": 1.0}
    assert set(OUTCOMES) == {"OK", "DEGRADED", "NO_DECISION", "SKIPPED"}


def test_skipped_needs_a_reason(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reason"):
        record_run_outcome(tmp_path, "morning", "SKIPPED", run_date=DAY, now=_now())
    event = record_run_outcome(tmp_path, "morning", "SKIPPED", run_date=DAY, now=_now(), reason="non_trading_day")
    assert event["outcome"] == "SKIPPED"
    assert load_run_outcomes(tmp_path, DAY) == {"morning": "SKIPPED"}


def test_unknown_outcome_rejected_and_reason_truncated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        record_run_outcome(tmp_path, "morning", "BOGUS", run_date=DAY, now=_now())
    with pytest.raises(ValueError, match="non-empty"):
        record_run_outcome(tmp_path, "", "OK", run_date=DAY, now=_now())
    event = record_run_outcome(tmp_path, "morning", "OK", run_date=DAY, now=_now(), reason="x" * 400)
    assert len(str(event["reason"])) == 300


def test_missing_file_loads_empty_and_naive_now_assumed_kst(tmp_path: Path) -> None:
    assert load_run_outcomes(tmp_path, DAY) == {}
    naive = datetime(2026, 10, 8, 9, 5, 7)
    event = record_run_outcome(tmp_path, "backup", "OK", run_date=DAY, now=naive)
    assert str(event["ts"]).endswith("+09:00")
