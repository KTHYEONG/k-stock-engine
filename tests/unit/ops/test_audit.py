"""Daily audit heartbeat contract invariants."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import src.ops.audit as audit_mod
from src.ops.audit import (
    AUDIT_HEARTBEAT_RELPATH,
    EXPECTED_JOBS,
    _backup_issue,
    _fresh_divergences,
    _measure_data_root_bytes,
    _stale_loose_bronze,
    _workspace_paths,
    run_daily_audit,
)
from src.ops.run_outcome import record_run_outcome

KRX_TZ = ZoneInfo("Asia/Seoul")
DAY = date(2026, 10, 8)


def _morning() -> datetime:
    return datetime(2026, 10, 8, 10, 0, tzinfo=KRX_TZ)


def _write_status(root: Path, *, rc: int = 0, age_h: float = 1.0, broken: bool = False) -> Path:
    target = root / "state" / "lake_backup_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if broken:
        target.write_text("not json", encoding="utf-8")
        return target
    finished = (_morning() - timedelta(hours=age_h)).astimezone(ZoneInfo("UTC"))
    target.write_text(
        json.dumps({"finished_at": finished.isoformat(), "rc": rc, "last_ok_at": finished.isoformat(),
                    "sources": {}}),
        encoding="utf-8",
    )
    return target


def _record_all_ok(root: Path, now: datetime) -> None:
    events = root / "ops" / "events"
    for job in EXPECTED_JOBS:
        record_run_outcome(events, job, "OK", run_date=DAY, now=now)


def _decision(execution: date, hour: int, minute: int) -> SimpleNamespace:
    return SimpleNamespace(
        data_session=execution - timedelta(days=1),
        execution_session=execution,
        sealed_at=datetime(execution.year, execution.month, execution.day, hour, minute, tzinfo=KRX_TZ),
    )


def test_heartbeat_matches_dashboard_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 8, 31),))
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert heartbeat["severity"] == "OK"
    assert heartbeat["day_kind"] == "trading"
    assert heartbeat["schema_version"] == 1
    assert heartbeat["snapshot_date"] == DAY.isoformat()
    assert heartbeat["undelivered_alerts"] == 0
    assert heartbeat["open_issues"] == []
    assert isinstance(heartbeat["subject"], str)
    assert "경고" not in str(heartbeat["subject"])
    assert datetime.fromisoformat(str(heartbeat["finished_at"])).tzinfo is not None
    target = tmp_path / AUDIT_HEARTBEAT_RELPATH
    assert target.is_file()
    assert (target.stat().st_mode & 0o777) == 0o644
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["severity"] == "OK"


def test_missing_morning_outcome_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 8, 31),))
    events = tmp_path / "ops" / "events"
    for job in EXPECTED_JOBS:
        if job != "morning":
            record_run_outcome(events, job, "OK", run_date=DAY, now=_morning())
    _write_status(tmp_path)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert heartbeat["severity"] == "WARNING"
    assert "경고" in str(heartbeat["subject"])
    assert any(item["key"] == "morning" for item in heartbeat["open_issues"])
    record_run_outcome(events, "morning", "NO_DECISION", run_date=DAY, now=_morning(), reason="x")
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert any(item["key"] == "morning" for item in heartbeat["open_issues"])


def test_decision_and_late_seal_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: ())
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert any(item["key"] == "decision" for item in heartbeat["open_issues"])
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 9, 4),))
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert any(item["key"] == "late_seal" for item in heartbeat["open_issues"])

    def _broken(path: Path) -> tuple[()]:
        raise ValueError("broken link")

    monkeypatch.setattr("src.research.operate.read_chain", _broken)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert any(item["key"] == "decision" for item in heartbeat["open_issues"])


def test_divergence_bronze_and_backup_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 8, 31),))
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path, age_h=30.0)
    decisions, shadow, bronze = _workspace_paths(tmp_path)
    assert decisions.name == "decisions"
    assert shadow.name == "shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    fresh = shadow / "official" / "divergence-2026-10-08.json"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_text("{}", encoding="utf-8")
    ghost = shadow / "official" / "divergence-2026-10-09.json"
    ghost.mkdir(parents=True, exist_ok=True)
    stale = shadow / "official" / "divergence-2026-10-06.json"
    stale.write_text("{}", encoding="utf-8")
    old_stamp = (_morning() - timedelta(days=3)).timestamp()
    os.utime(stale, (old_stamp, old_stamp))
    kind = bronze / "daily_market"
    kind.mkdir(parents=True, exist_ok=True)
    loose = kind / "loose.bin"
    loose.write_bytes(b"x" * 8)
    loose_stamp = (_morning() - timedelta(days=20)).timestamp()
    os.utime(loose, (loose_stamp, loose_stamp))
    dangling = kind / "gone.bin"
    os.symlink(tmp_path / "absent-target.bin", dangling)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    keys = {item["key"] for item in heartbeat["open_issues"]}
    assert {"divergence", "bronze_loose"} <= keys
    assert _fresh_divergences(shadow, DAY - timedelta(days=1)) == ["divergence-2026-10-08.json"]
    assert _stale_loose_bronze(bronze, _morning()) == ["daily_market/loose.bin"]
    assert _stale_loose_bronze(tmp_path / "missing", _morning()) == []
    assert _fresh_divergences(tmp_path / "missing", DAY) == []


def test_backup_issue_branches(tmp_path: Path) -> None:
    moment = _morning()
    assert _backup_issue(tmp_path / "absent.json", moment) is not None
    target = _write_status(tmp_path, broken=True)
    assert _backup_issue(target, moment) is not None
    target.write_text(json.dumps({"rc": 0}), encoding="utf-8")
    assert _backup_issue(target, moment) is not None
    naive = tmp_path / "naive.json"
    naive.write_text(json.dumps({"finished_at": "2026-10-08T01:00:00", "rc": 0}), encoding="utf-8")
    assert _backup_issue(naive, moment) is not None
    target = _write_status(tmp_path, rc=2)
    assert _backup_issue(target, moment) is not None
    target = _write_status(tmp_path, age_h=30.0)
    assert _backup_issue(target, moment) is not None
    target = _write_status(tmp_path)
    assert _backup_issue(target, moment) is None
    assert _backup_issue(target, moment.replace(tzinfo=None)) is None


def test_footprint_breach_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 8, 31),))
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path)
    assert _measure_data_root_bytes(tmp_path) > 0
    assert _measure_data_root_bytes(tmp_path / "absent") == 0
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "f.bin").write_bytes(b"x")
    blocked.chmod(0o000)
    try:
        assert _measure_data_root_bytes(tmp_path) >= 0
    finally:
        blocked.chmod(0o755)
    monkeypatch.setattr(audit_mod, "_measure_data_root_bytes", lambda root: 9 * 1024**3)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert not any(item["key"] == "footprint" for item in heartbeat["open_issues"])
    monkeypatch.setattr(audit_mod, "_measure_data_root_bytes", lambda root: 11 * 1024**3)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert any(item["key"] == "footprint" for item in heartbeat["open_issues"])


def test_closed_day_checks_only_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: False)
    saturday = date(2026, 10, 10)
    _write_status(tmp_path)
    heartbeat = run_daily_audit(tmp_path, today=saturday, now=_morning())
    assert heartbeat["severity"] == "HOLIDAY_SKIP"
    assert heartbeat["day_kind"] == "closed"
    assert heartbeat["open_issues"] == []
    assert "경고" not in str(heartbeat["subject"])
    _write_status(tmp_path, age_h=30.0)
    heartbeat = run_daily_audit(tmp_path, today=saturday, now=_morning())
    assert heartbeat["severity"] == "HOLIDAY_SKIP"
    assert "경고" in str(heartbeat["subject"])
    assert any(item["key"] == "backup" for item in heartbeat["open_issues"])


def test_stat_failure_skips_divergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    target = shadow / "divergence-2026-10-08.json"
    target.write_text("{}", encoding="utf-8")
    real_stat = Path.stat

    def _boom(self: Path, *args: object, **kwargs: object) -> object:
        if self.name.startswith("divergence-"):
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _boom)
    assert _fresh_divergences(shadow, DAY - timedelta(days=1)) == []


def test_naive_sealed_at_and_now(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    naive_decision = SimpleNamespace(
        data_session=DAY - timedelta(days=1),
        execution_session=DAY,
        sealed_at=datetime(DAY.year, DAY.month, DAY.day, 8, 31),
    )
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (naive_decision,))
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path)
    heartbeat = run_daily_audit(tmp_path, today=DAY, now=_morning().replace(tzinfo=None))
    assert heartbeat["severity"] == "OK"


def test_corrupt_events_still_write_warning_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    events = tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"truncated":', encoding="utf-8")
    result = run_daily_audit(tmp_path, today=DAY, now=_morning())
    assert result["severity"] == "WARNING"
    assert any(issue["key"] == "events" for issue in result["open_issues"])
    status = _write_status(tmp_path, age_h=-1.0)
    assert _backup_issue(status, _morning()) == "backup timestamp is in the future"


def test_heartbeat_carries_undelivered_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain", lambda path: (_decision(DAY, 8, 31),))
    _record_all_ok(tmp_path, _morning())
    _write_status(tmp_path)
    assert run_daily_audit(tmp_path, today=DAY, now=_morning(), undelivered_alerts=2)["undelivered_alerts"] == 2
    assert run_daily_audit(tmp_path, today=DAY, now=_morning())["undelivered_alerts"] == 0
    assert run_daily_audit(tmp_path, today=DAY, now=_morning(), undelivered_alerts=-1)["undelivered_alerts"] == 0
