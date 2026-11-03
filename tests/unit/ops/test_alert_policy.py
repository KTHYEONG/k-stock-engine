"""Operator page policy invariants: what pages, page shape, streaks, incident state."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops.alert_policy import (
    DEFAULT_ALERT_POLICY,
    FAILURE_LEDGER_RELPATH,
    INCIDENT_STATE_RELPATH,
    PAGE_TEXT,
    AlertPolicy,
    IncidentKind,
    audit_incidents,
    backup_is_stale,
    build_page,
    build_resolved,
    consecutive_failure_days,
    evaluated_kinds,
    incident_key,
    load_open_incidents,
    record_unit_failure,
    save_open_incidents,
    streak_issue_key,
    unit_incidents,
)
from src.ops.run_outcome import record_run_outcome

KRX_TZ = ZoneInfo("Asia/Seoul")
THU = date(2026, 10, 8)
EVE = "kse-collect-eve.service"


def _at(day: date, hour: int = 19) -> datetime:
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=KRX_TZ)


def _fail(root: Path, day: date, unit: str = EVE) -> None:
    record_unit_failure(root, unit=unit, result="exit-code", exit_status="1", cause="boom", now=_at(day))


def _outcome(root: Path, day: date, outcome: str, job: str = "collect_eve") -> None:
    record_run_outcome(root / "ops" / "events", job, outcome, run_date=day, now=_at(day),
                       reason="r" if outcome == "SKIPPED" else "")


def test_every_kind_has_three_line_korean_page_without_paths() -> None:
    assert set(PAGE_TEXT) == set(IncidentKind)
    for kind in IncidentKind:
        page = build_page(kind, THU)
        assert page.subject.startswith("[긴급][kse] ")
        lines = page.body.split("\n")
        assert [line.split(": ", 1)[0] for line in lines] == ["무슨 일", "자동 조치", "지금 할 일"]
        assert "{days}" not in page.subject + page.body
        assert "/" not in page.body
        assert "~" not in page.body
        assert page.key == incident_key(kind, THU) == f"incident|{kind.value}|2026-10-08"


def test_streak_text_follows_policy_threshold() -> None:
    page = build_page(IncidentKind.COLLECT_EVE_STREAK, THU, AlertPolicy(streak_threshold=4))
    assert "4거래일" in page.subject
    assert "4거래일" in page.body


def test_resolved_notice_is_one_line() -> None:
    notice = build_resolved(IncidentKind.BACKUP_STALE, THU)
    assert notice.subject == "[해결][kse] 오프사이트 백업 2일 이상 중단"
    assert notice.body == "해결됨: 오프사이트 백업 2일 이상 중단"
    assert notice.key == "resolved|backup_stale|2026-10-08"


@pytest.mark.parametrize(
    "kwargs",
    [{"streak_threshold": 0}, {"streak_threshold": 5, "streak_lookback_days": 4},
     {"backup_stale_after": timedelta(0)}],
)
def test_policy_rejects_incoherent_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must"):
        AlertPolicy(**kwargs)  # type: ignore[arg-type]


def test_audit_incidents_map_only_critical_keys_once() -> None:
    issues = [
        {"key": "late_seal"}, {"key": "decision"}, {"key": "divergence"}, {"key": "backup"},
        {"key": "bronze_loose"}, {"key": "footprint"}, {"key": "morning"}, {"key": "decision"},
        {"key": "events"}, {"key": "backup_stale"}, {"key": streak_issue_key("auction_open")},
    ]
    assert audit_incidents(issues) == (
        IncidentKind.NO_SEAL, IncidentKind.EVENTS_UNREADABLE, IncidentKind.BACKUP_STALE,
        IncidentKind.AUCTION_OPEN_STREAK,
    )
    assert audit_incidents([{"text": "no key"}]) == ()


def test_closed_day_evaluates_only_backup() -> None:
    assert evaluated_kinds("trading") == frozenset(IncidentKind)
    assert evaluated_kinds("closed") == frozenset({IncidentKind.BACKUP_STALE})


def test_backup_staleness_uses_last_success() -> None:
    now = _at(THU, 21)
    two_days = (now - timedelta(days=2)).isoformat()
    fresh = (now - timedelta(hours=30)).isoformat()
    assert backup_is_stale({"rc": 0, "finished_at": two_days}, now)
    assert not backup_is_stale({"rc": 0, "finished_at": fresh}, now)
    assert backup_is_stale({"rc": 1, "finished_at": fresh, "last_ok_at": two_days}, now)
    assert not backup_is_stale({"rc": 1, "finished_at": two_days, "last_ok_at": fresh}, now)
    assert not backup_is_stale({"rc": 1, "last_ok_at": None}, now)
    assert not backup_is_stale({"rc": 0, "finished_at": "not-a-date"}, now)
    assert not backup_is_stale({"rc": 0, "finished_at": "2026-10-01T00:00:00"}, now)
    assert backup_is_stale({"rc": 0, "finished_at": two_days}, now.replace(tzinfo=None))


def test_streak_counts_failed_days_across_weekend(tmp_path: Path) -> None:
    fri, mon, tue = date(2026, 10, 2), date(2026, 10, 5), date(2026, 10, 6)
    for day in (fri, mon, tue):
        _fail(tmp_path, day)
    _outcome(tmp_path, date(2026, 10, 3), "SKIPPED")
    assert consecutive_failure_days("collect_eve", EVE, today=tue, data_root=tmp_path) == 3
    assert consecutive_failure_days("collect_eve", EVE, today=date(2026, 10, 7), data_root=tmp_path) == 0


def test_streak_breaks_on_success_or_soft_degraded(tmp_path: Path) -> None:
    _fail(tmp_path, date(2026, 10, 6))
    _outcome(tmp_path, date(2026, 10, 7), "DEGRADED")
    _fail(tmp_path, THU)
    assert consecutive_failure_days("collect_eve", EVE, today=THU, data_root=tmp_path) == 1
    _outcome(tmp_path, date(2026, 10, 7), "OK")
    assert consecutive_failure_days("collect_eve", EVE, today=THU, data_root=tmp_path) == 1


def test_rerun_ok_clears_the_failed_day(tmp_path: Path) -> None:
    _fail(tmp_path, THU)
    _outcome(tmp_path, THU, "OK")
    assert consecutive_failure_days("collect_eve", EVE, today=THU, data_root=tmp_path) == 0


def test_streak_ignores_other_units_and_corrupt_lines(tmp_path: Path) -> None:
    _fail(tmp_path, THU)
    _fail(tmp_path, date(2026, 10, 7), unit="kse-backup.service")
    ledger = tmp_path / FAILURE_LEDGER_RELPATH
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("not json\n[1]\n" + json.dumps({"unit": EVE, "date": "bad"}) + "\n")
    events = tmp_path / "ops" / "events" / "2026-10" / "2026-10-07.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("{broken\n", encoding="utf-8")
    assert consecutive_failure_days("collect_eve", EVE, today=THU, data_root=tmp_path) == 1


def test_streak_is_bounded_by_lookback(tmp_path: Path) -> None:
    policy = AlertPolicy(streak_threshold=2, streak_lookback_days=2)
    for offset in range(5):
        _fail(tmp_path, THU - timedelta(days=offset))
    assert consecutive_failure_days("collect_eve", EVE, today=THU, data_root=tmp_path, policy=policy) == 2


def test_unit_classification(tmp_path: Path) -> None:
    assert unit_incidents("kse-morning.service", today=THU, data_root=tmp_path) == ()
    assert unit_incidents("kse-morning-late.service", today=THU, data_root=tmp_path) == (IncidentKind.NO_SEAL,)
    for unit in ("kse-backup.service", "kse-weekly-report.service", "kse-selftest.service"):
        assert unit_incidents(unit, today=THU, data_root=tmp_path) == ()
    assert unit_incidents("kse-daily-audit.service", today=THU, data_root=tmp_path) == ()
    assert unit_incidents(
        "kse-daily-audit.service", today=THU, data_root=tmp_path, audit_critical=["backup_stale", "bogus"],
    ) == (IncidentKind.BACKUP_STALE,)


def test_streak_unit_pages_only_at_threshold(tmp_path: Path) -> None:
    _fail(tmp_path, date(2026, 10, 7))
    _fail(tmp_path, THU)
    assert unit_incidents(EVE, today=THU, data_root=tmp_path) == ()
    _fail(tmp_path, date(2026, 10, 6))
    assert unit_incidents(EVE, today=THU, data_root=tmp_path) == (IncidentKind.COLLECT_EVE_STREAK,)
    assert DEFAULT_ALERT_POLICY.streak_threshold == 3


def test_failure_ledger_record_shape(tmp_path: Path) -> None:
    record_unit_failure(tmp_path, unit=EVE, result="timeout", exit_status="", cause="c",
                        now=datetime(2026, 10, 8, 23, 59))
    record = json.loads((tmp_path / FAILURE_LEDGER_RELPATH).read_text(encoding="utf-8"))
    assert record == {"ts": "2026-10-08T23:59:00+09:00", "date": "2026-10-08", "unit": EVE,
                      "result": "timeout", "exit_status": "", "cause": "c"}


def test_open_incident_state_roundtrip_and_tolerance(tmp_path: Path) -> None:
    assert load_open_incidents(tmp_path) == {}
    save_open_incidents(tmp_path, {IncidentKind.NO_SEAL: "2026-10-08"})
    assert load_open_incidents(tmp_path) == {IncidentKind.NO_SEAL: "2026-10-08"}
    target = tmp_path / INCIDENT_STATE_RELPATH
    target.write_text(json.dumps({"open": {"no_seal": "d", "retired_kind": "d"}}), encoding="utf-8")
    assert load_open_incidents(tmp_path) == {IncidentKind.NO_SEAL: "d"}
    target.write_text(json.dumps({"open": []}), encoding="utf-8")
    assert load_open_incidents(tmp_path) == {}
    target.write_text("[]", encoding="utf-8")
    assert load_open_incidents(tmp_path) == {}
    target.write_text("{", encoding="utf-8")
    assert load_open_incidents(tmp_path) == {}
