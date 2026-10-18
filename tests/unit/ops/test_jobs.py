"""Operations job entrypoint invariants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import src.ops.jobs as jobs
from src.core.pit import PITDataError
from src.ops.guards import GuardRefused
from src.ops.jobs import (
    EVENING_COLLECT_STEPS,
    KRX_COLLECT_STEPS,
    MORNING_RELAY_JOBS,
    audit,
    backup,
    collect_eve,
    main,
    morning,
    weekly_report,
)
from src.ops.run_outcome import load_run_outcomes

KRX_TZ = ZoneInfo("Asia/Seoul")
DAY = date(2026, 10, 8)


def _morning_now() -> datetime:
    return datetime(2026, 10, 8, 7, 30, tzinfo=KRX_TZ)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.current = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)


def _trading(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    monkeypatch.setattr(jobs, "trading_day", lambda day: value)


def _healthy_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "require_headroom", lambda **kwargs: None)


def test_holiday_is_clean_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, False)
    calls: list[str] = []
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: calls.append("refresh"))
    monkeypatch.setattr(jobs, "_wait_for_krx", lambda *a, **k: calls.append("wait") or True)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 0
    assert calls == []
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {
        "collect_eve": "SKIPPED",
        "morning": "SKIPPED",
    }


def test_low_memory_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)

    def _refuse(**kwargs: object) -> None:
        raise GuardRefused("insufficient headroom: MemAvailable 100 MB (need 3500)")

    monkeypatch.setattr(jobs, "require_headroom", _refuse)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 1
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert backup(tmp_path, now=_morning_now()) == 1
    outcomes = load_run_outcomes(tmp_path / "ops" / "events", DAY)
    assert outcomes == {"collect_eve": "DEGRADED", "morning": "DEGRADED", "backup": "DEGRADED"}
    line = (tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl").read_text(encoding="utf-8")
    assert "100 MB" in line


def test_collect_eve_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    assert {"kis_investor_flow", "ls_investor_flow", "dart_facts"} <= EVENING_COLLECT_STEPS
    monkeypatch.setattr(
        jobs, "_run_refresh",
        lambda root, **kwargs: SimpleNamespace(blocking_job=None, planned=["a"], built=["a"]),
    )
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_eve": "OK"}

    def _blocked(root: Path, **kwargs: object) -> SimpleNamespace:
        assert kwargs["collect_steps"] == EVENING_COLLECT_STEPS
        return SimpleNamespace(blocking_job="kis_investor_flow", planned=[], built=[])

    monkeypatch.setattr(jobs, "_run_refresh", _blocked)
    assert collect_eve(tmp_path, today=date(2026, 10, 12), now=_morning_now()) == 1

    def _boom(root: Path, **kwargs: object) -> SimpleNamespace:
        raise PITDataError("catalog unreadable")

    monkeypatch.setattr(jobs, "_run_refresh", _boom)
    assert collect_eve(tmp_path, today=date(2026, 10, 13), now=_morning_now()) == 1


def _morning_stubs(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], sealed_at: datetime
) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_previous_session", lambda day: date(2026, 10, 7))
    monkeypatch.setattr(jobs, "_build_all", lambda root, **kwargs: calls.append("build"))
    monkeypatch.setattr(
        jobs, "_decide_phase", lambda root: calls.append("decide") or ["2026-10-07"]
    )
    monkeypatch.setattr(jobs, "_shadow_phase", lambda root: calls.append("shadow"))
    monkeypatch.setattr(jobs, "_sealed_at_for", lambda decisions, execution: sealed_at)


def test_morning_waits_for_krx_then_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert {"krx_daily_market", "krx_cash_series"} <= KRX_COLLECT_STEPS
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    polls = iter([False, False, True])
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: (calls.append("poll"), next(polls))[1])
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 0
    assert calls == ["poll", "poll", "poll", "build", "decide", "shadow"]
    assert clock.sleeps == [60.0, 60.0]
    outcomes = load_run_outcomes(tmp_path / "ops" / "events", DAY)
    assert outcomes == {"morning": "OK"}


def test_krx_never_arrives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: calls.append("poll") or False)
    clock = _Clock(datetime(2026, 10, 8, 8, 54, tzinfo=KRX_TZ))
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert "decide" not in calls
    assert "shadow" not in calls
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"morning": "NO_DECISION"}


def test_late_seal_is_degraded_not_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 9, 4, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert "decide" in calls
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"morning": "DEGRADED"}


def test_morning_failure_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)

    def _pit(root: Path, **kwargs: object) -> None:
        raise PITDataError("cash return is missing")

    monkeypatch.setattr(jobs, "_build_all", _pit)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"morning": "NO_DECISION"}

    def _boom(root: Path, **kwargs: object) -> None:
        raise RuntimeError("scorer missing")

    monkeypatch.setattr(jobs, "_build_all", lambda root, **kwargs: None)
    monkeypatch.setattr(jobs, "_decide_phase", _boom)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=date(2026, 10, 12), now=clock.now, sleep=clock.sleep) == 1

    def _no_prev(day: date) -> date:
        raise PITDataError("no session before")

    monkeypatch.setattr(jobs, "_previous_session", _no_prev)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=date(2026, 10, 13), now=clock.now, sleep=clock.sleep) == 1


def test_poll_and_build_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as graph

    seen: dict[str, object] = {}

    def _refresh(root: Path, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(jobs, "_run_refresh", _refresh)
    monkeypatch.setattr(graph, "run_collection_jobs", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(graph, "_default_collection_steps", lambda: (("krx_daily_market", lambda: None),))
    monkeypatch.setattr(jobs, "_data_session_present", lambda root, session: session == DAY)
    assert jobs._poll_krx_once(tmp_path, DAY, _morning_now()) is True
    assert seen["steps"][0][0] == "krx_daily_market"
    assert jobs._poll_krx_once(tmp_path, date(2026, 10, 7), _morning_now()) is False
    jobs._build_all(tmp_path, decision_time=_morning_now())
    assert seen["collect"] is False
    assert seen["collect_steps"] is None


def test_wait_for_krx_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(_morning_now())
    deadline = datetime(2026, 10, 8, 8, 55, tzinfo=KRX_TZ)
    answers = iter([False, True])
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: next(answers))
    assert jobs._wait_for_krx(tmp_path, DAY, now=clock.now, sleep=clock.sleep, deadline=deadline) is True
    assert clock.sleeps == [60.0]
    clock = _Clock(deadline + timedelta(seconds=1))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: False)
    assert jobs._wait_for_krx(tmp_path, DAY, now=clock.now, sleep=clock.sleep, deadline=deadline) is False


def test_backup_failure_removes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _healthy_guard(monkeypatch)
    pruned: list[str] = []
    monkeypatch.setattr(jobs, "_prune_bronze", lambda *a, **k: pruned.append("prune") or {})

    def _fail(**kwargs: object) -> dict[str, object]:
        raise PITDataError("lake backup push failed rc=1")

    monkeypatch.setattr(jobs, "_push_backup_sources", _fail)
    assert backup(tmp_path, now=_morning_now()) == 1
    assert pruned == []
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"backup": "DEGRADED"}

    monkeypatch.setattr(jobs, "_push_backup_sources", lambda **kwargs: {"bronze": {}})

    def _prune_fail(root: Path, **kwargs: object) -> dict[str, object]:
        raise PITDataError("pack failed")

    monkeypatch.setattr(jobs, "_prune_bronze", _prune_fail)
    assert backup(tmp_path, now=datetime(2026, 10, 8, 2, 30, tzinfo=KRX_TZ)) == 1

    monkeypatch.setattr(jobs, "_prune_bronze", lambda *a, **k: {"removed": 3})
    assert backup(tmp_path, now=datetime(2026, 10, 8, 2, 30, tzinfo=KRX_TZ)) == 0


def test_backup_failure_records_redacted_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _healthy_guard(monkeypatch)
    monkeypatch.setenv("DRIVE_API_TOKEN", "s3cr3t-token-value")

    def _fail(**kwargs: object) -> dict[str, object]:
        raise PITDataError("lake backup push failed rc=1: auth s3cr3t-token-value rejected" + "x" * 500)

    monkeypatch.setattr(jobs, "_push_backup_sources", _fail)
    assert backup(tmp_path, now=_morning_now()) == 1
    records = [
        json.loads(line)
        for path in (tmp_path / "ops" / "events").rglob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    detail = next(item for item in records if item["job"] == "backup")["metrics"]["detail"]
    assert "rc=1" in detail
    assert "s3cr3t-token-value" not in detail
    assert len(detail) <= 300


def test_pruning_is_gated_by_offsite_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.runtime import resolve_data_runtime

    bronze = resolve_data_runtime(data_root=tmp_path).workspace.bronze_root
    (bronze / "catalog").mkdir(parents=True, exist_ok=True)
    kind = bronze / "daily_market"
    kind.mkdir(parents=True, exist_ok=True)
    confirmed = kind / "confirmed.bin"
    unconfirmed = kind / "unconfirmed.bin"
    confirmed.write_bytes(b"a" * 64)
    unconfirmed.write_bytes(b"b" * 64)
    old = (_morning_now() - timedelta(days=20)).timestamp()
    os.utime(confirmed, (old, old))
    os.utime(unconfirmed, (old, old))
    report = jobs._prune_bronze(tmp_path, now=_morning_now(), confirmed={"daily_market/confirmed.bin"})
    assert report["removed"] == 1
    assert not confirmed.exists()
    assert unconfirmed.exists()

    flaky = kind / "flaky.bin"
    flaky.write_bytes(b"c" * 64)
    os.utime(flaky, (old, old))
    real_older = jobs._is_older_than

    def _flaky_older(path: Path, cutoff_ts: float) -> bool:
        if path.name == "flaky.bin":
            raise OSError("gone")
        return real_older(path, cutoff_ts)

    monkeypatch.setattr(jobs, "_is_older_than", _flaky_older)
    report = jobs._prune_bronze(
        tmp_path, now=_morning_now(), confirmed={"daily_market/confirmed.bin", "daily_market/flaky.bin"}
    )
    assert flaky.exists()


def test_push_backup_sources_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status = tmp_path / "status.json"

    def _ok(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--no-lock" in command
        assert "--status" in command
        assert command.count("--source") == 4
        assert str(jobs._tool_path()) in command[1]
        payload = json.dumps({"bronze": {"scanned": 1}})
        return subprocess.CompletedProcess(command, 0, stdout=payload + "\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", _ok)
    assert jobs._push_backup_sources(status_path=status, no_lock=True) == {"bronze": {"scanned": 1}}

    def _timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1.0)

    monkeypatch.setattr(jobs.subprocess, "run", _timeout)
    with pytest.raises(PITDataError, match="timed out"):
        jobs._push_backup_sources(status_path=status, no_lock=False)

    def _rc1(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(jobs.subprocess, "run", _rc1)
    with pytest.raises(PITDataError, match="rc=1"):
        jobs._push_backup_sources(status_path=status, no_lock=True)

    def _bad(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="not json\n", stderr="")

    monkeypatch.setattr(jobs.subprocess, "run", _bad)
    with pytest.raises(PITDataError, match="no report"):
        jobs._push_backup_sources(status_path=status, no_lock=True)


def test_confirmed_paths_and_session_presence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import lake_backup as lake

    entries = [SimpleNamespace(path="a.bin"), SimpleNamespace(path="b.bin")]
    monkeypatch.setattr(lake, "RcloneStore", lambda remote, rclone_bin: SimpleNamespace(remote=remote))
    monkeypatch.setattr(lake, "_load_index", lambda store, work: entries)
    (tmp_path / "c.toml").write_text('remote = "gdrive:x"\n', encoding="utf-8")
    paths = jobs._confirmed_bronze_paths(config_path=tmp_path / "c.toml", work_root=tmp_path)
    assert paths == {"a.bin", "b.bin"}

    assert jobs._data_session_present(tmp_path, DAY) is False

    from src.data.receipt_catalog import EvidenceStatus

    class _Catalog:
        def __init__(self, root: Path) -> None:
            self._root = root

        def latest(self, *, source: str, natural_keys: set[str]) -> dict[str, SimpleNamespace]:
            assert source == "krx_daily_market"
            return {DAY.isoformat(): SimpleNamespace(status=EvidenceStatus.SUCCESS)}

    monkeypatch.setattr("src.data.receipt_catalog.ReceiptCatalog", _Catalog)
    assert jobs._data_session_present(tmp_path, DAY) is True

    def _fail(**kwargs: object) -> object:
        raise PITDataError("catalog locked")

    monkeypatch.setattr("src.data.runtime.resolve_data_runtime", _fail)
    assert jobs._data_session_present(tmp_path, DAY) is False


def test_run_refresh_and_research_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.pipeline_graph as graph
    import src.research.cli as research_cli
    import src.research.pipeline as pipeline

    seen: dict[str, object] = {}
    real_ctx = graph.build_refresh_context

    def _ctx(runtime: object, **kwargs: object) -> object:
        seen["runtime"] = runtime
        return real_ctx(runtime, **kwargs)

    def _scope(ctx: object, **kwargs: object) -> object:
        seen["scope"] = kwargs
        seen["ctx"] = ctx
        return SimpleNamespace(status="ok")

    monkeypatch.setattr(graph, "build_refresh_context", _ctx)
    monkeypatch.setattr(graph, "refresh_scope", _scope)
    jobs._run_refresh(tmp_path, decision_time=_morning_now(), collect=True, collect_steps=frozenset({"a"}))
    assert seen["scope"]["collect_steps"] == frozenset({"a"})
    jobs._run_refresh(tmp_path, decision_time=_morning_now(), collect=True, collect_steps=EVENING_COLLECT_STEPS)
    assert "daily_market" in seen["ctx"].deferred_build_kinds

    paths = jobs._research_paths(tmp_path)
    assert paths.decisions.name == "decisions"
    assert paths.shadow.name == "shadow"
    assert paths.state.name == "state.json"
    assert jobs._backup_config_path().name == "lake_backup.toml"

    monkeypatch.setattr(research_cli, "_load_spec", lambda path: SimpleNamespace(spec_hash="s"))
    monkeypatch.setattr(pipeline, "_scores_identity", lambda spec: "i" * 20)
    base = jobs._models_base(tmp_path / "spec.toml", tmp_path)
    assert base.name == "i" * 16

    monkeypatch.setattr(research_cli, "_run_decide", lambda args: {"sealed": ["2026-10-07"], "extra": 1})
    assert jobs._decide_phase(tmp_path) == ["2026-10-07"]
    monkeypatch.setattr(research_cli, "_run_decide", lambda args: {"sealed": "oops"})
    assert jobs._decide_phase(tmp_path) == []
    shadowed: list[str] = []
    monkeypatch.setattr(research_cli, "_run_shadow", lambda args: shadowed.append("shadow"))
    jobs._shadow_phase(tmp_path)
    assert shadowed == ["shadow"]


def test_sealed_at_and_previous_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.operate as operate

    first = SimpleNamespace(execution_session=DAY, sealed_at=datetime(2026, 10, 8, 8, 30, tzinfo=KRX_TZ))
    second = SimpleNamespace(execution_session=DAY, sealed_at=datetime(2026, 10, 8, 8, 45, tzinfo=KRX_TZ))
    other = SimpleNamespace(
        execution_session=date(2026, 10, 7), sealed_at=datetime(2026, 10, 7, 8, 30, tzinfo=KRX_TZ)
    )
    monkeypatch.setattr(operate, "read_chain", lambda path: (first, second, other))
    assert jobs._sealed_at_for(Path("/x"), DAY) == second.sealed_at
    monkeypatch.setattr(operate, "read_chain", lambda path: ())
    assert jobs._sealed_at_for(Path("/x"), DAY) is None

    from datetime import datetime as _dt

    sessions = (
        _dt(2026, 10, 7, 9, 0, tzinfo=KRX_TZ),
        _dt(2026, 10, 8, 9, 0, tzinfo=KRX_TZ),
    )
    monkeypatch.setattr(
        "src.core.krx_calendar.xkrx_calendar_through", lambda day: SimpleNamespace(sessions=sessions)
    )
    assert jobs._previous_session(DAY) == date(2026, 10, 7)
    with pytest.raises(PITDataError):
        jobs._previous_session(date(2026, 10, 9))
    lonely = SimpleNamespace(sessions=(_dt(2026, 10, 8, 9, 0, tzinfo=KRX_TZ),))
    monkeypatch.setattr("src.core.krx_calendar.xkrx_calendar_through", lambda day: lonely)
    with pytest.raises(PITDataError):
        jobs._previous_session(DAY)


def test_audit_and_weekly_report_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _alert_sender(monkeypatch, None)
    monkeypatch.setattr(jobs, "run_daily_audit", lambda root, **kwargs: {"severity": "OK", "open_issues": []})
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 0
    monkeypatch.setattr(
        jobs, "run_daily_audit", lambda root, **kwargs: {"severity": "WARNING", "open_issues": [{"key": "x"}]}
    )
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 1
    assert weekly_report(tmp_path, today=DAY, now=_morning_now()) == 0
    target = tmp_path / "ops" / "reports" / "weekly-2026-10-08.json"
    assert target.is_file()
    assert json.loads(capsys.readouterr().out)["week_ending"] == "2026-10-08"


def test_main_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(jobs, "collect_eve", lambda root, **kwargs: seen.append("collect-eve") or 0)
    assert main(["--data-root", str(tmp_path), "--date", "2026-10-08", "collect-eve"]) == 0
    clock = _Clock(_morning_now())
    monkeypatch.setattr("src.ops.jobs.time", SimpleNamespace(sleep=clock.sleep, monotonic=lambda: 0.0))
    monkeypatch.setattr(
        jobs, "morning", lambda root, **kwargs: seen.append("morning") or 3
    )
    assert main(["--data-root", str(tmp_path), "morning"]) == 3
    monkeypatch.setattr(jobs, "backup", lambda root, **kwargs: seen.append("backup") or 0)
    assert main(["--data-root", str(tmp_path), "backup", "--no-lock"]) == 0
    monkeypatch.setattr(jobs, "audit", lambda root, **kwargs: seen.append("audit") or 0)
    assert main(["--data-root", str(tmp_path), "audit"]) == 0
    monkeypatch.setattr(jobs, "weekly_report", lambda root, **kwargs: seen.append("weekly-report") or 0)
    assert main(["--data-root", str(tmp_path), "weekly-report"]) == 0
    assert seen == ["collect-eve", "morning", "backup", "audit", "weekly-report"]
    assert jobs._as_int(3) == 3
    assert jobs._as_int(True) == 0
    assert jobs._as_int("x") == 0
    assert jobs._default_data_root().name == "data"


def test_morning_poll_failure_and_missing_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, _morning_now())
    clock = _Clock(_morning_now())

    def fail(*args: object, **kwargs: object) -> bool:
        raise PITDataError("token=do-not-persist")

    monkeypatch.setattr(jobs, "_poll_krx_once", fail)
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY)["morning"] == "NO_DECISION"
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    monkeypatch.setattr(jobs, "_sealed_at_for", lambda *a: None)
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert "shadow" not in calls
    assert "do-not-persist" not in (tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl").read_text()


def test_session_presence_uses_real_keyed_receipt(tmp_path: Path) -> None:
    import hashlib

    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import BlobEntry, EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
    from src.data.runtime import resolve_data_runtime

    bronze = resolve_data_runtime(data_root=tmp_path).workspace.bronze_root
    bronze.mkdir(parents=True)
    payload = bronze / "page.json"
    payload.write_bytes(b'{"records": [1]}')
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    catalog = ReceiptCatalog(bronze / "catalog")
    receipt = ReceiptIndexEntry("krx_daily_market", DAY.isoformat(), DAY, None,
                                EvidenceStatus.SUCCESS, digest, _morning_now(), payload)
    blob = BlobEntry(digest, EvidenceKind.DAILY_MARKET, "krx_daily_market", True, None, _morning_now(), payload)
    catalog.publish([receipt], blobs=[blob])
    assert list(catalog.ranges(source="krx_daily_market")) == []
    assert jobs._data_session_present(tmp_path, DAY)
    assert not jobs._data_session_present(tmp_path, date(2026, 10, 7))


def test_initial_seal_waits_for_realized_shadow_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import cli as research_cli

    monkeypatch.setattr("src.research.operate.read_chain", lambda _: (
        SimpleNamespace(data_session=date(2026, 10, 7), execution_session=DAY),
    ))
    called: list[str] = []
    monkeypatch.setattr(research_cli, "_run_shadow", lambda _: called.append("shadow"))
    jobs._shadow_phase(tmp_path)
    assert called == []


def test_shadow_failure_is_degraded_after_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _morning_stubs(monkeypatch, [], _morning_now())
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)

    def fail(_: Path) -> None:
        raise PITDataError("shadow changed")

    monkeypatch.setattr(jobs, "_shadow_phase", fail)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY)["morning"] == "DEGRADED"


def test_backup_failure_retains_last_success_timestamp(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    stamp = "2026-10-07T00:00:00+00:00"
    status.write_text(json.dumps({"finished_at": stamp, "rc": 0}))
    jobs._ensure_backup_failure_status(status, _morning_now())
    failed = json.loads(status.read_text())
    assert failed["rc"] == 1
    assert failed["last_ok_at"] == stamp
    before = status.read_text()
    jobs._ensure_backup_failure_status(status, _morning_now())
    assert status.read_text() == before


class _AlertRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def send(self, subject: str, body: str) -> None:
        self.calls.append((subject, body))


def _alert_sender(monkeypatch: pytest.MonkeyPatch, sender: object | None) -> None:
    monkeypatch.setattr(jobs, "_alert_sender", lambda: (sender, frozenset()))


def _write_backup_status(root: Path, *, stale: bool = False) -> None:
    from zoneinfo import ZoneInfo as _Zone

    target = root / "state" / "lake_backup_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    age = timedelta(hours=30.0) if stale else timedelta(hours=1.0)
    finished = (_morning_now() - age).astimezone(_Zone("UTC"))
    target.write_text(json.dumps({"finished_at": finished.isoformat(), "rc": 0,
                                  "last_ok_at": finished.isoformat(), "sources": {}}))


def _healthy_audit_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.audit as audit_mod
    from src.ops.run_outcome import record_run_outcome

    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr(
        "src.research.operate.read_chain",
        lambda path: (SimpleNamespace(data_session=DAY - timedelta(days=1), execution_session=DAY,
                                      sealed_at=datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ)),),
    )
    for job_name in ("collect_eve", "collect_dbsec", "morning", "auction_open", "backup"):
        record_run_outcome(tmp_path / "ops" / "events", job_name, "OK", run_date=DAY, now=_morning_now())
    _write_backup_status(tmp_path)


def test_alert_command_exit_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert jobs.alert(tmp_path, unit="kse-backup.service", result="exit-code", exit_status="1",
                       journal_tail="boom", now=_morning_now()) == 0
    assert sender.calls[0][0] == "[kse] 유닛 실행 실패: kse-backup.service"
    assert jobs.alert(tmp_path, unit="kse-backup.service", result="exit-code", exit_status="1",
                       journal_tail="boom", now=_morning_now()) == 0
    assert len(sender.calls) == 1

    _alert_sender(monkeypatch, None)
    assert jobs.alert(tmp_path / "no-creds", unit="kse-backup.service", result="exit-code",
                       exit_status="1", journal_tail="boom", now=_morning_now()) == 1
    assert len(list((tmp_path / "no-creds" / "ops" / "alerts" / "outbox").glob("*.json"))) == 1


def test_alert_recovery_drains_without_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.alerts import count_undelivered

    _alert_sender(monkeypatch, None)
    kwargs = {"unit": "kse-backup.service", "result": "failed", "exit_status": "1", "journal_tail": "tail"}
    assert jobs.alert(tmp_path, now=_morning_now(), **kwargs) == 1
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert jobs.alert(tmp_path, now=_morning_now() + timedelta(minutes=1), **kwargs) == 0
    assert len(sender.calls) == 1
    assert count_undelivered(tmp_path / "ops" / "alerts") == 0


def test_alert_redacts_the_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.alerts as alerts_mod

    sender = _AlertRecorder()
    secret = "hunter22-secret-value"
    monkeypatch.setenv("ALERT_GMAIL_APP_PASSWORD", secret)
    monkeypatch.setattr(alerts_mod, "sender_from_env", lambda: sender)
    assert jobs.alert(tmp_path, unit="kse-morning.service", result="failed", exit_status="1",
                       journal_tail="x" * 290 + f" leak {secret}", now=_morning_now()) == 0
    assert secret not in sender.calls[0][1]
    assert "hunter22" not in sender.calls[0][1]
    assert "***" in sender.calls[0][1]


def test_alert_audit_unit_carries_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.audit import AUDIT_HEARTBEAT_RELPATH

    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    target = tmp_path / AUDIT_HEARTBEAT_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"open_issues": [{"key": "backup", "text": "backup is older than 26 h"}]}))
    assert jobs.alert(tmp_path, unit="kse-daily-audit.service", result="failed", exit_status="1",
                       journal_tail="tail", now=_morning_now()) == 0
    assert "- backup: backup is older than 26 h" in sender.calls[0][1]
    sender.calls.clear()
    (tmp_path / "ops" / "alerts" / "sent.jsonl").unlink(missing_ok=True)
    def _unexpected_read(root: Path) -> list[dict[str, str]]:
        pytest.fail("non-audit alert must not read the heartbeat")

    monkeypatch.setattr(jobs, "_heartbeat_open_issues", _unexpected_read)
    assert jobs.alert(tmp_path, unit="kse-morning.service", result="failed", exit_status="1",
                       journal_tail="tail", now=_morning_now()) == 0
    assert "backup is older than 26 h" not in sender.calls[0][1]
    assert not (tmp_path / "ops" / "events").exists()


def test_audit_warning_delivered_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.audit as audit_mod
    from src.ops.run_outcome import record_run_outcome

    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain",
                        lambda path: (SimpleNamespace(data_session=DAY - timedelta(days=1),
                                                      execution_session=DAY,
                                                      sealed_at=datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ)),))
    for job_name in ("collect_eve", "collect_dbsec", "morning", "auction_open"):
        record_run_outcome(tmp_path / "ops" / "events", job_name, "OK", run_date=DAY, now=_morning_now())
    _write_backup_status(tmp_path, stale=True)
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 0
    assert len(sender.calls) == 1
    assert "backup" in sender.calls[0][1]


def test_audit_warning_undeliverable_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.audit as audit_mod
    from src.ops.run_outcome import record_run_outcome

    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain",
                        lambda path: (SimpleNamespace(data_session=DAY - timedelta(days=1),
                                                      execution_session=DAY,
                                                      sealed_at=datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ)),))
    for job_name in ("collect_eve", "collect_dbsec", "morning", "auction_open"):
        record_run_outcome(tmp_path / "ops" / "events", job_name, "OK", run_date=DAY, now=_morning_now())
    _write_backup_status(tmp_path, stale=True)
    _alert_sender(monkeypatch, None)
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 1
    assert len(list((tmp_path / "ops" / "alerts" / "outbox").glob("*.json"))) == 1
    assert json.loads((tmp_path / "ops" / "heartbeat" / "daily_audit.json").read_text())["undelivered_alerts"] == 0
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 1
    assert json.loads((tmp_path / "ops" / "heartbeat" / "daily_audit.json").read_text())["undelivered_alerts"] == 1


def test_healthy_audit_stays_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _healthy_audit_state(tmp_path, monkeypatch)
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 0
    assert sender.calls == []
    assert not (tmp_path / "ops" / "alerts" / "outbox").exists() or \
        len(list((tmp_path / "ops" / "alerts" / "outbox").glob("*.json"))) == 0


def test_repeat_audit_same_day_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.audit as audit_mod
    from src.ops.run_outcome import record_run_outcome

    monkeypatch.setattr(audit_mod, "trading_day", lambda day: True)
    monkeypatch.setattr("src.research.operate.read_chain",
                        lambda path: (SimpleNamespace(data_session=DAY - timedelta(days=1),
                                                      execution_session=DAY,
                                                      sealed_at=datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ)),))
    for job_name in ("collect_eve", "collect_dbsec", "morning", "auction_open"):
        record_run_outcome(tmp_path / "ops" / "events", job_name, "OK", run_date=DAY, now=_morning_now())
    _write_backup_status(tmp_path, stale=True)
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 0
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 0
    assert len(sender.calls) == 1


def test_holiday_backup_issue_alerts_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.audit as audit_mod

    monkeypatch.setattr(audit_mod, "trading_day", lambda day: False)
    _write_backup_status(tmp_path, stale=True)
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert audit(tmp_path, today=date(2026, 10, 10), now=_morning_now()) == 0
    assert len(sender.calls) == 1
    assert "backup" in sender.calls[0][1]


def test_audit_recovery_counts_after_drain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.audit import AUDIT_HEARTBEAT_RELPATH

    _healthy_audit_state(tmp_path, monkeypatch)
    _write_backup_status(tmp_path, stale=True)
    _alert_sender(monkeypatch, None)
    assert audit(tmp_path, today=DAY, now=_morning_now()) == 1
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert audit(tmp_path, today=DAY, now=_morning_now() + timedelta(minutes=1)) == 0
    assert len(sender.calls) == 1
    assert "backup is older than 26 h" in sender.calls[0][1]
    assert json.loads((tmp_path / AUDIT_HEARTBEAT_RELPATH).read_text())["undelivered_alerts"] == 0


def test_alert_cli_reads_stdin_with_byte_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    seen: dict[str, object] = {}

    def _fake_alert(root: Path, **kwargs: object) -> int:
        seen.update(kwargs)
        seen["root"] = root
        return 0

    monkeypatch.setattr(jobs, "alert", _fake_alert)
    stream = io.BytesIO(b"x" * 65535 + b"\xff" + b"unread")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stream))
    assert main(["--data-root", str(tmp_path), "alert", "--unit", "kse-backup.service"]) == 0
    assert seen["unit"] == "kse-backup.service"
    assert seen["result"] == "unknown"
    assert seen["exit_status"] == "unknown"
    assert seen["journal_tail"] == "x" * 65535 + "\ufffd"
    assert stream.tell() == 65536
    monkeypatch.setattr(sys, "stdin", io.StringIO("한" * 65536))
    assert main(["--data-root", str(tmp_path), "alert", "--unit", "kse-backup.service"]) == 0
    assert seen["journal_tail"] == "한" * 21845 + "\ufffd"

    class _BrokenBuffer:
        def read(self, size: int) -> bytes:
            raise OSError("stdin gone")

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=_BrokenBuffer()))
    assert main(["--data-root", str(tmp_path), "alert", "--unit", "kse-backup.service"]) == 0
    assert seen["journal_tail"] == ""


def test_alert_best_effort_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ops.alerts as alerts_mod
    from src.ops.audit import AUDIT_HEARTBEAT_RELPATH

    real_sender = jobs._alert_sender
    sender = _AlertRecorder()
    _alert_sender(monkeypatch, sender)
    assert jobs.alert(tmp_path, unit="kse-daily-audit.service", result="failed", exit_status="1",
                       journal_tail="tail", now=_morning_now()) == 0
    assert "open issues" not in sender.calls[0][1]

    target = tmp_path / AUDIT_HEARTBEAT_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("[1, 2]", encoding="utf-8")
    sender.calls.clear()
    (tmp_path / "ops" / "alerts" / "sent.jsonl").unlink(missing_ok=True)
    assert jobs.alert(tmp_path, unit="kse-daily-audit.service", result="failed", exit_status="1",
                       journal_tail="tail", now=_morning_now()) == 0
    target.write_text(json.dumps({"open_issues": {}}), encoding="utf-8")
    sender.calls.clear()
    (tmp_path / "ops" / "alerts" / "sent.jsonl").unlink(missing_ok=True)
    assert jobs.alert(tmp_path, unit="kse-daily-audit.service", result="failed", exit_status="1",
                       journal_tail="tail", now=_morning_now()) == 0

    def _boom_secrets(environ: object) -> frozenset[str]:
        raise RuntimeError("env gone")

    monkeypatch.setattr(jobs, "_alert_sender", real_sender)
    monkeypatch.setattr(alerts_mod, "sender_from_env", lambda: sender)
    monkeypatch.setattr(alerts_mod, "collect_secret_values", _boom_secrets)
    sender.calls.clear()
    with pytest.raises(RuntimeError, match="env gone"):
        jobs.alert(tmp_path / "delegated", unit="kse-morning.service", result="failed",
                   exit_status="1", journal_tail="sensitive journal", now=_morning_now())
    assert sender.calls == []
    assert not (tmp_path / "delegated" / "ops" / "alerts").exists()

    def _boom_drain(**kwargs: object) -> int:
        raise OSError("drain gone")

    monkeypatch.setattr(alerts_mod, "sender_from_env", lambda *a, **k: sender)
    monkeypatch.setattr(alerts_mod, "collect_secret_values", lambda environ: frozenset())
    monkeypatch.setattr(alerts_mod, "drain_outbox", _boom_drain)
    sender.calls.clear()
    assert jobs.alert(tmp_path / "drain-fail", unit="kse-morning.service", result="failed",
                       exit_status="1", journal_tail="tail", now=_morning_now()) == 0
    assert len(sender.calls) == 1


def _collect_report(blocking: str | None, degraded: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(blocking_job=blocking, degraded_jobs=degraded, planned=["a"], built=["a"])


def test_collect_eve_degraded_exit_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report(None, ("kind_notice_search",)))
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_eve": "DEGRADED"}
    event = json.loads((tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl").read_text(encoding="utf-8"))
    assert event["reason"] == "optional:kind_notice_search"
    assert event["metrics"]["degraded_jobs"] == ["kind_notice_search"]
    assert event["metrics"]["planned"] == event["metrics"]["built"] == 1
    assert isinstance(event["metrics"]["peak_rss_mb"], float)


@pytest.mark.parametrize("degraded", [(), ("kind_notice_search",)])
def test_collect_eve_required_block_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degraded: tuple[str, ...],
) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report("kis_investor_flow", degraded))
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_eve": "DEGRADED"}
    event = json.loads((tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl").read_text(encoding="utf-8"))
    assert event["reason"] == "blocked on kis_investor_flow"
    assert event["metrics"] == {"blocking_job": "kis_investor_flow", "degraded_jobs": list(degraded)}


def test_collect_eve_clean_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report(None, ()))
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_eve": "OK"}


@pytest.mark.parametrize("degraded", [(), ("kind_notice_search",)])
@pytest.mark.parametrize("fails", [False, True])
def test_collect_eve_warm_cache_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degraded: tuple[str, ...], fails: bool,
) -> None:
    from src.data import pipeline_graph as graph
    from src.data.runtime import resolve_data_runtime

    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report(None, degraded))
    contexts: list[graph.RefreshContext] = []

    def warm(ctx: graph.RefreshContext) -> int:
        contexts.append(ctx)
        if fails:
            raise PITDataError("private payload details")
        return 3

    monkeypatch.setattr(graph, "warm_daily_flag_cache", warm)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    assert len(contexts) == 1
    assert contexts[0].runtime == resolve_data_runtime(data_root=tmp_path)
    assert contexts[0].decision_time == _morning_now()
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["outcome"] == ("DEGRADED" if fails or degraded else "OK")
    assert event["reason"] == (
        "optional:kind_notice_search" if degraded else "flag_cache_warm_failed" if fails else ""
    )
    assert event["metrics"].get("flag_cache_error") == ("PITDataError" if fails else None)
    assert event["metrics"].get("flag_cache_sessions") == (None if fails else 3)
    assert "private payload details" not in json.dumps(event)


@pytest.mark.parametrize("refused", [False, True])
def test_collect_eve_guard_or_block_skips_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refused: bool,
) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    if refused:
        monkeypatch.setattr(jobs, "_guard_headroom", lambda root: GuardRefused("low memory"))
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report("dart_facts", ()))

    def unexpected(ctx: object) -> int:
        pytest.fail("warm must not run after a refused guard or blocked collection")

    monkeypatch.setattr("src.data.pipeline_graph.warm_daily_flag_cache", unexpected)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 1


@pytest.mark.parametrize("mode", ["sealed", "absent", "pit", "error", "shadow", "guard", "late"])
def test_morning_records_peak_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    seal = datetime(2026, 10, 8, 9 if mode == "late" else 8, 31, tzinfo=KRX_TZ)
    _morning_stubs(monkeypatch, [], seal)
    monkeypatch.setattr(jobs, "_wait_for_krx", lambda *a, **k: mode != "absent")
    monkeypatch.setattr(jobs, "_peak_rss_mb", lambda: 123.5)
    if mode == "guard":
        monkeypatch.setattr(jobs, "_guard_headroom", lambda root: GuardRefused("low memory"))
    if mode in ("pit", "error", "shadow"):
        def fail(*args: object, **kwargs: object) -> None:
            raise (RuntimeError if mode == "error" else PITDataError)("phase failed")

        monkeypatch.setattr(jobs, "_shadow_phase" if mode == "shadow" else "_build_all", fail)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == (0 if mode == "sealed" else 1)
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["metrics"]["peak_rss_mb"] == 123.5


def _report_with_peaks(peaks: object) -> SimpleNamespace:
    return SimpleNamespace(blocking_job=None, degraded_jobs=(), planned=["a"], built=["a"],
                           node_peak_rss_mb=peaks)


def test_collect_eve_records_node_peaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _report_with_peaks({"x": 12.34, "y": 5.678}))
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["metrics"]["node_peak_rss_mb"] == {"x": 12.3, "y": 5.7}


def test_morning_records_node_peaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _morning_stubs(monkeypatch, [], datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    monkeypatch.setattr(jobs, "_build_all", lambda root, **k: _report_with_peaks({"x": 9.876}))
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 0
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["metrics"]["node_peak_rss_mb"] == {"x": 9.9}


def test_report_without_peaks_records_empty_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(
        jobs, "_run_refresh", lambda *a, **k: SimpleNamespace(blocking_job=None, degraded_jobs=(), planned=[], built=[]),
    )
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["metrics"]["node_peak_rss_mb"] == {}
    assert event["outcome"] == "OK"


def _prune_summary(**fields: object) -> dict[str, object]:
    return {"type": "summary", "candidates": 0, "unreadable": 0, "applied": True, "deleted": 0, "failed": 0, **fields}


def test_collect_eve_records_prune_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import dataset_maintenance

    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report(None, ()))
    applied: list[bool] = []

    def prune(runtime: object, registry: object, *, apply: bool, emit: object) -> dict[str, object]:
        applied.append(apply)
        return _prune_summary(deleted=4, failed=1, unreadable=2)

    monkeypatch.setattr(dataset_maintenance, "prune_datasets", prune)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    assert applied == [True]
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["outcome"] == "OK"
    assert event["metrics"]["pruned"] == 4
    assert event["metrics"]["prune_failed"] == 1
    assert event["metrics"]["prune_unreadable"] == 2


def test_collect_eve_survives_prune_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import dataset_maintenance

    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report(None, ()))

    def prune(*args: object, **kwargs: object) -> dict[str, object]:
        raise PITDataError("private manifest path")

    monkeypatch.setattr(dataset_maintenance, "prune_datasets", prune)
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 0
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["outcome"] == "OK"
    assert event["metrics"]["prune_error"] == "PITDataError"
    assert "private manifest path" not in json.dumps(event)


def test_collect_eve_prune_runs_only_after_unblocked_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    monkeypatch.setattr(jobs, "_run_refresh", lambda *a, **k: _collect_report("dart_facts", ()))
    monkeypatch.setattr(
        jobs, "_prune_unreferenced_datasets", lambda root: pytest.fail("prune must not run after a blocked collection"),
    )
    assert collect_eve(tmp_path, today=DAY, now=_morning_now()) == 1


def test_node_peak_metrics_ignores_malformed_reports() -> None:
    assert jobs._node_peak_metrics(SimpleNamespace(node_peak_rss_mb=["not-a-mapping"])) == {}
    assert jobs._node_peak_metrics(SimpleNamespace(node_peak_rss_mb={"x": "bad"})) == {}


def _dbsec_ok_report(steps: tuple) -> SimpleNamespace:
    return SimpleNamespace(
        steps=tuple(
            SimpleNamespace(job=name, status="complete", done=1, pending_left=0, requests_used=2) for name in steps
        ),
    )


def test_collect_dbsec_skips_non_trading_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, False)
    calls: list[str] = []
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs", lambda *a, **k: calls.append("run"),
    )

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    assert calls == []
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_dbsec": "SKIPPED"}


def test_collect_dbsec_runs_only_dbsec_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.pipeline_graph import _default_collection_steps
    from src.ops.jobs import DBSEC_COLLECT_STEPS, collect_dbsec

    _trading(monkeypatch, True)
    seen: list[str] = []

    def _fake(ctx: object, *, dry_run: bool, emit: object, steps: tuple) -> SimpleNamespace:
        seen.extend(name for name, _ in steps)
        return _dbsec_ok_report(tuple(name for name, _ in steps))

    monkeypatch.setattr("src.data.pipeline_graph.run_collection_jobs", _fake)

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    expected = [name for name, _ in _default_collection_steps() if name in DBSEC_COLLECT_STEPS]
    assert seen == expected == ["dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"]
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_dbsec": "OK"}


def test_collect_dbsec_degrades_on_failed_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)

    def _failed(ctx: object, *, dry_run: bool, emit: object, steps: tuple) -> SimpleNamespace:
        return SimpleNamespace(
            steps=(
                SimpleNamespace(job="dbsec_ticker_master", status="complete", done=1, pending_left=0, requests_used=1),
                SimpleNamespace(job="dbsec_daily_chart", status="complete", done=2, pending_left=0, requests_used=2),
                SimpleNamespace(job="dbsec_closing_bar", status="failed", done=0, pending_left=3, requests_used=0),
            ),
        )

    monkeypatch.setattr("src.data.pipeline_graph.run_collection_jobs", _failed)

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_dbsec": "DEGRADED"}
    event = _json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert "dbsec_closing_bar" in event["reason"]


def test_collect_dbsec_exception_is_degraded_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)

    def _boom(ctx: object, *, dry_run: bool, emit: object, steps: tuple) -> SimpleNamespace:
        raise PITDataError("catalog unreadable")

    monkeypatch.setattr("src.data.pipeline_graph.run_collection_jobs", _boom)

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_dbsec": "DEGRADED"}


def test_evening_and_krx_step_sets_exclude_dbsec() -> None:
    assert not (set(EVENING_COLLECT_STEPS) & {"dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"})
    assert not (set(KRX_COLLECT_STEPS) & {"dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar"})


def test_collect_dbsec_cli_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ops.jobs import main

    _trading(monkeypatch, True)
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs",
        lambda *a, **k: _dbsec_ok_report(("dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar")),
    )

    assert main(["--data-root", str(tmp_path), "--date", "2026-10-08", "collect-dbsec"]) == 0
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"collect_dbsec": "OK"}


_S1 = date(2026, 10, 6)
_S2 = date(2026, 10, 7)
_RECONCILE_TICKERS = ("005930", "000660")


def _reconcile_bronze(tmp_path: Path) -> Path:
    bronze = tmp_path / "reconcile-bronze"
    (bronze / "catalog").mkdir(parents=True, exist_ok=True)
    return bronze


def _stage_dbsec_session(bronze: Path, session: date, tickers: tuple[str, ...]) -> dict[tuple[str, str], SimpleNamespace]:
    import json as _json

    from src.data.receipt_catalog import EvidenceStatus

    entries: dict[tuple[str, str], SimpleNamespace] = {}
    master = {"rows": [
        {"Iscd": ticker, "StndIscd": f"KR7{ticker}003", "KorIsnm": ticker,
         "MrktClsCode": "STK", "MrktClsName": "거래소"}
        for ticker in tickers
    ]}
    master_path = bronze / f"master-{session.isoformat()}.json"
    master_path.write_text(_json.dumps(master), encoding="utf-8")
    entries[("dbsec_ticker_master", session.isoformat())] = SimpleNamespace(
        status=EvidenceStatus.SUCCESS, payload_path=master_path,
    )
    for ticker in tickers:
        stamp = session.strftime("%Y%m%d")
        daily = {"rows": [{"Date": stamp, "Oprc": "100", "Hprc": "110", "Lprc": "90",
                           "Prpr": "105", "AcmlVol": "1000"}]}
        bar = {"rows": [{"Date": stamp, "Hour": "153000", "Oprc": "100", "Hprc": "110",
                         "Lprc": "90", "Prpr": "105", "CntgVol": "10"}]}
        daily_path = bronze / f"daily-{session.isoformat()}-{ticker}.json"
        bar_path = bronze / f"bar-{session.isoformat()}-{ticker}.json"
        daily_path.write_text(_json.dumps(daily), encoding="utf-8")
        bar_path.write_text(_json.dumps(bar), encoding="utf-8")
        entries[("dbsec_daily_chart", f"{session.isoformat()}:{ticker}")] = SimpleNamespace(
            status=EvidenceStatus.SUCCESS, payload_path=daily_path,
        )
        entries[("dbsec_closing_bar", f"{session.isoformat()}:{ticker}")] = SimpleNamespace(
            status=EvidenceStatus.SUCCESS, payload_path=bar_path,
        )
    return entries


def _stub_reconcile_sources(
    monkeypatch: pytest.MonkeyPatch, entries: dict[tuple[str, str], SimpleNamespace], silver: dict[str, object],
) -> None:
    import src.data.jobs.dbsec as dbsec_jobs
    from src.data.dataset_registry import DatasetRegistry
    from src.data.receipt_catalog import ReceiptCatalog

    ordered = sorted({key[1].split(":")[0] for key in entries})
    monkeypatch.setattr(
        dbsec_jobs, "dbsec_final_sessions",
        lambda **kwargs: tuple(date.fromisoformat(day) for day in ordered),
    )
    monkeypatch.setattr(DatasetRegistry, "require", lambda self, kind: "daily_market_test")

    def _latest(self: object, *, source: str, natural_keys: set[str]) -> dict[str, SimpleNamespace]:
        return {key: entries[(source, key)] for key in natural_keys if (source, key) in entries}

    monkeypatch.setattr(ReceiptCatalog, "latest", _latest)
    monkeypatch.setattr(jobs, "_silver_session_frame", lambda root, dataset_id, session: silver.get(session.isoformat()))


def _reconcile_krx_frame() -> object:
    import polars as pl

    return pl.DataFrame(
        [
            {"ticker": ticker, "market": "KOSPI", "open": 100, "high": 110, "low": 90,
             "close": 105, "volume": 1000, "price_state": "tradable"}
            for ticker in _RECONCILE_TICKERS
        ],
        schema={
            "ticker": pl.String, "market": pl.String, "open": pl.Int64, "high": pl.Int64,
            "low": pl.Int64, "close": pl.Int64, "volume": pl.Int64, "price_state": pl.String,
        },
    )


def test_silver_session_frame_reads_partition_layouts(tmp_path: Path) -> None:
    import polars as pl

    frame = pl.DataFrame(
        [{"ticker": "005930", "close": 105}],
        schema={"ticker": pl.String, "close": pl.Int64},
    )
    direct = tmp_path / "silver" / "daily_market_test" / "session=2026-10-06"
    direct.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(direct / "part.parquet")

    assert jobs._silver_session_frame(tmp_path / "silver", "daily_market_test", _S1).equals(frame)
    assert jobs._silver_session_frame(tmp_path / "silver", "daily_market_test", _S2) is None
    assert jobs._silver_session_frame(tmp_path / "missing", "daily_market_test", _S1) is None

    corrupt = tmp_path / "silver" / "daily_market_test" / "session=2026-10-07"
    corrupt.mkdir(parents=True, exist_ok=True)
    (corrupt / "part.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(PITDataError):
        jobs._silver_session_frame(tmp_path / "silver", "daily_market_test", _S2)


def _collect_dbsec_event(tmp_path: Path) -> dict[str, object]:
    target = tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl"
    events = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    return next(event for event in events if event["job"] == "collect_dbsec")


def _reconcile_root(tmp_path: Path) -> Path:
    from src.data.runtime import resolve_data_runtime

    return resolve_data_runtime(data_root=tmp_path).workspace.state_root / "dbsec_reconcile"


def test_collect_dbsec_reconciles_sessions_with_krx_silver_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs",
        lambda *a, **k: _dbsec_ok_report(("dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar")),
    )
    bronze = _reconcile_bronze(tmp_path)
    entries = {
        **_stage_dbsec_session(bronze, _S1, _RECONCILE_TICKERS),
        **_stage_dbsec_session(bronze, _S2, _RECONCILE_TICKERS),
    }
    _stub_reconcile_sources(monkeypatch, entries, {_S1.isoformat(): _reconcile_krx_frame()})

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    root = _reconcile_root(tmp_path)
    assert sorted(path.name for path in root.glob("*.json")) == [f"{_S1.isoformat()}.json"]
    event = _collect_dbsec_event(tmp_path)
    assert event["outcome"] == "OK"
    (row,) = event["metrics"]["reconciled"]
    assert (row["session"], row["verdict"]) == (_S1.isoformat(), "pass")
    assert event["metrics"]["fallback_ready"] is False


def test_collect_dbsec_defers_session_with_unanswered_bar_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs",
        lambda *a, **k: _dbsec_ok_report(("dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar")),
    )
    bronze = _reconcile_bronze(tmp_path)
    entries = _stage_dbsec_session(bronze, _S1, _RECONCILE_TICKERS)
    del entries[("dbsec_closing_bar", f"{_S1.isoformat()}:{_RECONCILE_TICKERS[0]}")]
    _stub_reconcile_sources(monkeypatch, entries, {_S1.isoformat(): _reconcile_krx_frame()})

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    root = _reconcile_root(tmp_path)
    assert not root.exists() or not any(root.glob("*.json"))
    assert _collect_dbsec_event(tmp_path)["metrics"]["reconciled"] == []


def test_collect_dbsec_does_not_rewrite_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data.dbsec_reconcile import write_reconciliation
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs",
        lambda *a, **k: _dbsec_ok_report(("dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar")),
    )
    bronze = _reconcile_bronze(tmp_path)
    entries = _stage_dbsec_session(bronze, _S1, _RECONCILE_TICKERS)
    _stub_reconcile_sources(monkeypatch, entries, {_S1.isoformat(): _reconcile_krx_frame()})
    root = _reconcile_root(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / f"{_S1.isoformat()}.json"
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")
    before = (sentinel.stat().st_mtime_ns, sentinel.read_bytes())

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    assert (sentinel.stat().st_mtime_ns, sentinel.read_bytes()) == before
    assert write_reconciliation is not None


def test_collect_dbsec_reconcile_failure_degrades_without_failing_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data.dbsec_reconcile as reconcile
    from src.ops.jobs import collect_dbsec

    _trading(monkeypatch, True)
    monkeypatch.setattr(
        "src.data.pipeline_graph.run_collection_jobs",
        lambda *a, **k: _dbsec_ok_report(("dbsec_ticker_master", "dbsec_daily_chart", "dbsec_closing_bar")),
    )
    bronze = _reconcile_bronze(tmp_path)
    entries = _stage_dbsec_session(bronze, _S1, _RECONCILE_TICKERS)
    _stub_reconcile_sources(monkeypatch, entries, {_S1.isoformat(): _reconcile_krx_frame()})

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("silver unreadable")

    monkeypatch.setattr(reconcile, "reconcile_session", _boom)

    assert collect_dbsec(tmp_path, today=DAY, now=_morning_now()) == 0

    event = _collect_dbsec_event(tmp_path)
    assert event["outcome"] == "DEGRADED"
    assert event["reason"] == "reconcile:RuntimeError"
    root = _reconcile_root(tmp_path)
    assert not root.exists() or not any(root.iterdir())
    assert entries[("dbsec_ticker_master", _S1.isoformat())].payload_path.is_file()


def test_morning_build_never_reads_dbsec_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.dbsec_reconcile as reconcile

    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    bronze = _reconcile_bronze(tmp_path)
    _stage_dbsec_session(bronze, date(2026, 10, 7), _RECONCILE_TICKERS)
    monkeypatch.setattr(jobs, "_previous_session", lambda day: date(2026, 10, 7))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: False)
    clock = _Clock(datetime(2026, 10, 8, 8, 54, tzinfo=KRX_TZ))

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("morning must not read DBSEC evidence")

    monkeypatch.setattr(reconcile, "load_dbsec_session_rows", _forbidden)

    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"morning": "NO_DECISION"}
    target = tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl"
    morning_event = next(
        json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["job"] == "morning"
    )
    assert morning_event["reason"] == "krx_absent_at_deadline"


def test_morning_records_stale_investor_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.investor_flow_freshness import FlowFreshness, StaleInvestorFlowError

    _trading(monkeypatch, True)
    _healthy_guard(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(jobs, "_previous_session", lambda day: date(2026, 10, 7))
    monkeypatch.setattr(jobs, "_build_all", lambda root, **kwargs: calls.append("build"))
    freshness = (FlowFreshness(date(2026, 10, 7), 100, 90, 10, 10, 0.10),)

    def _stale(root: Path) -> list[str]:
        raise StaleInvestorFlowError(freshness)

    monkeypatch.setattr(jobs, "_decide_phase", _stale)
    monkeypatch.setattr(jobs, "_shadow_phase", lambda root: calls.append("shadow"))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    assert "shadow" not in calls
    assert load_run_outcomes(tmp_path / "ops" / "events", DAY) == {"morning": "NO_DECISION"}
    target = tmp_path / "ops" / "events" / "2026-10" / "2026-10-08.jsonl"
    morning_event = next(
        json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["job"] == "morning"
    )
    assert morning_event["reason"] == "stale_investor_flow"
    assert morning_event["metrics"]["flow_freshness"] == [
        {
            "source_session": "2026-10-07",
            "required_cells": 100,
            "present_cells": 90,
            "late_cells": 10,
            "missing_cells": 10,
            "missing_fraction": 0.10,
        }
    ]


def _raiser(error: BaseException):  # type: ignore[no-untyped-def]
    def _raise(*args: object, **kwargs: object) -> None:
        raise error

    return _raise


_RELAY_SUMMARIES: list[dict[str, object]] = [
    {"job": "kind_notice_search", "status": "complete", "imported": 3, "unchanged": 0, "rejected": 0,
     "pending_left": 0, "fetched_at": "2026-10-07T12:00:00+00:00"},
    {"job": "kind_notice_documents", "status": "complete", "imported": 1, "unchanged": 0, "rejected": 0,
     "pending_left": 0, "fetched_at": "2026-10-07T12:00:00+00:00"},
]


def test_morning_imports_relay_after_wait_before_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert MORNING_RELAY_JOBS == ("kind_notice_search", "kind_notice_documents")
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_wait_for_krx", lambda *a, **k: calls.append("wait") or True)
    seen: dict[str, datetime] = {}
    clock = _Clock(_morning_now())

    def relay(root: Path, *, now: datetime) -> list[dict[str, object]]:
        calls.append("relay")
        seen["now"] = now
        clock.sleep(1)
        return []

    def build(root: Path, *, decision_time: datetime) -> None:
        calls.append("build")
        seen["decision_time"] = decision_time

    monkeypatch.setattr(jobs, "_import_kind_relay", relay)
    monkeypatch.setattr(jobs, "_build_all", build)
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 0
    assert calls == ["wait", "relay", "build", "decide", "shadow"]
    assert seen["now"] < seen["decision_time"]
    assert seen["decision_time"] < datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ)


def test_morning_absent_krx_skips_relay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, _morning_now())
    monkeypatch.setattr(jobs, "_wait_for_krx", lambda *a, **k: False)
    monkeypatch.setattr(jobs, "_import_kind_relay", _raiser(AssertionError("relay must not run without KRX bars")))
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 1
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["outcome"] == "NO_DECISION"
    assert event["reason"] == "krx_absent_at_deadline"
    assert "kind_relay" not in event["metrics"]


@pytest.mark.parametrize("status", ["relay_absent", "relay_rejected", "relay_partial", "failed"])
def test_relay_status_never_changes_morning_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    summaries = [{**_RELAY_SUMMARIES[0], "status": status}]
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, datetime(2026, 10, 8, 8, 31, tzinfo=KRX_TZ))
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    monkeypatch.setattr(jobs, "_import_kind_relay", lambda root, *, now: [dict(item) for item in summaries])
    clock = _Clock(_morning_now())
    assert morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep) == 0
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["outcome"] == "OK"
    assert event["metrics"]["kind_relay"] == summaries
    assert isinstance(event["metrics"]["relay_s"], float)


@pytest.mark.parametrize("mode", ["ok", "late", "pit", "error", "shadow", "stale"])
def test_relay_metrics_ride_every_post_import_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from src.data.investor_flow_freshness import FlowFreshness, StaleInvestorFlowError

    seal = datetime(2026, 10, 8, 9 if mode == "late" else 8, 31, tzinfo=KRX_TZ)
    calls: list[str] = []
    _morning_stubs(monkeypatch, calls, seal)
    monkeypatch.setattr(jobs, "_poll_krx_once", lambda *a, **k: True)
    monkeypatch.setattr(jobs, "_import_kind_relay", lambda root, *, now: [dict(item) for item in _RELAY_SUMMARIES])
    if mode == "pit":
        monkeypatch.setattr(jobs, "_build_all", _raiser(PITDataError("cash return is missing")))
    elif mode == "error":
        monkeypatch.setattr(jobs, "_build_all", _raiser(RuntimeError("scorer missing")))
    elif mode == "shadow":
        monkeypatch.setattr(jobs, "_shadow_phase", _raiser(PITDataError("shadow changed")))
    elif mode == "stale":
        freshness = (FlowFreshness(date(2026, 10, 7), 10, 9, 1, 1, 0.1),)
        monkeypatch.setattr(jobs, "_decide_phase", _raiser(StaleInvestorFlowError(freshness)))
    clock = _Clock(_morning_now())
    exit_code = morning(tmp_path, today=DAY, now=clock.now, sleep=clock.sleep)
    event = json.loads((tmp_path / "ops/events/2026-10/2026-10-08.jsonl").read_text())
    assert event["metrics"]["kind_relay"] == _RELAY_SUMMARIES
    if mode == "ok":
        assert (exit_code, event["outcome"]) == (0, "OK")
    else:
        assert exit_code == 1
    if mode == "late":
        assert event["outcome"] == "DEGRADED"
        assert event["reason"] == "late_seal"
        assert isinstance(event["metrics"]["relay_s"], float)


@pytest.mark.parametrize("transport", [None, "direct"])
def test_relay_helper_is_file_only_under_direct_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str | None,
) -> None:
    import src.data.kind_relay_import as importer
    import src.data.pipeline_graph as graph

    if transport is None:
        monkeypatch.delenv("KSE_KIND_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("KSE_KIND_TRANSPORT", transport)
    calls: list[dict[str, object]] = []
    fetched = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

    def spy(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(job=kwargs["job"], status="complete", imported=1, unchanged=0,
                               rejected=0, pending_left=0, fetched_at=fetched)

    monkeypatch.setattr(importer, "import_relay_bundle", spy)
    monkeypatch.setattr(
        "src.integrations.krx.kind.build_scoped_kind_client", _raiser(AssertionError("no KIND client"))
    )
    monkeypatch.setattr(graph, "_run_kind_collection_step", _raiser(AssertionError("no collection step")))

    now = datetime(2026, 10, 8, 0, 30, tzinfo=KRX_TZ)
    summaries = jobs._import_kind_relay(tmp_path, now=now)

    assert [call["job"] for call in calls] == ["kind_notice_search", "kind_notice_documents"]
    assert all(call["dry_run"] is False and call["now"] == now for call in calls)
    assert [item["job"] for item in summaries] == ["kind_notice_search", "kind_notice_documents"]
    assert all(item["fetched_at"] == "2026-10-07T12:00:00+00:00" for item in summaries)


def test_relay_helper_isolates_job_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.kind_relay_import as importer

    invoked: list[str] = []

    def spy(**kwargs: object) -> SimpleNamespace:
        invoked.append(str(kwargs["job"]))
        if kwargs["job"] == "kind_notice_search":
            raise OSError("inbox unreadable")
        return SimpleNamespace(job=kwargs["job"], status="complete", imported=0, unchanged=0,
                               rejected=0, pending_left=0, fetched_at=None)

    monkeypatch.setattr(importer, "import_relay_bundle", spy)
    summaries = jobs._import_kind_relay(tmp_path, now=_morning_now())
    assert invoked == ["kind_notice_search", "kind_notice_documents"]
    assert [item["status"] for item in summaries] == ["failed", "complete"]
    assert (summaries[0]["imported"], summaries[0]["rejected"], summaries[0]["fetched_at"]) == (0, 0, None)


@pytest.mark.parametrize("target", ["src.data.runtime.resolve_data_runtime", "src.config.load_provider_policy"])
def test_relay_helper_resolution_failure_yields_two_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, target: str,
) -> None:
    monkeypatch.setattr(target, _raiser(PITDataError("configuration unavailable")))
    summaries = jobs._import_kind_relay(tmp_path, now=_morning_now())
    assert [item["job"] for item in summaries] == ["kind_notice_search", "kind_notice_documents"]
    assert [item["status"] for item in summaries] == ["failed", "failed"]
    assert all(item["imported"] == 0 and item["fetched_at"] is None for item in summaries)
    assert all(item[key] == 0 for item in summaries for key in ("unchanged", "rejected", "pending_left"))
    for job in MORNING_RELAY_JOBS:
        assert f"[DATA] stage=relay_import job={job} status=failed error=PITDataError" in caplog.text
