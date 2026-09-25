from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.pit import PITDataError
from src.data.remote_dart_worker import run_worker, seconds_until_window_end
from src.integrations.dart.xbrl import DartCircuitOpenError

NOON = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)  # 12:00 KST


class _Collector:
    def __init__(self, *, healthy: bool = True) -> None:
        self.aborted = False
        self.healthy = healthy

    def health_check(self) -> None:
        if not self.healthy:
            raise RuntimeError("connection reset")


def _job(root: Path, count: int, *, budget: int = 1000, windows: list[list[str]] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    identities = [{"corp_code": f"{i:08d}", "filing_id": f"2016051500{i:04d}", "biz_year": "2016", "reprt_code": "11013"} for i in range(count)]
    policy = {"daily_budget": budget, "daily_reserve": 10, "min_interval_seconds": 0.34, "chunk": 4, "avoid_kst_windows": windows or []}
    (root / "job.json").write_text(json.dumps({"identities": identities, "policy": policy}), encoding="utf-8")


def _collect_factory(root: Path, *, blocked: int = 0, unavailable: int = 0) -> Any:
    def collect(*, dart: object, identities: tuple[dict[str, str], ...], bronze_root: Path, retrieved_at: datetime) -> SimpleNamespace:
        report = root / f"report-{identities[0]['filing_id']}.json"
        report.write_text(
            json.dumps({"filing_ids": [i["filing_id"] for i in identities], "blocked": blocked, "unavailable": unavailable}), encoding="utf-8"
        )
        return SimpleNamespace(report_path=report)

    return collect


def _run(root: Path, monkeypatch: pytest.MonkeyPatch, collect: Any, collector: _Collector | None = None, **kwargs: Any) -> Any:
    monkeypatch.setenv("KEY_X", "secret")
    fixed = collector or _Collector()
    return run_worker(root=root, key_env="KEY_X", collect=collect, build_collector=lambda *_: fixed, now=lambda: NOON, sleep=lambda _s: None, **kwargs)


def test_worker_collects_every_identity_and_marks_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 10)

    result = _run(tmp_path, monkeypatch, _collect_factory(tmp_path))

    assert (result.status, result.done_total, result.pending_left) == ("complete", 10, 0)
    assert (tmp_path / "COMPLETE").is_file()
    assert len((tmp_path / "out" / "done.txt").read_text().split()) == 10


def test_worker_resumes_from_its_done_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 6)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "done.txt").write_text("\n".join(f"2016051500{i:04d}" for i in range(4)) + "\n")
    seen: list[int] = []

    def collect(*, identities: tuple[dict[str, str], ...], **_kw: Any) -> SimpleNamespace:
        seen.append(len(identities))
        return _collect_factory(tmp_path)(dart=None, identities=identities, bronze_root=tmp_path, retrieved_at=NOON)

    result = _run(tmp_path, monkeypatch, collect)

    assert seen == [2]
    assert result.status == "complete"


def test_worker_stops_when_daily_budget_is_spent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.integrations.dart.client import dart_quota_provider
    from src.integrations.quota import ProviderQuotaStateStore

    _job(tmp_path, 10, budget=100)
    monkeypatch.setenv("KEY_X", "secret")
    ProviderQuotaStateStore(tmp_path / "state").add_attempts(provider=dart_quota_provider("secret"), endpoint="x", day="2026-09-24", count=95)

    result = _run(tmp_path, monkeypatch, _collect_factory(tmp_path), _Collector(healthy=False))

    assert result.status == "budget_exhausted"  # 헬스체크(불건강 설정)에 도달하지 않고 종료
    assert result.pending_left == 10


def test_worker_reports_unreachable_without_collecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 4)

    def never(**_kw: Any) -> None:
        raise AssertionError("must not collect while unreachable")

    result = _run(tmp_path, monkeypatch, never, _Collector(healthy=False))

    assert result.status == "provider_unreachable"
    assert not (tmp_path / "out" / "done.txt").exists()


def test_worker_stops_on_circuit_error_and_on_aborted_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 8)

    def tripped(**_kw: Any) -> None:
        raise DartCircuitOpenError("down")

    assert _run(tmp_path, monkeypatch, tripped).status == "provider_unstable"

    aborting = _Collector()
    aborting.aborted = True
    result = _run(tmp_path, monkeypatch, _collect_factory(tmp_path), aborting)
    assert result.status == "provider_unstable"
    assert result.done_total == 4


def test_worker_stops_on_quota_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 8)

    assert _run(tmp_path, monkeypatch, _collect_factory(tmp_path, blocked=1)).status == "quota_blocked"


def test_worker_refuses_missing_job_and_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PITDataError, match="job file"):
        _run(tmp_path, monkeypatch, _collect_factory(tmp_path))
    _job(tmp_path, 2)
    monkeypatch.delenv("KEY_X", raising=False)
    with pytest.raises(PITDataError, match="not set"):
        run_worker(root=tmp_path, key_env="KEY_X", collect=_collect_factory(tmp_path), build_collector=lambda *_: _Collector())


def test_second_worker_backs_off_while_the_lock_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    _job(tmp_path, 2)
    handle = (tmp_path / "worker.lock").open("w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _run(tmp_path, monkeypatch, _collect_factory(tmp_path)).status == "busy"
    finally:
        handle.close()


def test_worker_waits_out_the_shared_ip_window_before_calling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 2, windows=[["11:50", "12:20"]])
    waits: list[float] = []
    monkeypatch.setenv("KEY_X", "secret")

    run_worker(
        root=tmp_path, key_env="KEY_X", collect=_collect_factory(tmp_path), build_collector=lambda *_: _Collector(),
        now=lambda: NOON, sleep=waits.append,
    )

    assert waits == [20 * 60.0]


def test_window_math_only_waits_inside_a_window() -> None:
    windows = [["21:25", "21:50"]]
    inside = datetime(2026, 9, 24, 12, 35, tzinfo=UTC)  # 21:35 KST
    assert seconds_until_window_end(inside, windows) == 15 * 60.0
    assert seconds_until_window_end(NOON, windows) == 0.0
    assert seconds_until_window_end(datetime(2026, 9, 24, 12, 50, tzinfo=UTC), windows) == 0.0


def test_worker_with_nothing_pending_marks_complete_without_health_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _job(tmp_path, 3)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "done.txt").write_text("\n".join(f"2016051500{i:04d}" for i in range(3)) + "\n")

    result = _run(tmp_path, monkeypatch, _collect_factory(tmp_path), _Collector(healthy=False))

    assert result.status == "complete"
    assert (tmp_path / "COMPLETE").is_file()


def test_worker_stops_mid_run_when_the_ledger_runs_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.integrations.dart.client import dart_quota_provider
    from src.integrations.quota import ProviderQuotaStateStore

    _job(tmp_path, 20, budget=60)
    monkeypatch.setenv("KEY_X", "secret")
    inner = _collect_factory(tmp_path)

    def spending(*, identities: tuple[dict[str, str], ...], **kwargs: Any) -> SimpleNamespace:
        ProviderQuotaStateStore(tmp_path / "state").add_attempts(
            provider=dart_quota_provider("secret"), endpoint="x", day="2026-09-24", count=3 * len(identities)
        )
        return inner(identities=identities, **kwargs)

    result = _run(tmp_path, monkeypatch, spending)

    assert result.status == "budget_exhausted"
    assert 0 < result.done_total < 20
