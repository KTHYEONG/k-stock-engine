"""Shard invariants: deterministic partition, stable ownership, lock exclusion."""
from __future__ import annotations

import fcntl
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
_JOB = "dart_shard_probe"


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data"
    )
    runtime.workspace.initialize()
    return runtime


def _unit(key: str):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import JobUnit

    return JobUnit(source="dart_facts", natural_key=key, payload={"id": key}, max_requests=1)


def _ctx(runtime, provider):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import build_job_context

    return build_job_context(
        runtime=runtime,
        provider=provider,
        key_env="OPENDART_API_KEY_2",
        collector=None,
        now=lambda: NOW,
        sleep=lambda seconds: None,
    )


class _ShardSpec:
    """Fake job answering every unit with a keyed disclosure page."""

    name = _JOB

    def __init__(self, units):  # type: ignore[no-untyped-def]
        self._units = list(units)

    def pending(self, ctx):  # type: ignore[no-untyped-def]
        return list(self._units)

    def fetch(self, ctx, units):  # type: ignore[no-untyped-def]
        from src.core.pit import EvidenceKind
        from src.data.receipt_catalog import EvidenceStatus
        from src.data.scoped_ingestion import ScopedRawPayload

        return [
            ScopedRawPayload(
                kind=EvidenceKind.DISCLOSURES,
                source="dart_disclosures",
                natural_key=unit.natural_key,
                as_of=date(2024, 1, 2),
                fiscal_period=None,
                status=EvidenceStatus.SUCCESS,
                payload=json.dumps({"key": unit.natural_key}).encode("utf-8"),
                retrieved_at=NOW,
                source_label=f"dart_disclosures:{unit.natural_key}",
            )
            for unit in units
        ]

    def health_check(self, ctx) -> None:  # type: ignore[no-untyped-def]
        return None


def _run(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import run_job

    params = {"chunk_size": 500, "max_chunks": None, "dry_run": True, "emit": lambda payload: None}
    params.update(kwargs)
    return run_job(spec, ctx, **params)


def _hold(path: Path, kind: int):  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    fcntl.flock(handle.fileno(), kind)
    return handle


def test_shards_partition_pending_units() -> None:
    from src.data.jobs.runner import Shard

    units = [_unit(f"unit-{i}") for i in range(1000)]
    owned = [{unit.natural_key for unit in units if Shard(index=i, count=4).owns(unit)} for i in range(4)]

    assert len(set().union(*owned)) == 1000
    for first in range(4):
        for second in range(first + 1, 4):
            assert owned[first].isdisjoint(owned[second])
    for shard_keys in owned:
        assert 150 <= len(shard_keys) <= 350


def test_shard_ownership_is_process_stable() -> None:
    from src.data.jobs.runner import Shard

    unit = _unit("corp:2019:11011")
    assert [Shard(index=index, count=4).owns(unit) for index in range(4)] == [False, True, False, False]


def test_same_shard_cannot_run_twice(tmp_path: Path) -> None:
    from src.data.jobs.runner import Shard

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    jobs_dir = runtime.workspace.state_root / "jobs"
    held = _hold(jobs_dir / f"{_JOB}.shard-1-of-4.lock", fcntl.LOCK_EX)
    try:
        report = _run(_ShardSpec([_unit("a")]), ctx, shard=Shard.parse("1/4"))
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()

    assert report.status == "busy"


def test_different_shards_run_concurrently(tmp_path: Path) -> None:
    from src.data.jobs.runner import Shard

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    jobs_dir = runtime.workspace.state_root / "jobs"
    job_held = _hold(jobs_dir / f"{_JOB}.lock", fcntl.LOCK_SH)
    shard_held = _hold(jobs_dir / f"{_JOB}.shard-0-of-4.lock", fcntl.LOCK_EX)
    try:
        report = _run(_ShardSpec([_unit("a")]), ctx, shard=Shard.parse("1/4"))
    finally:
        fcntl.flock(shard_held.fileno(), fcntl.LOCK_UN)
        shard_held.close()
        fcntl.flock(job_held.fileno(), fcntl.LOCK_UN)
        job_held.close()

    assert report.status != "busy"


def test_unsharded_and_sharded_runs_exclude_each_other(tmp_path: Path) -> None:
    from src.data.jobs.runner import Shard

    runtime = _runtime(tmp_path)
    provider = _provider()
    jobs_dir = runtime.workspace.state_root / "jobs"

    held = _hold(jobs_dir / f"{_JOB}.lock", fcntl.LOCK_EX)
    try:
        report = _run(_ShardSpec([_unit("a")]), _ctx(runtime, provider), shard=Shard.parse("0/2"))
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    assert report.status == "busy"

    job_held = _hold(jobs_dir / f"{_JOB}.lock", fcntl.LOCK_SH)
    shard_held = _hold(jobs_dir / f"{_JOB}.shard-0-of-2.lock", fcntl.LOCK_EX)
    try:
        report = _run(_ShardSpec([_unit("a")]), _ctx(runtime, provider))
    finally:
        fcntl.flock(shard_held.fileno(), fcntl.LOCK_UN)
        shard_held.close()
        fcntl.flock(job_held.fileno(), fcntl.LOCK_UN)
        job_held.close()
    assert report.status == "busy"


def test_shard_parse_validation() -> None:
    from src.data.jobs.runner import Shard

    assert Shard.parse("0/1") == Shard(index=0, count=1)
    for bad in ("4/4", "-1/2", "1", "a/b", "1/2/3", "", "0/0"):
        with pytest.raises(ValueError, match="shard"):
            Shard.parse(bad)
    with pytest.raises(ValueError, match="shard"):
        Shard(index=0, count=0)
    with pytest.raises(ValueError, match="shard"):
        Shard(index=2, count=2)
    with pytest.raises(ValueError, match="shard"):
        Shard(index=True, count=2)


def test_collect_cli_forwards_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import json

    import src.data.jobs.runner as runner_mod
    from src.data.cli import main
    from src.data.jobs.runner import Shard

    seen: dict = {}

    def _fake_run_job(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return runner_mod.JobReport(status="dry_run", done=0, pending_left=0, requests_used=0)

    monkeypatch.setattr(runner_mod, "run_job", _fake_run_job)
    runtime = _runtime(tmp_path)
    exit_code = main([
        "collect-dart-facts",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(runtime.workspace.root),
        "--dry-run", "--shard", "0/2",
    ])

    assert exit_code == 0
    assert seen["shard"] == Shard(index=0, count=2)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1]["shard"] == "0/2"

    seen.clear()
    exit_code = main([
        "collect-dart-disclosures",
        "--scope-config", "config/research/kr_swing_2019_v1.toml",
        "--data-root", str(runtime.workspace.root),
        "--dry-run",
    ])

    assert exit_code == 0
    assert seen["shard"] is None
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1]["shard"] is None
