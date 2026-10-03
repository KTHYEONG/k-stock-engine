"""Runner invariants: resume, budget sizing, quota, breaker, windows, lock, ledger."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)  # 12:00 KST

# A registered keyed source, so the fake job writes through the real contract gate.
_FAKE_SOURCE = "dart_disclosures"


def _provider(*, budget=1000, reserve=10, windows=(), threshold=3, batch=500):  # type: ignore[no-untyped-def]
    from src.config.providers import (
        DartKeyPolicy, DartPolicy, KindPolicy, KisPolicy, KrxPolicy, LsPolicy, ProviderPolicy,
    )

    return ProviderPolicy(
        dart=DartPolicy(
            circuit_threshold=threshold,
            requests_per_identity=3,
            batch_identities=batch,
            host_min_interval_seconds=0.1,
            shared_ip_avoid_windows_kst=[tuple(window) for window in windows],
            keys={
                "TEST_DART_KEY": DartKeyPolicy(
                    daily_limit=20000,
                    daily_budget=budget,
                    daily_reserve=reserve,
                    min_interval_seconds=0.2,
                    max_workers=2,
                )
            },
        ),
        kis=KisPolicy(
            app_key_env="K1",
            app_secret_env="S1",
            account_no_env="A1",
            account_product_code_env="P1",
            env_env="E1",
            circuit_threshold=3,
            daily_limit=20000,
            investor_flow_rows_per_page=30,
            min_interval_seconds=1.0,
            max_attempts=3,
        ),
        krx=KrxPolicy(
            api_key_env="KRX_TEST_KEY",
            circuit_threshold=3,
            min_interval_seconds=1.5,
            daily_limit=10000,
        ),
        ls=LsPolicy(
            app_key_env="L1",
            app_secret_env="S1",
            circuit_threshold=3,
            max_sessions_per_request=700,
            min_interval_seconds=1.05,
            daily_limit=10000,
        ),
        kind=KindPolicy(
            circuit_threshold=3,
            min_interval_seconds=1.0,
            daily_limit=5000,
            search_keywords=("상장폐지",),
            document_titles=("상장폐지",),
        ),
    )


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    from src.data.runtime import load_data_runtime

    return load_data_runtime(
        scope_config=Path("config/research/kr_swing_2019_v1.toml"), data_root=tmp_path / "data"
    )


def _ctx(runtime, provider, *, now=None, sleep=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import build_job_context

    return build_job_context(
        runtime=runtime,
        provider=provider,
        key_env="TEST_DART_KEY",
        collector=None,
        now=now or (lambda: NOW),
        sleep=sleep or (lambda seconds: None),
    )


def _unit(key: str, *, max_requests: int = 3):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import JobUnit

    return JobUnit(source=_FAKE_SOURCE, natural_key=key, payload={"id": key}, max_requests=max_requests)


def _payload(key: str, retrieved_at: datetime):  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    return ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES,
        source=_FAKE_SOURCE,
        natural_key=key,
        as_of=date(2024, 1, 2),
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=json.dumps({"key": key}).encode("utf-8"),
        retrieved_at=retrieved_at,
        source_label=f"{_FAKE_SOURCE}:{key}",
    )


class _FakeSpec:
    """Catalog-resuming fake job with per-unit scripted outcomes."""

    name = "fake_job"

    def __init__(self, units, *, behavior=None, healthy=True, attempts_per_unit=0):  # type: ignore[no-untyped-def]
        self._units = list(units)
        self._behavior = dict(behavior or {})
        self._healthy = healthy
        self._attempts_per_unit = attempts_per_unit
        self.fetch_calls = 0
        self.health_checks = 0

    def pending(self, ctx):  # type: ignore[no-untyped-def]
        from src.data.receipt_catalog import EvidenceStatus

        answered = ctx.catalog.latest(source=_FAKE_SOURCE, natural_keys={unit.natural_key for unit in self._units})
        terminal = {EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED}
        return [
            unit
            for unit in self._units
            if unit.natural_key not in answered or answered[unit.natural_key].status not in terminal
        ]

    def fetch(self, ctx, units):  # type: ignore[no-untyped-def]
        from src.integrations.dart.client import ProviderRetryableError
        from src.integrations.errors import ProviderQuotaExhaustedError

        out = []
        for unit in units:
            self.fetch_calls += 1
            mode = self._behavior.get(unit.natural_key, "ok")
            if mode == "quota":
                raise ProviderQuotaExhaustedError("quota out")
            if mode == "transport":
                raise ProviderRetryableError("transport failed")
            if mode == "empty":
                return []
            for _ in range(self._attempts_per_unit):
                ctx.quota_store.record_attempt(
                    provider=ctx.runner.quota_provider, endpoint="fake", now=ctx.now(), daily_limit=ctx.runner.daily_budget
                )
            out.append(_payload(unit.natural_key, ctx.now()))
        return out

    def health_check(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self.health_checks += 1
        if not self._healthy:
            raise RuntimeError("connection reset")


def _run(spec, ctx, **kwargs):  # type: ignore[no-untyped-def]
    from src.data.jobs.runner import run_job

    emitted: list[dict] = []
    params = {"chunk_size": 100, "max_chunks": None, "dry_run": False, "emit": emitted.append}
    params.update(kwargs)
    return run_job(spec, ctx, **params), emitted  # type: ignore[arg-type]


def test_second_run_is_noop(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u1"), _unit("u2")])

    first, _ = _run(spec, ctx)
    assert (first.status, first.done, first.pending_left) == ("complete", 2, 0)
    calls_after_first = spec.fetch_calls

    second, _ = _run(spec, ctx)
    assert (second.status, second.done, second.pending_left) == ("complete", 0, 0)
    assert spec.fetch_calls == calls_after_first


def test_first_chunk_is_sized_to_headroom(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider(budget=12, reserve=2))
    spec = _FakeSpec([_unit(f"u{i}") for i in range(5)])

    report, _ = _run(spec, ctx, max_chunks=1)

    assert (report.status, report.done, report.pending_left) == ("chunk_limit", 3, 2)


def test_quota_block_discards_chunk_and_stops(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u0"), _unit("u1"), _unit("u2"), _unit("u3")], behavior={"u1": "quota"})

    report, _ = _run(spec, ctx, chunk_size=2)

    assert (report.status, report.done, report.pending_left) == ("quota_blocked", 0, 4)
    assert spec.fetch_calls == 2
    assert ctx.catalog.latest(source=_FAKE_SOURCE, natural_keys={"u0", "u1", "u2", "u3"}) == {}


def test_breaker_persists_chunk_successes_and_retries_failures(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec(
        [_unit(f"u{i}") for i in range(5)],
        behavior={"u1": "transport", "u2": "transport", "u3": "transport"},
    )

    report, _ = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("provider_unstable", 1, 4)
    assert spec.fetch_calls == 4
    answered = ctx.catalog.latest(source=_FAKE_SOURCE, natural_keys={f"u{i}" for i in range(5)})
    assert set(answered) == {"u0"}

    spec._behavior.clear()
    retry, _ = _run(spec, ctx)
    assert (retry.status, retry.done, retry.pending_left) == ("complete", 4, 0)


def test_avoid_window_wait_precedes_first_request(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    events: list[tuple[str, object]] = []
    ctx = _ctx(
        runtime,
        _provider(windows=[("12:00", "12:30")]),
        sleep=lambda seconds: events.append(("sleep", seconds)),
    )
    spec = _FakeSpec([_unit("u0")])
    original_fetch = spec.fetch

    def _recording(ctx, units):  # type: ignore[no-untyped-def]
        events.append(("fetch", units[0].natural_key))
        return original_fetch(ctx, units)

    spec.fetch = _recording  # type: ignore[method-assign]

    report, _ = _run(spec, ctx)

    assert report.status == "complete"
    assert events[0] == ("sleep", 1800.0)
    assert ("fetch", "u0") in events


def test_second_process_reports_busy_without_requests(tmp_path: Path) -> None:
    import fcntl

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u0")])
    lock_path = runtime.workspace.state_root / "jobs" / "fake_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report, _ = _run(spec, ctx)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    assert (report.status, report.done, report.pending_left, report.requests_used) == ("busy", 0, 0, 0)
    assert spec.fetch_calls == 0
    assert spec.health_checks == 0


def test_requests_used_matches_ledger_delta(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit(f"u{i}") for i in range(3)], attempts_per_unit=2)

    report, _ = _run(spec, ctx)

    assert report.status == "complete"
    assert report.requests_used == 6


def test_dry_run_plans_without_requests(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u1"), _unit("u2")])

    report, emitted = _run(spec, ctx, dry_run=True)

    assert (report.status, report.done, report.pending_left, report.requests_used) == ("dry_run", 0, 2, 0)
    assert spec.fetch_calls == 0
    assert spec.health_checks == 0
    assert [line["phase"] for line in emitted] == ["plan", "done"]


def test_empty_pending_completes_without_health_check(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([])

    report, _ = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("complete", 0, 0)
    assert spec.health_checks == 0


def test_budget_exhausted_when_headroom_covers_nothing(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider(budget=3, reserve=2))
    spec = _FakeSpec([_unit("u0")])

    report, _ = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("budget_exhausted", 0, 1)
    assert spec.fetch_calls == 0


def test_transient_failures_reset_breaker_and_stay_pending(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec(
        [_unit(f"u{i}") for i in range(4)],
        behavior={"u0": "transport", "u2": "transport"},
    )

    report, _ = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("complete", 2, 2)
    assert spec.fetch_calls == 4


def test_empty_fetch_result_stays_pending(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u0")], behavior={"u0": "empty"})

    report, emitted = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("complete", 0, 1)
    assert ctx.catalog.latest(source=_FAKE_SOURCE, natural_keys={"u0"}) == {}
    assert [line["phase"] for line in emitted] == ["plan", "chunk", "done"]


def test_unreachable_provider_before_first_chunk(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u0")], healthy=False)

    report, _ = _run(spec, ctx)

    assert (report.status, report.done, report.pending_left) == ("provider_unreachable", 0, 1)
    assert spec.fetch_calls == 0


def test_invalid_arguments_fail_closed(tmp_path: Path) -> None:
    from src.data.jobs.runner import JobUnit, run_job

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([])
    with pytest.raises(ValueError, match="chunk_size"):
        run_job(spec, ctx, chunk_size=0, max_chunks=None, dry_run=True, emit=lambda _p: None)
    with pytest.raises(ValueError, match="max_chunks"):
        run_job(spec, ctx, chunk_size=1, max_chunks=0, dry_run=True, emit=lambda _p: None)
    with pytest.raises(ValueError, match="natural key"):
        JobUnit(source="fake", natural_key="  ", payload={}, max_requests=1)
    with pytest.raises(ValueError, match="max_requests"):
        JobUnit(source="fake", natural_key="k", payload={}, max_requests=0)


def test_window_wait_handles_naive_clock() -> None:
    from datetime import datetime

    from src.data.jobs.runner import seconds_until_window_end

    assert seconds_until_window_end(datetime(2026, 9, 24, 3, 0), [["12:00", "12:30"]]) == 1800.0
    assert seconds_until_window_end(datetime(2026, 9, 24, 4, 0), [["12:00", "12:30"]]) == 0.0


def test_build_job_context_defaults_to_default_key(tmp_path: Path) -> None:
    from src.config import load_provider_policy, load_runtime_config
    from src.data.jobs.runner import build_job_context

    runtime = _runtime(tmp_path)
    provider = load_provider_policy(load_runtime_config())
    ctx = build_job_context(runtime=runtime, provider=provider, key_env=None, collector=None)

    assert ctx.key_env == provider.default_key_env == "OPENDART_API_KEY_2"
    assert provider.default_key_env != provider.primary_key_env


def test_runner_policy_rejects_invalid_budgets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from src.config.providers import RunnerPolicy

    with pytest.raises(ValueError, match="quota provider"):
        RunnerPolicy(quota_provider="  ", daily_budget=10, daily_reserve=0, circuit_threshold=3, avoid_windows_kst=())
    with pytest.raises(ValueError, match="daily_budget"):
        RunnerPolicy(quota_provider="X", daily_budget=0, daily_reserve=0, circuit_threshold=3, avoid_windows_kst=())
    with pytest.raises(ValueError, match="daily_reserve"):
        RunnerPolicy(quota_provider="X", daily_budget=10, daily_reserve=-1, circuit_threshold=3, avoid_windows_kst=())
    with pytest.raises(ValueError, match="daily_reserve"):
        RunnerPolicy(quota_provider="X", daily_budget=10, daily_reserve=True, circuit_threshold=3, avoid_windows_kst=())
    with pytest.raises(ValueError, match="circuit_threshold"):
        RunnerPolicy(quota_provider="X", daily_budget=10, daily_reserve=0, circuit_threshold=0, avoid_windows_kst=())


def test_chunk_progress_carries_elapsed_and_eta(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    spec = _FakeSpec([_unit("u0")])

    _, emitted = _run(spec, ctx)

    chunk = next(line for line in emitted if line["phase"] == "chunk")
    assert chunk["done"] == 1
    assert chunk["pending"] == 0
    assert chunk["requests_used"] == 0
    assert chunk["elapsed_s"] == 0.0
    assert chunk["eta_s"] == 0.0
