import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest


def test_record_attempt_under_concurrent_threads_never_crashes_and_counts_exactly(tmp_path) -> None:
    import threading

    from src.integrations.quota import ProviderQuotaStateStore

    # Given: a single store shared by many threads, as happens when DartXbrlCollector's
    # ThreadPoolExecutor drives one DartApiClient's requests concurrently.
    store = ProviderQuotaStateStore(tmp_path / "quota")
    errors: list[BaseException] = []

    def _hit() -> None:
        try:
            store.record_attempt(provider="OpenDART", endpoint="fnlttSinglAcntAll.json", now=datetime.now(UTC))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_hit) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Then: no thread ever raced on the shared temp file, and every increment landed
    # (no lost updates from an unserialized read-modify-write).
    assert errors == []
    state = store._load()
    assert state["OpenDART|fnlttSinglAcntAll.json"]["attempted_requests"] == 50


def test_daily_limit_is_shared_across_endpoints_and_resets_by_kst_day(tmp_path) -> None:
    import pytest

    from src.integrations.quota import ProviderQuotaExhaustedError, ProviderQuotaStateStore

    store = ProviderQuotaStateStore(tmp_path / "quota", daily_limit=2)
    moment = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)
    store.record_attempt(provider="OpenDART", endpoint="list.json", now=moment)
    store.record_attempt(provider="OpenDART", endpoint="fnlttSinglAcntAll.json", now=moment)
    with pytest.raises(ProviderQuotaExhaustedError):
        store.record_attempt(provider="OpenDART", endpoint="list.json", now=moment)
    store.record_attempt(provider="OpenDART", endpoint="list.json", now=datetime(2026, 9, 21, 15, 0, tzinfo=UTC))


def test_remaining_daily_attempts_sums_all_provider_endpoints(tmp_path) -> None:
    import pytest
    from src.integrations.quota import ProviderQuotaExhaustedError, ProviderQuotaStateStore

    store = ProviderQuotaStateStore(tmp_path / "quota")
    moment = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)
    store.record_attempt(provider="OpenDART", endpoint="list.json", now=moment)
    store.record_attempt(provider="OpenDART", endpoint="fnlttSinglAcntAll.json", now=moment)

    assert store.remaining_daily_attempts(provider="OpenDART", now=moment, daily_limit=10) == 8
    with pytest.raises(ValueError, match="daily_limit"):
        ProviderQuotaStateStore(tmp_path / "invalid", daily_limit=0)
    with pytest.raises(ValueError, match="daily_limit"):
        store.remaining_daily_attempts(provider="OpenDART", now=moment, daily_limit=0)
    limited = ProviderQuotaStateStore(tmp_path / "limited")
    limited.record_attempt(provider="OpenDART", endpoint="list.json", now=moment, daily_limit=1)
    with pytest.raises(ProviderQuotaExhaustedError, match="daily quota"):
        limited.acquire(provider="OpenDART", endpoint="fnlttSinglAcntAll.json", now=moment, daily_limit=1)


def test_add_attempts_folds_remote_usage_into_the_kst_day(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.integrations.quota import ProviderQuotaStateStore

    store = ProviderQuotaStateStore(tmp_path)
    now = datetime(2026, 9, 24, 3, tzinfo=UTC)
    store.record_attempt(provider="P", endpoint="e", now=now)
    store.add_attempts(provider="P", endpoint="e", day="2026-09-24", count=40)
    store.add_attempts(provider="P", endpoint="e", day="2026-09-24", count=0)

    assert store.remaining_daily_attempts(provider="P", now=now, daily_limit=100) == 59
    with pytest.raises(ValueError, match="negative"):
        store.add_attempts(provider="P", endpoint="e", day="2026-09-24", count=-1)
    with pytest.raises(ValueError, match="Invalid isoformat"):
        store.add_attempts(provider="P", endpoint="e", day="not-a-date", count=1)


def _record_many(root: str, count: int) -> None:
    from datetime import UTC, datetime

    from src.integrations.quota import ProviderQuotaStateStore

    store = ProviderQuotaStateStore(root)
    now = datetime(2026, 9, 25, tzinfo=UTC)
    for _ in range(count):
        store.record_attempt(provider="P", endpoint="e", now=now)


def test_two_processes_do_not_lose_attempts(tmp_path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from src.integrations.quota import ProviderQuotaStateStore

    root = str(tmp_path / "ledger")
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(_record_many, [root, root], [200, 200]))

    store = ProviderQuotaStateStore(root)
    state = store._load()
    assert state["P|e"]["attempted_requests"] == 400


def test_corrupt_ledger_reads_as_empty_and_helpers_roundtrip(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.integrations.quota import ProviderQuotaStateStore

    root = tmp_path / "ledger"
    root.mkdir(parents=True)
    (root / "quota_state.json").write_text("not json", encoding="utf-8")
    store = ProviderQuotaStateStore(root)
    assert store._load() == {}
    assert store.remaining_daily_attempts(provider="P", now=datetime(2026, 9, 25, tzinfo=UTC), daily_limit=10) == 10

    (root / "quota_state.json").write_text("[1, 2]", encoding="utf-8")
    assert store._load() == {}

    now = datetime(2026, 9, 25, tzinfo=UTC)
    store.record_attempt(provider="P", endpoint="e", now=now)
    assert store._load()["P|e"]["attempted_requests"] == 1
    store._save(store._load())
    assert store.remaining_daily_attempts(provider="P", now=now, daily_limit=10) == 9


def test_ledger_gate_uses_wall_clock_without_binding(tmp_path) -> None:
    from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore

    store = ProviderQuotaStateStore(tmp_path / "ledger")
    gate = LedgerQuotaGate(store, provider="P", daily_limit=100)
    gate.acquire(endpoint="e")
    gate.record_rate_limit(endpoint="e", retry_after=1.0)


_NOW = datetime(2026, 11, 10, 3, tzinfo=UTC)


def _seed(root, state: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
    root.mkdir(parents=True, exist_ok=True)
    (root / "quota_state.json").write_text(json.dumps(state), encoding="utf-8")


def _saved(root) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return json.loads((root / "quota_state.json").read_text(encoding="utf-8"))


def _aged(days: int, **extra: Any) -> dict[str, Any]:
    moment = _NOW - timedelta(days=days)
    return {"attempted_requests": 5, "daily_attempt_day": moment.date().isoformat(),
            "daily_attempted_requests": 5, "updated_at": moment.isoformat(), **extra}


def test_stale_entry_pruned_on_write(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    _seed(tmp_path, {"KIND|old": _aged(45)})
    ProviderQuotaStateStore(tmp_path).acquire_and_record(provider="KIND", endpoint="new", now=_NOW)
    saved = _saved(tmp_path)
    assert "KIND|old" not in saved
    assert saved["KIND|new"]["attempted_requests"] == 1


def test_recent_entry_kept(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    _seed(tmp_path, {"KIND|recent": _aged(10)})
    ProviderQuotaStateStore(tmp_path).record_attempt(provider="KIND", endpoint="new", now=_NOW)
    assert "KIND|recent" in _saved(tmp_path)


def test_active_block_never_pruned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderQuotaExhaustedError
    from src.integrations.quota import ProviderQuotaStateStore

    blocked = (_NOW + timedelta(hours=1)).isoformat()
    _seed(tmp_path, {"KIND|blocked": _aged(90, blocked_until=blocked)})
    store = ProviderQuotaStateStore(tmp_path)
    store.record_attempt(provider="KIND", endpoint="new", now=_NOW)
    assert "KIND|blocked" in _saved(tmp_path)
    with pytest.raises(ProviderQuotaExhaustedError):
        store.acquire(provider="KIND", endpoint="blocked", now=_NOW)


def test_expired_block_old_entry_pruned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    expired = (_NOW - timedelta(hours=1)).isoformat()
    _seed(tmp_path, {"KIND|expired": _aged(90, blocked_until=expired)})
    ProviderQuotaStateStore(tmp_path).record_rate_limit(provider="KIND", endpoint="new", now=_NOW, retry_after=60.0)
    assert "KIND|expired" not in _saved(tmp_path)


def test_missing_dates_and_malformed_entries_kept(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    state: dict[str, Any] = {"KIND|empty": {}, "KIND|bad": {"updated_at": "not-a-date"}, "KIND|scalar": "garbage",
                             "KIND|badblock": _aged(90, blocked_until=12345)}
    _seed(tmp_path, state)
    ProviderQuotaStateStore(tmp_path).acquire_and_record(provider="KIND", endpoint="new", now=_NOW)
    saved = _saved(tmp_path)
    assert {k: saved[k] for k in state} == state


def test_daily_limit_accounting_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    today = _NOW.date().isoformat()
    recent = (_NOW - timedelta(hours=1)).isoformat()
    _seed(tmp_path, {
        "P|a": {"daily_attempt_day": today, "daily_attempted_requests": 3, "updated_at": recent},
        "P|b": {"daily_attempt_day": today, "daily_attempted_requests": 2, "updated_at": recent},
        "P|old": _aged(60),
    })
    store = ProviderQuotaStateStore(tmp_path)
    assert store.remaining_daily_attempts(provider="P", now=_NOW, daily_limit=10) == 5
    store.record_attempt(provider="P", endpoint="c", now=_NOW)
    assert "P|old" not in _saved(tmp_path)
    assert store.remaining_daily_attempts(provider="P", now=_NOW, daily_limit=10) == 4


def test_fallback_to_daily_attempt_day(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    day = (_NOW - timedelta(days=60)).date().isoformat()
    _seed(tmp_path, {"KIND|folded": {"daily_attempt_day": day, "daily_attempted_requests": 1, "attempted_requests": 1}})
    ProviderQuotaStateStore(tmp_path).acquire_and_record(provider="KIND", endpoint="new", now=_NOW)
    assert "KIND|folded" not in _saved(tmp_path)
