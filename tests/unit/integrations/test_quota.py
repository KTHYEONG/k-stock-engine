from datetime import UTC, datetime


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

    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

    store = ProviderQuotaStateStore(tmp_path / "quota", daily_limit=2)
    moment = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)
    store.record_attempt(provider="OpenDART", endpoint="list.json", now=moment)
    store.record_attempt(provider="OpenDART", endpoint="fnlttSinglAcntAll.json", now=moment)
    with pytest.raises(ProviderQuotaBlocked):
        store.record_attempt(provider="OpenDART", endpoint="list.json", now=moment)
    store.record_attempt(provider="OpenDART", endpoint="list.json", now=datetime(2026, 9, 21, 15, 0, tzinfo=UTC))


def test_remaining_daily_attempts_sums_all_provider_endpoints(tmp_path) -> None:
    import pytest
    from src.integrations.quota import ProviderQuotaBlocked, ProviderQuotaStateStore

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
    with pytest.raises(ProviderQuotaBlocked, match="daily quota"):
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
