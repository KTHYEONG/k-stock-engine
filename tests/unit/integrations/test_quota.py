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
