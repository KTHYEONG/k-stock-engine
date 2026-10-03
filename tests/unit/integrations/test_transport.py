"""Shared transport invariants: pacing, ledgering, retry classification."""
from __future__ import annotations

import threading


def _ok_response(status_code: int = 200, payload=None, headers=None):
    from types import SimpleNamespace

    payload = {"ok": True} if payload is None else payload
    return SimpleNamespace(status_code=status_code, json=lambda: payload, headers=headers or {}, content=b"{}")


def _transport(**overrides):
    from src.integrations.transport import HttpTransport, RetryPolicy

    params = {
        "provider": "TEST",
        "base_url": "https://example.invalid",
        "min_interval_seconds": 0.0,
        "retry": RetryPolicy(max_attempts=3, base_backoff_seconds=0.1, max_backoff_seconds=1.0),
        "quota": None,
        "timeout_seconds": 5.0,
    }
    params.update(overrides)
    return HttpTransport(**params)


def test_retry_after_wins_over_backoff() -> None:
    from types import SimpleNamespace

    sleeps: list[float] = []
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(status_code=429, headers={"Retry-After": "7"}, json=lambda: {})
        return _ok_response()

    transport = _transport(session=SimpleNamespace(get=fake_get), sleep=sleeps.append)

    assert transport.get("ep", {}).status_code == 200
    assert calls["n"] == 2
    assert sleeps != []
    assert max(sleeps) >= 7.0


def test_4xx_is_not_retried() -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError

    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return _ok_response(status_code=400)

    transport = _transport(session=SimpleNamespace(get=fake_get), sleep=lambda _s: None)

    with pytest.raises(ProviderTerminalError):
        transport.get("ep", {})
    assert calls["n"] == 1


def test_retryable_exhaustion_records_every_attempt(tmp_path) -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore

    from datetime import UTC, datetime

    root = tmp_path / "ledger"
    store = ProviderQuotaStateStore(root)
    now = datetime(2026, 9, 25, tzinfo=UTC)
    gate = LedgerQuotaGate(store, provider="TEST", daily_limit=1000)
    gate.bind_now(lambda: now)

    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return SimpleNamespace(status_code=503, headers={}, json=lambda: {})

    transport = _transport(session=SimpleNamespace(get=fake_get), quota=gate, sleep=lambda _s: None)

    with pytest.raises(ProviderRetryableError):
        transport.get("ep", {})
    assert calls["n"] == 3
    assert store.remaining_daily_attempts(provider="TEST", now=now, daily_limit=1000) == 997


def test_quota_blocks_before_sending() -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderQuotaExhaustedError

    class _BlockedGate:
        def acquire(self, *, endpoint: str) -> None:
            raise ProviderQuotaExhaustedError("blocked", provider="TEST", endpoint=endpoint)

        def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
            pass

    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return _ok_response()

    transport = _transport(session=SimpleNamespace(get=fake_get), quota=_BlockedGate())

    with pytest.raises(ProviderQuotaExhaustedError):
        transport.get("ep", {})
    assert calls["n"] == 0


def test_pacing_across_threads() -> None:
    from types import SimpleNamespace

    clock = {"t": 1000.0}
    lock = threading.Lock()
    stamps: list[float] = []

    def monotonic() -> float:
        with lock:
            return clock["t"]

    def sleep(seconds: float) -> None:
        with lock:
            clock["t"] += seconds

    def fake_get(*_a, **_kw):
        with lock:
            stamps.append(clock["t"])
        return _ok_response()

    transport = _transport(
        session=SimpleNamespace(get=fake_get),
        min_interval_seconds=0.2,
        sleep=sleep,
        monotonic=monotonic,
    )

    def _work() -> None:
        for _ in range(5):
            transport.get("ep", {})

    threads = [threading.Thread(target=_work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(stamps) == 10
    from itertools import pairwise

    ordered = sorted(stamps)
    for first, second in pairwise(ordered):
        assert second - first >= 0.2 - 1e-9


def test_token_cache_is_private_and_atomic(tmp_path) -> None:
    import threading

    from src.integrations.transport import TokenCache

    cache = TokenCache(tmp_path, provider="kis", env="demo")
    cache.save("token-1", "2030-01-01T00:00:00")
    assert (cache.path.stat().st_mode & 0o777) == 0o600

    errors: list[BaseException] = []
    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            try:
                loaded = cache.load()
                if loaded is not None:
                    token, _ = loaded
                    assert token in {"token-1", "token-2"}
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()

    readers = [threading.Thread(target=_reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    for _ in range(50):
        cache.save("token-2", "2030-01-01T00:00:00")
        cache.save("token-1", "2030-01-01T00:00:00")
    stop.set()
    for thread in readers:
        thread.join()

    assert errors == []


def test_retry_policy_rejects_non_positive_attempts() -> None:
    import pytest

    from src.integrations.transport import RetryPolicy

    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=True)


def test_parse_retry_after_branches() -> None:
    from src.integrations.transport import parse_retry_after

    assert parse_retry_after(None) is None
    assert parse_retry_after("   ") is None
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("not-a-date-nor-number") is None
    future = "Fri, 01 Jan 2038 00:00:00 GMT"
    assert parse_retry_after(future, now=0.0) > 0.0
    past = "Thu, 01 Jan 1970 00:00:00 GMT"
    assert parse_retry_after(past) == 0.0
    assert parse_retry_after(future, now=9999999999.0) == 0.0


def test_provider_property_and_post() -> None:
    from types import SimpleNamespace

    seen: dict[str, object] = {}

    def fake_post(url, *, params=None, json=None, headers=None, timeout=None):
        seen.update({"url": url, "json": json})
        return _ok_response()

    def fake_get(*_args, **_kwargs):
        raise AssertionError("GET must not be used")

    transport = _transport(session=SimpleNamespace(get=fake_get, post=fake_post), sleep=lambda _s: None)

    assert transport.provider == "TEST"
    response = transport.post("order", {"a": 1})

    assert response.status_code == 200
    assert seen["url"] == "https://example.invalid/order"
    assert seen["json"] == {"a": 1}


def test_form_post_sends_urlencoded_data() -> None:
    from types import SimpleNamespace

    seen: dict[str, object] = {}
    acquires: list[str] = []

    class _Gate:
        def acquire(self, *, endpoint: str) -> None:
            acquires.append(endpoint)

        def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
            pass

    def fake_post(url, *, params=None, data=None, json=None, headers=None, timeout=None):
        seen.update({"url": url, "data": data, "json": json})
        return _ok_response()

    transport = _transport(
        session=SimpleNamespace(get=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("GET must not be used")), post=fake_post),
        quota=_Gate(), sleep=lambda _s: None,
    )

    response = transport.post_form("disclosure/details.do", {"method": "searchDetailsSub", "pageIndex": "2"})

    assert response.status_code == 200
    assert seen["url"] == "https://example.invalid/disclosure/details.do"
    assert seen["data"] == {"method": "searchDetailsSub", "pageIndex": "2"}
    assert seen["json"] is None
    assert acquires == ["disclosure/details.do"]


def test_429_records_rate_limit_and_retries() -> None:
    from types import SimpleNamespace

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.transport import RetryPolicy

    recorded: list[dict[str, object]] = []

    class _Gate:
        def acquire(self, *, endpoint: str) -> None:
            pass

        def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
            recorded.append({"endpoint": endpoint, "retry_after": retry_after})

    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(status_code=429, headers={}, json=lambda: {})
        return _ok_response()

    transport = _transport(session=SimpleNamespace(get=fake_get), quota=_Gate(), sleep=lambda _s: None)

    assert transport.get("ep", {}).status_code == 200
    assert calls["n"] == 3
    assert len(recorded) == 2

    def always_429(*_args, **_kwargs):
        return SimpleNamespace(status_code=429, headers={"Retry-After": "1"}, json=lambda: {})

    transport = _transport(
        session=SimpleNamespace(get=always_429),
        quota=_Gate(),
        sleep=lambda _s: None,
        retry=RetryPolicy(max_attempts=2),
    )
    try:
        transport.get("ep", {})
    except ProviderRetryableError:
        pass
    else:
        raise AssertionError("expected retryable exhaustion")


def test_unlisted_5xx_is_terminal() -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderTerminalError

    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        return SimpleNamespace(status_code=501, headers={}, json=lambda: {})

    transport = _transport(session=SimpleNamespace(get=fake_get), sleep=lambda _s: None)

    with pytest.raises(ProviderTerminalError, match="501"):
        transport.get("ep", {})
    assert calls["n"] == 1


def test_connection_exhaustion_raises_retryable() -> None:
    import pytest
    from types import SimpleNamespace

    import requests

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.transport import RetryPolicy

    def always_fail(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("down")

    transport = _transport(
        session=SimpleNamespace(get=always_fail),
        retry=RetryPolicy(max_attempts=2, base_backoff_seconds=0.0, max_backoff_seconds=0.0),
        sleep=lambda _s: None,
    )

    with pytest.raises(ProviderRetryableError, match="transport failed"):
        transport.get("ep", {})


def test_quota_error_from_classify_is_recorded() -> None:
    import pytest
    from types import SimpleNamespace

    from src.integrations.errors import ProviderQuotaExhaustedError

    recorded: list[str] = []

    class _Gate:
        def acquire(self, *, endpoint: str) -> None:
            pass

        def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
            recorded.append(endpoint)

    def fake_get(*_args, **_kwargs):
        return _ok_response()

    def _classify(_response) -> None:
        raise ProviderQuotaExhaustedError("quota", provider="TEST", endpoint="ep")

    transport = _transport(session=SimpleNamespace(get=fake_get), quota=_Gate(), sleep=lambda _s: None)

    with pytest.raises(ProviderQuotaExhaustedError):
        transport.get("ep", {}, classify=_classify)
    assert recorded == ["ep"]


def test_token_cache_load_branches(tmp_path) -> None:
    import json

    from src.integrations.transport import TokenCache

    cache = TokenCache(tmp_path, provider="kis", env="demo")
    assert cache.path.name == "kis_token_demo.json"
    assert cache.load() is None

    cache.path.write_text("not json", encoding="utf-8")
    cache.path.chmod(0o600)
    assert cache.load() is None

    cache.path.write_text("[1, 2]", encoding="utf-8")
    cache.path.chmod(0o600)
    assert cache.load() is None

    cache.path.write_text(json.dumps({"access_token": "", "expire_at": ""}), encoding="utf-8")
    cache.path.chmod(0o600)
    assert cache.load() is None

    cache.save("tok", "2030-01-01T00:00:00")
    assert cache.load() == ("tok", "2030-01-01T00:00:00")

    cache.path.write_text("{}", encoding="utf-8")
    cache.path.chmod(0o644)
    assert cache.load() is None


def _host_pacer(state_path, *, interval=0.1, clock=None, sleeps=None):  # type: ignore[no-untyped-def]
    from src.integrations.transport import HostPacer

    params: dict = {"min_interval_seconds": interval}
    if clock is not None:
        params["clock"] = clock
    if sleeps is not None:
        params["sleep"] = sleeps.append
    else:
        params["sleep"] = lambda _s: None
    return HostPacer(state_path, **params)


def test_host_pacer_spaces_attempts_across_holders(tmp_path) -> None:
    import json

    import pytest

    state = tmp_path / "quota" / "dart_host_pacer.json"
    state.parent.mkdir(parents=True)
    clock = {"t": 1000.0}
    sleeps: list[float] = []
    first = _host_pacer(state, clock=lambda: clock["t"], sleeps=sleeps)
    second = _host_pacer(state, clock=lambda: clock["t"], sleeps=sleeps)

    slots: list[float] = []
    for turn in range(5):
        (first if turn % 2 == 0 else second).wait_turn()
        slots.append(json.loads(state.read_text(encoding="utf-8"))["next_allowed"] - 0.1)

    assert len(slots) == 5
    assert len(set(slots)) == 5
    assert slots == sorted(slots)
    from itertools import pairwise

    for before, after in pairwise(slots):
        assert after - before >= 0.1 - 1e-9
    for slot, slept in zip(slots[1:], sleeps, strict=True):
        assert slept == pytest.approx(slot - 1000.0)


def test_host_pacer_recovers_corrupt_state_file(tmp_path) -> None:
    import json

    state = tmp_path / "quota" / "dart_host_pacer.json"
    state.parent.mkdir(parents=True)
    state.write_text("garbage{{{", encoding="utf-8")
    _host_pacer(state, clock=lambda: 500.0).wait_turn()
    assert json.loads(state.read_text(encoding="utf-8"))["next_allowed"] == 500.1

    state.write_text(json.dumps({"next_allowed": "soon"}), encoding="utf-8")
    _host_pacer(state, clock=lambda: 500.0).wait_turn()
    assert json.loads(state.read_text(encoding="utf-8"))["next_allowed"] == 500.1


def test_transport_without_pacer_is_unchanged() -> None:
    from types import SimpleNamespace

    clock = {"t": 2000.0}
    sleeps: list[float] = []
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        return _ok_response()

    transport = _transport(
        session=SimpleNamespace(get=fake_get),
        min_interval_seconds=1.0,
        sleep=sleeps.append,
        monotonic=lambda: clock["t"],
        host_pacer=None,
    )

    assert transport.get("ep", {}).status_code == 200
    assert transport.get("ep", {}).status_code == 200
    assert calls["n"] == 2
    assert sleeps == [1.0]


def test_retries_also_wait_for_the_host_slot(tmp_path) -> None:
    from types import SimpleNamespace

    state = tmp_path / "quota" / "dart_host_pacer.json"
    state.parent.mkdir(parents=True)
    turns = {"n": 0}
    inner = _host_pacer(state, clock=lambda: 300.0)
    real_wait = inner.wait_turn

    def _counted() -> None:
        turns["n"] += 1
        real_wait()

    inner.wait_turn = _counted  # type: ignore[method-assign]
    calls = {"n": 0}

    def fake_get(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(status_code=503, headers={}, json=lambda: {})
        return _ok_response()

    transport = _transport(session=SimpleNamespace(get=fake_get), sleep=lambda _s: None, host_pacer=inner)

    assert transport.get("ep", {}).status_code == 200
    assert calls["n"] == 2
    assert turns["n"] == 2


def test_host_pacer_rejects_non_positive_interval(tmp_path) -> None:
    import pytest

    from src.integrations.transport import HostPacer

    with pytest.raises(ValueError, match="min_interval_seconds"):
        HostPacer(tmp_path / "pacer.json", min_interval_seconds=0)
    with pytest.raises(ValueError, match="min_interval_seconds"):
        HostPacer(tmp_path / "pacer.json", min_interval_seconds=-0.5)
