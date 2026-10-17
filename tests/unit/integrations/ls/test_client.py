"""LS client invariants: contract token store, shared host pacing, trend reads."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

SESSION = date(2026, 1, 5)


class _FakeResponse:
    def __init__(self, payload=None, *, json_error=False, status_code=200):  # type: ignore[no-untyped-def]
        self._payload = payload
        self._json_error = json_error
        self.status_code = status_code
        self.headers: dict = {}

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_error:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeTransport:
    """Scripted stand-in for HttpTransport.post."""

    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self._handler = handler
        self.calls: list[tuple] = []

    def post(self, endpoint, json_body, *, headers=None, classify=None):  # type: ignore[no-untyped-def]
        self.calls.append((endpoint, dict(json_body), dict(headers or {})))
        response = self._handler(endpoint, dict(json_body))
        if classify is not None:
            classify(response)
        return response


class _FakeSession:
    """Scripted stand-in for the OAuth HTTP session."""

    def __init__(self, handler):  # type: ignore[no-untyped-def]
        self._handler = handler
        self.posts = 0

    def post(self, url, *, data=None, timeout=None):  # type: ignore[no-untyped-def]
        self.posts += 1
        return self._handler(url, dict(data or {}))


def _trend_rows():  # type: ignore[no-untyped-def]
    return [
        {
            "date": "20260105",
            "tjj0008": "100",
            "tjj0009": "10",
            "tjj0010": "20",
            "tjj0016": "30",
            "tjj0007": "5",
            "tjj0011": "-15",
            "tjj0017": "-10",
            "tjj0000": "1",
            "tjj0001": "2",
            "tjj0002": "3",
            "tjj0003": "4",
            "tjj0004": "5",
            "tjj0005": "6",
            "tjj0006": "-41",
            "tjj0018": "-20",
        }
    ]


def _credentials():  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsCredentials

    return LsCredentials(app_key="key", app_secret="secret")


def _sha12(app_key: str) -> str:
    return hashlib.sha256(app_key.encode()).hexdigest()[:12]


def _marked_broker(tmp_path: Path, name: str = "broker") -> Path:
    broker = tmp_path / name
    broker.mkdir(parents=True, exist_ok=True)
    (broker / ".host-admission").write_text("ok", encoding="utf-8")
    return broker


def _contract_store(broker: Path, app_key: str = "key", *, clock=None, lock_timeout: float = 30.0, sleep=None):  # type: ignore[no-untyped-def]
    from src.integrations.host_broker import open_token_store

    params: dict = {"lock_timeout_seconds": lock_timeout, "expiry_margin_seconds": 180.0}
    if clock is not None:
        params["clock"] = clock
    if sleep is not None:
        params["sleep"] = sleep
    return open_token_store(broker, "ls", app_key, **params)


def _client(tmp_path, *, transport=None, session=None, token_store=None, app_key: str = "key", **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient, LsCredentials

    broker = _marked_broker(tmp_path, "client-broker")
    store = token_store if token_store is not None else _contract_store(broker, app_key)
    token_response = _FakeResponse({"access_token": "tok", "expires_in": 3600})
    return LsClient(
        LsCredentials(app_key=app_key, app_secret="secret"),
        transport=transport,
        token_store=store,
        session=session or _FakeSession(lambda url, data: token_response),
        **kwargs,
    )


def test_trend_success_returns_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    transport = _FakeTransport(lambda endpoint, body: _FakeResponse({"rsp_cd": "00000", "t1702OutBlock1": _trend_rows()}))
    client = _client(tmp_path, transport=transport)

    rows = client.inquire_investor_trend("005930", SESSION, SESSION)

    assert len(rows) == 1
    assert transport.calls[0][0] == "stock/frgr-itt"
    assert transport.calls[0][1]["t1702InBlock"]["shcode"] == "005930"


def test_retryable_rsp_code_surfaces_for_transport_retry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderRetryableError

    transport = _FakeTransport(lambda endpoint, body: _FakeResponse({"rsp_cd": "IGW00201"}))
    client = _client(tmp_path, transport=transport)

    with pytest.raises(ProviderRetryableError, match="IGW00201"):
        client.inquire_investor_trend("005930", SESSION, SESSION)


def test_terminal_rsp_code_fails_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderTerminalError

    transport = _FakeTransport(lambda endpoint, body: _FakeResponse({"rsp_cd": "99999"}))
    client = _client(tmp_path, transport=transport)

    with pytest.raises(ProviderTerminalError, match="99999"):
        client.inquire_investor_trend("005930", SESSION, SESSION)


def test_malformed_trend_payloads_fail_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError

    invalid_json = _client(
        tmp_path, transport=_FakeTransport(lambda e, b: _FakeResponse(json_error=True))
    )
    with pytest.raises(ProviderRetryableError):
        invalid_json.inquire_investor_trend("005930", SESSION, SESSION)

    non_object = _client(tmp_path, transport=_FakeTransport(lambda e, b: _FakeResponse([1])))
    with pytest.raises(ProviderTerminalError):
        non_object.inquire_investor_trend("005930", SESSION, SESSION)

    missing_block = _client(
        tmp_path, transport=_FakeTransport(lambda e, b: _FakeResponse({"rsp_cd": "00000"}))
    )
    with pytest.raises(ProviderTerminalError, match="t1702OutBlock1"):
        missing_block.inquire_investor_trend("005930", SESSION, SESSION)


def test_amount_unit_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    client = _client(tmp_path, transport=_FakeTransport(lambda e, b: _FakeResponse({})))

    with pytest.raises(ValueError, match="shares"):
        client.inquire_investor_trend("005930", SESSION, SESSION, unit="amount")


def test_ls_token_shared_across_clients(tmp_path) -> None:
    import json

    broker = _marked_broker(tmp_path)
    now = datetime(2026, 1, 5, tzinfo=UTC)
    first_store = _contract_store(broker, "shared-key", clock=lambda: now)
    second_store = _contract_store(broker, "shared-key", clock=lambda: now)
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "shared-tok", "expires_in": 3600}))
    from src.integrations.ls.client import LsClient
    from src.integrations.ls.client import LsCredentials as _Creds

    creds = _Creds(app_key="shared-key", app_secret="secret")
    first = LsClient(creds, token_store=first_store, session=session, transport=_FakeTransport(lambda e, b: _FakeResponse({})))
    second = LsClient(creds, token_store=second_store, session=session, transport=_FakeTransport(lambda e, b: _FakeResponse({})))

    assert first.ensure_token() == "shared-tok"
    assert second.ensure_token() == "shared-tok"
    assert session.posts == 1
    payload = json.loads((broker / f"token_ls_{_sha12('shared-key')}.json").read_text(encoding="utf-8"))
    assert payload["generation"] == 1


def test_ls_peer_token_reused(tmp_path) -> None:
    from src.integrations.host_broker import IssuedToken
    from src.integrations.ls.client import LsClient
    from src.integrations.ls.client import LsCredentials as _Creds

    broker = _marked_broker(tmp_path)
    now = datetime(2026, 1, 5, tzinfo=UTC)
    peer = _contract_store(broker, "peer-key", clock=lambda: now)
    peer.get_or_issue(lambda: IssuedToken("tok1", 3600.0))
    peer.replace_rejected("tok1", lambda: IssuedToken("tok2", 3600.0))
    peer.replace_rejected("tok2", lambda: IssuedToken("peer3", 3600.0))
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "new", "expires_in": 3600}))
    client = LsClient(
        _Creds(app_key="peer-key", app_secret="secret"),
        token_store=_contract_store(broker, "peer-key", clock=lambda: now),
        session=session,
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )

    assert client.ensure_token() == "peer3"
    assert session.posts == 0


def test_ls_expiring_token_reissued(tmp_path) -> None:
    import json

    from src.integrations.host_broker import IssuedToken
    from src.integrations.ls.client import LsClient
    from src.integrations.ls.client import LsCredentials as _Creds

    broker = _marked_broker(tmp_path)
    now = [datetime(2026, 1, 5, tzinfo=UTC)]
    peer = _contract_store(broker, "exp-key", clock=lambda: now[0])
    peer.get_or_issue(lambda: IssuedToken("old", 179.0))
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "fresh", "expires_in": 3600}))
    client = LsClient(
        _Creds(app_key="exp-key", app_secret="secret"),
        token_store=_contract_store(broker, "exp-key", clock=lambda: now[0]),
        session=session,
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )

    assert client.ensure_token() == "fresh"
    assert session.posts == 1
    payload = json.loads((broker / f"token_ls_{_sha12('exp-key')}.json").read_text(encoding="utf-8"))
    assert payload["generation"] == 2


def test_ls_memory_short_circuits_store(tmp_path) -> None:
    from src.integrations.ls.client import LsClient

    broker = _marked_broker(tmp_path)
    now = datetime(2026, 1, 5, tzinfo=UTC)
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "mem", "expires_in": 3600}))
    client = LsClient(
        _credentials(),
        token_store=_contract_store(broker, "key", clock=lambda: now),
        session=session,
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
        now=lambda: now,
    )

    assert client.ensure_token() == "mem"
    assert session.posts == 1
    (broker / f"token_ls_{_sha12('key')}.json").unlink()
    assert client.ensure_token() == "mem"
    assert session.posts == 1
    client.health_check()
    assert session.posts == 1


def test_ls_issue_failures_classify(tmp_path) -> None:
    import requests

    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.host_broker import token_record_path
    from src.integrations.ls.client import LsClient

    def _boom(url, data):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError("down")

    broker = _marked_broker(tmp_path, "fail-broker")
    now = datetime(2026, 1, 5, tzinfo=UTC)
    down = LsClient(
        _credentials(),
        token_store=_contract_store(broker, "key", clock=lambda: now),
        session=_FakeSession(_boom),
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )
    with pytest.raises(ProviderRetryableError):
        down.ensure_token()

    bad_json = LsClient(
        _credentials(),
        token_store=_contract_store(_marked_broker(tmp_path, "fail-broker-2"), "key", clock=lambda: now),
        session=_FakeSession(lambda u, d: _FakeResponse(json_error=True)),
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )
    with pytest.raises(ProviderRetryableError):
        bad_json.ensure_token()

    no_token = LsClient(
        _credentials(),
        token_store=_contract_store(_marked_broker(tmp_path, "fail-broker-3"), "key", clock=lambda: now),
        session=_FakeSession(lambda u, d: _FakeResponse({"rsp_cd": "00000"})),
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )
    with pytest.raises(ProviderTerminalError):
        no_token.ensure_token()

    broker4 = _marked_broker(tmp_path, "fail-broker-4")
    fallback_expiry = LsClient(
        _credentials(),
        token_store=_contract_store(broker4, "key", clock=lambda: now),
        session=_FakeSession(lambda u, d: _FakeResponse({"access_token": "t", "expires_in": "bad"})),
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )
    assert fallback_expiry.ensure_token() == "t"
    assert fallback_expiry._memory is not None
    assert fallback_expiry._memory.expires_at == now + timedelta(seconds=86400)

    broker5 = _marked_broker(tmp_path, "fail-broker-5")
    bool_expiry = LsClient(
        _credentials(),
        token_store=_contract_store(broker5, "key", clock=lambda: now),
        session=_FakeSession(lambda u, d: _FakeResponse({"access_token": "b", "expires_in": True})),
        transport=_FakeTransport(lambda e, b: _FakeResponse({})),
    )
    assert bool_expiry.ensure_token() == "b"
    assert bool_expiry._memory is not None
    assert bool_expiry._memory.expires_at == now + timedelta(seconds=86400)

    assert token_record_path(broker, "ls", "key").parent == broker
    assert _contract_store(broker, "key", clock=lambda: now).read() is None


def test_ls_lock_timeout_is_retryable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import fcntl
    import os
    import time as _time

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.ls.client import LsClient

    broker = _marked_broker(tmp_path)
    now = datetime(2026, 1, 5, tzinfo=UTC)
    lock_path = broker / f"token_ls_{_sha12('key')}.json.lock"
    lock_path.touch()
    fd = os.open(str(lock_path), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        mono = [1000.0]
        monkeypatch.setattr(_time, "monotonic", lambda: mono[0])

        def _sleep(seconds: float) -> None:
            mono[0] += 300.0

        client = LsClient(
            _credentials(),
            token_store=_contract_store(broker, "key", clock=lambda: now, sleep=_sleep),
            session=_FakeSession(lambda u, d: _FakeResponse({"access_token": "t", "expires_in": 3600})),
            transport=_FakeTransport(lambda e, b: _FakeResponse({})),
        )
        with pytest.raises(ProviderRetryableError):
            client.ensure_token()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_ls_requests_book_shared_host_slot(tmp_path) -> None:
    from src.integrations.host_broker import AdmissionSlot

    broker = _marked_broker(tmp_path)
    key = "pace-key"
    moment = [1000.0]
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        moment[0] += seconds

    clock = lambda: moment[0]  # noqa: E731
    peer = AdmissionSlot(broker, "ls", key, clock=clock, sleep=_sleep)
    peer.wait_turn()
    moment[0] += 0.5
    first = AdmissionSlot(broker, "ls", key, clock=clock, sleep=_sleep)
    second = AdmissionSlot(broker, "ls", key, clock=clock, sleep=_sleep)
    first.wait_turn()
    grants = [moment[0]]
    second.wait_turn()
    grants.append(moment[0])

    assert grants[0] >= 1000.55 - 1e-9
    assert grants[1] - grants[0] >= 1.05 - 1e-9
    names = sorted(p.name for p in broker.iterdir() if p.name != ".host-admission")
    assert names == [f"admission_ls_{_sha12(key)}.state"]

    from src.integrations.ls.client import LsClient

    broker2 = _marked_broker(tmp_path, "pace-client")
    now2 = datetime(2026, 1, 5, tzinfo=UTC)
    wired = LsClient(
        _credentials(),
        token_store=_contract_store(broker2, "key", clock=lambda: now2),
        session=_FakeSession(lambda u, d: _FakeResponse({"access_token": "t", "expires_in": 3600})),
        now=lambda: now2,
    )
    assert wired._transport._min_interval == pytest.approx(1.05)
    assert wired._transport._host_pacer.state_path.name == f"admission_ls_{_sha12('key')}.state"


def test_ls_pacing_file_name_matches_kca(tmp_path) -> None:
    from src.integrations.host_broker import AdmissionSlot

    broker = _marked_broker(tmp_path)
    slot = AdmissionSlot(broker, "ls", "K")

    assert slot.state_path.name == f"admission_ls_{_sha12('K')}.state"


def test_ls_pacer_unavailable_is_retryable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderRetryableError
    from src.integrations.ls.client import LsClient
    from src.integrations.transport import HttpTransport, RetryPolicy

    class _BlockedPacer:
        def wait_turn(self) -> None:
            raise OSError("slot locked")

    class _TrendSession:
        def __init__(self) -> None:
            self.sent = 0

        def post(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.sent += 1
            return _FakeResponse({"rsp_cd": "00000", "t1702OutBlock1": []})

    broker = _marked_broker(tmp_path)
    now = datetime(2026, 1, 5, tzinfo=UTC)
    session = _FakeSession(lambda u, d: _FakeResponse({"access_token": "t", "expires_in": 3600}))
    trend_session = _TrendSession()

    client = LsClient(
        _credentials(),
        token_store=_contract_store(broker, "key", clock=lambda: now),
        session=session,
        transport=HttpTransport(
            provider="LS",
            base_url="https://openapi.ls-sec.co.kr:8080",
            min_interval_seconds=1.05,
            retry=RetryPolicy(max_attempts=1),
            quota=None,
            timeout_seconds=15.0,
            session=trend_session,  # type: ignore[arg-type]
            host_pacer=_BlockedPacer(),  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(ProviderRetryableError):
        client.inquire_investor_trend("005930", SESSION, SESSION)
    assert session.posts == 1
    assert trend_session.sent == 0


def test_ls_no_local_min_interval_in_config() -> None:
    import pydantic

    from src.config.providers import LsPolicy

    values = _policy_ls_dump()
    values["min_interval_seconds"] = 1.05
    with pytest.raises(pydantic.ValidationError):
        LsPolicy.model_validate(values)


def _policy_ls_dump():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config()).ls.model_dump()


def test_ls_builder_uses_broker_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.ls.client import build_scoped_ls_client

    broker = _marked_broker(tmp_path)
    provider = load_provider_policy(load_runtime_config())
    monkeypatch.setenv(provider.ls.app_key_env, "k")
    monkeypatch.setenv(provider.ls.app_secret_env, "s")
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(broker))

    client = build_scoped_ls_client(policy=provider.ls)
    assert client.token_store._path.parent == broker
    assert not (load_runtime_config().logs_root / f"token_ls_{_sha12('k')}.json").exists()


def test_default_policy_used_when_not_given(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.ls.client import LsClient

    expected = load_provider_policy(load_runtime_config()).ls
    client = LsClient(_credentials(), token_store=_contract_store(_marked_broker(tmp_path), "key"))

    assert client._retryable_codes == tuple(expected.retryable_rsp_codes)
    assert client._transport._min_interval == pytest.approx(1.05)


def test_default_policy_without_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import src.integrations.ls.client as client_module

    def _missing(runtime):  # type: ignore[no-untyped-def]
        raise RuntimeError("no config")

    monkeypatch.setattr("src.config.providers.load_provider_policy", _missing)

    with pytest.raises(RuntimeError, match="no config"):
        client_module._default_ls_policy()


def test_credentials_reject_empty_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient, LsCredentials

    with pytest.raises(ValueError, match="credentials"):
        LsClient(
            LsCredentials(app_key=" ", app_secret="s"),
            token_store=_contract_store(_marked_broker(tmp_path), "key"),
        )


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.ls.client import LsCredentials

    provider = load_provider_policy(load_runtime_config())
    monkeypatch.setenv(provider.ls.app_key_env, "k")
    monkeypatch.setenv(provider.ls.app_secret_env, "s")

    credentials = LsCredentials.from_env(provider.ls)
    assert (credentials.app_key, credentials.app_secret) == ("k", "s")

    monkeypatch.delenv(provider.ls.app_key_env)
    with pytest.raises(ValueError, match="is not set"):
        LsCredentials.from_env(provider.ls)


def test_trend_rejects_non_date_arguments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pytest
    from datetime import date

    client = _client(tmp_path)

    with pytest.raises(ValueError, match="must be dates"):
        client.inquire_investor_trend("005930", "2026-01-05", date(2026, 1, 6))  # type: ignore[arg-type]


def test_quota_store_builds_ledger_gate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.quota import ProviderQuotaStateStore

    client = _client(tmp_path, quota_store=ProviderQuotaStateStore(tmp_path / "quota"))

    assert client._transport._quota is not None


def test_injected_store_controls_host_pacing_directory(tmp_path, monkeypatch) -> None:
    from src.integrations.ls.client import LsClient

    broker = _marked_broker(tmp_path)
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(tmp_path / "unmarked"))
    client = LsClient(_credentials(), token_store=_contract_store(broker))
    assert client._transport._host_pacer.state_path.parent == broker


def test_real_locked_slot_fails_closed(tmp_path) -> None:
    import fcntl

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.host_broker import AdmissionSlot, IssuedToken
    from src.integrations.ls.client import LsClient

    broker = _marked_broker(tmp_path)
    store = _contract_store(broker)
    store.get_or_issue(lambda: IssuedToken("cached", 3600.0))
    slot = AdmissionSlot(broker, "ls", "key", lock_timeout_seconds=0.01)
    slot.state_path.write_text("0", encoding="ascii")
    session = _FakeSession(lambda u, d: pytest.fail("HTTP must not be called"))
    client = LsClient(_credentials(), token_store=store, host_pacer=slot, session=session)
    with slot.state_path.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(ProviderRetryableError, match="admission lock"):
            client.inquire_investor_trend("005930", SESSION, SESSION)
    assert session.posts == 0


def test_client_requests_observe_peer_slots(tmp_path, monkeypatch) -> None:
    from src.integrations.host_broker import AdmissionSlot, IssuedToken
    from src.integrations.ls.client import LsClient

    broker = _marked_broker(tmp_path)
    moment = [1000.0]

    def sleep(seconds):
        moment[0] += seconds

    slot = AdmissionSlot(broker, "ls", "key", clock=lambda: moment[0], sleep=sleep)
    slot.state_path.write_text("1000.5", encoding="ascii")
    store = _contract_store(broker)
    store.get_or_issue(lambda: IssuedToken("cached", 3600.0))
    grants = []

    class Session:
        def post(self, *_args, **_kwargs):
            grants.append(moment[0])
            return _FakeResponse({"rsp_cd": "00000", "t1702OutBlock1": []})

    monkeypatch.setattr("src.integrations.ls.client.time.sleep", sleep)
    monkeypatch.setattr("src.integrations.ls.client.time.monotonic", lambda: moment[0])
    client = LsClient(_credentials(), token_store=store, host_pacer=slot, session=Session())
    for _ in range(2):
        assert client.inquire_investor_trend("005930", SESSION, SESSION) == ()
    assert grants[0] >= 1000.5
    assert grants[1] - grants[0] >= 1.05 - 1e-9


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "broken"])
def test_slot_recovers_invalid_peer_state(tmp_path, raw) -> None:
    from src.integrations.host_broker import AdmissionSlot

    broker = _marked_broker(tmp_path)
    slot = AdmissionSlot(broker, "ls", "key", clock=lambda: 1000.0)
    slot.state_path.write_text(raw, encoding="ascii")
    slot.wait_turn()
    assert float(slot.state_path.read_text(encoding="ascii")) == pytest.approx(1001.05)


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), float("inf")])
def test_slot_rejects_unbounded_lock_wait(tmp_path, bound) -> None:
    from src.integrations.host_broker import AdmissionSlot

    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        AdmissionSlot(tmp_path, "ls", "key", lock_timeout_seconds=bound)
