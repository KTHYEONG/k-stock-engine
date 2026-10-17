"""DB Securities client invariants: contract token store, legacy fallback, paced reads."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path


class _FakeResponse:
    def __init__(self, payload=None, *, status_code=200, json_error=False):  # type: ignore[no-untyped-def]
        self._payload = payload
        self._json_error = json_error
        self.status_code = status_code
        self.headers: dict = {}

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_error:
            raise ValueError("not json")
        return self._payload


class _FakeTransport:
    def __init__(self, script):  # type: ignore[no-untyped-def]
        self._script = list(script)
        self._index = 0
        self.calls: list[tuple] = []

    def post(self, endpoint, json_body, *, headers=None, classify=None, error_classify=None):  # type: ignore[no-untyped-def]
        from src.integrations.errors import ProviderRetryableError

        last = None
        while self._index < len(self._script):
            entry = self._script[self._index]
            self._index += 1
            status_code, payload = entry[0], entry[1]
            json_error = entry[2] if len(entry) > 2 else False
            response = _FakeResponse(payload, status_code=status_code, json_error=json_error)
            self.calls.append((endpoint, dict(json_body), dict(headers or {})))
            try:
                if status_code == 200:
                    if classify is not None:
                        classify(response)
                elif error_classify is not None:
                    error_classify(response)
                else:
                    raise AssertionError("non-200 without hook")
            except ProviderRetryableError as exc:
                last = exc
                continue
            return response
        raise last  # type: ignore[misc]


class _FakeSession:
    def __init__(self, tokens):  # type: ignore[no-untyped-def]
        self._tokens = list(tokens)
        self.posts = 0

    def post(self, url, *, data=None, timeout=None):  # type: ignore[no-untyped-def]
        self.posts += 1
        token = self._tokens[min(self.posts - 1, len(self._tokens) - 1)]
        return _FakeResponse({"access_token": token, "expires_in": 86400})


class _ScriptedSession:
    def __init__(self, outcomes):  # type: ignore[no-untyped-def]
        self._outcomes = list(outcomes)
        self.posts = 0

    def post(self, url, *, data=None, timeout=None):  # type: ignore[no-untyped-def]
        self.posts += 1
        outcome = self._outcomes[min(self.posts - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _NoOpPacer:
    def wait_turn(self) -> None:
        return None


class _RecordingPacer:
    def __init__(self, order: list[str] | None = None, name: str = "") -> None:
        self.calls = 0
        self._order = order
        self._name = name

    def wait_turn(self) -> None:
        self.calls += 1
        if self._order is not None:
            self._order.append(self._name)


class _OrderTransport:
    def __init__(self, order: list[str], recording):  # type: ignore[no-untyped-def]
        self._order = order
        self._recording = recording
        self.calls: list[tuple] = []

    def post(self, endpoint, json_body, *, headers=None, classify=None, error_classify=None):  # type: ignore[no-untyped-def]
        self.calls.append((endpoint, dict(json_body), dict(headers or {})))
        self._order.append("app")
        self._recording.calls += 1
        response = _FakeResponse({"rsp_cd": "00000", "Out": [{"JCODE": "005930"}]})
        if classify is not None:
            classify(response)
        return response


class _ScriptedTokenStore:
    def __init__(self, tokens, broker_dir: Path):  # type: ignore[no-untyped-def]
        self._tokens = list(tokens)
        self.rejects: list = []
        self._broker_dir = Path(broker_dir)

    @property
    def broker_dir(self) -> Path:
        return self._broker_dir

    def get_or_issue(self, issue, *, reject=None):  # type: ignore[no-untyped-def]
        self.rejects.append(reject)
        return self._tokens[min(len(self.rejects) - 1, len(self._tokens) - 1)]


def _policy():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config()).dbsec


def _credentials(key="key", secret="sec"):  # type: ignore[no-untyped-def]  # noqa: S107
    from src.integrations.dbsec.client import DbsecCredentials

    return DbsecCredentials(app_key=key, app_secret=secret)


def _sha12(app_key: str) -> str:
    return hashlib.sha256(app_key.encode()).hexdigest()[:12]


def _marked_broker(tmp_path: Path, name: str = "broker") -> Path:
    broker = tmp_path / name
    broker.mkdir(parents=True, exist_ok=True)
    (broker / ".host-admission").write_text("ok", encoding="utf-8")
    return broker


def _contract_store(broker: Path, app_key: str = "contract-key", *, legacy_dir=None, clock=None, sleep=None):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecTokenStore

    params: dict = {
        "legacy_dir": Path(legacy_dir) if legacy_dir is not None else None,
        "lock_timeout_seconds": 240.0,
        "expiry_margin_seconds": 720.0,
    }
    if clock is not None:
        params["clock"] = clock
    if sleep is not None:
        params["sleep"] = sleep
    return DbsecTokenStore(broker, app_key, **params)


def _write_legacy(legacy_dir: Path, *, legacy_token: str = "legacy-live", now: datetime, ttl: float = 3600.0, mode: int = 0o600, expire_raw=None, raw_text: str | None = None) -> Path:  # type: ignore[no-untyped-def]  # noqa: S107 - test fixture token, not a credential
    legacy_dir.mkdir(parents=True, exist_ok=True)
    path = legacy_dir / "dbsec_token_prod.json"
    if raw_text is not None:
        path.write_text(raw_text, encoding="utf-8")
    else:
        payload = {
            "access_token": legacy_token,
            "expire_at": expire_raw if expire_raw is not None else now.timestamp() + ttl,
            "issued_at": now.timestamp(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(mode)
    return path


def _client(tmp_path, *, script, tokens=("tok",), pacers=None, token_store=None):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path, "client-broker")
    store = token_store
    if store is None:
        store = _ScriptedTokenStore(["tok"], broker)
    return DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=store,
        transport=_FakeTransport(script),
        scope_pacers=pacers or {"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(tokens),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )


def _ok_script(rows):  # type: ignore[no-untyped-def]
    return [(200, {"rsp_cd": "00000", "Out": rows})]


def test_token_store_writes_contract_file_only(tmp_path) -> None:
    from src.integrations.host_broker import IssuedToken

    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    before = sorted(p.name for p in legacy_dir.iterdir())
    store = _contract_store(broker, "contract-key", legacy_dir=legacy_dir, clock=lambda: now)

    assert store.get_or_issue(lambda: IssuedToken("fresh", 7200)) == "fresh"

    digest = _sha12("contract-key")
    names = sorted(p.name for p in broker.iterdir() if p.name != ".host-admission")
    assert names == [f"token_dbsec_{digest}.json", f"token_dbsec_{digest}.json.lock", f"token_issue_dbsec_{digest}.state"]
    payload = json.loads((broker / f"token_dbsec_{digest}.json").read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "access_token", "issued_at", "expires_at", "generation"}
    assert payload["generation"] == 1
    assert (broker / f"token_dbsec_{digest}.json").stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in legacy_dir.iterdir()) == before


def test_token_store_reads_legacy_when_contract_absent(tmp_path) -> None:
    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    legacy_dir = tmp_path / "legacy"
    _write_legacy(legacy_dir, legacy_token="legacy-live", now=now)
    store = _contract_store(broker, "contract-key", legacy_dir=legacy_dir, clock=lambda: now)

    def _boom():  # type: ignore[no-untyped-def]
        raise AssertionError("issue must not be called")

    assert store.get_or_issue(_boom) == "legacy-live"
    digest = _sha12("contract-key")
    assert not (broker / f"token_dbsec_{digest}.json").exists()


def test_token_store_ignores_legacy_after_rejection(tmp_path) -> None:
    from src.integrations.host_broker import IssuedToken

    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    legacy_dir = tmp_path / "legacy"
    legacy_path = _write_legacy(legacy_dir, legacy_token="L", now=now)
    before = legacy_path.read_bytes()
    store = _contract_store(broker, "contract-key", legacy_dir=legacy_dir, clock=lambda: now)

    calls: list[str] = []

    def _issue():  # type: ignore[no-untyped-def]
        calls.append("issued")
        return IssuedToken("fresh", 7200)

    assert store.get_or_issue(_issue, reject="L") == "fresh"
    assert calls == ["issued"]
    assert legacy_path.read_bytes() == before


def _run_stale_legacy_case(tmp_path: Path, index: int, *, ttl: float, mode: int, expire_raw: object, now: datetime) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.host_broker import IssuedToken

    broker = _marked_broker(tmp_path, f"broker-{index}")
    legacy_dir = tmp_path / f"legacy-{index}"
    _write_legacy(
        legacy_dir,
        legacy_token="stale",
        now=now,
        ttl=ttl,
        mode=mode,
        expire_raw=expire_raw,
    )
    store = _contract_store(broker, "contract-key", legacy_dir=legacy_dir, clock=lambda: now)
    seen: list[str] = []

    def _issue() -> IssuedToken:
        seen.append("issued")
        return IssuedToken("fresh", 7200)

    assert store.get_or_issue(_issue) == "fresh"
    assert seen == ["issued"]


def test_token_store_ignores_stale_or_loose_legacy(tmp_path) -> None:
    now = datetime(2026, 10, 15, tzinfo=UTC)
    _run_stale_legacy_case(tmp_path, 0, ttl=500.0, mode=0o600, expire_raw=None, now=now)
    _run_stale_legacy_case(tmp_path, 1, ttl=3600.0, mode=0o644, expire_raw=None, now=now)
    _run_stale_legacy_case(tmp_path, 2, ttl=3600.0, mode=0o600, expire_raw=True, now=now)
    _run_stale_legacy_case(tmp_path, 3, ttl=3600.0, mode=0o600, expire_raw=float("nan"), now=now)


def test_token_store_prefers_contract_over_legacy(tmp_path) -> None:
    from src.integrations.host_broker import IssuedToken

    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    legacy_dir = tmp_path / "legacy"
    _write_legacy(legacy_dir, legacy_token="L", now=now)
    peer = _contract_store(broker, "contract-key", legacy_dir=None, clock=lambda: now)
    assert peer.get_or_issue(lambda: IssuedToken("C", 7200)) == "C"
    store = _contract_store(broker, "contract-key", legacy_dir=legacy_dir, clock=lambda: now)

    def _boom():  # type: ignore[no-untyped-def]
        raise AssertionError("issue must not be called")

    assert store.get_or_issue(_boom) == "C"


def test_token_store_rejects_short_expiry(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError
    from src.integrations.host_broker import IssuedToken

    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    for index, expiry in enumerate([720.0, 0.0, None, float("nan"), float("inf"), float("-inf")]):
        key = f"short-expiry-{index}"
        store = _contract_store(broker, key, clock=lambda: now)
        with pytest.raises(ProviderTerminalError):
            store.get_or_issue(lambda expiry=expiry: IssuedToken("t1", expiry))
        assert not (broker / f"token_dbsec_{_sha12(key)}.json").exists()


def test_token_store_preserves_reuse_window(tmp_path) -> None:
    from datetime import timedelta

    from src.integrations.host_broker import IssuedToken

    now = [datetime(2026, 10, 15, tzinfo=UTC)]
    broker = _marked_broker(tmp_path)

    def _sleep(seconds: float) -> None:
        now[0] = now[0] + timedelta(seconds=seconds)

    store = _contract_store(broker, "contract-key", clock=lambda: now[0], sleep=_sleep)

    assert store.get_or_issue(lambda: IssuedToken("t1", 721)) == "t1"

    def _boom():  # type: ignore[no-untyped-def]
        raise AssertionError("issue must not be called")

    assert store.get_or_issue(_boom) == "t1"

    now[0] = now[0] + timedelta(seconds=2)
    assert store.get_or_issue(lambda: IssuedToken("t2", 721)) == "t2"


def test_token_store_lock_timeout_is_retryable(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fcntl
    import os
    import time as _time

    import pytest

    from src.integrations.errors import ProviderRetryableError
    from src.integrations.host_broker import IssuedToken

    now = datetime(2026, 10, 15, tzinfo=UTC)
    broker = _marked_broker(tmp_path)
    digest = _sha12("contract-key")
    lock_path = broker / f"token_dbsec_{digest}.json.lock"
    lock_path.touch()
    fd = os.open(str(lock_path), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        mono = [1000.0]
        monkeypatch.setattr(_time, "monotonic", lambda: mono[0])

        def _sleep(seconds: float) -> None:
            mono[0] += 300.0

        store = _contract_store(broker, "contract-key", clock=lambda: now, sleep=_sleep)
        with pytest.raises(ProviderRetryableError, match="oauth2/token"):
            store.get_or_issue(lambda: IssuedToken("t1", 7200))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_client_default_slots_use_contract_files(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path)
    store = _contract_store(broker, "slot-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC))
    client = DbsecClient(
        _credentials(key="slot-key"),
        policy=_policy(),
        token_store=store,
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    digest = _sha12("slot-key")
    assert set(client._scope_pacers) == {"chart", "quote"}
    assert client._scope_pacers["chart"].state_path.name == f"admission_dbsec_{digest}_chart.state"
    assert client._scope_pacers["quote"].state_path.name == f"admission_dbsec_{digest}_quote.state"
    assert client._transport._host_pacer.state_path.name == f"admission_dbsec_{digest}.state"


def test_client_rejects_incomplete_scope_pacers(tmp_path) -> None:
    import pytest

    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path)
    store = _contract_store(broker, "slot-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC))
    with pytest.raises(ValueError, match="scope_pacers"):
        DbsecClient(
            _credentials(key="slot-key"),
            policy=_policy(),
            token_store=store,
            transport=_FakeTransport(_ok_script([])),
            scope_pacers={"chart": _NoOpPacer()},
            session=_FakeSession(("tok",)),
            now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        )


def test_client_waits_scope_then_app_slot(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path)
    store = _contract_store(broker, "order-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC))
    order: list[str] = []
    quote, chart = _RecordingPacer(order, "quote"), _RecordingPacer(order, "chart")
    app_recording = _RecordingPacer()
    client = DbsecClient(
        _credentials(key="order-key"),
        policy=_policy(),
        token_store=store,
        transport=_OrderTransport(order, app_recording),  # type: ignore[arg-type]
        scope_pacers={"chart": chart, "quote": quote},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    client.fetch_ticker_master("J")

    assert order == ["quote", "app"]
    assert (quote.calls, chart.calls, app_recording.calls) == (1, 0, 1)


def test_builder_resolves_broker_dir_and_legacy_env(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_runtime_config
    from src.integrations.dbsec.client import build_scoped_dbsec_client

    broker = _marked_broker(tmp_path, "shared")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setenv("DB_APP_KEY", "k")
    monkeypatch.setenv("DB_APP_SECRET", "s")
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(broker))
    monkeypatch.setenv("DBSEC_TOKEN_CACHE_DIR", str(legacy))

    scoped = build_scoped_dbsec_client(policy=_policy())
    assert scoped._token_store.broker_dir == broker

    monkeypatch.delenv("DBSEC_TOKEN_CACHE_DIR")
    without_legacy = build_scoped_dbsec_client(policy=_policy())
    assert without_legacy._token_store._legacy_dir is None
    assert without_legacy._token_store.broker_dir == broker
    assert without_legacy._token_store.broker_dir != load_runtime_config().logs_root


def test_builder_fails_closed_without_marker_in_container(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pathlib

    import pytest

    from src.integrations.host_broker import HostBrokerDirNotSharedError
    from src.integrations.dbsec.client import build_scoped_dbsec_client

    broker = tmp_path / "unshared"
    broker.mkdir()
    monkeypatch.setenv("DB_APP_KEY", "k")
    monkeypatch.setenv("DB_APP_SECRET", "s")
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(broker))
    monkeypatch.setenv("DBSEC_TOKEN_CACHE_DIR", str(tmp_path / "legacy"))

    orig_exists = pathlib.Path.exists

    def _fake_exists(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return True
        return orig_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", _fake_exists)
    with pytest.raises(HostBrokerDirNotSharedError):
        build_scoped_dbsec_client(policy=_policy())
    assert list(broker.iterdir()) == []


def test_builder_checks_credentials_first(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from src.config import ConfigError
    from src.integrations.dbsec.client import build_scoped_dbsec_client

    broker = tmp_path / "untouched"
    monkeypatch.setenv("KIS_TOKEN_CACHE_DIR", str(broker))
    monkeypatch.setenv("DBSEC_TOKEN_CACHE_DIR", str(tmp_path / "legacy"))
    monkeypatch.delenv("DB_APP_KEY", raising=False)
    monkeypatch.setenv("DB_APP_SECRET", "s")

    with pytest.raises(ConfigError):
        build_scoped_dbsec_client(policy=_policy())
    assert not broker.exists()


def test_client_retries_throttle_code_on_http_403(tmp_path) -> None:
    rows = [{"JCODE": "005930"}]
    client = _client(tmp_path, script=[(403, {"rsp_cd": "IGW00201"}), *_ok_script(rows)])

    assert client.fetch_ticker_master("J") == tuple(rows)


def test_client_refreshes_token_once_on_auth_code(tmp_path) -> None:
    broker = _marked_broker(tmp_path, "refresh-broker")
    store = _ScriptedTokenStore(["tok", "tok2"], broker)
    rows = [{"Date": "20261015"}]
    client = _client(
        tmp_path,
        script=[(200, {"rsp_cd": "IGW00123"}), *_ok_script(rows)],
        token_store=store,
    )

    assert client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=True) == tuple(rows)
    assert store.rejects == [None, "tok"]


def test_client_fails_terminal_on_second_auth_failure(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    broker = _marked_broker(tmp_path, "auth-broker")
    store = _ScriptedTokenStore(["tok", "tok2"], broker)
    client = _client(
        tmp_path,
        script=[(200, {"rsp_cd": "IGW00121"}), (200, {"rsp_cd": "IGW00121"})],
        token_store=store,
    )

    with pytest.raises(ProviderTerminalError):
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)


def test_daily_chart_drops_zero_date_padding(tmp_path) -> None:
    rows = [{"Date": "00000000"}, {"Date": "20261015"}, {"Date": "20261014"}]
    client = _client(tmp_path, script=_ok_script(rows))

    got = client.fetch_daily_chart("005930", date(2026, 10, 14), date(2026, 10, 15), adjusted=False)

    assert tuple(row["Date"] for row in got) == ("20261015", "20261014")


def test_daily_chart_sends_adjustment_flag(tmp_path) -> None:
    broker = _marked_broker(tmp_path, "flag-broker")
    transport = _FakeTransport(_ok_script([]) + _ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    for adjusted in (True, False):
        client = DbsecClient(
            _credentials(),
            policy=_policy(),
            token_store=_ScriptedTokenStore(["tok"], broker),
            transport=transport,
            scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
            session=_FakeSession(("tok",)),
            now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        )
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=adjusted)

    flags = [body["In"]["InputOrgAdjPrc"] for _, body, _ in transport.calls]
    assert flags == ["1", "0"]


def test_minute_chart_rejects_out_of_range_count(tmp_path) -> None:
    import pytest

    broker = _marked_broker(tmp_path, "bar-broker")
    transport = _FakeTransport(_ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="bar_count"):
        client.fetch_minute_chart("005930", date(2026, 10, 15), bar_count=2001)
    with pytest.raises(ValueError, match="bar_count"):
        client.fetch_minute_chart("005930", date(2026, 10, 15), bar_count=0)
    assert transport.calls == []


def test_client_rejects_malformed_ticker(tmp_path) -> None:
    import pytest

    broker = _marked_broker(tmp_path, "ticker-broker")
    transport = _FakeTransport(_ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="ticker"):
        client.fetch_daily_chart("5930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)
    with pytest.raises(ValueError, match="ticker"):
        client.fetch_minute_chart("N-005930", date(2026, 10, 15), bar_count=10)
    assert transport.calls == []


def test_client_waits_scope_pacer_before_send(tmp_path) -> None:
    chart, quote = _RecordingPacer(), _RecordingPacer()
    client = _client(
        tmp_path,
        script=_ok_script([{"Date": "20261015"}]),
        pacers={"chart": chart, "quote": quote},  # type: ignore[dict-item]
    )

    client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)

    assert chart.calls == 1
    assert quote.calls == 0


def test_client_never_logs_credentials(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    import pytest

    from src.integrations.dbsec.client import DbsecClient, DbsecCredentials
    from src.integrations.errors import ProviderTerminalError

    broker = _marked_broker(tmp_path, "nolog-broker")
    key, secret, token = "DBKEY-UNIQUE-1", "DBSECRET-UNIQUE-2", "DBTOKEN-UNIQUE-3"
    store = _ScriptedTokenStore([token, token], broker)
    client = DbsecClient(
        DbsecCredentials(app_key=key, app_secret=secret),
        policy=_policy(),
        token_store=store,
        transport=_FakeTransport([(200, {"rsp_cd": "IGW00121"}), (200, {"rsp_cd": "IGW00121"})]),
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession((token,)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderTerminalError) as excinfo:
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)

    for record in caplog.records:
        text = f"{record.getMessage()} {record.args}"
        for needle in (key, secret, token):
            assert needle not in text
    for needle in (key, secret, token):
        assert needle not in str(excinfo.value)


def test_credentials_from_env_round_trip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from src.config import ConfigError
    from src.integrations.dbsec.client import DbsecCredentials

    monkeypatch.setenv("DB_APP_KEY", "k")
    monkeypatch.setenv("DB_APP_SECRET", "s")
    assert DbsecCredentials.from_env(_policy()) == DbsecCredentials(app_key="k", app_secret="s")

    monkeypatch.delenv("DB_APP_KEY")
    with pytest.raises(ConfigError):
        DbsecCredentials.from_env(_policy())

    monkeypatch.setenv("DB_APP_KEY", "   ")
    with pytest.raises(ConfigError):
        DbsecCredentials.from_env(_policy())


def test_issue_token_delegates_with_dbsec_hooks(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.host_broker import IssuedToken

    broker = _marked_broker(tmp_path, "delegate-broker")
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=_FakeTransport(_ok_script([])),
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_ScriptedSession([_FakeResponse({"access_token": "t", "expires_in": 3600}, status_code=403)]),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    assert client._issue_token() == IssuedToken("t", 3600.0)


def test_issue_token_branches(tmp_path) -> None:
    import pytest
    import requests

    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.host_broker import IssuedToken

    broker = _marked_broker(tmp_path, "issue-broker")

    def _issued(session, expected_posts=1):  # type: ignore[no-untyped-def]
        client = DbsecClient(
            _credentials(),
            policy=_policy(),
            token_store=_ScriptedTokenStore(["tok"], broker),
            transport=_FakeTransport(_ok_script([])),
            scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
            session=session,
            now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        )
        issued = client._issue_token()
        assert session.posts == expected_posts
        return issued

    assert _issued(_ScriptedSession([_FakeResponse({"access_token": "t", "expires_in": 3600})])) == IssuedToken(
        "t", 3600.0
    )
    with pytest.raises(ProviderTerminalError):
        _issued(_ScriptedSession([_FakeResponse({"access_token": "t", "expires_in": "soon"})]))
    with pytest.raises(ProviderTerminalError):
        _issued(_ScriptedSession([_FakeResponse({"access_token": "t"})]))
    with pytest.raises(ProviderTerminalError):
        _issued(_ScriptedSession([_FakeResponse({"access_token": "t", "expires_in": True})]))

    with pytest.raises(ProviderRetryableError):
        _issued(_ScriptedSession([requests.ConnectionError("down")]))
    with pytest.raises(ProviderRetryableError):
        _issued(_ScriptedSession([_FakeResponse(None, json_error=True)]))
    with pytest.raises(ProviderRetryableError):
        _issued(_ScriptedSession([_FakeResponse({"rsp_cd": "IGW00201"})]))
    with pytest.raises(ProviderTerminalError):
        _issued(_ScriptedSession([_FakeResponse(["not", "object"])]))
    with pytest.raises(ProviderTerminalError):
        _issued(_ScriptedSession([_FakeResponse({"rsp_cd": "00000"})]))


def test_health_check_reuses_live_token(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path, "health-broker")
    session = _FakeSession(("tok",))
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_contract_store(broker, "health-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC)),
        transport=_FakeTransport(_ok_script([])),
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=session,
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    assert client.health_check() is None
    assert client.health_check() is None
    assert session.posts == 1


def test_minute_chart_success_sends_single_page_body(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient

    broker = _marked_broker(tmp_path, "minute-broker")
    rows = [{"Time": "150000"}]
    transport = _FakeTransport(_ok_script(rows))
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    got = client.fetch_minute_chart("005930", date(2026, 10, 15), bar_count=120)

    assert got == tuple(rows)
    _, body, _ = transport.calls[0]
    assert body["In"] == {
        "dataCnt": "120",
        "InputCondMrktDivCode": "J",
        "InputIscd1": "005930",
        "InputDate1": "20261015",
        "InputDivXtick": "60",
        "InputOrgAdjPrc": "0",
    }


def test_quote_response_shapes_are_guarded(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    with pytest.raises(ProviderTerminalError):
        _client(tmp_path, script=[(200, {"rsp_cd": "00000"})]).fetch_ticker_master("J")
    with pytest.raises(ProviderTerminalError):
        _client(tmp_path, script=[(200, {"rsp_cd": "00000", "Out": {"Date": "x"}})]).fetch_daily_chart(
            "005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False
        )
    with pytest.raises(ProviderTerminalError):
        _client(tmp_path, script=[(200, {"rsp_cd": "00000", "Out": "rows"})]).fetch_minute_chart(
            "005930", date(2026, 10, 15), bar_count=10
        )


def test_quote_request_validation(tmp_path) -> None:
    import pytest

    client = _client(tmp_path, script=_ok_script([]))

    with pytest.raises(ValueError, match="market_div"):
        client.fetch_ticker_master("K")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="range"):
        client.fetch_daily_chart("005930", date(2026, 10, 15), date(2026, 10, 1), adjusted=False)
    with pytest.raises(ValueError, match="bar_count"):
        client.fetch_minute_chart("005930", date(2026, 10, 15), bar_count=True)  # type: ignore[arg-type]


def test_quote_unknown_and_malformed_bodies_are_terminal(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    with pytest.raises(ProviderTerminalError, match="kr-chart/day"):
        _client(tmp_path, script=[(200, {"rsp_cd": "IGW99999"})]).fetch_daily_chart(
            "005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False
        )
    with pytest.raises(ProviderTerminalError):
        _client(tmp_path, script=[(200, ["not", "object"])]).fetch_daily_chart(
            "005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False
        )
    with pytest.raises(ProviderTerminalError):
        _client(tmp_path, script=[(200, None, True)]).fetch_daily_chart(
            "005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False
        )
    transport = _FakeTransport([(500, {"rsp_cd": "IGW99999"})])
    broker = _marked_broker(tmp_path, "terminal-broker")
    from src.integrations.dbsec.client import DbsecClient as _DbsecClient2

    client = _DbsecClient2(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )
    with pytest.raises(ProviderTerminalError):
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)
    assert len(transport.calls) == 1


def test_client_builds_default_transport_and_pacers(tmp_path) -> None:
    import pytest

    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.quota import ProviderQuotaStateStore

    broker = _marked_broker(tmp_path, "default-broker")
    client = DbsecClient(
        _credentials(key="default-key"),
        policy=_policy(),
        token_store=_contract_store(broker, "default-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC)),
        quota_store=ProviderQuotaStateStore(tmp_path / "quota"),
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    assert client._transport.provider == "DBSEC"
    assert set(client._scope_pacers) == {"chart", "quote"}
    with pytest.raises(ValueError, match="credentials"):
        DbsecClient(
            _credentials(key=" ", secret="s"),
            policy=_policy(),
            token_store=_contract_store(broker, "default-key", clock=lambda: datetime(2026, 10, 15, tzinfo=UTC)),
        )


def _real_transport_client(tmp_path, responses):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.transport import HttpTransport, RetryPolicy

    broker = _marked_broker(tmp_path, "real-broker")

    class _QuoteSession:
        def __init__(self) -> None:
            self.sent = 0

        def post(self, url, **kwargs):  # type: ignore[no-untyped-def]
            self.sent += 1
            return responses[self.sent - 1]

    session = _QuoteSession()
    transport = HttpTransport(
        provider="DBSEC",
        base_url="https://example.invalid",
        min_interval_seconds=0.001,
        retry=RetryPolicy(max_attempts=3),
        quota=None,
        timeout_seconds=1.0,
        session=session,  # type: ignore[arg-type]
        sleep=lambda _s: None,
    )
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"], broker),
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )
    return client, session


def test_real_transport_returns_final_200_body_after_retried_5xx(tmp_path) -> None:
    stale = _FakeResponse({"rsp_cd": "00000", "Out": [{"JCODE": "STALE1"}]}, status_code=500)
    fresh = _FakeResponse({"rsp_cd": "00000", "Out": [{"JCODE": "005930"}]})
    client, session = _real_transport_client(tmp_path, [stale, fresh])

    assert client.fetch_ticker_master("J") == ({"JCODE": "005930"},)
    assert session.sent == 2


def test_real_transport_retries_gateway_page_without_rsp_cd(tmp_path) -> None:
    gateway = _FakeResponse(None, status_code=502, json_error=True)
    fresh = _FakeResponse({"rsp_cd": "00000", "Out": [{"JCODE": "005930"}]})
    client, session = _real_transport_client(tmp_path, [gateway, fresh])

    assert client.fetch_ticker_master("J") == ({"JCODE": "005930"},)
    assert session.sent == 2


def test_real_transport_retries_uncoded_5xx_bodies(tmp_path) -> None:
    non_object = _FakeResponse(["gateway"], status_code=503)
    no_code = _FakeResponse({"message": "busy"}, status_code=503)
    fresh = _FakeResponse({"rsp_cd": "00000", "Out": [{"JCODE": "005930"}]})
    client, session = _real_transport_client(tmp_path, [non_object, no_code, fresh])

    assert client.fetch_ticker_master("J") == ({"JCODE": "005930"},)
    assert session.sent == 3


def test_real_transport_keeps_coded_403_terminal(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    client, session = _real_transport_client(tmp_path, [_FakeResponse({"rsp_cd": "IGW40011"}, status_code=403)])

    with pytest.raises(ProviderTerminalError, match="IGW40011"):
        client.fetch_ticker_master("J")
    assert session.sent == 1
