"""DB Securities client invariants: token store, rsp-code classes, paced reads."""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime


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
    def __init__(self) -> None:
        self.calls = 0

    def wait_turn(self) -> None:
        self.calls += 1


class _ScriptedTokenStore:
    def __init__(self, tokens):  # type: ignore[no-untyped-def]
        self._tokens = list(tokens)
        self.rejects: list = []
        self.cache_dir = None

    def get_or_issue(self, issue, *, reject=None):  # type: ignore[no-untyped-def]
        self.rejects.append(reject)
        return self._tokens[min(len(self.rejects) - 1, len(self._tokens) - 1)]


def _policy():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config()).dbsec


def _credentials(key="key", secret="sec"):  # type: ignore[no-untyped-def]  # noqa: S107
    from src.integrations.dbsec.client import DbsecCredentials

    return DbsecCredentials(app_key=key, app_secret=secret)


def _store(tmp_path, **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecTokenStore

    params: dict = {"min_issue_interval_seconds": 61.0}
    params.update(kwargs)
    return DbsecTokenStore(tmp_path / "cache", **params)


def _client(tmp_path, *, script, tokens=("tok",), pacers=None, token_store=None):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecClient

    store = token_store
    if store is None:
        store = _store(tmp_path)
    if getattr(store, "cache_dir", None) is None:
        store.cache_dir = tmp_path / "cache"
    return DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=store,
        transport=_FakeTransport(script),
        scope_pacers=pacers or {"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
        session=_FakeSession(tokens),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )


def _ok_script(rows):  # type: ignore[no-untyped-def]
    return [(200, {"rsp_cd": "00000", "Out": rows})]


def test_token_store_reuses_live_token_without_issuing(tmp_path) -> None:
    now = [1_000_000.0]
    store = _store(tmp_path, clock=lambda: now[0])

    first = store.get_or_issue(lambda: ("t1", 3600))

    def _boom():  # type: ignore[no-untyped-def]
        raise AssertionError("issue must not be called")

    assert store.get_or_issue(_boom) == first == "t1"


def test_token_store_issues_once_under_concurrent_callers(tmp_path) -> None:
    import time

    store = _store(tmp_path)
    calls = {"n": 0}
    lock = threading.Lock()

    def _issue():  # type: ignore[no-untyped-def]
        with lock:
            calls["n"] += 1
        time.sleep(0.2)
        return "shared", 7200

    results: list[str] = []

    def _work() -> None:
        results.append(store.get_or_issue(_issue))

    threads = [threading.Thread(target=_work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["shared", "shared"]
    assert calls["n"] == 1


def test_token_store_spaces_issuance_by_min_interval(tmp_path) -> None:
    now = [1_000_000.0]
    sleeps: list[float] = []
    store = _store(tmp_path, clock=lambda: now[0], sleep=sleeps.append)

    old = store.get_or_issue(lambda: ("old", 7200))
    now[0] += 10.0

    assert store.get_or_issue(lambda: ("new", 7200), reject=old) == "new"
    assert sleeps
    assert sleeps[0] >= 51.0 - 1e-9


def test_token_store_treats_loose_permissions_as_miss(tmp_path) -> None:
    sleeps: list[float] = []
    store = _store(tmp_path, sleep=sleeps.append)
    assert store.get_or_issue(lambda: ("t1", 7200)) == "t1"

    (tmp_path / "cache" / "dbsec_token_prod.json").chmod(0o644)
    seen: list[str] = []

    def _issue():  # type: ignore[no-untyped-def]
        seen.append("issued")
        return "t2", 7200

    assert store.get_or_issue(_issue) == "t2"
    assert seen == ["issued"]
    assert len(sleeps) == 1


def test_token_store_rejects_short_expiry(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    store = _store(tmp_path)

    with pytest.raises(ProviderTerminalError):
        store.get_or_issue(lambda: ("t1", 300))


def test_client_retries_throttle_code_on_http_403(tmp_path) -> None:
    rows = [{"JCODE": "005930"}]
    client = _client(tmp_path, script=[(403, {"rsp_cd": "IGW00201"}), *_ok_script(rows)])

    assert client.fetch_ticker_master("J") == tuple(rows)


def test_client_refreshes_token_once_on_auth_code(tmp_path) -> None:
    store = _ScriptedTokenStore(["tok", "tok2"])
    rows = [{"Date": "20261015"}]
    client = _client(
        tmp_path,
        script=[(200, {"rsp_cd": "IGW00123"}), *_ok_script(rows)],
        token_store=store,  # type: ignore[arg-type]
    )

    assert client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=True) == tuple(rows)
    assert store.rejects == [None, "tok"]


def test_client_fails_terminal_on_second_auth_failure(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderTerminalError

    store = _ScriptedTokenStore(["tok", "tok2"])
    client = _client(
        tmp_path,
        script=[(200, {"rsp_cd": "IGW00121"}), (200, {"rsp_cd": "IGW00121"})],
        token_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderTerminalError):
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)


def test_daily_chart_drops_zero_date_padding(tmp_path) -> None:
    rows = [{"Date": "00000000"}, {"Date": "20261015"}, {"Date": "20261014"}]
    client = _client(tmp_path, script=_ok_script(rows))

    got = client.fetch_daily_chart("005930", date(2026, 10, 14), date(2026, 10, 15), adjusted=False)

    assert tuple(row["Date"] for row in got) == ("20261015", "20261014")


def test_daily_chart_sends_adjustment_flag(tmp_path) -> None:
    transport = _FakeTransport(_ok_script([]) + _ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    for adjusted in (True, False):
        client = DbsecClient(
            _credentials(),
            policy=_policy(),
            token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
            transport=transport,
            scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
            session=_FakeSession(("tok",)),
            now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        )
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=adjusted)

    flags = [body["In"]["InputOrgAdjPrc"] for _, body, _ in transport.calls]
    assert flags == ["1", "0"]


def test_minute_chart_rejects_out_of_range_count(tmp_path) -> None:
    import pytest

    transport = _FakeTransport(_ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
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

    transport = _FakeTransport(_ok_script([]))
    from src.integrations.dbsec.client import DbsecClient

    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="ticker"):
        client.fetch_daily_chart("5930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)
    with pytest.raises(ValueError, match="ticker"):
        client.fetch_minute_chart("N-005930", date(2026, 10, 15), bar_count=10)
    assert transport.calls == []


def test_client_waits_scope_pacer_before_send(tmp_path) -> None:
    chart, quote_master = _RecordingPacer(), _RecordingPacer()
    client = _client(
        tmp_path,
        script=_ok_script([{"Date": "20261015"}]),
        pacers={"chart": chart, "quote_master": quote_master},  # type: ignore[dict-item]
    )

    client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)

    assert chart.calls == 1
    assert quote_master.calls == 0


def test_client_never_logs_credentials(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    import pytest

    from src.integrations.dbsec.client import DbsecClient, DbsecCredentials
    from src.integrations.errors import ProviderTerminalError

    key, secret, token = "DBKEY-UNIQUE-1", "DBSECRET-UNIQUE-2", "DBTOKEN-UNIQUE-3"
    store = _ScriptedTokenStore([token, token])
    client = DbsecClient(
        DbsecCredentials(app_key=key, app_secret=secret),
        policy=_policy(),
        token_store=store,  # type: ignore[arg-type]
        transport=_FakeTransport([(200, {"rsp_cd": "IGW00121"}), (200, {"rsp_cd": "IGW00121"})]),
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
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


def test_token_store_treats_corrupt_payloads_as_miss(tmp_path) -> None:
    import json

    from src.integrations.dbsec.client import DbsecTokenStore

    cases = [
        "not json{{{",
        json.dumps(["tok"]),
        json.dumps({}),
        json.dumps({"access_token": "", "expire_at": 9999999999, "issued_at": 1000}),
        json.dumps({"access_token": "t", "expire_at": True, "issued_at": 1000}),
        json.dumps({"access_token": "t", "expire_at": 9999999999, "issued_at": False}),
        json.dumps({"access_token": "t"}),
    ]
    for index, raw in enumerate(cases):
        cache = tmp_path / f"case-{index}"
        cache.mkdir(parents=True)
        (cache / "dbsec_token_prod.json").write_text(raw, encoding="utf-8")
        (cache / "dbsec_token_prod.json").chmod(0o600)
        store = DbsecTokenStore(cache, min_issue_interval_seconds=61.0)
        assert store.get_or_issue(lambda: ("fresh", 7200)) == "fresh"
        assert store.cache_dir == cache


def test_token_store_returns_peer_token_issued_during_spacing(tmp_path) -> None:
    import json

    now = [1_000_000.0]
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        payload = {"access_token": "peer", "expire_at": now[0] + 7000.0, "issued_at": now[0]}
        path = tmp_path / "cache" / "dbsec_token_prod.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)

    store = _store(tmp_path, clock=lambda: now[0], sleep=_sleep)
    old = store.get_or_issue(lambda: ("old", 7200))
    now[0] += 10.0

    def _boom():  # type: ignore[no-untyped-def]
        raise AssertionError("issue must not be called")

    assert store.get_or_issue(_boom, reject=old) == "peer"
    assert sleeps == [51.0]


def test_issue_token_branches(tmp_path) -> None:
    import pytest
    import requests

    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError

    def _issued(session, expected_posts=1):  # type: ignore[no-untyped-def]
        client = DbsecClient(
            _credentials(),
            policy=_policy(),
            token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
            transport=_FakeTransport(_ok_script([])),
            scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
            session=session,
            now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        )
        token, expires_in = client._issue_token()
        assert session.posts == expected_posts
        return token, expires_in

    assert _issued(_ScriptedSession([_FakeResponse({"access_token": "t", "expires_in": 3600})])) == ("t", 3600)
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

    session = _FakeSession(("tok",))
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_store(tmp_path),
        transport=_FakeTransport(_ok_script([])),
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
        session=session,
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    assert client.health_check() is None
    assert client.health_check() is None
    assert session.posts == 1


def test_minute_chart_success_sends_single_page_body(tmp_path) -> None:
    from src.integrations.dbsec.client import DbsecClient

    rows = [{"Time": "150000"}]
    transport = _FakeTransport(_ok_script(rows))
    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
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
    from src.integrations.dbsec.client import DbsecClient as _DbsecClient2

    client = _DbsecClient2(
        _credentials(),
        policy=_policy(),
        token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )
    with pytest.raises(ProviderTerminalError):
        client.fetch_daily_chart("005930", date(2026, 10, 1), date(2026, 10, 15), adjusted=False)
    assert len(transport.calls) == 1


def test_build_scoped_client_uses_env_cache_dir(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_runtime_config
    from src.integrations.dbsec.client import DbsecClient, build_scoped_dbsec_client

    monkeypatch.setenv("DB_APP_KEY", "k")
    monkeypatch.setenv("DB_APP_SECRET", "s")
    monkeypatch.setenv("DBSEC_TOKEN_CACHE_DIR", str(tmp_path / "shared"))

    scoped = build_scoped_dbsec_client(policy=_policy())
    assert isinstance(scoped, DbsecClient)
    assert scoped._token_store.cache_dir == tmp_path / "shared"

    monkeypatch.delenv("DBSEC_TOKEN_CACHE_DIR")
    fallback = build_scoped_dbsec_client(policy=_policy())
    assert fallback._token_store.cache_dir == load_runtime_config().logs_root


def test_client_builds_default_transport_and_pacers(tmp_path) -> None:
    import pytest

    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.quota import ProviderQuotaStateStore

    client = DbsecClient(
        _credentials(),
        policy=_policy(),
        token_store=_store(tmp_path),
        quota_store=ProviderQuotaStateStore(tmp_path / "quota"),
        session=_FakeSession(("tok",)),
        now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
    )

    assert client._transport.provider == "DBSEC"
    assert set(client._scope_pacers) == {"chart", "quote_master"}
    with pytest.raises(ValueError, match="credentials"):
        DbsecClient(_credentials(key=" ", secret="s"), policy=_policy(), token_store=_store(tmp_path))


def test_token_store_spaces_issuance_after_failed_attempt(tmp_path) -> None:
    import pytest

    from src.integrations.errors import ProviderRetryableError

    now = [1_000_000.0]
    sleeps: list[float] = []
    store = _store(tmp_path, clock=lambda: now[0], sleep=sleeps.append)

    def _throttled():  # type: ignore[no-untyped-def]
        raise ProviderRetryableError("throttled", provider="DBSEC", endpoint="oauth2/token")

    with pytest.raises(ProviderRetryableError):
        store.get_or_issue(_throttled)
    now[0] += 5.0

    assert store.get_or_issue(lambda: ("t1", 7200)) == "t1"
    assert sleeps == [56.0]


def _real_transport_client(tmp_path, responses):  # type: ignore[no-untyped-def]
    from src.integrations.dbsec.client import DbsecClient
    from src.integrations.transport import HttpTransport, RetryPolicy

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
        token_store=_ScriptedTokenStore(["tok"]),  # type: ignore[arg-type]
        transport=transport,
        scope_pacers={"chart": _NoOpPacer(), "quote_master": _NoOpPacer()},
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
