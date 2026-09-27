"""LS client invariants: OAuth cache, retryable rsp codes, paced transport."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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


def _token_cache(tmp_path):  # type: ignore[no-untyped-def]
    from src.integrations.transport import TokenCache

    return TokenCache(tmp_path, provider="ls", env="test")


def _client(tmp_path, *, transport=None, session=None, **kwargs):  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient

    token_response = _FakeResponse({"access_token": "tok", "expires_in": 3600})
    return LsClient(
        _credentials(),
        transport=transport,
        token_cache=_token_cache(tmp_path),
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


def test_token_is_cached_across_clients(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient

    cache = _token_cache(tmp_path)
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "cached-tok", "expires_in": 3600}))
    first = LsClient(_credentials(), token_cache=cache, session=session)
    second = LsClient(_credentials(), token_cache=cache, session=session)

    assert first.ensure_token() == "cached-tok"
    assert second.ensure_token() == "cached-tok"
    assert session.posts == 1


def test_expired_cache_is_refreshed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient

    cache = _token_cache(tmp_path)
    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    cache.save("stale", past)
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "fresh", "expires_in": 3600}))
    client = LsClient(_credentials(), token_cache=cache, session=session)

    assert client.ensure_token() == "fresh"
    assert session.posts == 1


def test_token_failures_classify(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import requests

    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError

    def _boom(url, data):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError("down")

    down = _client(tmp_path, session=_FakeSession(_boom))
    with pytest.raises(ProviderRetryableError):
        down.ensure_token()

    bad_json = _client(tmp_path, session=_FakeSession(lambda u, d: _FakeResponse(json_error=True)))
    with pytest.raises(ProviderRetryableError):
        bad_json.ensure_token()

    no_token = _client(tmp_path, session=_FakeSession(lambda u, d: _FakeResponse({"rsp_cd": "00000"})))
    with pytest.raises(ProviderTerminalError):
        no_token.ensure_token()

    fallback_expiry = _client(
        tmp_path, session=_FakeSession(lambda u, d: _FakeResponse({"access_token": "t", "expires_in": "bad"}))
    )
    assert fallback_expiry.ensure_token() == "t"


def test_default_policy_used_when_not_given(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.ls.client import LsClient

    expected = load_provider_policy(load_runtime_config()).ls
    client = LsClient(_credentials(), token_cache=_token_cache(tmp_path))

    assert client._retryable_codes == tuple(expected.retryable_rsp_codes)
    assert client._transport._min_interval == expected.min_interval_seconds


def test_default_policy_without_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import src.integrations.ls.client as client_module

    def _missing(runtime):  # type: ignore[no-untyped-def]
        raise RuntimeError("no config")

    monkeypatch.setattr("src.config.providers.load_provider_policy", _missing)

    with pytest.raises(RuntimeError, match="no config"):
        client_module._default_ls_policy()


def test_default_token_cache_dir_without_explicit_cache() -> None:
    from src.integrations.ls.client import LsClient

    client = LsClient(_credentials())
    assert client.token_cache.path.name.startswith("ls_token_")


def test_credentials_reject_empty_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient, LsCredentials

    with pytest.raises(ValueError, match="credentials"):
        LsClient(LsCredentials(app_key=" ", app_secret="s"), token_cache=_token_cache(tmp_path))


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


def test_scoped_builder_uses_declared_envs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config
    from src.integrations.ls.client import build_scoped_ls_client

    provider = load_provider_policy(load_runtime_config())
    monkeypatch.setenv(provider.ls.app_key_env, "k")
    monkeypatch.setenv(provider.ls.app_secret_env, "s")

    client = build_scoped_ls_client(policy=provider.ls, token_cache_dir=tmp_path)
    assert client.credentials.app_key == "k"


def test_memory_token_short_circuits_cache_and_network(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime, timedelta

    client = _client(tmp_path)
    client._token = "live"
    client._token_expire_at = datetime.now(UTC) + timedelta(hours=1)

    assert client.ensure_token() == "live"
    client.health_check()


def test_garbled_cache_expiry_falls_back_to_refresh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.ls.client import LsClient

    cache = _token_cache(tmp_path)
    cache.save("stale", "not-a-datetime")
    session = _FakeSession(lambda url, data: _FakeResponse({"access_token": "fresh", "expires_in": 3600}))
    client = LsClient(_credentials(), token_cache=cache, session=session)

    assert client.ensure_token() == "fresh"


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
