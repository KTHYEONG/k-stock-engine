"""Shared client-credentials issuance invariants."""

from __future__ import annotations

import json

import pytest
import requests


class _FakeResponse:
    def __init__(self, payload=None, *, status_code=200, json_error=False):  # type: ignore[no-untyped-def]
        self._payload = payload
        self._json_error = json_error
        self.status_code = status_code

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_error:
            raise ValueError("not json")
        return self._payload

class _RecordingSession:
    def __init__(self, response):  # type: ignore[no-untyped-def]
        self._response = response
        self.calls: list[dict] = []

    def post(self, url, *, data=None, timeout=None):  # type: ignore[no-untyped-def]
        self.calls.append({"url": url, "data": dict(data or {}), "timeout": timeout})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _grants():  # type: ignore[no-untyped-def]
    from src.integrations.oauth import ClientCredentialsGrant

    ls = ClientCredentialsGrant(
        provider="LS",
        url="https://openapi.ls-sec.co.kr:8080/oauth2/token",
        endpoint="oauth2/token",
        timeout_seconds=10.0,
        require_http_ok=True,
        throttle_rsp_codes=frozenset(),
        default_expires_in_seconds=86400,
    )
    dbsec = ClientCredentialsGrant(
        provider="DBSEC",
        url="https://openapi.dbsec.co.kr:8443/oauth2/token",
        endpoint="oauth2/token",
        timeout_seconds=10.0,
        require_http_ok=False,
        throttle_rsp_codes=frozenset({"IGW00201"}),
        default_expires_in_seconds=None,
    )
    return ls, dbsec


def test_form_body_and_timeout() -> None:
    from src.integrations.oauth import issue_client_credentials_token

    ls, dbsec = _grants()
    for grant in (ls, dbsec):
        session = _RecordingSession(_FakeResponse({"access_token": "t", "expires_in": 3600}))
        got = issue_client_credentials_token(session, grant, app_key="K", app_secret="S")
        assert (got.access_token, got.expires_in_seconds) == ("t", 3600.0)
        assert len(session.calls) == 1
        call = session.calls[0]
        assert call["url"] == grant.url
        assert call["data"] == {
            "grant_type": "client_credentials",
            "appkey": "K",
            "appsecretkey": "S",
            "scope": "oob",
        }
        assert call["timeout"] == 10.0


@pytest.mark.parametrize(("raw", "expected"), [("3600", 3600.0), (3600.9, 3600.0), (0, 0.0), (-1, -1.0)])
def test_expiry_preserves_vendor_lifetime(raw: object, expected: float) -> None:
    from src.integrations.host_broker import IssuedToken
    from src.integrations.oauth import issue_client_credentials_token

    for grant in _grants():
        session = _RecordingSession(_FakeResponse({"access_token": "t", "expires_in": raw}))
        assert issue_client_credentials_token(session, grant, app_key="K", app_secret="S") == IssuedToken("t", expected)


def test_transport_error_is_retryable() -> None:
    from src.integrations.errors import ProviderRetryableError
    from src.integrations.oauth import issue_client_credentials_token

    ls, _ = _grants()
    original = requests.ConnectionError("down")
    session = _RecordingSession(original)
    with pytest.raises(ProviderRetryableError) as excinfo:
        issue_client_credentials_token(session, ls, app_key="K", app_secret="S")
    assert excinfo.value.__cause__ is original
    assert len(session.calls) == 1


@pytest.mark.parametrize("status_code", [199, 300, 302, 304, 403, 503])
def test_non_2xx_retryable_only_when_required(status_code: int) -> None:
    from src.integrations.errors import ProviderRetryableError
    from src.integrations.oauth import issue_client_credentials_token

    ls, dbsec = _grants()
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps({"access_token": "t", "expires_in": 3600}).encode()
    with pytest.raises(ProviderRetryableError) as excinfo:
        issue_client_credentials_token(
            _RecordingSession(response),
            ls,
            app_key="K",
            app_secret="S",
        )
    assert isinstance(excinfo.value.__cause__, requests.HTTPError)
    got = issue_client_credentials_token(
        _RecordingSession(response),
        dbsec,
        app_key="K",
        app_secret="S",
    )
    assert (got.access_token, got.expires_in_seconds) == ("t", 3600.0)


def test_invalid_json_retryable_non_object_terminal() -> None:
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.oauth import issue_client_credentials_token

    ls, _ = _grants()
    with pytest.raises(ProviderRetryableError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse(json_error=True)), ls, app_key="K", app_secret="S"
        )
    with pytest.raises(ProviderTerminalError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse(["a"])), ls, app_key="K", app_secret="S"
        )


def test_throttle_code_retryable() -> None:
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.oauth import issue_client_credentials_token

    ls, dbsec = _grants()
    with pytest.raises(ProviderRetryableError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse({"rsp_cd": "IGW00201"})), dbsec, app_key="K", app_secret="S"
        )
    with pytest.raises(ProviderTerminalError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse({"rsp_cd": "IGW00201"})), ls, app_key="K", app_secret="S"
        )
    payload = {"rsp_cd": " IGW00201 ", "access_token": "t", "expires_in": 3600}
    with pytest.raises(ProviderRetryableError):
        issue_client_credentials_token(_RecordingSession(_FakeResponse(payload)), dbsec, app_key="K", app_secret="S")
    assert issue_client_credentials_token(
        _RecordingSession(_FakeResponse(payload)), ls, app_key="K", app_secret="S"
    ).access_token == "t"


def test_missing_or_non_str_token_terminal() -> None:
    from src.integrations.errors import ProviderTerminalError
    from src.integrations.oauth import issue_client_credentials_token

    ls, _ = _grants()
    with pytest.raises(ProviderTerminalError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse({"rsp_cd": "00000"})), ls, app_key="K", app_secret="S"
        )
    with pytest.raises(ProviderTerminalError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse({"access_token": 5})), ls, app_key="K", app_secret="S"
        )


@pytest.mark.parametrize("raw", ["bad", True, None, float("inf"), float("nan"), 10**400])
def test_expiry_default_vs_required(raw) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderTerminalError
    from src.integrations.oauth import issue_client_credentials_token

    ls, dbsec = _grants()
    payload: dict = {"access_token": "t"}
    if raw is not None:
        payload["expires_in"] = raw
    got = issue_client_credentials_token(_RecordingSession(_FakeResponse(payload)), ls, app_key="K", app_secret="S")
    assert (got.access_token, got.expires_in_seconds) == ("t", 86400.0)
    with pytest.raises(ProviderTerminalError):
        issue_client_credentials_token(
            _RecordingSession(_FakeResponse(dict(payload))), dbsec, app_key="K", app_secret="S"
        )


def test_secrets_never_in_errors_or_logs(caplog) -> None:  # type: ignore[no-untyped-def]
    from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
    from src.integrations.oauth import issue_client_credentials_token

    ls, dbsec = _grants()
    key, secret, token = "SECRETKEY", "SECRETSEC", "SECRETTOKEN"
    branches = [
        (ls, _RecordingSession(requests.ConnectionError(f"{key} {secret} {token}"))),
        (ls, _RecordingSession(_FakeResponse(json_error=True))),
        (ls, _RecordingSession(_FakeResponse([key, secret, token]))),
        (ls, _RecordingSession(_FakeResponse({"access_token": token, "expires_in": 3600}, status_code=403))),
        (dbsec, _RecordingSession(_FakeResponse({"rsp_cd": "IGW00201", "access_token": token}))),
        (ls, _RecordingSession(_FakeResponse({"access_token": 5, "error": f"{key} {secret} {token}"}))),
        (dbsec, _RecordingSession(_FakeResponse({"access_token": token, "expires_in": secret}))),
    ]
    with caplog.at_level("DEBUG"):
        messages: list[str] = []
        for grant, session in branches:
            with pytest.raises((ProviderRetryableError, ProviderTerminalError)) as excinfo:
                issue_client_credentials_token(session, grant, app_key=key, app_secret=secret)
            messages.append(str(excinfo.value))
        for message in messages:
            for needle in (key, secret, token):
                assert needle not in message
        for record in caplog.records:
            text = f"{record.getMessage()} {record.args}"
            for needle in (key, secret, token):
                assert needle not in text
