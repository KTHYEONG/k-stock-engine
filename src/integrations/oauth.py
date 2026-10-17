"""Shared OAuth2 client-credentials issuance for form-grant vendors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import requests

from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.host_broker import IssuedToken

__all__ = ["ClientCredentialsGrant", "issue_client_credentials_token"]


@dataclass(frozen=True, slots=True)
class ClientCredentialsGrant:
    """Vendor hooks for one OAuth2 client-credentials token endpoint."""

    provider: str
    url: str
    endpoint: str
    timeout_seconds: float
    require_http_ok: bool
    throttle_rsp_codes: frozenset[str]
    default_expires_in_seconds: int | None


def issue_client_credentials_token(
    session: requests.Session,
    grant: ClientCredentialsGrant,
    *,
    app_key: str,
    app_secret: str,
) -> IssuedToken:
    """POST the client-credentials form once and return the issued token."""
    try:
        response = session.post(
            grant.url,
            data={
                "grant_type": "client_credentials",
                "appkey": app_key,
                "appsecretkey": app_secret,
                "scope": "oob",
            },
            timeout=grant.timeout_seconds,
        )
        if grant.require_http_ok and not 200 <= response.status_code < 300:
            raise requests.HTTPError("Token endpoint returned non-2xx status", response=response)
    except requests.RequestException as exc:
        raise ProviderRetryableError(
            f"{grant.provider} token issuance failed for {grant.endpoint}",
            provider=grant.provider,
            endpoint=grant.endpoint,
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderRetryableError(
            f"{grant.provider} token issuance returned invalid JSON for {grant.endpoint}",
            provider=grant.provider,
            endpoint=grant.endpoint,
        ) from exc
    if not isinstance(data, Mapping):
        raise ProviderTerminalError(
            f"{grant.provider} token issuance returned non-object for {grant.endpoint}",
            provider=grant.provider,
            endpoint=grant.endpoint,
        )
    if str(data.get("rsp_cd", "")).strip() in grant.throttle_rsp_codes:
        raise ProviderRetryableError(
            f"{grant.provider} token issuance throttled for {grant.endpoint}",
            provider=grant.provider,
            endpoint=grant.endpoint,
        )
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise ProviderTerminalError(
            f"{grant.provider} token issuance failed for {grant.endpoint}",
            provider=grant.provider,
            endpoint=grant.endpoint,
        )
    try:
        raw_expires = data.get("expires_in")
        if isinstance(raw_expires, bool):
            raise TypeError
        expires_in = float(int(raw_expires))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        if grant.default_expires_in_seconds is None:
            raise ProviderTerminalError(
                f"{grant.provider} token issuance returned no valid expires_in for {grant.endpoint}",
                provider=grant.provider,
                endpoint=grant.endpoint,
            ) from exc
        expires_in = grant.default_expires_in_seconds
    return IssuedToken(token, float(expires_in))
