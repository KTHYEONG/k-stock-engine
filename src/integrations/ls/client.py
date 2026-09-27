"""LS Securities OpenAPI client."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import requests

from src.config.providers import LsPolicy
from src.config.runtime import load_runtime_config
from src.config.secrets import read_secret
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HttpTransport, RetryPolicy, TokenCache

__all__ = [
    "LsClient",
    "LsCredentials",
    "build_scoped_ls_client",
]

_PROVIDER = "LS"
_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
_TOKEN_ENDPOINT = "oauth2/token"  # noqa: S105 - URL path, not a credential
_TREND_ENDPOINT = "stock/frgr-itt"
_TIMEOUT_SECONDS: Final = 15.0
_TOKEN_TIMEOUT_SECONDS: Final = 10.0
_MAX_HTTP_ATTEMPTS: Final = 3
_SUCCESS_RSP_CD: Final = "00000"


@dataclass(frozen=True, slots=True)
class LsCredentials:
    app_key: str
    app_secret: str

    @classmethod
    def from_env(cls, policy: LsPolicy | None = None) -> LsCredentials:
        """Read LS credentials using the env names declared in ``policy``."""
        resolved = policy if policy is not None else _default_ls_policy()
        app_key = read_secret(resolved.app_key_env).strip()
        app_secret = read_secret(resolved.app_secret_env).strip()
        return cls(app_key=app_key, app_secret=app_secret)


def _default_ls_policy() -> LsPolicy:
    from src.config.providers import load_provider_policy

    return load_provider_policy(load_runtime_config()).ls


class LsClient:
    """LS Securities OpenAPI client for investor flow (t1702).

    Transport-only: pacing, retries, and quotas belong to ``HttpTransport``
    and the quota ledger. Bronze writes belong to the data layer.
    """

    def __init__(
        self,
        credentials: LsCredentials,
        *,
        base_url: str = _BASE_URL,
        policy: LsPolicy | None = None,
        transport: HttpTransport | None = None,
        token_cache: TokenCache | None = None,
        session: requests.Session | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not credentials.app_key.strip() or not credentials.app_secret.strip():
            raise ValueError("LS credentials require a non-empty app key and secret")
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._now = now or (lambda: datetime.now(UTC))
        resolved = policy if policy is not None else _default_ls_policy()
        pace = float(resolved.min_interval_seconds)
        self._retryable_codes = tuple(resolved.retryable_rsp_codes)
        self._token: str | None = None
        self._token_expire_at: datetime | None = None
        if token_cache is not None:
            self.token_cache = token_cache
        else:
            self.token_cache = TokenCache(load_runtime_config().logs_root, provider="ls", env="prod")
        if transport is not None:
            self._transport = transport
            return
        gate: LedgerQuotaGate | None = None
        if quota_store is not None:
            gate = LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=resolved.daily_limit)
            gate.bind_now(self._now)
        self._transport = HttpTransport(
            provider=_PROVIDER,
            base_url=self.base_url,
            min_interval_seconds=pace,
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=self._session,
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
        )

    def ensure_token(self) -> str:
        """Return a live OAuth token, reusing the file cache across runs."""
        now = self._now()
        if self._token and self._token_expire_at and self._token_expire_at > now + timedelta(minutes=1):
            return self._token
        cached = self.token_cache.load()
        if cached is not None:
            token, expire_at_iso = cached
            try:
                expire_at = datetime.fromisoformat(expire_at_iso)
            except ValueError:
                expire_at = now
            if token and expire_at > now + timedelta(minutes=1):
                self._token = token
                self._token_expire_at = expire_at
                return token
        return self._request_new_token()

    def _request_new_token(self) -> str:
        try:
            response = self._session.post(
                f"{self.base_url}/{_TOKEN_ENDPOINT}",
                data={
                    "grant_type": "client_credentials",
                    "appkey": self.credentials.app_key,
                    "appsecretkey": self.credentials.app_secret,
                    "scope": "oob",
                },
                timeout=_TOKEN_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderRetryableError(
                f"LS token issuance failed for {_TOKEN_ENDPOINT}: {exc}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderRetryableError(
                f"LS token issuance returned invalid JSON for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc
        token = str(data.get("access_token", "")) if isinstance(data, Mapping) else ""
        if not token:
            raise ProviderTerminalError(
                f"LS token issuance failed for {_TOKEN_ENDPOINT}", provider=_PROVIDER, endpoint=_TOKEN_ENDPOINT
            )
        try:
            expires_in = int(data.get("expires_in", 86400))
        except (TypeError, ValueError):
            expires_in = 86400
        expire_at = self._now() + timedelta(seconds=max(expires_in - 120, 60))
        self._token = token
        self._token_expire_at = expire_at
        self.token_cache.save(token, expire_at.isoformat())
        return token

    def health_check(self) -> None:
        """Confirm collection credentials are live without touching Bronze."""
        self.ensure_token()

    def inquire_investor_trend(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        unit: str = "shares",
    ) -> tuple[dict[str, Any], ...]:
        """Request daily per-investor net trading for one symbol over a date range.

        Only the share-quantity mode is supported: it is the only mode whose unit
        has been verified against exchange volume. Amount mode is rejected rather
        than guessed.
        """
        if unit != "shares":
            raise ValueError(f"LS t1702 supports only unit='shares', got {unit!r}")
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            raise ValueError("start_date and end_date must be dates")
        token = self.ensure_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "tr_cd": "t1702",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body: dict[str, Any] = {
            "t1702InBlock": {
                "shcode": symbol.strip(),
                "fromdt": start_date.strftime("%Y%m%d"),
                "todt": end_date.strftime("%Y%m%d"),
                "volvalgb": "1",
                "msmdgb": "0",
                "gubun": "0",
                "exchgubun": "K",
            },
            "tr_cd": "t1702",
        }
        parsed: list[dict[str, Any]] = []

        def _classify(response: requests.Response) -> None:
            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderRetryableError(
                    f"LS t1702 returned invalid JSON for {_TREND_ENDPOINT}",
                    provider=_PROVIDER,
                    endpoint=_TREND_ENDPOINT,
                ) from exc
            if not isinstance(data, Mapping):
                raise ProviderTerminalError(
                    f"LS t1702 response must be an object for {_TREND_ENDPOINT}",
                    provider=_PROVIDER,
                    endpoint=_TREND_ENDPOINT,
                )
            rsp_cd = str(data.get("rsp_cd", "")).strip()
            if rsp_cd == _SUCCESS_RSP_CD:
                parsed.append(dict(data))
                return
            if rsp_cd in self._retryable_codes:
                raise ProviderRetryableError(
                    f"LS t1702 throttled ({rsp_cd}); certification blocked",
                    provider=_PROVIDER,
                    endpoint=_TREND_ENDPOINT,
                )
            raise ProviderTerminalError(
                f"LS t1702 non-success rsp_cd {rsp_cd!r}; certification blocked",
                provider=_PROVIDER,
                endpoint=_TREND_ENDPOINT,
            )

        self._transport.post(_TREND_ENDPOINT, body, headers=headers, classify=_classify)
        rows = parsed[0].get("t1702OutBlock1")
        if not isinstance(rows, list):
            raise ProviderTerminalError(
                f"LS t1702 t1702OutBlock1 missing or not a list for {_TREND_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TREND_ENDPOINT,
            )
        return tuple(dict(row) for row in rows if isinstance(row, dict))


def build_scoped_ls_client(
    *,
    policy: LsPolicy,
    quota_store: ProviderQuotaStateStore | None = None,
    token_cache_dir: Path | str | None = None,
    now: Callable[[], datetime] | None = None,
) -> LsClient:
    """Build the LS collector for one declared key from its provider policy.

    Raises:
        ValueError: the key's environment variables are unset.
    """
    cache_dir = Path(token_cache_dir) if token_cache_dir is not None else load_runtime_config().logs_root
    return LsClient(
        LsCredentials.from_env(policy),
        policy=policy,
        token_cache=TokenCache(cache_dir, provider="ls", env="prod"),
        quota_store=quota_store,
        now=now,
    )
