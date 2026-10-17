"""LS Securities OpenAPI client."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import requests

from src.config.providers import LsPolicy
from src.config.runtime import load_runtime_config
from src.config.secrets import read_secret
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.host_broker import (
    AdmissionSlot,
    IssuedToken,
    TokenRecord,
    TokenStore,
    TokenStoreLockTimeout,
    contract_rate,
    open_token_store,
    resolve_broker_dir,
)
from src.integrations.oauth import ClientCredentialsGrant, issue_client_credentials_token
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HostSlotGate, HttpTransport, RetryPolicy

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
_DEFAULT_EXPIRES_IN_SECONDS: Final = 86400


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
    """LS Securities OpenAPI client for investor flow (t1702)."""

    def __init__(
        self,
        credentials: LsCredentials,
        *,
        base_url: str = _BASE_URL,
        policy: LsPolicy | None = None,
        transport: HttpTransport | None = None,
        token_store: TokenStore | None = None,
        host_pacer: HostSlotGate | None = None,
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
        self._policy = resolved
        self._retryable_codes = tuple(resolved.retryable_rsp_codes)
        self._margin = float(resolved.token_expiry_margin_seconds)
        self._memory: TokenRecord | None = None
        if token_store is not None:
            self._token_store = token_store
        else:
            self._token_store = open_token_store(
                resolve_broker_dir(resolved.broker_dir_env),
                "ls",
                credentials.app_key,
                lock_timeout_seconds=float(resolved.token_lock_timeout_seconds),
                expiry_margin_seconds=float(resolved.token_expiry_margin_seconds),
                clock=self._now,
            )
        if transport is not None:
            self._transport = transport
            return
        gate: LedgerQuotaGate | None = None
        if quota_store is not None:
            gate = LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=resolved.daily_limit)
            gate.bind_now(self._now)
        pacer = host_pacer if host_pacer is not None else AdmissionSlot(
            self._token_store.broker_dir, "ls", credentials.app_key
        )
        self._transport = HttpTransport(
            provider=_PROVIDER,
            base_url=self.base_url,
            min_interval_seconds=1.0 / contract_rate("ls"),
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=self._session,
            sleep=lambda seconds: time.sleep(seconds),
            monotonic=lambda: time.monotonic(),
            host_pacer=pacer,
        )

    @property
    def token_store(self) -> TokenStore:
        return self._token_store

    def ensure_token(self) -> str:
        """Return a usable OAuth token from memory or the host-shared store."""
        now = self._now()
        memory = self._memory
        if (
            memory is not None
            and memory.expires_at is not None
            and now < memory.expires_at - timedelta(seconds=self._margin)
        ):
            return memory.access_token
        try:
            record = self._token_store.get_or_issue(self._issue_token)
        except TokenStoreLockTimeout as exc:
            raise ProviderRetryableError(
                f"LS token issuance failed for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc
        self._memory = record
        return record.access_token

    def _issue_token(self) -> IssuedToken:
        return issue_client_credentials_token(
            self._session,
            ClientCredentialsGrant(
                provider=_PROVIDER,
                url=f"{self.base_url}/{_TOKEN_ENDPOINT}",
                endpoint=_TOKEN_ENDPOINT,
                timeout_seconds=_TOKEN_TIMEOUT_SECONDS,
                require_http_ok=True,
                throttle_rsp_codes=frozenset(),
                default_expires_in_seconds=_DEFAULT_EXPIRES_IN_SECONDS,
            ),
            app_key=self.credentials.app_key,
            app_secret=self.credentials.app_secret,
        )

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
        """Request daily per-investor net trading for one symbol over a date range."""
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

        try:
            self._transport.post(_TREND_ENDPOINT, body, headers=headers, classify=_classify)
        except OSError as exc:
            raise ProviderRetryableError(
                f"LS t1702 pacing failed for {_TREND_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TREND_ENDPOINT,
            ) from exc
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
    now: Callable[[], datetime] | None = None,
) -> LsClient:
    """Build the LS collector for one declared key."""
    return LsClient(
        LsCredentials.from_env(policy),
        policy=policy,
        quota_store=quota_store,
        now=now,
    )
