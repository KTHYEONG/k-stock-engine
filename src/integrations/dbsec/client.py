"""DB Securities quote/chart REST client."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import requests

from src.config.providers import DbsecPolicy
from src.config.secrets import read_secret
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.host_broker import (
    AdmissionSlot,
    IssuedToken,
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
    "DbsecClient",
    "DbsecCredentials",
    "DbsecTokenStore",
    "build_scoped_dbsec_client",
]

_LOG = logging.getLogger(__name__)

_PROVIDER: Final = "DBSEC"
_BASE_URL: Final = "https://openapi.dbsec.co.kr:8443"
_TOKEN_ENDPOINT: Final = "oauth2/token"  # noqa: S105 - URL path, not a credential
_TICKER_ENDPOINT: Final = "api/v1/quote/kr-stock/inquiry/stock-ticker"
_DAILY_ENDPOINT: Final = "api/v1/quote/kr-chart/day"
_MINUTE_ENDPOINT: Final = "api/v1/quote/kr-chart/min"
_TICKER_SCOPE: Final = "quote"
_CHART_SCOPE: Final = "chart"
_TIMEOUT_SECONDS: Final = 15.0
_TOKEN_TIMEOUT_SECONDS: Final = 10.0
_TOKEN_THROTTLE_RSP_CODES: Final = frozenset({"IGW00201"})
_MAX_HTTP_ATTEMPTS: Final = 3
_SUCCESS_RSP_CD: Final = "00000"
_LEGACY_FILENAME: Final = "dbsec_token_prod.json"
_LEGACY_LIVE_MARGIN_SECONDS: Final = 600.0
_TICKER_PATTERN: Final = re.compile(r"^[0-9A-Z]{6}$")
_LEGACY_FALLBACK_LOGGED: Final = {"done": False}


@dataclass(frozen=True, slots=True)
class DbsecCredentials:
    app_key: str
    app_secret: str

    @classmethod
    def from_env(cls, policy: DbsecPolicy) -> DbsecCredentials:
        """Read DB credentials from the env names declared in ``policy``."""
        app_key = read_secret(policy.app_key_env).strip()
        app_secret = read_secret(policy.app_secret_env).strip()
        return cls(app_key=app_key, app_secret=app_secret)


class _DbsecAuthError(ProviderTerminalError):
    """One DB auth-coded response awaiting a single token refresh."""


class DbsecTokenStore:
    """Contract v1 dbsec token store with a one-release read-only fallback."""

    def __init__(
        self,
        broker_dir: Path,
        app_key: str,
        *,
        legacy_dir: Path | None,
        lock_timeout_seconds: float,
        expiry_margin_seconds: float,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._broker_dir = Path(broker_dir)
        self._legacy_dir = Path(legacy_dir) if legacy_dir is not None else None
        self._expiry_margin = float(expiry_margin_seconds)
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._store: TokenStore = open_token_store(
            self._broker_dir,
            "dbsec",
            app_key,
            lock_timeout_seconds=float(lock_timeout_seconds),
            expiry_margin_seconds=float(expiry_margin_seconds),
            clock=self._clock,
            sleep=sleep,
        )

    @property
    def broker_dir(self) -> Path:
        return self._broker_dir

    def _read_legacy(self, now: datetime) -> str | None:
        if self._legacy_dir is None:
            return None
        try:
            path = self._legacy_dir / _LEGACY_FILENAME
            if path.stat().st_mode & 0o077:
                return None
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        token = payload.get("access_token")
        expire_at = payload.get("expire_at")
        issued_at = payload.get("issued_at")
        if not isinstance(token, str) or not token:
            return None
        if isinstance(expire_at, bool) or not isinstance(expire_at, (int, float)):
            return None
        if isinstance(issued_at, bool) or not isinstance(issued_at, (int, float)):
            return None
        if not float(expire_at) - now.timestamp() > _LEGACY_LIVE_MARGIN_SECONDS:
            return None
        return token

    def _wrap_issue(self, issue: Callable[[], IssuedToken]) -> Callable[[], IssuedToken]:
        def _issue() -> IssuedToken:
            issued = issue()
            expires = issued.expires_in_seconds
            if expires is None or not math.isfinite(expires) or expires <= self._expiry_margin:
                raise ProviderTerminalError(
                    f"{_PROVIDER} token expiry too short for {_TOKEN_ENDPOINT}",
                    provider=_PROVIDER,
                    endpoint=_TOKEN_ENDPOINT,
                )
            return issued

        return _issue

    def get_or_issue(self, issue: Callable[[], IssuedToken], *, reject: str | None = None) -> str:
        """Return a usable token, issuing through the host-shared store when needed."""
        try:
            if reject is None:
                if self._store.read() is None:
                    legacy = self._read_legacy(self._clock())
                    if legacy is not None:
                        if not _LEGACY_FALLBACK_LOGGED["done"]:
                            _LEGACY_FALLBACK_LOGGED["done"] = True
                            _LOG.info(
                                "[SYS] stage=dbsec_token status=LEGACY_FALLBACK path=%s",
                                _LEGACY_FILENAME,
                            )
                        return legacy
                return self._store.get_or_issue(self._wrap_issue(issue)).access_token
            return self._store.replace_rejected(reject, self._wrap_issue(issue)).access_token
        except TokenStoreLockTimeout as exc:
            raise ProviderRetryableError(
                f"{_PROVIDER} token issuance failed for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc


class DbsecClient:
    """DB Securities quote/chart REST client (read-only TRs)."""

    def __init__(
        self,
        credentials: DbsecCredentials,
        *,
        policy: DbsecPolicy,
        token_store: DbsecTokenStore,
        transport: HttpTransport | None = None,
        scope_pacers: Mapping[str, HostSlotGate] | None = None,
        quota_store: ProviderQuotaStateStore | None = None,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not credentials.app_key.strip() or not credentials.app_secret.strip():
            raise ValueError("DBSEC credentials require a non-empty app key and secret")
        self._credentials = credentials
        self._policy = policy
        self._token_store = token_store
        self._session = session if session is not None else requests.Session()
        self._now = now or (lambda: datetime.now(UTC))
        self._retryable_codes = tuple(policy.retryable_rsp_codes)
        self._auth_codes = tuple(policy.auth_rsp_codes)
        broker_dir = token_store.broker_dir
        app_key = credentials.app_key
        if scope_pacers is not None:
            if not {_CHART_SCOPE, _TICKER_SCOPE}.issubset(scope_pacers):
                raise ValueError("scope_pacers must declare chart and quote")
            self._scope_pacers: Mapping[str, HostSlotGate] = scope_pacers
        else:
            self._scope_pacers = {
                _TICKER_SCOPE: AdmissionSlot(broker_dir, "dbsec", app_key, _TICKER_SCOPE),
                _CHART_SCOPE: AdmissionSlot(broker_dir, "dbsec", app_key, _CHART_SCOPE),
            }
        if transport is not None:
            self._transport = transport
            return
        gate: LedgerQuotaGate | None = None
        if quota_store is not None:
            gate = LedgerQuotaGate(quota_store, provider=_PROVIDER, daily_limit=policy.daily_limit)
            gate.bind_now(self._now)
        self._transport = HttpTransport(
            provider=_PROVIDER,
            base_url=_BASE_URL,
            min_interval_seconds=1.0 / contract_rate("dbsec"),
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=self._session,
            host_pacer=AdmissionSlot(broker_dir, "dbsec", app_key),
        )

    def _issue_token(self) -> IssuedToken:
        return issue_client_credentials_token(
            self._session,
            ClientCredentialsGrant(
                provider=_PROVIDER,
                url=f"{_BASE_URL}/{_TOKEN_ENDPOINT}",
                endpoint=_TOKEN_ENDPOINT,
                timeout_seconds=_TOKEN_TIMEOUT_SECONDS,
                require_http_ok=False,
                throttle_rsp_codes=_TOKEN_THROTTLE_RSP_CODES,
                default_expires_in_seconds=None,
            ),
            app_key=self._credentials.app_key,
            app_secret=self._credentials.app_secret,
        )

    def _hooks(self, endpoint: str, parsed: list[dict[str, Any]]) -> Callable[[requests.Response], None]:
        def _hook(response: requests.Response) -> None:
            is_ok = response.status_code == 200
            try:
                data = response.json()
            except ValueError as exc:
                if not is_ok:
                    return
                raise ProviderTerminalError(
                    f"{_PROVIDER} {endpoint} returned invalid JSON with rsp_cd unknown",
                    provider=_PROVIDER,
                    endpoint=endpoint,
                ) from exc
            if not isinstance(data, Mapping):
                if not is_ok:
                    return
                raise ProviderTerminalError(
                    f"{_PROVIDER} {endpoint} returned non-object with rsp_cd unknown",
                    provider=_PROVIDER,
                    endpoint=endpoint,
                )
            rsp_cd = str(data.get("rsp_cd", "") or "")
            if not is_ok and not rsp_cd:
                return
            if rsp_cd == _SUCCESS_RSP_CD:
                if is_ok:
                    parsed.append(dict(data))
                return
            if rsp_cd in self._retryable_codes:
                raise ProviderRetryableError(
                    f"{_PROVIDER} {endpoint} throttled with rsp_cd {rsp_cd}",
                    provider=_PROVIDER,
                    endpoint=endpoint,
                )
            if rsp_cd in self._auth_codes:
                raise _DbsecAuthError(
                    f"{_PROVIDER} {endpoint} auth failure with rsp_cd {rsp_cd}",
                    provider=_PROVIDER,
                    endpoint=endpoint,
                )
            raise ProviderTerminalError(
                f"{_PROVIDER} {endpoint} rejected with rsp_cd {rsp_cd}",
                provider=_PROVIDER,
                endpoint=endpoint,
            )

        return _hook

    def _post_once(self, endpoint: str, scope: str, body_in: dict[str, Any], token: str) -> dict[str, Any]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "cont_yn": "N",
            "cont_key": "",
        }
        parsed: list[dict[str, Any]] = []
        hook = self._hooks(endpoint, parsed)
        self._scope_pacers[scope].wait_turn()
        self._transport.post(endpoint, {"In": body_in}, headers=headers, classify=hook, error_classify=hook)
        return parsed[-1]

    def _post_with_auth_refresh(self, endpoint: str, scope: str, body_in: dict[str, Any]) -> dict[str, Any]:
        token = self._token_store.get_or_issue(self._issue_token)
        try:
            return self._post_once(endpoint, scope, body_in, token)
        except _DbsecAuthError:
            refreshed = self._token_store.get_or_issue(self._issue_token, reject=token)
            try:
                return self._post_once(endpoint, scope, body_in, refreshed)
            except _DbsecAuthError as exc:
                raise ProviderTerminalError(
                    f"{_PROVIDER} {endpoint} auth failure repeated after token refresh",
                    provider=_PROVIDER,
                    endpoint=endpoint,
                ) from exc

    def health_check(self) -> None:
        """Confirm a live token is obtainable without touching Bronze."""
        self._token_store.get_or_issue(self._issue_token)

    @staticmethod
    def _check_ticker(ticker: str) -> str:
        text = ticker.strip()
        if not _TICKER_PATTERN.fullmatch(text):
            raise ValueError(f"invalid ticker {ticker!r}: expected 6 chars [0-9A-Z]")
        return text

    def fetch_ticker_master(self, market_div: Literal["J", "NJ", "E", "EN"]) -> tuple[dict[str, Any], ...]:
        """Return JCODES rows exactly as returned for one market division."""
        if market_div not in ("J", "NJ", "E", "EN"):
            raise ValueError(f"invalid market_div {market_div!r}: expected J, NJ, E or EN")
        data = self._post_with_auth_refresh(_TICKER_ENDPOINT, _TICKER_SCOPE, {"InputCondMrktDivCode": market_div})
        rows = data.get("Out")
        if not isinstance(rows, list):
            raise ProviderTerminalError(
                f"{_PROVIDER} {_TICKER_ENDPOINT} Out missing or not a list",
                provider=_PROVIDER,
                endpoint=_TICKER_ENDPOINT,
            )
        return tuple(dict(row) for row in rows if isinstance(row, dict))

    def fetch_daily_chart(self, ticker: str, start: date, end: date, *, adjusted: bool) -> tuple[dict[str, Any], ...]:
        """Return CHARTDAY rows for ``[start, end]`` with zero-date padding rows removed."""
        symbol = self._check_ticker(ticker)
        if start > end:
            raise ValueError(f"invalid range {start!s} > {end!s}")
        data = self._post_with_auth_refresh(
            _DAILY_ENDPOINT,
            _CHART_SCOPE,
            {
                "InputOrgAdjPrc": "1" if adjusted else "0",
                "InputCondMrktDivCode": "J",
                "InputIscd1": symbol,
                "InputDate1": start.strftime("%Y%m%d"),
                "InputDate2": end.strftime("%Y%m%d"),
            },
        )
        rows = data.get("Out")
        if not isinstance(rows, list):
            raise ProviderTerminalError(
                f"{_PROVIDER} {_DAILY_ENDPOINT} Out missing or not a list",
                provider=_PROVIDER,
                endpoint=_DAILY_ENDPOINT,
            )
        return tuple(dict(row) for row in rows if isinstance(row, dict) and row.get("Date") != "00000000")

    def fetch_minute_chart(self, ticker: str, anchor: date, *, bar_count: int) -> tuple[dict[str, Any], ...]:
        """Return the newest ``bar_count`` 1-minute J bars at or before the end of ``anchor``."""
        symbol = self._check_ticker(ticker)
        if isinstance(bar_count, bool) or not 1 <= int(bar_count) <= 2000:
            raise ValueError(f"invalid bar_count {bar_count!r}: expected [1, 2000]")
        data = self._post_with_auth_refresh(
            _MINUTE_ENDPOINT,
            _CHART_SCOPE,
            {
                "dataCnt": str(int(bar_count)),
                "InputCondMrktDivCode": "J",
                "InputIscd1": symbol,
                "InputDate1": anchor.strftime("%Y%m%d"),
                "InputDivXtick": "60",
                "InputOrgAdjPrc": "0",
            },
        )
        rows = data.get("Out")
        if not isinstance(rows, list):
            raise ProviderTerminalError(
                f"{_PROVIDER} {_MINUTE_ENDPOINT} Out missing or not a list",
                provider=_PROVIDER,
                endpoint=_MINUTE_ENDPOINT,
            )
        return tuple(dict(row) for row in rows if isinstance(row, dict))


def build_scoped_dbsec_client(
    *,
    policy: DbsecPolicy,
    quota_store: ProviderQuotaStateStore | None = None,
    now: Callable[[], datetime] | None = None,
) -> DbsecClient:
    """Build the DB client on the host broker contract."""
    credentials = DbsecCredentials.from_env(policy)
    broker_dir = resolve_broker_dir(policy.broker_dir_env)
    raw_legacy = read_secret(policy.legacy_token_dir_env, default="")
    legacy_dir = Path(raw_legacy.strip()) if raw_legacy.strip() else None
    store = DbsecTokenStore(
        broker_dir,
        credentials.app_key,
        legacy_dir=legacy_dir,
        lock_timeout_seconds=float(policy.token_lock_timeout_seconds),
        expiry_margin_seconds=float(policy.token_expiry_margin_seconds),
        clock=now,
    )
    return DbsecClient(credentials, policy=policy, token_store=store, quota_store=quota_store, now=now)
