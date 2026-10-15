"""DB Securities quote/chart REST client."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import requests

from src.config.providers import DbsecPolicy
from src.config.runtime import load_runtime_config
from src.config.secrets import read_secret
from src.integrations.errors import ProviderRetryableError, ProviderTerminalError
from src.integrations.quota import LedgerQuotaGate, ProviderQuotaStateStore
from src.integrations.transport import HostPacer, HttpTransport, RetryPolicy

__all__ = [
    "DbsecClient",
    "DbsecCredentials",
    "DbsecTokenStore",
    "build_scoped_dbsec_client",
]

_PROVIDER: Final = "DBSEC"
_BASE_URL: Final = "https://openapi.dbsec.co.kr:8443"
_TOKEN_ENDPOINT: Final = "oauth2/token"  # noqa: S105 - URL path, not a credential
_TICKER_ENDPOINT: Final = "api/v1/quote/kr-stock/inquiry/stock-ticker"
_DAILY_ENDPOINT: Final = "api/v1/quote/kr-chart/day"
_MINUTE_ENDPOINT: Final = "api/v1/quote/kr-chart/min"
_TICKER_SCOPE: Final = "quote_master"
_CHART_SCOPE: Final = "chart"
_TIMEOUT_SECONDS: Final = 15.0
_TOKEN_TIMEOUT_SECONDS: Final = 10.0
_MAX_HTTP_ATTEMPTS: Final = 3
_SUCCESS_RSP_CD: Final = "00000"
_TOKEN_LIVE_MARGIN_SECONDS: Final = 600.0
_TOKEN_EXPIRY_MARGIN_SECONDS: Final = 120
_TICKER_PATTERN: Final = re.compile(r"^[0-9A-Z]{6}$")


@dataclass(frozen=True, slots=True)
class DbsecCredentials:
    app_key: str
    app_secret: str

    @classmethod
    def from_env(cls, policy: DbsecPolicy) -> DbsecCredentials:
        """Read DB credentials from the env names declared in ``policy``; raises ConfigError when unset."""
        app_key = read_secret(policy.app_key_env).strip()
        app_secret = read_secret(policy.app_secret_env).strip()
        return cls(app_key=app_key, app_secret=app_secret)


class _DbsecAuthError(ProviderTerminalError):
    """One DB auth-coded response awaiting a single token refresh."""


class DbsecTokenStore:
    """Host-shared DB OAuth token file with single-flight issuance under a file lock."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        min_issue_interval_seconds: float,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._dir = Path(cache_dir)
        self._min_interval = float(min_issue_interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._mutex = threading.Lock()

    @property
    def cache_dir(self) -> Path:
        return self._dir

    @property
    def _path(self) -> Path:
        return self._dir / "dbsec_token_prod.json"

    @property
    def _lock_path(self) -> Path:
        return self._dir / "dbsec_token_prod.json.lock"

    @property
    def _attempt_path(self) -> Path:
        return self._dir / "dbsec_token_prod.attempt"

    def _last_attempt(self) -> float | None:
        try:
            return float(self._attempt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _stamp_attempt(self, at: float) -> None:
        tmp_path = self._dir / f".dbsec_token_prod.attempt.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_path.write_text(repr(at), encoding="utf-8")
        os.replace(tmp_path, self._attempt_path)

    def _read(self) -> tuple[str, float, float] | None:
        try:
            if self._path.stat().st_mode & 0o077:
                return None
            payload = json.loads(self._path.read_text(encoding="utf-8"))
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
        return token, float(expire_at), float(issued_at)

    def _write(self, token: str, expire_at: float, issued_at: float) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._dir / f".dbsec_token_prod.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"access_token": token, "expire_at": expire_at, "issued_at": issued_at}))
            os.replace(tmp_path, self._path)
            self._path.chmod(0o600)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def get_or_issue(self, issue: Callable[[], tuple[str, int]], *, reject: str | None = None) -> str:
        """Return a live token, issuing via ``issue`` (returns token, expires_in seconds) only when none is live; ``reject`` marks a token the server refused."""
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._mutex, open(self._lock_path, "w") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                stored = self._read()
                now = self._clock()
                if stored is not None:
                    token, expire_at, _ = stored
                    if (reject is None or token != reject) and expire_at - now > _TOKEN_LIVE_MARGIN_SECONDS:
                        return token
                # Failed issuances (vendor IGW00201) leave no token file, so spacing keys on the last attempt.
                anchors = [at for at in (stored[2] if stored else None, self._last_attempt()) if at is not None]
                if anchors:
                    wait = self._min_interval - (now - max(anchors))
                    if wait > 0:
                        self._sleep(wait)
                        stored = self._read()
                        now = self._clock()
                        if stored is not None:
                            token, expire_at, _ = stored
                            if (reject is None or token != reject) and expire_at - now > _TOKEN_LIVE_MARGIN_SECONDS:
                                return token
                self._stamp_attempt(self._clock())
                fresh_token, expires_in = issue()
                if int(expires_in) < _TOKEN_LIVE_MARGIN_SECONDS:
                    raise ProviderTerminalError(
                        f"{_PROVIDER} token expiry too short for {_TOKEN_ENDPOINT}",
                        provider=_PROVIDER,
                        endpoint=_TOKEN_ENDPOINT,
                    )
                issued = self._clock()
                self._write(fresh_token, issued + int(expires_in) - _TOKEN_EXPIRY_MARGIN_SECONDS, issued)
                return fresh_token
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DbsecClient:
    """DB Securities quote/chart REST client (read-only TRs)."""

    def __init__(
        self,
        credentials: DbsecCredentials,
        *,
        policy: DbsecPolicy,
        token_store: DbsecTokenStore,
        transport: HttpTransport | None = None,
        scope_pacers: Mapping[str, HostPacer] | None = None,
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
        cache_dir = token_store.cache_dir
        if scope_pacers is not None:
            self._scope_pacers: Mapping[str, HostPacer] = scope_pacers
        else:
            self._scope_pacers = {
                _TICKER_SCOPE: HostPacer(
                    cache_dir / "dbsec_quote_master.state",
                    min_interval_seconds=float(policy.scope_min_interval_seconds["quote_master"]),
                ),
                _CHART_SCOPE: HostPacer(
                    cache_dir / "dbsec_chart.state",
                    min_interval_seconds=float(policy.scope_min_interval_seconds["chart"]),
                ),
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
            min_interval_seconds=float(policy.app_min_interval_seconds),
            retry=RetryPolicy(max_attempts=_MAX_HTTP_ATTEMPTS),
            quota=gate,
            timeout_seconds=_TIMEOUT_SECONDS,
            session=self._session,
            host_pacer=HostPacer(
                cache_dir / "dbsec_app.state",
                min_interval_seconds=float(policy.app_min_interval_seconds),
            ),
        )

    def _issue_token(self) -> tuple[str, int]:
        try:
            response = self._session.post(
                f"{_BASE_URL}/{_TOKEN_ENDPOINT}",
                data={
                    "grant_type": "client_credentials",
                    "appkey": self._credentials.app_key,
                    "appsecretkey": self._credentials.app_secret,
                    "scope": "oob",
                },
                timeout=_TOKEN_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise ProviderRetryableError(
                f"{_PROVIDER} token issuance failed for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderRetryableError(
                f"{_PROVIDER} token issuance returned invalid JSON for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc
        if not isinstance(data, Mapping):
            raise ProviderTerminalError(
                f"{_PROVIDER} token issuance returned non-object for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            )
        rsp_cd = str(data.get("rsp_cd", "") or "").strip()
        if rsp_cd == "IGW00201":
            raise ProviderRetryableError(
                f"{_PROVIDER} token issuance throttled for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            )
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise ProviderTerminalError(
                f"{_PROVIDER} token issuance failed for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            )
        expires_in = data.get("expires_in")
        try:
            if isinstance(expires_in, bool):
                raise TypeError
            return token, int(expires_in)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ProviderTerminalError(
                f"{_PROVIDER} token issuance returned no valid expires_in for {_TOKEN_ENDPOINT}",
                provider=_PROVIDER,
                endpoint=_TOKEN_ENDPOINT,
            ) from exc

    def _hooks(self, endpoint: str, parsed: list[dict[str, Any]]) -> Callable[[requests.Response], None]:
        def _hook(response: requests.Response) -> None:
            is_ok = response.status_code == 200
            try:
                data = response.json()
            except ValueError as exc:
                if not is_ok:
                    # Gateway error pages carry no rsp_cd; the transport status mapping decides retry.
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
        """Return the newest ``bar_count`` 1-minute J bars at or before the end of ``anchor`` (single page)."""
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
    """Build the DB client with host-shared token store and pacers; raises ConfigError when credentials are unset."""
    raw = read_secret(policy.token_cache_dir_env, default="")
    cache_dir = Path(raw) if raw.strip() else load_runtime_config().logs_root
    credentials = DbsecCredentials.from_env(policy)
    store = DbsecTokenStore(cache_dir, min_issue_interval_seconds=float(policy.token_min_interval_seconds))
    return DbsecClient(credentials, policy=policy, token_store=store, quota_store=quota_store, now=now)
