"""Paced, ledgered, retrying HTTP for one provider credential."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol

import requests

from src.integrations.errors import (
    ProviderQuotaExhaustedError,
    ProviderRetryableError,
    ProviderTerminalError,
)

_DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

ClassifyHook = Callable[[requests.Response], None]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff policy."""

    max_attempts: int
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 5.0
    retry_statuses: frozenset[int] = field(default_factory=lambda: _DEFAULT_RETRY_STATUSES)

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or int(self.max_attempts) < 1:
            raise ValueError("max_attempts must be >= 1")


class QuotaGate(Protocol):
    """Pre-flight quota check consulted before every attempt."""

    def acquire(self, *, endpoint: str) -> None:
        """Record one attempt or raise ``ProviderQuotaExhaustedError`` without sending."""
        ...

    def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
        """Note a 429/quota signal so the ledger blocks the endpoint."""
        ...


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Parse a ``Retry-After`` header value into seconds."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    base = time.time() if now is None else now
    timestamp: float = moment.timestamp()
    base_value: float = float(base)
    delta: float = timestamp - base_value
    if delta < 0.0:
        return 0.0
    return delta


class HttpTransport:
    """Paced, ledgered, retrying HTTP for one provider credential.

    Every attempt (including retries) is paced by ``min_interval_seconds``
    across threads, recorded in the quota ledger before it is sent, and
    classified into the ``ProviderError`` tree. ``Retry-After`` (seconds or
    HTTP date) wins over the computed backoff when larger. Response bodies
    are never logged.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        min_interval_seconds: float,
        retry: RetryPolicy,
        quota: QuotaGate | None,
        timeout_seconds: float,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._min_interval = float(min_interval_seconds)
        self._retry = retry
        self._quota = quota
        self._timeout = float(timeout_seconds)
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._monotonic = monotonic
        self._pace_lock = threading.Lock()
        self._last_attempt: float | None = None

    @property
    def provider(self) -> str:
        return self._provider

    def _pace(self) -> None:
        if self._min_interval <= 0:
            with self._pace_lock:
                self._last_attempt = self._monotonic()
            return
        with self._pace_lock:
            now = self._monotonic()
            if self._last_attempt is not None:
                wait = self._min_interval - (now - self._last_attempt)
                if wait > 0:
                    self._sleep(wait)
            self._last_attempt = self._monotonic()

    def _backoff(self, attempt: int) -> float:
        result: float = min(self._retry.base_backoff_seconds * (2**attempt), self._retry.max_backoff_seconds)
        return result

    def _sleep_before_retry(self, *, attempt: int, retry_after: float | None) -> None:
        delay = self._backoff(attempt)
        if retry_after is not None and retry_after > delay:
            delay = retry_after
        if delay > 0:
            self._sleep(delay)

    def _send(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
        form_body: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        classify: ClassifyHook | None = None,
    ) -> requests.Response:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        last_error: ProviderRetryableError | None = None
        for attempt in range(self._retry.max_attempts):
            if self._quota is not None:
                self._quota.acquire(endpoint=endpoint)
            self._pace()
            try:
                if method == "GET":
                    response = self._session.get(
                        url, params=dict(params or {}), headers=dict(headers or {}), timeout=self._timeout
                    )
                elif form_body is not None:
                    response = self._session.post(
                        url,
                        params=dict(params or {}),
                        data=dict(form_body),
                        headers=dict(headers or {}),
                        timeout=self._timeout,
                    )
                else:
                    response = self._session.post(
                        url,
                        params=dict(params or {}),
                        json=dict(json_body) if json_body is not None else None,  # type: ignore[arg-type]
                        headers=dict(headers or {}),
                        timeout=self._timeout,
                    )
            except requests.exceptions.RequestException as exc:
                # reason: adapter boundary — requests transport failure has no status to classify.
                last_error = ProviderRetryableError(
                    f"{self._provider} transport failed for {endpoint}: {exc}",
                    provider=self._provider,
                    endpoint=endpoint,
                )
                if attempt + 1 >= self._retry.max_attempts:
                    raise last_error from exc
                self._sleep_before_retry(attempt=attempt, retry_after=None)
                continue
            if response.status_code == 200:
                if classify is not None:
                    try:
                        classify(response)
                    except ProviderRetryableError:
                        if attempt + 1 >= self._retry.max_attempts:
                            raise
                        self._sleep_before_retry(attempt=attempt, retry_after=None)
                        continue
                    except ProviderQuotaExhaustedError:
                        if self._quota is not None:
                            with contextlib.suppress(ProviderQuotaExhaustedError):
                                self._quota.record_rate_limit(endpoint=endpoint, retry_after=None)
                        raise
                    except ProviderTerminalError:
                        raise
                return response
            retry_after = parse_retry_after((getattr(response, "headers", None) or {}).get("Retry-After"))
            if response.status_code in self._retry.retry_statuses:
                if response.status_code == 429 and self._quota is not None:
                    with contextlib.suppress(ProviderQuotaExhaustedError):
                        self._quota.record_rate_limit(endpoint=endpoint, retry_after=retry_after)
                last_error = ProviderRetryableError(
                    f"{self._provider} HTTP {response.status_code} for {endpoint}",
                    provider=self._provider,
                    endpoint=endpoint,
                )
                if attempt + 1 >= self._retry.max_attempts:
                    raise last_error
                self._sleep_before_retry(attempt=attempt, retry_after=retry_after)
                continue
            if 400 <= response.status_code < 500:
                raise ProviderTerminalError(
                    f"{self._provider} HTTP {response.status_code} for {endpoint}",
                    provider=self._provider,
                    endpoint=endpoint,
                )
            raise ProviderTerminalError(
                f"{self._provider} HTTP {response.status_code} for {endpoint}",
                provider=self._provider,
                endpoint=endpoint,
            )
        assert last_error is not None  # pragma: no cover - loop always returns or raises
        raise last_error  # pragma: no cover - loop always returns or raises

    def get(
        self,
        endpoint: str,
        params: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        classify: ClassifyHook | None = None,
    ) -> requests.Response:
        """Send one ledgered GET with bounded retries."""
        return self._send("GET", endpoint, params=params, headers=headers, classify=classify)

    def post(
        self,
        endpoint: str,
        json_body: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
        classify: ClassifyHook | None = None,
    ) -> requests.Response:
        """Send one ledgered POST with bounded retries."""
        return self._send("POST", endpoint, json_body=json_body, headers=headers, classify=classify)

    def post_form(
        self,
        endpoint: str,
        form: Mapping[str, str],
        *,
        headers: Mapping[str, str] | None = None,
        classify: ClassifyHook | None = None,
    ) -> requests.Response:
        """Send one ledgered form-encoded POST with bounded retries."""
        return self._send("POST", endpoint, form_body=form, headers=headers, classify=classify)


class TokenCache:
    """File-backed OAuth token cache (mode 0600, atomic replace)."""

    def __init__(self, cache_dir: Path | str, *, provider: str, env: str) -> None:
        self._dir = Path(cache_dir)
        self._path = self._dir / f"{provider}_token_{env}.json"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> tuple[str, str] | None:
        """Return ``(token, expire_at_iso)`` or ``None`` when absent/invalid."""
        try:
            if not self._path.exists():
                return None
            if self._path.stat().st_mode & 0o077:
                return None
            import json

            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # reason: adapter boundary — a corrupt cache file must read as a miss, never crash collection.
            return None
        if not isinstance(payload, dict):
            return None
        token = payload.get("access_token")
        expire_at = payload.get("expire_at")
        if not isinstance(token, str) or not token or not isinstance(expire_at, str) or not expire_at:
            return None
        return token, expire_at

    def save(self, token: str, expire_at_iso: str) -> None:
        """Persist one token atomically so readers never see a partial file."""
        import json

        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_name = f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp_path = self._dir / tmp_name
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"access_token": token, "expire_at": expire_at_iso}))
                os.replace(tmp_path, self._path)
                self._path.chmod(0o600)
            finally:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
