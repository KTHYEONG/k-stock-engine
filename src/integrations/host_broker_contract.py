# canonical: krx-alpha/deploy/shared/host_broker_contract.py contract=v1
"""Host-shared broker token and admission contract (canonical v1)."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

logger = logging.getLogger(__name__)

CONTRACT_VERSION: Final[int] = 1
VENDOR_IDS: Final[frozenset[str]] = frozenset({"ls", "toss", "dbsec", "kiwoom"})
HOST_ADMISSION_MARKER: Final[str] = ".host-admission"
DEFAULT_UNKNOWN_EXPIRY_MAX_AGE_S: Final[float] = 43200.0
MIN_ISSUE_INTERVAL_S: Final[Mapping[str, float]] = MappingProxyType({"dbsec": 61.0})
WILDCARD_SCOPE: Final[str] = "*"

_TOSS_RATES: Final[Mapping[str, float]] = MappingProxyType(
    {
        "AUTH": 5.0,
        "ACCOUNT": 1.0,
        "ASSET": 5.0,
        "STOCK": 5.0,
        "STOCK_ALL": 1.0,
        "STOCK_TRADING_TREND": 10.0,
        "MARKET_INFO": 3.0,
        "MARKET_DATA": 15.0,
        "MARKET_DATA_CHART": 20.0,
        "RANKING": 5.0,
        "MARKET_INDICATOR": 10.0,
        "MARKET_INDICATOR_CHART": 5.0,
        "ORDER": 10.0,
        "ORDER_HISTORY": 5.0,
        "ORDER_INFO": 6.0,
        "CONDITIONAL_ORDER": 5.0,
        "CONDITIONAL_ORDER_HISTORY": 10.0,
    }
)

_RATES: dict[tuple[str, str | None], float] = {
    ("dbsec", None): 20.0,
    ("dbsec", "chart"): 4.0,
    ("dbsec", "quote"): 3.0,
    ("ls", None): 1.0 / 1.05,
    ("kiwoom", WILDCARD_SCOPE): 5.0,
}
for _scope, _rate in _TOSS_RATES.items():
    _RATES[("toss", _scope)] = _rate
DEFAULT_RATES: Final[Mapping[tuple[str, str | None], float]] = MappingProxyType(dict(_RATES))

_SCOPE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
_SCHEMA_VERSION: Final[int] = 1
_POLL_INTERVAL_SECONDS: Final[float] = 0.01


class HostBrokerDirNotSharedError(RuntimeError):
    """The broker state directory lacks the host marker while enforcement is active."""


class TokenStoreLockTimeout(RuntimeError):  # noqa: N818 - spec-mandated protocol name
    """The token lock was not obtained within the configured bound."""


class TokenIssueThrottled(RuntimeError):  # noqa: N818 - spec-mandated protocol name
    """Issuance refused locally (min-issue interval) or by the vendor's issuance limit."""


@dataclass(frozen=True)
class IssuedToken:
    access_token: str = field(repr=False)
    expires_in_seconds: float | None


@dataclass(frozen=True)
class TokenRecord:
    access_token: str = field(repr=False)
    issued_at: datetime
    expires_at: datetime | None
    generation: int


IssueThrottlePolicy = Literal["wait", "raise"]


def validate_vendor(vendor: str) -> str:
    """Return vendor unchanged; raise ValueError unless it is a member of VENDOR_IDS."""
    if vendor not in VENDOR_IDS:
        raise ValueError(f"invalid vendor: {vendor!r}")
    return vendor


def credential_digest(app_key: str) -> str:
    """Return sha256(app_key utf-8)[:12]; raise ValueError on empty app_key."""
    if not app_key:
        raise ValueError("credential must be non-empty")
    return hashlib.sha256(app_key.encode("utf-8")).hexdigest()[:12]


def _validate_scope(scope: str) -> str:
    if scope == WILDCARD_SCOPE or not _SCOPE_RE.fullmatch(scope):
        raise ValueError(f"invalid scope: {scope!r}")
    return scope


def resolve_host_broker_dir(
    configured: Path,
    *,
    require_shared: Literal["auto", "always", "never"] = "auto",
    in_container: bool | None = None,
) -> Path:
    """Return the caller-injected broker dir; raise HostBrokerDirNotSharedError when enforced and the marker is absent."""
    directory = Path(configured)
    container = Path("/.dockerenv").exists() if in_container is None else bool(in_container)
    enforced = require_shared == "always" or (require_shared == "auto" and container)
    if enforced and not (directory / HOST_ADMISSION_MARKER).is_file():
        raise HostBrokerDirNotSharedError(f"broker dir is not host-shared: {directory}")
    return directory


def token_record_path(directory: Path, vendor: str, app_key: str) -> Path:
    """Return directory/token_<vendor>_<sha12>.json; raise ValueError on bad vendor or empty key."""
    validate_vendor(vendor)
    digest = credential_digest(app_key)
    return Path(directory) / f"token_{vendor}_{digest}.json"


def issue_guard_path(directory: Path, vendor: str, app_key: str) -> Path:
    """Return directory/token_issue_<vendor>_<sha12>.state; raise ValueError on bad vendor or empty key."""
    validate_vendor(vendor)
    digest = credential_digest(app_key)
    return Path(directory) / f"token_issue_{vendor}_{digest}.state"


def admission_state_path(directory: Path, vendor: str, app_key: str, scope: str | None = None) -> Path:
    """Return directory/admission_<vendor>_<sha12>[_<scope>].state; raise ValueError on bad vendor/scope or empty key."""
    validate_vendor(vendor)
    digest = credential_digest(app_key)
    if scope is None:
        return Path(directory) / f"admission_{vendor}_{digest}.state"
    _validate_scope(scope)
    return Path(directory) / f"admission_{vendor}_{digest}_{scope}.state"


def rate_for(vendor: str, scope: str | None) -> float:
    """Return the canonical max_rate for (vendor, scope); raise ValueError when undeclared."""
    key = (vendor, scope)
    if key in DEFAULT_RATES:
        return DEFAULT_RATES[key]
    if scope is not None and (vendor, WILDCARD_SCOPE) in DEFAULT_RATES:
        return DEFAULT_RATES[(vendor, WILDCARD_SCOPE)]
    raise ValueError(f"undeclared rate: {vendor}/{scope}")


def _parse_record(payload: object) -> TokenRecord | None:
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"schema_version", "access_token", "issued_at", "expires_at", "generation"}:
        return None
    try:
        if type(payload.get("schema_version")) is not int or payload.get("schema_version") != _SCHEMA_VERSION:
            return None
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None
        issued_raw = payload.get("issued_at")
        expires_raw = payload.get("expires_at")
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            return None
        if not isinstance(issued_raw, str):
            return None
        issued_at = datetime.fromisoformat(issued_raw)
        if issued_at.tzinfo is None:
            return None
        expires_at: datetime | None = None
        if expires_raw is not None:
            if not isinstance(expires_raw, str):
                return None
            expires_at = datetime.fromisoformat(expires_raw)
            if expires_at.tzinfo is None:
                return None
        return TokenRecord(
            access_token=access_token,
            issued_at=issued_at,
            expires_at=expires_at,
            generation=generation,
        )
    except (ValueError, TypeError):
        return None


def _lock_path_for(path: Path) -> Path:
    return Path(str(path) + ".lock")


def _read_record_file(path: Path) -> TokenRecord | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        logger.warning(
            "[SYS] stage=shared_token status=UNREADABLE path=%s reason=%s",
            path.name,
            type(exc).__name__,
        )
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("[SYS] stage=shared_token status=SCHEMA_INVALID path=%s", path.name)
        return None
    record = _parse_record(payload)
    if record is None:
        logger.warning("[SYS] stage=shared_token status=SCHEMA_INVALID path=%s", path.name)
        return None
    return record


def _write_record_file(path: Path, record: TokenRecord) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "access_token": record.access_token,
        "issued_at": record.issued_at.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at is not None else None,
        "generation": record.generation,
    }
    _atomic_write_text(path, json.dumps(payload))
    logger.info(
        "[SYS] stage=shared_token status=PUBLISHED path=%s generation=%d",
        path.name,
        record.generation,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _build_record(issued: IssuedToken, previous: TokenRecord | None, now: datetime) -> TokenRecord:
    expires_at: datetime | None = None
    if issued.expires_in_seconds is not None:
        expires_at = now + timedelta(seconds=float(issued.expires_in_seconds))
    generation = previous.generation + 1 if previous is not None else 1
    return TokenRecord(
        access_token=issued.access_token,
        issued_at=now,
        expires_at=expires_at,
        generation=generation,
    )


def _is_usable(record: TokenRecord, now: datetime, margin: float, max_age: float) -> bool:
    if record.expires_at is None:
        return (now - record.issued_at).total_seconds() < max_age
    return now < record.expires_at - timedelta(seconds=margin)


def _guard_path_for_token(token_path: Path, vendor: str) -> Path:
    name = Path(token_path).name
    prefix = f"token_{vendor}_"
    if name.startswith(prefix) and name.endswith(".json"):
        digest = name[len(prefix) : -len(".json")]
        if digest:
            return Path(token_path).parent / f"token_issue_{vendor}_{digest}.state"
    fallback = hashlib.sha256(str(token_path).encode("utf-8")).hexdigest()[:12]
    return Path(token_path).parent / f"token_issue_{vendor}_{fallback}.state"


def _read_guard_value(guard_path: Path) -> float | None:
    try:
        fd = os.open(str(guard_path), os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            raw = os.read(fd, 64).decode("ascii", errors="replace").strip()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
        if not raw:
            return None
        try:
            value = float(raw)
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    finally:
        os.close(fd)


def _write_guard_value(guard_path: Path, now_epoch: float) -> None:
    _atomic_write_text(guard_path, repr(float(now_epoch)))


def _validate_store_args(
    vendor: str,
    lock_timeout_seconds: float,
    expiry_margin_seconds: float,
    unknown_expiry_max_age_seconds: float,
    issue_throttle_policy: str,
) -> None:
    validate_vendor(vendor)
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
        raise ValueError("lock_timeout_seconds must be positive")
    if not math.isfinite(expiry_margin_seconds) or expiry_margin_seconds < 0:
        raise ValueError("expiry_margin_seconds must be non-negative")
    if not math.isfinite(unknown_expiry_max_age_seconds) or unknown_expiry_max_age_seconds < 0:
        raise ValueError("unknown_expiry_max_age_seconds must be non-negative")
    if issue_throttle_policy not in ("wait", "raise"):
        raise ValueError(f"invalid issue_throttle_policy: {issue_throttle_policy!r}")


def _throttle_delay_seconds(vendor: str, guard_path: Path | None, now_epoch: float) -> float:
    interval = MIN_ISSUE_INTERVAL_S.get(vendor, 0.0)
    if interval <= 0 or guard_path is None:
        return 0.0
    last = _read_guard_value(guard_path)
    if last is None:
        return 0.0
    wait = last + interval - now_epoch
    return wait if wait > 0 else 0.0


class AsyncTokenStore:
    """Host-shared single-issuer token cache (asyncio surface)."""

    def __init__(
        self,
        path: Path,
        *,
        vendor: str,
        lock_timeout_seconds: float,
        expiry_margin_seconds: float,
        clock: Callable[[], datetime],
        issue_throttle_policy: IssueThrottlePolicy,
        unknown_expiry_max_age_seconds: float = DEFAULT_UNKNOWN_EXPIRY_MAX_AGE_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Bind a C3 token record; raise ValueError on non-positive lock timeout, negative margin/max-age, bad vendor or policy."""
        _validate_store_args(
            vendor, lock_timeout_seconds, expiry_margin_seconds, unknown_expiry_max_age_seconds, issue_throttle_policy
        )
        self._path = Path(path)
        self._vendor = vendor
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._expiry_margin_seconds = float(expiry_margin_seconds)
        self._clock = clock
        self._issue_throttle_policy: IssueThrottlePolicy = issue_throttle_policy
        self._unknown_expiry_max_age_seconds = float(unknown_expiry_max_age_seconds)
        self._sleep = sleep
        self._guard_path = _guard_path_for_token(self._path, vendor)

    def _usable(self, record: TokenRecord, now: datetime) -> bool:
        return _is_usable(record, now, self._expiry_margin_seconds, self._unknown_expiry_max_age_seconds)

    def read(self) -> TokenRecord | None:
        """Return the stored record, or None when absent/unreadable/schema-invalid (logged, never raised)."""
        return _read_record_file(self._path)

    def _publish(self, record: TokenRecord) -> None:
        _write_record_file(self._path, record)

    def _build(self, issued: IssuedToken, previous: TokenRecord | None) -> TokenRecord:
        return _build_record(issued, previous, self._clock())

    def _build_record(self, issued: IssuedToken, previous: TokenRecord | None) -> TokenRecord:
        return self._build(issued, previous)

    async def _hold_lock(self) -> int:
        lock_path = _lock_path_for(self._path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self._lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TokenStoreLockTimeout(f"token lock not acquired: {lock_path}") from None
                    await self._sleep(_POLL_INTERVAL_SECONDS)
        except BaseException:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise

    async def _guard_before_issue(self) -> None:
        interval = MIN_ISSUE_INTERVAL_S.get(self._vendor, 0.0)
        if interval <= 0:
            return
        wait = _throttle_delay_seconds(self._vendor, self._guard_path, self._clock().timestamp())
        if wait > 0:
            if self._issue_throttle_policy == "raise":
                raise TokenIssueThrottled("token issuance throttled by host guard")
            if wait > self._lock_timeout_seconds:
                raise TokenIssueThrottled("token issuance throttled by host guard")
            await self._sleep(wait)
        _write_guard_value(self._guard_path, self._clock().timestamp())

    async def get_or_issue(self, issue: Callable[[], Awaitable[IssuedToken]]) -> TokenRecord:
        """Return a usable record, issuing at most once host-wide; raise TokenStoreLockTimeout, TokenIssueThrottled, or issue's error."""
        now = self._clock()
        cached = self.read()
        if cached is not None and self._usable(cached, now):
            return cached
        fd = await self._hold_lock()
        try:
            now = self._clock()
            cached = self.read()
            if cached is not None and self._usable(cached, now):
                return cached
            await self._guard_before_issue()
            issued = await issue()
            record = _build_record(issued, cached, self._clock())
            self._publish(record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def replace_rejected(self, rejected_token: str, issue: Callable[[], Awaitable[IssuedToken]]) -> TokenRecord:
        """Adopt a peer-rotated usable token or issue a new one; raise TokenStoreLockTimeout, TokenIssueThrottled, or issue's error."""
        fd = await self._hold_lock()
        try:
            cached = self.read()
            if cached is not None and cached.access_token != rejected_token and self._usable(cached, self._clock()):
                return cached
            await self._guard_before_issue()
            issued = await issue()
            record = _build_record(issued, cached, self._clock())
            self._publish(record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class TokenStore:
    """Host-shared single-issuer token cache (blocking surface)."""

    def __init__(
        self,
        path: Path,
        *,
        vendor: str,
        lock_timeout_seconds: float,
        expiry_margin_seconds: float,
        clock: Callable[[], datetime],
        issue_throttle_policy: IssueThrottlePolicy,
        unknown_expiry_max_age_seconds: float = DEFAULT_UNKNOWN_EXPIRY_MAX_AGE_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind a C3 token record; raise ValueError on non-positive lock timeout, negative margin/max-age, bad vendor or policy."""
        _validate_store_args(
            vendor, lock_timeout_seconds, expiry_margin_seconds, unknown_expiry_max_age_seconds, issue_throttle_policy
        )
        self._path = Path(path)
        self._vendor = vendor
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._expiry_margin_seconds = float(expiry_margin_seconds)
        self._clock = clock
        self._issue_throttle_policy: IssueThrottlePolicy = issue_throttle_policy
        self._unknown_expiry_max_age_seconds = float(unknown_expiry_max_age_seconds)
        self._sleep = sleep
        self._guard_path = _guard_path_for_token(self._path, vendor)

    def _usable(self, record: TokenRecord, now: datetime) -> bool:
        return _is_usable(record, now, self._expiry_margin_seconds, self._unknown_expiry_max_age_seconds)

    def read(self) -> TokenRecord | None:
        """Return the stored record, or None when absent/unreadable/schema-invalid (logged, never raised)."""
        return _read_record_file(self._path)

    def _hold_lock(self) -> int:
        lock_path = _lock_path_for(self._path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self._lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TokenStoreLockTimeout(f"token lock not acquired: {lock_path}") from None
                    self._sleep(_POLL_INTERVAL_SECONDS)
        except BaseException:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise

    def _guard_before_issue(self) -> None:
        interval = MIN_ISSUE_INTERVAL_S.get(self._vendor, 0.0)
        if interval <= 0:
            return
        wait = _throttle_delay_seconds(self._vendor, self._guard_path, self._clock().timestamp())
        if wait > 0:
            if self._issue_throttle_policy == "raise":
                raise TokenIssueThrottled("token issuance throttled by host guard")
            if wait > self._lock_timeout_seconds:
                raise TokenIssueThrottled("token issuance throttled by host guard")
            self._sleep(wait)
        _write_guard_value(self._guard_path, self._clock().timestamp())

    def get_or_issue(self, issue: Callable[[], IssuedToken]) -> TokenRecord:
        """Return a usable record, issuing at most once host-wide; raise TokenStoreLockTimeout, TokenIssueThrottled, or issue's error."""
        now = self._clock()
        cached = self.read()
        if cached is not None and self._usable(cached, now):
            return cached
        fd = self._hold_lock()
        try:
            now = self._clock()
            cached = self.read()
            if cached is not None and self._usable(cached, now):
                return cached
            self._guard_before_issue()
            issued = issue()
            record = _build_record(issued, cached, self._clock())
            _write_record_file(self._path, record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def replace_rejected(self, rejected_token: str, issue: Callable[[], IssuedToken]) -> TokenRecord:
        """Adopt a peer-rotated usable token or issue a new one; raise TokenStoreLockTimeout, TokenIssueThrottled, or issue's error."""
        fd = self._hold_lock()
        try:
            cached = self.read()
            if cached is not None and cached.access_token != rejected_token and self._usable(cached, self._clock()):
                return cached
            self._guard_before_issue()
            issued = issue()
            record = _build_record(issued, cached, self._clock())
            _write_record_file(self._path, record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read_next_free_locked(fd: int, state_path: Path) -> float:
    raw = os.read(fd, 64).decode("ascii", errors="replace").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite admission state")
        return value
    except ValueError:
        logger.warning(
            "[SYS] stage=host_admission status=STATE_RESET path=%s",
            state_path.name,
        )
        return 0.0


def _book_slot(
    state_path: Path, interval: float, clock: Callable[[], float], max_lead: float | None
) -> tuple[bool, float]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(state_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            next_free = _read_next_free_locked(fd, state_path)
            now = clock()
            slot = max(now, next_free)
            if max_lead is not None and slot - now > max_lead:
                return False, max(slot - now - max_lead, interval)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, repr(slot + interval).encode("ascii"))
            return True, slot - now
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class AsyncHostPacer:
    """Host-wide file-backed pacing bucket (asyncio surface)."""

    def __init__(
        self,
        state_path: Path,
        max_rate: float,
        time_period: float = 1.0,
        *,
        max_lead_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Bind a C4 pacing file; raise ValueError on non-positive rate/period/lead."""
        if not math.isfinite(max_rate) or not math.isfinite(time_period) or max_rate <= 0 or time_period <= 0:
            raise ValueError("max_rate and time_period must be positive")
        if max_lead_seconds is not None and (not math.isfinite(max_lead_seconds) or max_lead_seconds <= 0):
            raise ValueError("max_lead_seconds must be positive")
        self.max_rate = float(max_rate)
        self.time_period = float(time_period)
        self._state_path = Path(state_path)
        self._max_rate = float(max_rate)
        self._time_period = float(time_period)
        self._interval = self._time_period / self._max_rate
        self._max_lead_seconds = max_lead_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _loop_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _read_next_free_locked(self, fd: int) -> float:
        return _read_next_free_locked(fd, self._state_path)

    def _book_unconditional(self) -> float:
        booked, wait = _book_slot(self._state_path, self._interval, self._clock, None)
        assert booked
        return wait

    def _try_book_or_wait(self) -> tuple[bool, float]:
        assert self._max_lead_seconds is not None
        return _book_slot(self._state_path, self._interval, self._clock, self._max_lead_seconds)

    def _reserve_slot(self) -> float:
        return self._book_unconditional()

    async def _acquire_bounded(self) -> None:
        assert self._max_lead_seconds is not None
        async with self._loop_lock():
            while True:
                booked, wait = self._try_book_or_wait()
                if booked:
                    if wait > 0:
                        await self._sleep(wait)
                    return
                await self._sleep(wait)

    async def acquire(self) -> None:
        """Block until this caller owns the next host-wide slot of the state file."""
        if self._max_lead_seconds is not None:
            await self._acquire_bounded()
            return
        async with self._loop_lock():
            delay = self._book_unconditional()
        if delay > 0:
            await self._sleep(delay)


class HostPacer:
    """Host-wide file-backed pacing bucket (blocking surface)."""

    def __init__(
        self,
        state_path: Path,
        max_rate: float,
        time_period: float = 1.0,
        *,
        max_lead_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind a C4 pacing file; raise ValueError on non-positive rate/period/lead."""
        if not math.isfinite(max_rate) or not math.isfinite(time_period) or max_rate <= 0 or time_period <= 0:
            raise ValueError("max_rate and time_period must be positive")
        if max_lead_seconds is not None and (not math.isfinite(max_lead_seconds) or max_lead_seconds <= 0):
            raise ValueError("max_lead_seconds must be positive")
        self.max_rate = float(max_rate)
        self.time_period = float(time_period)
        self._state_path = Path(state_path)
        self._max_rate = float(max_rate)
        self._time_period = float(time_period)
        self._interval = self._time_period / self._max_rate
        self._max_lead_seconds = max_lead_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def _book_unconditional(self) -> float:
        booked, wait = _book_slot(self._state_path, self._interval, self._clock, None)
        assert booked
        return wait

    def _try_book_or_wait(self) -> tuple[bool, float]:
        assert self._max_lead_seconds is not None
        return _book_slot(self._state_path, self._interval, self._clock, self._max_lead_seconds)

    def acquire(self) -> None:
        """Block until this caller owns the next host-wide slot of the state file."""
        if self._max_lead_seconds is not None:
            with self._lock:
                while True:
                    booked, wait = self._try_book_or_wait()
                    if booked:
                        if wait > 0:
                            self._sleep(wait)
                        return
                    self._sleep(wait)
        with self._lock:
            delay = self._book_unconditional()
        if delay > 0:
            self._sleep(delay)


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_RATES",
    "DEFAULT_UNKNOWN_EXPIRY_MAX_AGE_S",
    "HOST_ADMISSION_MARKER",
    "MIN_ISSUE_INTERVAL_S",
    "VENDOR_IDS",
    "WILDCARD_SCOPE",
    "AsyncHostPacer",
    "AsyncTokenStore",
    "HostBrokerDirNotSharedError",
    "HostPacer",
    "IssueThrottlePolicy",
    "IssuedToken",
    "TokenIssueThrottled",
    "TokenRecord",
    "TokenStore",
    "TokenStoreLockTimeout",
    "admission_state_path",
    "credential_digest",
    "issue_guard_path",
    "rate_for",
    "resolve_host_broker_dir",
    "token_record_path",
    "validate_vendor",
]
