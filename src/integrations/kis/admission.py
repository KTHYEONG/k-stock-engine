"""Shared KIS pool admission pacing and token cache."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

HOST_ADMISSION_MARKER: Final = ".host-admission"
KIS_POOL_MAX_RATE: Final = 18.0
TOKEN_MIN_REMAINING: Final = timedelta(minutes=10)
TOKEN_LOCK_POLL_SECONDS: Final = 0.05
TOKEN_LOCK_TIMEOUT_SECONDS: Final = 240.0

_SEOUL = ZoneInfo("Asia/Seoul")
_STATE_READ_BYTES: Final = 64


class AdmissionClass(StrEnum):
    CRITICAL = "critical"
    STANDARD = "standard"
    BULK = "bulk"


class AdmissionDirNotSharedError(RuntimeError):
    """The admission directory lacks the host marker, so pacing would not be shared with other projects."""


class TokenStoreLockTimeout(TimeoutError):  # noqa: N818
    """The token lock was not obtained within the bound."""


def kis_key_id(app_key: str) -> str:
    """First 12 hex chars of sha256(app_key utf-8)."""
    return hashlib.sha256(app_key.encode("utf-8")).hexdigest()[:12]


def max_lead_for(admission_class: AdmissionClass) -> float | None:
    """Booking lead in seconds: None for critical, 1.0 for standard, 0.25 for bulk."""
    if admission_class is AdmissionClass.CRITICAL:
        return None
    if admission_class is AdmissionClass.STANDARD:
        return 1.0
    return 0.25


class HostPacedLimiter:
    """Host-wide token-bucket slot booking on a shared state file."""

    def __init__(
        self,
        state_path: Path,
        *,
        max_rate: float = KIS_POOL_MAX_RATE,
        max_lead_seconds: float | None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._state_path = Path(state_path)
        self._max_rate = float(max_rate)
        self._max_lead = None if max_lead_seconds is None else float(max_lead_seconds)
        self._clock = clock
        self._sleep = sleep

    def acquire(self) -> float:
        """Block until a host-wide slot is booked; returns the seconds waited."""
        interval = 1.0 / self._max_rate
        waited = 0.0
        while True:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._state_path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    raw = os.read(fd, _STATE_READ_BYTES)
                    try:
                        text = raw.decode("ascii").strip()
                        next_free = float(text) if text else 0.0
                        if not math.isfinite(next_free):
                            next_free = 0.0
                    except (UnicodeDecodeError, ValueError):
                        next_free = 0.0
                    now = self._clock()
                    slot = max(now, next_free)
                    wait = slot - now
                    if self._max_lead is None or wait <= self._max_lead:
                        os.lseek(fd, 0, os.SEEK_SET)
                        os.ftruncate(fd, 0)
                        os.write(fd, repr(slot + interval).encode("ascii"))
                        action: tuple[str, float] = ("book", wait)
                    else:
                        action = ("yield", max(wait - self._max_lead, interval))
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            kind, amount = action
            if kind == "book":
                if amount > 0:
                    self._sleep(amount)
                    waited += amount
                return waited
            self._sleep(amount)
            waited += amount


class SharedTokenStore:
    """Per-key KIS token cache shared across host processes."""

    def __init__(
        self,
        cache_dir: Path,
        app_key: str,
        *,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._dir = Path(cache_dir)
        self._app_key = app_key
        self._now = now if now is not None else (lambda: datetime.now(_SEOUL))
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def _token_path(self) -> Path:
        return self._dir / f"token_{kis_key_id(self._app_key)}.json"

    @property
    def _lock_path(self) -> Path:
        return Path(str(self._token_path) + ".lock")

    def _read(self) -> tuple[str, datetime] | None:
        try:
            payload = json.loads(self._token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("app_key") != self._app_key:
            return None
        token = payload.get("access_token")
        expired_raw = payload.get("expired_at")
        issued_raw = payload.get("issued_at")
        if not isinstance(token, str) or not token:
            return None
        if not isinstance(expired_raw, str) or not isinstance(issued_raw, str):
            return None
        try:
            expired_at = datetime.fromisoformat(expired_raw)
            issued_at = datetime.fromisoformat(issued_raw)
        except ValueError:
            return None
        if expired_at.tzinfo is None or issued_at.tzinfo is None:
            return None
        return token, expired_at

    def load_valid(self, *, min_remaining: timedelta = TOKEN_MIN_REMAINING) -> str | None:
        """Cached access token valid for at least min_remaining, else None."""
        cached = self._read()
        if cached is None or cached[1] - self._now() < min_remaining:
            return None
        return cached[0]

    def _write_token(self, access_token: str, expires_in: int) -> None:
        issued_at = self._now()
        expired_at = issued_at + timedelta(seconds=int(expires_in))
        data = {
            "access_token": access_token,
            "expired_at": expired_at.isoformat(timespec="seconds"),
            "app_key": self._app_key,
            "issued_at": issued_at.isoformat(timespec="seconds"),
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._dir / f".{self._token_path.name}.{os.getpid()}.tmp"
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(data, ensure_ascii=False))
            os.replace(tmp_path, self._token_path)
            os.chmod(self._token_path, 0o600)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    def _acquire_lock(self) -> int:
        try:
            fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        except PermissionError:
            fd = os.open(str(self._lock_path), os.O_RDONLY)
        deadline = self._clock() + TOKEN_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                if self._clock() >= deadline:
                    os.close(fd)
                    raise TokenStoreLockTimeout(
                        f"KIS token lock not obtained within {TOKEN_LOCK_TIMEOUT_SECONDS:g} s"
                    ) from None
                self._sleep(TOKEN_LOCK_POLL_SECONDS)

    def ensure(self, issue: Callable[[], tuple[str, int]], *, rejected: str | None = None) -> str:
        """Valid token, issuing through issue() under the host lock only when the cache has none; raises TokenStoreLockTimeout."""
        cached = self.load_valid()
        if cached is not None and cached != rejected:
            return cached
        self._dir.mkdir(parents=True, exist_ok=True)
        fd = self._acquire_lock()
        try:
            reread = self.load_valid()
            if reread is not None and reread != rejected:
                return reread
            access_token, expires_in = issue()
            self._write_token(access_token, int(expires_in))
            return access_token
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def limiter_for(
    cache_dir: Path, app_key: str, admission_class: AdmissionClass, *, require_marker: bool
) -> HostPacedLimiter:
    """Limiter on the shared state file of this key; raises AdmissionDirNotSharedError when the marker is required and missing."""
    base = Path(cache_dir)
    if require_marker and not (base / HOST_ADMISSION_MARKER).exists():
        raise AdmissionDirNotSharedError(f"shared admission marker missing in {base}")
    return HostPacedLimiter(
        base / f"tps_{kis_key_id(app_key)}.state",
        max_lead_seconds=max_lead_for(admission_class),
    )
