"""Host broker contract adapter (kse-local)."""
from __future__ import annotations

import fcntl
import logging
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.config.secrets import read_secret
from src.integrations.errors import ProviderRetryableError
from src.integrations.host_broker_contract import (
    DEFAULT_RATES,
    MIN_ISSUE_INTERVAL_S,
    VENDOR_IDS,
    HostBrokerDirNotSharedError,
    IssuedToken,
    TokenRecord,
    TokenStoreLockTimeout,
    admission_state_path,
    resolve_host_broker_dir,
    token_record_path,
)
from src.integrations.host_broker_contract import TokenStore as ContractTokenStore

__all__ = [
    "AdmissionSlot",
    "HostBrokerDirNotSharedError",
    "IssuedToken",
    "TokenRecord",
    "TokenStore",
    "TokenStoreLockTimeout",
    "contract_rate",
    "open_token_store",
    "resolve_broker_dir",
]

_LOG = logging.getLogger(__name__)


class TokenStore(ContractTokenStore):
    """Contract token store exposing its shared directory to client adapters."""

    @property
    def broker_dir(self) -> Path:
        return self._path.parent


def resolve_broker_dir(dir_env: str) -> Path:
    """Return the host-shared broker dir for env ``dir_env``."""
    raw = read_secret(dir_env, default="")
    if not raw.strip():
        return resolve_host_broker_dir(None)
    return resolve_host_broker_dir(Path(raw.strip()))


def contract_rate(vendor: str, scope: str | None = None) -> float:
    """Return the canonical admission rate for (vendor, scope)."""
    try:
        return DEFAULT_RATES[(vendor, scope)]
    except KeyError as exc:
        raise ValueError(f"undeclared broker rate for vendor={vendor!r} scope={scope!r}") from exc


class AdmissionSlot:
    """Contract C4 admission slot exposing the transport ``wait_turn`` hook."""

    def __init__(
        self,
        broker_dir: Path,
        vendor: str,
        app_key: str,
        scope: str | None = None,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self._state = admission_state_path(Path(broker_dir), vendor, app_key, scope)
        if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        self._interval = 1.0 / contract_rate(vendor, scope)
        self._vendor = vendor
        self._clock = clock
        self._sleep = sleep
        self._lock_timeout = lock_timeout_seconds

    @property
    def state_path(self) -> Path:
        return self._state

    def wait_turn(self) -> None:
        """Block until this process owns the next host-wide slot."""
        self._state.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._state, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            deadline = time.monotonic() + self._lock_timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ProviderRetryableError(
                            "Host admission lock timed out",
                            provider=self._vendor.upper(),
                            endpoint="host_admission",
                        ) from None
                    self._sleep(0.01)
            raw = os.read(fd, 64).decode("ascii", errors="replace").strip()
            try:
                next_free = float(raw)
                if not math.isfinite(next_free):
                    raise ValueError("non-finite admission state")
            except ValueError:
                next_free = 0.0
                if raw:
                    _LOG.warning("[SYS] stage=host_admission status=STATE_RESET path=%s", self._state.name)
            now = self._clock()
            slot = max(now, next_free)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, repr(slot + self._interval).encode("ascii"))
        finally:
            os.close(fd)
        if slot > now:
            self._sleep(slot - now)


def open_token_store(
    broker_dir: Path,
    vendor: str,
    app_key: str,
    *,
    lock_timeout_seconds: float,
    expiry_margin_seconds: float,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> TokenStore:
    """Return the contract C3 token store for (vendor, app_key)."""
    if vendor not in VENDOR_IDS:
        raise ValueError(f"invalid vendor {vendor!r}")
    bound = MIN_ISSUE_INTERVAL_S.get(vendor, 0.0)
    if not float(lock_timeout_seconds) > float(bound):
        raise ValueError(f"lock_timeout_seconds must exceed {bound:g} s for vendor={vendor!r}")
    resolved_clock = clock if clock is not None else (lambda: datetime.now(UTC))
    path = token_record_path(Path(broker_dir), vendor, app_key)
    _LOG.debug("[SYS] token store path=%s", path.name)
    return TokenStore(
        path,
        vendor=vendor,
        lock_timeout_seconds=float(lock_timeout_seconds),
        expiry_margin_seconds=float(expiry_margin_seconds),
        clock=resolved_clock,
        sleep=sleep,
        issue_throttle_policy="wait",
    )
