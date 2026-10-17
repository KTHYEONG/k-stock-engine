"""Host broker contract adapter (kse-local)."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.config.secrets import read_secret
from src.integrations.host_broker_contract import (
    DEFAULT_RATES,
    MIN_ISSUE_INTERVAL_S,
    VENDOR_IDS,
    HostBrokerDirNotSharedError,
    IssuedToken,
    TokenRecord,
    TokenStore,
    TokenStoreLockTimeout,
    admission_state_path,
    resolve_host_broker_dir,
    token_record_path,
)
from src.integrations.host_broker_contract import (
    HostPacer as ContractHostPacer,
)

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
    ) -> None:
        self._state = admission_state_path(Path(broker_dir), vendor, app_key, scope)
        self._pacer = ContractHostPacer(self._state, contract_rate(vendor, scope), clock=clock, sleep=sleep)

    @property
    def state_path(self) -> Path:
        return self._state

    def wait_turn(self) -> None:
        """Block until this process owns the next host-wide slot."""
        self._pacer.acquire()


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
