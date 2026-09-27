"""Durable provider quota ledger with atomic JSON state."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.integrations.errors import ProviderQuotaExhaustedError

__all__ = [
    "LedgerQuotaGate",
    "ProviderQuotaExhaustedError",
    "ProviderQuotaState",
    "ProviderQuotaStateStore",
]


@dataclass(frozen=True, slots=True)
class ProviderQuotaState:
    provider: str
    endpoint: str
    blocked_until: datetime | None
    attempted_requests: int
    rate_limited_requests: int


def _next_kst_midnight_utc(now: datetime) -> datetime:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    kst = moment + timedelta(hours=9)
    nxt = (kst + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - timedelta(hours=9)).astimezone(UTC)


class ProviderQuotaStateStore:
    def __init__(self, root: Path | str, *, daily_limit: int | None = None) -> None:
        self._root = Path(root)
        if daily_limit is not None and (isinstance(daily_limit, bool) or int(daily_limit) < 1):
            raise ValueError("daily_limit must be a positive integer")
        self._daily_limit = int(daily_limit) if daily_limit is not None else None
        self._lock = threading.Lock()

    def _path(self) -> Path:
        return self._root / "quota_state.json"

    def _load_nolock(self) -> dict[str, dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # reason: adapter boundary — a torn ledger read must surface as empty rather than crash the gate.
            return {}
        return dict(raw) if isinstance(raw, dict) else {}

    def _load(self) -> dict[str, dict[str, Any]]:
        return self._load_nolock()

    def _save_nolock(self, state: dict[str, dict[str, Any]]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._root / f".quota_state.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(state, sort_keys=True, ensure_ascii=False))
            os.replace(tmp, self._path())
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _save(self, state: dict[str, dict[str, Any]]) -> None:
        self._save_nolock(state)

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold an inter-process exclusive lock across one read-modify-write."""
        self._root.mkdir(parents=True, exist_ok=True)
        lock_path = self._root / ".quota_state.lock"
        with self._lock, open(lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _key(self, *, provider: str, endpoint: str) -> str:
        return f"{provider}|{endpoint}"

    @staticmethod
    def _kst_day(now: datetime) -> str:
        moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        return (moment + timedelta(hours=9)).date().isoformat()

    def _daily_attempts(self, state: dict[str, dict[str, Any]], *, provider: str, day: str) -> int:
        return sum(
            int(entry.get("daily_attempted_requests", 0))
            for key, entry in state.items()
            if key.startswith(f"{provider}|") and entry.get("daily_attempt_day") == day
        )

    def remaining_daily_attempts(self, *, provider: str, now: datetime, daily_limit: int) -> int:
        """Return provider-wide KST-day request headroom under one local limit."""
        if daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        with self._exclusive():
            state = self._load_nolock()
            used = self._daily_attempts(state, provider=provider, day=self._kst_day(now))
            return max(0, int(daily_limit) - used)

    def _check_blocked(self, state: dict[str, dict[str, Any]], *, provider: str, endpoint: str, now: datetime) -> None:
        entry = state.get(self._key(provider=provider, endpoint=endpoint), {})
        raw_blocked = entry.get("blocked_until")
        blocked_until = datetime.fromisoformat(str(raw_blocked)) if raw_blocked else None
        if blocked_until is not None and now < blocked_until:
            raise ProviderQuotaExhaustedError(f"{provider} {endpoint} blocked until {blocked_until.isoformat()}")

    def acquire(self, *, provider: str, endpoint: str, now: datetime, daily_limit: int | None = None) -> None:
        with self._exclusive():
            state = self._load_nolock()
            self._check_blocked(state, provider=provider, endpoint=endpoint, now=now)
            limit = int(daily_limit) if daily_limit is not None else self._daily_limit
            day = self._kst_day(now)
            if limit is not None and self._daily_attempts(state, provider=provider, day=day) >= limit:
                raise ProviderQuotaExhaustedError(f"{provider} daily quota safety limit reached ({limit})")

    def acquire_and_record(
        self, *, provider: str, endpoint: str, now: datetime, daily_limit: int | None = None
    ) -> None:
        """Check the gate and record one attempt atomically (one ledger increment)."""
        with self._exclusive():
            state = self._load_nolock()
            self._check_blocked(state, provider=provider, endpoint=endpoint, now=now)
            limit = int(daily_limit) if daily_limit is not None else self._daily_limit
            day = self._kst_day(now)
            if limit is not None and self._daily_attempts(state, provider=provider, day=day) >= limit:
                raise ProviderQuotaExhaustedError(f"{provider} daily quota safety limit reached ({limit})")
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
            previous_day = entry.get("daily_attempt_day")
            entry["daily_attempt_day"] = day
            entry["daily_attempted_requests"] = (
                int(entry.get("daily_attempted_requests", 0)) + 1 if previous_day == day else 1
            )
            entry["updated_at"] = now.isoformat()
            state[key] = entry
            self._save_nolock(state)

    def record_attempt(self, *, provider: str, endpoint: str, now: datetime, daily_limit: int | None = None) -> None:
        with self._exclusive():
            state = self._load_nolock()
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            limit = int(daily_limit) if daily_limit is not None else self._daily_limit
            day = self._kst_day(now)
            if limit is not None and self._daily_attempts(state, provider=provider, day=day) >= limit:
                raise ProviderQuotaExhaustedError(f"{provider} daily quota safety limit reached ({limit})")
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
            previous_day = entry.get("daily_attempt_day")
            entry["daily_attempt_day"] = day
            entry["daily_attempted_requests"] = (
                int(entry.get("daily_attempted_requests", 0)) + 1 if previous_day == day else 1
            )
            entry["updated_at"] = now.isoformat()
            state[key] = entry
            self._save_nolock(state)

    def add_attempts(self, *, provider: str, endpoint: str, day: str, count: int) -> None:
        """Fold ``count`` requests made elsewhere into one KST day."""
        if count < 0:
            raise ValueError("count must not be negative")
        datetime.fromisoformat(day)
        if count == 0:
            return
        with self._exclusive():
            state = self._load_nolock()
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + count
            same_day = entry.get("daily_attempt_day") == day
            entry["daily_attempt_day"] = day
            entry["daily_attempted_requests"] = (int(entry.get("daily_attempted_requests", 0)) if same_day else 0) + count
            state[key] = entry
            self._save_nolock(state)

    def record_rate_limit(self, *, provider: str, endpoint: str, now: datetime, retry_after: float | None) -> ProviderQuotaState:
        blocked_until = (now + timedelta(seconds=float(retry_after))) if retry_after is not None else _next_kst_midnight_utc(now)
        with self._exclusive():
            state = self._load_nolock()
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
            entry["rate_limited_requests"] = int(entry.get("rate_limited_requests", 0)) + 1
            entry["blocked_until"] = blocked_until.isoformat()
            entry["updated_at"] = now.isoformat()
            state[key] = entry
            self._save_nolock(state)
            return ProviderQuotaState(provider=provider, endpoint=endpoint, blocked_until=blocked_until, attempted_requests=int(entry["attempted_requests"]), rate_limited_requests=int(entry["rate_limited_requests"]))


class LedgerQuotaGate:
    """``QuotaGate`` adapter that meters one provider in a shared ledger."""

    def __init__(self, store: ProviderQuotaStateStore, *, provider: str, daily_limit: int | None) -> None:
        self._store = store
        self._provider = provider
        self._daily_limit = daily_limit
        self._now: Any = None

    def bind_now(self, now: Any) -> None:
        """Bind the clock used for KST-day accounting (test seam)."""
        self._now = now

    def _current_time(self) -> datetime:
        if self._now is not None:
            value = self._now() if callable(self._now) else self._now
            assert isinstance(value, datetime)
            return value
        return datetime.now(UTC)

    def acquire(self, *, endpoint: str) -> None:
        self._store.acquire_and_record(
            provider=self._provider, endpoint=endpoint, now=self._current_time(), daily_limit=self._daily_limit
        )

    def record_rate_limit(self, *, endpoint: str, retry_after: float | None) -> None:
        self._store.record_rate_limit(
            provider=self._provider, endpoint=endpoint, now=self._current_time(), retry_after=retry_after
        )
