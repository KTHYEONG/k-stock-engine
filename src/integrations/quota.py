"""Durable provider quota ledger with atomic JSON state."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class ProviderQuotaBlocked(RuntimeError):  # noqa: N818
    """Provider endpoint is blocked until its quota window resets."""


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
        # 동일 프로세스 내 여러 스레드(예: DartXbrlCollector의 ThreadPoolExecutor)가
        # 하나의 DartApiClient를 공유해 동시에 상태를 읽고-쓰면, 고정된 임시 파일명 위에서
        # os.replace가 서로의 임시 파일을 소비해 FileNotFoundError로 경합한다.
        # read-modify-write 구간 전체를 직렬화해 경합과 갱신 유실을 동시에 막는다.
        self._lock = threading.Lock()

    def _path(self) -> Path:
        return self._root / "quota_state.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else {}

    def _save(self, state: dict[str, dict[str, Any]]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._path().with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path())

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
        with self._lock:
            state = self._load()
            used = self._daily_attempts(state, provider=provider, day=self._kst_day(now))
            return max(0, int(daily_limit) - used)

    def acquire(self, *, provider: str, endpoint: str, now: datetime, daily_limit: int | None = None) -> None:
        with self._lock:
            state = self._load()
            entry = state.get(self._key(provider=provider, endpoint=endpoint), {})
            raw_blocked = entry.get("blocked_until")
            blocked_until = datetime.fromisoformat(str(raw_blocked)) if raw_blocked else None
            if blocked_until is not None and now < blocked_until:
                raise ProviderQuotaBlocked(f"{provider} {endpoint} blocked until {blocked_until.isoformat()}")
            limit = int(daily_limit) if daily_limit is not None else self._daily_limit
            day = self._kst_day(now)
            if limit is not None and self._daily_attempts(state, provider=provider, day=day) >= limit:
                raise ProviderQuotaBlocked(f"{provider} daily quota safety limit reached ({limit})")

    def record_attempt(self, *, provider: str, endpoint: str, now: datetime, daily_limit: int | None = None) -> None:
        with self._lock:
            state = self._load()
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            limit = int(daily_limit) if daily_limit is not None else self._daily_limit
            day = self._kst_day(now)
            if limit is not None and self._daily_attempts(state, provider=provider, day=day) >= limit:
                raise ProviderQuotaBlocked(f"{provider} daily quota safety limit reached ({limit})")
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
            previous_day = entry.get("daily_attempt_day")
            entry["daily_attempt_day"] = day
            entry["daily_attempted_requests"] = (
                int(entry.get("daily_attempted_requests", 0)) + 1
                if previous_day == day
                else 1
            )
            entry["updated_at"] = now.isoformat()
            state[key] = entry
            self._save(state)

    def record_rate_limit(self, *, provider: str, endpoint: str, now: datetime, retry_after: float | None) -> ProviderQuotaState:
        blocked_until = (now + timedelta(seconds=float(retry_after))) if retry_after is not None else _next_kst_midnight_utc(now)
        with self._lock:
            state = self._load()
            key = self._key(provider=provider, endpoint=endpoint)
            entry = state.get(key, {})
            entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
            entry["rate_limited_requests"] = int(entry.get("rate_limited_requests", 0)) + 1
            entry["blocked_until"] = blocked_until.isoformat()
            entry["updated_at"] = now.isoformat()
            state[key] = entry
            self._save(state)
            return ProviderQuotaState(provider=provider, endpoint=endpoint, blocked_until=blocked_until, attempted_requests=int(entry["attempted_requests"]), rate_limited_requests=int(entry["rate_limited_requests"]))
