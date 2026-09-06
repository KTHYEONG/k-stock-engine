"""Durable provider quota ledger with atomic JSON state."""
from __future__ import annotations

import json
import os
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
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

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

    def acquire(self, *, provider: str, endpoint: str, now: datetime) -> None:
        state = self._load()
        entry = state.get(self._key(provider=provider, endpoint=endpoint), {})
        raw_blocked = entry.get("blocked_until")
        blocked_until = datetime.fromisoformat(str(raw_blocked)) if raw_blocked else None
        if blocked_until is not None and now < blocked_until:
            raise ProviderQuotaBlocked(f"{provider} {endpoint} blocked until {blocked_until.isoformat()}")

    def record_attempt(self, *, provider: str, endpoint: str, now: datetime) -> None:
        state = self._load()
        key = self._key(provider=provider, endpoint=endpoint)
        entry = state.get(key, {})
        entry["attempted_requests"] = int(entry.get("attempted_requests", 0)) + 1
        entry["updated_at"] = now.isoformat()
        state[key] = entry
        self._save(state)

    def record_rate_limit(self, *, provider: str, endpoint: str, now: datetime, retry_after: float | None) -> ProviderQuotaState:
        blocked_until = (now + timedelta(seconds=float(retry_after))) if retry_after is not None else _next_kst_midnight_utc(now)
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
