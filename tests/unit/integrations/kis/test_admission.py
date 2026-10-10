"""Shared-pool admission invariants: byte-identical host pacing and token protocol."""
from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.integrations.kis.admission import (
    AdmissionClass,
    AdmissionDirNotSharedError,
    HostPacedLimiter,
    SharedTokenStore,
    TokenStoreLockTimeout,
    kis_key_id,
    limiter_for,
    max_lead_for,
)

SEOUL = ZoneInfo("Asia/Seoul")
INTERVAL = 1.0 / 18.0


def _critical(path: Path, now: float, sleeps: list[float]) -> HostPacedLimiter:
    state = {"t": now}
    return HostPacedLimiter(
        path,
        max_lead_seconds=None,
        clock=lambda: state["t"],
        sleep=sleeps.append,
    )


def test_state_file_bytes_match_shared_protocol(tmp_path: Path) -> None:
    state_path = tmp_path / "tps_abc.state"
    sleeps: list[float] = []
    limiter = _critical(state_path, 1000.0, sleeps)
    w1 = limiter.acquire()
    w2 = limiter.acquire()
    assert (w1, w2) == (0.0, pytest.approx(INTERVAL))
    assert state_path.read_text(encoding="utf-8") == repr(1000.0 + 2 * INTERVAL)
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600


def test_bulk_yields_to_booked_ahead_queue(tmp_path: Path) -> None:
    state_path = tmp_path / "tps_abc.state"
    state_path.write_text(repr(1002.0), encoding="utf-8")
    state_path.chmod(0o600)
    clock = {"t": 1000.0}
    sleeps: list[float] = []
    limiter = HostPacedLimiter(
        state_path,
        max_lead_seconds=0.25,
        clock=lambda: clock["t"],
        sleep=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)),
    )
    limiter.acquire()
    assert sleeps[0] == pytest.approx(1.75)
    assert sum(sleeps) == pytest.approx(2.0)
    assert state_path.read_text(encoding="utf-8") == repr(1002.0 + INTERVAL)


def test_unparsable_state_is_treated_as_free(tmp_path: Path) -> None:
    state_path = tmp_path / "tps_abc.state"
    state_path.write_text("garbage", encoding="utf-8")
    state_path.chmod(0o600)
    limiter = _critical(state_path, 1000.0, [])
    assert limiter.acquire() == 0.0
    assert state_path.read_text(encoding="utf-8") == repr(1000.0 + INTERVAL)


def test_two_processes_never_share_a_slot(tmp_path: Path) -> None:
    state_path = tmp_path / "tps_abc.state"
    clock = {"t": 500.0}
    first = HostPacedLimiter(state_path, max_lead_seconds=None, clock=lambda: clock["t"], sleep=lambda s: None)
    second = HostPacedLimiter(state_path, max_lead_seconds=None, clock=lambda: clock["t"], sleep=lambda s: None)
    seen: list[float] = []
    for limiter in (first, second, first, second):
        limiter.acquire()
        seen.append(float(state_path.read_text(encoding="utf-8")))
    from itertools import pairwise

    for before, after in pairwise(seen):
        assert after - before >= INTERVAL - 1e-9


def test_max_lead_and_key_id_contract() -> None:
    assert max_lead_for(AdmissionClass.CRITICAL) is None
    assert max_lead_for(AdmissionClass.STANDARD) == 1.0
    assert max_lead_for(AdmissionClass.BULK) == 0.25
    import hashlib

    assert kis_key_id("k") == hashlib.sha256(b"k").hexdigest()[:12]


def _store(tmp_path: Path, key: str, moment: datetime) -> SharedTokenStore:
    state = {"now": moment}
    return SharedTokenStore(tmp_path, key, now=lambda: state["now"])


def test_token_file_matches_sibling_reader_contract(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    store = _store(tmp_path, "my-key", moment)
    assert store.ensure(lambda: ("tok123", 86400)) == "tok123"
    token_path = tmp_path / f"token_{kis_key_id('my-key')}.json"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    raw = token_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert list(payload.keys()) == ["access_token", "expired_at", "app_key", "issued_at"]
    assert payload["access_token"] == "tok123"
    assert payload["app_key"] == "my-key"
    assert payload["issued_at"] == "2026-01-05T09:00:00+09:00"
    assert payload["expired_at"] == "2026-01-06T09:00:00+09:00"


def test_valid_cached_token_reused_without_issuing(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    store = _store(tmp_path, "k", moment)
    store.ensure(lambda: ("T1", 1800))
    calls: list[int] = []

    def _issue() -> tuple[str, int]:
        calls.append(1)
        return ("T2", 1800)

    assert store.ensure(_issue) == "T1"
    assert calls == []


def test_rotated_token_is_adopted(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    store = _store(tmp_path, "k", moment)
    store.ensure(lambda: ("T1", 1800))
    token_path = tmp_path / f"token_{kis_key_id('k')}.json"
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    payload["access_token"] = "T2"
    token_path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.ensure(lambda: ("T3", 1800), rejected="T1") == "T2"

    store2 = _store(tmp_path, "k2", moment)
    store2.ensure(lambda: ("T1", 1800))
    calls: list[int] = []

    def _issue() -> tuple[str, int]:
        calls.append(1)
        return ("T9", 1800)

    assert store2.ensure(_issue, rejected="T1") == "T9"
    assert len(calls) == 1


def test_near_expiry_and_naive_timestamps_miss(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    store = _store(tmp_path, "k", moment)
    store.ensure(lambda: ("T1", 300))
    assert store.load_valid() is None
    token_path = tmp_path / f"token_{kis_key_id('k')}.json"
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    payload["expired_at"] = "2026-01-05T09:30:00"
    token_path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.load_valid() is None


def test_lock_timeout_is_bounded(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    probe = SharedTokenStore(tmp_path, "k", now=lambda: moment)
    lock_path = Path(str(probe._token_path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        clock = {"t": 0.0}
        store = SharedTokenStore(
            tmp_path,
            "k",
            now=lambda: moment,
            clock=lambda: clock["t"],
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        )
        with pytest.raises(TokenStoreLockTimeout):
            store.ensure(lambda: ("T", 1800))
        assert clock["t"] >= 240.0
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


def test_marker_is_required_on_shared_host(tmp_path: Path) -> None:
    with pytest.raises(AdmissionDirNotSharedError):
        limiter_for(tmp_path, "k", AdmissionClass.BULK, require_marker=True)
    (tmp_path / ".host-admission").write_text("", encoding="utf-8")
    limiter = limiter_for(tmp_path, "k", AdmissionClass.BULK, require_marker=True)
    assert isinstance(limiter, HostPacedLimiter)
    assert limiter_for(tmp_path, "k", AdmissionClass.BULK, require_marker=False) is not None


def test_nonfinite_and_corrupt_cache_reads_miss(tmp_path: Path) -> None:
    state_path = tmp_path / "tps_abc.state"
    state_path.write_text("nan", encoding="utf-8")
    state_path.chmod(0o600)
    assert _critical(state_path, 1000.0, []).acquire() == 0.0
    state_path.write_text("inf", encoding="utf-8")
    state_path.chmod(0o600)
    assert _critical(state_path, 1000.0, []).acquire() == 0.0

    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    store = _store(tmp_path, "k", moment)
    token_path = tmp_path / f"token_{kis_key_id('k')}.json"
    token_path.write_text("not json", encoding="utf-8")
    assert store.load_valid() is None
    token_path.write_text('["x"]', encoding="utf-8")
    assert store.load_valid() is None
    token_path.write_text(json.dumps({"access_token": "", "expired_at": "x", "app_key": "k", "issued_at": "y"}), encoding="utf-8")
    assert store.load_valid() is None
    token_path.write_text(
        json.dumps({"access_token": "T", "expired_at": "2026-01-06T09:00:00+09:00", "app_key": "other", "issued_at": "2026-01-05T09:00:00+09:00"}),
        encoding="utf-8",
    )
    assert store.load_valid() is None
    token_path.write_text(
        json.dumps({"access_token": "T", "expired_at": 1, "app_key": "k", "issued_at": 2}),
        encoding="utf-8",
    )
    assert store.load_valid() is None
    token_path.write_text(
        json.dumps({"access_token": "T", "expired_at": "bad", "app_key": "k", "issued_at": "bad"}),
        encoding="utf-8",
    )
    assert store.load_valid() is None


def _token_payload(token: str, key: str) -> str:
    return json.dumps(
        {
            "access_token": token,
            "expired_at": "2026-01-06T09:00:00+09:00",
            "app_key": key,
            "issued_at": "2026-01-05T09:00:00+09:00",
        }
    )


def test_token_rotated_while_waiting_for_lock_is_adopted(tmp_path: Path) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    token_path = tmp_path / f"token_{kis_key_id('k')}.json"
    holder = os.open(f"{token_path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    polls: list[float] = []

    def _sibling_finishes(seconds: float) -> None:
        polls.append(seconds)
        token_path.write_text(_token_payload("T2", "k"), encoding="utf-8")
        fcntl.flock(holder, fcntl.LOCK_UN)

    store = SharedTokenStore(tmp_path, "k", now=lambda: moment, clock=lambda: 0.0, sleep=_sibling_finishes)
    issued: list[int] = []

    def _issue() -> tuple[str, int]:
        issued.append(1)
        return ("T3", 1800)

    try:
        assert store.ensure(_issue) == "T2"
    finally:
        os.close(holder)
    assert issued == []
    assert polls == [0.05]


def test_read_only_lock_file_still_serialises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    moment = datetime(2026, 1, 5, 9, 0, 0, tzinfo=SEOUL)
    lock_path = tmp_path / f"token_{kis_key_id('k')}.json.lock"
    lock_path.write_text("", encoding="utf-8")
    real_open = os.open
    lock_flags: list[int] = []

    def _foreign_owned(path: str, flags: int, mode: int = 0o777) -> int:
        if path == str(lock_path):
            lock_flags.append(flags)
            if flags & os.O_RDWR:
                raise PermissionError(path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _foreign_owned)
    store = SharedTokenStore(tmp_path, "k", now=lambda: moment)
    assert store.ensure(lambda: ("T1", 1800)) == "T1"
    assert lock_flags == [os.O_RDWR | os.O_CREAT, os.O_RDONLY]
    assert store.load_valid() == "T1"
