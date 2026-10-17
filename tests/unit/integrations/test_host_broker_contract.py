"""Canonical contract invariants (via adapter)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.integrations import host_broker as hb
from src.integrations.host_broker_contract import (
    AsyncHostPacer,
    AsyncTokenStore,
    HostBrokerDirNotSharedError,
    HostPacer,
    IssuedToken,
    TokenIssueThrottled,
    TokenStore,
    TokenStoreLockTimeout,
    token_record_path,
)


def _fake_datetime_clock(start: float):
    state = {"t": start}

    def clock() -> datetime:
        return datetime.fromtimestamp(state["t"], tz=UTC)

    def sleep(seconds: float) -> None:
        state["t"] += seconds

    return clock, sleep, state


def _store(tmp_path: Path, vendor: str = "dbsec", key: str = "K", **overrides):
    from src.integrations.host_broker_contract import token_record_path as _tp

    path = _tp(tmp_path, vendor, key)
    clock, sleep, _ = _fake_datetime_clock(1_700_000_000.0)
    params = {
        "vendor": vendor,
        "issue_throttle_policy": "wait",
        "lock_timeout_seconds": 240.0,
        "expiry_margin_seconds": 0.0,
        "clock": clock,
        "sleep": sleep,
    }
    params.update(overrides)
    return TokenStore(path, **params), path


def test_token_record_path_naming(tmp_path) -> None:
    path = token_record_path(tmp_path, "dbsec", "K")
    expect = f"token_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.json"
    assert path.name == expect


def test_rejects_alias_vendor(tmp_path) -> None:
    for bad in ("db", "DBSEC"):
        with pytest.raises(ValueError, match=r"."):
            hb.open_token_store(tmp_path, bad, "K", lock_timeout_seconds=240.0, expiry_margin_seconds=0.0)


def test_schema_strict(tmp_path) -> None:
    store, path = _store(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert store.read() is None
    payload = {
        "schema_version": 2,
        "access_token": "t",
        "issued_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "generation": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.read() is None
    extra = {
        "schema_version": 1,
        "access_token": "t",
        "issued_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "generation": 1,
        "extra": 1,
    }
    path.write_text(json.dumps(extra), encoding="utf-8")
    assert store.read() is None


def test_single_issuance_under_contention(tmp_path) -> None:
    from src.integrations.host_broker_contract import token_record_path as _tp

    path = _tp(tmp_path, "dbsec", "K")
    calls = {"n": 0}

    def issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken("tok", 3600.0)

    def make() -> TokenStore:
        return TokenStore(
            path, vendor="dbsec", lock_timeout_seconds=10.0, expiry_margin_seconds=0.0,
            clock=lambda: datetime.now(UTC), issue_throttle_policy="wait",
        )

    results: list = []
    errors: list = []

    def work() -> None:
        try:
            results.append(make().get_or_issue(issue))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert calls["n"] == 1
    assert len(results) == 2
    assert results[0].generation == 1
    assert results[1].generation == 1


def test_replace_rejected_no_pingpong(tmp_path) -> None:
    store, _ = _store(tmp_path)
    first = store.get_or_issue(lambda: IssuedToken("A", 3600.0))
    assert first.generation == 1
    second = store.get_or_issue(lambda: IssuedToken("B", 3600.0))
    # second call reuses A since usable; force B via replace
    replaced = store.replace_rejected("A", lambda: IssuedToken("B", 3600.0))
    assert replaced.access_token == "B"
    calls = {"n": 0}

    def issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken("C", 3600.0)

    out = store.replace_rejected("A", issue)
    assert out.access_token == "B"
    assert calls["n"] == 0
    assert out.generation == 2


def test_dbsec_min_issue_interval_wait(tmp_path) -> None:
    clock, sleep, state = _fake_datetime_clock(1_700_000_000.0)
    store = TokenStore(
        token_record_path(tmp_path, "dbsec", "K"),
        vendor="dbsec",
        lock_timeout_seconds=240.0,
        expiry_margin_seconds=0.0,
        issue_throttle_policy="wait",
        clock=clock,
        sleep=sleep,
    )
    guard = tmp_path / f"token_issue_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.state"
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(repr(state["t"] - 10.0), encoding="utf-8")
    seen: list[float] = []

    def issue() -> IssuedToken:
        seen.append(state["t"])
        return IssuedToken("tok", 3600.0)

    record = store.get_or_issue(issue)
    assert record.generation == 1
    assert seen
    assert seen[0] - 1_700_000_000.0 >= 51.0 - 1e-9


def test_dbsec_min_issue_interval_raise(tmp_path) -> None:
    clock, sleep, state = _fake_datetime_clock(1_700_000_000.0)
    path = token_record_path(tmp_path, "dbsec", "K")
    store = TokenStore(
        path, vendor="dbsec", lock_timeout_seconds=240.0, expiry_margin_seconds=0.0,
        clock=clock, sleep=sleep, issue_throttle_policy="raise",
    )
    guard = tmp_path / f"token_issue_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.state"
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(repr(state["t"] - 10.0), encoding="utf-8")
    before = guard.read_text(encoding="utf-8")
    calls = {"n": 0}

    def issue() -> IssuedToken:
        calls["n"] += 1
        return IssuedToken("tok", 3600.0)

    with pytest.raises(TokenIssueThrottled):
        store.get_or_issue(issue)
    assert calls["n"] == 0
    assert guard.read_text(encoding="utf-8") == before
    assert not path.exists()
    with pytest.raises(TokenIssueThrottled):
        store.get_or_issue(issue)


def test_sync_async_byte_identical(tmp_path) -> None:
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    sync = TokenStore(
        token_record_path(dir_a, "dbsec", "K"), vendor="dbsec",
        lock_timeout_seconds=240.0, expiry_margin_seconds=0.0,
        clock=lambda: fixed, sleep=lambda _s: None, issue_throttle_policy="wait",
    )
    async_store = AsyncTokenStore(
        token_record_path(dir_b, "dbsec", "K"), vendor="dbsec",
        lock_timeout_seconds=240.0, expiry_margin_seconds=0.0,
        clock=lambda: fixed, issue_throttle_policy="wait",
    )
    sync.get_or_issue(lambda: IssuedToken("tok", 3600.0))
    async def issue() -> IssuedToken:
        return IssuedToken("tok", 3600.0)

    asyncio.run(async_store.get_or_issue(issue))
    token_a = (dir_a / token_record_path(dir_a, "dbsec", "K").name).read_bytes()
    token_b = (dir_b / token_record_path(dir_b, "dbsec", "K").name).read_bytes()
    assert token_a == token_b
    sha = hashlib.sha256(b"K").hexdigest()[:12]
    assert (dir_a / f"token_issue_dbsec_{sha}.state").read_bytes() == (
        dir_b / f"token_issue_dbsec_{sha}.state"
    ).read_bytes()
    assert async_store.read() is not None
    assert asyncio.run(async_store.replace_rejected("other", issue)).access_token == "tok"


def test_pacing_shared_slots(tmp_path) -> None:
    broker = tmp_path / "broker"
    broker.mkdir()
    (broker / ".host-admission").write_text("", encoding="utf-8")
    clock_state = {"t": 1000.0}
    sleeps: list[float] = []

    def clock() -> float:
        return clock_state["t"]

    def sleep(seconds: float) -> None:
        clock_state["t"] += seconds
        sleeps.append(seconds)

    first = hb.AdmissionSlot(broker, "dbsec", "K", clock=clock, sleep=sleep)
    second = hb.AdmissionSlot(broker, "dbsec", "K", clock=clock, sleep=sleep)
    grants: list[float] = []
    for turn in range(40):
        (first if turn % 2 == 0 else second).wait_turn()
        grants.append(clock_state["t"])
    ordered = sorted(grants)
    from itertools import pairwise

    for before, after in pairwise(ordered):
        assert after - before >= 0.05 - 1e-9


def test_container_requires_marker(tmp_path, monkeypatch) -> None:
    target = tmp_path / "unmarked"
    target.mkdir()
    monkeypatch.setenv("X_BROKER_DIR", str(target))
    real_exists = Path.exists

    def _exists(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _exists)
    with pytest.raises(HostBrokerDirNotSharedError, match=r"."):
        hb.resolve_broker_dir("X_BROKER_DIR")


def test_vendored_copy_hash_pinned() -> None:
    path = Path("src/integrations/host_broker_contract.py")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# canonical: krx-alpha/deploy/shared/host_broker_contract.py contract=v1"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "0bfab4ccae0f0921a32493492683014e1e080e708db3d92473731a5c4eb0b0c1"


def test_contract_branches(tmp_path) -> None:
    from src.integrations.host_broker_contract import _read_guard_value, resolve_host_broker_dir

    assert resolve_host_broker_dir(tmp_path, require_shared="never") == tmp_path
    assert _read_guard_value(tmp_path / "missing.state") is None
    bad = tmp_path / "bad.state"
    for text in ("", "garbage", "nan", "inf", "123.5"):
        bad.write_text(text, encoding="utf-8")
        assert _read_guard_value(bad) == (123.5 if text == "123.5" else None)
    for rate in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            HostPacer(tmp_path / "s.state", rate)
    state = tmp_path / "pace.state"
    for text in ("", "garbage", "nan", "inf"):
        state.write_text(text, encoding="ascii")
        HostPacer(state, 10.0, clock=lambda: 5.0).acquire()
        assert float(state.read_text(encoding="ascii")) == pytest.approx(5.1)
        assert state.stat().st_size <= 64
    asyncio.run(AsyncHostPacer(tmp_path / "async.state", 10.0).acquire())


def test_token_parse_branches(tmp_path) -> None:
    store, path = _store(tmp_path, vendor="ls")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert store.read() is None
    path.write_text("[1,2]", encoding="utf-8")
    assert store.read() is None
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert store.read() is None
    base = {
        "schema_version": 1,
        "access_token": "t",
        "issued_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "generation": 1,
    }
    bad_token = dict(base, access_token="")
    path.write_text(json.dumps(bad_token), encoding="utf-8")
    assert store.read() is None
    bad_gen = dict(base, generation=True)
    path.write_text(json.dumps(bad_gen), encoding="utf-8")
    assert store.read() is None
    bad_gen2 = dict(base, generation=0)
    path.write_text(json.dumps(bad_gen2), encoding="utf-8")
    assert store.read() is None
    bad_issued = dict(base, issued_at="not-a-date")
    path.write_text(json.dumps(bad_issued), encoding="utf-8")
    assert store.read() is None
    naive = datetime.now().isoformat()
    bad_tz = dict(base, issued_at=naive)
    path.write_text(json.dumps(bad_tz), encoding="utf-8")
    assert store.read() is None
    bad_exp_type = dict(base, expires_at=123)
    path.write_text(json.dumps(bad_exp_type), encoding="utf-8")
    assert store.read() is None
    bad_exp_naive = dict(base, expires_at=naive)
    path.write_text(json.dumps(bad_exp_naive), encoding="utf-8")
    assert store.read() is None
    bad_issued_type = dict(base, issued_at=123)
    path.write_text(json.dumps(bad_issued_type), encoding="utf-8")
    assert store.read() is None


def test_lock_timeout(tmp_path) -> None:
    import fcntl

    store, path = _store(tmp_path, lock_timeout_seconds=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(TokenStoreLockTimeout):
            store.get_or_issue(lambda: IssuedToken("t", 3600.0))
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_ls_no_guard_interval(tmp_path) -> None:
    clock, sleep, _ = _fake_datetime_clock(1_700_000_000.0)
    path = token_record_path(tmp_path, "ls", "K")
    store = TokenStore(
        path, vendor="ls", lock_timeout_seconds=240.0, expiry_margin_seconds=0.0, clock=clock, sleep=sleep, issue_throttle_policy="wait"
    )
    record = store.get_or_issue(lambda: IssuedToken("tok", None))
    assert record.expires_at is None
    assert record.generation == 1
    reread = store.get_or_issue(lambda: IssuedToken("other", 10.0))
    assert reread.access_token == "tok"
    replaced = store.replace_rejected("tok", lambda: IssuedToken("tok2", None))
    assert replaced.access_token == "tok2"


def test_expiry_margin_reissue(tmp_path) -> None:
    clock, sleep, _ = _fake_datetime_clock(1_700_000_000.0)
    store = TokenStore(
        token_record_path(tmp_path, "ls", "K"), vendor="ls", lock_timeout_seconds=240.0,
        expiry_margin_seconds=100.0, clock=clock, sleep=sleep, issue_throttle_policy="wait",
    )
    store.get_or_issue(lambda: IssuedToken("t1", 50.0))
    second = store.get_or_issue(lambda: IssuedToken("t2", 5000.0))
    assert second.access_token == "t2"


def test_read_missing_returns_none(tmp_path) -> None:
    store, _ = _store(tmp_path)
    assert store.read() is None


def test_guard_expired_no_wait(tmp_path) -> None:
    clock, sleep, state = _fake_datetime_clock(1_700_000_000.0)
    store = TokenStore(
        token_record_path(tmp_path, "dbsec", "K"), vendor="dbsec",
        lock_timeout_seconds=240.0, expiry_margin_seconds=0.0, clock=clock, sleep=sleep, issue_throttle_policy="wait",
    )
    guard = tmp_path / f"token_issue_dbsec_{hashlib.sha256(b'K').hexdigest()[:12]}.state"
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(repr(state["t"] - 100.0), encoding="utf-8")
    sleeps: list[float] = []
    orig_sleep = sleep

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        orig_sleep(seconds)

    store._sleep = _sleep  # type: ignore[method-assign]
    record = store.get_or_issue(lambda: IssuedToken("tok", 3600.0))
    assert record.generation == 1
    assert sleeps == []


def test_get_or_issue_rereads_under_lock(tmp_path, monkeypatch) -> None:
    from src.integrations.host_broker_contract import TokenRecord, _write_record_file

    store, path = _store(tmp_path)
    original = store._hold_lock
    now = datetime.fromtimestamp(1_700_000_000.0, tz=UTC)

    def peer_publish():
        fd = original()
        _write_record_file(path, TokenRecord("peer", now, now + timedelta(hours=1), 2))
        return fd

    monkeypatch.setattr(store, "_hold_lock", peer_publish)
    assert store.get_or_issue(lambda: pytest.fail("duplicate issuance")).access_token == "peer"


def test_replace_rejected_rotates_expired_peer(tmp_path) -> None:
    store, _ = _store(tmp_path, vendor="ls", expiry_margin_seconds=100.0)
    store.get_or_issue(lambda: IssuedToken("expired", 50.0))
    out = store.replace_rejected("older", lambda: IssuedToken("fresh", 3600.0))
    assert out.access_token == "fresh"
    assert out.generation == 2
