"""Cross-process protocol boundaries and sync/async parity."""
from __future__ import annotations

import asyncio
import fcntl
from datetime import UTC, datetime, timedelta

import pytest

from src.integrations import host_broker_contract as c


def test_paths_and_rate_validation(tmp_path):
    for vendor in ("db", "DBSEC"):
        with pytest.raises(ValueError, match="vendor"):
            c.token_record_path(tmp_path, vendor, "K")
    with pytest.raises(ValueError, match="credential"):
        c.token_record_path(tmp_path, "dbsec", "")
    for scope in ("*", "../escape", "", "with space"):
        with pytest.raises(ValueError, match="scope"):
            c.admission_state_path(tmp_path, "toss", "K", scope)
    assert c.issue_guard_path(tmp_path, "dbsec", "K").name == "token_issue_dbsec_86be9a55762d.state"
    assert c.rate_for("dbsec", None) == 20.0
    assert c.rate_for("kiwoom", "orders") == 5.0
    with pytest.raises(ValueError, match="undeclared"):
        c.rate_for("dbsec", "orders")
    assert c._guard_path_for_token(tmp_path / "custom.json", "dbsec").parent == tmp_path


def test_unreadable_record_and_atomic_publish_failure(tmp_path, monkeypatch):
    assert c._read_record_file(tmp_path) is None
    path = tmp_path / "token.json"
    path.write_bytes(b"\xff")
    assert c._read_record_file(path) is None
    original = path.read_bytes()

    def fail_replace(*_args):
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(c.os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk failure"):
            c._atomic_write_text(path, "new token")
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"lock_timeout_seconds": float("inf")}, "lock_timeout"),
    ({"expiry_margin_seconds": -1.0}, "expiry_margin"),
    ({"unknown_expiry_max_age_seconds": -1.0}, "unknown_expiry"),
    ({"issue_throttle_policy": "invalid"}, "policy"),
])
def test_store_argument_boundaries(tmp_path, kwargs, match):
    params = {"vendor": "ls", "lock_timeout_seconds": 30.0, "expiry_margin_seconds": 0.0,
              "clock": lambda: datetime.now(UTC), "issue_throttle_policy": "wait"}
    params.update(kwargs)
    for store in (c.TokenStore, c.AsyncTokenStore):
        with pytest.raises(ValueError, match=match):
            store(tmp_path / "token.json", **params)


def test_unknown_expiry_has_bounded_lifetime(tmp_path):
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = c.TokenStore(c.token_record_path(tmp_path, "ls", "K"), vendor="ls",
                         lock_timeout_seconds=30.0, expiry_margin_seconds=0.0,
                         clock=lambda: now[0], issue_throttle_policy="wait")
    store.get_or_issue(lambda: c.IssuedToken("first", None))
    now[0] += timedelta(hours=12)
    record = store.get_or_issue(lambda: c.IssuedToken("second", None))
    assert record.access_token == "second"
    assert record.generation == 2


def test_guard_wait_is_bounded(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    guard = c.issue_guard_path(tmp_path, "dbsec", "K")
    guard.write_text(str(now.timestamp()), encoding="ascii")
    path = c.token_record_path(tmp_path, "dbsec", "K")
    store = c.TokenStore(path, vendor="dbsec", lock_timeout_seconds=1.0,
                         expiry_margin_seconds=0.0, clock=lambda: now, issue_throttle_policy="wait")
    with pytest.raises(c.TokenIssueThrottled):
        store.get_or_issue(lambda: pytest.fail("must not issue within guard"))
    assert not path.exists()
    assert c._throttle_delay_seconds("dbsec", None, now.timestamp()) == 0.0


def test_async_store_contention_guard_and_rejection(tmp_path):
    async def scenario():
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        path = c.token_record_path(tmp_path, "dbsec", "K")
        guard = c.issue_guard_path(tmp_path, "dbsec", "K")
        guard.write_text(str(now[0].timestamp() - 10), encoding="ascii")

        async def sleep(seconds):
            now[0] += timedelta(seconds=seconds)
            await asyncio.sleep(0)

        def store(vendor="dbsec", policy="wait", timeout=240.0):
            return c.AsyncTokenStore(path, vendor=vendor, lock_timeout_seconds=timeout,
                                     expiry_margin_seconds=0.0, clock=lambda: now[0],
                                     issue_throttle_policy=policy, sleep=sleep)

        calls = []

        async def issue():
            calls.append(now[0])
            await asyncio.sleep(0)
            return c.IssuedToken(f"token-{len(calls)}", 3600.0)

        first, second = await asyncio.gather(store().get_or_issue(issue), store().get_or_issue(issue))
        assert first == second
        assert len(calls) == 1
        assert calls[0] >= datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=51)
        assert await store().get_or_issue(issue) == first
        for policy, timeout in (("raise", 240.0), ("wait", 1.0)):
            with pytest.raises(c.TokenIssueThrottled):
                await store(policy=policy, timeout=timeout).replace_rejected(first.access_token, issue)
        rotated = await store().replace_rejected(first.access_token, issue)
        assert rotated.generation == 2
        assert len(calls) == 2
        ls = store(vendor="ls")
        assert ls._build_record(c.IssuedToken("built", None), rotated).generation == 3
        assert ls._build(c.IssuedToken("built", None), None).generation == 1
        await ls.replace_rejected(rotated.access_token, issue)
        assert len(calls) == 3

    asyncio.run(scenario())


def test_async_lock_timeout_closes_descriptor(tmp_path):
    path = c.token_record_path(tmp_path, "ls", "K")
    store = c.AsyncTokenStore(path, vendor="ls", lock_timeout_seconds=0.02,
                             expiry_margin_seconds=0.0, clock=lambda: datetime.now(UTC),
                             issue_throttle_policy="wait")

    async def issue():
        pytest.fail("lock held: no issuance")

    with open(str(path) + ".lock", "w", encoding="ascii") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(c.TokenStoreLockTimeout):
            asyncio.run(store.get_or_issue(issue))
    assert not path.exists()
    assert asyncio.run(store.get_or_issue(_async_issue)).generation == 1


async def _async_issue():
    return c.IssuedToken("after-timeout", 3600.0)


@pytest.mark.parametrize("kwargs", [
    {"max_rate": 0.0}, {"max_rate": float("nan")},
    {"max_rate": 20.0, "max_lead_seconds": -1.0},
])
def test_pacer_argument_boundaries(tmp_path, kwargs):
    for pacer in (c.HostPacer, c.AsyncHostPacer):
        with pytest.raises(ValueError, match="positive"):
            pacer(tmp_path / "slot.state", **kwargs)


def test_bounded_sync_async_slots_share_ascii_state(tmp_path):
    state = tmp_path / "slot.state"
    now = [1000.0]
    state.write_text("1001.0", encoding="ascii")

    def sleep(seconds):
        now[0] += seconds

    sync = c.HostPacer(state, 20.0, max_lead_seconds=0.1, clock=lambda: now[0], sleep=sleep)
    sync.acquire()
    assert now[0] == pytest.approx(1001.0)
    assert float(state.read_text(encoding="ascii")) == pytest.approx(1001.05)

    async def asleep(seconds):
        sleep(seconds)

    def read_next(pacer):
        with open(state, encoding="ascii") as handle:
            return pacer._read_next_free_locked(handle.fileno())

    async def scenario():
        pacer = c.AsyncHostPacer(state, 20.0, max_lead_seconds=0.1, clock=lambda: now[0], sleep=asleep)
        now[0] = 1000.0
        await pacer.acquire()
        assert now[0] == pytest.approx(1001.05)
        assert read_next(pacer) == pytest.approx(1001.1)
        unbounded = c.AsyncHostPacer(state, 20.0, clock=lambda: now[0], sleep=asleep)
        assert unbounded._reserve_slot() == pytest.approx(0.05)
        await unbounded.acquire()
        assert now[0] == pytest.approx(1001.15)

    asyncio.run(scenario())
