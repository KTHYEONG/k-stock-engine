"""Invariant guard tests for the content-addressed parse cache."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.parse_cache import ParseCache, source_fingerprint


def _cache(tmp_path: Path, *, namespace: str = "ns-v1", fingerprint: str = "fp-a") -> ParseCache:
    return ParseCache(tmp_path / "parse.sqlite", namespace=namespace, fingerprint=fingerprint)


def test_round_trip_returns_only_present(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({"a": {"x": 1}, "b": [1, 2], "c": "v"})
        assert cache.get_many(["a", "b", "c", "missing"]) == {"a": {"x": 1}, "b": [1, 2], "c": "v"}
    finally:
        cache.close()


def test_fingerprint_change_clears(tmp_path: Path) -> None:
    path = tmp_path / "parse.sqlite"
    first = ParseCache(path, namespace="ns-v1", fingerprint="fp-a")
    try:
        first.put_many({"a": 1})
        assert first.get_many(["a"]) == {"a": 1}
    finally:
        first.close()
    second = ParseCache(path, namespace="ns-v1", fingerprint="fp-b")
    try:
        assert second.get_many(["a"]) == {}
        rows = second._conn.execute("SELECT COUNT(*) FROM entries").fetchone() if second._conn else (0,)
        assert rows[0] == 0
    finally:
        second.close()


def test_namespace_change_clears(tmp_path: Path) -> None:
    path = tmp_path / "parse.sqlite"
    first = ParseCache(path, namespace="ns-v1", fingerprint="fp-a")
    try:
        first.put_many({"a": 1})
    finally:
        first.close()
    second = ParseCache(path, namespace="ns-v2", fingerprint="fp-a")
    try:
        assert second.get_many(["a"]) == {}
    finally:
        second.close()


def test_same_fingerprint_retains(tmp_path: Path) -> None:
    path = tmp_path / "parse.sqlite"
    first = ParseCache(path, namespace="ns-v1", fingerprint="fp-a")
    try:
        first.put_many({"a": {"v": 1}})
    finally:
        first.close()
    second = ParseCache(path, namespace="ns-v1", fingerprint="fp-a")
    try:
        assert second.get_many(["a"]) == {"a": {"v": 1}}
    finally:
        second.close()


def test_corrupt_row_is_a_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({"good": {"x": 1}, "bad": {"y": 2}})
        assert cache._conn is not None
        cache._conn.execute("UPDATE entries SET value=? WHERE content_hash=?", (b"not-zlib", "bad"))
        cache._conn.commit()
        assert cache.get_many(["good", "bad"]) == {"good": {"x": 1}}
        cache._conn.execute("UPDATE entries SET value=? WHERE content_hash=?", ("plain-text", "bad"))
        cache._conn.commit()
        assert cache.get_many(["bad"]) == {}
    finally:
        cache.close()


def test_unwritable_path_degrades(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="src.data.parse_cache"):
        caplog.clear()
        cache = ParseCache(blocker / "parse.sqlite", namespace="ns-v1", fingerprint="fp-a")
        try:
            cache.put_many({"a": 1})
            assert cache.get_many(["a"]) == {}
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1
            assert "[SYS]" in warnings[0].getMessage()
        finally:
            cache.close()


def test_prune_retires_pages(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({h: idx for idx, h in enumerate(["a", "b", "c", "d", "e"])})
        assert cache.prune({"a", "b"}) == 3
        assert cache.get_many(["a", "b", "c", "d", "e"]) == {"a": 0, "b": 1}
        assert cache.prune({"a", "b"}) == 0
    finally:
        cache.close()


def test_prune_empty_keep_clears_all(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({"a": 1, "b": 2})
        assert cache.prune(set()) == 2
        assert cache.get_many(["a", "b"]) == {}
    finally:
        cache.close()


def test_atomic_write(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({"ok": {"x": 1}})
        cache.put_many({"ok": {"x": 2}, "broken": {"bad": object()}})
        assert cache.get_many(["ok", "broken"]) == {"ok": {"x": 1}}
    finally:
        cache.close()


def test_sql_write_failure_rolls_back_entire_batch(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.put_many({"existing": 1})
        assert cache._conn is not None
        cache._conn.execute(
            "CREATE TRIGGER reject_bad BEFORE INSERT ON entries "
            "WHEN NEW.content_hash = 'bad' BEGIN SELECT RAISE(ABORT, 'rejected'); END"
        )
        cache._conn.commit()
        cache.put_many({"existing": 2, "new": 3, "bad": 4})
        assert cache.get_many(["existing", "new", "bad"]) == {"existing": 1}


def test_missing_metadata_invalidates_existing_rows(tmp_path: Path) -> None:
    with _cache(tmp_path) as cache:
        cache.put_many({"a": 1})
        assert cache._conn is not None
        cache._conn.execute("DROP TABLE meta")
        cache._conn.commit()
    with _cache(tmp_path) as cache:
        assert cache.get_many(["a"]) == {}


def test_put_many_empty_is_noop(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({})
        assert cache.get_many([]) == {}
    finally:
        cache.close()


def test_close_is_idempotent_and_context_manager(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    with cache:
        cache.put_many({"a": 1})
        assert cache.get_many(["a"]) == {"a": 1}
    cache.close()
    cache.close()
    assert cache.get_many(["a"]) == {}
    assert cache.prune({"a"}) == 0
    cache.put_many({"a": 1})
    assert cache.get_many(["a"]) == {}


def test_get_many_chunks_over_limit(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        items = {f"k{i:04d}": i for i in range(600)}
        cache.put_many(items)
        assert cache.get_many(list(items)) == items
    finally:
        cache.close()


def test_put_many_sqlite_failure_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)
    try:
        assert cache._conn is not None
        cache._conn.execute("DROP TABLE entries")
        cache._conn.commit()
        cache.put_many({"a": 1})
        assert cache.get_many(["a"]) == {}
    finally:
        cache.close()
        monkeypatch.undo()


def test_get_many_outer_failure_returns_empty(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        assert cache.get_many(None) == {}  # type: ignore[arg-type]
    finally:
        cache.close()


def test_prune_query_failure_returns_zero(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        assert cache._conn is not None
        cache.put_many({"a": 1})
        cache._conn.execute("DROP TABLE entries")
        cache._conn.commit()
        assert cache.prune({"a"}) == 0
    finally:
        cache.close()


def test_close_failure_is_silent(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache._conn is not None

    class _BoomConn:
        def close(self) -> None:
            raise RuntimeError("close boom")

    cache._conn.close()
    cache._conn = _BoomConn()  # type: ignore[assignment]
    cache.close()
    cache.close()


def test_source_fingerprint_is_ordered_and_stable() -> None:
    first = source_fingerprint(sys.modules["json"], sys.modules["pathlib"])
    assert first == source_fingerprint(sys.modules["json"], sys.modules["pathlib"])
    assert first != source_fingerprint(sys.modules["pathlib"], sys.modules["json"])


def test_source_fingerprint_rejects_missing_file() -> None:
    with pytest.raises(PITDataError):
        source_fingerprint(type(sys)("ghost"))
    ghost = type(sys)("ghost-file")
    ghost.__file__ = "/nonexistent-parse-cache-source-xyz.py"
    with pytest.raises(PITDataError):
        source_fingerprint(ghost)


def test_zlib_and_sqlite_failures_degrade_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zlib as _zlib

    cache = _cache(tmp_path)
    try:
        original = _zlib.compress
        monkeypatch.setattr("src.data.parse_cache.zlib.compress", lambda _raw: (_ for _ in ()).throw(OSError("boom")))
        cache.put_many({"a": {"x": 1}})
        monkeypatch.setattr("src.data.parse_cache.zlib.compress", original)
        cache.put_many({"a": {"x": 1}})
        assert cache.get_many(["a"]) == {"a": {"x": 1}}
    finally:
        cache.close()


def test_put_many_outer_failure_skips(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        class _BadMapping(dict):  # type: ignore[type-arg]
            def items(self) -> object:  # type: ignore[override]
                raise RuntimeError("items boom")

        cache.put_many(_BadMapping({"a": 1}))  # type: ignore[arg-type]
        assert cache.get_many(["a"]) == {}
    finally:
        cache.close()


def test_corrupt_text_row_is_a_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    try:
        cache.put_many({"t": {"x": 1}})
        assert cache._conn is not None
        cache._conn.execute("UPDATE entries SET value=? WHERE content_hash=?", ("plain-text", "t"))
        cache._conn.commit()
        assert cache.get_many(["t"]) == {}
    finally:
        cache.close()


def test_sqlite_file_is_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "parse.sqlite"
    cache = ParseCache(path, namespace="ns-v1", fingerprint="fp-a")
    try:
        assert path.is_file()
        connection = sqlite3.connect(str(path))
        try:
            assert connection.execute("SELECT name FROM sqlite_master").fetchall()
        finally:
            connection.close()
    finally:
        cache.close()


def test_init_inner_failure_closes_and_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3 as _sqlite3

    real_connect = _sqlite3.connect
    closed: list[bool] = []

    class _FailingConn:
        def execute(self, *args: object, **kwargs: object) -> object:
            raise _sqlite3.OperationalError("boom")

        def close(self) -> None:
            closed.append(True)

        def commit(self) -> None:
            pass

    def _boom_connect(*args: object, **kwargs: object) -> object:
        real = real_connect(str(tmp_path / "inner.sqlite"), timeout=30.0)
        real.close()
        return _FailingConn()

    monkeypatch.setattr("src.data.parse_cache.sqlite3.connect", _boom_connect)
    cache = ParseCache(tmp_path / "inner.sqlite", namespace="ns-v1", fingerprint="fp-a")
    try:
        assert closed == [True]
        assert cache.get_many(["a"]) == {}
    finally:
        cache.close()
