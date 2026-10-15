"""Best-effort content-addressed cache for Bronze parse results."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import zlib
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from types import ModuleType
from typing import Any

from src.core.pit import PITDataError

_LOG = logging.getLogger(__name__)

_CHUNK_SIZE = 500


class ParseCache:
    """Best-effort sqlite KV of JSON parse results keyed by Bronze content hash; never alters build results."""

    def __init__(self, path: Path, *, namespace: str, fingerprint: str) -> None:
        self._path = Path(path)
        self._namespace = namespace
        self._fingerprint = fingerprint
        self._conn: sqlite3.Connection | None = None
        self._warned = False
        self._closed = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), timeout=30.0)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS entries(content_hash TEXT PRIMARY KEY, value BLOB NOT NULL)"
                )
                conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
                if rows.get("namespace") != namespace or rows.get("fingerprint") != fingerprint:
                    conn.execute("DELETE FROM entries")
                    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("namespace", namespace))
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("fingerprint", fingerprint)
                    )
                    conn.commit()
            except Exception:
                conn.close()
                raise
            self._conn = conn
        except Exception:
            self._conn = None
            self._note_degraded("open")

    def _note_degraded(self, stage: str) -> None:
        if not self._warned:
            self._warned = True
            _LOG.warning("[SYS] parse cache degraded stage=%s path=%s", stage, str(self._path))

    def get_many(self, content_hashes: Sequence[str]) -> dict[str, Any]:
        """Return cached values for the hashes present; absent, corrupt or unreadable entries are simply omitted."""
        if self._closed or self._conn is None:
            return {}
        result: dict[str, Any] = {}
        try:
            for offset in range(0, len(content_hashes), _CHUNK_SIZE):
                chunk = list(content_hashes[offset : offset + _CHUNK_SIZE])
                placeholders = ",".join("?" for _ in chunk)
                try:
                    rows = self._conn.execute(
                        f"SELECT content_hash, value FROM entries WHERE content_hash IN ({placeholders})",  # noqa: S608
                        chunk,
                    ).fetchall()
                except Exception:
                    self._note_degraded("read")
                    return {}
                for content_hash, blob in rows:
                    try:
                        payload = blob if isinstance(blob, bytes) else bytes(blob)
                        result[str(content_hash)] = json.loads(zlib.decompress(payload))
                    except Exception:
                        self._note_degraded("read")
                        continue
            return result
        except Exception:
            self._note_degraded("read")
            return {}

    def put_many(self, items: Mapping[str, Any]) -> None:
        """Persist new results atomically; silently skips when the cache is unavailable."""
        if self._closed or self._conn is None or not items:
            return
        try:
            encoded: dict[str, bytes] = {}
            for key, value in items.items():
                try:
                    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
                except Exception:
                    self._note_degraded("write")
                    return
                try:
                    encoded[str(key)] = zlib.compress(raw)
                except Exception:
                    self._note_degraded("write")
                    return
            try:
                with self._conn:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO entries(content_hash, value) VALUES (?, ?)",
                        [(key, encoded[key]) for key in encoded],
                    )
            except Exception:
                self._note_degraded("write")
                return
        except Exception:
            self._note_degraded("write")
            return

    def prune(self, keep: Set[str]) -> int:
        """Delete cached rows whose hash is not in keep; return the deleted count."""
        if self._closed or self._conn is None:
            return 0
        try:
            rows = self._conn.execute("SELECT content_hash FROM entries").fetchall()
            stale = [str(row[0]) for row in rows if str(row[0]) not in keep]
            if not stale:
                return 0
            deleted = 0
            with self._conn:
                for offset in range(0, len(stale), _CHUNK_SIZE):
                    chunk = stale[offset : offset + _CHUNK_SIZE]
                    placeholders = ",".join("?" for _ in chunk)
                    cursor = self._conn.execute(
                        f"DELETE FROM entries WHERE content_hash IN ({placeholders})", chunk  # noqa: S608
                    )
                    deleted += cursor.rowcount
            return deleted
        except Exception:
            self._note_degraded("prune")
            return 0

    def close(self) -> None:
        """Release the sqlite connection; idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            self._note_degraded("close")
        finally:
            self._conn = None

    def __enter__(self) -> ParseCache:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def source_fingerprint(*modules: ModuleType) -> str:
    """Hex sha256 over the source bytes of the given modules, in the order given."""
    digest = hashlib.sha256()
    for module in modules:
        filename = getattr(module, "__file__", None)
        if not filename:
            raise PITDataError("parse cache source module has no readable file")
        try:
            digest.update(Path(str(filename)).read_bytes())
        except OSError as exc:
            raise PITDataError("parse cache source module has no readable file") from exc
    return digest.hexdigest()
