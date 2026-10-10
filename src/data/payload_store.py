"""Read-through packed store for content-addressed Bronze payloads."""
from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa

from src.core.pit import PITDataError

PACK_DIR_NAME: Final = "_packs"
_PACK_CAP_BYTES: Final = 256 * 1024 * 1024
_INDEX_NAME: Final = "index.sqlite3"

_CODEC: Final = pa.Codec("zstd", compression_level=10)


class PayloadMissingError(PITDataError):
    """A Bronze file is present neither as a loose file nor in any pack."""


@dataclass(frozen=True, slots=True)
class PackReport:
    packed: int
    skipped_unverified: int
    raw_bytes: int
    packed_bytes: int
    removed: int


def _bronze_root_for(path: Path) -> Path:
    candidate = path if path.is_dir() else path.parent
    current = candidate.resolve()
    for ancestor in (current, *current.parents):
        if (ancestor / "catalog").is_dir():
            return ancestor
    raise PayloadMissingError(f"Bronze file is outside any Bronze root: {path}")


def _index_path(bronze_root: Path) -> Path:
    return Path(bronze_root) / PACK_DIR_NAME / _INDEX_NAME


def _connect_index(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS members("
        "rel_path TEXT PRIMARY KEY, pack TEXT NOT NULL, "
        "offset INTEGER NOT NULL, length INTEGER NOT NULL, "
        "raw_size INTEGER NOT NULL, sha256 TEXT NOT NULL) WITHOUT ROWID"
    )
    return connection


def _rel_posix(bronze_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(bronze_root.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        raise PayloadMissingError(f"Bronze file is outside its Bronze root: {path}") from exc


def _lookup(bronze_root: Path, rel_path: str) -> tuple[str, int, int, int] | None:
    db_path = _index_path(bronze_root)
    if not db_path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=30.0)
        try:
            row = connection.execute(
                "SELECT pack, offset, length, raw_size FROM members WHERE rel_path = ?",
                (rel_path,),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        return (str(row[0]), int(row[1]), int(row[2]), int(row[3]))
    except (TypeError, ValueError, OverflowError):
        return None


def read_payload(path: Path) -> bytes:
    """Bytes of a Bronze file under a scope's bronze root, from the loose file or its pack; raises PayloadMissingError."""
    candidate = Path(path)
    bronze_root = _bronze_root_for(candidate)
    rel_path = _rel_posix(bronze_root, candidate)
    try:
        loose = candidate.read_bytes()
    except FileNotFoundError:
        loose = None
    except OSError as exc:
        raise PayloadMissingError(f"Bronze payload is unreadable: {path}") from exc
    if loose is not None:
        return loose
    return _read_packed(bronze_root, rel_path)


def _read_packed(bronze_root: Path, rel_path: str) -> bytes:
    found = _lookup(bronze_root, rel_path)
    if found is None:
        raise PayloadMissingError(f"Bronze payload is missing: {rel_path}")
    pack_rel, offset, length, raw_size = found
    pack_path = (bronze_root / pack_rel).resolve()
    if not pack_path.is_relative_to(bronze_root / PACK_DIR_NAME) or min(offset, length, raw_size) < 0:
        raise PayloadMissingError(f"invalid Bronze pack member: {rel_path}")
    try:
        with pack_path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read(length)
    except OSError as exc:
        raise PayloadMissingError(f"Bronze pack is unreadable: {pack_rel}") from exc
    if len(blob) != length:
        raise PayloadMissingError(f"Bronze pack member is truncated: {rel_path}")
    try:
        out = bytes(_CODEC.decompress(blob, raw_size))
    except Exception as exc:
        raise PayloadMissingError(f"Bronze pack member is corrupt: {rel_path}") from exc
    return out


def payload_exists(path: Path) -> bool:
    """True when the Bronze file is readable through read_payload."""
    candidate = Path(path)
    try:
        read_payload(candidate)
    except PayloadMissingError:
        return False
    return True


def _current_pack(bronze_root: Path, kind: str, connection: sqlite3.Connection) -> tuple[str, Path]:
    rows = connection.execute(
        "SELECT pack FROM members WHERE pack LIKE ? ORDER BY pack DESC LIMIT 1",
        (f"_packs/{kind}/pack-%.bin",),
    ).fetchone()
    if rows is not None:
        pack_rel = str(rows[0])
        pack_path = bronze_root / pack_rel
        indexed_end = connection.execute(
            "SELECT MAX(offset + length) FROM members WHERE pack = ?", (pack_rel,)
        ).fetchone()[0]
        if not pack_path.is_file() or pack_path.stat().st_size < int(indexed_end):
            raise PITDataError(f"Bronze pack is missing or truncated: {pack_rel}")
        if pack_path.is_file() and pack_path.stat().st_size < _PACK_CAP_BYTES:
            return pack_rel, pack_path
    existing = sorted((bronze_root / PACK_DIR_NAME / kind).glob("pack-*.bin")) if (bronze_root / PACK_DIR_NAME / kind).is_dir() else []
    indexes = []
    for item in existing:
        try:
            indexes.append(int(item.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    nxt = (max(indexes) + 1) if indexes else 1
    pack_rel = f"_packs/{kind}/pack-{nxt:05d}.bin"
    pack_path = bronze_root / pack_rel
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    return pack_rel, pack_path


def pack_loose(
    bronze_root: Path, *, kinds: Sequence[str], offsite_paths: Collection[str] | None, remove: bool
) -> PackReport:
    """Pack listed Bronze kind directories, removing only offsite-confirmed loose files."""
    root = Path(bronze_root).resolve()
    if not (root / "catalog").is_dir():
        raise PITDataError(f"Bronze root has no catalog: {root}")
    packs = root / PACK_DIR_NAME
    if packs.is_symlink():
        raise PITDataError(f"Bronze pack directory must not be a symlink: {packs}")
    packs.mkdir(parents=True, exist_ok=True)
    with (packs / "writer.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _pack_loose(root, kinds=kinds, offsite_paths=offsite_paths, remove=remove)


def _pack_loose(
    bronze_root: Path, *, kinds: Sequence[str], offsite_paths: Collection[str] | None, remove: bool
) -> PackReport:
    """Move loose payload/receipt files of the given kind directories into packs; returns counts."""
    root = Path(bronze_root).resolve()
    offsite = set(offsite_paths) if offsite_paths is not None else None
    db_path = _index_path(root)
    connection = _connect_index(db_path)
    packed = 0
    skipped_unverified = 0
    raw_bytes = 0
    packed_bytes = 0
    removed = 0
    try:
        for kind in kinds:
            if not kind or Path(kind).name != kind or kind in (".", "..", "catalog", PACK_DIR_NAME):
                continue
            kind_dir = root / kind
            if not kind_dir.is_dir() or kind_dir.is_symlink():
                continue
            if (root / PACK_DIR_NAME / kind).is_symlink():
                raise PITDataError(f"Bronze kind pack directory must not be a symlink: {kind}")
            for file_path in sorted(kind_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                if PACK_DIR_NAME in file_path.relative_to(root).parts:
                    continue
                if file_path.is_symlink() or not file_path.resolve().is_relative_to(kind_dir):
                    raise PITDataError(f"Bronze member escapes its kind directory: {file_path}")
                rel_path = file_path.relative_to(root).as_posix()
                try:
                    raw = file_path.read_bytes()
                except OSError:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                row = connection.execute(
                    "SELECT sha256 FROM members WHERE rel_path = ?", (rel_path,)
                ).fetchone()
                if row is not None:
                    if str(row[0]) != digest:
                        raise PITDataError(f"Bronze payload changed after packing: {rel_path}")
                    packed_raw = _read_packed(root, rel_path)
                    if hashlib.sha256(packed_raw).hexdigest() != digest:
                        raise PITDataError(f"Bronze pack hash mismatch: {rel_path}")
                    needs_remove = remove and (offsite is None or rel_path in offsite)
                    if needs_remove:
                        try:
                            file_path.unlink()
                            removed += 1
                        except OSError as exc:
                            raise PITDataError(f"Bronze payload cannot be removed: {rel_path}") from exc
                    elif remove and offsite is not None and rel_path not in offsite:
                        skipped_unverified += 1
                    continue
                blob = bytes(_CODEC.compress(raw))
                if len(blob) > _PACK_CAP_BYTES:
                    raise PITDataError(f"Bronze member exceeds pack capacity: {rel_path}")
                pack_rel, pack_path = _current_pack(root, kind, connection)
                if pack_path.is_file() and pack_path.stat().st_size + len(blob) > _PACK_CAP_BYTES:
                    existing = sorted((root / PACK_DIR_NAME / kind).glob("pack-*.bin"))
                    indexes = []
                    for item in existing:
                        try:
                            indexes.append(int(item.stem.split("-")[1]))
                        except (IndexError, ValueError):
                            continue
                    nxt = (max(indexes) + 1) if indexes else 1
                    pack_rel = f"_packs/{kind}/pack-{nxt:05d}.bin"
                    pack_path = root / pack_rel
                    pack_path.parent.mkdir(parents=True, exist_ok=True)
                offset = pack_path.stat().st_size if pack_path.is_file() else 0
                try:
                    with pack_path.open("ab") as handle:
                        handle.write(blob)
                        handle.flush()
                        os.fsync(handle.fileno())
                    directory_fd = os.open(pack_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError as exc:
                    raise PITDataError(f"Bronze pack is not writable: {pack_rel}") from exc
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO members(rel_path, pack, offset, length, raw_size, sha256)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (rel_path, pack_rel, offset, len(blob), len(raw), digest),
                )
                connection.execute("COMMIT")
                packed += 1
                raw_bytes += len(raw)
                packed_bytes += len(blob)
                if remove:
                    if offsite is not None and rel_path not in offsite:
                        skipped_unverified += 1
                    else:
                        try:
                            file_path.unlink()
                            removed += 1
                        except OSError as exc:
                            raise PITDataError(f"Bronze payload cannot be removed: {rel_path}") from exc
    except (OSError, sqlite3.Error, pa.ArrowException) as exc:
        raise PITDataError("Bronze packing failed; loose files retained for uncommitted members") from exc
    finally:
        connection.close()
    return PackReport(
        packed=packed,
        skipped_unverified=skipped_unverified,
        raw_bytes=raw_bytes,
        packed_bytes=packed_bytes,
        removed=removed,
    )
