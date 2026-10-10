"""Content-addressed, deduplicated archive backup of raw data trees to an object store."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import logging
import os
import sqlite3
import subprocess
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

import pyarrow as pa

__all__ = [
    "BackupError",
    "BackupSource",
    "LocalDirStore",
    "PushReport",
    "RcloneStore",
    "RemoteStore",
    "RestoreReport",
    "push",
    "restore",
    "verify",
]

_LOG = logging.getLogger(__name__)

INDEX_NAME: Final = "index.jsonl.zst"
MANIFEST_NAME: Final = "MANIFEST.json"
_CHUNK: Final = 1 << 20
# Level 10 stores sampled Bronze payloads 38% smaller than level 1 at ~90 MB/s; 15 gains 5% more at 14 MB/s.
_ZSTD_LEVEL: Final = 10
_SQLITE_SUFFIX: Final = ".sqlite3"
_RCLONE_DIR_NOT_FOUND: Final = 3


class BackupError(RuntimeError):
    """The remote copy cannot be trusted or a requested file cannot be restored."""


@dataclass(frozen=True, slots=True)
class BackupSource:
    """One local tree and its remote namespace.

    immutable: paths never change once written, so a path already in the index is skipped without hashing.
    segment_bytes: raw-byte cap per archive; one archive is built in memory, so this bounds peak RAM.
    """

    name: str
    root: Path
    exclude: tuple[str, ...] = ()
    immutable: bool = False
    segment_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PushReport:
    scanned: int
    stored: int
    referenced: int
    archives: int
    raw_bytes: int
    stored_bytes: int


@dataclass(frozen=True, slots=True)
class RestoreReport:
    restored: int
    skipped: int
    archives: int


class RemoteStore(Protocol):
    """Flat object namespace; names are relative to the source prefix."""

    def fetch(self, name: str, dest: Path) -> bool:
        """Copy the object to ``dest``; False when it does not exist."""
        ...

    def put(self, src: Path, name: str) -> None: ...

    def sha256(self, name: str) -> str | None:
        """Remote content hash, or None when the object does not exist."""
        ...


class LocalDirStore:
    """Directory-backed store (external disk or tests)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def fetch(self, name: str, dest: Path) -> bool:
        src = self._root / name
        if not src.is_file():
            return False
        dest.write_bytes(src.read_bytes())
        return True

    def put(self, src: Path, name: str) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, target)

    def sha256(self, name: str) -> str | None:
        src = self._root / name
        return _file_sha256(src) if src.is_file() else None


class RcloneStore:
    """rclone remote prefix such as ``gdrive:quant-lake/live/k-stock-engine/lake/bronze``."""

    def __init__(self, remote: str, *, rclone_bin: str = "rclone", timeout_s: int = 3600) -> None:
        self._remote = remote.rstrip("/")
        self._bin = rclone_bin
        self._timeout = timeout_s

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [self._bin, *args], capture_output=True, text=True, timeout=self._timeout, check=False
        )

    def _stat(self, name: str) -> dict[str, Any] | None:
        parent, _, leaf = f"{self._remote}/{name}".rpartition("/")
        res = self._run("lsjson", "--files-only", "--hash", "--hash-type", "sha256", parent)
        if res.returncode == _RCLONE_DIR_NOT_FOUND:
            return None
        if res.returncode != 0:
            raise BackupError(f"rclone lsjson failed for {name}: {res.stderr.strip()[-300:]}")
        for row in json.loads(res.stdout or "[]"):
            if row.get("Name") == leaf:
                return dict(row)
        return None

    def fetch(self, name: str, dest: Path) -> bool:
        if self._stat(name) is None:
            return False
        res = self._run("copyto", f"{self._remote}/{name}", str(dest))
        if res.returncode != 0:
            raise BackupError(f"rclone download failed for {name}: {res.stderr.strip()[-300:]}")
        return True

    def put(self, src: Path, name: str) -> None:
        res = self._run("copyto", str(src), f"{self._remote}/{name}")
        if res.returncode != 0:
            raise BackupError(f"rclone upload failed for {name}: {res.stderr.strip()[-300:]}")

    def sha256(self, name: str) -> str | None:
        row = self._stat(name)
        if row is None:
            return None
        digest = (row.get("Hashes") or {}).get("sha256")
        if not digest:
            raise BackupError(f"remote reports no sha256 for {name}")
        return str(digest)


@dataclass(frozen=True, slots=True)
class _Entry:
    path: str
    sha256: str
    size: int
    archive: str
    member: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(source: BackupSource) -> Iterator[str]:
    """Relative paths of regular files (a dangling symlink has no bytes to back up and is skipped)."""
    for folder, dirs, files in os.walk(source.root):
        dirs.sort()
        for name in sorted(files):
            path = Path(folder) / name
            if not path.is_file():
                continue
            rel = path.relative_to(source.root).as_posix()
            if not any(fnmatch.fnmatch(rel, pattern) for pattern in source.exclude):
                yield rel


def _materialize(source: BackupSource, rel: str, workdir: Path) -> Path:
    """Path whose bytes are backed up; a live SQLite database is copied through the backup API first."""
    path = source.root / rel
    if not rel.endswith(_SQLITE_SUFFIX):
        return path
    snapshot = workdir / "sqlite" / hashlib.sha256(rel.encode()).hexdigest()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    dst = sqlite3.connect(snapshot)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return snapshot


def _load_index(store: RemoteStore, workdir: Path) -> list[_Entry]:
    local = workdir / INDEX_NAME
    if not store.fetch(INDEX_NAME, local):
        return []
    with pa.OSFile(str(local), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as stream:
        data = stream.read()
    return [_Entry(**json.loads(line)) for line in bytes(data).splitlines() if line]


def _load_manifest(store: RemoteStore, workdir: Path, source_name: str) -> dict[str, Any]:
    local = workdir / MANIFEST_NAME
    if not store.fetch(MANIFEST_NAME, local):
        return {"source": source_name, "archives": {}}
    manifest: dict[str, Any] = json.loads(local.read_text(encoding="utf-8"))
    return manifest


def _save_state(
    store: RemoteStore, workdir: Path, entries: Sequence[_Entry], manifest: dict[str, Any], now: datetime
) -> None:
    lines = b"".join(json.dumps(asdict(entry), sort_keys=True).encode() + b"\n" for entry in entries)
    index_path = workdir / INDEX_NAME
    index_path.write_bytes(pa.Codec("zstd", compression_level=_ZSTD_LEVEL).compress(lines, asbytes=True))
    latest = {entry.path: entry for entry in entries}
    manifest["updated_at"] = now.isoformat()
    manifest["files"] = len(latest)
    manifest["raw_bytes"] = sum(entry.size for entry in latest.values())
    manifest_path = workdir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8")
    store.put(index_path, INDEX_NAME)
    store.put(manifest_path, MANIFEST_NAME)


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    # Byte-identical archives for identical members make a rerun after a crash an upload no-op.
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def _write_archive(members: Sequence[tuple[str, Path]], dest: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for member, path in members:
            tar.add(str(path), arcname=member, recursive=False, filter=_normalized)
    dest.write_bytes(pa.Codec("zstd", compression_level=_ZSTD_LEVEL).compress(buffer.getbuffer(), asbytes=True))


def _upload_verified(store: RemoteStore, local: Path, name: str) -> str:
    digest = _file_sha256(local)
    remote = store.sha256(name)
    if remote is None:
        store.put(local, name)
        remote = store.sha256(name)
    if remote != digest:
        raise BackupError(f"remote sha256 mismatch for {name}: local {digest} remote {remote}")
    return digest


@dataclass(frozen=True, slots=True)
class _Pending:
    rel: str
    path: Path
    sha256: str
    size: int


def _segments(pending: Sequence[_Pending], cap: int) -> Iterator[list[_Pending]]:
    batch: list[_Pending] = []
    total = 0
    for item in pending:
        if batch and total + item.size > cap:
            yield batch
            batch, total = [], 0
        batch.append(item)
        total += item.size
    if batch:
        yield batch


def _scan(
    source: BackupSource, workdir: Path, entries: Sequence[_Entry]
) -> tuple[int, list[_Pending], list[_Entry], list[_Pending]]:
    """Scanned count, files to store, references to stored content, and in-run duplicates of files to store."""
    by_path = {entry.path: entry for entry in entries}
    by_hash = {entry.sha256: entry for entry in reversed(entries)}
    pending: list[_Pending] = []
    queued: set[str] = set()
    refs: list[_Entry] = []
    duplicates: list[_Pending] = []
    scanned = 0
    for rel in _iter_files(source):
        scanned += 1
        known = by_path.get(rel)
        if known is not None and source.immutable:
            continue
        path = _materialize(source, rel, workdir)
        item = _Pending(rel, path, _file_sha256(path), path.stat().st_size)
        if known is not None and known.sha256 == item.sha256:
            continue
        stored = by_hash.get(item.sha256)
        if stored is not None:
            refs.append(_Entry(rel, item.sha256, item.size, stored.archive, stored.member))
        elif item.sha256 in queued:
            duplicates.append(item)
        else:
            queued.add(item.sha256)
            pending.append(item)
    return scanned, pending, refs, duplicates


def push(source: BackupSource, store: RemoteStore, workdir: Path, *, now: datetime) -> PushReport:
    """Upload every file of ``source`` that the remote index does not hold yet.

    A file whose content already exists under another path is recorded as a reference and not uploaded again.
    Each archive is verified against the remote sha256 before the index that names it is written.

    Raises:
        BackupError: a remote archive hash differs from the local one, or a store operation fails.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    entries = _load_index(store, workdir)
    manifest = _load_manifest(store, workdir, source.name)
    scanned, pending, refs, duplicates = _scan(source, workdir, entries)
    stored_bytes = 0
    archives = 0
    placed: dict[str, _Entry] = {}
    for batch in _segments(pending, source.segment_bytes):
        key = hashlib.sha256("\n".join(item.sha256 for item in batch).encode()).hexdigest()[:16]
        name = f"seg-{key}.tar.zst"
        local = workdir / name
        _write_archive([(item.rel, item.path) for item in batch], local)
        digest = _upload_verified(store, local, name)
        size = local.stat().st_size
        local.unlink()
        manifest["archives"][name] = {
            "sha256": digest,
            "bytes": size,
            "files": len(batch),
            "raw_bytes": sum(item.size for item in batch),
        }
        for item in batch:
            placed[item.sha256] = _Entry(item.rel, item.sha256, item.size, name, item.rel)
        entries.extend(placed[item.sha256] for item in batch)
        stored_bytes += size
        archives += 1
        _LOG.info("[DATA] lake_backup source=%s archive=%s files=%d bytes=%d", source.name, name, len(batch), size)
    refs.extend(
        _Entry(item.rel, item.sha256, item.size, placed[item.sha256].archive, placed[item.sha256].member)
        for item in duplicates
    )
    entries.extend(refs)
    if pending or refs:
        _save_state(store, workdir, entries, manifest, now)
    return PushReport(
        scanned=scanned,
        stored=len(pending),
        referenced=len(refs),
        archives=archives,
        raw_bytes=sum(item.size for item in pending),
        stored_bytes=stored_bytes,
    )


def _wanted(entries: Sequence[_Entry], patterns: Sequence[str]) -> dict[str, _Entry]:
    latest = {entry.path: entry for entry in entries}
    if not patterns:
        return latest
    return {path: entry for path, entry in latest.items() if any(fnmatch.fnmatch(path, p) for p in patterns)}


def _restore_target(dest: Path, relative: str) -> Path:
    root = dest.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise BackupError(f"index path {relative!r} would escape the restore root")
    return target


def _extract(local: Path, archive: str, members: dict[str, list[_Entry]], dest: Path) -> int:
    written = 0
    found: set[str] = set()
    with (
        pa.OSFile(str(local), "rb") as raw,
        pa.CompressedInputStream(raw, "zstd") as stream,
        tarfile.open(fileobj=stream, mode="r|") as tar,
    ):
        for info in tar:
            wanted = members.get(info.name)
            handle = tar.extractfile(info) if wanted else None
            if wanted is None or handle is None:
                continue
            data = handle.read()
            found.add(info.name)
            for entry in wanted:
                if hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise BackupError(f"restored bytes of {entry.path} do not match the index sha256")
                target = _restore_target(dest, entry.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written += 1
    missing = sorted(set(members) - found)
    if missing:
        raise BackupError(f"member {missing[0]} is missing from {archive}")
    return written


def restore(
    source: BackupSource, store: RemoteStore, dest: Path, workdir: Path, *, patterns: Sequence[str] = ()
) -> RestoreReport:
    """Write the latest backed-up version of each selected path under ``dest``; matching files are left alone.

    Raises:
        BackupError: an archive hash differs from the manifest, a member is missing, a restored file's hash
            differs from the index, or an index path would escape ``dest``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    entries = _load_index(store, workdir)
    manifest = _load_manifest(store, workdir, source.name)
    by_archive: dict[str, dict[str, list[_Entry]]] = {}
    skipped = 0
    for path, entry in _wanted(entries, patterns).items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise BackupError(f"index path {path!r} would escape the restore root")
        target = _restore_target(dest, path)
        if target.is_file() and target.stat().st_size == entry.size and _file_sha256(target) == entry.sha256:
            skipped += 1
            continue
        by_archive.setdefault(entry.archive, {}).setdefault(entry.member, []).append(entry)
    restored = 0
    for archive, members in sorted(by_archive.items()):
        if Path(archive).name != archive or archive in (".", "..", ""):
            raise BackupError(f"archive name {archive!r} would escape the work directory")
        local = workdir / archive
        if not store.fetch(archive, local):
            raise BackupError(f"archive {archive} is missing from the remote")
        if _file_sha256(local) != manifest["archives"].get(archive, {}).get("sha256"):
            raise BackupError(f"archive {archive} does not match the manifest sha256")
        restored += _extract(local, archive, members, dest)
        local.unlink()
    return RestoreReport(restored=restored, skipped=skipped, archives=len(by_archive))


def verify(store: RemoteStore, workdir: Path, source_name: str) -> list[str]:
    """Problems found by comparing the remote archives with the manifest and the index; empty when sound."""
    workdir.mkdir(parents=True, exist_ok=True)
    entries = _load_index(store, workdir)
    manifest = _load_manifest(store, workdir, source_name)
    problems: list[str] = []
    for name, meta in sorted(manifest["archives"].items()):
        remote = store.sha256(name)
        if remote != meta["sha256"]:
            problems.append(f"{name}: remote sha256 {remote} != manifest {meta['sha256']}")
    orphans = {entry.archive for entry in entries} - set(manifest["archives"])
    problems.extend(f"{name}: referenced by the index but absent from the manifest" for name in sorted(orphans))
    return problems
