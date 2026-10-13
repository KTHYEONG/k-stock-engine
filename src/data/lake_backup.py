"""Content-addressed, deduplicated archive backup of raw data trees to an object store."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import mmap
import os
import shutil
import sqlite3
import subprocess
import tarfile
import threading
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
_INDEX_CHUNK: Final = 8 * 1024 * 1024
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
    optional: the root may legitimately be absent on a host; a push then reports the source as skipped.
    segment_bytes: raw-byte cap per archive.
    """

    name: str
    root: Path
    exclude: tuple[str, ...] = ()
    immutable: bool = False
    segment_bytes: int = 256 * 1024 * 1024
    optional: bool = False


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
        shutil.copyfile(src, dest)
        return True

    def put(self, src: Path, name: str) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(src, tmp)
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


_ENTRY_FIELDS: Final = frozenset({"path", "sha256", "size", "archive", "member"})


def _validate_index_line(line: bytes, lineno: int) -> None:
    try:
        obj = json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BackupError(f"index line {lineno} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict) or obj.keys() != _ENTRY_FIELDS:
        raise BackupError(f"index line {lineno} has invalid _Entry fields")
    if (
        not all(isinstance(obj[key], str) for key in ("path", "sha256", "archive", "member"))
        or type(obj["size"]) is not int
        or obj["size"] < 0
        or len(obj["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in obj["sha256"])
    ):
        raise BackupError(f"index line {lineno} has invalid _Entry values")


def _iter_index_lines(local: Path) -> Iterator[bytes]:
    """Yield non-empty JSONL lines of a zstd index file in file order without materialising the payload."""
    carry = bytearray()
    lineno = 0
    with pa.OSFile(str(local), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as stream:
        while True:
            chunk = stream.read(_INDEX_CHUNK)
            if not chunk:
                break
            carry.extend(chunk)
            del chunk
            start = 0
            while True:
                nl = carry.find(b"\n", start)
                if nl < 0:
                    break
                lineno += 1
                line = bytes(carry[start:nl])
                start = nl + 1
                if not line.strip():
                    continue
                _validate_index_line(line, lineno)
                yield line
            del carry[:start]
        if carry.strip():
            lineno += 1
            line = bytes(carry)
            _validate_index_line(line, lineno)
            yield line


def _load_index(store: RemoteStore, workdir: Path) -> list[_Entry]:
    local = workdir / INDEX_NAME
    if not store.fetch(INDEX_NAME, local):
        return []
    return [_Entry(**json.loads(line)) for line in _iter_index_lines(local)]


@dataclass(frozen=True, slots=True)
class _IndexSummary:
    """Compact view of an index: latest (sha256, size) per path, first location per sha256, totals."""

    by_path: dict[str, tuple[bytes, int]]
    hashes: set[bytes]
    index_local: Path | None
    files: int
    raw_bytes: int


def _summarize_index(store: RemoteStore, workdir: Path) -> _IndexSummary:
    """Download and stream the index once; an absent index yields an empty summary."""
    local = workdir / INDEX_NAME
    if not store.fetch(INDEX_NAME, local):
        return _IndexSummary({}, set(), None, 0, 0)
    by_path: dict[str, tuple[bytes, int]] = {}
    hashes: set[bytes] = set()
    for line in _iter_index_lines(local):
        obj = json.loads(line)
        digest = bytes.fromhex(obj["sha256"])
        hashes.add(digest)
        by_path[obj["path"]] = (digest, int(obj["size"]))
    return _IndexSummary(by_path, hashes, local, len(by_path), sum(size for _, size in by_path.values()))


def _resolve_first_locations(summary: _IndexSummary, wanted: set[bytes]) -> dict[bytes, tuple[str, str]]:
    """First (archive, member) per sha256 for the requested hashes via a second streaming pass."""
    found: dict[bytes, tuple[str, str]] = {}
    if not wanted:
        return found
    if summary.index_local is None:
        raise BackupError("index has no location for referenced sha256")
    for line in _iter_index_lines(summary.index_local):
        obj = json.loads(line)
        digest = bytes.fromhex(obj["sha256"])
        if digest in wanted and digest not in found:
            found[digest] = (obj["archive"], obj["member"])
            if len(found) == len(wanted):
                break
    missing = wanted - found.keys()
    if missing:
        raise BackupError(f"index has no location for {len(missing)} referenced sha256")
    return found


def _load_manifest(store: RemoteStore, workdir: Path, source_name: str) -> dict[str, Any]:
    local = workdir / MANIFEST_NAME
    if not store.fetch(MANIFEST_NAME, local):
        return {"source": source_name, "archives": {}}
    manifest: dict[str, Any] = json.loads(local.read_text(encoding="utf-8"))
    return manifest


def _save_state(
    store: RemoteStore,
    workdir: Path,
    summary: _IndexSummary,
    new_entries: Sequence[_Entry],
    manifest: dict[str, Any],
    now: datetime,
) -> None:
    index_path = workdir / INDEX_NAME
    tmp_raw = workdir / (INDEX_NAME + ".raw.tmp")
    last_byte = b"\n"
    try:
        with tmp_raw.open("wb") as out:
            if summary.index_local is not None:
                with (
                    pa.OSFile(str(summary.index_local), "rb") as raw,
                    pa.CompressedInputStream(raw, "zstd") as stream,
                ):
                    while chunk := stream.read(_INDEX_CHUNK):
                        out.write(chunk)
                        last_byte = chunk[-1:]
            if last_byte != b"\n":
                out.write(b"\n")
            for entry in new_entries:
                out.write(json.dumps(asdict(entry), sort_keys=True).encode() + b"\n")
        with (
            tmp_raw.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            _compress_mapped(mapped, index_path)
    finally:
        tmp_raw.unlink(missing_ok=True)
    latest: dict[str, int] = {}
    files, raw_bytes = summary.files, summary.raw_bytes
    for entry in new_entries:
        previous = latest.get(entry.path)
        if previous is None:
            known = summary.by_path.get(entry.path)
            if known is None:
                files += 1
                previous = 0
            else:
                previous = known[1]
        raw_bytes += entry.size - previous
        latest[entry.path] = entry.size
    manifest["updated_at"] = now.isoformat()
    manifest["files"] = files
    manifest["raw_bytes"] = raw_bytes
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


def _discard_mapped_pages(mapped: mmap.mmap, stop: threading.Event) -> None:
    while not stop.wait(0.02):
        # Read-only file pages can be faulted back without changing the codec's input bytes.
        mapped.madvise(mmap.MADV_DONTNEED)


def _compress_mapped(mapped: mmap.mmap, dest: Path) -> None:
    """Compress a read-only file map while bounding its resident input pages."""
    stop = threading.Event()
    reclaimer = threading.Thread(target=_discard_mapped_pages, args=(mapped, stop))
    reclaimer.start()
    try:
        compressed = pa.Codec("zstd", compression_level=_ZSTD_LEVEL).compress(mapped)
        with dest.open("wb") as out:
            out.write(compressed)
    finally:
        stop.set()
        reclaimer.join()


def _write_archive(members: Sequence[tuple[str, Path]], dest: Path) -> None:
    """Build a normalized tar on disk and compress its memory map."""
    tmp = dest.parent / (dest.name + ".part.tar")
    try:
        with tarfile.open(name=str(tmp), mode="w") as tar:
            for member, path in members:
                tar.add(str(path), arcname=member, recursive=False, filter=_normalized)
        with (
            tmp.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            _compress_mapped(mapped, dest)
    finally:
        tmp.unlink(missing_ok=True)


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
    source: BackupSource, workdir: Path, summary: _IndexSummary
) -> tuple[int, list[_Pending], list[_Pending], list[_Pending]]:
    """Return scanned count, pending files, stored-content references, and in-run duplicates."""
    pending: list[_Pending] = []
    queued: set[str] = set()
    refs: list[_Pending] = []
    duplicates: list[_Pending] = []
    scanned = 0
    for rel in _iter_files(source):
        scanned += 1
        known = summary.by_path.get(rel)
        if known is not None and source.immutable:
            continue
        path = _materialize(source, rel, workdir)
        item = _Pending(rel, path, _file_sha256(path), path.stat().st_size)
        if known is not None and known[0] == bytes.fromhex(item.sha256):
            continue
        if bytes.fromhex(item.sha256) in summary.hashes:
            refs.append(item)
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
    summary = _summarize_index(store, workdir)
    manifest = _load_manifest(store, workdir, source.name)
    scanned, pending, ref_pendings, duplicates = _scan(source, workdir, summary)
    changed_paths = {item.rel for item in (*pending, *ref_pendings, *duplicates)}
    summary = _IndexSummary(
        {path: value for path, value in summary.by_path.items() if path in changed_paths},
        set(),
        summary.index_local,
        summary.files,
        summary.raw_bytes,
    )
    stored_bytes = 0
    archives = 0
    placed: dict[str, _Entry] = {}
    placed_ordered: list[_Entry] = []
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
            entry = _Entry(item.rel, item.sha256, item.size, name, item.rel)
            placed[item.sha256] = entry
            placed_ordered.append(entry)
        stored_bytes += size
        archives += 1
        _LOG.info("[DATA] lake_backup source=%s archive=%s files=%d bytes=%d", source.name, name, len(batch), size)
    ref_entries: list[_Entry] = []
    if ref_pendings:
        locations = _resolve_first_locations(summary, {bytes.fromhex(item.sha256) for item in ref_pendings})
        for item in ref_pendings:
            archive, member = locations[bytes.fromhex(item.sha256)]
            ref_entries.append(_Entry(item.rel, item.sha256, item.size, archive, member))
    dup_entries: list[_Entry] = []
    for item in duplicates:
        loc = placed[item.sha256]
        dup_entries.append(_Entry(item.rel, item.sha256, item.size, loc.archive, loc.member))
    new_entries = [*placed_ordered, *ref_entries, *dup_entries]
    if new_entries:
        _save_state(store, workdir, summary, new_entries, manifest, now)
    return PushReport(
        scanned=scanned,
        stored=len(pending),
        referenced=len(ref_entries) + len(dup_entries),
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
