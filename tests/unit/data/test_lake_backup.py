from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from src.data import lake_backup as lb
from src.data.lake_backup import BackupError, BackupSource, LocalDirStore, RcloneStore, push, restore, verify

NOW = datetime(2026, 10, 10, tzinfo=UTC)


def _write(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _archives(remote: Path) -> list[Path]:
    return sorted(remote.glob("seg-*.tar.zst"))


def _replace_archive(remote: Path, members: dict[str, bytes]) -> None:
    """Swap the single archive for one holding ``members`` and make the manifest agree with it."""
    (archive,) = _archives(remote)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(pa.Codec("zstd").compress(buffer.getvalue(), asbytes=True))
    manifest = json.loads((remote / "MANIFEST.json").read_text())
    manifest["archives"][archive.name]["sha256"] = LocalDirStore(remote).sha256(archive.name)
    (remote / "MANIFEST.json").write_text(json.dumps(manifest))


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, remote, work = tmp_path / "root", tmp_path / "remote", tmp_path / "work"
    root.mkdir()
    return root, remote, work


def test_roundtrip_restores_identical_bytes_and_honours_excludes(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "kind/aa/payload.json", b'{"a": 1}' * 100)
    _write(root, "kind/aa/receipt.json", b"{}")
    _write(root, "db/catalog.sqlite3-wal", b"junk")
    source = BackupSource("s", root, exclude=("*-wal",))
    report = push(source, LocalDirStore(remote), work, now=NOW)
    assert (report.scanned, report.stored, report.referenced, report.archives) == (2, 2, 0, 1)
    assert report.raw_bytes == 802
    dest = tmp_path / "dest"
    out = restore(source, LocalDirStore(remote), dest, work)
    assert (out.restored, out.skipped, out.archives) == (2, 0, 1)
    assert _tree(dest) == {k: v for k, v in _tree(root).items() if not k.endswith("-wal")}
    assert verify(LocalDirStore(remote), work, "s") == []


def test_rerun_uploads_nothing_and_leaves_remote_state_untouched(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.json", b"x" * 10)
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    before = (remote / "MANIFEST.json").read_bytes()
    again = push(source, LocalDirStore(remote), work, now=datetime(2027, 1, 1, tzinfo=UTC))
    assert (again.stored, again.referenced, again.archives, again.stored_bytes) == (0, 0, 0, 0)
    assert (remote / "MANIFEST.json").read_bytes() == before


def test_identical_content_is_stored_once_across_and_within_runs(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "a/one.bin", b"same")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    _write(root, "b/copy_of_one.bin", b"same")
    _write(root, "c/new1.bin", b"fresh")
    _write(root, "c/new2.bin", b"fresh")
    report = push(source, LocalDirStore(remote), work, now=NOW)
    assert (report.stored, report.referenced, report.archives) == (1, 2, 1)
    dest = tmp_path / "dest"
    restore(source, LocalDirStore(remote), dest, work)
    assert _tree(dest) == _tree(root)


def test_changed_file_gets_a_new_version_unless_source_is_immutable(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "state.json", b"v1")
    push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    _write(root, "state.json", b"v2")
    frozen = push(BackupSource("s", root, immutable=True), LocalDirStore(remote), work, now=NOW)
    assert frozen.stored == 0
    assert push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW).stored == 1
    dest = tmp_path / "dest"
    restore(BackupSource("s", root), LocalDirStore(remote), dest, work)
    assert (dest / "state.json").read_bytes() == b"v2"


def test_sqlite_database_is_backed_up_as_a_consistent_snapshot(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    db = sqlite3.connect(root / "catalog.sqlite3")
    db.execute("create table t (v integer)")
    db.execute("insert into t values (42)")
    db.commit()
    db.close()
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    assert push(source, LocalDirStore(remote), work, now=NOW).stored == 0
    dest = tmp_path / "dest"
    restore(source, LocalDirStore(remote), dest, work)
    restored = sqlite3.connect(dest / "catalog.sqlite3")
    assert restored.execute("select v from t").fetchall() == [(42,)]
    restored.close()


def test_segment_cap_splits_archives_and_oversized_file_stands_alone(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.bin", b"a" * 6)
    _write(root, "b.bin", b"b" * 6)
    _write(root, "c.bin", b"c" * 50)
    report = push(BackupSource("s", root, segment_bytes=10), LocalDirStore(remote), work, now=NOW)
    assert report.archives == 3
    assert len(_archives(remote)) == 3


def test_dangling_symlink_is_skipped_not_fatal(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "real.json", b"x")
    (root / "ghost").symlink_to(tmp_path / "does_not_exist")
    report = push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    assert (report.scanned, report.stored) == (1, 1)


def test_restore_skips_matching_files_and_filters_by_pattern(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "keep/a.json", b"a")
    _write(root, "skip/b.json", b"b")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    dest = tmp_path / "dest"
    first = restore(source, LocalDirStore(remote), dest, work, patterns=("keep/*",))
    assert (first.restored, sorted(_tree(dest))) == (1, ["keep/a.json"])
    second = restore(source, LocalDirStore(remote), dest, work)
    assert (second.restored, second.skipped) == (1, 1)


def test_restore_fails_closed_on_missing_or_corrupt_archive(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "a.json", b"a")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    (archive,) = _archives(remote)
    good = archive.read_bytes()
    archive.write_bytes(good + b"x")
    assert verify(LocalDirStore(remote), work, "s") != []
    with pytest.raises(BackupError, match="manifest sha256"):
        restore(source, LocalDirStore(remote), tmp_path / "d1", work)
    archive.unlink()
    with pytest.raises(BackupError, match="missing from the remote"):
        restore(source, LocalDirStore(remote), tmp_path / "d2", work)


def test_restore_rejects_missing_member_and_wrong_member_bytes(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "a.json", b"a")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    _replace_archive(remote, {"other.json": b"zz"})
    with pytest.raises(BackupError, match=r"member a\.json is missing"):
        restore(source, LocalDirStore(remote), tmp_path / "d1", work)
    _replace_archive(remote, {"a.json": b"tampered"})
    with pytest.raises(BackupError, match="do not match the index"):
        restore(source, LocalDirStore(remote), tmp_path / "d2", work)


def test_restore_rejects_index_paths_that_escape_the_root(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    remote.mkdir()
    line = json.dumps({"path": "../evil", "sha256": "0" * 64, "size": 1, "archive": "seg-x.tar.zst", "member": "m"})
    (remote / "index.jsonl.zst").write_bytes(pa.Codec("zstd").compress(line.encode() + b"\n", asbytes=True))
    with pytest.raises(BackupError, match="escape"):
        restore(BackupSource("s", root), LocalDirStore(remote), tmp_path / "dest", work)
    assert verify(LocalDirStore(remote), work, "s") == ["seg-x.tar.zst: referenced by the index but absent from the manifest"]


def test_restore_cannot_overwrite_files_through_an_external_symlink(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "linked/payload.json", b"backup")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    outside = tmp_path / "outside"
    _write(outside, "payload.json", b"keep")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BackupError, match="escape"):
        restore(source, LocalDirStore(remote), dest, work)
    assert (outside / "payload.json").read_bytes() == b"keep"


def test_restore_rejects_archive_names_outside_workdir(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "payload.json", b"backup")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    index = remote / "index.jsonl.zst"
    with pa.OSFile(str(index), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as stream:
        row = json.loads(stream.read())
    row["archive"] = "../external.tar.zst"
    index.write_bytes(pa.Codec("zstd").compress(json.dumps(row).encode(), asbytes=True))
    with pytest.raises(BackupError, match="work directory"):
        restore(source, LocalDirStore(remote), tmp_path / "dest", work)


class _LyingStore(LocalDirStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.puts = 0

    def put(self, src: Path, name: str) -> None:
        self.puts += 1
        super().put(src, name)

    def sha256(self, name: str) -> str | None:
        real = super().sha256(name)
        return None if real is None else "f" * 64


def test_push_refuses_an_archive_the_remote_hashes_differently(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.json", b"a")
    store = _LyingStore(remote)
    with pytest.raises(BackupError, match="sha256 mismatch"):
        push(BackupSource("s", root), store, work, now=NOW)
    assert not (remote / "index.jsonl.zst").exists()


def test_push_reuses_an_identical_archive_left_by_an_interrupted_run(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.json", b"a")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    (remote / "index.jsonl.zst").unlink()
    (remote / "MANIFEST.json").unlink()
    uploaded: list[str] = []

    class Counting(LocalDirStore):
        def put(self, src: Path, name: str) -> None:
            uploaded.append(name)
            super().put(src, name)

    report = push(source, Counting(remote), work, now=NOW)
    assert report.archives == 1
    assert uploaded == ["index.jsonl.zst", "MANIFEST.json"]


def _fake_rclone(monkeypatch: pytest.MonkeyPatch, responses: dict[str, tuple[int, str, str]]) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        code, out, err = responses[cmd[1]]
        return subprocess.CompletedProcess(cmd, code, out, err)

    monkeypatch.setattr(lb.subprocess, "run", run)
    return calls


def test_rclone_store_reads_hash_and_transfers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    listing = json.dumps([{"Name": "seg-a.tar.zst", "Hashes": {"sha256": "ab"}}, {"Name": "nohash"}])
    calls = _fake_rclone(monkeypatch, {"lsjson": (0, listing, ""), "copyto": (0, "", "")})
    store = RcloneStore("gdrive:lake/s/", rclone_bin="rc")
    assert store.sha256("seg-a.tar.zst") == "ab"
    assert store.sha256("absent") is None
    with pytest.raises(BackupError, match="no sha256"):
        store.sha256("nohash")
    assert store.fetch("seg-a.tar.zst", tmp_path / "x") is True
    assert store.fetch("absent", tmp_path / "y") is False
    store.put(tmp_path / "x", "sub/seg-b.tar.zst")
    assert calls[-1] == ["rc", "copyto", str(tmp_path / "x"), "gdrive:lake/s/sub/seg-b.tar.zst"]
    assert calls[0][-1] == "gdrive:lake/s"


def test_rclone_store_treats_missing_directory_as_absent_and_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_rclone(monkeypatch, {"lsjson": (3, "", "directory not found")})
    assert RcloneStore("gdrive:lake/s").sha256("a") is None
    _fake_rclone(monkeypatch, {"lsjson": (1, "", "quota exceeded")})
    with pytest.raises(BackupError, match="lsjson failed"):
        RcloneStore("gdrive:lake/s").sha256("a")
    listing = json.dumps([{"Name": "a", "Hashes": {"sha256": "ab"}}])
    _fake_rclone(monkeypatch, {"lsjson": (0, listing, ""), "copyto": (1, "", "boom")})
    with pytest.raises(BackupError, match="download failed"):
        RcloneStore("gdrive:lake/s").fetch("a", tmp_path / "a")
    with pytest.raises(BackupError, match="upload failed"):
        RcloneStore("gdrive:lake/s").put(tmp_path / "a", "a")

