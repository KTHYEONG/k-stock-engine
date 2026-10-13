from __future__ import annotations

import hashlib
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


def _golden_tree(root: Path) -> None:
    _write(root, "a1.bin", b"A" * 20)
    _write(root, "a2.bin", b"B" * 20)
    _write(root, "a3.bin", b"C" * 20)
    _write(root, "dup_copy.bin", b"A" * 20)
    _write(root, "sub/c.bin", b"D" * 20)
    db = sqlite3.connect(root / "catalog.sqlite3")
    db.execute("create table t (v integer)")
    db.execute("insert into t values (42)")
    db.commit()
    db.close()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_push_outputs_are_byte_identical_to_recorded_golden_digests(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _golden_tree(root)
    push(BackupSource("golden", root, segment_bytes=40), LocalDirStore(remote), work, now=NOW)
    assert _sha(remote / "seg-0b6c11a1d219df95.tar.zst") == "055e152eb0df57a85e2834db3940b4139aa97914e126304780cd34d93509015c"
    assert _sha(remote / "seg-8d0b2acde74edb2a.tar.zst") == "42c06b7e500d4fe783e113fda0047c7d710da5a56767cdd912212648fdb7a0d9"
    assert _sha(remote / "seg-a9060aca05adc2d9.tar.zst") == "6df185e68d67ddd69c7b2edc20668590c96a8f66f1a7eda8e175f93a937e10ea"
    assert _sha(remote / "seg-e53b9d1e47629f2f.tar.zst") == "8381ef06c12a00c6dd3413b6343afbbdba5263fd4662df28751d97b0f7c8bfc1"
    assert _sha(remote / "index.jsonl.zst") == "fe01668bea5a21ca6eb6c4ffb39119ec3b21f01bc735cb8cc74402c53737a441"
    assert _sha(remote / "MANIFEST.json") == "48e7e419e177dd7c65b540321a5fc6328cd666d8c1b9d0a0c8b5d04d8d27798c"


def test_second_push_after_new_files_matches_golden_incremental_digests(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _golden_tree(root)
    source = BackupSource("golden", root, segment_bytes=40)
    push(source, LocalDirStore(remote), work, now=NOW)
    before = {p.name: p.read_bytes() for p in remote.glob("seg-*.tar.zst")}
    _write(root, "new1.bin", b"E" * 20)
    _write(root, "new2.bin", b"F" * 20)
    _write(root, "new3.bin", b"G" * 20)
    uploaded: list[str] = []

    class Counting(LocalDirStore):
        def put(self, src: Path, name: str) -> None:
            uploaded.append(name)
            super().put(src, name)

    push(source, Counting(remote), work, now=datetime(2026, 10, 11, tzinfo=UTC))
    assert set(before).isdisjoint(uploaded)
    assert _sha(remote / "index.jsonl.zst") == "1a2f85139865b74e47bd98e81a6d5ffdec0de937fe88e3c632d1ea631134426c"
    assert _sha(remote / "MANIFEST.json") == "254d61cfcd4d340a3d7d8575a766555b8287587db2d3e9ba8719b1f1cbfe4365"
    for name, data in before.items():
        assert (remote / name).read_bytes() == data


def test_index_append_equals_full_reserialisation(env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    root, remote, work = env
    _write(root, "a/one.bin", b"same")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    with pa.OSFile(str(remote / "index.jsonl.zst"), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as s:
        old_raw = s.read()
    _write(root, "b/copy_of_one.bin", b"same")
    _write(root, "c/new1.bin", b"fresh")
    _write(root, "c/new2.bin", b"fresh")
    push(source, LocalDirStore(remote), work, now=NOW)
    with pa.OSFile(str(remote / "index.jsonl.zst"), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as s:
        new_raw = bytes(s.read())
    assert new_raw.startswith(bytes(old_raw))
    tail = [json.loads(line) for line in new_raw[len(old_raw):].splitlines()]
    assert [row["path"] for row in tail] == ["c/new1.bin", "b/copy_of_one.bin", "c/new2.bin"]
    assert tail[0]["archive"] == tail[2]["archive"] != tail[1]["archive"]
    old = json.loads(old_raw)
    digest = hashlib.sha256(b"fresh").hexdigest()
    archive = f"seg-{hashlib.sha256(digest.encode()).hexdigest()[:16]}.tar.zst"
    expected = [
        {"path": "c/new1.bin", "sha256": digest, "size": 5, "archive": archive, "member": "c/new1.bin"},
        {**old, "path": "b/copy_of_one.bin"},
        {"path": "c/new2.bin", "sha256": digest, "size": 5, "archive": archive, "member": "c/new1.bin"},
    ]
    assert new_raw == bytes(old_raw) + b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in expected)


def test_iter_index_lines_handles_chunk_boundaries(tmp_path: Path) -> None:
    pad = "p" * lb._INDEX_CHUNK
    rows = [
        {"path": pad, "sha256": "aa" * 32, "size": 1, "archive": "seg-x.tar.zst", "member": "m"},
        {"path": "b", "sha256": "bb" * 32, "size": 2, "archive": "seg-x.tar.zst", "member": "b"},
        {"path": "c", "sha256": "cc" * 32, "size": 3, "archive": "seg-y.tar.zst", "member": "c"},
    ]
    payload = b"\n\n" + json.dumps(rows[0]).encode() + b"\n" + json.dumps(rows[1]).encode() + b"\n\n" + json.dumps(rows[2]).encode()
    local = tmp_path / "index.jsonl.zst"
    local.write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
    assert [json.loads(line)["path"] for line in lb._iter_index_lines(local)] == [pad, "b", "c"]


def test_corrupt_index_line_fails_closed_with_line_number(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    remote.mkdir(exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    good = {"path": "a", "sha256": "aa" * 32, "size": 1, "archive": "s.tar.zst", "member": "a"}
    payload = json.dumps(good).encode() + b"\n" + json.dumps(good).encode() + b"\nnot-json\n"
    (remote / "index.jsonl.zst").write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
    with pytest.raises(BackupError, match="line 3"):
        lb._summarize_index(LocalDirStore(remote), work)
    with pytest.raises(BackupError, match="line 3"):
        lb._load_index(LocalDirStore(remote), work)
    _write(root, "new.bin", b"new")
    before = _tree(remote)
    with pytest.raises(BackupError, match="line 3"):
        push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    assert _tree(remote) == before
    assert list(remote.glob("seg-*.tar.zst")) == []


def test_summary_latest_by_path_and_first_by_hash_match_legacy(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    remote.mkdir(exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rows = [
        {"path": "p", "sha256": "aa" * 32, "size": 10, "archive": "seg-1.tar.zst", "member": "p"},
        {"path": "q", "sha256": "aa" * 32, "size": 10, "archive": "seg-later.tar.zst", "member": "q"},
        {"path": "p", "sha256": "bb" * 32, "size": 20, "archive": "seg-2.tar.zst", "member": "p"},
    ]
    payload = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)
    (remote / "index.jsonl.zst").write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
    summary = lb._summarize_index(LocalDirStore(remote), work)
    assert summary.by_path["p"][0] == bytes.fromhex("bb" * 32)
    assert summary.files == 2
    assert summary.raw_bytes == 30
    assert all(isinstance(k, bytes) for k in summary.hashes)
    assert all(isinstance(v, tuple) and isinstance(v[0], bytes) for v in summary.by_path.values())
    locations = lb._resolve_first_locations(summary, {bytes.fromhex("aa" * 32)})
    assert locations[bytes.fromhex("aa" * 32)] == ("seg-1.tar.zst", "p")


def test_reference_resolves_to_first_stored_location(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a/one.bin", b"same")
    source = BackupSource("s", root)
    push(source, LocalDirStore(remote), work, now=NOW)
    (first,) = _archives(remote)
    _write(root, "b/copy.bin", b"same")
    report = push(source, LocalDirStore(remote), work, now=NOW)
    assert (report.stored, report.archives) == (0, 0)
    entries = lb._load_index(LocalDirStore(remote), work)
    copy = next(e for e in entries if e.path == "b/copy.bin")
    assert (copy.archive, copy.member) == (first.name, "a/one.bin")


def test_archive_temp_tar_is_removed_on_success_and_failure(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.bin", b"data")
    dest = work / "seg-test.tar.zst"
    work.mkdir(parents=True, exist_ok=True)
    lb._write_archive([("a.bin", root / "a.bin")], dest)
    assert dest.is_file()
    assert list(work.glob("*.tar")) == []
    with pytest.raises(OSError, match="does-not-exist"):
        lb._write_archive([("m", root / "does-not-exist")], dest)
    assert list(work.glob("*.tar")) == []


def test_archive_codec_failure_removes_temporary_tar(
    env: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, remote, work = env
    work.mkdir()
    _write(root, "a.bin", b"data")

    class FailingCodec:
        def compress(self, mapped: object) -> None:
            raise RuntimeError("codec failure")

    monkeypatch.setattr(lb.pa, "Codec", lambda *args, **kwargs: FailingCodec())
    with pytest.raises(RuntimeError, match="codec failure"):
        lb._write_archive([("a.bin", root / "a.bin")], work / "seg-test.tar.zst")
    assert list(work.glob("*.tar")) == []


def test_archive_bytes_independent_of_member_buffering(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.bin", hashlib.shake_256(b"archive fixture").digest(16 * 1024 * 1024))
    _write(root, "b.bin", b"b" * 100)
    dest = work / "seg-test.tar.zst"
    work.mkdir(parents=True, exist_ok=True)
    lb._write_archive([("a.bin", root / "a.bin"), ("b.bin", root / "b.bin")], dest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in ("a.bin", "b.bin"):
            tar.add(str(root / name), arcname=name, recursive=False, filter=lb._normalized)
    expected = pa.Codec("zstd", compression_level=lb._ZSTD_LEVEL).compress(buffer.getbuffer(), asbytes=True)
    assert dest.read_bytes() == expected


def test_push_memory_scales_with_paths_not_with_payload(env: tuple[Path, Path, Path]) -> None:
    import tracemalloc

    root, remote, work = env
    remote.mkdir(exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {"path": f"f{i:06d}", "sha256": f"{i:064x}"[-64:], "size": i, "archive": "seg-x.tar.zst", "member": f"f{i:06d}"},
            sort_keys=True,
        ).encode()
        + b"\n"
        for i in range(100_000)
    ]
    (remote / "index.jsonl.zst").write_bytes(pa.Codec("zstd").compress(b"".join(rows), asbytes=True))
    tracemalloc.start()
    try:
        lb._summarize_index(LocalDirStore(remote), work)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 60 * 1024 * 1024


def test_malformed_index_lines_fail_closed_with_line_number(tmp_path: Path) -> None:
    good = {"path": "a", "sha256": "aa" * 32, "size": 1, "archive": "s.tar.zst", "member": "a"}
    line = json.dumps(good).encode()
    missing = json.dumps({"path": "a", "size": 1}).encode()
    cases = [
        line + b"\n" + missing + b"\n",
        line + b"\n" + b"oops",
        missing,
    ]
    for n, payload in enumerate(cases):
        local = tmp_path / f"bad{n}.jsonl.zst"
        local.write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
        with pytest.raises(BackupError, match="line 2" if n == 0 else "line 1"):
            list(lb._iter_index_lines(local))


def test_resolve_first_locations_empty_and_missing(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.bin", b"x")
    push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    summary = lb._summarize_index(LocalDirStore(remote), work)
    assert lb._resolve_first_locations(summary, set()) == {}
    with pytest.raises(BackupError, match="no location"):
        lb._resolve_first_locations(summary, {b"\x00" * 32})


@pytest.mark.parametrize("suffix", [b"", b"\n\n", b"\n  "])
def test_push_appends_to_unterminated_index(env: tuple[Path, Path, Path], suffix: bytes) -> None:
    root, remote, work = env
    _write(root, "a.bin", b"old")
    source = BackupSource("s", root)
    store = LocalDirStore(remote)
    push(source, store, work, now=NOW)
    index = remote / lb.INDEX_NAME
    with pa.input_stream(index, compression="zstd") as stream:
        old = stream.read().rstrip(b"\n") + suffix
    index.write_bytes(pa.Codec("zstd").compress(old, asbytes=True))
    _write(root, "b.bin", b"new")
    push(source, store, work, now=NOW)
    with pa.input_stream(index, compression="zstd") as stream:
        saved = stream.read()
    assert saved.startswith(old)
    assert [e.path for e in lb._load_index(store, work)] == ["a.bin", "b.bin"]
    assert verify(store, work, "s") == []


@pytest.mark.parametrize("payload", [b"\n  ", b""])
def test_empty_index_bytes_are_preserved(env: tuple[Path, Path, Path], payload: bytes) -> None:
    root, remote, work = env
    remote.mkdir()
    index = remote / lb.INDEX_NAME
    index.write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
    _write(root, "a.bin", b"new")
    push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    with pa.input_stream(index, compression="zstd") as stream:
        assert stream.read().startswith(payload)


def test_absent_index_does_not_reuse_stale_local_copy(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    work.mkdir()
    (work / lb.INDEX_NAME).write_bytes(b"stale")
    store = LocalDirStore(remote)
    summary = lb._summarize_index(store, work)
    with pytest.raises(BackupError, match="no location"):
        lb._resolve_first_locations(summary, {b"\x00" * 32})
    _write(root, "a.bin", b"new")
    push(BackupSource("s", root), store, work, now=NOW)
    assert [e.path for e in lb._load_index(store, work)] == ["a.bin"]


@pytest.mark.parametrize("change", [{"sha256": "zz" * 32}, {"size": "1"}, {"size": -1}, {"extra": 1}, {"path": []}])
def test_invalid_index_values_fail_before_upload(env: tuple[Path, Path, Path], change: dict[str, object]) -> None:
    root, remote, work = env
    remote.mkdir()
    row = {"path": "a", "sha256": "aa" * 32, "size": 1, "archive": "s.tar.zst", "member": "a"}
    row.update(change)
    payload = b"\n" + json.dumps(row).encode()
    (remote / lb.INDEX_NAME).write_bytes(pa.Codec("zstd").compress(payload, asbytes=True))
    _write(root, "new.bin", b"new")
    with pytest.raises(BackupError, match="line 2"):
        push(BackupSource("s", root), LocalDirStore(remote), work, now=NOW)
    with pytest.raises(BackupError, match="line 2"):
        lb._load_index(LocalDirStore(remote), work)
    assert set(_tree(remote)) == {lb.INDEX_NAME}


def test_manifest_totals_follow_latest_path_with_zero_size(env: tuple[Path, Path, Path]) -> None:
    root, remote, work = env
    _write(root, "a.bin", b"old")
    source, store = BackupSource("s", root), LocalDirStore(remote)
    push(source, store, work, now=NOW)
    summary = lb._summarize_index(store, work)
    entry = lb._load_index(store, work)[0]
    entries = [
        lb._Entry("a.bin", entry.sha256, 0, entry.archive, entry.member),
        lb._Entry("a.bin", entry.sha256, 7, entry.archive, entry.member),
        lb._Entry("b.bin", entry.sha256, 0, entry.archive, entry.member),
        lb._Entry("b.bin", entry.sha256, 2, entry.archive, entry.member),
    ]
    manifest = lb._load_manifest(store, work, "s")
    lb._save_state(store, work, summary, entries, manifest, NOW)
    assert manifest["files"] == 2
    assert manifest["raw_bytes"] == 9
