"""Packed Bronze payload store tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.payload_store import (
    PACK_DIR_NAME,
    PayloadMissingError,
    pack_loose,
    payload_exists,
    read_payload,
    release_payload_readers,
)


@pytest.fixture(autouse=True)
def _release_readers() -> Iterator[None]:
    release_payload_readers()
    yield
    release_payload_readers()


def _bronze(tmp_path: Path) -> Path:
    root = tmp_path / "bronze"
    (root / "catalog").mkdir(parents=True)
    return root


def _loose(root: Path, kind: str, name: str, data: bytes) -> Path:
    digest = hashlib.sha256(data).hexdigest()
    path = root / kind / digest[:16] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_packed_file_reads_back_identically(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    bodies = {f"payload-{idx}.json": b'{"row": %d, "pad": "%s"}' % (idx, b"x" * 500) for idx in range(3)}
    paths = {name: _loose(root, "daily_market", name, body) for name, body in bodies.items()}
    report = pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert report.packed == 3
    assert report.removed == 3
    for name, body in bodies.items():
        assert not paths[name].exists()
        assert read_payload(paths[name]) == body
        assert payload_exists(paths[name])


def test_removal_is_gated_by_offsite_confirmation(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    kept = _loose(root, "daily_market", "a.json", b"alpha" * 100)
    confirmed = _loose(root, "daily_market", "b.json", b"beta" * 100)
    rel_confirmed = confirmed.relative_to(root).as_posix()
    report = pack_loose(root, kinds=["daily_market"], offsite_paths={rel_confirmed}, remove=True)
    assert report.skipped_unverified == 1
    assert kept.exists()
    assert not confirmed.exists()
    assert read_payload(kept) == b"alpha" * 100
    assert read_payload(confirmed) == b"beta" * 100


def test_loose_file_wins_over_pack(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"original" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"replacement" * 100)
    assert read_payload(path) == b"replacement" * 100


def test_immutable_violation_fails_closed(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"original" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"tampered" * 100)
    with pytest.raises(PITDataError):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)


def test_crash_orphan_bytes_are_preserved_without_indexing(tmp_path: Path) -> None:
    import sqlite3

    root = _bronze(tmp_path)
    first = _loose(root, "daily_market", "first.json", b"first" * 200)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    packs = sorted((root / PACK_DIR_NAME / "daily_market").glob("pack-*.bin"))
    assert len(packs) == 1
    committed_end = packs[0].stat().st_size
    with packs[0].open("ab") as handle:
        handle.write(b"\x00" * 4096)
    second = _loose(root, "daily_market", "second.json", b"second" * 200)
    report = pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert read_payload(first) == b"first" * 200
    assert read_payload(second) == b"second" * 200
    assert report.packed == 1
    connection = sqlite3.connect(str(root / PACK_DIR_NAME / "index.sqlite3"))
    try:
        rows = connection.execute("SELECT pack, offset, length FROM members").fetchall()
    finally:
        connection.close()
    by_pack: dict[str, int] = {}
    for pack_rel, offset, length in rows:
        by_pack[str(pack_rel)] = max(by_pack.get(str(pack_rel), 0), int(offset) + int(length))
    for pack_rel, expected in by_pack.items():
        assert (root / pack_rel).stat().st_size == expected
    assert any(int(offset) >= committed_end + 4096 for _, offset, _ in rows)
    assert packs[0].read_bytes()[committed_end:committed_end + 4096] == b"\x00" * 4096


def test_missing_everywhere_raises(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    ghost = root / "daily_market" / ("0" * 64) / "payload.json"
    with pytest.raises(PayloadMissingError):
        read_payload(ghost)
    assert payload_exists(ghost) is False


def test_catalog_and_other_directories_untouched(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    catalog_db = root / "catalog" / "catalog.sqlite3"
    catalog_db.write_bytes(b"catalog-bytes" * 100)
    other = _loose(root, "security_master", "payload.json", b"other" * 100)
    target = _loose(root, "daily_market", "payload.json", b"target" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert catalog_db.read_bytes() == b"catalog-bytes" * 100
    assert other.read_bytes() == b"other" * 100
    assert read_payload(target) == b"target" * 100


def test_pack_skips_unlisted_unreadable_and_stray_files(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    good = _loose(root, "daily_market", "good.json", b"good" * 200)
    locked = _loose(root, "daily_market", "locked.json", b"locked" * 200)
    locked.chmod(0o000)
    stray_dir = root / "daily_market" / PACK_DIR_NAME
    stray_dir.mkdir(parents=True, exist_ok=True)
    (stray_dir / "stray.json").write_bytes(b"stray")
    catalog_db = root / "catalog" / "catalog.sqlite3"
    catalog_db.write_bytes(b"catalog")
    try:
        report = pack_loose(
            root, kinds=["catalog", "bad/kind", "", "daily_market"], offsite_paths=None, remove=False
        )
    finally:
        locked.chmod(0o644)
    assert report.packed == 1
    assert read_payload(good) == b"good" * 200
    assert catalog_db.read_bytes() == b"catalog"


def test_gated_rerun_counts_unverified_and_stuck_removal_fails(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    kept = _loose(root, "daily_market", "a.json", b"alpha" * 100)
    confirmed = _loose(root, "daily_market", "b.json", b"beta" * 100)
    rel_confirmed = confirmed.relative_to(root).as_posix()
    pack_loose(root, kinds=["daily_market"], offsite_paths={rel_confirmed}, remove=True)
    rerun = pack_loose(root, kinds=["daily_market"], offsite_paths={rel_confirmed}, remove=True)
    assert rerun.packed == 0
    assert rerun.skipped_unverified == 1
    assert read_payload(kept) == b"alpha" * 100
    kept.parent.chmod(0o555)
    try:
        with pytest.raises(PITDataError):
            pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    finally:
        kept.parent.chmod(0o755)
    fresh = _loose(root, "daily_market", "fresh.json", b"fresh" * 100)
    fresh.parent.chmod(0o555)
    try:
        with pytest.raises(PITDataError, match="cannot be removed"):
            pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    finally:
        fresh.parent.chmod(0o755)
    assert read_payload(fresh) == b"fresh" * 100


def test_pack_rotation_caps_pack_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.payload_store as store_module

    root = _bronze(tmp_path)
    bodies = [(f"payload-{idx}.json", bytes([idx]) * 300) for idx in range(8)]
    paths = {name: _loose(root, "daily_market", name, body) for name, body in bodies}
    foreign = root / PACK_DIR_NAME / "daily_market" / "pack-abc.bin"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"foreign")
    (root / PACK_DIR_NAME / "daily_market" / "pack-00002.bin").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_module, "_PACK_CAP_BYTES", 64)
    report = pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert report.packed == 8
    packs = sorted((root / PACK_DIR_NAME / "daily_market").glob("pack-*.bin"))
    assert len([item for item in packs if item.is_file()]) >= 2
    for name, body in bodies:
        assert read_payload(paths[name]) == body


def test_damaged_store_fails_closed(tmp_path: Path) -> None:
    import sqlite3

    root = _bronze(tmp_path)
    first = _loose(root, "daily_market", "first.json", b"first" * 200)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    ghost = root / "daily_market" / ("1" * 64) / "payload.json"
    with pytest.raises(PayloadMissingError):
        read_payload(ghost)
    assert payload_exists(ghost) is False
    outside = tmp_path / "elsewhere" / "payload.json"
    assert payload_exists(outside) is False
    connection = sqlite3.connect(str(root / PACK_DIR_NAME / "index.sqlite3"))
    try:
        row = connection.execute("SELECT pack, offset, length, raw_size FROM members").fetchone()
        pack_rel, offset, length = (str(row[0]), int(row[1]), int(row[2]))
    finally:
        connection.close()
    pack_bin = root / pack_rel
    pack_bin.unlink()
    with pytest.raises(PayloadMissingError):
        read_payload(first)
    pack_bin.parent.mkdir(parents=True, exist_ok=True)
    pack_bin.write_bytes(b"\x00" * (offset + length - 1))
    with pytest.raises(PayloadMissingError):
        read_payload(first)
    pack_bin.write_bytes(b"\xff" * (offset + length))
    with pytest.raises(PayloadMissingError):
        read_payload(first)


def test_corrupt_index_reads_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "bronze-broken"
    (root / "catalog").mkdir(parents=True)
    target = _loose(root, "daily_market", "payload.json", b"body" * 200)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    (root / PACK_DIR_NAME / "index.sqlite3").write_bytes(b"corrupt")
    with pytest.raises(PayloadMissingError):
        read_payload(target)
    assert payload_exists(target) is False


def test_rel_posix_rejects_paths_outside_the_root(tmp_path: Path) -> None:
    from src.data.payload_store import _rel_posix

    root = _bronze(tmp_path)
    with pytest.raises(PayloadMissingError):
        _rel_posix(root, tmp_path / "elsewhere" / "payload.json")


def test_unwritable_pack_file_keeps_orphan_for_next_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.payload_store as store_module

    monkeypatch.setattr(store_module, "_PACK_CAP_BYTES", 128)
    root = _bronze(tmp_path)
    first = _loose(root, "daily_market", "first.json", b"first" * 200)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    packs = sorted((root / PACK_DIR_NAME / "daily_market").glob("pack-*.bin"))
    assert len(packs) == 1
    with packs[0].open("ab") as handle:
        handle.write(b"\x00" * 4096)
    packs[0].chmod(0o444)
    second = _loose(root, "daily_market", "second.json", b"second" * 200)
    try:
        report = pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    finally:
        packs[0].chmod(0o644)
    assert report.packed == 1
    assert read_payload(first) == b"first" * 200
    assert read_payload(second) == b"second" * 200


def test_unwritable_pack_directory_fails_closed(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    _loose(root, "daily_market", "payload.json", b"body" * 200)
    pack_dir = root / PACK_DIR_NAME / "daily_market"
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.chmod(0o555)
    try:
        with pytest.raises(PITDataError):
            pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    finally:
        pack_dir.chmod(0o755)


def test_existing_loose_file_outside_bronze_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_bytes(b"outside")
    with pytest.raises(PayloadMissingError):
        read_payload(path)
    assert not payload_exists(path)


def test_corrupt_indexed_member_never_removes_the_remaining_loose_copy(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"original" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    pack = next((root / PACK_DIR_NAME / "daily_market").glob("pack-*.bin"))
    pack.write_bytes(b"\xff" * pack.stat().st_size)
    with pytest.raises(PITDataError):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert path.read_bytes() == b"original" * 100
    path.unlink()
    assert not payload_exists(path)


def test_kind_traversal_and_symlinks_cannot_pack_catalog_or_external_files(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    catalog = root / "catalog" / "catalog.sqlite3"
    catalog.write_bytes(b"catalog")
    outside = tmp_path / "external.json"
    outside.write_bytes(b"outside")
    report = pack_loose(root, kinds=[".", "..", "catalog"], offsite_paths=None, remove=True)
    assert report.packed == 0
    assert catalog.read_bytes() == b"catalog"
    directory = root / "daily_market"
    directory.mkdir()
    (directory / "external.json").symlink_to(outside)
    with pytest.raises(PITDataError):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert outside.read_bytes() == b"outside"


def test_oversized_member_stays_loose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.payload_store as store_module

    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"large")
    monkeypatch.setattr(store_module, "_PACK_CAP_BYTES", 1)
    with pytest.raises(PITDataError, match="capacity"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert path.read_bytes() == b"large"


def _pack_in_process(root: Path) -> int:
    return pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True).packed


def test_concurrent_packers_preserve_every_member(tmp_path: Path) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    root = _bronze(tmp_path)
    paths = [_loose(root, "daily_market", f"{i}.json", str(i).encode() * 50) for i in range(16)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert sum(pool.map(_pack_in_process, [root, root])) == len(paths)
    for i, path in enumerate(paths):
        assert not path.exists()
        assert read_payload(path) == str(i).encode() * 50


def test_invalid_index_location_and_truncated_pack_fail_closed(tmp_path: Path) -> None:
    import sqlite3

    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "first.json", b"first" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    db = root / PACK_DIR_NAME / "index.sqlite3"
    with sqlite3.connect(db) as connection:
        original = connection.execute("SELECT pack FROM members").fetchone()[0]
        connection.execute("UPDATE members SET pack = '../external.bin'")
    assert not payload_exists(path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE members SET pack = ?", (original,))
    (root / original).write_bytes(b"")
    fresh = _loose(root, "daily_market", "second.json", b"second" * 100)
    with pytest.raises(PITDataError, match="truncated"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert fresh.read_bytes() == b"second" * 100


def test_valid_but_wrong_packed_bytes_do_not_authorize_removal(tmp_path: Path) -> None:
    import sqlite3

    import pyarrow as pa

    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"original" * 100)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    altered = bytes(pa.Codec("zstd", compression_level=10).compress(b"altered" * 100))
    with sqlite3.connect(root / PACK_DIR_NAME / "index.sqlite3") as connection:
        pack = connection.execute("SELECT pack FROM members").fetchone()[0]
        connection.execute("UPDATE members SET offset=0, length=?, raw_size=?", (len(altered), 700))
    (root / pack).write_bytes(altered)
    with pytest.raises(PITDataError, match="hash mismatch"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert path.read_bytes() == b"original" * 100


def test_index_commit_failure_preserves_loose_copy_and_orphan_bytes(tmp_path: Path) -> None:
    import sqlite3

    root = _bronze(tmp_path)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=False)
    db = root / PACK_DIR_NAME / "index.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TRIGGER block_insert BEFORE INSERT ON members BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    path = _loose(root, "daily_market", "payload.json", b"body" * 100)
    with pytest.raises(PITDataError, match="uncommitted"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert read_payload(path) == b"body" * 100
    pack = next((root / PACK_DIR_NAME / "daily_market").glob("pack-*.bin"))
    orphan = pack.read_bytes()
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0
        connection.execute("DROP TRIGGER block_insert")
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert read_payload(path) == b"body" * 100
    assert pack.read_bytes() == orphan


def test_packing_requires_a_catalog_root(tmp_path: Path) -> None:
    with pytest.raises(PITDataError, match="catalog"):
        pack_loose(tmp_path, kinds=["daily_market"], offsite_paths=None, remove=True)


def test_symlink_pack_directories_cannot_receive_external_writes(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    _loose(root, "daily_market", "payload.json", b"body")
    outside = tmp_path / "outside"
    outside.mkdir()
    packs = root / PACK_DIR_NAME
    packs.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PITDataError, match="symlink"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    packs.unlink()
    packs.mkdir()
    (packs / "daily_market").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PITDataError, match="symlink"):
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert list(outside.iterdir()) == []


def test_malformed_index_offsets_are_not_readable(tmp_path: Path) -> None:
    import sqlite3

    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"body")
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    with sqlite3.connect(root / PACK_DIR_NAME / "index.sqlite3") as connection:
        connection.execute("UPDATE members SET offset='invalid'")
    assert not payload_exists(path)


def test_index_replaced_mid_process_finds_new_member(tmp_path: Path) -> None:
    import sqlite3

    import src.data.payload_store as store_module

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        first = _loose(root, "daily_market", "first.json", b"first" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        assert read_payload(first) == b"first" * 200
        db_path = root / PACK_DIR_NAME / "index.sqlite3"
        readers = store_module._reader_cache()
        old_conn, _, _ = readers[str(db_path)]
        second = _loose(root, "daily_market", "second.json", b"second" * 200)
        db_path.unlink()
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        assert read_payload(second) == b"second" * 200
        with pytest.raises(sqlite3.ProgrammingError):
            old_conn.execute("SELECT 1")
    finally:
        release_payload_readers()


def test_concurrent_writer_visible_without_release(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import src.data.payload_store as store_module

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        first = _loose(root, "daily_market", "first.json", b"first" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        assert read_payload(first) == b"first" * 200
        second = _loose(root, "daily_market", "second.json", b"second" * 200)
        second.unlink()
        with pytest.raises(PayloadMissingError, match="Bronze payload is missing"):
            read_payload(second)
        reader = store_module._reader_cache()[str(root / PACK_DIR_NAME / "index.sqlite3")][0]
        assert not reader.in_transaction
        second.write_bytes(b"second" * 200)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_pack_in_process, root).result()
        assert read_payload(second) == b"second" * 200
        assert store_module._reader_cache()[str(root / PACK_DIR_NAME / "index.sqlite3")][0] is reader
        assert not reader.in_transaction
    finally:
        release_payload_readers()


def test_one_connection_per_thread_and_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3 as sqlite3_module
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock, get_ident

    import src.data.payload_store as store_module

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        path = _loose(root, "daily_market", "payload.json", b"body" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        release_payload_readers()
        real_connect = sqlite3_module.connect
        calls: list[int] = []
        lock = Lock()
        barrier = Barrier(3)

        def _counted(*args: object, **kwargs: object) -> sqlite3_module.Connection:
            with lock:
                calls.append(get_ident())
            return real_connect(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(store_module.sqlite3, "connect", _counted)

        def _read_many(_: int) -> bytes:
            barrier.wait(timeout=10)
            try:
                for _ in range(5):
                    assert read_payload(path) == b"body" * 200
                return b"ok"
            finally:
                release_payload_readers()

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_read_many, range(3)))
        assert results == [b"ok", b"ok", b"ok"]
        assert len(calls) == len(set(calls)) == 3
    finally:
        release_payload_readers()


def test_outside_root_error_preserved(tmp_path: Path) -> None:
    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        _loose(root, "daily_market", "payload.json", b"body")
        outside = tmp_path / "elsewhere" / "payload.json"
        with pytest.raises(PayloadMissingError) as first_exc:
            read_payload(outside)
        with pytest.raises(PayloadMissingError) as second_exc:
            read_payload(outside)
        expected = f"Bronze file is outside any Bronze root: {outside}"
        assert str(first_exc.value) == str(second_exc.value) == expected
    finally:
        release_payload_readers()


def test_release_reclaims_connections(tmp_path: Path) -> None:
    import sqlite3

    import src.data.payload_store as store_module

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        path = _loose(root, "daily_market", "payload.json", b"body" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        assert read_payload(path) == b"body" * 200
        assert len(store_module._reader_cache()) >= 1
        old_connection = store_module._reader_cache()[str(root / PACK_DIR_NAME / "index.sqlite3")][0]
        release_payload_readers()
        assert len(store_module._reader_cache()) == 0
        assert not store_module._root_cache()
        assert not store_module._pack_path_cache()
        with pytest.raises(sqlite3.ProgrammingError):
            old_connection.execute("SELECT 1")
        assert read_payload(path) == b"body" * 200
    finally:
        release_payload_readers()


def test_bronze_root_for_caches_per_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.payload_store import _bronze_root_for

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        path = _loose(root, "daily_market", "payload.json", b"body")
        assert _bronze_root_for(path) == root.resolve()
        with monkeypatch.context() as patch:
            def _unexpected_resolve(self: Path, strict: bool = False) -> Path:
                pytest.fail("cached root discovery must not resolve paths")

            patch.setattr(Path, "resolve", _unexpected_resolve)
            assert _bronze_root_for(path) == root
    finally:
        release_payload_readers()


def test_lookup_with_directory_index_fails_closed(tmp_path: Path) -> None:
    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        path = _loose(root, "daily_market", "payload.json", b"body" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        assert read_payload(path) == b"body" * 200
        release_payload_readers()
        (root / PACK_DIR_NAME / "index.sqlite3").unlink()
        (root / PACK_DIR_NAME / "index.sqlite3").mkdir()
        with pytest.raises(PayloadMissingError):
            read_payload(path)
    finally:
        release_payload_readers()


def test_lookup_connect_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3 as sqlite3_module

    import src.data.payload_store as store_module

    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        path = _loose(root, "daily_market", "payload.json", b"body" * 200)
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        release_payload_readers()

        def _broken(*args: object, **kwargs: object) -> sqlite3_module.Connection:
            raise sqlite3_module.OperationalError("locked")

        monkeypatch.setattr(store_module.sqlite3, "connect", _broken)
        with pytest.raises(PayloadMissingError):
            read_payload(path)
    finally:
        release_payload_readers()


@pytest.mark.parametrize("warm_cache", [False, True])
def test_symlinked_payload_outside_root_rejected(tmp_path: Path, warm_cache: bool) -> None:
    release_payload_readers()
    try:
        root = _bronze(tmp_path)
        sibling = _loose(root, "daily_market", "sibling.json", b"sibling" * 100)
        if warm_cache:
            assert read_payload(sibling) == b"sibling" * 100
        outside = tmp_path / "outside.json"
        outside.write_bytes(b"outside")
        link = sibling.parent / "link.json"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(outside)
        with pytest.raises(PayloadMissingError, match="outside its Bronze root"):
            read_payload(link)
    finally:
        release_payload_readers()


def test_relative_paths_follow_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = []
    for name in ("first", "second"):
        base = tmp_path / name
        root = _bronze(base)
        path = root / "daily_market" / "payload.json"
        path.parent.mkdir()
        path.write_bytes(name.encode())
        pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
        roots.append(base)
    for base in roots:
        monkeypatch.chdir(base)
        assert read_payload(Path("bronze/daily_market/payload.json")) == base.name.encode()


def test_symlinked_loose_payload_uses_its_parent_bronze_root(tmp_path: Path) -> None:
    root = _bronze(tmp_path)
    target = root / "target.json"
    target.write_bytes(b"target")
    link = root / "daily_market" / "payload.json"
    link.parent.mkdir()
    link.symlink_to(target)
    assert read_payload(link) == b"target"


def test_warm_payload_read_resolves_candidate_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _bronze(tmp_path)
    path = _loose(root, "daily_market", "payload.json", b"body" * 200)
    pack_loose(root, kinds=["daily_market"], offsite_paths=None, remove=True)
    assert read_payload(path) == b"body" * 200
    calls: list[Path] = []
    real_resolve = Path.resolve

    def _counted(self: Path, strict: bool = False) -> Path:
        calls.append(self)
        return real_resolve(self, strict=strict)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", _counted)
        assert read_payload(path) == b"body" * 200
    assert calls == [path]
