"""Invariant scenarios for the derived fact-page metadata cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
RCEPT = "20240315001111"
CORP = "00126380"


def _write_page(directory: Path, name: str, page: dict) -> tuple[Path, str]:
    path = directory / f"{name}.json"
    raw = json.dumps(page, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _entry(natural_key: str, path: Path, content_hash: str):
    from src.data.receipt_catalog import EvidenceStatus, ReceiptIndexEntry

    return ReceiptIndexEntry(
        source="financial_facts",
        natural_key=natural_key,
        as_of=None,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        content_hash=content_hash,
        retrieved_at=NOW,
        payload_path=path,
    )


def _identity(**overrides):
    base = {
        "corp_code": CORP,
        "filing_id": RCEPT,
        "rcept_no": RCEPT,
        "biz_year": "2023",
        "reprt_code": "11011",
        "fs_div": "CFS",
        "published_at": "2024-03-15",
    }
    return {**base, **overrides}


def _standard_page(**overrides):
    page = {
        "source_kind": "opendart_standard",
        "status": "000",
        "identity": _identity(),
        "records": [
            {
                "rcept_no": RCEPT,
                "sj_div": "BS",
                "account_id": "ifrs-full_Assets",
                "account_nm": "자산총계",
                "thstrm_amount": "5,000",
            }
        ],
        "raw_document_hash": None,
        **_identity(),
    }
    return {**page, **overrides}


def _legacy_page(**overrides):
    page = {
        "source_kind": "legacy_document",
        "status": "013",
        "identity": _identity(),
        "records": [],
        "diagnostics": ("legacy_fallback",),
        "raw_document_hash": "abc123",
        **_identity(),
    }
    return {**page, **overrides}


def test_resolve_metadata_matches_direct_derivation(tmp_path: Path) -> None:
    """Each cached field equals the value derived from the page itself."""
    from src.data.dart_document_benchmark import (
        _is_financial_page,
        _page_field,
        _page_receipt,
        standard_labels,
    )
    from src.data.fact_page_meta import FactPageMetaStore
    from src.data.jobs.dart_documents import _is_document_not_found
    from src.integrations.dart.document_statements import REVISION as DOCUMENT_STATEMENTS_REVISION

    pages = {
        f"{CORP}:2023:11011": _standard_page(),
        f"{CORP}:2022:11011": _legacy_page(),
        f"{CORP}:2021:11011": _legacy_page(
            source_kind="document_verified",
            parser_version="v0-old",
            diagnostics=("legacy_fallback",),
        ),
        f"{CORP}:2020:11011": _legacy_page(
            source_kind="document_verified",
            parser_version=DOCUMENT_STATEMENTS_REVISION,
            diagnostics=("document_not_found",),
        ),
    }
    entries = []
    digests: dict[str, str] = {}
    for key, page in pages.items():
        path, digest = _write_page(tmp_path, key.replace(":", "_"), page)
        digests[key] = digest
        entries.append(_entry(key, path, digest))
    store = FactPageMetaStore(tmp_path / "catalog")
    resolved = {entry.natural_key: meta for entry, meta in store.resolve(entries)}
    assert set(resolved) == set(pages)
    for key, page in pages.items():
        meta = resolved[key]
        parts = key.split(":")
        assert meta.content_hash == digests[key]
        assert meta.source_kind == str(page.get("source_kind") or "")
        assert meta.status == str(page.get("status") or "")
        assert meta.parser_version == str(page.get("parser_version") or "")
        assert meta.raw_document_hash == str(page.get("raw_document_hash") or "")
        assert meta.document_not_found == bool(_is_document_not_found(page))
        assert meta.has_labels == bool(standard_labels(page))
        assert meta.is_financial == bool(_is_financial_page(page))
        assert meta.reprt_code == (_page_field(page, "reprt_code") or parts[-1])
        assert meta.biz_year == (_page_field(page, "biz_year") or parts[1])
        assert meta.rcept_no == _page_receipt(page)


def test_resolve_second_pass_reads_no_payloads(tmp_path: Path, monkeypatch) -> None:
    """A warm cache returns identical metadata without touching payload files."""
    from src.data.fact_page_meta import FactPageMetaStore

    entries = []
    for year in ("2021", "2022", "2023"):
        key = f"{CORP}:{year}:11011"
        path, digest = _write_page(tmp_path, key.replace(":", "_"), _standard_page())
        entries.append(_entry(key, path, digest))
    store = FactPageMetaStore(tmp_path / "catalog")
    first = list(store.resolve(entries))
    reads = 0
    real_read_bytes = Path.read_bytes

    def _counted_read_bytes(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal reads
        reads += 1
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _counted_read_bytes)
    second = list(store.resolve(entries))
    assert reads == 0
    assert [meta for _, meta in second] == [meta for _, meta in first]


def test_resolve_changed_content_uses_new_key(tmp_path: Path) -> None:
    """A new content hash for an identity is read and reflects the new page."""
    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    old_path, old_digest = _write_page(tmp_path, "old", _legacy_page())
    store = FactPageMetaStore(tmp_path / "catalog")
    (old_entry, old_meta) = next(iter(store.resolve([_entry(key, old_path, old_digest)])))
    assert old_meta.source_kind == "legacy_document"
    new_path, new_digest = _write_page(tmp_path, "new", _standard_page())
    assert new_digest != old_digest
    (new_entry, new_meta) = next(iter(store.resolve([_entry(key, new_path, new_digest)])))
    assert new_entry.natural_key == key
    assert new_meta.source_kind == "opendart_standard"
    assert new_meta.has_labels is True


def test_resolve_derivation_version_change_rederives(tmp_path: Path, monkeypatch) -> None:
    """Rows stored under an older derivation version are derived again."""
    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    path, digest = _write_page(tmp_path, "page", _standard_page())
    store = FactPageMetaStore(tmp_path / "catalog")
    assert len(list(store.resolve([_entry(key, path, digest)]))) == 1
    db_path = tmp_path / "catalog" / "fact_page_meta.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE fact_page_meta SET derivation_version = 0")
        connection.commit()
    reads = 0
    real_read_bytes = Path.read_bytes

    def _counted_read_bytes(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal reads
        reads += 1
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _counted_read_bytes)
    resolved = list(store.resolve([_entry(key, path, digest)]))
    assert reads == 1
    assert resolved[0][1].source_kind == "opendart_standard"
    with sqlite3.connect(db_path) as connection:
        (version,) = connection.execute(
            "SELECT derivation_version FROM fact_page_meta"
        ).fetchone()
    assert version == 1


def test_resolve_unreadable_page_skipped_and_recovers(tmp_path: Path) -> None:
    """Missing or non-object payloads are absent; a later file appears on retry."""
    from src.data.fact_page_meta import FactPageMetaStore

    good_path, good_digest = _write_page(tmp_path, "good", _standard_page())
    missing_path = tmp_path / "missing.json"
    missing_hash = "0" * 64
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not json", encoding="utf-8")
    corrupt_hash = hashlib.sha256(b"not json").hexdigest()
    tail_path, tail_digest = _write_page(tmp_path, "tail", _legacy_page())
    entries = [
        _entry(f"{CORP}:2023:11011", good_path, good_digest),
        _entry("missing:2023:11011", missing_path, missing_hash),
        _entry("corrupt:2023:11011", corrupt_path, corrupt_hash),
        _entry(f"{CORP}:2022:11011", tail_path, tail_digest),
    ]
    store = FactPageMetaStore(tmp_path / "catalog")
    first = list(store.resolve(entries))
    assert [entry.natural_key for entry, _ in first] == [
        f"{CORP}:2023:11011",
        f"{CORP}:2022:11011",
    ]
    assert list(store.resolve(entries[1:3])) == []
    raw = json.dumps(_standard_page(), ensure_ascii=False).encode("utf-8")
    missing_path.write_bytes(raw)
    recovered_hash = hashlib.sha256(raw).hexdigest()
    second = list(
        store.resolve([_entry("missing:2023:11011", missing_path, recovered_hash)])
    )
    assert len(second) == 1
    assert second[0][1].source_kind == "opendart_standard"


def test_resolve_corrupt_cache_file_rebuilds(tmp_path: Path) -> None:
    """A garbage cache file is discarded; results stay correct afterwards."""
    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    path, digest = _write_page(tmp_path, "page", _standard_page())
    store = FactPageMetaStore(tmp_path / "catalog")
    assert len(list(store.resolve([_entry(key, path, digest)]))) == 1
    db_path = tmp_path / "catalog" / "fact_page_meta.sqlite3"
    db_path.write_bytes(b"not a sqlite database")
    rebuilt = list(store.resolve([_entry(key, path, digest)]))
    assert rebuilt[0][1].source_kind == "opendart_standard"
    again = list(store.resolve([_entry(key, path, digest)]))
    assert again[0][1] == rebuilt[0][1]
    with sqlite3.connect(db_path) as connection:
        (count,) = connection.execute("SELECT COUNT(*) FROM fact_page_meta").fetchone()
    assert count == 1


def test_resolve_preserves_input_order_across_batches(tmp_path: Path, monkeypatch) -> None:
    """Entries spanning several batches yield in input order."""
    import src.data.fact_page_meta as meta_mod
    from src.data.fact_page_meta import FactPageMetaStore

    monkeypatch.setattr(meta_mod, "_BATCH_SIZE", 2)
    entries = []
    for index in range(5):
        key = f"{CORP}:202{index}:11011"
        page = _standard_page() if index % 2 == 0 else _legacy_page()
        path, digest = _write_page(tmp_path, f"page{index}", page)
        entries.append(_entry(key, path, digest))
    store = FactPageMetaStore(tmp_path / "catalog")
    resolved = list(store.resolve(reversed(entries)))
    assert [entry.natural_key for entry, _ in resolved] == [
        entry.natural_key for entry in reversed(entries)
    ]


def test_store_discards_garbage_file_and_rebuilds(tmp_path: Path) -> None:
    """A direct store over a garbage file replaces it with valid rows."""
    import sqlite3 as _sqlite3

    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    path, digest = _write_page(tmp_path, "page", _standard_page())
    store = FactPageMetaStore(tmp_path / "catalog")
    [( _, meta)] = list(store.resolve([_entry(key, path, digest)]))
    db_path = tmp_path / "catalog" / "fact_page_meta.sqlite3"
    db_path.write_bytes(b"not a sqlite database")
    store._store([meta])
    with _sqlite3.connect(db_path) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM fact_page_meta WHERE content_hash = ?", (digest,)
        ).fetchone()
    assert count == 1


def test_resolve_succeeds_when_store_fails(tmp_path: Path, monkeypatch) -> None:
    """A failing cache write never raises and never blocks planning."""
    import sqlite3 as _sqlite3

    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    path, digest = _write_page(tmp_path, "page", _standard_page())
    store = FactPageMetaStore(tmp_path / "catalog")

    def _failing_write(metas) -> None:  # type: ignore[no-untyped-def]
        raise _sqlite3.InterfaceError("bad parameter")

    monkeypatch.setattr(store, "_write_rows", _failing_write)
    resolved = list(store.resolve([_entry(key, path, digest)]))
    assert len(resolved) == 1
    assert resolved[0][1].source_kind == "opendart_standard"


def test_resolve_succeeds_when_store_locked(tmp_path: Path, monkeypatch) -> None:
    """A locked cache database never raises and never blocks planning."""
    import sqlite3 as _sqlite3

    from src.data.fact_page_meta import FactPageMetaStore

    key = f"{CORP}:2023:11011"
    path, digest = _write_page(tmp_path, "page", _standard_page())
    store = FactPageMetaStore(tmp_path / "catalog")

    def _locked_write(metas) -> None:  # type: ignore[no-untyped-def]
        raise _sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_write_rows", _locked_write)
    resolved = list(store.resolve([_entry(key, path, digest)]))
    assert len(resolved) == 1
    assert resolved[0][1].source_kind == "opendart_standard"
