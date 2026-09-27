"""Streaming Bronze index tests: classification, resume, corruption, dry-run and memory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tracemalloc
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.pit import PITDataError
from src.data.bronze_index import index_bronze
from src.data.receipt_catalog import ReceiptCatalog
from src.data.runtime import load_data_runtime

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
RETRIEVED_AT = datetime(2024, 6, 1, tzinfo=UTC)


def _runtime(tmp_path: Path):
    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _bronze_root(tmp_path: Path) -> Path:
    return _runtime(tmp_path).workspace.bronze_root


def _write_blob(
    bronze_root: Path,
    kind_dir: str,
    payload: bytes,
    *,
    retrieved_at: datetime | None = RETRIEVED_AT,
    suffix: str = ".json",
) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    blob_dir = bronze_root / kind_dir / digest
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / f"payload{suffix}").write_bytes(payload)
    if retrieved_at is not None:
        (blob_dir / "receipt.json").write_text(
            json.dumps({
                "kind": kind_dir,
                "content_hash": digest,
                "source_path": f"test:{kind_dir}",
                "retrieved_at": retrieved_at.isoformat(),
                "ingested_at": retrieved_at.isoformat(),
            }, sort_keys=True),
            encoding="utf-8",
        )
    return digest


def _json_blob(
    bronze_root: Path, kind_dir: str, document: Mapping[str, object], retrieved_at: datetime | None = RETRIEVED_AT
) -> str:
    return _write_blob(
        bronze_root,
        kind_dir,
        json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        retrieved_at=retrieved_at,
    )


def _seed_legacy_receipts(
    bronze_root: Path, rows: list[tuple[str, str, str, str, str, str | None]]
) -> None:
    """Seed a v1 catalog: (source, natural_key, content_hash, status, payload_rel, retrieved_at)."""
    database = bronze_root / "catalog" / "catalog.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE catalog_metadata(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "schema_version INTEGER NOT NULL, sequence INTEGER NOT NULL, row_count INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE receipts(source TEXT NOT NULL, natural_key TEXT NOT NULL, as_of TEXT, "
            "fiscal_period TEXT, status TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "retrieved_at TEXT NOT NULL, payload_path TEXT NOT NULL, "
            "PRIMARY KEY (source, natural_key)) WITHOUT ROWID"
        )
        connection.execute("CREATE INDEX idx_receipts_source_status ON receipts(source, status)")
        for source, natural_key, content_hash, status, payload_rel, moment in rows:
            connection.execute(
                "INSERT INTO receipts(source, natural_key, as_of, fiscal_period, status, "
                "content_hash, retrieved_at, payload_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source, natural_key, None, None, status, content_hash,
                    moment if moment is not None else RETRIEVED_AT.isoformat(), payload_rel,
                ),
            )
        connection.execute(
            "INSERT INTO catalog_metadata(singleton, schema_version, sequence, row_count) VALUES (1, 1, 0, ?)",
            (len(rows),),
        )
        connection.commit()
    finally:
        connection.close()


def _stored_hashes(bronze_root: Path) -> set[str]:
    with sqlite3.connect(bronze_root / "catalog" / "catalog.sqlite3") as connection:
        return {
            str(row[0]) for row in connection.execute("SELECT content_hash FROM blobs").fetchall()
        }


def _blob_row(bronze_root: Path, content_hash: str) -> tuple[str, int, str | None]:
    connection = sqlite3.connect(bronze_root / "catalog" / "catalog.sqlite3")
    try:
        row = connection.execute(
            "SELECT source, usable, unusable_reason FROM blobs WHERE content_hash = ?", (content_hash,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None, content_hash
    return (str(row[0]), int(row[1]), str(row[2]) if row[2] is not None else None)


def _sequence(bronze_root: Path) -> int:
    connection = sqlite3.connect(bronze_root / "catalog" / "catalog.sqlite3")
    try:
        row = connection.execute("SELECT sequence FROM catalog_metadata WHERE singleton = 1").fetchone()
    finally:
        connection.close()
    return int(row[0])


def test_index_bronze_classifies_every_table_row(tmp_path: Path) -> None:
    """One synthetic blob per classification-table row earns exactly its verdict."""
    bronze_root = _bronze_root(tmp_path)
    heartbeats: list[Mapping[str, object]] = []

    ls_usable = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "endpoint": "t1702",
        "query": {"symbol": "005930", "start": "2024-01-02", "end": "2024-01-04"},
        "rows": [{"date": "20240102"}], "records": [],
    })
    marker = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "endpoint": "frgr-itt", "symbol": "005930",
        "status": "missing_sessions", "missing_sessions": ["2024-01-02", "2024-01-03"],
    })
    kis_usable = _json_blob(bronze_root, "investor_flow", {
        "provider": "KIS", "endpoint": "investor-trade-by-stock-daily",
        "query": {"symbol": "005930", "anchor": "2024-01-04"},
        "rows": [{"stck_bsop_date": "20240102"}, {"stck_bsop_date": "20240103"}],
        "records": [],
    })
    kis_mapped = _json_blob(bronze_root, "investor_flow", {
        "provider": "KIS", "query": {"symbol": "005930", "anchor": "2024-01-04"},
        "rows": [], "records": [{"ticker": "005930"}],
    })
    ls_mapped = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "symbol": "005930", "anchor": "2024-01-04", "records": [{"ticker": "005930"}],
    })
    legacy = _json_blob(
        bronze_root, "investor_flow",
        {"provider": "ls", "status": "provider_error", "sessions": ["2024-01-02"]},
        retrieved_at=datetime(2024, 6, 1),
    )
    unknown_provider = _json_blob(bronze_root, "investor_flow", {"provider": "XX", "rows": [{"x": 1}]})
    ls_bad_range = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "query": {"symbol": "005930", "start": "2024-01-05", "end": "2024-01-02"},
        "rows": [{"date": "20240102"}],
    })
    kis_bad_anchor = _json_blob(bronze_root, "investor_flow", {
        "provider": "KIS", "query": {"symbol": "005930", "anchor": "2024-01-02"},
        "rows": [{"stck_bsop_date": "20240105"}],
    })
    missing = _json_blob(bronze_root, "investor_flow", {
        "provider": "KIS", "query": {"symbol": "005930", "anchor": "2024-01-04"},
        "rows": [{"stck_bsop_date": "20240102"}],
    }, retrieved_at=None)
    missing_ls = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "query": {"symbol": "005930", "start": "2024-01-02", "end": "2024-01-02"},
        "rows": [{"date": "20240102"}],
    }, retrieved_at=None)
    broken = _write_blob(bronze_root, "investor_flow", b"{not json")
    ls_bad_start = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "query": {"symbol": "005930", "start": "not-a-date", "end": "2024-01-02"},
        "rows": [{"date": "20240102"}],
    })
    stray_marker = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "symbol": "005930", "status": "missing_sessions", "missing_sessions": "oops",
    })
    industry = _json_blob(bronze_root, "industry", {
        "provider": "KIS", "endpoint": "inquire-price", "symbol": "005930",
        "collected_at": "2024-06-01T09:00:00+09:00",
        "output": {"price": 1}, "records": [{"ticker": "005930", "industry": "x"}],
    })
    industry_naive = _json_blob(bronze_root, "industry", {
        "provider": "KIS", "endpoint": "search-stock-info", "symbol": "000020",
        "collected_at": "2024-06-01T09:00:00",
        "output": {"price": 1}, "records": [{"ticker": "000020", "industry": "y"}],
    })
    industry_broken_time = _json_blob(bronze_root, "industry", {
        "provider": "KIS", "endpoint": "inquire-price", "symbol": "005930",
        "collected_at": "oops", "records": [],
    })
    industry_bad = _json_blob(bronze_root, "industry", {
        "provider": "KIS", "endpoint": "weird", "symbol": "005930",
        "collected_at": "2024-06-01T09:00:00+09:00", "records": [],
    })
    industry_foreign = _json_blob(bronze_root, "industry", {"provider": "XX", "endpoint": "weird"})
    daily = _json_blob(bronze_root, "daily_market", {"close": 1})
    daily_orphan = _json_blob(bronze_root, "daily_market", {"close": 2})
    zipped = _write_blob(bronze_root, "dart_documents", b"PK\x03\x04zip-bytes", suffix=".zip")
    (bronze_root / "dart_documents" / zipped / "receipt.json").write_text(
        json.dumps({"rcept_no": "20240101000001", "retrieved_at": RETRIEVED_AT.isoformat(),
                    "sha256": zipped, "byte_length": 12}, sort_keys=True), encoding="utf-8",
    )
    corp = _write_blob(bronze_root, "dart_corp_codes", b'[{"ticker": "005930"}]', retrieved_at=None)
    _seed_legacy_receipts(bronze_root, [
        ("investor_flow", "005930:2024-01-02", marker, "success", f"investor_flow/{marker}/payload.json", None),
        ("investor_flow", "005930:2024-01-03", stray_marker, "success",
         f"investor_flow/{stray_marker}/payload.json", None),
        ("krx_daily_market", "2024-01-02", daily, "success", f"daily_market/{daily}/payload.json", None),
        ("krx_daily_market", "unmatched", "0" * 64, "success", "daily_market/" + "0" * 64 + "/payload.json",
         "not-a-date"),
        ("krx_daily_market", "2024-01-03", daily, "success", f"daily_market/{daily}/payload.json",
         "2024-01-01T00:00:00"),
    ])

    report = index_bronze(
        _runtime(tmp_path), dry_run=False, batch_size=100, emit=heartbeats.append,
    )

    assert report.scanned == 23
    assert report.registered == 23
    assert report.already_registered == 0
    assert dict(report.usable) == {
        "ls_investor_flow": 1, "kis_investor_flow": 1, "kis_industry": 2, "krx_daily_market": 1,
    }
    assert dict(report.unusable) == {
        "negative_marker": 1, "kis_mapped_only": 1, "ls_mapped_only": 1, "legacy_error_page": 1,
        "unrecognized": 9, "missing_receipt": 3, "unreferenced": 2,
    }
    assert dict(report.ranges_published) == {"ls_investor_flow": 2, "kis_investor_flow": 1}
    assert dict(report.receipts_removed) == {"investor_flow": 2}
    assert len(heartbeats) == 1
    assert heartbeats[0]["scanned"] == 23
    assert heartbeats[0]["registered"] == 23

    assert _blob_row(bronze_root, ls_usable) == ("ls_investor_flow", 1, None)
    assert _blob_row(bronze_root, marker) == ("ls_investor_flow", 0, "negative_marker")
    assert _blob_row(bronze_root, kis_usable) == ("kis_investor_flow", 1, None)
    assert _blob_row(bronze_root, kis_mapped) == ("kis_investor_flow", 0, "kis_mapped_only")
    assert _blob_row(bronze_root, ls_mapped) == ("ls_investor_flow", 0, "ls_mapped_only")
    assert _blob_row(bronze_root, legacy) == ("ls_investor_flow", 0, "legacy_error_page")
    assert _blob_row(bronze_root, unknown_provider) == ("investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, ls_bad_range) == ("ls_investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, kis_bad_anchor) == ("kis_investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, missing) == ("kis_investor_flow", 0, "missing_receipt")
    assert _blob_row(bronze_root, missing_ls) == ("ls_investor_flow", 0, "missing_receipt")
    assert _blob_row(bronze_root, broken) == ("investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, ls_bad_start) == ("ls_investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, stray_marker) == ("ls_investor_flow", 0, "unrecognized")
    assert _blob_row(bronze_root, industry) == ("kis_industry", 1, None)
    assert _blob_row(bronze_root, industry_naive) == ("kis_industry", 1, None)
    assert _blob_row(bronze_root, industry_broken_time) == ("kis_industry", 0, "unrecognized")
    assert _blob_row(bronze_root, industry_bad) == ("kis_industry", 0, "unrecognized")
    assert _blob_row(bronze_root, industry_foreign) == ("industry", 0, "unrecognized")
    assert _blob_row(bronze_root, daily) == ("krx_daily_market", 1, None)
    assert _blob_row(bronze_root, daily_orphan) == ("daily_market", 0, "unreferenced")
    assert _blob_row(bronze_root, zipped) == ("dart_documents", 0, "unreferenced")
    assert _blob_row(bronze_root, corp) == ("dart_corp_codes", 0, "missing_receipt")

    catalog = ReceiptCatalog(bronze_root / "catalog")
    ls_ranges = list(catalog.ranges(source="ls_investor_flow"))
    assert [(item.subject, item.start, item.end, item.status.value, item.content_hash) for item in ls_ranges] == [
        ("005930", date(2024, 1, 2), date(2024, 1, 3), "empty", None),
        ("005930", date(2024, 1, 2), date(2024, 1, 4), "success", ls_usable),
    ]
    kis_ranges = list(catalog.ranges(source="kis_investor_flow"))
    assert [(item.subject, item.start, item.end, item.content_hash) for item in kis_ranges] == [
        ("005930", date(2024, 1, 2), date(2024, 1, 4), kis_usable),
    ]
    found = catalog.latest(source="kis_industry", natural_keys=["inquire-price:005930:2024-06-01"])
    assert found["inquire-price:005930:2024-06-01"].as_of == date(2024, 6, 1)
    assert list(catalog.entries(source="investor_flow")) == []


def test_index_bronze_collapses_cell_receipts_into_ranges(tmp_path: Path) -> None:
    """Three per-cell receipts over one raw window become one range, then vanish."""
    bronze_root = _bronze_root(tmp_path)
    raw = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "query": {"symbol": "005930", "start": "2024-01-02", "end": "2024-01-04"},
        "rows": [{"date": "20240102"}],
    })
    _seed_legacy_receipts(bronze_root, [
        ("investor_flow", f"005930:{day}", raw, "success", f"investor_flow/{raw}/payload.json", None)
        for day in ("2024-01-02", "2024-01-03", "2024-01-04")
    ])

    report = index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)

    catalog = ReceiptCatalog(bronze_root / "catalog")
    assert [(item.start, item.end) for item in catalog.ranges(source="ls_investor_flow")] == [
        (date(2024, 1, 2), date(2024, 1, 4)),
    ]
    assert list(catalog.entries(source="investor_flow")) == []
    assert dict(report.receipts_removed) == {"investor_flow": 3}


def test_index_bronze_resumes_without_reopening_indexed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted run continues where it stopped and never re-reads batch one."""
    from src.data.receipt_catalog import ReceiptCatalog as Catalog

    bronze_a = _bronze_root(tmp_path / "a")
    bronze_b = _bronze_root(tmp_path / "b")
    for bronze_root in (bronze_a, bronze_b):
        for index in range(5):
            _json_blob(bronze_root, "daily_market", {"close": index})

    real_publish = Catalog.publish
    calls = 0

    def _flaky(self: Catalog, entries: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PITDataError("interrupted after the first batch")
        return real_publish(self, entries, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Catalog, "publish", _flaky)
    opens: dict[str, int] = {}
    real_open = Path.open

    def _counting_open(self: Path, *args: object, **kwargs: object) -> object:
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if "r" in mode and self.name in ("payload.json", "receipt.json"):
            opens[str(self)] = opens.get(str(self), 0) + 1
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", _counting_open)
    with pytest.raises(PITDataError, match="interrupted after the first batch"):
        index_bronze(_runtime(tmp_path / "a"), dry_run=False, batch_size=2, emit=lambda _: None)

    committed = _stored_hashes(bronze_a)
    assert len(committed) == 2
    before = dict(opens)

    rerun = index_bronze(_runtime(tmp_path / "a"), dry_run=False, batch_size=2, emit=lambda _: None)
    for content_hash in committed:
        assert opens.get(str(bronze_a / "daily_market" / content_hash / "payload.json"), 0) == before.get(
            str(bronze_a / "daily_market" / content_hash / "payload.json"), 0
        )
    assert rerun.scanned == 5
    assert rerun.registered == 3
    assert rerun.already_registered == 2
    assert dict(rerun.unusable) == {"unreferenced": 3}

    clean = index_bronze(_runtime(tmp_path / "b"), dry_run=False, batch_size=2, emit=lambda _: None)
    assert rerun.scanned == clean.scanned
    assert dict(rerun.ranges_published) == dict(clean.ranges_published) == {}
    assert dict(rerun.receipts_removed) == dict(clean.receipts_removed) == {}
    assert rerun.registered == clean.registered - len(committed)
    assert rerun.already_registered == len(committed)
    assert _stored_hashes(bronze_a) == _stored_hashes(bronze_b)


def test_index_bronze_rejects_corrupt_bytes(tmp_path: Path) -> None:
    """Bytes that do not hash to their directory name stop the run and name the path."""
    bronze_root = _bronze_root(tmp_path)
    blob_dir = bronze_root / "investor_flow" / ("f" * 64)
    blob_dir.mkdir(parents=True, exist_ok=True)
    payload_path = blob_dir / "payload.json"
    payload_path.write_bytes(b"tampered")

    with pytest.raises(PITDataError, match="hash mismatch"):
        index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)

    with pytest.raises(PITDataError, match="hash mismatch") as exc_info:
        index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)
    assert str(payload_path) in str(exc_info.value)
    assert not (bronze_root / "catalog" / "catalog.sqlite3").exists()


def test_index_bronze_dry_run_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run classifies everything yet leaves the catalog sequence untouched."""
    from src.data.receipt_catalog import ReceiptCatalog as Catalog

    bronze_root = _bronze_root(tmp_path)
    raw = _json_blob(bronze_root, "investor_flow", {
        "provider": "LS", "query": {"symbol": "005930", "start": "2024-01-02", "end": "2024-01-02"},
        "rows": [{"date": "20240102"}],
    })
    _json_blob(bronze_root, "daily_market", {"close": 1})
    _seed_legacy_receipts(bronze_root, [
        ("investor_flow", "005930:2024-01-02", raw, "success", f"investor_flow/{raw}/payload.json", None),
    ])

    publishes = 0
    real_publish = Catalog.publish

    def _spy(self: Catalog, entries: object, **kwargs: object) -> object:
        nonlocal publishes
        publishes += 1
        return real_publish(self, entries, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Catalog, "publish", _spy)
    dry = index_bronze(_runtime(tmp_path), dry_run=True, batch_size=10, emit=lambda _: None)
    assert publishes == 0
    assert _sequence(bronze_root) == 0

    monkeypatch.setattr(Catalog, "publish", real_publish)
    real = index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)
    assert dry == real
    assert _sequence(bronze_root) > 0


def test_index_bronze_memory_is_independent_of_blob_count(tmp_path: Path) -> None:
    """Tracing 5,000 blobs peaks within 20% of tracing 1,000 blobs."""
    peaks: list[int] = []
    for name, count in (("small", 1000), ("big", 5000)):
        bronze_root = _bronze_root(tmp_path / name)
        for index in range(count):
            _json_blob(bronze_root, "investor_flow", {"provider": "XX", "rows": [], "n": index})
        tracemalloc.start()
        try:
            report = index_bronze(
                _runtime(tmp_path / name), dry_run=False, batch_size=500, emit=lambda _: None
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert report.scanned == count
        peaks.append(peak)
    assert peaks[1] <= peaks[0] * 1.2


def test_index_bronze_skips_unknown_layouts_and_rejects_bad_inputs(tmp_path: Path) -> None:
    """Unknown directories and stray files are skipped; bad flags and catalogs fail closed."""
    bronze_root = _bronze_root(tmp_path)
    mystery = bronze_root / "mystery" / ("a" * 64)
    mystery.mkdir(parents=True, exist_ok=True)
    (mystery / "payload.json").write_bytes(b"{}")
    (bronze_root / "stray.txt").write_text("stray", encoding="utf-8")
    (bronze_root / ".hidden").write_text("hidden", encoding="utf-8")
    hidden_blob = bronze_root / "daily_market" / ".hidden"
    hidden_blob.mkdir(parents=True, exist_ok=True)
    empty_blob = bronze_root / "daily_market" / ("b" * 64)
    empty_blob.mkdir(parents=True, exist_ok=True)
    kept = _json_blob(bronze_root, "daily_market", {"close": 1})
    (bronze_root / "catalog").mkdir(parents=True, exist_ok=True)
    sqlite3.connect(bronze_root / "catalog" / "catalog.sqlite3").close()

    report = index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)

    assert report.scanned == 1
    assert report.registered == 1
    assert _blob_row(bronze_root, kept) == ("daily_market", 0, "unreferenced")

    with pytest.raises(PITDataError, match="batch_size"):
        index_bronze(_runtime(tmp_path), dry_run=False, batch_size=0, emit=lambda _: None)

    (bronze_root / "catalog" / "catalog.sqlite3").write_bytes(b"not a database")
    with pytest.raises(PITDataError, match="unreadable"):
        index_bronze(_runtime(tmp_path), dry_run=False, batch_size=10, emit=lambda _: None)


def test_index_bronze_cli_reports_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The wired command emits the report summary as its final JSON line."""
    import json as _json

    from src.data.cli import main

    bronze_root = _bronze_root(tmp_path / "cli")
    _json_blob(bronze_root, "daily_market", {"close": 1})
    args = ["--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "cli" / "data")]
    assert main(["index-bronze", *args, "--dry-run", "--batch-size", "10"]) == 0
    summary = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["scanned"] == 1
    assert summary["registered"] == 1
