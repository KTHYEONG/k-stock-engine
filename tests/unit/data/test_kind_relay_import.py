"""KIND relay import invariants: bundle gates, verified persistence, coverage."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from tests.fixtures import scope_runtime as _scope_runtime
from tests.fixtures.kind_html import kind_row_html, kind_search_html

FETCHED_AT = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
NOW = FETCHED_AT + timedelta(hours=1)
FETCH_DATE = date(2026, 9, 24)
OPEN_START = date(2026, 7, 1)


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _keywords() -> tuple[str, ...]:
    return tuple(_provider().kind.search_keywords)


class _Collector:
    def __init__(self, rows_by_keyword):  # type: ignore[no-untyped-def]
        self._rows = dict(rows_by_keyword)

    def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
        rows = self._rows[keyword]
        return kind_search_html(rows, len(rows))


def _search_item(keyword: str, rows: list[str]):  # type: ignore[no-untyped-def]
    from src.data.jobs.kind import collect_search_window

    return collect_search_window(
        _Collector({keyword: rows}),
        keyword=keyword,
        start=OPEN_START,
        end=FETCH_DATE,
        today=FETCH_DATE,
        retrieved_at=FETCHED_AT,
    )


def _row(acptno: str, *, title: str = "상장폐지", submitter: str = "유가증권시장본부") -> str:
    return kind_row_html(acptno, "2024-01-02 15:30", title, "005930", submitter)


def _bundle(items, *, fetched_at: datetime = FETCHED_AT):  # type: ignore[no-untyped-def]
    from src.data.kind_relay import RelayBundle

    return RelayBundle(
        fetched_at=fetched_at,
        fetch_date_kst=(fetched_at + timedelta(hours=9)).date(),
        producer={"run_id": "test"},
        items=tuple(items),
    )


def _write_bundle(runtime, bundle) -> Path:  # type: ignore[no-untyped-def]
    from src.data.kind_relay import encode_relay_bundle
    from src.data.kind_relay_import import relay_inbox_path

    path = relay_inbox_path(runtime.workspace.state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_relay_bundle(bundle))
    return path


def _search_bundle(keywords=None):  # type: ignore[no-untyped-def]
    selected = tuple(keywords) if keywords is not None else _keywords()
    items = [
        _search_item(keyword, [_row(f"20260924{position:06d}")])
        for position, keyword in enumerate(selected)
    ]
    return _bundle(items)


def _import(runtime, provider, job: str, *, now: datetime = NOW, dry_run: bool = False):  # type: ignore[no-untyped-def]
    from src.data.kind_relay_import import import_relay_bundle

    return import_relay_bundle(runtime=runtime, provider=provider, job=job, now=now, dry_run=dry_run)


def _catalog(runtime):  # type: ignore[no-untyped-def]
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def _state_file(runtime) -> Path:  # type: ignore[no-untyped-def]
    return runtime.workspace.state_root / "kind_relay" / "last_accepted.json"


def _sequence(bronze_root: Path) -> int:
    connection = sqlite3.connect(bronze_root / "catalog" / "catalog.sqlite3")
    try:
        row = connection.execute("SELECT sequence FROM catalog_metadata WHERE singleton = 1").fetchone()
    finally:
        connection.close()
    return int(row[0])


def test_missing_bundle_is_absent_and_writes_nothing(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE

    runtime = _scope_runtime(tmp_path)
    provider = _provider()

    for job in ("kind_notice_search", "kind_notice_documents"):
        report = _import(runtime, provider, job)
        assert report.status == "relay_absent"
        assert report.fetched_at is None
    assert _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={"k"}) == {}
    assert not (runtime.workspace.bronze_root / "catalog" / "catalog.sqlite3").exists()
    assert not _state_file(runtime).exists()


def test_corrupt_or_stale_bundle_is_rejected_whole(tmp_path: Path) -> None:
    from src.data.kind_relay import encode_relay_bundle

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    good = _search_bundle()

    tampered = json.loads(encode_relay_bundle(good))
    tampered["items"][0]["payload_b64"] = tampered["items"][0]["payload_b64"][:-2] + "AA"
    _write_bundle(runtime, good)
    inbox = runtime.workspace.state_root / "kind_relay" / "inbox" / "bundle.json"
    inbox.write_text(json.dumps(tampered), encoding="utf-8")
    assert _import(runtime, provider, "kind_notice_search").status == "relay_rejected"

    stale = _bundle(good.items, fetched_at=NOW - timedelta(hours=40))
    _write_bundle(runtime, stale)
    assert _import(runtime, provider, "kind_notice_search").status == "relay_rejected"

    future = _bundle(good.items, fetched_at=NOW + timedelta(minutes=10))
    _write_bundle(runtime, future)
    assert _import(runtime, provider, "kind_notice_search").status == "relay_rejected"
    assert _import(runtime, provider, "kind_notice_documents").status == "relay_rejected"

    assert not (runtime.workspace.bronze_root / "catalog" / "catalog.sqlite3").exists()
    assert not _state_file(runtime).exists()


def test_valid_bundle_imports_with_importer_retrieved_at(tmp_path: Path) -> None:
    import json as _json

    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.kind_relay import encode_relay_bundle

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    bundle = _search_bundle()
    _write_bundle(runtime, bundle)

    report = _import(runtime, provider, "kind_notice_search")

    assert report.status == "complete"
    assert (report.imported, report.unchanged, report.rejected, report.pending_left) == (3, 0, 0, 0)
    assert report.fetched_at == FETCHED_AT
    expected_keys = {f"{keyword}:{OPEN_START.isoformat()}..{FETCH_DATE.isoformat()}" for keyword in _keywords()}
    entries = _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys=expected_keys)
    assert set(entries) == expected_keys
    for key in expected_keys:
        assert entries[key].retrieved_at == NOW
        receipt = _json.loads((entries[key].payload_path.parent / "receipt.json").read_text(encoding="utf-8"))
        assert f"relay:fetched_at={FETCHED_AT.isoformat()}" in receipt["source_path"]
    state = _json.loads(_state_file(runtime).read_text(encoding="utf-8"))
    assert state["fetched_at"] == FETCHED_AT.isoformat()
    assert state["items_sha256"] == _json.loads(encode_relay_bundle(bundle))["items_sha256"]


def test_second_import_is_idempotent(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    _write_bundle(runtime, _search_bundle())

    first = _import(runtime, provider, "kind_notice_search")
    sequence = _sequence(runtime.workspace.bronze_root)
    keys = {f"{keyword}:{OPEN_START.isoformat()}..{FETCH_DATE.isoformat()}" for keyword in _keywords()}
    before = _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys=keys)

    second = _import(runtime, provider, "kind_notice_search")

    assert (first.imported, first.unchanged) == (3, 0)
    assert (second.imported, second.unchanged, second.rejected) == (0, 3, 0)
    assert second.status == "complete"
    assert _sequence(runtime.workspace.bronze_root) == sequence
    assert _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys=keys) == before


def test_one_bad_item_does_not_poison_others(tmp_path: Path) -> None:
    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    keywords = _keywords()
    bad, *rest = keywords
    items = [_search_item(keyword, [_row(f"20260924{position:06d}")]) for position, keyword in enumerate(rest)]
    key = f"{bad}:{OPEN_START.isoformat()}..{FETCH_DATE.isoformat()}"
    items.append(
        ScopedRawPayload(
            kind=EvidenceKind.DISCLOSURES,
            source=KIND_NOTICE_SEARCH_SOURCE,
            natural_key=key,
            as_of=FETCH_DATE,
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps(
                {"keyword": bad, "start": OPEN_START.isoformat(), "end": FETCH_DATE.isoformat(),
                 "reported_total": 2, "pages": [kind_search_html([_row("20260924999999")], 2)]}
            ).encode("utf-8"),
            retrieved_at=FETCHED_AT,
            source_label="test",
        )
    )
    _write_bundle(runtime, _bundle(items))

    report = _import(runtime, provider, "kind_notice_search")

    assert (report.imported, report.rejected) == (2, 1)
    assert report.status == "relay_partial"
    assert _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={key}) == {}


def test_missing_open_window_keyword_is_partial(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    keywords = [keyword for keyword in _keywords() if keyword != "정리매매"]
    assert len(keywords) == len(_keywords()) - 1
    _write_bundle(runtime, _search_bundle(keywords))

    report = _import(runtime, provider, "kind_notice_search")

    assert report.pending_left == 1
    assert report.status == "relay_partial"


def test_future_window_end_is_rejected(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.jobs.kind import collect_search_window

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    keywords = _keywords()
    future_end = FETCH_DATE + timedelta(days=1)
    future = collect_search_window(
        _Collector({keywords[0]: [_row("20260924999999")]}),
        keyword=keywords[0],
        start=OPEN_START,
        end=future_end,
        today=FETCH_DATE,
        retrieved_at=FETCHED_AT,
    )
    items = [future] + [
        _search_item(keyword, [_row(f"20260924{position:06d}")])
        for position, keyword in enumerate(keywords[1:], start=1)
    ]
    _write_bundle(runtime, _bundle(items))

    report = _import(runtime, provider, "kind_notice_search")

    assert report.rejected == 1
    assert report.status == "relay_partial"
    assert _catalog(runtime).latest(
        source=KIND_NOTICE_SEARCH_SOURCE,
        natural_keys={f"{keywords[0]}:{OPEN_START.isoformat()}..{future_end.isoformat()}"},
    ) == {}


def _target_bundle():  # type: ignore[no-untyped-def]
    from src.data.jobs.kind import collect_notice_document
    from src.data.scoped_ingestion import ScopedRawPayload
    from src.integrations.krx.kind import KindNoticeDocument, parse_kind_search_page

    class _Docs:
        def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
            return KindNoticeDocument(acptno=acptno, doc_no="1", template="a.htm", html="<html>x</html>")

    keywords = _keywords()
    rows = [_row("20240101000001"), _row("20240101000002", submitter="삼성전자")]
    search = _search_item(keywords[0], rows)
    (target,) = [row for row in parse_kind_search_page(kind_search_html(rows, 2)).rows
                 if row.acptno == "20240101000001"]
    document = collect_notice_document(_Docs(), target, retrieved_at=FETCHED_AT)
    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE
    from src.data.receipt_catalog import EvidenceStatus

    ignored = ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES,
        source=KIND_NOTICE_DOCUMENT_SOURCE,
        natural_key="20240101000002",
        as_of=date(2024, 1, 2),
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=b"not-json",
        retrieved_at=FETCHED_AT,
        source_label="test",
    )
    others = [
        _search_item(keyword, [_row(f"20260924{position:06d}", submitter="삼성전자")])
        for position, keyword in enumerate(keywords[1:], start=1)
    ]
    return _bundle((search, document, ignored, *others))


def test_documents_only_for_current_targets(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    _write_bundle(runtime, _target_bundle())

    search_report = _import(runtime, provider, "kind_notice_search")
    assert search_report.imported == 3
    report = _import(runtime, provider, "kind_notice_documents")

    catalog = _catalog(runtime)
    assert set(catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys={"20240101000001"})) == {"20240101000001"}
    assert catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys={"20240101000002"}) == {}
    assert (report.imported, report.rejected, report.pending_left) == (1, 0, 0)
    assert report.status == "complete"


def test_documents_step_reports_unanswered_targets(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    keywords = _keywords()
    rows = [_row("20240101000001")]
    _write_bundle(runtime, _bundle([_search_item(keywords[0], rows)]))

    assert _import(runtime, provider, "kind_notice_search").imported == 1
    report = _import(runtime, provider, "kind_notice_documents")

    assert (report.imported, report.pending_left) == (0, 1)
    assert report.status == "relay_partial"


def test_state_file_never_regresses(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    bundle = _search_bundle()
    _write_bundle(runtime, bundle)
    assert _import(runtime, provider, "kind_notice_search").status == "complete"
    state_before = _state_file(runtime).read_bytes()

    older = _bundle(bundle.items, fetched_at=FETCHED_AT - timedelta(hours=1))
    _write_bundle(runtime, older)
    report = _import(runtime, provider, "kind_notice_search")

    assert report.status == "relay_rejected"
    assert _state_file(runtime).read_bytes() == state_before


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    _write_bundle(runtime, _search_bundle())

    report = _import(runtime, provider, "kind_notice_search", dry_run=True)

    assert report.status == "dry_run"
    assert (report.imported, report.pending_left) == (3, 0)
    assert not (runtime.workspace.bronze_root / "catalog" / "catalog.sqlite3").exists()
    assert not _state_file(runtime).exists()


def test_inbox_file_is_never_modified(tmp_path: Path) -> None:
    from src.data.kind_relay import encode_relay_bundle

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    path = _write_bundle(runtime, _search_bundle())

    before = (path.read_bytes(), path.stat().st_mtime_ns)
    _import(runtime, provider, "kind_notice_search")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before

    tampered = bytearray(encode_relay_bundle(_search_bundle()))
    tampered[100] ^= 1
    path.write_bytes(bytes(tampered))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert _import(runtime, provider, "kind_notice_search").status == "relay_rejected"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_unknown_relay_job_raises(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import PITDataError

    runtime = _scope_runtime(tmp_path)
    with pytest.raises(PITDataError, match="unknown relay job"):
        _import(runtime, _provider(), "kind_notice_searches")


def test_out_of_scope_windows_are_rejected(tmp_path: Path) -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    valid = _search_bundle().items
    foreign = _search_item("불성실공시", [_row("20260924999998")])
    key = f"불성실공시:{OPEN_START.isoformat()}..{FETCH_DATE.isoformat()}"
    _write_bundle(runtime, _bundle((*valid, foreign)))

    report = _import(runtime, provider, "kind_notice_search")

    assert (report.imported, report.rejected, report.pending_left) == (3, 1, 0)
    assert report.status == "relay_partial"
    assert _catalog(runtime).latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={key}) == {}


def test_colon_keyword_window_is_rejected_without_crashing(tmp_path: Path) -> None:
    from src.data.jobs.kind import collect_search_window

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    tricky = collect_search_window(
        _Collector({"a:b": [_row("20260924999997")]}),
        keyword="a:b",
        start=OPEN_START,
        end=FETCH_DATE,
        today=FETCH_DATE,
        retrieved_at=FETCHED_AT,
    )
    _write_bundle(runtime, _bundle((*_search_bundle().items, tricky)))

    report = _import(runtime, provider, "kind_notice_search")

    assert (report.imported, report.rejected) == (3, 1)
    assert report.status == "relay_partial"


def test_invalid_target_body_is_rejected_and_stays_pending(tmp_path: Path) -> None:
    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload
    from src.integrations.krx.kind import parse_kind_search_page

    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    keywords = _keywords()
    rows = [_row("20240101000001"), _row("20240101000003")]
    search = _search_item(keywords[0], rows)
    (target,) = [row for row in parse_kind_search_page(kind_search_html(rows, 2)).rows
                 if row.acptno == "20240101000001"]
    others = [
        _search_item(keyword, [_row(f"20260924{position:06d}", submitter="삼성전자")])
        for position, keyword in enumerate(keywords[1:], start=1)
    ]
    from src.data.jobs.kind import collect_notice_document
    from src.integrations.krx.kind import KindNoticeDocument

    class _Docs:
        def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
            return KindNoticeDocument(acptno=acptno, doc_no="1", template="a.htm", html="<html>x</html>")

    document = collect_notice_document(_Docs(), target, retrieved_at=FETCHED_AT)
    broken = ScopedRawPayload(
        kind=EvidenceKind.DISCLOSURES,
        source=KIND_NOTICE_DOCUMENT_SOURCE,
        natural_key="20240101000003",
        as_of=date(2024, 1, 2),
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS,
        payload=b"not-json",
        retrieved_at=FETCHED_AT,
        source_label="test",
    )
    _write_bundle(runtime, _bundle((search, document, broken, *others)))

    assert _import(runtime, provider, "kind_notice_search").status == "complete"
    report = _import(runtime, provider, "kind_notice_documents")

    assert (report.imported, report.rejected, report.pending_left) == (1, 1, 1)
    assert report.status == "relay_partial"


def test_documents_second_import_is_idempotent(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    provider = _provider()
    _write_bundle(runtime, _target_bundle())

    assert _import(runtime, provider, "kind_notice_search").imported == 3
    first = _import(runtime, provider, "kind_notice_documents")
    second = _import(runtime, provider, "kind_notice_documents")

    assert (first.imported, first.unchanged) == (1, 0)
    assert (second.imported, second.unchanged, second.rejected) == (0, 1, 0)
    assert second.status == "complete"
    assert second.pending_left == 0


def test_unreadable_and_oversize_inbox_write_nothing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import src.data.kind_relay_import as importer

    runtime = _scope_runtime(tmp_path)
    path = importer.relay_inbox_path(runtime.workspace.state_root)
    path.mkdir(parents=True)
    assert _import(runtime, _provider(), "kind_notice_search").status == "relay_rejected"
    runtime = _scope_runtime(tmp_path / "oversize")
    path = _write_bundle(runtime, _search_bundle())
    monkeypatch.setattr(importer, "RELAY_MAX_BYTES", 64)
    assert path.stat().st_size > 64
    assert _import(runtime, _provider(), "kind_notice_documents").status == "relay_rejected"
    assert not _state_file(runtime).exists()
    assert not (runtime.workspace.bronze_root / "catalog" / "catalog.sqlite3").exists()


def test_bad_storage_contract_does_not_poison_search_batch(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.core.pit import EvidenceKind

    for dry_run in (False, True):
        runtime = _scope_runtime(tmp_path / str(dry_run))
        first, *rest = _search_bundle().items
        broken = replace(first, kind=EvidenceKind.DAILY_MARKET)
        _write_bundle(runtime, _bundle((broken, *rest)))
        report = _import(runtime, _provider(), "kind_notice_search", dry_run=dry_run)
        assert (report.imported, report.rejected, report.pending_left) == (2, 1, 1)
        assert report.status == ("dry_run" if dry_run else "relay_partial")


def test_document_dry_run_simulates_answers_without_writing(tmp_path: Path) -> None:
    runtime = _scope_runtime(tmp_path)
    _write_bundle(runtime, _target_bundle())
    _import(runtime, _provider(), "kind_notice_search")
    sequence = _sequence(runtime.workspace.bronze_root)
    state_before = _state_file(runtime).read_bytes()
    report = _import(runtime, _provider(), "kind_notice_documents", dry_run=True)
    assert (report.status, report.imported, report.pending_left) == ("dry_run", 1, 0)
    assert _sequence(runtime.workspace.bronze_root) == sequence
    assert _state_file(runtime).read_bytes() == state_before


def test_watermark_advanced_during_persistence_never_regresses(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.data.scoped_ingestion import ScopedBronzeWriter

    runtime = _scope_runtime(tmp_path)
    _write_bundle(runtime, _search_bundle())
    persist = ScopedBronzeWriter.persist_many
    newer = NOW.isoformat()

    def advance(writer, payloads):  # type: ignore[no-untyped-def]
        result = persist(writer, payloads)
        _state_file(runtime).write_text(json.dumps({"fetched_at": newer, "items_sha256": "newer"}))
        return result

    monkeypatch.setattr(ScopedBronzeWriter, "persist_many", advance)
    _import(runtime, _provider(), "kind_notice_search")
    assert json.loads(_state_file(runtime).read_bytes()) == {"fetched_at": newer, "items_sha256": "newer"}


def test_malformed_document_is_isolated_before_storage(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE

    for bad_payload in (b"[" * 2000 + b"]" * 2000, None):
        runtime = _scope_runtime(tmp_path / str(bad_payload is None))
        bundle = _target_bundle()
        items = tuple(
            replace(item, payload=bad_payload) if bad_payload is not None
            else replace(item, kind=EvidenceKind.DAILY_MARKET)
            for item in bundle.items
            if item.source == KIND_NOTICE_DOCUMENT_SOURCE and item.natural_key == "20240101000001"
        )
        searches = tuple(item for item in bundle.items if item.source != KIND_NOTICE_DOCUMENT_SOURCE)
        _write_bundle(runtime, _bundle((*searches, *items)))
        _import(runtime, _provider(), "kind_notice_search")
        report = _import(runtime, _provider(), "kind_notice_documents")
        assert (report.status, report.rejected, report.pending_left) == ("relay_partial", 1, 1)
