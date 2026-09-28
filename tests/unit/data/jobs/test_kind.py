"""KIND job invariants: keyword windows, declared-total completeness, document targets."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from tests.fixtures.kind_html import kind_row_html, kind_search_html

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)
SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")


def _runtime(tmp_path: Path, *, evidence_start: date | None = None):  # type: ignore[no-untyped-def]
    import dataclasses

    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    if evidence_start is None:
        return runtime
    return dataclasses.replace(runtime, scope=runtime.scope.model_copy(update={"evidence_start": evidence_start}))


def _provider():  # type: ignore[no-untyped-def]
    from src.config import load_provider_policy, load_runtime_config

    return load_provider_policy(load_runtime_config())


def _provider_with_keywords(keywords: tuple[str, ...]):  # type: ignore[no-untyped-def]
    from src.config.providers import KindPolicy

    provider = _provider()
    return provider.model_copy(
        update={
            "kind": KindPolicy(
                circuit_threshold=3,
                min_interval_seconds=1.0,
                daily_limit=5000,
                search_keywords=list(keywords),
                document_titles=["상장폐지", "관리종목지정", "관리종목지정해제"],
            )
        }
    )


def _ctx(runtime, provider, *, collector=None, now=None):  # type: ignore[no-untyped-def]
    from src.data.jobs.kind import build_kind_job_context

    return build_kind_job_context(runtime=runtime, provider=provider, collector=collector, now=now or (lambda: NOW))






def _acptno(index: int) -> str:
    return f"2024010100{index:04d}"


def _rows(count: int, *, title: str = "상장폐지", company: str | None = "005930",
           submitter: str = "유가증권시장본부") -> list[str]:
    return [
        kind_row_html(_acptno(index), "2024-01-02 15:30", title, company, submitter)
        for index in range(count)
    ]


class _FakeKindClient:
    """Scripted KIND collector serving prebuilt search pages and documents."""

    def __init__(self, *, pages: dict | None = None, documents: dict | None = None):  # type: ignore[no-untyped-def]
        self._pages = pages or {}
        self._documents = documents or {}
        self.search_calls: list[tuple] = []

    def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
        from src.integrations.krx.kind import parse_kind_search_page

        self.search_calls.append((keyword, start, end, page_index))
        html = self._pages.get((keyword, start.isoformat(), end.isoformat(), page_index))
        if html is None:
            html = kind_search_html([], 0)
        parse_kind_search_page(html)
        return html

    def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
        from src.integrations.krx.kind import KindNoticeDocument

        doc_no, template, html = self._documents[acptno]
        return KindNoticeDocument(acptno=acptno, doc_no=doc_no, template=template, html=html)

    def health_check(self) -> None:
        pass


def _persist_search(ctx, *, keyword: str, start: str, end: str, reported_total: int, pages: list[str]):  # type: ignore[no-untyped-def]
    from src.core.pit import EvidenceKind
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.DISCLOSURES,
            source=KIND_NOTICE_SEARCH_SOURCE,
            natural_key=f"{keyword}:{start}..{end}",
            as_of=date.fromisoformat(end),
            fiscal_period=None,
            status=EvidenceStatus.EMPTY if reported_total == 0 else EvidenceStatus.SUCCESS,
            payload=json.dumps(
                {"keyword": keyword, "start": start, "end": end,
                 "reported_total": reported_total, "pages": pages}
            ).encode("utf-8"),
            retrieved_at=NOW,
            source_label=f"test:{keyword}:{start}..{end}",
        )
    )


def test_windows_per_keyword_tile_the_scope(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path, evidence_start=date(2016, 1, 1))
    provider = _provider_with_keywords(("상장폐지", "관리종목"))
    ctx = _ctx(runtime, provider, now=lambda: datetime(2016, 7, 15, 3, 0, tzinfo=UTC))

    keys = {unit.natural_key for unit in KindNoticeSearchJob().pending(ctx)}

    assert keys == {
        f"{keyword}:{start}..{end}"
        for keyword in ("상장폐지", "관리종목")
        for start, end in (
            ("2016-01-01", "2016-03-31"), ("2016-04-01", "2016-06-30"), ("2016-07-01", "2016-07-15")
        )
    }


def test_complete_completed_window_is_not_replanned(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    rows = _rows(2)
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=2, pages=[kind_search_html(rows, 2)])

    units = KindNoticeSearchJob().pending(ctx)

    assert "상장폐지:2019-07-01..2019-09-30" not in {unit.natural_key for unit in units}


def test_short_completed_window_is_replanned(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=2, pages=[kind_search_html(_rows(1), 2)])

    units = KindNoticeSearchJob().pending(ctx)

    assert "상장폐지:2019-07-01..2019-09-30" in {unit.natural_key for unit in units}


def test_fetch_pages_through_the_declared_total(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    rows = _rows(250)
    pages = {
        ("상장폐지", "2019-07-01", "2019-09-30", 1): kind_search_html(rows[:100], 250),
        ("상장폐지", "2019-07-01", "2019-09-30", 2): kind_search_html(rows[100:200], 250),
        ("상장폐지", "2019-07-01", "2019-09-30", 3): kind_search_html(rows[200:], 250),
    }
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(pages=pages))
    (unit,) = [
        unit for unit in KindNoticeSearchJob().pending(ctx)
        if unit.natural_key == "상장폐지:2019-07-01..2019-09-30"
    ]

    (payload,) = KindNoticeSearchJob().fetch(ctx, [unit])

    body = json.loads(payload.payload)
    assert body["reported_total"] == 250
    assert list(body["pages"]) == [pages[("상장폐지", "2019-07-01", "2019-09-30", index)] for index in (1, 2, 3)]
    assert payload.status.value == "success"


def test_row_mismatch_on_completed_window_fails_closed(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    rows = _rows(249)
    pages = {
        ("상장폐지", "2019-07-01", "2019-09-30", 1): kind_search_html(rows[:100], 250),
        ("상장폐지", "2019-07-01", "2019-09-30", 2): kind_search_html(rows[100:200], 250),
        ("상장폐지", "2019-07-01", "2019-09-30", 3): kind_search_html(rows[200:], 250),
    }
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(pages=pages))
    (unit,) = [
        unit for unit in KindNoticeSearchJob().pending(ctx)
        if unit.natural_key == "상장폐지:2019-07-01..2019-09-30"
    ]

    with pytest.raises(PITDataError):
        KindNoticeSearchJob().fetch(ctx, [unit])

    assert ctx.catalog.latest(source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={unit.natural_key}) == {}


def test_page_bound_is_never_truncated(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    pages = {("상장폐지", "2019-07-01", "2019-09-30", 1): kind_search_html(_rows(100), 2500)}
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(pages=pages))
    (unit,) = [
        unit for unit in KindNoticeSearchJob().pending(ctx)
        if unit.natural_key == "상장폐지:2019-07-01..2019-09-30"
    ]

    with pytest.raises(PITDataError):
        KindNoticeSearchJob().fetch(ctx, [unit])


def _target_rows() -> list[str]:
    return [
        kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부"),
        kind_row_html("20240101000002", "2024-01-02 15:31", "상장폐지", "005930", "삼성전자"),
        kind_row_html("20240101000003", "2024-01-02 15:32", "관리종목 지정", "003540", "유가증권시장본부"),
        kind_row_html("20240101000004", "2024-01-02 15:33", "관리종목지정해제(회생절차 종결결정)", "003540", "유가증권시장본부"),
        kind_row_html("20240101000005", "2024-01-02 15:34", "ETF 관리종목지정 해제", "069500", "유가증권시장본부"),
        kind_row_html("20240101000006", "2024-01-02 15:35", "기타시장안내(상장폐지 결정)", "005930", "유가증권시장본부"),
    ]


def test_document_targets_are_exchange_forms_only(tmp_path: Path) -> None:
    from src.data.jobs.kind import kind_document_targets

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    rows = _target_rows()
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=6, pages=[kind_search_html(rows, 6)])

    targets = kind_document_targets(ctx.catalog, _provider().kind)

    assert [row.acptno for row in targets] == ["20240101000001", "20240101000003", "20240101000004"]


def test_title_normalization_strips_tags_qualifiers_and_spaces() -> None:
    from src.data.jobs.kind import normalize_kind_title

    assert normalize_kind_title("[정정] 관리종목 지정 해제(사유 해소)") == "관리종목지정해제"
    assert normalize_kind_title("관리종목 지정") == "관리종목지정"
    assert normalize_kind_title("관리종목지정해제(회생절차 종결결정)") == "관리종목지정해제"


def test_document_receipts_are_keyed_by_acceptance_number(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeDocumentJob

    runtime = _runtime(tmp_path)
    rows = _target_rows()
    documents = {"20240101000001": ("1234567", "68051.htm", "<html>단축코드 A005980</html>")}
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(documents=documents))
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=6, pages=[kind_search_html(rows, 6)])
    (unit,) = [
        unit for unit in KindNoticeDocumentJob().pending(ctx) if unit.natural_key == "20240101000001"
    ]

    (payload,) = KindNoticeDocumentJob().fetch(ctx, [unit])

    assert payload.natural_key == "20240101000001"
    body = json.loads(payload.payload)
    assert body["disclosed_at"] == "2024-01-02T15:30:00+09:00"
    assert payload.status.value == "success"


def test_resolve_kind_job_registry() -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.kind import KIND_JOBS, resolve_kind_job

    assert set(KIND_JOBS) == {"kind_notice_search", "kind_notice_documents"}
    assert resolve_kind_job("kind_notice_search").name == "kind_notice_search"
    with pytest.raises(PITDataError, match="unknown KIND job"):
        resolve_kind_job("nope")


def test_search_receipt_helpers_cover_corrupt_branches(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace

    from src.data.jobs.kind import _search_max_requests, _search_receipt_complete

    def _entry(payload: bytes):  # type: ignore[no-untyped-def]
        target = tmp_path / "payload.json"
        target.write_bytes(payload)
        return SimpleNamespace(
            payload_path=target, content_hash=hashlib.sha256(payload).hexdigest(), natural_key="k"
        )

    assert _search_receipt_complete(SimpleNamespace(payload_path=tmp_path / "missing.json", content_hash="x", natural_key="k")) is False

    good = json.dumps({"reported_total": 0, "pages": []}).encode("utf-8")
    assert _search_receipt_complete(_entry(good)) is True
    assert _search_max_requests(_entry(good)) == 1

    assert _search_receipt_complete(_entry(b"{broken")) is False
    assert _search_receipt_complete(_entry(b"[1]")) is False
    assert _search_receipt_complete(_entry(json.dumps({"pages": []}).encode("utf-8"))) is False
    assert _search_receipt_complete(_entry(json.dumps({"reported_total": True, "pages": []}).encode("utf-8"))) is False
    assert _search_receipt_complete(_entry(json.dumps({"reported_total": "2", "pages": []}).encode("utf-8"))) is False
    assert _search_receipt_complete(_entry(json.dumps({"reported_total": 2, "pages": "nope"}).encode("utf-8"))) is False

    tampered = _entry(good)
    (tmp_path / "payload.json").write_bytes(b"tampered")
    assert _search_receipt_complete(tampered) is False
    assert _search_max_requests(tampered) == 20

    big = _entry(json.dumps({"reported_total": 250, "pages": []}).encode("utf-8"))
    assert _search_max_requests(big) == 3
    assert _search_max_requests(_entry(json.dumps({"reported_total": -1, "pages": []}).encode("utf-8"))) == 20


def test_corrupt_search_payload_is_replanned_and_rejected(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.jobs.kind import KindNoticeSearchJob, kind_document_targets

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider())
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=1, pages=[kind_search_html(_rows(1), 1)])
    entry = ctx.catalog.latest(source=KIND_NOTICE_SEARCH_SOURCE,
                               natural_keys={"상장폐지:2019-07-01..2019-09-30"})["상장폐지:2019-07-01..2019-09-30"]
    Path(entry.payload_path).write_bytes(b"tampered")

    assert "상장폐지:2019-07-01..2019-09-30" in {unit.natural_key for unit in KindNoticeSearchJob().pending(ctx)}
    with pytest.raises(PITDataError, match="hash mismatch"):
        kind_document_targets(ctx.catalog, _provider().kind)


def test_fetch_rejects_contradictory_totals_and_duplicates(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.jobs.kind import KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    rows = _rows(150)
    pages = {
        ("상장폐지", "2019-07-01", "2019-09-30", 1): kind_search_html(rows[:100], 150),
        ("상장폐지", "2019-07-01", "2019-09-30", 2): kind_search_html(rows[100:], 999),
    }
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(pages=pages))
    (unit,) = [
        unit for unit in KindNoticeSearchJob().pending(ctx)
        if unit.natural_key == "상장폐지:2019-07-01..2019-09-30"
    ]
    with pytest.raises(PITDataError, match="contradictory"):
        KindNoticeSearchJob().fetch(ctx, [unit])

    dup_rows = [kind_row_html("20240101000001", "2024-01-02 15:30", "상장폐지", "005930", "유가증권시장본부")] * 2
    dup_pages = {("상장폐지", "2019-07-01", "2019-09-30", 1): kind_search_html(dup_rows, 2)}
    dup_ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(pages=dup_pages))
    with pytest.raises(PITDataError, match="duplicate"):
        KindNoticeSearchJob().fetch(dup_ctx, [unit])


def test_document_pending_skips_answered_and_fetch_needs_a_target(tmp_path: Path) -> None:
    import pytest

    from src.core.pit import EvidenceKind, PITDataError
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE
    from src.data.jobs.kind import KindNoticeDocumentJob
    from src.data.receipt_catalog import EvidenceStatus
    from src.data.scoped_ingestion import ScopedRawPayload

    runtime = _runtime(tmp_path)
    rows = _target_rows()
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(documents={}))
    _persist_search(ctx, keyword="상장폐지", start="2019-07-01", end="2019-09-30",
                    reported_total=6, pages=[kind_search_html(rows, 6)])
    assert len(KindNoticeDocumentJob().pending(ctx)) == 3

    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.DISCLOSURES,
            source=KIND_NOTICE_DOCUMENT_SOURCE,
            natural_key="20240101000001",
            as_of=date(2024, 1, 2),
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=json.dumps({"acptno": "20240101000001"}).encode("utf-8"),
            retrieved_at=NOW,
            source_label="test:20240101000001",
        )
    )
    assert [unit.natural_key for unit in KindNoticeDocumentJob().pending(ctx)] == [
        "20240101000003", "20240101000004",
    ]

    from src.data.jobs.runner import JobUnit

    with pytest.raises(PITDataError, match="no search row"):
        KindNoticeDocumentJob().fetch(
            ctx, [JobUnit(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_key="20000101000000",
                           payload={"acptno": "20000101000000"}, max_requests=3)])


def test_job_health_checks_delegate_to_collector(tmp_path: Path) -> None:
    from src.data.jobs.kind import KindNoticeDocumentJob, KindNoticeSearchJob

    runtime = _runtime(tmp_path)
    ctx = _ctx(runtime, _provider(), collector=_FakeKindClient(documents={}))

    for spec in (KindNoticeSearchJob(), KindNoticeDocumentJob()):
        spec.health_check(ctx)
