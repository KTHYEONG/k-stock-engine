"""Relay producer invariants: window coverage, retry discipline, body omissions."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from tests.fixtures.kind_html import kind_row_html, kind_search_html

TODAY = date(2026, 10, 13)
NOW = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
EVIDENCE_2019 = date(2019, 1, 1)
EVIDENCE_Q3 = date(2026, 7, 1)
OPEN_START = date(2026, 10, 1)


def _policy(keywords=("상장폐지",), lookback=14, doc_lookback=14, attempts=3):  # type: ignore[no-untyped-def]
    from src.config.providers import KindPolicy

    return KindPolicy(
        circuit_threshold=3,
        min_interval_seconds=1.0,
        daily_limit=5000,
        search_keywords=list(keywords),
        document_titles=["상장폐지", "관리종목지정", "관리종목지정해제"],
        max_stale_sessions=5,
        relay_window_lookback_days=lookback,
        relay_document_lookback_days=doc_lookback,
        relay_max_age_hours=36,
        relay_clock_skew_seconds=300,
        relay_attempts=attempts,
    )


def _row(acptno: str, disclosed: str, title: str = "상장폐지") -> str:
    return kind_row_html(acptno, disclosed, title, "005930", "유가증권시장본부")


class _FakeCollector:
    """Scripted KIND surface: per-window page sequences plus documents and failures."""

    def __init__(self) -> None:
        self.scripts: dict[tuple[str, str, str], list[str]] = {}
        self.documents: dict[str, tuple[str, str, str]] = {}
        self.failures: set[str] = set()
        self.search_calls: list[tuple] = []
        self.document_calls: list[str] = []

    def script(self, keyword: str, start: date, end: date, htmls: list[str]) -> None:
        self.scripts[(keyword, start.isoformat(), end.isoformat())] = list(htmls)

    def search_page(self, keyword, start, end, *, page_index):  # type: ignore[no-untyped-def]
        self.search_calls.append((keyword, start, end, page_index))
        seq = self.scripts.get((keyword, start.isoformat(), end.isoformat()))
        if seq is None:
            return kind_search_html([], 0)
        attempt = sum(1 for call in self.search_calls if call[:3] == (keyword, start, end))
        return seq[min(attempt, len(seq)) - 1]

    def fetch_document(self, acptno):  # type: ignore[no-untyped-def]
        from src.integrations.errors import ProviderRetryableError
        from src.integrations.krx.kind import KindNoticeDocument

        self.document_calls.append(acptno)
        if acptno in self.failures:
            raise ProviderRetryableError("boom", provider="KIND", endpoint="common/disclsviewer.do")
        doc_no, template, html = self.documents[acptno]
        return KindNoticeDocument(acptno=acptno, doc_no=doc_no, template=template, html=html)


def _window_parts(natural_key: str) -> tuple[str, str, str]:
    keyword, _, window = natural_key.partition(":")
    start, _, end = window.partition("..")
    return keyword, start, end


def test_bundle_contains_open_and_recent_closed_windows_per_keyword() -> None:
    from src.data.jobs.dart import disclosure_windows
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy(keywords=("상장폐지", "관리종목"))
    bundle = build_relay_bundle(
        _FakeCollector(), policy=policy, evidence_start=EVIDENCE_2019, now=NOW, producer={}, clock=lambda: NOW
    )
    assert bundle.fetch_date_kst == TODAY
    assert bundle.fetched_at == NOW
    expected = [
        (start.isoformat(), end.isoformat())
        for start, end in disclosure_windows(EVIDENCE_2019, TODAY)
        if end >= TODAY - timedelta(days=14)
    ]
    assert expected
    assert expected[-1][1] == TODAY.isoformat()
    for keyword in ("상장폐지", "관리종목"):
        got = sorted(
            (start, end)
            for name, start, end in (_window_parts(item.natural_key) for item in bundle.items)
            if name == keyword
        )
        assert got == expected
    assert bundle.producer["document_failures"] == "0"


def test_bundle_items_verify_with_shared_verifiers() -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.jobs.kind import verify_document_payload, verify_search_payload
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy()
    collector = _FakeCollector()
    acptno = "20261012000001"
    collector.script("상장폐지", OPEN_START, TODAY, [kind_search_html([_row(acptno, "2026-10-12 15:30")], 1)])
    collector.documents[acptno] = ("7", "a.htm", "<html>body</html>")
    bundle = build_relay_bundle(
        collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={"run_id": "t"}, clock=lambda: NOW
    )
    assert len(bundle.items) == 3
    assert bundle.producer["run_id"] == "t"
    for item in bundle.items:
        if item.source == KIND_NOTICE_SEARCH_SOURCE:
            verify_search_payload(item, fetch_date=TODAY)
        else:
            verify_document_payload(item)


def test_inconsistent_open_window_is_retried_then_fails_closed() -> None:
    import pytest

    from src.core.pit import PITDataError
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy(attempts=3)
    collector = _FakeCollector()
    collector.script(
        "상장폐지", OPEN_START, TODAY, [kind_search_html([_row("20261012000001", "2026-10-12 15:30")], 2)]
    )
    with pytest.raises(PITDataError):
        build_relay_bundle(
            collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW
        )
    assert sum(1 for call in collector.search_calls if call[1] == OPEN_START) == 3


def test_inconsistency_that_resolves_on_retry_is_accepted() -> None:
    from src.data.evidence_sources import KIND_NOTICE_SEARCH_SOURCE
    from src.data.jobs.kind import verify_search_payload
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy(attempts=3)
    collector = _FakeCollector()
    bad = kind_search_html([_row("20261012000001", "2026-10-12 15:30")], 2)
    good = kind_search_html([_row("20261012000001", "2026-10-12 15:30")], 1)
    collector.script("상장폐지", OPEN_START, TODAY, [bad, good])
    collector.documents["20261012000001"] = ("7", "a.htm", "<html>body</html>")
    bundle = build_relay_bundle(
        collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW
    )
    assert sum(1 for call in collector.search_calls if call[1] == OPEN_START) == 2
    assert bundle.producer["document_failures"] == "0"
    for item in bundle.items:
        if item.source == KIND_NOTICE_SEARCH_SOURCE:
            verify_search_payload(item, fetch_date=TODAY)


def test_document_failure_is_omitted_not_fatal() -> None:
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy()
    collector = _FakeCollector()
    first, second = "20261012000001", "20261012000002"
    collector.script(
        "상장폐지",
        OPEN_START,
        TODAY,
        [kind_search_html([_row(first, "2026-10-12 15:30"), _row(second, "2026-10-12 15:31")], 2)],
    )
    collector.failures.add(first)
    collector.documents[second] = ("7", "a.htm", "<html>body</html>")
    bundle = build_relay_bundle(
        collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW
    )
    bodies = {item.natural_key for item in bundle.items if item.source == KIND_NOTICE_DOCUMENT_SOURCE}
    assert bodies == {second}
    assert bundle.producer["document_failures"] == "1"


def test_old_notices_get_no_bodies() -> None:
    from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy(doc_lookback=14)
    collector = _FakeCollector()
    collector.script(
        "상장폐지", OPEN_START, TODAY, [kind_search_html([_row("20260913000001", "2026-09-13 15:30")], 1)]
    )
    collector.documents["20260913000001"] = ("7", "a.htm", "<html>body</html>")
    bundle = build_relay_bundle(
        collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW
    )
    assert collector.document_calls == []
    assert all(item.source != KIND_NOTICE_DOCUMENT_SOURCE for item in bundle.items)


def test_retrieved_at_never_exceeds_fetched_at() -> None:
    from src.data.kind_relay_producer import build_relay_bundle

    policy = _policy()
    collector = _FakeCollector()
    acptno = "20261012000001"
    collector.script("상장폐지", OPEN_START, TODAY, [kind_search_html([_row(acptno, "2026-10-12 15:30")], 1)])
    collector.documents[acptno] = ("7", "a.htm", "<html>body</html>")
    bundle = build_relay_bundle(
        collector, policy=policy, evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW
    )
    assert bundle.items
    assert all(item.retrieved_at <= bundle.fetched_at for item in bundle.items)


def test_invalid_time_or_scope_fails_closed() -> None:
    import pytest
    from src.core.pit import PITDataError
    from src.data.kind_relay_producer import build_relay_bundle

    for moment, start in [(NOW.replace(tzinfo=None), EVIDENCE_Q3), (NOW, TODAY + timedelta(days=1))]:
        collector = _FakeCollector()
        with pytest.raises(PITDataError):
            build_relay_bundle(collector, policy=_policy(), evidence_start=start, now=moment, producer={})
        assert collector.search_calls == []


def test_invalid_document_is_counted_and_programming_errors_propagate() -> None:
    import pytest
    from src.data.kind_relay_producer import build_relay_bundle

    collector = _FakeCollector()
    acptno = "20261012000001"
    collector.script("상장폐지", OPEN_START, TODAY, [kind_search_html([_row(acptno, "2026-10-12 15:30")], 1)])
    collector.documents[acptno] = ("7", "a.htm", "")
    bundle = build_relay_bundle(collector, policy=_policy(), evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW)
    assert bundle.producer["document_failures"] == "1"
    del collector.documents[acptno]
    with pytest.raises(KeyError):
        build_relay_bundle(collector, policy=_policy(), evidence_start=EVIDENCE_Q3, now=NOW, producer={}, clock=lambda: NOW)


def test_each_item_is_stamped_after_collection_and_bundle_after_all_items() -> None:
    from datetime import timezone
    from src.data.kind_relay import decode_relay_bundle, encode_relay_bundle
    from src.data.kind_relay_producer import build_relay_bundle

    collector = _FakeCollector()
    acptno = "20261012000001"
    collector.script("상장폐지", OPEN_START, TODAY, [kind_search_html([_row(acptno, "2026-10-12 15:30")], 1)])
    collector.documents[acptno] = ("7", "a.htm", "<html>body</html>")
    moments = [NOW + timedelta(seconds=second) for second in range(1, 5)]
    readings = iter(moments)

    def clock() -> datetime:
        moment = next(readings)
        if moment == moments[0]:
            assert len(collector.search_calls) == 1
        elif moment == moments[1]:
            assert len(collector.search_calls) == 2
        else:
            assert collector.document_calls == [acptno]
        return moment

    bundle = build_relay_bundle(
        collector, policy=_policy(), evidence_start=EVIDENCE_Q3,
        now=NOW.astimezone(timezone(timedelta(hours=9))), producer={}, clock=clock,
    )
    assert [item.retrieved_at for item in bundle.items] == moments[:3]
    assert bundle.fetched_at == moments[3]
    assert bundle.fetched_at.utcoffset() == timedelta(0)
    assert bundle.fetch_date_kst == TODAY
    assert decode_relay_bundle(encode_relay_bundle(bundle)).fetched_at == moments[3]


def test_clock_faults_in_documents_or_completion_fail_the_entire_build() -> None:
    import pytest
    from src.core.pit import PITDataError
    from src.data.kind_relay_producer import build_relay_bundle

    for faulty in (NOW.replace(tzinfo=None), NOW - timedelta(seconds=1), NOW + timedelta(days=1)):
        for fault_index in range(4):
            collector = _FakeCollector()
            acptno = "20261012000001"
            collector.script("상장폐지", OPEN_START, TODAY, [kind_search_html([_row(acptno, "2026-10-12 15:30")], 1)])
            collector.documents[acptno] = ("7", "a.htm", "<html>body</html>")
            moments = [NOW] * 4
            moments[fault_index] = faulty
            readings = iter(moments)
            with pytest.raises(PITDataError, match=r"clock|date boundary"):
                build_relay_bundle(
                    collector, policy=_policy(), evidence_start=EVIDENCE_Q3,
                    now=NOW, producer={}, clock=lambda readings=readings: next(readings),
                )


def test_completion_rejects_kst_midnight_crossing() -> None:
    import pytest
    from src.core.pit import PITDataError
    from src.data.kind_relay_producer import build_relay_bundle

    start = datetime(2026, 10, 13, 14, 59, 59, tzinfo=UTC)
    readings = iter([start, start, start + timedelta(seconds=1)])
    with pytest.raises(PITDataError, match="KST date boundary"):
        build_relay_bundle(
            _FakeCollector(), policy=_policy(), evidence_start=EVIDENCE_Q3,
            now=start, producer={}, clock=lambda: next(readings),
        )


def test_collection_error_in_closed_window_is_retried() -> None:
    from src.data.kind_relay_producer import build_relay_bundle

    collector = _FakeCollector()
    collector.script("상장폐지", EVIDENCE_Q3, date(2026, 9, 30), ["malformed", kind_search_html([], 0)])
    bundle = build_relay_bundle(
        collector, policy=_policy(), evidence_start=EVIDENCE_Q3,
        now=NOW, producer={}, clock=lambda: NOW,
    )
    assert len(bundle.items) == 2
    assert sum(call[1] == EVIDENCE_Q3 for call in collector.search_calls) == 2
