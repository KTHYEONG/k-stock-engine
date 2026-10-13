"""Off-VPS KIND relay assembly: recent search windows plus notice bodies as one bundle."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

from src.config.providers import KindPolicy
from src.core.pit import PITDataError
from src.data.jobs.dart import disclosure_windows
from src.data.jobs.kind import (
    KindPageCollector,
    collect_notice_document,
    collect_search_window,
    select_document_targets,
    verify_document_payload,
    verify_search_payload,
)
from src.data.kind_relay import RelayBundle
from src.data.scoped_ingestion import ScopedRawPayload
from src.integrations.errors import ProviderError
from src.integrations.krx.kind import KindNoticeRow, parse_kind_search_page

__all__ = ["build_relay_bundle"]

_KST = timedelta(hours=9)


def _kst_today(now: datetime) -> date:
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) + _KST).date()


def _window_rows(items: list[ScopedRawPayload]) -> list[KindNoticeRow]:
    rows: list[KindNoticeRow] = []
    for item in items:
        document = json.loads(item.payload)
        for page in document["pages"]:
            rows.extend(parse_kind_search_page(page).rows)
    return rows


def build_relay_bundle(
    collector: KindPageCollector,
    *,
    policy: KindPolicy,
    evidence_start: date,
    now: datetime,
    producer: Mapping[str, str],
    clock: Callable[[], datetime] | None = None,
) -> RelayBundle:
    """Collect the recent KIND windows and notice bodies and return a verified bundle; raises PITDataError."""
    if now.utcoffset() is None:
        raise PITDataError("KIND relay requires a timezone-aware timestamp")
    started_at = now.astimezone(UTC)
    today = _kst_today(started_at)
    if evidence_start > today:
        raise PITDataError("KIND relay evidence start is after the collection date")
    read_clock = clock or (lambda: datetime.now(UTC))
    last_observed = started_at

    def observe() -> datetime:
        nonlocal last_observed
        moment = read_clock()
        if moment.utcoffset() is None:
            raise PITDataError("KIND relay clock returned a naive timestamp")
        moment = moment.astimezone(UTC)
        if moment < last_observed:
            raise PITDataError("KIND relay clock regressed during collection")
        if _kst_today(moment) != today:
            raise PITDataError("KIND relay collection crossed a KST date boundary")
        last_observed = moment
        return moment
    window_cutoff = today - timedelta(days=int(policy.relay_window_lookback_days))
    attempts = max(1, int(policy.relay_attempts))
    search_items: list[ScopedRawPayload] = []
    for keyword in dict.fromkeys(policy.search_keywords):
        for start, end in disclosure_windows(evidence_start, today):
            if end < window_cutoff:
                continue
            last_error: PITDataError | None = None
            payload: ScopedRawPayload | None = None
            for _ in range(attempts):
                try:
                    candidate = collect_search_window(
                        collector, keyword=keyword, start=start, end=end, today=today, retrieved_at=started_at
                    )
                except PITDataError as exc:
                    last_error = exc
                    continue
                candidate = replace(candidate, retrieved_at=observe())
                try:
                    verify_search_payload(candidate, fetch_date=today)
                except PITDataError as exc:
                    last_error = exc
                    continue
                payload = candidate
                break
            if payload is None:
                raise PITDataError(
                    f"KIND relay window {keyword}:{start.isoformat()}..{end.isoformat()} failed verification"
                ) from last_error
            search_items.append(payload)
    document_cutoff = today - timedelta(days=int(policy.relay_document_lookback_days))
    targets = [
        row
        for row in select_document_targets(_window_rows(search_items), policy)
        if row.disclosed_at.date() >= document_cutoff
    ]
    documents: list[ScopedRawPayload] = []
    failures = 0
    for row in targets:
        try:
            body = collect_notice_document(collector, row, retrieved_at=started_at)
        except (PITDataError, ProviderError):
            failures += 1
            continue
        body = replace(body, retrieved_at=observe())
        try:
            verify_document_payload(body)
        except PITDataError:
            failures += 1
            continue
        documents.append(body)
    stamped = dict(producer)
    stamped["document_failures"] = str(failures)
    return RelayBundle(
        fetched_at=observe(), fetch_date_kst=today, producer=stamped, items=tuple(search_items + documents)
    )
