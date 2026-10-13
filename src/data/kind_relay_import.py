"""VPS-side KIND relay import: verify a producer bundle and persist its passing items."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

from src.config.providers import ProviderPolicy
from src.core.pit import PITDataError
from src.data.evidence_sources import KIND_NOTICE_DOCUMENT_SOURCE, KIND_NOTICE_SEARCH_SOURCE
from src.data.jobs.kind import kind_document_targets, verify_document_payload, verify_search_payload
from src.data.kind_relay import RELAY_MAX_BYTES, RelayBundle, check_relay_freshness, decode_relay_bundle
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

__all__ = [
    "RelayImportReport",
    "import_relay_bundle",
    "relay_inbox_path",
]

_LOG = logging.getLogger(__name__)

_RELAY_JOBS = frozenset({"kind_notice_search", "kind_notice_documents"})
_ANSWERED = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY, EvidenceStatus.EXTRACTION_FAILED})


@dataclass(frozen=True, slots=True)
class RelayImportReport:
    """Outcome of importing one bundle for one KIND step."""

    job: str
    status: str
    imported: int
    unchanged: int
    rejected: int
    pending_left: int
    fetched_at: datetime | None


def relay_inbox_path(state_root: Path) -> Path:
    """Fixed bundle location: ``<state_root>/kind_relay/inbox/bundle.json``."""
    return Path(state_root) / "kind_relay" / "inbox" / "bundle.json"


def _accepted_state_path(state_root: Path) -> Path:
    return Path(state_root) / "kind_relay" / "last_accepted.json"


def _read_last_accepted_at(state_root: Path) -> datetime | None:
    try:
        document = json.loads(_accepted_state_path(state_root).read_bytes())
        moment = datetime.fromisoformat(document["fetched_at"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None
    return moment if moment.tzinfo is not None else None


def _write_accepted_state(state_root: Path, *, fetched_at: datetime, items_sha256: str) -> None:
    path = _accepted_state_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        {"fetched_at": fetched_at.isoformat(), "items_sha256": items_sha256},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = _read_last_accepted_at(state_root)
        if current is not None and fetched_at < current:
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)


def _split_window_key(natural_key: str) -> tuple[str, date, date]:
    keyword, _, window = natural_key.partition(":")
    start_raw, _, end_raw = window.partition("..")
    try:
        return keyword, date.fromisoformat(start_raw), date.fromisoformat(end_raw)
    except ValueError as exc:
        raise PITDataError(f"relay item has a malformed window key {natural_key!r}") from exc


def _relabel(item: ScopedRawPayload, *, fetched_at: datetime, now: datetime) -> ScopedRawPayload:
    return replace(
        item,
        retrieved_at=now,
        source_label=f"{item.source_label}|relay:fetched_at={fetched_at.astimezone(UTC).isoformat()}",
    )


def _import_search(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    bundle: RelayBundle,
    items_sha256: str,
    now: datetime,
    dry_run: bool,
) -> RelayImportReport:
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    keywords = set(provider.kind.search_keywords)
    evidence_start = runtime.scope.evidence_start
    items = sorted(
        (item for item in bundle.items if item.source == KIND_NOTICE_SEARCH_SOURCE),
        key=lambda item: item.natural_key,
    )
    latest = catalog.latest(
        source=KIND_NOTICE_SEARCH_SOURCE, natural_keys={item.natural_key for item in items}
    )
    to_store: list[ScopedRawPayload] = []
    covered: set[str] = set()
    rejected = 0
    unchanged = 0
    for item in items:
        try:
            verify_search_payload(item, fetch_date=bundle.fetch_date_kst)
            keyword, start, end = _split_window_key(item.natural_key)
            if keyword not in keywords or start < evidence_start:
                raise PITDataError(f"relay window {item.natural_key!r} is outside the collection scope")
            writer.validate(_relabel(item, fetched_at=bundle.fetched_at, now=now))
        except (PITDataError, ValueError, TypeError, OverflowError, RecursionError):
            rejected += 1
            continue
        entry = latest.get(item.natural_key)
        if (
            entry is not None
            and entry.status in _ANSWERED
            and entry.content_hash == hashlib.sha256(item.payload).hexdigest()
        ):
            unchanged += 1
        else:
            to_store.append(item)
        if end == bundle.fetch_date_kst:
            covered.add(keyword)
    if to_store and not dry_run:
        writer.persist_many(
            [_relabel(item, fetched_at=bundle.fetched_at, now=now) for item in to_store]
        )
    if not dry_run:
        _write_accepted_state(
            runtime.workspace.state_root, fetched_at=bundle.fetched_at, items_sha256=items_sha256
        )
    missing = sum(1 for keyword in provider.kind.search_keywords if keyword not in covered)
    status = "dry_run" if dry_run else ("complete" if missing == 0 and rejected == 0 else "relay_partial")
    _LOG.info(
        "[DATA] stage=relay_import job=kind_notice_search status=%s imported=%d unchanged=%d rejected=%d",
        status,
        len(to_store),
        unchanged,
        rejected,
    )
    return RelayImportReport(
        job="kind_notice_search",
        status=status,
        imported=len(to_store),
        unchanged=unchanged,
        rejected=rejected,
        pending_left=missing,
        fetched_at=bundle.fetched_at,
    )


def _import_documents(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    bundle: RelayBundle,
    now: datetime,
    dry_run: bool,
) -> RelayImportReport:
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    wanted = {row.acptno for row in kind_document_targets(catalog, provider.kind)}
    items = sorted(
        (
            item
            for item in bundle.items
            if item.source == KIND_NOTICE_DOCUMENT_SOURCE and item.natural_key in wanted
        ),
        key=lambda item: item.natural_key,
    )
    latest = catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys=set(wanted))
    to_store: list[ScopedRawPayload] = []
    rejected = 0
    unchanged = 0
    for item in items:
        try:
            verify_document_payload(item)
            writer.validate(_relabel(item, fetched_at=bundle.fetched_at, now=now))
        except (PITDataError, ValueError, TypeError, OverflowError, RecursionError):
            rejected += 1
            continue
        entry = latest.get(item.natural_key)
        if (
            entry is not None
            and entry.status in _ANSWERED
            and entry.content_hash == hashlib.sha256(item.payload).hexdigest()
        ):
            unchanged += 1
        else:
            to_store.append(item)
    if to_store and not dry_run:
        writer.persist_many(
            [_relabel(item, fetched_at=bundle.fetched_at, now=now) for item in to_store]
        )
    if not dry_run:
        latest = catalog.latest(source=KIND_NOTICE_DOCUMENT_SOURCE, natural_keys=set(wanted))
    simulated_answers = {item.natural_key for item in to_store} if dry_run else set()
    pending_left = sum(
        1
        for acptno in wanted
        if acptno not in simulated_answers
        and ((entry := latest.get(acptno)) is None or entry.status not in _ANSWERED)
    )
    status = (
        "dry_run"
        if dry_run
        else ("complete" if pending_left == 0 and rejected == 0 else "relay_partial")
    )
    _LOG.info(
        "[DATA] stage=relay_import job=kind_notice_documents status=%s imported=%d unchanged=%d rejected=%d",
        status,
        len(to_store),
        unchanged,
        rejected,
    )
    return RelayImportReport(
        job="kind_notice_documents",
        status=status,
        imported=len(to_store),
        unchanged=unchanged,
        rejected=rejected,
        pending_left=pending_left,
        fetched_at=bundle.fetched_at,
    )


def import_relay_bundle(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    job: str,
    now: datetime,
    dry_run: bool = False,
) -> RelayImportReport:
    """Verify the inbox bundle and persist its passing items for one KIND step; never raises for bad input."""
    if job not in _RELAY_JOBS:
        raise PITDataError(f"unknown relay job {job!r}: expected one of {sorted(_RELAY_JOBS)}")
    inbox = relay_inbox_path(runtime.workspace.state_root)
    try:
        with inbox.open("rb") as stream:
            raw = stream.read(RELAY_MAX_BYTES + 1)
    except FileNotFoundError:
        return RelayImportReport(
            job=job, status="relay_absent", imported=0, unchanged=0,
            rejected=0, pending_left=0, fetched_at=None,
        )
    except OSError:
        return RelayImportReport(
            job=job, status="relay_rejected", imported=0, unchanged=0,
            rejected=0, pending_left=0, fetched_at=None,
        )
    try:
        bundle = decode_relay_bundle(raw)
        check_relay_freshness(
            bundle, now=now, policy=provider.kind,
            last_accepted_at=_read_last_accepted_at(runtime.workspace.state_root),
        )
        items_sha256 = str(json.loads(raw.decode("utf-8"))["items_sha256"])
    except Exception:
        return RelayImportReport(
            job=job, status="relay_rejected", imported=0, unchanged=0,
            rejected=0, pending_left=0, fetched_at=None,
        )
    if job == "kind_notice_search":
        return _import_search(
            runtime=runtime, provider=provider, bundle=bundle,
            items_sha256=items_sha256, now=now, dry_run=dry_run,
        )
    return _import_documents(
        runtime=runtime, provider=provider, bundle=bundle, now=now, dry_run=dry_run,
    )
