"""Verify and register Bronze pages that a remote DART worker collected."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.data.collection import dart_fact_scoped_payload
from src.data.dart_documents import DartDocumentStore
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.quota import ProviderQuotaStateStore

_BATCH = 500


@dataclass(frozen=True, slots=True)
class InboxResult:
    """Outcome of one ingest pass.

    Attributes:
        accepted: Page directories (``<kind>/<hash>``) now safely local; only these may be deleted remotely.
        rejected: Directories that failed verification and must be kept for inspection.
    """

    accepted: tuple[str, ...]
    rejected: tuple[str, ...]
    fact_pages: int
    document_pages: int


@dataclass(slots=True)
class _Tally:
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def _read_receipt(directory: Path) -> dict[str, object] | None:
    receipt = directory / "receipt.json"
    if not receipt.is_file():
        return None  # 원격에서 아직 쓰는 중인 페이지: 다음 회차에 처리한다.
    try:
        loaded = json.loads(receipt.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def ingest_remote_bronze(*, inbox_bronze: Path, bronze_root: Path, writer: ScopedBronzeWriter) -> InboxResult:
    """Register verified remote pages locally and report which may be deleted remotely.

    Documents are stored first because fact pages reference them. A page is accepted only when its
    payload hash equals its directory name and its receipt; anything else is rejected, never repaired.
    Re-running over already-ingested pages is idempotent.

    Args:
        inbox_bronze: Local copy of the remote ``out/bronze`` tree.
        bronze_root: Scope Bronze root receiving verified pages.
        writer: Scoped writer that also publishes catalog entries for fact pages.

    Returns:
        Accepted and rejected page directories with counts.
    """
    tally = _Tally()
    documents = 0
    document_store = DartDocumentStore(bronze_root)
    for directory in sorted((inbox_bronze / "dart_documents").glob("*")) if (inbox_bronze / "dart_documents").is_dir() else []:
        meta = _read_receipt(directory)
        payload_path = directory / "payload.zip"
        name = f"dart_documents/{directory.name}"
        if meta is None or not payload_path.is_file():
            continue
        raw = payload_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != directory.name or meta.get("sha256") != directory.name:
            tally.rejected.append(name)
            continue
        document_store.store_archive(raw, rcept_no=str(meta["rcept_no"]), retrieved_at=datetime.fromisoformat(str(meta["retrieved_at"])))
        tally.accepted.append(name)
        documents += 1

    facts = 0
    batch: list[tuple[str, ScopedRawPayload]] = []

    def flush() -> None:
        nonlocal facts
        if batch:
            writer.persist_many(tuple(payload for _, payload in batch))
            tally.accepted.extend(name for name, _ in batch)
            facts += len(batch)
            batch.clear()

    facts_dir = inbox_bronze / "financial_facts"
    for directory in sorted(facts_dir.glob("*")) if facts_dir.is_dir() else []:
        meta = _read_receipt(directory)
        payload_path = directory / "payload.json"
        name = f"financial_facts/{directory.name}"
        if meta is None or not payload_path.is_file():
            continue
        raw = payload_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != directory.name or meta.get("content_hash") != directory.name:
            tally.rejected.append(name)
            continue
        scoped = dart_fact_scoped_payload(page=json.loads(raw), retrieved_at=datetime.fromisoformat(str(meta["retrieved_at"])))
        if hashlib.sha256(scoped.payload).hexdigest() != directory.name:
            tally.rejected.append(name)  # 로컬 직렬화와 바이트가 다르면 같은 증거로 볼 수 없다.
            continue
        batch.append((name, scoped))
        if len(batch) >= _BATCH:
            flush()
    flush()
    return InboxResult(tuple(tally.accepted), tuple(tally.rejected), facts, documents)


def fold_remote_quota(*, store: ProviderQuotaStateStore, remote_state: Path, watermark_path: Path) -> int:
    """Add requests the remote worker made (per KST day) to the local ledger exactly once.

    Args:
        store: Local quota ledger for the shared key.
        remote_state: Copy of the remote ``quota_state.json``.
        watermark_path: JSON file remembering how much of each (provider, endpoint, day) was already folded.

    Returns:
        Number of requests newly folded.
    """
    remote = json.loads(remote_state.read_text(encoding="utf-8")) if remote_state.is_file() else {}
    marks: dict[str, int] = json.loads(watermark_path.read_text(encoding="utf-8")) if watermark_path.is_file() else {}
    folded = 0
    for key, entry in remote.items():
        day = entry.get("daily_attempt_day")
        count = int(entry.get("daily_attempted_requests", 0))
        if not day or "|" not in key:
            continue
        mark_key = f"{key}|{day}"
        delta = count - int(marks.get(mark_key, 0))
        if delta > 0:
            provider, endpoint = key.split("|", 1)
            store.add_attempts(provider=provider, endpoint=endpoint, day=str(day), count=delta)
            marks[mark_key] = count
            folded += delta
    watermark_path.parent.mkdir(parents=True, exist_ok=True)
    watermark_path.write_text(json.dumps(marks, sort_keys=True), encoding="utf-8")
    return folded
