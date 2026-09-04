"""Multi-receipt Bronze discovery and content-addressed aggregation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from src.data.bronze import BronzeStore
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

_SMALL_STREAMING_KINDS: frozenset[EvidenceKind] = frozenset(
    {EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER}
)
_DERIVED_PREFIXES: tuple[str, ...] = ("aggregated:", "merged:", "manifest:")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a payload digest without materializing the file in memory."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PITDataError(f"missing Bronze payload for {path}") from exc
    return digest.hexdigest()


def select_streaming_receipts(
    *, kind: EvidenceKind, receipts: tuple[BronzeReceipt, ...]
) -> tuple[BronzeReceipt, ...]:
    """Exclude redundant materialized inputs when original receipts exist."""
    ordered = tuple(sorted(receipts, key=lambda item: (item.retrieved_at, item.content_hash)))
    original = tuple(
        item for item in ordered if not item.source_path.startswith(_DERIVED_PREFIXES)
    )
    if original:
        selected = original
    else:
        aggregates = tuple(item for item in ordered if item.source_path.startswith("aggregated:"))
        selected = aggregates[-1:] if kind not in _SMALL_STREAMING_KINDS and aggregates else ordered
    if not selected:
        raise PITDataError(f"no Bronze receipts selected for {kind.value}")
    return selected


def discover_verified_bronze_receipts(
    *, bronze_root: Path
) -> dict[EvidenceKind, tuple[BronzeReceipt, ...]]:
    """Verify every receipt payload and return all pages in stable order."""
    root = Path(bronze_root)
    grouped: dict[EvidenceKind, tuple[BronzeReceipt, ...]] = {}
    for kind in EvidenceKind:
        kind_dir = root / kind.value
        if not kind_dir.exists():
            continue
        receipts: list[BronzeReceipt] = []
        metadata_rows: list[tuple[Path, dict[str, object], Path]] = []
        for receipt_path in sorted(kind_dir.rglob("receipt.json")):
            try:
                meta = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PITDataError(f"malformed Bronze receipt {receipt_path}: {exc}") from exc
            if not isinstance(meta, dict):
                raise PITDataError(f"malformed Bronze receipt {receipt_path}")
            payload_path = receipt_path.parent / "payload.json"
            if not payload_path.exists():
                raise PITDataError(f"missing Bronze payload for {receipt_path}")
            content_hash = meta.get("content_hash")
            if not isinstance(content_hash, str) or not content_hash:
                raise PITDataError(f"malformed Bronze receipt {receipt_path}")
            meta_kind = meta.get("kind")
            if isinstance(meta_kind, str) and meta_kind and meta_kind != kind.value:
                raise PITDataError(f"kind mismatch in Bronze receipt {receipt_path}")
            try:
                retrieved_at = datetime.fromisoformat(str(meta["retrieved_at"]))
            except (KeyError, ValueError) as exc:
                raise PITDataError(f"malformed Bronze receipt {receipt_path}") from exc
            try:
                ingested_at = datetime.fromisoformat(str(meta["ingested_at"]))
            except (KeyError, ValueError) as exc:
                raise PITDataError(f"malformed Bronze receipt {receipt_path}") from exc
            metadata_rows.append((receipt_path, meta, payload_path))
        has_original = any(
            not str(meta.get("source_path", "")).startswith(_DERIVED_PREFIXES)
            for _, meta, _ in metadata_rows
        )
        for receipt_path, meta, payload_path in metadata_rows:
            if has_original and str(meta.get("source_path", "")).startswith(_DERIVED_PREFIXES):
                continue
            content_hash = str(meta["content_hash"])
            computed = sha256_file(payload_path)
            if computed != content_hash:
                raise PITDataError(f"hash mismatch for Bronze payload {payload_path}")
            retrieved_at = datetime.fromisoformat(str(meta["retrieved_at"]))
            ingested_at = datetime.fromisoformat(str(meta["ingested_at"]))
            receipts.append(
                BronzeReceipt(
                    kind=kind,
                    content_hash=content_hash,
                    source_path=str(meta.get("source_path", "")),
                    retrieved_at=retrieved_at,
                    ingested_at=ingested_at,
                    payload_path=payload_path,
                    metadata_path=receipt_path,
                )
            )
        if receipts:
            ordered = sorted(receipts, key=lambda r: (r.retrieved_at, r.content_hash))
            grouped[kind] = tuple(ordered)
    return grouped


def aggregate_small_bronze_pages(
    *,
    kind: EvidenceKind,
    receipts: tuple[BronzeReceipt, ...],
    store: BronzeStore,
) -> BronzeReceipt:
    """Aggregate small JSON pages preserving every source page object."""
    if not receipts:
        raise PITDataError(f"no Bronze receipts to aggregate for {kind.value}")
    if kind in _SMALL_STREAMING_KINDS:
        raise PITDataError(f"streaming kind {kind.value} must not be aggregated into one JSON list")
    ordered = tuple(sorted(receipts, key=lambda r: (r.retrieved_at, r.content_hash)))
    input_hashes = [item.content_hash for item in ordered]
    pages: list[dict[str, object]] = []
    merged_records: list[object] = []
    merged_sessions: list[object] = []
    merged_intervals: list[object] = []
    merged_list: list[object] = []
    commission: object = None
    has_commission = False
    for receipt in ordered:
        try:
            raw = receipt.payload_path.read_bytes()
        except OSError as exc:
            raise PITDataError(f"missing Bronze payload for {kind.value}") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PITDataError(f"invalid Bronze payload for {kind.value}") from exc
        if isinstance(payload, dict):
            page: dict[str, object] = dict(payload)
            page["source_receipt_hash"] = receipt.content_hash
            page.setdefault("retrieved_at", receipt.retrieved_at.isoformat())
            pages.append(page)
            records = payload.get("records")
            if isinstance(records, list):
                merged_records.extend(records)
            sessions = payload.get("sessions")
            if isinstance(sessions, list):
                merged_sessions.extend(sessions)
            intervals = payload.get("intervals")
            if isinstance(intervals, list):
                merged_intervals.extend(intervals)
            items = payload.get("list")
            if isinstance(items, list):
                merged_list.extend(items)
            if not has_commission and payload.get("commission") is not None:
                commission = payload.get("commission")
                has_commission = True
        elif isinstance(payload, list):
            pages.append(
                {
                    "records": list(payload),
                    "source_receipt_hash": receipt.content_hash,
                    "retrieved_at": receipt.retrieved_at.isoformat(),
                }
            )
            merged_records.extend(payload)
        else:
            raise PITDataError(f"invalid Bronze payload for {kind.value}")
    aggregate: dict[str, object] = {
        "kind": kind.value,
        "input_receipt_hashes": list(input_hashes),
        "pages": pages,
    }
    if merged_records:
        aggregate["records"] = merged_records
    if merged_sessions:
        aggregate["sessions"] = sorted({str(value) for value in merged_sessions})
    if merged_intervals:
        aggregate["intervals"] = merged_intervals
    if merged_list:
        aggregate["list"] = merged_list
    if has_commission:
        aggregate["commission"] = commission
    canonical = json.dumps(aggregate, sort_keys=True, ensure_ascii=False).encode("utf-8")
    latest = max(item.retrieved_at for item in ordered)
    return store.import_bytes(
        canonical,
        kind=kind,
        retrieved_at=latest,
        source_label=f"aggregated:{kind.value}:{len(ordered)}",
    )
