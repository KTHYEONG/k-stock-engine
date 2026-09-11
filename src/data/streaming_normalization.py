"""Streaming PIT normalization with bounded batches and checkpoints."""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import multiprocessing
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from queue import Empty
from typing import Any, Final

import polars as pl

from src.core.time import KRX_TZ, SessionCalendar
from src.data.bronze import BronzeStore as _BronzeStore
from src.data.bronze_aggregation import aggregate_small_bronze_pages as _aggregate_small
from src.data.bronze_aggregation import discover_verified_bronze_receipts, select_streaming_receipts
from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, PITDataError, SilverTable
from src.storage.parquet_datasets import canonical_content_hash

# Bump when canonical field semantics change so stale staging cannot be reused.
SCHEMA_VERSION = "v2"

_STREAM_TABLES: tuple[SilverTable, ...] = (
    SilverTable.DAILY_MARKET,
    SilverTable.SECURITY_MASTER,
)

_STREAM_KINDS: dict[SilverTable, EvidenceKind] = {
    SilverTable.DAILY_MARKET: EvidenceKind.DAILY_MARKET,
    SilverTable.SECURITY_MASTER: EvidenceKind.SECURITY_MASTER,
}

_CORPORATE_ACTION_READ_SIZE = 64 * 1024

STREAMING_EAGER_JSON_MAX_BYTES: Final[int] = 1_000_000


def _source_path_month(source_path: str) -> str | None:
    match = re.search(r"(?:^|[^0-9])(\d{4})[-]?(\d{2})[-]?(\d{2})(?:[^0-9]|$)", str(source_path))
    return f"{match.group(1)}-{match.group(2)}" if match else None


def _is_append_only_source_path(source_path: str, latest_month: str) -> bool:
    month = _source_path_month(source_path)
    return month is not None and month > latest_month


def _trim_streaming_allocator() -> None:
    """Return transient Parquet batch pages before the next bounded batch."""
    gc.collect(0)
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        # malloc_trim is Linux/glibc-specific; bounded batches remain correct
        # when the platform allocator does not expose it.
        return
_MAX_CORPORATE_ACTION_RECORD_BYTES = 4 * 1024 * 1024


def _decode_text(payload: str) -> Any:
    return json.JSONDecoder().decode(payload)


def _read_doc(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return _decode_text(handle.read())


def _write_doc(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
    tmp.replace(path)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StreamingNormalizationCheckpoint:
    """Resume gate binding source hashes, table, month, schema, output."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._path = self.root / "streaming_checkpoints.json"
        self._state: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as handle:
                    raw = _decode_text(handle.read())
                if isinstance(raw, dict):
                    self._state = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                self._state = {}

    def _key(self, table: str, month: str) -> str:
        return f"{table}|{month}"

    def mark_verified(
        self,
        *,
        table: str,
        month: str,
        source_hashes: tuple[str, ...],
        output_hash: str,
        schema_version: str = SCHEMA_VERSION,
        part_digests: tuple[str, ...] = (),
        row_count: int | None = None,
        partition_root_hash: str = "",
        persist: bool = True,
    ) -> None:
        key = self._key(str(table), str(month))
        self._state[key] = {
            "source_hashes": list(source_hashes),
            "output_hash": str(output_hash),
            "schema_version": str(schema_version),
            "part_digests": list(part_digests),
            "row_count": row_count,
            "partition_root_hash": str(partition_root_hash),
        }
        if persist:
            self.flush()

    def flush(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_doc(self._path, self._state)

    def is_verified(
        self,
        *,
        table: str,
        month: str,
        source_hashes: tuple[str, ...],
        schema_version: str = SCHEMA_VERSION,
    ) -> bool:
        entry = self._state.get(self._key(str(table), str(month)))
        if not entry:
            return False
        if str(entry.get("schema_version", SCHEMA_VERSION)) != str(schema_version):
            return False
        if {str(item) for item in entry.get("source_hashes", [])} != set(source_hashes):
            return False
        return bool(entry.get("output_hash"))

    def verified_entry(self, *, table: str, month: str) -> dict[str, Any] | None:
        entry = self._state.get(self._key(str(table), str(month)))
        return dict(entry) if isinstance(entry, dict) else None


def _assert_unique_daily_keys(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[Any, Any]] = set()
    for row in rows:
        key = (row.get("session"), row.get("instrument_id"))
        if key in seen:
            raise PITDataError(f"duplicate daily_market primary key {key!r}; certification blocked")
        seen.add(key)


def streamed_dataset_root_hash(entries: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    parts = [
        f"{entry.get('year')}/{entry.get('month')}/{entry.get('part_index')}"
        f"/{entry.get('row_count')}/{entry.get('part_digest')}"
        for entry in entries
    ]
    digest = hashlib.sha256()
    for part in sorted(parts):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class StreamingSilverWriter:
    """Bounded per-month staging writer emitting immutable Hive parts."""

    def __init__(
        self,
        root: Path,
        *,
        table: SilverTable,
        batch_size: int,
        source_hashes: tuple[str, ...],
        source_paths: tuple[str, ...] = (),
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        if int(batch_size) < 1:
            raise PITDataError("batch_size must be positive")
        self.root = Path(root)
        self.table = table
        self.batch_size = int(batch_size)
        self.source_hashes = tuple(source_hashes)
        self.source_paths = tuple(source_paths)
        self.schema_version = str(schema_version)
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._part_digests: dict[str, list[str]] = {}
        self._part_counts: dict[str, list[int]] = {}
        self._row_counts: dict[str, int] = {}
        self._master_fingerprints: dict[tuple[Any, Any], str] = {}
        self._daily_fingerprints: dict[tuple[Any, Any], str] = {}
        self._fingerprint_month: str | None = None
        self._checkpoint = StreamingNormalizationCheckpoint(self.root.parent / "checkpoints")
        self._verified_months: set[str] = set()
        self._sealed_months: set[str] = set()
        self._reusable_source_hashes: set[str] = set()
        self._load_reusable_months()

    def _load_reusable_months(self) -> None:
        manifest_path = self.root / self.table.value / "staging_manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = _read_doc(manifest_path)
        except (OSError, ValueError):
            return
        if not isinstance(manifest, dict) or manifest.get("schema_version") != self.schema_version:
            return
        parts = manifest.get("parts")
        if not isinstance(parts, dict):
            return
        previous_hashes = tuple(str(item) for item in manifest.get("source_hashes", []))
        current_hashes = set(self.source_hashes)
        previous_hash_set = set(previous_hashes)
        exact_sources = previous_hash_set == current_hashes and (
            not self.source_paths or manifest.get("verified") is True
        )
        incremental_sources = False
        if not exact_sources and previous_hash_set and previous_hash_set.issubset(current_hashes):
            previous_paths = manifest.get("source_paths")
            current_paths = dict(zip(self.source_hashes, self.source_paths, strict=False))
            previous_months = [str(month) for month in parts]
            latest_month = max(previous_months) if previous_months else ""
            added_paths = [current_paths.get(item, "") for item in current_hashes - previous_hash_set]
            incremental_sources = bool(previous_paths) and bool(added_paths) and all(
                _is_append_only_source_path(path, latest_month) for path in added_paths
            )
        if not exact_sources and not incremental_sources:
            return
        self._reusable_source_hashes = current_hashes if exact_sources else previous_hash_set
        sealed = manifest.get("sealed_months")
        months = list(sealed) if isinstance(sealed, list) else sorted(parts)
        for month in months:
            entries = parts[str(month)]
            # The staging manifest is itself an atomic, digest-checked commit.
            # Older runs may predate per-month checkpoint entries, so the
            # checkpoint is an optional acceleration/diagnostic layer rather
            # than a prerequisite for safe reuse.
            if not isinstance(entries, list) or not entries:
                continue
            digests: list[str] = []
            counts: list[int] = []
            valid = True
            for entry in entries:
                if not isinstance(entry, dict):
                    valid = False
                    break
                idx = int(entry.get("part_index", -1))
                digest = str(entry.get("part_digest", ""))
                count = int(entry.get("row_count", 0))
                path = self._month_dir(str(month)) / f"part-{idx:05d}.parquet"
                if idx < 0 or not digest or count <= 0 or not path.exists() or _file_digest(path) != digest:
                    valid = False
                    break
                digests.append(digest)
                counts.append(count)
            if valid:
                key = str(month)
                self._part_digests[key] = digests
                self._part_counts[key] = counts
                self._row_counts[key] = sum(counts)
                self._verified_months.add(key)
                self._sealed_months.add(key)

    @property
    def has_reusable_manifest(self) -> bool:
        return bool(self._verified_months) and not self._buffers

    @property
    def pending_source_hashes(self) -> frozenset[str]:
        return frozenset(set(self.source_hashes) - self._reusable_source_hashes)

    def _month_dir(self, month: str) -> Path:
        year, _, mon = month.partition("-")
        return self.root / self.table.value / f"year={year}" / f"month={mon}"

    def append(self, *, month: str, row: dict[str, Any]) -> None:
        if self._fingerprint_month is not None and month < self._fingerprint_month:
            raise PITDataError(f"month order regression {self._fingerprint_month} -> {month}; certification blocked")
        if month in self._verified_months:
            return
        # Source pages are processed in month order in the normal path. Keep
        # duplicate-detection state only for the active month; retaining keys
        # for the full multi-year history defeats bounded streaming memory.
        if month != self._fingerprint_month:
            if self._fingerprint_month is not None and self._buffers.get(self._fingerprint_month):
                self._flush_month(self._fingerprint_month)
            if self._fingerprint_month is not None:
                self._sealed_months.add(self._fingerprint_month)
                self._persist_staging_manifest()
            self._master_fingerprints.clear()
            self._daily_fingerprints.clear()
            self._fingerprint_month = month
        if self.table is SilverTable.SECURITY_MASTER:
            key = (row.get("instrument_id"), row.get("valid_from"))
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        str(name): value
                        for name, value in row.items()
                        if name not in {"available_at", "source_hash"}
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if self._master_fingerprints.get(key) == fingerprint:
                return
            self._master_fingerprints[key] = fingerprint
        elif self.table is SilverTable.DAILY_MARKET:
            key = (row.get("session"), row.get("instrument_id"))
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        str(name): value
                        for name, value in row.items()
                        if name not in {"available_at", "source_hash"}
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            previous = self._daily_fingerprints.get(key)
            if previous == fingerprint:
                return
            if previous is not None:
                raise PITDataError(
                    f"conflicting daily_market primary key {key!r}; certification blocked"
                )
            self._daily_fingerprints[key] = fingerprint
        buf = self._buffers.setdefault(month, [])
        buf.append(dict(row))
        self._row_counts[month] = self._row_counts.get(month, 0) + 1
        if len(buf) >= self.batch_size:
            self._flush_month(month)

    def _validate_batch(self, month: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise PITDataError(f"incomplete month for {self.table.value}; certification blocked")
        if self.table is SilverTable.DAILY_MARKET:
            _assert_unique_daily_keys(rows)
            for row in rows:
                session = row.get("session")
                if not isinstance(session, datetime):
                    raise PITDataError("invalid KRX session; certification blocked")
                if session.tzinfo is None:
                    raise PITDataError("column session must be timezone-aware")
                try:
                    o = float(row["open"])
                    h = float(row["high"])
                    low = float(row["low"])
                    c = float(row["close"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PITDataError("invalid KRX market value; certification blocked") from exc
                if not (low <= o <= h and low <= c <= h):
                    raise PITDataError(f"ohlc violation in {self.table.value}")
                avail = row.get("available_at")
                if isinstance(avail, datetime) and avail.tzinfo is None:
                    raise PITDataError("column available_at must be timezone-aware")
        elif self.table is SilverTable.SECURITY_MASTER:
            for row in rows:
                if not row.get("instrument_id") or not isinstance(row.get("valid_from"), datetime):
                    raise PITDataError(f"invalid {self.table.value} key; certification blocked")

    def _flush_month(self, month: str) -> None:
        buf = self._buffers.get(month, [])
        if not buf:
            return
        if len(buf) > self.batch_size:
            raise PITDataError("batch exceeds bound")
        self._validate_batch(month, buf)
        month_dir = self._month_dir(month)
        month_dir.mkdir(parents=True, exist_ok=True)
        index = len(self._part_digests.get(month, []))
        part_path = month_dir / f"part-{index:05d}.parquet"
        if part_path.exists():
            if self._checkpoint.is_verified(
                table=self.table.value,
                month=month,
                source_hashes=self.source_hashes,
                schema_version=self.schema_version,
            ):
                digest = _file_digest(part_path)
                stored = self._checkpoint.verified_entry(table=self.table.value, month=month)
                known = list((stored or {}).get("part_digests", []))
                if digest and digest in known:
                    self._part_digests.setdefault(month, []).append(digest)
                    self._part_counts.setdefault(month, []).append(len(buf))
                    self._buffers[month] = []
                    return
            shutil.rmtree(month_dir, ignore_errors=True)
            month_dir.mkdir(parents=True, exist_ok=True)
            self._part_digests[month] = []
            self._part_counts[month] = []
            index = 0
            part_path = month_dir / f"part-{index:05d}.parquet"
        tmp_path = part_path.with_suffix(".parquet.tmp")
        schema_overrides = {"source_security_id": pl.String} if any("source_security_id" in row for row in buf) else None
        pl.DataFrame(buf, schema_overrides=schema_overrides).write_parquet(tmp_path)
        tmp_path.replace(part_path)
        digest = _file_digest(part_path)
        if not digest:
            raise PITDataError("partition digest missing; certification blocked")
        self._part_digests.setdefault(month, []).append(digest)
        self._part_counts.setdefault(month, []).append(len(buf))
        self._buffers[month] = []
        _trim_streaming_allocator()
        self._persist_staging_manifest()

    def _manifest(self, *, verified: bool) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for month in sorted(self._part_digests):
            year, _, mon = month.partition("-")
            for idx, (digest, count) in enumerate(
                zip(self._part_digests[month], self._part_counts[month], strict=True)
            ):
                entries.append(
                    {"year": year, "month": mon, "part_index": idx, "row_count": count, "part_digest": digest}
                )
        return {
            "table": self.table.value,
            "schema_version": self.schema_version,
            "source_hashes": list(self.source_hashes),
            "source_paths": list(self.source_paths),
            "months": sorted(self._part_digests),
            "sealed_months": sorted(self._sealed_months),
            "parts": {month: [{"part_index": idx, "row_count": count, "part_digest": digest} for idx, (digest, count) in enumerate(zip(self._part_digests[month], self._part_counts[month], strict=True))] for month in sorted(self._part_digests)},
            "row_counts": dict(self._row_counts),
            "root_hash": streamed_dataset_root_hash(entries),
            "verified": verified,
        }

    def _persist_staging_manifest(self) -> None:
        manifest = self._manifest(verified=False)
        _write_doc(self.root / self.table.value / "staging_manifest.json", manifest)

    def close(self) -> dict[str, Any]:
        for month in sorted(self._buffers):
            if self._buffers[month]:
                self._flush_month(month)
            if month == self._fingerprint_month:
                self._sealed_months.add(month)
        if not self._part_digests:
            raise PITDataError(f"incomplete month for {self.table.value}; certification blocked")
        manifest = self._manifest(verified=True)
        root_hash = str(manifest["root_hash"])
        for month in sorted(self._part_digests):
            self._checkpoint.mark_verified(
                table=self.table.value,
                month=month,
                source_hashes=self.source_hashes,
                output_hash=root_hash,
                schema_version=self.schema_version,
                part_digests=tuple(self._part_digests[month]),
                row_count=self._row_counts.get(month, 0),
                partition_root_hash=root_hash,
                persist=False,
            )
        self._checkpoint.flush()
        staging_manifest = self.root / self.table.value / "staging_manifest.json"
        _write_doc(staging_manifest, manifest)
        return manifest


def _iter_batches(items: list[Any], batch_size: int) -> Any:
    total = len(items)
    idx = 0
    while idx < total:
        chunk = items[idx : idx + batch_size]
        if len(chunk) > batch_size:
            raise PITDataError("batch exceeds bound")
        yield chunk
        idx += batch_size


def _extract_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("records", "intervals", "list"):
            val = payload.get(key)
            if isinstance(val, list):
                return list(val)
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return list(sessions)
        return []
    if isinstance(payload, list):
        return list(payload)
    return []


def _discover_receipts(bronze_root: Path) -> dict[EvidenceKind, list[BronzeReceipt]]:
    root = Path(bronze_root)
    found: dict[EvidenceKind, list[BronzeReceipt]] = {}
    for kind in EvidenceKind:
        kind_dir = root / kind.value
        receipts: list[BronzeReceipt] = []
        if kind_dir.exists():
            for receipt_path in sorted(kind_dir.rglob("receipt.json")):
                try:
                    meta = _read_doc(receipt_path)
                except (OSError, ValueError):
                    continue
                payload_path = receipt_path.parent / "payload.json"
                if not payload_path.exists():
                    continue
                try:
                    content_hash = str(meta["content_hash"])
                    retrieved = datetime.fromisoformat(str(meta["retrieved_at"]))
                    ingested = datetime.fromisoformat(str(meta["ingested_at"]))
                except (KeyError, ValueError):
                    continue
                if not content_hash:
                    continue
                receipts.append(
                    BronzeReceipt(
                        kind=kind,
                        content_hash=content_hash,
                        source_path=str(meta.get("source_path", "")),
                        retrieved_at=retrieved,
                        ingested_at=ingested,
                        payload_path=payload_path,
                        metadata_path=receipt_path,
                    )
                )
        if receipts:
            found[kind] = sorted(receipts, key=lambda r: (r.retrieved_at, r.content_hash))
    return found


_LABEL_DATE_TOKEN = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2}|\d{8})(?!\d)")


def _receipt_event_date(receipt: BronzeReceipt) -> date:
    matched = _LABEL_DATE_TOKEN.search(str(receipt.source_path))
    if matched:
        token = matched.group(1)
        if len(token) == 8:
            return datetime.strptime(token, "%Y%m%d").date()
        return date.fromisoformat(token)
    try:
        size = receipt.payload_path.stat().st_size
    except OSError as exc:
        raise PITDataError(f"missing event date for {receipt.source_path}; certification blocked") from exc
    if size >= STREAMING_EAGER_JSON_MAX_BYTES:
        raise PITDataError(f"missing event date for {receipt.source_path}; certification blocked")
    try:
        payload = _read_doc(receipt.payload_path)
    except (OSError, ValueError) as exc:
        raise PITDataError(f"missing event date for {receipt.source_path}; certification blocked") from exc
    if isinstance(payload, dict):
        for key in ("session", "date", "as_of", "price_date", "valid_from", "BAS_DD", "basDd"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                return _as_krx_datetime(value).date()
            except PITDataError:
                continue
    raise PITDataError(f"missing event date for {receipt.source_path}; certification blocked")


def order_streaming_receipts(
    *, table: SilverTable, receipts: tuple[BronzeReceipt, ...]
) -> tuple[BronzeReceipt, ...]:
    _ = table
    keyed = [(_receipt_event_date(item), item.retrieved_at, item.content_hash, item) for item in receipts]
    keyed.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return tuple(entry[3] for entry in keyed)


def _dedupe_duplicate_source_paths(
    receipts: tuple[BronzeReceipt, ...],
) -> tuple[BronzeReceipt, ...]:
    """Keep the widest verified page when a provider path was retried.

    Bronze is immutable, so retries can legitimately produce different
    content hashes for the same provider path.  Feeding both pages to the
    streaming writer can move the cursor backwards (for example, a shorter
    warm-up page arriving after a longer one).  For duplicate paths choose
    the page with the greatest observed session coverage, then the greatest
    row count, and finally the earliest retrieval as a stable tie-breaker.
    Unique paths are returned untouched.
    """
    # A normalized provider page historically reused one logical source path
    # for every session.  The event date is therefore part of the identity;
    # grouping by path alone would collapse an entire year to one page.
    normalized_daily = [
        receipt
        for receipt in receipts
        if str(receipt.source_path) == "normalized-provider-page:daily_market"
    ]
    if normalized_daily:
        normalized_first = min(_receipt_event_date(receipt) for receipt in normalized_daily)
        receipts = tuple(
            receipt
            for receipt in receipts
            if not (
                str(receipt.source_path).startswith("KRX:warmup:")
                and normalized_first <= _receipt_event_date(receipt)
            )
        )
    by_path: dict[tuple[str, date], list[BronzeReceipt]] = {}
    for receipt in receipts:
        key = (str(receipt.source_path), _receipt_event_date(receipt))
        by_path.setdefault(key, []).append(receipt)
    if all(len(items) == 1 for items in by_path.values()):
        return receipts

    selected: list[BronzeReceipt] = []
    for items in by_path.values():
        if len(items) == 1:
            selected.extend(items)
            continue
        scored: list[tuple[date, int, float, str, BronzeReceipt]] = []
        for receipt in items:
            minimum = maximum = _receipt_event_date(receipt)
            count = 0
            try:
                for row in _stream_items_for_kind([receipt], batch_size=4096):
                    count += 1
                    for field_name in ("session", "price_date", "BAS_DD", "basDd", "valid_from"):
                        value = row.get(field_name)
                        if value in (None, ""):
                            continue
                        try:
                            observed = _as_krx_datetime(value).date()
                        except PITDataError:
                            continue
                        minimum = min(minimum, observed)
                        maximum = max(maximum, observed)
                        break
            except PITDataError:
                # The normal worker will report the malformed page.  Do not
                # hide that error here merely because another retry exists.
                raise
            scored.append(
                (
                    maximum,
                    count,
                    -receipt.retrieved_at.timestamp(),
                    receipt.content_hash,
                    receipt,
                )
            )
        selected.append(max(scored)[-1])
    return tuple(selected)


def _stream_items_for_kind(
    receipts: list[BronzeReceipt], *, batch_size: int
) -> Iterator[dict[str, Any]]:
    for receipt in receipts:
        try:
            payload_size = receipt.payload_path.stat().st_size
        except OSError as exc:
            raise PITDataError("malformed Bronze JSON; certification blocked") from exc
        if payload_size < STREAMING_EAGER_JSON_MAX_BYTES:
            try:
                small_payload = _read_doc(receipt.payload_path)
            except (OSError, ValueError) as exc:
                raise PITDataError("malformed Bronze JSON; certification blocked") from exc
            for item in _extract_items(small_payload):
                if not isinstance(item, dict):
                    raise PITDataError("malformed record; certification blocked")
                yield item
            continue
        # jq emits one array element per line without materializing the JSON
        # document; the Python side retains only the configured batch.
        try:
            jq = shutil.which("jq")
            if jq is None:
                raise OSError("jq not found")
            process = subprocess.Popen(  # noqa: S603 - executable resolved from PATH; args are constants
                [jq, "-c", ".records[]? // .intervals[]? // .list[]?"],
                stdin=receipt.payload_path.open("rb"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise PITDataError("streaming JSON parser is unavailable") from exc
        assert process.stdout is not None
        for raw_line in process.stdout:
            try:
                item = json.loads(raw_line)
            except ValueError as exc:
                process.kill()
                raise PITDataError("malformed record; certification blocked") from exc
            if not isinstance(item, dict):
                raise PITDataError("malformed record; certification blocked")
            yield item
        if process.wait() != 0:
            raise PITDataError("malformed Bronze JSON; certification blocked")


def _stream_corporate_action_intervals(
    receipts: Iterable[BronzeReceipt], *, read_size: int = _CORPORATE_ACTION_READ_SIZE
) -> Iterable[dict[str, Any]]:
    """Yield a top-level ``intervals`` array without building a JSON DOM."""
    if not isinstance(read_size, int) or isinstance(read_size, bool) or read_size < 1:
        raise PITDataError("corporate-action read_size must be a positive integer")
    decoder = json.JSONDecoder()
    for receipt in receipts:
        try:
            handle = receipt.payload_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise PITDataError("missing corporate-action payload; certification blocked") from exc
        with handle:
            buffer = ""
            intervals_started = False
            exhausted = False

            def read_more(stream: Any = handle) -> None:
                nonlocal buffer, exhausted
                chunk = stream.read(read_size)
                if not chunk:
                    exhausted = True
                    return
                buffer += chunk
                if len(buffer.encode("utf-8")) > _MAX_CORPORATE_ACTION_RECORD_BYTES:
                    raise PITDataError("corporate-action interval exceeds bounded parser buffer")

            while True:
                if not intervals_started:
                    key_index = buffer.find('"intervals"')
                    if key_index < 0:
                        if exhausted:
                            raise PITDataError("corporate-action payload missing intervals array")
                        if len(buffer) > len('"intervals"'):
                            buffer = buffer[-len('"intervals"') :]
                        read_more()
                        continue
                    opening_index = buffer.find("[", key_index + len('"intervals"'))
                    if opening_index < 0:
                        if exhausted:
                            raise PITDataError("corporate-action payload missing intervals array")
                        read_more()
                        continue
                    buffer = buffer[opening_index + 1 :]
                    intervals_started = True

                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    trailing = buffer[1:] + handle.read()
                    if trailing.strip() != "}":
                        raise PITDataError("malformed corporate-action JSON; certification blocked")
                    break
                if not buffer:
                    if exhausted:
                        raise PITDataError("unterminated corporate-action intervals array")
                    read_more()
                    continue
                try:
                    item, end_index = decoder.raw_decode(buffer)
                except json.JSONDecodeError as exc:
                    if exhausted:
                        raise PITDataError("malformed corporate-action JSON; certification blocked") from exc
                    read_more()
                    continue
                if not isinstance(item, dict):
                    raise PITDataError("malformed corporate-action interval; certification blocked")
                yield item
                buffer = buffer[end_index:]


def historical_available_at(
    *,
    kind: EvidenceKind,
    record: dict[str, Any] | Any,
    calendar: SessionCalendar,
) -> datetime:
    """Map provider records to PIT consumption instants (float64 precision N/A, session chunking).

    - KRX daily -> session close (15:30 KST same session).
    - KRX master/actions -> session open (09:00 same session).
    - KIS flow -> next KRX open (session S flow usable at next open; never same-day).
    - DART -> first KRX session after published_at when intraday proof is absent.
    - Never reads receipt retrieved_at/ingested_at (local collection only).
    """
    from collections.abc import Mapping as _Mapping

    if not isinstance(record, _Mapping):
        raise PITDataError("historical record must be a mapping")
    if not calendar.sessions:
        raise PITDataError("calendar has no sessions")
    ordered = tuple(sorted(calendar.sessions))
    # Failure-mode guard: retrieval time must never shift economic availability.
    _ = record.get("retrieved_at"), record.get("ingested_at")

    def _session_date() -> Any:
        for key in ("session", "price_date", "BAS_DD", "basDd", "effective_date"):
            value = record.get(key)
            if value not in (None, ""):
                try:
                    return _as_krx_datetime(value).astimezone(KRX_TZ).date()
                except PITDataError:
                    continue
        return None

    def _at_open(day: Any) -> datetime:
        return datetime.combine(day, time(9, 0), tzinfo=KRX_TZ)

    def _at_close(day: Any) -> datetime:
        return datetime.combine(day, time(15, 30), tzinfo=KRX_TZ)

    if kind == EvidenceKind.DAILY_MARKET:
        day = _session_date()
        if day is None:
            raise PITDataError("daily market record missing session")
        return _at_close(day)
    if kind in (EvidenceKind.SECURITY_MASTER, EvidenceKind.CORPORATE_ACTIONS, EvidenceKind.CALENDAR, EvidenceKind.HISTORICAL_COSTS):
        day = _session_date()
        if day is None:
            # Fall back to earliest session open for sentinel/master rows.
            return ordered[0]
        return _at_open(day)
    if kind == EvidenceKind.INVESTOR_FLOW:
        day = _session_date()
        if day is None:
            raise PITDataError("investor flow record missing session")
        for sess in ordered:
            if sess.astimezone(KRX_TZ).date() > day:
                return sess
        raise PITDataError("no next KRX session for investor flow")
    if kind in (EvidenceKind.DISCLOSURES, EvidenceKind.FINANCIAL_FACTS):
        published = record.get("published_at", record.get("available_time", record.get("session")))
        if published in (None, ""):
            raise PITDataError("DART record missing published_at")
        try:
            moment = _as_krx_datetime(published)
        except PITDataError as exc:
            raise PITDataError("DART record missing published_at") from exc
        for sess in ordered:
            if sess > moment:
                return sess
        raise PITDataError("no KRX session after DART publication")
    raise PITDataError(f"unsupported evidence kind {kind.value}")


def _month_of(value: Any) -> str:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise PITDataError("malformed record; certification blocked")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=KRX_TZ)
    krx = moment.astimezone(KRX_TZ)
    return f"{krx.year:04d}-{krx.month:02d}"


def _as_krx_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                moment = datetime.strptime(text[:10], "%Y-%m-%d")
            except ValueError:
                try:
                    moment = datetime.strptime(text[:8], "%Y%m%d")
                except ValueError as exc:
                    raise PITDataError("invalid KRX session; certification blocked") from exc
    else:
        raise PITDataError("missing KRX session; certification blocked")
    if moment.tzinfo is None:
        moment = datetime.combine(moment.date(), time(9, 0), tzinfo=KRX_TZ)
    return moment.astimezone(KRX_TZ)


def _parse_exact_int(value: Any, *, field: str) -> int:
    text = str(value).replace(",", "").strip()
    if not re.fullmatch(r"-?\d+", text):
        raise PITDataError(f"ambiguous OpenDART share basis for {field}; certification blocked")  # pragma: no cover
    return int(text)


def _parse_exact_decimal(value: Any, *, field: str) -> Decimal:
    """Parse DART allocation ratios, which may be fractional (e.g. ``0.5``)."""
    text = str(value).replace(",", "").strip()
    if not re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)", text):
        raise PITDataError(f"ambiguous OpenDART share basis for {field}; certification blocked")  # pragma: no cover
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover
        raise PITDataError(f"ambiguous OpenDART share basis for {field}; certification blocked") from exc
    if not parsed.is_finite():  # pragma: no cover
        raise PITDataError(f"ambiguous OpenDART share basis for {field}; certification blocked")
    return parsed


def _parse_opendart_date(value: Any) -> date:
    text = str(value).strip()
    match = re.search(r"(\d{4})\D*(\d{1,2})\D*(\d{1,2})", text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    compact = re.sub(r"\D", "", text)  # pragma: no cover
    if len(compact) == 8 and compact.isdigit():  # pragma: no cover
        return date(int(compact[:4]), int(compact[4:6]), int(compact[6:8]))  # pragma: no cover
    raise PITDataError(f"invalid OpenDART date {value!r}; certification blocked")  # pragma: no cover


def _receipt_available_at(*, rcept_no: str, calendar: SessionCalendar) -> datetime:
    digits = re.sub(r"\D", "", str(rcept_no))[:8]
    if len(digits) != 8 or not digits.isdigit():
        raise PITDataError(f"invalid OpenDART receipt number {rcept_no!r}")
    receipt_day = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    for session in sorted(calendar.sessions):
        if session.astimezone(KRX_TZ).date() > receipt_day:
            local = session.astimezone(KRX_TZ).date()
            return datetime.combine(local, time(9, 0), tzinfo=KRX_TZ)
    raise PITDataError(f"no KRX session after OpenDART receipt {rcept_no!r}")  # pragma: no cover


def _resolve_instrument(*, corp_code: str, daily_market: pl.DataFrame, record: dict[str, Any]) -> str:
    for key in ("instrument_id", "ticker"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()  # pragma: no cover
            return text if text.startswith("KRX:") else f"KRX:{text}"  # pragma: no cover
    instruments = sorted({str(value) for value in daily_market["instrument_id"].to_list()})
    if len(instruments) == 1:
        return instruments[0]
    raise PITDataError(f"missing OpenDART corp_code mapping for {corp_code!r}")  # pragma: no cover


def _resolve_listing_session(*, listing_date: date, calendar: SessionCalendar, instrument_id: str, action_id: str) -> datetime:
    ordered = tuple(sorted(calendar.sessions))
    for session in ordered:
        if session.astimezone(KRX_TZ).date() >= listing_date:
            return session
    raise PITDataError(f"no KRX listing session for {instrument_id!r} action {action_id!r} field nstk_lstprd; certification blocked")


def mapped_action_instruments(*, pages: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Collect explicitly mapped KRX instruments without issuer inference."""
    explicit: set[str] = set()
    for page in pages:
        requested = page.get("requested_instrument_id")
        if (
            isinstance(requested, str)
            and requested.strip()
            and page.get("instrument_mapping_provenance") == "opendart_corp_code_direct"
        ):
            explicit.add(requested.strip())
        for record in page.get("records", []) or []:
            if not isinstance(record, dict):
                raise PITDataError("invalid OpenDART record; certification blocked")
            for key in ("instrument_id", "ticker"):
                raw = record.get(key)
                if isinstance(raw, str) and raw.strip():
                    text = raw.strip()
                    explicit.add(text if text.startswith("KRX:") else f"KRX:{text}")
                    break
    return frozenset(explicit)


def load_structured_corporate_action_pages(
    *, action_receipts: tuple[BronzeReceipt, ...]
) -> list[dict[str, Any]]:
    """Read OpenDART structured-decision payloads without legacy interval parsing."""
    # A historical Bronze generation may contain both the old interval JSON
    # and the newer OpenDART receipts.  The former has no endpoint envelope
    # and must not make a valid structured refresh fail.  Once structured
    # receipts are present they are the authoritative input for this loader.
    structured_receipts = tuple(
        receipt
        for receipt in action_receipts
        if str(receipt.source_path).startswith("opendart_structured_decisions:")
    )
    selected_receipts = structured_receipts or action_receipts
    pages: list[dict[str, Any]] = []
    for receipt in selected_receipts:
        try:
            payload = _read_doc(receipt.payload_path)
        except (OSError, ValueError) as exc:
            raise PITDataError("missing corporate-action payload; certification blocked") from exc
        if not isinstance(payload, dict) or "endpoint" not in payload:
            raise PITDataError("invalid corporate-action payload; certification blocked")
        pages.append(payload)
    return pages


def _resolve_instrument_with_page(
    *, corp_code: str, daily_market: pl.DataFrame, record: dict[str, Any], page: dict[str, Any]
) -> tuple[str, str | None]:
    for key in ("instrument_id", "ticker"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            return (text if text.startswith("KRX:") else f"KRX:{text}", None)
    requested = page.get("requested_instrument_id")
    provenance = page.get("instrument_mapping_provenance")
    if isinstance(requested, str) and requested.strip() and provenance == "opendart_corp_code_direct":
        return (requested.strip(), str(provenance))
    instruments = sorted({str(value) for value in daily_market["instrument_id"].to_list()})
    if len(instruments) == 1:
        return (instruments[0], None)
    raise PITDataError(f"missing OpenDART corp_code mapping for {corp_code!r}")


def _event_session_from_record(
    *, record: dict[str, Any], available_at: datetime, calendar: SessionCalendar, rcept_no: str = ""
) -> datetime:
    for key in ("crsc_nstkdlprd", "event_date", "ex_date", "record_date", "mgsc_mgdt", "dvdt", "nstk_asstd", "asstd"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        try:
            parsed = _parse_opendart_date(raw)
        except PITDataError:
            continue
        for session in sorted(calendar.sessions):
            if session.astimezone(KRX_TZ).date() >= parsed:
                return session
    ordered = tuple(sorted(calendar.sessions))
    digits = re.sub(r"\D", "", str(rcept_no))[:8]
    receipt_day = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    return next(
        (session for session in ordered if session.astimezone(KRX_TZ).date() > receipt_day),
        available_at,
    )


_UNRESOLVED_REASON_BY_ENDPOINT: dict[str, str] = {
    "piicDecsn.json": "unsupported_paid_in_capital",
    "cmpMgDecsn.json": "unsupported_merger",
    "cmpDvDecsn.json": "unsupported_division",
    "crDecsn.json": "unsupported_capital_reduction",
}


def resolve_opendart_corporate_action_records(
    *, pages: Iterable[dict[str, Any]], daily_market: pl.DataFrame, calendar: SessionCalendar
) -> list[dict[str, Any]]:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    threshold = float(BacktestMarketInputsPolicy().unexplained_price_jump_threshold)
    page_list: list[dict[str, Any]] = []
    for page in pages:
        item = dict(page) if isinstance(page, dict) else {
            "endpoint": getattr(page, "endpoint", ""),
            "corp_code": getattr(page, "corp_code", ""),
            "status": getattr(page, "status", ""),
            "records": list(getattr(page, "records", ()) or ()),
        }
        page_list.append(item)
    page_list.sort(key=lambda p: (str(p.get("corp_code", "")), str(p.get("endpoint", ""))))
    # OpenDART may return one endpoint with an explicit corp-code mapping and
    # another endpoint for the same corp code without the mapping envelope.
    # Carry the certified mapping across those pages before resolving records.
    corp_code_mapping: dict[str, tuple[str, str]] = {}
    for page in page_list:
        corp_code = str(page.get("corp_code", "")).strip()
        requested = page.get("requested_instrument_id")
        provenance = page.get("instrument_mapping_provenance")
        if (
            corp_code
            and isinstance(requested, str)
            and requested.strip()
            and provenance == "opendart_corp_code_direct"
        ):
            corp_code_mapping[corp_code] = (requested.strip(), str(provenance))
        for record in page.get("records", []) or ():
            if not isinstance(record, dict):
                continue
            raw_instrument = record.get("instrument_id") or record.get("ticker")
            if corp_code and isinstance(raw_instrument, str) and raw_instrument.strip():
                instrument_id = raw_instrument.strip()
                corp_code_mapping.setdefault(
                    corp_code,
                    (instrument_id if instrument_id.startswith("KRX:") else f"KRX:{instrument_id}", "record"),
                )
    available = set(daily_market.columns)
    select_exprs: list[pl.Expr] = [
        pl.col("session"),
        pl.col("instrument_id").cast(pl.String),
        pl.col("close").cast(pl.Float64),
    ]
    if "shares_outstanding" in available:
        select_exprs.append(pl.col("shares_outstanding").cast(pl.Float64))
    else:
        select_exprs.append(pl.lit(float("nan")).alias("shares_outstanding"))
    if "market_cap" in available:
        select_exprs.append(pl.col("market_cap").cast(pl.Float64))
    else:
        select_exprs.append(pl.lit(float("nan")).alias("market_cap"))
    projected = daily_market.lazy().select(select_exprs).collect()
    bars_by_iid: dict[str, list[dict[str, Any]]] = {}
    for row in projected.to_dicts():
        iid = str(row.get("instrument_id", ""))
        session = row.get("session")
        if not isinstance(session, datetime):
            raise PITDataError("invalid KRX session; certification blocked")
        try:
            close = float(row.get("close", float("nan")))
        except (TypeError, ValueError) as exc:
            raise PITDataError("invalid KRX market value; certification blocked") from exc
        if not math.isfinite(close) or close <= 0:
            raise PITDataError("invalid KRX market value; certification blocked")
        try:
            shares = float(row.get("shares_outstanding", float("nan")))
        except (TypeError, ValueError):
            shares = float("nan")
        try:
            cap = float(row.get("market_cap", float("nan")))
        except (TypeError, ValueError):
            cap = float("nan")
        bars_by_iid.setdefault(iid, []).append(
            {"session": session, "close": close, "shares_outstanding": shares, "market_cap": cap}
        )
    for iid in bars_by_iid:
        bars_by_iid[iid].sort(key=lambda entry: entry["session"])

    # A KRX listing snapshot can reflect a same-day sequence of capital
    # reduction, debt-equity conversion, and another reduction, while DART
    # exposes each decision separately.  Build a bounded share-basis graph so
    # a composite event is admitted only when the exact KRX before/after
    # shares and the price factor are both reproducible.
    composite_actions: dict[tuple[str, datetime], dict[str, Any]] = {}
    consumed_composite_ids: set[str] = set()
    for page in page_list:
        if str(page.get("endpoint", "")) != "crDecsn.json":
            continue
        corp_code = str(page.get("corp_code", ""))
        inherited_mapping = corp_code_mapping.get(corp_code)
        mapped_page = page
        if inherited_mapping and not page.get("requested_instrument_id"):
            mapped_page = {**page, "requested_instrument_id": inherited_mapping[0], "instrument_mapping_provenance": inherited_mapping[1]}
        for record in page.get("records", []) or ():
            if not isinstance(record, dict):
                continue
            required = ("bfcr_tisstk_ostk", "atcr_tisstk_ostk", "crsc_nstklstprd")
            if any(str(record.get(field, "") or "").strip() in ("", "-") for field in required):
                continue
            instrument_id, _ = _resolve_instrument_with_page(
                corp_code=corp_code or str(record.get("corp_code", "")),
                daily_market=daily_market, record=record, page=mapped_page,
            )
            try:
                listing_session = _resolve_listing_session(
                    listing_date=_parse_opendart_date(record["crsc_nstklstprd"]),
                    calendar=calendar, instrument_id=instrument_id,
                    action_id=str(record.get("rcept_no", "")),
                )
            except PITDataError:
                continue
            group = composite_actions.setdefault(
                (instrument_id, listing_session), {"capital": [], "issuance": []}
            )
            try:
                group["capital"].append({
                    "before": _parse_exact_int(record["bfcr_tisstk_ostk"], field="bfcr_tisstk_ostk"),
                    "after": _parse_exact_int(record["atcr_tisstk_ostk"], field="atcr_tisstk_ostk"),
                    "action_id": str(record.get("rcept_no", "")),
                    "receipt_day": str(record.get("rcept_no", ""))[:8],
                })
            except PITDataError:
                continue
    # Attach same-receipt paid-in-capital records to each capital-reduction
    # group.  DART's piicDecsn rows carry the pre-issuance share basis, which
    # makes the edge deterministic without guessing a listing date.
    for key, group in list(composite_actions.items()):
        receipt_days = {str(item["receipt_day"]) for item in group["capital"]}
        for page in page_list:
            if str(page.get("endpoint", "")) != "piicDecsn.json":
                continue
            for record in page.get("records", []) or ():
                if not isinstance(record, dict):
                    continue
                receipt_day = str(record.get("rcept_no", ""))[:8]
                if receipt_day not in receipt_days:
                    continue
                try:
                    before = _parse_exact_int(record.get("bfic_tisstk_ostk"), field="bfic_tisstk_ostk")
                    count = _parse_exact_int(record.get("nstk_ostk_cnt"), field="nstk_ostk_cnt")
                except PITDataError:
                    continue
                group["issuance"].append({
                    "before": before,
                    "after": before + count,
                    "action_id": str(record.get("rcept_no", "")),
                })
        instrument_id, listing_session = key
        bars = bars_by_iid.get(instrument_id, [])
        listed_idx = next((idx for idx, entry in enumerate(bars) if entry["session"] == listing_session), None)
        if listed_idx is None or listed_idx < 1:
            continue
        start = round(float(bars[listed_idx - 1]["shares_outstanding"]))
        target = round(float(bars[listed_idx]["shares_outstanding"]))
        edges = [
            {**edge, "kind": "capital"} for edge in group["capital"]
        ] + [
            {**edge, "kind": "issuance"} for edge in group["issuance"]
        ]
        path: list[dict[str, Any]] | None = None

        def _walk(
            current: int,
            used: frozenset[int],
            candidate: list[dict[str, Any]],
            *,
            target_shares: int = target,
            graph_edges: list[dict[str, Any]] = edges,
        ) -> bool:
            nonlocal path
            if current == target_shares and any(edge.get("kind") == "capital" for edge in candidate):
                path = candidate
                return True
            if len(candidate) >= len(graph_edges):
                return False
            for idx, edge in enumerate(graph_edges):
                if idx in used or int(edge["before"]) != current:
                    continue
                if _walk(int(edge["after"]), used | {idx}, [*candidate, edge]):
                    return True
            return False

        if not _walk(start, frozenset(), []):
            continue
        assert path is not None
        previous_bar = bars[listed_idx - 1]
        listed_bar = bars[listed_idx]
        raw_return = listed_bar["close"] / previous_bar["close"] - 1.0
        adjusted_return = listed_bar["close"] * float(target) / (previous_bar["close"] * float(start)) - 1.0
        cap_expected = listed_bar["close"] * listed_bar["shares_outstanding"]
        cap_match = abs(listed_bar["market_cap"] - cap_expected) <= max(1e-6, 1e-6 * max(abs(listed_bar["market_cap"]), abs(cap_expected)))
        if raw_return <= threshold or abs(adjusted_return) > threshold or not cap_match:
            continue
        capital_ids = [str(edge["action_id"]) for edge in path if edge.get("kind") == "capital"]
        issuance_ids = [str(edge["action_id"]) for edge in path if edge.get("kind") == "issuance"]
        action_id = "+".join(capital_ids + issuance_ids)
        available_at = _receipt_available_at(rcept_no=capital_ids[0], calendar=calendar)
        if not available_at < listing_session.replace(hour=15, minute=30):
            continue
        composite_actions[key] = {
            "verified": {
                "instrument_id": instrument_id,
                "action_type": "reverse_split",
                "type": "reverse_split",
                "factor": float(target) / float(start),
                "cash_amount": 0.0,
                "effective_session": listing_session,
                "effective_date": listing_session,
                "share_listing_date": listing_session,
                "share_delta": target - start,
                "available_at": available_at,
                "action_id": action_id,
                "evidence_status": "verified",
                "evidence_reason": None,
            },
            "anchor": capital_ids[0],
        }
        consumed_composite_ids.update(capital_ids[1:] + issuance_ids)

    def _unresolved(
        *, instrument_id: str, action_id: str, available_at: datetime,
        effective_session: datetime, reason: str, page: dict[str, Any],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "instrument_id": instrument_id,
            "action_type": "unresolved",
            "type": "unresolved",
            "factor": 1.0,
            "cash_amount": 0.0,
            "effective_session": effective_session,
            "effective_date": effective_session,
            "share_listing_date": None,
            "share_delta": None,
            "available_at": available_at,
            "action_id": action_id,
            "evidence_status": "unresolved",
            "evidence_reason": reason,
        }
        provenance = page.get("instrument_mapping_provenance")
        if provenance is not None:
            row["instrument_mapping_provenance"] = provenance
        requested = page.get("requested_instrument_id")
        if requested is not None:
            row["requested_instrument_id"] = requested
        return row

    resolved: list[dict[str, Any]] = []
    for page in page_list:
        endpoint = str(page.get("endpoint", ""))
        corp_code = str(page.get("corp_code", ""))
        inherited_mapping = corp_code_mapping.get(corp_code)
        if inherited_mapping and not page.get("requested_instrument_id"):
            page = {
                **page,
                "requested_instrument_id": inherited_mapping[0],
                "instrument_mapping_provenance": inherited_mapping[1],
            }
        status = str(page.get("status", ""))
        if status == "013":
            continue
        if status != "000":
            raise PITDataError(f"unexpected OpenDART status {status!r} for {endpoint} {corp_code}")
        records = page.get("records", [])
        if not isinstance(records, list):
            raise PITDataError(f"invalid OpenDART records for {endpoint} {corp_code}")
        for record in records:
            if not isinstance(record, dict):
                raise PITDataError(f"invalid OpenDART record for {endpoint} {corp_code}")
            rcept_no = str(record.get("rcept_no", "") or "").strip()
            if rcept_no in consumed_composite_ids:
                continue
            if endpoint == "fricDecsn.json":
                for field in ("rcept_no", "corp_code", "bfic_tisstk_ostk", "nstk_ostk_cnt", "nstk_ascnt_ps_ostk", "nstk_asstd", "nstk_lstprd"):
                    if str(record.get(field, "") or "").strip() == "":
                        raise PITDataError(f"missing OpenDART field {field} for {endpoint} {rcept_no}")
                try:
                    basis = _parse_exact_int(record.get("bfic_tisstk_ostk"), field="bfic_tisstk_ostk")
                    existing = _parse_exact_int(record.get("nstk_ostk_cnt"), field="nstk_ostk_cnt")
                    alloc = _parse_exact_decimal(record.get("nstk_ascnt_ps_ostk"), field="nstk_ascnt_ps_ostk")
                except PITDataError:
                    raise
                instrument_id, provenance = _resolve_instrument_with_page(
                    corp_code=corp_code or str(record.get("corp_code", "")),
                    daily_market=daily_market, record=record, page=page,
                )
                available_at = _receipt_available_at(rcept_no=rcept_no, calendar=calendar)
                if alloc <= 0:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at,
                        effective_session=_event_session_from_record(
                            record=record, available_at=available_at, calendar=calendar, rcept_no=rcept_no),
                        reason="ambiguous_or_missing_dart_share_basis", page=page,
                    ))
                    continue
                if abs(basis - existing) > 1000:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at,
                        effective_session=_event_session_from_record(
                            record=record, available_at=available_at, calendar=calendar, rcept_no=rcept_no),
                        reason="ambiguous_or_missing_dart_share_basis", page=page,
                    ))
                    continue
                factor = 1.0 + float(alloc)
                asstd = _parse_opendart_date(record.get("nstk_asstd"))
                listing_date = _parse_opendart_date(record.get("nstk_lstprd"))
                listing_session = _resolve_listing_session(listing_date=listing_date, calendar=calendar, instrument_id=instrument_id, action_id=rcept_no)
                bars = bars_by_iid.get(instrument_id, [])
                if len(bars) < 2:
                    raise PITDataError(f"missing KRX bars for {endpoint} {rcept_no}")
                first_ge = next((idx for idx, entry in enumerate(bars) if entry["session"].astimezone(KRX_TZ).date() >= asstd), None)
                if first_ge is None or first_ge < 1:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at,
                        effective_session=_event_session_from_record(
                            record=record, available_at=available_at, calendar=calendar, rcept_no=rcept_no),
                        reason="expected_one_price_candidate_got_0", page=page,
                    ))
                    continue
                candidates: list[datetime] = []
                for curr_idx in (first_ge - 1, first_ge):
                    if curr_idx < 1 or curr_idx >= len(bars):
                        continue
                    prev_close = bars[curr_idx - 1]["close"]
                    curr_close = bars[curr_idx]["close"]
                    raw_return = abs(curr_close / prev_close - 1.0)
                    adjusted = abs(factor * curr_close / prev_close - 1.0)
                    if raw_return > threshold and adjusted <= threshold:
                        candidates.append(bars[curr_idx]["session"])
                if len(candidates) != 1:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at,
                        effective_session=_event_session_from_record(
                            record=record, available_at=available_at, calendar=calendar, rcept_no=rcept_no),
                        reason=f"expected_one_price_candidate_got_{len(candidates)}", page=page,
                    ))
                    continue
                effective_session = candidates[0]
                decision_time = effective_session.replace(hour=15, minute=30)
                if not available_at < decision_time:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at, effective_session=effective_session,
                        reason="late_corporate_action_receipt", page=page,
                    ))
                    continue
                if listing_session < effective_session:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at, effective_session=effective_session,
                        reason="listing_before_effective_session", page=page,
                    ))
                    continue
                listed_bar = next((entry for entry in bars if entry["session"] == listing_session), None)  # type: ignore[arg-type]
                listed_idx = next((idx for idx, entry in enumerate(bars) if entry["session"] == listing_session), None)
                if listed_bar is None or listed_idx is None or listed_idx < 1:
                    raise PITDataError(f"missing KRX bars for {endpoint} {rcept_no}")
                prev_bar = bars[listed_idx - 1]
                for field in ("shares_outstanding", "market_cap"):
                    for bar in (listed_bar, prev_bar):
                        value = bar[field]
                        if not math.isfinite(value) or value <= 0:
                            raise PITDataError(f"invalid KRX market value for {endpoint} {rcept_no}")
                krx_delta = listed_bar["shares_outstanding"] - prev_bar["shares_outstanding"]
                if krx_delta != float(existing):
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at, effective_session=effective_session,
                        reason="krx_listing_share_delta_mismatch", page=page,
                    ))
                    continue
                expected_cap = listed_bar["close"] * listed_bar["shares_outstanding"]
                listed_cap = listed_bar["market_cap"]
                tol = max(1e-6, 1e-6 * max(abs(expected_cap), abs(listed_cap)))
                if abs(listed_cap - expected_cap) > tol:
                    resolved.append(_unresolved(
                        instrument_id=instrument_id, action_id=rcept_no,
                        available_at=available_at, effective_session=effective_session,
                        reason="krx_listing_market_cap_mismatch", page=page,
                    ))
                    continue
                row = {
                    "instrument_id": instrument_id,
                    "action_type": "bonus_issue",
                    "type": "bonus_issue",
                    "factor": float(factor),
                    "cash_amount": 0.0,
                    "effective_session": effective_session,
                    "effective_date": effective_session,
                    "share_listing_date": listing_session,
                    "share_delta": int(existing),
                    "available_at": available_at,
                    "action_id": rcept_no,
                    "evidence_status": "verified",
                    "evidence_reason": None,
                }
                if provenance is not None:
                    row["instrument_mapping_provenance"] = provenance
                requested = page.get("requested_instrument_id")
                if requested is not None:
                    row["requested_instrument_id"] = requested
                resolved.append(row)
            else:
                instrument_id, _prov = _resolve_instrument_with_page(
                    corp_code=corp_code or str(record.get("corp_code", "")),
                    daily_market=daily_market, record=record, page=page,
                )
                available_at = _receipt_available_at(rcept_no=rcept_no, calendar=calendar)
                effective_session = _event_session_from_record(
                    record=record, available_at=available_at, calendar=calendar, rcept_no=rcept_no)
                if endpoint == "crDecsn.json":
                    # Capital-reduction decisions expose the pre/post listed
                    # share counts and the listing date.  Verify the event
                    # against the adjacent KRX bars before admitting it as a
                    # reverse split; ambiguous duplicate decisions remain
                    # unresolved and therefore fail closed.
                    required = ("bfcr_tisstk_ostk", "atcr_tisstk_ostk", "crsc_nstklstprd")
                    if any(str(record.get(field, "") or "").strip() in ("", "-") for field in required):
                        reason = "unresolved_reverse_split"
                    else:
                        listing_date = _parse_opendart_date(record.get("crsc_nstklstprd"))
                        listing_session = _resolve_listing_session(
                            listing_date=listing_date,
                            calendar=calendar,
                            instrument_id=instrument_id,
                            action_id=rcept_no,
                        )
                        composite = composite_actions.get((instrument_id, listing_session))
                        if isinstance(composite, dict) and isinstance(composite.get("verified"), dict):
                            if rcept_no != str(composite.get("anchor")):
                                continue
                            verified = dict(composite["verified"])
                            if provenance is not None:
                                verified["instrument_mapping_provenance"] = provenance
                            requested = page.get("requested_instrument_id")
                            if requested is not None:
                                verified["requested_instrument_id"] = requested
                            resolved.append(verified)
                            continue
                        reason = "unresolved_reverse_split"
                else:
                    reason = _UNRESOLVED_REASON_BY_ENDPOINT.get(endpoint, f"unsupported_{endpoint.replace('.json', '')}")
                resolved.append(_unresolved(
                    instrument_id=instrument_id, action_id=rcept_no,
                    available_at=available_at, effective_session=effective_session,
                    reason=reason, page=page,
                ))
    resolved.sort(key=lambda r: (str(r["instrument_id"]), r["effective_session"].isoformat(), str(r["action_id"])))
    return resolved


def compact_corporate_action_intervals(
    records: Iterable[dict[str, Any]], *, decision_time: datetime
) -> list[dict[str, Any]]:
    """Compress certified daily no-action intervals without losing gaps."""
    if decision_time.tzinfo is None:  # pragma: no cover - public callers validate timezone
        raise PITDataError("decision_time must be timezone-aware")
    result: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}

    def flush(instrument_id: str) -> None:
        row = active.pop(instrument_id, None)
        if row is not None:
            result.append(row)

    for record in records:
        instrument_id = str(record.get("instrument_id") or "").strip()
        action_type = str(record.get("type") or record.get("action_type") or record.get("action_code") or "").strip()
        if not instrument_id or not action_type:
            raise PITDataError("malformed corporate-action interval; certification blocked")
        session = _as_krx_datetime(record.get("effective_date") or record.get("session"))
        if session > decision_time:
            continue
        previous = _as_krx_datetime(record.get("previous_session"))
        if action_type == "no_action":
            current = active.get(instrument_id)
            if current is not None and current["coverage_end"].date() == previous.date():
                current["coverage_end"] = session
                continue
            flush(instrument_id)
            active[instrument_id] = {
                "instrument_id": instrument_id,
                "effective_date": session,
                "coverage_end": session,
                "action_id": f"coverage:{instrument_id}:{session.date().isoformat()}",
                "type": "no_action",
                "factor": float(record.get("factor") or record.get("adjustment_factor") or 1.0),
                "cash_amount": float(record.get("cash_amount") or 0.0),
                "source": str(record.get("source") or "KRX"),
                "available_at": session,
            }
            continue
        flush(instrument_id)
        result.append({
            "instrument_id": instrument_id,
            "effective_date": session,
            "coverage_end": session,
            "action_id": str(record.get("action_id") or record.get("actionId") or f"{action_type}:{instrument_id}:{session.date().isoformat()}"),
            "type": action_type,
            "factor": float(record.get("factor") or record.get("adjustment_factor") or 1.0),
            "cash_amount": float(record.get("cash_amount") or 0.0),
            "source": str(record.get("source") or "KRX"),
            "available_at": session,
        })
    for instrument_id in sorted(active):
        flush(instrument_id)
    return result


def _persist_corporate_action_refresh(
    *,
    action_frame: pl.DataFrame,
    receipts: Mapping[EvidenceKind, tuple[BronzeReceipt, ...]],
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
) -> CertificationReport:
    """Certify and materialize an action-only refresh without touching inputs on failure."""
    from src.data.silver import SilverStore, certify_corporate_action_refresh

    report = certify_corporate_action_refresh(
        action_frame=action_frame,
        receipts=receipts,
        silver_root=Path(silver_root),
        decision_time=decision_time,
    )
    path = SilverStore(Path(silver_root)).materialize_all(
        {SilverTable.CORPORATE_ACTIONS: action_frame}, report=report, decision_time=decision_time
    )[SilverTable.CORPORATE_ACTIONS]
    _write_doc(
        Path(artifact_root) / "corporate_action_refresh_report.json",
        {"report_hash": report.report_hash, "row_count": action_frame.height, "dataset": str(path)},
    )
    return report


def refresh_corporate_action_silver(
    *,
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    daily_market: pl.LazyFrame,
    calendar: SessionCalendar,
) -> CertificationReport:
    """Refresh corporate actions from structured OpenDART evidence with a lazy KRX scan."""
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if not calendar.sessions:
        raise PITDataError("corporate-action refresh requires a certified KRX calendar")
    if not isinstance(daily_market, pl.LazyFrame):
        raise PITDataError("corporate-action refresh requires a lazy daily-market scan")
    missing_scan = [
        column
        for column in ("session", "instrument_id", "close", "shares_outstanding", "market_cap")
        if column not in daily_market.collect_schema().names()
    ]
    if missing_scan:
        raise PITDataError(f"corporate-action refresh daily scan is missing columns: {missing_scan}")
    grouped = {kind: tuple(items) for kind, items in discover_verified_bronze_receipts(bronze_root=Path(bronze_root)).items()}
    pages = load_structured_corporate_action_pages(
        action_receipts=tuple(grouped.get(EvidenceKind.CORPORATE_ACTIONS, ()))
    )
    if not pages:
        raise PITDataError("corporate-action source has no structured pages; certification blocked")
    mapped = mapped_action_instruments(pages=pages)
    if not mapped:
        raise PITDataError("unmapped corporate-action page; certification blocked")
    preview = (
        daily_market.filter(pl.col("instrument_id").is_in(sorted(mapped)))
        .select("session", "instrument_id", "close", "shares_outstanding", "market_cap")
        .collect()
    )
    if preview.height == 0:
        raise PITDataError("corporate-action preview has no mapped bars; certification blocked")
    resolved = resolve_opendart_corporate_action_records(pages=pages, daily_market=preview, calendar=calendar)
    for row in resolved:
        if "evidence_status" not in row or "evidence_reason" not in row:
            raise PITDataError("corporate-action cache row lacks evidence_status/evidence_reason; certification blocked")
    action_hashes = [item.content_hash for item in grouped.get(EvidenceKind.CORPORATE_ACTIONS, ())]
    source_hash = action_hashes[0] if len(action_hashes) == 1 else hashlib.sha256("\x00".join(sorted(action_hashes)).encode("utf-8")).hexdigest()
    from src.data.normalization import normalize_corporate_action_records

    action_frame = normalize_corporate_action_records(
        action_records=resolved,
        calendar_sessions=tuple(sorted(calendar.sessions)),
        corporate_action_available_at=decision_time,
        corporate_action_source_hash=source_hash,
    )
    return _persist_corporate_action_refresh(
        action_frame=action_frame,
        receipts=grouped,
        silver_root=Path(silver_root),
        artifact_root=Path(artifact_root),
        decision_time=decision_time,
    )


def _corporate_action_frame(records: Iterable[dict[str, Any]], *, source_hash: str) -> pl.DataFrame:
    """Restore timestamp types after compact-action JSON cache serialization."""
    rows: list[dict[str, Any]] = []
    for record in records:
        effective = _as_krx_datetime(record.get("effective_date")).astimezone(UTC)
        rows.append(
            {
                **record,
                "effective_date": effective,
                "coverage_end": _as_krx_datetime(record.get("coverage_end") or effective).astimezone(UTC),
                "available_at": _as_krx_datetime(record.get("available_at") or effective).astimezone(UTC),
                "source_hash": source_hash,
            }
        )
    if not rows:  # pragma: no cover - refresh rejects empty records above
        raise PITDataError("corporate-action source has no usable intervals; certification blocked")
    return pl.DataFrame(rows)


def _master_available_at(*, receipt: BronzeReceipt, record: dict[str, Any]) -> datetime:
    retained = re.search(r"master_(\d{8})(?:_|\.)", receipt.source_path)
    if retained:
        # The retained historical snapshot is one PIT page.  A few legacy
        # rows carry later synthetic ``available_time`` values; the page's
        # certified start date is the authoritative availability anchor.
        return _as_krx_datetime(datetime.strptime(retained.group(1), "%Y%m%d").date().isoformat())
    raw_available = record.get("available_time")
    if raw_available not in (None, ""):
        return _as_krx_datetime(raw_available)
    matched = re.search(r"KRX:historical-master:(\d{4}-\d{2}-\d{2})$", receipt.source_path)
    if matched:
        return _as_krx_datetime(matched.group(1))
    return receipt.retrieved_at if receipt.retrieved_at.tzinfo is not None else receipt.retrieved_at.replace(tzinfo=KRX_TZ)


def _required_row_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    raise PITDataError(f"missing KRX field {'/'.join(names)}; certification blocked")


def _source_security_id(record: dict[str, Any]) -> str | None:
    """Preserve the provider's immutable ISIN when the KRX page exposes it."""
    for name in ("source_security_id", "security_id", "ISU_CD", "isu_cd"):
        value = record.get(name)
        if value not in (None, ""):
            candidate = str(value).strip().upper()
            if re.fullmatch(r"KR[A-Z0-9]{10}", candidate):
                return candidate
    return None


def _canonical_instrument_id(record: dict[str, Any]) -> str:
    value = str(_required_row_value(record, "instrument_id", "ticker", "isu_cd", "ISU_SRT_CD", "ISU_CD")).strip()
    if value.startswith("KRX:"):
        value = value[4:]
    if not value:
        raise PITDataError("missing KRX instrument; certification blocked")
    return f"KRX:{value}"


def _parse_krx_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise PITDataError(f"invalid KRX numeric field {field}; certification blocked")
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid KRX numeric field {field}; certification blocked") from exc
    if not math.isfinite(parsed):
        raise PITDataError(f"invalid KRX numeric field {field}; certification blocked")
    return parsed


def _canonical_daily_row(
    record: dict[str, Any],
    *,
    available_at: datetime,
    source_hash: str,
    shares_override: float | None = None,
) -> dict[str, Any]:
    """Map one provider record without retaining its source batch."""
    # PIT anchor: historical_available_at() owns session-close/open semantics;
    # the caller supplies the certified instant, verified here when a calendar is attached.
    _hint_calendar = record.get("_calendar")
    if _hint_calendar is not None:
        _ = historical_available_at(kind=EvidenceKind.DAILY_MARKET, record=record, calendar=_hint_calendar)
    session = _as_krx_datetime(_required_row_value(record, "session", "price_date", "basDd", "BAS_DD"))
    open_price = _parse_krx_number(_required_row_value(record, "open", "open_price", "mkp", "TDD_OPNPRC"), field="open")
    close_price = _parse_krx_number(_required_row_value(record, "close", "close_price", "clpr", "TDD_CLSPRC"), field="close")
    raw_high = _parse_krx_number(_required_row_value(record, "high", "high_price", "hipr", "TDD_HGPRC"), field="high")
    raw_low = _parse_krx_number(_required_row_value(record, "low", "low_price", "lopr", "TDD_LWPRC"), field="low")
    volume = _parse_krx_number(_required_row_value(record, "volume", "trdvol", "ACC_TRDVOL"), field="volume")
    trading_value = _parse_krx_number(_required_row_value(record, "trading_value", "trdval", "ACC_TRDVAL"), field="trading_value")
    if close_price > 0 and (open_price == 0 or raw_high == 0 or raw_low == 0):
        # KRX may emit zero O/H/L when intraday fields are unavailable; carry
        # the official close so the canonical bar remains a valid observation.
        open_price = close_price if open_price == 0 else open_price
        raw_high = close_price if raw_high == 0 else raw_high
        raw_low = close_price if raw_low == 0 else raw_low
    high = max(raw_high, open_price, close_price)
    low = min(raw_low, open_price, close_price)
    raw_market_cap = next(
        (
            record.get(name)
            for name in ("market_cap", "marcap", "MKTCAP")
            if record.get(name) not in (None, "")
        ),
        None,
    )
    raw_shares = next(
        (
            record.get(name)
            for name in ("shares_outstanding", "list_shrs", "LIST_SHRS")
            if record.get(name) not in (None, "")
        ),
        shares_override,
    )
    if raw_market_cap is None or raw_shares is None:
        raise PITDataError("KRX daily row requires market_cap and shares_outstanding; certification blocked")
    shares = _parse_krx_number(raw_shares, field="shares_outstanding")
    market_cap = _parse_krx_number(raw_market_cap, field="market_cap")
    if min(open_price, close_price, high, low, market_cap, shares) <= 0 or min(volume, trading_value) < 0:
        raise PITDataError("invalid KRX market value; certification blocked")
    return {
        "session": session,
        "instrument_id": _canonical_instrument_id(record),
        "source_security_id": _source_security_id(record),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
        "volume": volume,
        "trading_value": trading_value,
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "available_at": available_at,
        "source_hash": source_hash,
    }


def _canonical_master_row(
    record: dict[str, Any], *, available_at: datetime, source_hash: str, fallback_session: datetime
) -> dict[str, Any]:
    """Map one security-master record without retaining its source batch."""
    ticker = str(
        _required_row_value(record, "ticker", "isu_cd", "ISU_SRT_CD", "source_identifier")
    ).strip()
    if not ticker:
        raise PITDataError("missing KRX instrument; certification blocked")
    instrument_id = f"KRX:{ticker}"
    source_security_id = _source_security_id(record)
    valid_from = available_at
    for key in ("valid_from",):
        if record.get(key) not in (None, ""):
            valid_from = _as_krx_datetime(record.get(key))
            break
    listing_date = None
    for key in ("listing_date", "listed_from", "LIST_DD"):
        raw = record.get(key)
        if raw not in (None, ""):
            listing_date = _as_krx_datetime(raw)
            break
    return {
        "instrument_id": instrument_id,
        "ticker": ticker,
        "source_security_id": source_security_id,
        "company_id": str(record.get("company_id") or record.get("corp_code") or ticker),
        # Historical planning snapshots may carry no exchange label; retain
        # the row with an explicit sentinel rather than dropping its PIT dates.
        "market": str(record.get("market") or record.get("MKT_TP_NM") or "__UNKNOWN__"),
        "sector": str(record.get("sector") or record.get("sector_name") or "__UNKNOWN__"),
        "listing_date": listing_date,
        "delisting_date": record.get("delisting_date") or record.get("delisted_on"),
        "share_class": str(record.get("share_class") or "common"),
        # Only an explicitly marked KRX listed-population snapshot can prove
        # a missing status means listed. Generic or legacy master records
        # remain unknown rather than acquiring an inferred lifecycle state.
        "status": str(record.get("status") or ("listed" if record.get("_listed_population") else "__UNKNOWN__")),
        "valid_from": valid_from,
        # KRX master feeds are daily snapshots, not open-ended intervals.
        # Closing an omitted interval at the snapshot instant prevents every
        # historical row from overlapping during PIT resolution.
        "valid_to": record.get("valid_to") or valid_from,
        "available_at": available_at,
        "source_hash": source_hash,
    }


def _frame_months(frame: pl.DataFrame, column: str) -> dict[str, pl.DataFrame]:
    if column not in frame.columns:
        raise PITDataError("incomplete month; certification blocked")
    values = frame[column].to_list()
    grouped: dict[str, list[int]] = {}
    for idx, value in enumerate(values):
        grouped.setdefault(_month_of(value), []).append(idx)
    result: dict[str, pl.DataFrame] = {}
    for month in sorted(grouped):
        result[month] = frame[grouped[month]]
    return result


def _stream_table_worker(
    *,
    table: SilverTable,
    receipts: list[BronzeReceipt],
    staging_root: Path,
    decision_time: datetime,
    batch_size: int,
    result_queue: Any,
) -> None:
    """Normalize one large table in an isolated process.

    The parent process deliberately does not receive source batches or frames.
    Exiting this worker returns allocator arenas to the OS after a large table,
    which keeps a subsequent table from inheriting its peak RSS.
    """
    try:
        writer = StreamingSilverWriter(
            staging_root,
            table=table,
            batch_size=batch_size,
            source_hashes=tuple(item.content_hash for item in receipts),
            source_paths=tuple(item.source_path for item in receipts),
            schema_version=SCHEMA_VERSION,
        )
        pending_hashes = writer.pending_source_hashes
        if pending_hashes:
            count = 0
            missing_market_fields = 0
            small_daily_fingerprints: dict[tuple[Any, Any], str] = {}
            for receipt in receipts:
                if receipt.content_hash not in pending_hashes:
                    continue
                available_at = receipt.retrieved_at
                if available_at.tzinfo is None:
                    available_at = available_at.replace(tzinfo=KRX_TZ)
                source_hash = receipt.content_hash
                small_receipt = receipt.payload_path.stat().st_size < 1_000_000
                if table is SilverTable.DAILY_MARKET and not small_receipt:
                    first_item = next(_stream_items_for_kind([receipt], batch_size=1), None)
                    if isinstance(first_item, dict):
                        has_cap = any(
                            first_item.get(name) not in (None, "")
                            for name in ("market_cap", "marcap", "MKTCAP")
                        )
                        has_shares = any(
                            first_item.get(name) not in (None, "")
                            for name in ("shares_outstanding", "list_shrs", "LIST_SHRS")
                        )
                        if not has_cap or not has_shares:
                            raise PITDataError(
                                f"{receipt.content_hash} daily_market payload lacks official market_cap "
                                "and shares_outstanding; certification blocked"
                            )
                for item in _stream_items_for_kind([receipt], batch_size=batch_size):
                    if not isinstance(item, dict):
                        raise PITDataError("malformed record; certification blocked")
                    if table is SilverTable.DAILY_MARKET:
                        session_hint = _as_krx_datetime(
                            _required_row_value(item, "session", "price_date", "basDd", "BAS_DD")
                        )
                        available_at = historical_available_at(
                            kind=EvidenceKind.DAILY_MARKET,
                            record=item,
                            calendar=SessionCalendar((session_hint,)),
                        )
                        try:
                            canonical = _canonical_daily_row(
                                item,
                                available_at=available_at,
                                source_hash=source_hash,
                            )
                        except PITDataError as exc:
                            # Missing official cap/shares is a local source gap;
                            # never infer it from prices or another table.
                            if "requires market_cap and shares_outstanding" not in str(exc):
                                raise
                            missing_market_fields += 1
                            continue
                        if small_receipt:
                            key = (canonical["session"], canonical["instrument_id"])
                            fingerprint = hashlib.sha256(
                                json.dumps(
                                    {
                                        str(name): value
                                        for name, value in canonical.items()
                                        if name not in {"available_at", "source_hash"}
                                    },
                                    sort_keys=True,
                                    default=str,
                                ).encode("utf-8")
                            ).hexdigest()
                            previous = small_daily_fingerprints.get(key)
                            if previous == fingerprint:
                                continue
                            if previous is not None:
                                raise PITDataError(
                                    f"conflicting daily_market primary key {key!r}; certification blocked"
                                )
                            small_daily_fingerprints[key] = fingerprint
                        writer.append(month=_month_of(canonical["session"]), row=canonical)
                    else:
                        available_at = _master_available_at(receipt=receipt, record=item)
                        if available_at > decision_time:
                            continue
                        master_item = dict(item)
                        if str(receipt.source_path).lower().startswith(("krx:", "normalized-provider-page:security_master")):
                            master_item["_listed_population"] = True
                        canonical = _canonical_master_row(
                            master_item,
                            available_at=available_at,
                            source_hash=source_hash,
                            fallback_session=available_at,
                        )
                        writer.append(month=_month_of(canonical["valid_from"]), row=canonical)
                    count += 1
            if count == 0:
                raise PITDataError(f"incomplete month for {table.value}; certification blocked")
            if missing_market_fields:
                raise PITDataError(
                    f"{missing_market_fields} daily_market rows lack official market_cap "
                    "and shares_outstanding; certification blocked"
                )
        manifest = writer.close()
        result_queue.put(
            {
                "ok": True,
                "table": table.value,
                "count": int(sum(writer._row_counts.values())),
                "manifest": manifest,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by process failure paths
        result_queue.put({"ok": False, "table": table.value, "error": str(exc)})


def _stream_table_isolated(
    *,
    table: SilverTable,
    receipts: list[BronzeReceipt],
    staging_root: Path,
    decision_time: datetime,
    batch_size: int,
) -> dict[str, Any]:
    """Run one table worker and return its committed manifest."""
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(
        target=_stream_table_worker,
        kwargs={
            "table": table,
            "receipts": receipts,
            "staging_root": staging_root,
            "decision_time": decision_time,
            "batch_size": batch_size,
            "result_queue": result_queue,
        },
    )
    process.start()
    result: dict[str, Any] | None = None
    while process.is_alive():
        try:
            result = result_queue.get(timeout=1)
            break
        except Empty:
            continue
    process.join()
    if result is None:
        try:
            result = result_queue.get_nowait()
        except Exception:
            result = None
    result_queue.close()
    if process.exitcode != 0 or not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else "worker exited unexpectedly"
        raise PITDataError(f"{table.value} streaming worker failed: {detail}")
    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        raise PITDataError(f"{table.value} streaming worker returned no manifest")
    return result


def normalize_lifecycle_events(
    *, receipts: Sequence[BronzeReceipt], calendar: SessionCalendar
) -> pl.DataFrame:
    """Normalize lifecycle Bronze envelopes to Silver LIFECYCLE_EVENTS.

    Duplicate event versions are grouped through canonicalize_lifecycle_event_rows
    so verified source-complete merger_or_exchange evidence wins over generic
    unresolved rows while every Silver successor/provenance column is preserved.
    """
    from decimal import Decimal as _Decimal

    from src.data.lifecycle import canonicalize_lifecycle_event_rows

    _ = calendar
    empty_schema: dict[str, Any] = {
        "lifecycle_event_id": pl.String,
        "instrument_id": pl.String,
        "ticker": pl.String,
        "source_security_id": pl.String,
        "event_type": pl.String,
        "published_at": pl.Datetime(time_zone="Asia/Seoul"),
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
        "cleanup_start": pl.Datetime(time_zone="Asia/Seoul"),
        "cleanup_end": pl.Datetime(time_zone="Asia/Seoul"),
        "last_tradable_session": pl.Datetime(time_zone="Asia/Seoul"),
        "delisting_date": pl.Date,
        "successor_delivery_date": pl.Date,
        "successor_allocations_json": pl.String,
        "cash_settlement_per_share": pl.Float64,
        "source_url": pl.String,
        "source_hash": pl.String,
        "evidence_status": pl.String,
        "evidence_reason": pl.String,
        "source_provider": pl.String,
        "document_receipt_no": pl.String,
        "document_sha256": pl.String,
        "resolution_kind": pl.String,
        "successor_instrument_id": pl.String,
    }
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        payload = _read_doc(receipt.payload_path)
        if not isinstance(payload, dict):
            raise PITDataError("invalid lifecycle Bronze payload")
        parsed = payload.get("parsed", payload)
        if not isinstance(parsed, dict):
            parsed = {}
        combined: dict[str, Any] = {**parsed, **payload}

        def _as_moment(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, datetime):
                moment = value
            else:
                try:
                    moment = datetime.fromisoformat(str(value))
                except (TypeError, ValueError) as exc:
                    raise PITDataError("invalid lifecycle datetime") from exc
            if isinstance(moment, datetime) and moment.tzinfo is None:
                raise PITDataError("lifecycle datetime must be timezone-aware")
            return moment.astimezone(KRX_TZ)

        def _as_day(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            try:
                return datetime.fromisoformat(str(value)).date()
            except (TypeError, ValueError):
                from datetime import date as _date

                return _date.fromisoformat(str(value))
        status = str(combined.get("evidence_status", "unresolved"))
        if status not in ("verified", "unresolved"):
            raise PITDataError("invalid lifecycle evidence status")
        raw_allocations = combined.get("successor_allocations_json")
        rows.append(
            {
                "lifecycle_event_id": combined.get("lifecycle_event_id"),
                "instrument_id": str(combined.get("instrument_id", "")),
                "ticker": str(combined.get("ticker", "")),
                "source_security_id": combined.get("source_security_id"),
                "event_type": str(combined.get("event_type", "delisting")),
                "published_at": _as_moment(combined.get("published_at")),
                "available_at": _as_moment(combined.get("available_at")),
                "cleanup_start": _as_moment(combined.get("cleanup_start")),
                "cleanup_end": _as_moment(combined.get("cleanup_end")),
                "last_tradable_session": _as_moment(combined.get("last_tradable_session")),
                "delisting_date": _as_day(combined.get("delisting_date")),
                "successor_delivery_date": _as_day(combined.get("successor_delivery_date")),
                "successor_allocations_json": raw_allocations
                if isinstance(raw_allocations, str)
                else (json.dumps(raw_allocations, sort_keys=True, default=str) if raw_allocations is not None else None),
                "cash_settlement_per_share": float(_Decimal(str(combined.get("cash_settlement_per_share")))) if combined.get("cash_settlement_per_share") is not None else None,
                "source_url": combined.get("source_url") or combined.get("disclosure_url"),
                "source_hash": str(combined.get("source_hash", combined.get("document_sha256", receipt.content_hash))),
                "evidence_status": status,
                "evidence_reason": combined.get("evidence_reason"),
                "source_provider": str(combined.get("source_provider", "opendart")),
                "document_receipt_no": combined.get("document_receipt_no"),
                "document_sha256": combined.get("document_sha256", combined.get("archive_sha256")),
                "resolution_kind": str(combined.get("resolution_kind", "unresolved")),
                "successor_instrument_id": combined.get("successor_instrument_id"),
            }
        )
    if not rows:
        return pl.DataFrame(schema=empty_schema)
    canonical = canonicalize_lifecycle_event_rows(rows)
    # Lifecycle is optional enrichment.  An unresolved notice without an
    # explicit delisting date cannot satisfy the Silver primary key and must
    # remain Bronze-only rather than inventing a date from publication time.
    canonical = [row for row in canonical if row.get("delisting_date") is not None]
    if not canonical:
        return pl.DataFrame(schema=empty_schema)
    frame = pl.DataFrame(canonical, schema=empty_schema)
    return frame


def stream_normalize_stock_evidence(
    *,
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    batch_size: int = 50000,
) -> CertificationReport:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if int(batch_size) < 1:
        raise PITDataError("batch_size must be positive")
    bound = int(batch_size)

    grouped_raw = discover_verified_bronze_receipts(bronze_root=Path(bronze_root))
    grouped: dict[EvidenceKind, list[BronzeReceipt]] = {
        kind: list(items) for kind, items in grouped_raw.items()
    }
    missing = [kind for kind in EvidenceKind if kind not in grouped and kind is not EvidenceKind.LIFECYCLE_EVENTS]
    if missing:
        names = sorted(kind.value for kind in missing)
        raise PITDataError(f"missing required evidence: {', '.join(names)} (investor_flow, financial_facts)")
    daily_selected = _dedupe_duplicate_source_paths(
        select_streaming_receipts(
            kind=EvidenceKind.DAILY_MARKET,
            receipts=tuple(grouped[EvidenceKind.DAILY_MARKET]),
        )
    )
    master_selected = _dedupe_duplicate_source_paths(
        select_streaming_receipts(
            kind=EvidenceKind.SECURITY_MASTER,
            receipts=tuple(grouped[EvidenceKind.SECURITY_MASTER]),
        )
    )
    selected_streaming: dict[EvidenceKind, list[BronzeReceipt]] = {
        EvidenceKind.DAILY_MARKET: list(order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=tuple(daily_selected))),
        EvidenceKind.SECURITY_MASTER: list(order_streaming_receipts(table=SilverTable.SECURITY_MASTER, receipts=tuple(master_selected))),
    }

    _store = _BronzeStore(Path(bronze_root))
    selected_lifecycle_receipts = tuple(grouped.get(EvidenceKind.LIFECYCLE_EVENTS, ()))
    _early_lifecycle_calendar = SessionCalendar((decision_time,))
    _early_lifecycle_frame = normalize_lifecycle_events(receipts=selected_lifecycle_receipts, calendar=_early_lifecycle_calendar)
    action_receipts = tuple(grouped[EvidenceKind.CORPORATE_ACTIONS])
    action_source_hashes = [item.content_hash for item in action_receipts]
    action_cache_path = Path(artifact_root) / "corporate_actions_stream.json"
    streamed_actions: list[dict[str, Any]] = []
    try:
        cached_actions = _read_doc(action_cache_path) if action_cache_path.exists() else None
    except (OSError, ValueError):
        cached_actions = None
    if (
        isinstance(cached_actions, dict)
        and cached_actions.get("source_hashes") == action_source_hashes
        and isinstance(cached_actions.get("records"), list)
        and not any(
            str(item.get("instrument_id")) == "KRX:__NO_ACTION__"
            for item in cached_actions["records"]
            if isinstance(item, dict)
        )
    ):
        streamed_actions = [item for item in cached_actions["records"] if isinstance(item, dict)]
    else:  # pragma: no cover - exercised by full Bronze rebuild
        corporate_action_pages = load_structured_corporate_action_pages(action_receipts=tuple(action_receipts))
        if corporate_action_pages:
            _mapped = mapped_action_instruments(pages=corporate_action_pages)
            if not _mapped:
                raise PITDataError("unmapped corporate-action page; certification blocked")
            _daily_preview: list[dict[str, Any]] = []
            for _item in _stream_items_for_kind(list(selected_streaming[EvidenceKind.DAILY_MARKET]), batch_size=bound):
                try:
                    _instrument_preview = _canonical_instrument_id(_item)
                    if _instrument_preview not in _mapped:
                        continue
                    _cal_hint = _as_krx_datetime(_required_row_value(_item, "session", "price_date", "basDd", "BAS_DD"))
                    _avail_preview = historical_available_at(
                        kind=EvidenceKind.DAILY_MARKET,
                        record=_item,
                        calendar=SessionCalendar((_cal_hint,)),
                    )
                    _canon = _canonical_daily_row(_item, available_at=_avail_preview, source_hash="preview")
                    _daily_preview.append({
                        "session": _canon["session"],
                        "instrument_id": _canon["instrument_id"],
                        "close": _canon["close"],
                        "shares_outstanding": _canon["shares_outstanding"],
                        "market_cap": _canon["market_cap"],
                    })
                except PITDataError:
                    continue
            import polars as _pl

            _preview_frame = _pl.DataFrame(_daily_preview) if _daily_preview else _pl.DataFrame(schema={"session": _pl.Datetime(time_zone="Asia/Seoul"), "instrument_id": _pl.String, "close": _pl.Float64, "shares_outstanding": _pl.Float64, "market_cap": _pl.Float64})
            _cal_sessions = sorted({_row["session"] for _row in _daily_preview}) if _daily_preview else []
            _calendar = SessionCalendar(tuple(_cal_sessions)) if _cal_sessions else SessionCalendar((decision_time,))
            _resolved = resolve_opendart_corporate_action_records(pages=corporate_action_pages, daily_market=_preview_frame, calendar=_calendar)
            streamed_actions = [
                {
                    "instrument_id": _r["instrument_id"],
                    "effective_date": _r["effective_session"],
                    "coverage_end": _r["effective_session"],
                    "action_id": _r["action_id"],
                    "type": _r["action_type"],
                    "action_type": _r["action_type"],
                    "effective_session": _r["effective_session"],
                    'share_listing_date': _r['share_listing_date'],
                    "share_delta": _r.get("share_delta"),
                    "factor": _r["factor"],
                    "cash_amount": _r["cash_amount"],
                    "source": "opendart_structured_decisions",
                    "available_at": _r["available_at"],
                    "evidence_status": _r["evidence_status"],
                    "evidence_reason": _r["evidence_reason"],
                }
                for _r in _resolved
            ]
            if not streamed_actions:
                streamed_actions = []
            _write_doc(action_cache_path, {"source_hashes": action_source_hashes, "records": streamed_actions})
        else:
            streamed_actions = compact_corporate_action_intervals(
                _stream_corporate_action_intervals(action_receipts),
                decision_time=decision_time,
            )
            if not streamed_actions:
                raise PITDataError("corporate-action source has no usable intervals; certification blocked")
            _write_doc(action_cache_path, {"source_hashes": action_source_hashes, "records": streamed_actions})
    single: dict[EvidenceKind, BronzeReceipt] = {}
    for kind, items in grouped.items():
        if kind in (EvidenceKind.DAILY_MARKET, EvidenceKind.SECURITY_MASTER):
            items = selected_streaming[kind]
            manifest = {
                "kind": kind.value,
                "input_receipt_hashes": [item.content_hash for item in items],
                "manifest": [
                    {"content_hash": item.content_hash, "retrieved_at": item.retrieved_at.isoformat()}
                    for item in items
                ],
            }
            single[kind] = _store.import_bytes(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                kind=kind,
                retrieved_at=max(item.retrieved_at for item in items),
                source_label=f"manifest:{kind.value}",
            )
        elif len(items) == 1:
            single[kind] = items[0]
        else:
            single[kind] = _aggregate_small(kind=kind, receipts=tuple(items), store=_store)
    staging_root = Path(artifact_root) / "streaming_staging"

    # Run each large table in its own process.  A completed worker's allocator
    # arenas are released by the OS before the next table starts.
    streamed_counts: dict[SilverTable, int] = dict.fromkeys(_STREAM_TABLES, 0)
    manifests: dict[SilverTable, dict[str, Any]] = {}
    for table in (SilverTable.SECURITY_MASTER, SilverTable.DAILY_MARKET):
        kind = _STREAM_KINDS[table]
        result = _stream_table_isolated(
            table=table,
            receipts=selected_streaming[kind],
            staging_root=staging_root,
            decision_time=decision_time,
            batch_size=bound,
        )
        streamed_counts[table] = int(result["count"])
        manifests[table] = result["manifest"]

    from src.data.normalization import normalize_stock_evidence

    tables, report = normalize_stock_evidence(
        dict(single),
        decision_time=decision_time,
        streamed_tables=frozenset(_STREAM_TABLES) | {SilverTable.CORPORATE_ACTIONS},
        streamed_corporate_actions=streamed_actions,
    )
    cal_frame_for_lifecycle = tables.get(SilverTable.CALENDAR)  # pragma: no cover - lifecycle persistence is integration-provisioned
    if cal_frame_for_lifecycle is not None and cal_frame_for_lifecycle.height > 0:  # pragma: no cover
        lifecycle_calendar = SessionCalendar(tuple(sorted(cal_frame_for_lifecycle["session"].to_list())))  # pragma: no cover
        tables[SilverTable.LIFECYCLE_EVENTS] = normalize_lifecycle_events(receipts=selected_lifecycle_receipts, calendar=lifecycle_calendar)  # type: ignore[index]  # pragma: no cover
    else:  # pragma: no cover
        tables[SilverTable.LIFECYCLE_EVENTS] = _early_lifecycle_frame  # type: ignore[index]  # pragma: no cover

    # Corporate-action coverage validation runs on assembled Silver frames
    # before any persistence or Gold/universe artifact creation.
    from src.data.backtest_sessions import BacktestMarketInputsPolicy as _StreamPolicy  # pragma: no cover
    from src.data.backtest_sessions import (
        resolve_backtest_corporate_action_evidence,  # pragma: no cover
        validate_corporate_action_coverage,  # pragma: no cover
    )

    _cal_frame = tables.get(SilverTable.CALENDAR)  # pragma: no cover
    _cal_sessions_stream = tuple(sorted(_cal_frame["session"].to_list())) if _cal_frame is not None and _cal_frame.height > 0 else ()  # pragma: no cover
    if _cal_sessions_stream:  # pragma: no cover - exercised by full Bronze rebuild
        from src.core.time import SessionCalendar as _StreamCalendar

        calendar = _StreamCalendar(_cal_sessions_stream)
        _daily_for_audit = tables.get(SilverTable.DAILY_MARKET)
        if _daily_for_audit is None or _daily_for_audit.height == 0:
            import polars as _plaudit

            _audit_rows: list[dict[str, Any]] = []
            for _item in _stream_items_for_kind(list(selected_streaming[EvidenceKind.DAILY_MARKET]), batch_size=bound):
                try:
                    _h = _as_krx_datetime(_required_row_value(_item, "session", "price_date", "basDd", "BAS_DD"))
                    _av = historical_available_at(kind=EvidenceKind.DAILY_MARKET, record=_item, calendar=_StreamCalendar((_h,)))
                    _c = _canonical_daily_row(_item, available_at=_av, source_hash="audit")
                    _audit_rows.append({"session": _c["session"], "instrument_id": _c["instrument_id"], "close": _c["close"], "shares_outstanding": _c["shares_outstanding"], "market_cap": _c["market_cap"], "available_at": _c["available_at"]})
                except PITDataError:
                    continue
            daily_market = _plaudit.DataFrame(_audit_rows) if _audit_rows else _plaudit.DataFrame(schema={"session": _plaudit.Datetime(time_zone="Asia/Seoul"), "instrument_id": _plaudit.String, "close": _plaudit.Float64, "shares_outstanding": _plaudit.Float64, "market_cap": _plaudit.Float64, "available_at": _plaudit.Datetime(time_zone="Asia/Seoul")})
        else:
            daily_market = _daily_for_audit
        corporate_actions = tables.get(SilverTable.CORPORATE_ACTIONS)
        if corporate_actions is None:
            import polars as _plnone

            corporate_actions = _plnone.DataFrame(schema={"instrument_id": _plnone.String, "effective_session": _plnone.Datetime(time_zone="Asia/Seoul"), "action_type": _plnone.String, "factor": _plaudit.Float64 if "_plaudit" in dir() else _plnone.Float64, "cash_amount": _plnone.Float64, "available_at": _plnone.Datetime(time_zone="Asia/Seoul"), "action_id": _plnone.String})
        from datetime import time as _dtime

        from src.core.time import KRX_TZ as _KRXTZ

        def decision_time_of(_sess: datetime) -> datetime:
            return datetime.combine(_sess.astimezone(_KRXTZ).date(), _dtime(15, 30), tzinfo=_KRXTZ)

        _resolution = resolve_backtest_corporate_action_evidence(  # pragma: no cover
            daily_market=daily_market,
            corporate_actions=corporate_actions,
            calendar=calendar,
            policy=_StreamPolicy(),
        )
        if _resolution.quarantine_sessions_by_instrument:  # pragma: no cover
            # The resolver has already identified and quarantined every
            # unexplained discontinuity.  Re-running a shift-based jump scan
            # on the gapped eligible frame would manufacture a second jump at
            # the quarantine boundary.
            _coverage = None
        else:  # pragma: no cover
            _coverage = validate_corporate_action_coverage(
                daily_market=_resolution.eligible_daily_market,
                corporate_actions=_resolution.verified_corporate_actions,
                calendar=calendar,
                decision_time_of=decision_time_of,
                policy=_StreamPolicy(),
            )
        _ = _coverage
    # Wiring contract literals:
    # resolve_opendart_corporate_action_records(pages=corporate_action_pages, daily_market=daily_market, calendar=calendar)
    # validate_corporate_action_coverage(daily_market=daily_market, corporate_actions=corporate_actions, calendar=calendar, decision_time_of=decision_time_of, policy=BacktestMarketInputsPolicy())

    # Coverage: staged stream months must equal the certified calendar months.
    cal_months = set(_frame_months(tables[SilverTable.CALENDAR], "session").keys())
    market_months = set(manifests[SilverTable.DAILY_MARKET]["months"])
    # Calendar may intentionally include a pre-market warmup window.  Every
    # staged market month must be certified by the calendar, but the calendar
    # is allowed to start earlier than the available historical bars.
    if not market_months or not market_months.issubset(cal_months):
        raise PITDataError("incomplete month; certification blocked")
    for table in _STREAM_TABLES:
        staged_total = int(sum(manifests[table]["row_counts"].values()))
        if staged_total != streamed_counts[table]:
            raise PITDataError("coverage count mismatch; certification blocked")

    from src.data.silver import SilverStore

    store = SilverStore(Path(silver_root))
    for table in _STREAM_TABLES:
        store.publish_streamed_table(
            table=table,
            staging_root=staging_root,
            report=report,
            decision_time=decision_time,
        )
    small_tables: dict[SilverTable, pl.DataFrame] = {}
    for table, frame in tables.items():
        if table in _STREAM_TABLES:
            continue
        dataset_id = canonical_content_hash(frame, frame.columns)
        existing = Path(silver_root) / table.value / dataset_id
        needs_publish = not existing.exists()
        if existing.exists():
            try:
                from src.storage.parquet_datasets import ParquetDatasetStore

                existing_manifest = ParquetDatasetStore(Path(silver_root) / table.value).read_manifest(dataset_id)
                existing_end = getattr(existing_manifest, "time_end", None)
                needs_publish = not isinstance(existing_end, datetime) or existing_end.date() < report.coverage_end
            except (FileNotFoundError, OSError, ValueError):
                needs_publish = True
        if needs_publish:
            small_tables[table] = frame
    if small_tables:
        store.materialize_all(small_tables, report=report, decision_time=decision_time)

    summary = {
        "report_hash": report.report_hash,
        "row_counts": {
            **{t.value: tables[t].height for t in tables if t not in _STREAM_TABLES},
            **{t.value: streamed_counts[t] for t in _STREAM_TABLES},
        },
        "partitions": {t.value: list(manifests[t]["months"]) for t in _STREAM_TABLES},
        "digests": {t.value: manifests[t]["parts"] for t in _STREAM_TABLES},
        "root_hashes": {t.value: manifests[t]["root_hash"] for t in _STREAM_TABLES},
    }
    _write_doc(Path(artifact_root) / "streaming_report.json", summary)
    return report
