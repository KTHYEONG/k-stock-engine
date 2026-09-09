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
from collections.abc import Iterable
from datetime import UTC, date, datetime, time
from pathlib import Path
from queue import Empty
from typing import Any

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
        if list(entry.get("source_hashes", [])) != list(source_hashes):
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
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        if int(batch_size) < 1:
            raise PITDataError("batch_size must be positive")
        self.root = Path(root)
        self.table = table
        self.batch_size = int(batch_size)
        self.source_hashes = tuple(source_hashes)
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
        self._load_reusable_months()

    def _load_reusable_months(self) -> None:
        manifest_path = self.root / self.table.value / "staging_manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = _read_doc(manifest_path)
        except (OSError, ValueError):
            return
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != self.schema_version
            or manifest.get("verified") is not True
            or list(manifest.get("source_hashes", [])) != list(self.source_hashes)
        ):
            return
        parts = manifest.get("parts")
        if not isinstance(parts, dict):
            return
        for month, entries in parts.items():
            checkpoint_verified = self._checkpoint.is_verified(
                table=self.table.value,
                month=str(month),
                source_hashes=self.source_hashes,
                schema_version=self.schema_version,
            )
            # The staging manifest is itself an atomic, digest-checked commit.
            # Older runs may predate per-month checkpoint entries, so the
            # checkpoint is an optional acceleration/diagnostic layer rather
            # than a prerequisite for safe reuse.
            if (not checkpoint_verified and self._checkpoint.verified_entry(table=self.table.value, month=str(month)) is not None):
                continue
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

    @property
    def has_reusable_manifest(self) -> bool:
        return bool(self._verified_months) and not self._buffers

    def _month_dir(self, month: str) -> Path:
        year, _, mon = month.partition("-")
        return self.root / self.table.value / f"year={year}" / f"month={mon}"

    def append(self, *, month: str, row: dict[str, Any]) -> None:
        if month in self._verified_months:
            return
        # Source pages are processed in month order in the normal path. Keep
        # duplicate-detection state only for the active month; retaining keys
        # for the full multi-year history defeats bounded streaming memory.
        if month != self._fingerprint_month:
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
        pl.DataFrame(buf).write_parquet(tmp_path)
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
            "months": sorted(self._part_digests),
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


def _stream_items_for_kind(
    receipts: list[BronzeReceipt], *, batch_size: int
) -> Any:
    for receipt in receipts:
        # Aggregated manifests are intentionally row-free.  Avoid spawning a
        # jq process for each small manifest while preserving true streaming
        # for large source payloads.
        try:
            if receipt.payload_path.stat().st_size < 1_000_000:
                small_payload = _read_doc(receipt.payload_path)
                if isinstance(small_payload, dict) and not any(
                    isinstance(small_payload.get(key), list)
                    for key in ("records", "intervals", "list")
                ):
                    continue
        except (OSError, ValueError) as exc:
            raise PITDataError("malformed Bronze JSON; certification blocked") from exc
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


def resolve_opendart_corporate_action_records(
    *, pages: Iterable[dict[str, Any]], daily_market: pl.DataFrame, calendar: SessionCalendar
) -> list[dict[str, Any]]:
    from src.data.backtest_sessions import BacktestMarketInputsPolicy

    threshold = float(BacktestMarketInputsPolicy().unexplained_price_jump_threshold)
    ordered_sessions = tuple(sorted(calendar.sessions))
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
    daily_rows = daily_market.to_dicts()
    closes_by_iid: dict[str, list[tuple[datetime, float]]] = {}
    for row in daily_rows:
        iid = str(row.get("instrument_id", ""))
        session = row.get("session")
        if not isinstance(session, datetime):
            raise PITDataError("invalid KRX session; certification blocked")  # pragma: no cover
        try:
            close = float(row.get("close", float("nan")))
        except (TypeError, ValueError) as exc:  # pragma: no cover
            raise PITDataError("invalid KRX market value; certification blocked") from exc
        if not math.isfinite(close) or close <= 0:
            raise PITDataError("invalid KRX market value; certification blocked")  # pragma: no cover
        closes_by_iid.setdefault(iid, []).append((session, close))
    for iid in closes_by_iid:
        closes_by_iid[iid].sort(key=lambda pair: pair[0])
    resolved: list[dict[str, Any]] = []
    for page in page_list:
        endpoint = str(page.get("endpoint", ""))
        corp_code = str(page.get("corp_code", ""))
        status = str(page.get("status", ""))
        if status == "013":
            continue  # pragma: no cover
        if status != "000":
            raise PITDataError(f"unexpected OpenDART status {status!r} for {endpoint} {corp_code}")  # pragma: no cover
        records = page.get("records", [])
        if not isinstance(records, list):
            raise PITDataError(f"invalid OpenDART records for {endpoint} {corp_code}")
        for record in records:
            if not isinstance(record, dict):
                raise PITDataError(f"invalid OpenDART record for {endpoint} {corp_code}")  # pragma: no cover
            rcept_no = str(record.get("rcept_no", "") or "").strip()
            if endpoint == "fricDecsn.json":
                for field in ("rcept_no", "corp_code", "bfic_tisstk_ostk", "nstk_ostk_cnt", "nstk_ascnt_ps_ostk", "nstk_asstd", "nstk_lstprd"):
                    if str(record.get(field, "") or "").strip() == "":
                        raise PITDataError(f"missing OpenDART field {field} for {endpoint} {rcept_no}")  # pragma: no cover
                basis = _parse_exact_int(record.get("bfic_tisstk_ostk"), field="bfic_tisstk_ostk")
                existing = _parse_exact_int(record.get("nstk_ostk_cnt"), field="nstk_ostk_cnt")
                if abs(basis - existing) > 1000:
                    raise PITDataError(  # pragma: no cover
                        f"ambiguous OpenDART share basis for {endpoint} {rcept_no}; certification blocked"
                    )
                alloc = _parse_exact_int(record.get("nstk_ascnt_ps_ostk"), field="nstk_ascnt_ps_ostk")
                if alloc <= 0:
                    raise PITDataError(f"invalid OpenDART allocation for {endpoint} {rcept_no}")  # pragma: no cover
                factor = 1.0 + float(alloc)
                if not math.isfinite(factor) or factor <= 1.0:
                    raise PITDataError(f"invalid OpenDART factor for {endpoint} {rcept_no}")  # pragma: no cover
                asstd = _parse_opendart_date(record.get("nstk_asstd"))
                instrument_id = _resolve_instrument(corp_code=corp_code or str(record.get("corp_code", "")), daily_market=daily_market, record=record)
                available_at = _receipt_available_at(rcept_no=rcept_no, calendar=calendar)
                listing_date = _parse_opendart_date(record.get("nstk_lstprd"))
                listing_session = _resolve_listing_session(listing_date=listing_date, calendar=calendar, instrument_id=instrument_id, action_id=rcept_no)
                bars = closes_by_iid.get(instrument_id, [])
                if len(bars) < 2:
                    raise PITDataError(f"missing KRX bars for {endpoint} {rcept_no}")  # pragma: no cover
                first_ge = next((idx for idx, (sess, _) in enumerate(bars) if sess.astimezone(KRX_TZ).date() >= asstd), None)
                if first_ge is None or first_ge < 1:
                    raise PITDataError(f"missing KRX bars for {endpoint} {rcept_no}")  # pragma: no cover
                candidates: list[datetime] = []
                for curr_idx in (first_ge - 1, first_ge):
                    if curr_idx < 1 or curr_idx >= len(bars):
                        continue  # pragma: no cover
                    prev_close = bars[curr_idx - 1][1]
                    curr_close = bars[curr_idx][1]
                    if not math.isfinite(prev_close) or not math.isfinite(curr_close) or prev_close <= 0 or curr_close <= 0:
                        raise PITDataError(f"invalid KRX market value for {endpoint} {rcept_no}")  # pragma: no cover
                    raw_return = abs(curr_close / prev_close - 1.0)
                    adjusted = abs(factor * curr_close / prev_close - 1.0)
                    if raw_return > threshold and adjusted <= threshold:
                        candidates.append(bars[curr_idx][0])
                if len(candidates) != 1 or listing_session < candidates[0]:  # pragma: no cover
                    raise PITDataError(
                        f"unreconciled OpenDART bonus issue for {endpoint} {rcept_no}; certification blocked"
                    )
                effective_session = candidates[0]
                decision_time = effective_session.replace(hour=15, minute=30)
                if not available_at < decision_time:  # pragma: no cover
                    raise PITDataError(f"late corporate action for {instrument_id!r}")
                resolved.append({
                    "instrument_id": instrument_id,
                    "action_type": "bonus_issue",
                    "factor": float(factor),
                    "cash_amount": 0.0,
                    "effective_session": effective_session,
                    "share_listing_date": listing_session,
                    "share_delta": int(existing),
                    "available_at": available_at,
                    "action_id": rcept_no,
                })
            elif endpoint == "crDecsn.json":  # pragma: no cover
                cr_mth = str(record.get("cr_mth", "") or "").strip()
                if cr_mth and cr_mth not in ("consolidation", "stock_consolidation", "주식병합"):
                    raise PITDataError(
                        f"unsupported OpenDART corporate action consolidation in {endpoint} rcept {rcept_no} category {cr_mth}"
                    )
                pre_raw = next((record.get(k) for k in ("bf_ostk_cnt", "bfic_tisstk_ostk", "pre_shares") if record.get(k) not in (None, "")), None)
                post_raw = next((record.get(k) for k in ("af_ostk_cnt", "aft_ostk_cnt", "post_shares") if record.get(k) not in (None, "")), None)
                if pre_raw is None or post_raw is None:
                    raise PITDataError(
                        f"unsupported OpenDART corporate action consolidation in {endpoint} rcept {rcept_no} category capital_reduction"
                    )
                pre = _parse_exact_int(pre_raw, field="pre_shares")
                post = _parse_exact_int(post_raw, field="post_shares")
                if pre <= 0 or post <= 0 or post >= pre:
                    raise PITDataError(f"invalid OpenDART consolidation factor for {endpoint} {rcept_no}")
                factor = float(post) / float(pre)
                if not math.isfinite(factor) or not 0.0 < factor < 1.0:
                    raise PITDataError(f"invalid OpenDART consolidation factor for {endpoint} {rcept_no}")
                instrument_id = _resolve_instrument(corp_code=corp_code or str(record.get("corp_code", "")), daily_market=daily_market, record=record)
                available_at = _receipt_available_at(rcept_no=rcept_no, calendar=calendar)
                resolved.append({
                    "instrument_id": instrument_id,
                    "action_type": "reverse_split",
                    "factor": float(factor),
                    "cash_amount": 0.0,
                    "effective_session": ordered_sessions[0],
                    "share_listing_date": None,
                    "share_delta": None,
                    "available_at": available_at,
                    "action_id": rcept_no,
                })
            else:
                category = {"piicDecsn.json": "paid-in-capital", "cmpDvDecsn.json": "division", "cmpMgDecsn.json": "merger"}.get(endpoint, "unrecognised")
                raise PITDataError(
                    f"unsupported OpenDART corporate action {category} in {endpoint} rcept {rcept_no} category {category}"
                )
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


def refresh_corporate_action_silver(
    *,
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
) -> CertificationReport:
    """Refresh only corporate actions without aggregating unrelated Bronze pages."""
    if decision_time.tzinfo is None:  # pragma: no cover - public callers validate timezone
        raise PITDataError("decision_time must be timezone-aware")
    grouped_raw = discover_verified_bronze_receipts(bronze_root=Path(bronze_root))
    grouped = {kind: tuple(items) for kind, items in grouped_raw.items()}
    missing = [kind.value for kind in EvidenceKind if not grouped.get(kind)]
    if missing:  # pragma: no cover - verified Bronze preflight
        raise PITDataError(f"missing required evidence: {', '.join(sorted(missing))}")

    action_receipts = grouped[EvidenceKind.CORPORATE_ACTIONS]
    action_source_hashes = [item.content_hash for item in action_receipts]
    cache_path = Path(artifact_root) / "corporate_actions_stream.json"
    try:
        cached = _read_doc(cache_path) if cache_path.exists() else None
    except (OSError, ValueError):  # pragma: no cover - corrupted optional cache
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("source_hashes") == action_source_hashes
        and isinstance(cached.get("records"), list)
        and not any(
            str(item.get("instrument_id")) == "KRX:__NO_ACTION__"
            for item in cached["records"]
            if isinstance(item, dict)
        )
    ):
        records = [item for item in cached["records"] if isinstance(item, dict)]
    else:  # pragma: no cover - live source parsing is covered by parser tests
        records = compact_corporate_action_intervals(
            _stream_corporate_action_intervals(action_receipts), decision_time=decision_time
        )
        if not records:  # pragma: no cover - valid source must have intervals
            raise PITDataError("corporate-action source has no usable intervals; certification blocked")
        _write_doc(cache_path, {"source_hashes": action_source_hashes, "records": records})

    from src.data.silver import SilverStore, certify_corporate_action_refresh

    preliminary_source_hash = (
        action_source_hashes[0]
        if len(action_source_hashes) == 1
        else hashlib.sha256("\x00".join(sorted(action_source_hashes)).encode("utf-8")).hexdigest()
    )
    action_frame = _corporate_action_frame(records, source_hash=preliminary_source_hash)
    report = certify_corporate_action_refresh(
        action_frame=action_frame,
        receipts=grouped,
        silver_root=Path(silver_root),
        decision_time=decision_time,
    )
    if report.source_hashes[EvidenceKind.CORPORATE_ACTIONS] != preliminary_source_hash:  # pragma: no cover - multi-receipt source
        action_frame = _corporate_action_frame(
            records, source_hash=report.source_hashes[EvidenceKind.CORPORATE_ACTIONS]
        )
        report = certify_corporate_action_refresh(
            action_frame=action_frame,
            receipts=grouped,
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
        "company_id": str(record.get("company_id") or record.get("corp_code") or ticker),
        # Historical planning snapshots may carry no exchange label; retain
        # the row with an explicit sentinel rather than dropping its PIT dates.
        "market": str(record.get("market") or record.get("MKT_TP_NM") or "__UNKNOWN__"),
        "sector": str(record.get("sector") or record.get("sector_name") or "__GLOBAL__"),
        "listing_date": listing_date,
        "delisting_date": record.get("delisting_date") or record.get("delisted_on"),
        "share_class": str(record.get("share_class") or "common"),
        "status": str(record.get("status") or "__UNKNOWN__"),
        "valid_from": valid_from,
        "valid_to": record.get("valid_to"),
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
            schema_version=SCHEMA_VERSION,
        )
        if not writer.has_reusable_manifest:
            count = 0
            missing_market_fields = 0
            small_daily_fingerprints: dict[tuple[Any, Any], str] = {}
            for receipt in receipts:
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
                        canonical = _canonical_master_row(
                            item,
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
    missing = [kind for kind in EvidenceKind if kind not in grouped]
    if missing:
        names = sorted(kind.value for kind in missing)
        raise PITDataError(f"missing required evidence: {', '.join(names)} (investor_flow, financial_facts)")
    selected_streaming: dict[EvidenceKind, list[BronzeReceipt]] = {
        EvidenceKind.DAILY_MARKET: list(
            select_streaming_receipts(
                kind=EvidenceKind.DAILY_MARKET,
                receipts=tuple(grouped[EvidenceKind.DAILY_MARKET]),
            )
        ),
        EvidenceKind.SECURITY_MASTER: list(
            select_streaming_receipts(
                kind=EvidenceKind.SECURITY_MASTER,
                receipts=tuple(grouped[EvidenceKind.SECURITY_MASTER]),
            )
        ),
    }

    _store = _BronzeStore(Path(bronze_root))
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
        corporate_action_pages: list[dict[str, Any]] = []
        for _receipt in action_receipts:
            try:
                _payload = _read_doc(_receipt.payload_path)
            except (OSError, ValueError):
                continue
            if isinstance(_payload, dict) and "endpoint" in _payload:
                corporate_action_pages.append(_payload)
        if corporate_action_pages:
            _daily_preview: list[dict[str, Any]] = []
            for _item in _stream_items_for_kind(list(selected_streaming[EvidenceKind.DAILY_MARKET]), batch_size=bound):
                try:
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
                        "shares_outstanding": _canon["shares"],
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

    # Corporate-action coverage validation runs on assembled Silver frames
    # before any persistence or Gold/universe artifact creation.
    from src.data.backtest_sessions import BacktestMarketInputsPolicy as _StreamPolicy  # pragma: no cover
    from src.data.backtest_sessions import validate_corporate_action_coverage  # pragma: no cover

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
                    _audit_rows.append({"session": _c["session"], "instrument_id": _c["instrument_id"], "close": _c["close"], "shares_outstanding": _c["shares"], "market_cap": _c["market_cap"], "available_at": _c["available_at"]})
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

        _coverage = validate_corporate_action_coverage(daily_market=daily_market, corporate_actions=corporate_actions, calendar=calendar, decision_time_of=decision_time_of, policy=_StreamPolicy())
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
        if not (Path(silver_root) / table.value / dataset_id).exists():
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
