"""Streaming PIT normalization with bounded batches and checkpoints."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ
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


def _canonical_daily_row(
    record: dict[str, Any],
    *,
    available_at: datetime,
    source_hash: str,
    shares_override: float | None = None,
) -> dict[str, Any]:
    """Map one provider record without retaining its source batch."""
    session = _as_krx_datetime(_required_row_value(record, "session", "price_date", "basDd", "BAS_DD"))
    open_price = float(_required_row_value(record, "open", "open_price", "mkp", "TDD_OPNPRC"))
    close_price = float(_required_row_value(record, "close", "close_price", "clpr", "TDD_CLSPRC"))
    high = max(float(_required_row_value(record, "high", "high_price", "hipr", "TDD_HGPRC")), open_price, close_price)
    low = min(float(_required_row_value(record, "low", "low_price", "lopr", "TDD_LWPRC")), open_price, close_price)
    volume = float(_required_row_value(record, "volume", "trdvol", "ACC_TRDVOL"))
    trading_value = float(_required_row_value(record, "trading_value", "trdval", "ACC_TRDVAL"))
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
    if raw_market_cap is None and raw_shares is None:
        raise PITDataError("missing KRX field market_cap/marcap/MKTCAP; certification blocked")
    shares = float(raw_shares) if raw_shares is not None else 0.0
    market_cap = float(raw_market_cap) if raw_market_cap is not None else close_price * shares
    if raw_shares is None and close_price > 0:
        shares = market_cap / close_price
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
    # A historical master snapshot becomes effective when it is observed.
    # Keep listing_date as descriptive metadata; using it as valid_from would
    # collapse distinct PIT snapshots onto one primary key.
    valid_from = available_at
    for key in ("valid_from",):
        if record.get(key) not in (None, ""):
            valid_from = _as_krx_datetime(record.get(key))
            break
    return {
        "instrument_id": instrument_id,
        "ticker": ticker,
        "company_id": str(record.get("company_id") or record.get("corp_code") or ticker),
        # Historical planning snapshots may carry no exchange label; retain
        # the row with an explicit sentinel rather than dropping its PIT dates.
        "market": str(record.get("market") or record.get("MKT_TP_NM") or "__UNKNOWN__"),
        "sector": str(record.get("sector") or record.get("sector_name") or "__GLOBAL__"),
        "listing_date": valid_from,
        "delisting_date": record.get("delisting_date") or record.get("delisted_on"),
        "share_class": str(record.get("share_class") or "common"),
        "status": str(record.get("status") or "listed"),
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
    action_source_hashes = [item.content_hash for item in grouped[EvidenceKind.CORPORATE_ACTIONS]]
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
    ):
        streamed_actions = [item for item in cached_actions["records"] if isinstance(item, dict)]
    else:
        for receipt in grouped[EvidenceKind.CORPORATE_ACTIONS]:
            for item in _stream_items_for_kind([receipt], batch_size=bound):
                action_code = str(item.get("type") or item.get("action_type") or item.get("action_code") or "no_action")
                if action_code != "no_action":
                    streamed_actions.append(item)
        if not streamed_actions:
            streamed_actions.append({"type": "no_action", "effective_date": decision_time, "instrument_id": "KRX:__NO_ACTION__"})
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

    writers: dict[SilverTable, StreamingSilverWriter] = {}
    for table in _STREAM_TABLES:
        kind = _STREAM_KINDS[table]
        writers[table] = StreamingSilverWriter(
            staging_root,
            table=table,
            batch_size=bound,
            source_hashes=tuple(r.content_hash for r in selected_streaming[kind]),
            schema_version=SCHEMA_VERSION,
        )

    # One record at a time for large payloads; never retain the source batch.
    # Process master first so legacy daily rows can use its share count when
    # the provider omitted LIST_SHRS and MKTCAP from the historical page.
    streamed_counts: dict[SilverTable, int] = dict.fromkeys(_STREAM_TABLES, 0)
    master_shares: dict[str, float] = {}
    for table in (SilverTable.SECURITY_MASTER, SilverTable.DAILY_MARKET):
        kind = _STREAM_KINDS[table]
        writer = writers[table]
        if writer.has_reusable_manifest:
            streamed_counts[table] = int(sum(writer._row_counts.values()))
            continue
        count = 0
        for receipt in selected_streaming[kind]:
            avail = receipt.retrieved_at
            if avail.tzinfo is None:
                avail = avail.replace(tzinfo=KRX_TZ)
            source_hash = receipt.content_hash
            for item in _stream_items_for_kind([receipt], batch_size=bound):
                if not isinstance(item, dict):
                    raise PITDataError("malformed record; certification blocked")
                if table is SilverTable.DAILY_MARKET:
                    ticker = str(
                        _required_row_value(
                            item,
                            "instrument_id",
                            "ticker",
                            "isu_cd",
                            "ISU_SRT_CD",
                            "ISU_CD",
                        )
                    ).strip()
                    canonical = _canonical_daily_row(
                        item,
                        available_at=avail,
                        source_hash=source_hash,
                        shares_override=master_shares.get(ticker.removeprefix("KRX:")),
                    )
                    writer.append(month=_month_of(canonical["session"]), row=canonical)
                else:
                    avail = _master_available_at(receipt=receipt, record=item)
                    if avail > decision_time:
                        continue
                    ticker = str(
                        _required_row_value(
                            item, "ticker", "isu_cd", "ISU_SRT_CD", "source_identifier"
                        )
                    ).strip()
                    raw_shares = next(
                        (
                            item.get(name)
                            for name in ("shares_outstanding", "list_shrs", "LIST_SHRS")
                            if item.get(name) not in (None, "")
                        ),
                        None,
                    )
                    if raw_shares is not None:
                        try:
                            parsed_shares = float(raw_shares)
                        except (TypeError, ValueError):
                            parsed_shares = 0.0
                        if parsed_shares > 0:
                            master_shares[ticker.removeprefix("KRX:")] = parsed_shares
                    canonical = _canonical_master_row(
                        item, available_at=avail, source_hash=source_hash, fallback_session=avail
                    )
                    writer.append(month=_month_of(canonical["valid_from"]), row=canonical)
                count += 1
        if count == 0:
            raise PITDataError(f"incomplete month for {table.value}; certification blocked")
        # Writer counts accepted canonical rows (duplicates are intentionally
        # dropped), not raw provider records encountered in the stream.
        streamed_counts[table] = int(sum(writer._row_counts.values()))

    manifests: dict[SilverTable, dict[str, Any]] = {}
    for table in _STREAM_TABLES:
        manifests[table] = writers[table].close()

    from src.data.normalization import normalize_stock_evidence

    tables, report = normalize_stock_evidence(
        dict(single),
        decision_time=decision_time,
        streamed_tables=frozenset(_STREAM_TABLES) | {SilverTable.CORPORATE_ACTIONS},
        streamed_corporate_actions=streamed_actions,
    )

    # Corporate-action gate: unadjusted prices are forbidden unless every
    # affected instrument/date is excluded from the eligible universe.
    actions = tables.get(SilverTable.CORPORATE_ACTIONS)
    if actions is not None and actions.height > 0:
        eligible = set(tables[SilverTable.SECURITY_MASTER]["instrument_id"].to_list())
        for row in actions.to_dicts():
            if str(row.get("type")) not in ("no_action", "", None) and str(row.get("instrument_id")) in eligible:
                raise PITDataError("unadjusted prices forbidden; corporate action affects universe")

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
