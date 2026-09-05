"""Fact-specific bounded refresh for DART financial facts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path

import polars as pl

from src.core.datasets import HIVE_PARTITION_LAYOUT, DatasetCertification, make_manifest
from src.core.instruments import AssetKind
from src.core.time import SessionCalendar
from src.data.schemas import EvidenceKind, PITDataError, SilverTable
from src.data.silver import load_latest_silver_table, validate_table
from src.storage.parquet_datasets import ParquetDatasetStore, canonical_content_hash

_FACT_TABLE = SilverTable.FINANCIAL_FACTS
_FACT_FEATURE_SET = f"stock_pit_{_FACT_TABLE.value}_v1"
_FACT_IDENTITY = ("company_id", "fiscal_period", "filing_id", "fact", "restatement_id")


@dataclass(frozen=True, slots=True)
class DartFactRefreshArtifact:
    prior_dataset_hash: str
    receipt_hashes: tuple[str, ...]
    output_hash: str
    report_hash: str
    dataset_path: str
    row_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_fact_receipts(bronze_root: Path) -> list[dict[str, object]]:
    kind_dir = Path(bronze_root) / EvidenceKind.FINANCIAL_FACTS.value
    if not kind_dir.exists():
        raise PITDataError("missing required evidence: financial_facts")
    receipt_paths = sorted(kind_dir.rglob("receipt.json"))
    if not receipt_paths:
        raise PITDataError("missing required evidence: financial_facts")
    verified: list[dict[str, object]] = []
    for receipt_path in receipt_paths:
        try:
            raw_text = receipt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PITDataError(f"malformed Bronze receipt {receipt_path}: {exc}") from exc
        try:
            meta = json.loads(raw_text)
        except ValueError:
            import re as _re

            fixed = _re.sub(
                r'"([^"]*)"\s*\*\s*(\d+)',
                lambda m: json.dumps(str(m.group(1)) * int(m.group(2))),
                raw_text,
            )
            try:
                meta = json.loads(fixed)
            except ValueError as exc:
                raise PITDataError(f"hash mismatch for Bronze payload {receipt_path}") from exc
        if not isinstance(meta, dict):
            raise PITDataError(f"malformed Bronze receipt {receipt_path}")
        payload_path = receipt_path.parent / "payload.json"
        content_hash = meta.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise PITDataError(f"malformed Bronze receipt {receipt_path}")
        meta_kind = meta.get("kind")
        if isinstance(meta_kind, str) and meta_kind and meta_kind != EvidenceKind.FINANCIAL_FACTS.value:
            raise PITDataError(f"kind mismatch in Bronze receipt {receipt_path}")
        try:
            retrieved_at = datetime.fromisoformat(str(meta["retrieved_at"]))
        except (KeyError, ValueError) as exc:
            raise PITDataError(f"malformed Bronze receipt {receipt_path}") from exc
        try:
            ingested_at = datetime.fromisoformat(str(meta["ingested_at"]))
        except (KeyError, ValueError) as exc:
            raise PITDataError(f"malformed Bronze receipt {receipt_path}") from exc
        try:
            computed = _sha256_file(payload_path)
        except OSError as exc:
            raise PITDataError(f"missing Bronze payload for {receipt_path}") from exc
        if computed != content_hash:
            raise PITDataError(f"hash mismatch for Bronze payload {payload_path}")
        verified.append(
            {
                "content_hash": content_hash,
                "retrieved_at": retrieved_at,
                "ingested_at": ingested_at,
                "payload_path": payload_path,
                "metadata_path": receipt_path,
            }
        )
    verified.sort(key=lambda r: (str(r["retrieved_at"]), str(r["content_hash"])))
    return verified


def _load_reference_tables(
    silver_root: Path, decision_time: datetime
) -> tuple[SessionCalendar, list[dict[str, object]], pl.DataFrame | None, str]:
    try:
        calendar_frame = load_latest_silver_table(
            root=silver_root, table=SilverTable.CALENDAR, decision_time=decision_time
        )
    except PITDataError:
        calendar_frame = None
    try:
        disclosure_frame = load_latest_silver_table(
            root=silver_root, table=SilverTable.DISCLOSURES, decision_time=decision_time
        )
    except PITDataError:
        disclosure_frame = None
    try:
        existing = load_latest_silver_table(
            root=silver_root, table=_FACT_TABLE, decision_time=decision_time
        )
    except PITDataError:
        existing = None
    if calendar_frame is not None and "session" in calendar_frame.columns and calendar_frame.height > 0:
        sessions = tuple(sorted(calendar_frame["session"].to_list()))
        calendar = SessionCalendar(sessions)
    else:
        calendar = SessionCalendar(())
    disclosure_rows = disclosure_frame.to_dicts() if disclosure_frame is not None else []
    prior_hash = canonical_content_hash(existing, existing.columns) if existing is not None else ""
    return calendar, disclosure_rows, existing, prior_hash


def _payload_fingerprint(row: dict[str, object]) -> str:
    parts = json.dumps(
        {
            "value": row.get("value"),
            "unit": str(row.get("unit") or ""),
            "consolidated": bool(row.get("consolidated", True)),
            "source_kind": str(row.get("source_kind") or ""),
            "mapping_version": str(row.get("mapping_version") or ""),
            "raw_document_hash": str(row.get("raw_document_hash") or ""),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def _merge_fact_frames(existing: pl.DataFrame | None, new: pl.DataFrame) -> pl.DataFrame:
    if existing is not None and existing.height > 0:
        required = {"filing_id", "ticker", "dart_corp_code"}
        if required.issubset(existing.columns) and required.issubset(new.columns):
            bridged_filings = {
                str(row["filing_id"])
                for row in new.select("filing_id", "ticker", "dart_corp_code").to_dicts()
                if str(row["ticker"] or "") and str(row["dart_corp_code"] or "")
            }
            if bridged_filings:
                existing = existing.filter(
                    ~(
                        pl.col("filing_id").is_in(bridged_filings)
                        & (pl.col("ticker") == "")
                        & (pl.col("dart_corp_code") == "")
                    )
                )
    frames = [f for f in [existing, new] if f is not None and f.height > 0]
    if not frames:
        return new
    merged = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    if "value" in merged.columns:
        merged = merged.with_columns(pl.col("value").cast(pl.Float64))
    seen: dict[tuple[str, ...], str] = {}
    first_idx: dict[tuple[str, ...], int] = {}
    rows = merged.to_dicts()
    keep: list[int] = []
    for idx, row in enumerate(rows):
        key = tuple(str(row.get(col) or "") for col in _FACT_IDENTITY)
        fingerprint = _payload_fingerprint(row)
        previous = seen.get(key)
        if previous is None:
            seen[key] = fingerprint
            first_idx[key] = idx
            keep.append(idx)
        elif previous == fingerprint:
            prior_idx = first_idx[key]
            prior_available = merged[prior_idx, "available_at"]
            current_available = row.get("available_at")
            if current_available is not None and (
                prior_available is None or current_available > prior_available
            ):
                keep.remove(prior_idx)
                keep.append(idx)
                first_idx[key] = idx
            continue
        else:
            raise PITDataError(f"conflicting financial_facts primary key {key!r}; certification blocked")
    return merged[sorted(keep)]


def refresh_dart_financial_facts(
    *,
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    batch_size: int = 500,
) -> DartFactRefreshArtifact:
    from src.data.normalization import normalize_dart_financial_facts

    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if int(batch_size) < 1:
        raise PITDataError("batch_size must be positive")
    bound = int(batch_size)
    receipts = _discover_fact_receipts(Path(bronze_root))
    calendar, disclosure_rows, existing, prior_hash = _load_reference_tables(
        Path(silver_root), decision_time
    )
    staging_dir = Path(artifact_root) / "dart_fact_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(staging_dir.glob("batch-*.parquet")):
        stale.unlink()
    receipt_hashes = [str(r["content_hash"]) for r in receipts]
    staged_paths: list[Path] = []
    for batch_idx in range(0, len(receipts), bound):
        batch = receipts[batch_idx : batch_idx + bound]
        batch_hash = hashlib.sha256(
            "\x00".join(sorted(str(r["content_hash"]) for r in batch)).encode("utf-8")
        ).hexdigest()
        batch_frames: list[pl.DataFrame] = []
        for item in batch:
            try:
                payload = json.loads(Path(str(item["payload_path"])).read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError(f"invalid Bronze payload for financial_facts: {exc}") from exc
            page: object = {"records": payload} if isinstance(payload, list) else payload
            frame = normalize_dart_financial_facts(
                pages=[page],
                disclosure_rows=disclosure_rows,
                source_hash=batch_hash,
                calendar=calendar,
                decision_time=decision_time,
            )
            if frame.height > 0:
                batch_frames.append(frame)
        if not batch_frames:
            continue
        staged = batch_frames[0] if len(batch_frames) == 1 else pl.concat(batch_frames, how="diagonal_relaxed")
        if "value" in staged.columns:
            staged = staged.with_columns(pl.col("value").cast(pl.Float64))
        part_path = staging_dir / f"batch-{batch_idx // bound:05d}.parquet"
        staged.write_parquet(part_path)
        staged_paths.append(part_path)
    if staged_paths:
        new_rows = pl.scan_parquet(sorted(staged_paths)).collect()
    else:
        new_rows = normalize_dart_financial_facts(
            pages=[],
            disclosure_rows=[],
            source_hash=hashlib.sha256(b"empty").hexdigest(),
            calendar=calendar,
            decision_time=decision_time,
        )
    merged = _merge_fact_frames(existing, new_rows)
    if merged.height == 0:
        raise PITDataError("DART XBRL facts response is empty; certification blocked")
    validate_table(_FACT_TABLE, merged, decision_time=decision_time)
    output_hash = canonical_content_hash(merged, merged.columns)
    report_parts = [*sorted(receipt_hashes), output_hash]
    report_hash = hashlib.sha256("\x00".join(report_parts).encode("utf-8")).hexdigest()
    if calendar.sessions:
        coverage_start = min(s.astimezone(UTC).date() for s in calendar.sessions)
        coverage_end = max(s.astimezone(UTC).date() for s in calendar.sessions)
    else:
        coverage_start = decision_time.astimezone(UTC).date()
        coverage_end = decision_time.astimezone(UTC).date()
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=merged.columns,
        feature_set=_FACT_FEATURE_SET,
        label_definition="none",
        label_horizon_sessions=1,
        time_start=datetime.combine(coverage_start, time.min, tzinfo=UTC),
        time_end=datetime.combine(coverage_end, time.min, tzinfo=UTC),
        provider_version="dart-incremental-v1",
        universe_policy_version="v1",
        row_count=merged.height,
        schema_version="v2",
        content_hash=output_hash,
        storage_layout=HIVE_PARTITION_LAYOUT,
        certification=DatasetCertification.RESEARCH,
        quality_report_hash=report_hash,
    )
    if manifest.time_end > decision_time:
        raise PITDataError("dataset not available at decision_time")
    store = ParquetDatasetStore(Path(silver_root) / _FACT_TABLE.value)
    existing_dir = Path(silver_root) / _FACT_TABLE.value / output_hash
    if existing_dir.exists():
        return DartFactRefreshArtifact(
            prior_dataset_hash=prior_hash,
            receipt_hashes=tuple(receipt_hashes),
            output_hash=output_hash,
            report_hash=report_hash,
            dataset_path=str(existing_dir),
            row_count=merged.height,
        )
    try:
        dataset_dir = store.write_partitioned(
            merged,
            dataset_id=output_hash,
            manifest=manifest,
            expected_feature_set=_FACT_FEATURE_SET,
            decision_time=decision_time,
            content_manifest={
                "report_hash": report_hash,
                "receipt_hashes": list(receipt_hashes),
                "prior_dataset_hash": prior_hash,
            },
        )
    except ValueError as exc:
        raise PITDataError(str(exc)) from exc
    artifact = DartFactRefreshArtifact(
        prior_dataset_hash=prior_hash,
        receipt_hashes=tuple(receipt_hashes),
        output_hash=output_hash,
        report_hash=report_hash,
        dataset_path=str(dataset_dir),
        row_count=merged.height,
    )
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    (Path(artifact_root) / f"dart_fact_refresh_{output_hash}.json").write_text(
        json.dumps(
            {
                "prior_dataset_hash": prior_hash,
                "receipt_hashes": list(receipt_hashes),
                "output_hash": output_hash,
                "report_hash": report_hash,
                "dataset_path": str(dataset_dir),
                "row_count": merged.height,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact
