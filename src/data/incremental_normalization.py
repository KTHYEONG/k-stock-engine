"""Fact-specific bounded refresh for DART financial facts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from src.core.time import SessionCalendar
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    canonical_content_hash,
    dataset_digest,
    dataset_reference,
    load_manifest,
    publish_dataset,
    read_dataset,
)
from src.data.normalization import (
    TRUSTED_FACT_SOURCE_KINDS,
    QuarantinedFiling,
    normalize_dart_financial_facts_with_quarantine,
)
from src.data.schemas import EvidenceKind, PITDataError

AVAILABILITY_POLICY: Final = "next-session-after-effective-receipt-v2"

_FACT_IDENTITY = ("company_id", "fiscal_period", "filing_id", "fact", "restatement_id", "consolidated")


@dataclass(frozen=True, slots=True)
class DartFactRefreshArtifact:
    prior_dataset_hash: str
    receipt_hashes: tuple[str, ...]
    output_hash: str
    report_hash: str
    dataset_path: str
    row_count: int
    quarantine_path: str  # JSON list of quarantined filings, bound to output_hash
    quarantined_filings: int


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
    silver_root: Path,
    decision_time: datetime,
    *,
    disclosures_dataset_id: str | None = None,
    financial_facts_dataset_id: str | None = None,
) -> tuple[list[dict[str, object]], str, str]:
    """Load disclosure and prior-fact inputs from verified v2 datasets."""

    def _flat(kind: str, explicit_id: str | None = None) -> tuple[pl.DataFrame | None, str | None]:
        if explicit_id is not None:
            candidates = [Path(silver_root) / explicit_id]
        else:
            candidates = sorted(
                path for path in Path(silver_root).glob(f"{kind}_*") if path.is_dir() and not path.is_symlink()
            )
            candidates.reverse()
        for candidate in candidates:
            try:
                load_manifest(candidate)
                return read_dataset(candidate).collect(), candidate.name
            except PITDataError:
                continue
        return None, None

    disclosure_frame, disclosure_id = _flat("disclosures", disclosures_dataset_id)
    existing, _existing_id = _flat("financial_facts", financial_facts_dataset_id)
    disclosure_rows = disclosure_frame.to_dicts() if disclosure_frame is not None else []
    prior_hash = canonical_content_hash(existing, existing.columns) if existing is not None else ""
    disclosure_digest = (
        dataset_reference(disclosure_id, kind="disclosures")
        if disclosure_id is not None
        else dataset_digest([])
    )
    _ = decision_time
    return disclosure_rows, prior_hash, disclosure_digest


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


def load_frozen_dart_ticker_bridge(
    *, bronze_root: Path, decision_time: datetime
) -> tuple[dict[str, str], str]:
    _ = decision_time
    bridge_dir = Path(bronze_root) / "dart_corp_codes"
    payloads = sorted(bridge_dir.glob("*/payload.json")) if bridge_dir.exists() else []
    if not payloads:
        raise PITDataError("ticker bridge missing: no retained dart_corp_codes receipt")
    payload_path = payloads[-1]
    receipt_hash = payload_path.parent.name
    if len(receipt_hash) != 64 or any(character not in "0123456789abcdef" for character in receipt_hash):
        raise PITDataError("ticker bridge receipt hash is invalid")
    try:
        raw_bytes = payload_path.read_bytes()
    except OSError as exc:
        raise PITDataError("ticker bridge payload is unreadable") from exc
    if hashlib.sha256(raw_bytes).hexdigest() != receipt_hash:
        raise PITDataError("ticker bridge payload hash mismatch")
    try:
        raw = json.loads(raw_bytes)
    except (TypeError, ValueError) as exc:
        raise PITDataError("ticker bridge payload is invalid") from exc
    if not isinstance(raw, list):
        raise PITDataError("ticker bridge payload must be a list")
    mapping: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise PITDataError("ticker bridge row must be an object")
        corp_code = str(row.get("corp_code") or "").strip()
        ticker = str(row.get("ticker") or "").strip()
        if not corp_code or not ticker:
            raise PITDataError("ticker bridge row lacks corp_code or ticker")
        previous = mapping.get(corp_code)
        if previous is not None and previous != ticker:
            raise PITDataError(f"ticker bridge corp_code maps to multiple tickers: {corp_code}")
        mapping[corp_code] = ticker
    if not mapping:
        raise PITDataError("ticker bridge payload is empty")
    return mapping, receipt_hash


def _quarantine_to_json_record(record: QuarantinedFiling) -> dict[str, str]:
    return {
        "company_id": record.company_id,
        "dart_corp_code": record.dart_corp_code,
        "fiscal_period": record.fiscal_period,
        "filing_id": record.filing_id,
        "source_kind": record.source_kind,
        "published_at": record.published_at.astimezone(UTC).isoformat(),
        "available_at": record.available_at.astimezone(UTC).isoformat(),
    }


def _deduplicate_quarantine(
    records: list[QuarantinedFiling],
) -> list[QuarantinedFiling]:
    best: dict[tuple[str, str, str], QuarantinedFiling] = {}
    for record in records:
        key = (record.company_id, record.fiscal_period, record.filing_id)
        previous = best.get(key)
        if previous is None or (record.available_at, record.published_at, record.source_kind) > (
            previous.available_at,
            previous.published_at,
            previous.source_kind,
        ):
            best[key] = record
    return sorted(
        best.values(),
        key=lambda r: (r.available_at, r.company_id, r.fiscal_period, r.filing_id),
    )


def _write_quarantine_file(
    *, artifact_root: Path, output_hash: str, records: list[QuarantinedFiling]
) -> Path:
    payload = json.dumps(
        [_quarantine_to_json_record(r) for r in records],
        indent=2,
        sort_keys=True,
    )
    target = Path(artifact_root) / f"dart_fact_quarantine_{output_hash}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(target)
    return target


def _validate_fact_frame(frame: pl.DataFrame, *, decision_time: datetime) -> None:
    required = {
        "company_id",
        "fiscal_period",
        "filing_id",
        "fact",
        "published_at",
        "available_at",
        "value",
        "unit",
        "consolidated",
        "restatement_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PITDataError(f"financial facts frame lacks columns: {missing}")
    keys = ["company_id", "fiscal_period", "filing_id", "fact", "restatement_id", "consolidated"]
    if frame.select(keys).null_count().sum_horizontal().item() > 0:
        raise PITDataError("financial facts contain a null primary key")
    if frame.select(keys).is_duplicated().any():
        raise PITDataError("financial facts contain a duplicate primary key")
    if frame.filter(pl.col("available_at") > decision_time).height:
        raise PITDataError("financial facts contain a row after decision_time")


def refresh_dart_financial_facts(
    *,
    bronze_root: Path,
    silver_root: Path,
    artifact_root: Path,
    decision_time: datetime,
    calendar: SessionCalendar,
    batch_size: int = 500,
    superseded_receipt_hashes: frozenset[str] = frozenset(),
    disclosures_dataset_id: str | None = None,
    financial_facts_dataset_id: str | None = None,
) -> DartFactRefreshArtifact:
    """Rebuild the financial-facts Silver table from every verified Bronze fact receipt.

    The table is rebuilt in full rather than merged with a prior dataset: rows
    written under an earlier availability policy would otherwise survive the
    merge with stale ``available_at`` values. The prior dataset hash is still
    reported for lineage.

    Facts from untrusted source kinds are withheld from the published table and recorded,
    per filing, in a quarantine list written beside the refresh report; the count is also
    stored in the content manifest so the exclusion is auditable.

    Args:
        bronze_root: Scope Bronze root containing ``financial_facts/``.
        silver_root: Scope Silver root receiving a flat ``financial_facts_<hash16>/`` dataset.
        artifact_root: Destination for staging parquet and the refresh report.
        decision_time: Availability cutoff; must be timezone-aware.
        calendar: KRX sessions (see ``xkrx_session_calendar``) extending past
            ``decision_time``.
        batch_size: Receipts normalized per staged parquet batch.
        superseded_receipt_hashes: Bronze receipts known to be erroneous
            re-recoveries of a filing that a later receipt corrects. They are
            excluded and listed in the content manifest so the exclusion is
            auditable; an unknown hash is rejected to catch typos.

    Returns:
        Hashes, dataset path and row count of the published table.

    Raises:
        PITDataError: Naive ``decision_time``, non-positive ``batch_size``,
            missing or corrupt Bronze, empty result, conflicting primary keys,
            or calendar coverage failure.
    """
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if int(batch_size) < 1:
        raise PITDataError("batch_size must be positive")
    bound = int(batch_size)
    receipts = _discover_fact_receipts(Path(bronze_root))
    unknown = superseded_receipt_hashes - {str(r["content_hash"]) for r in receipts}
    if unknown:
        raise PITDataError(f"superseded receipts not found in Bronze: {sorted(unknown)}")
    receipts = [r for r in receipts if str(r["content_hash"]) not in superseded_receipt_hashes]
    disclosure_rows, prior_hash, disclosure_digest = _load_reference_tables(
        Path(silver_root),
        decision_time,
        disclosures_dataset_id=disclosures_dataset_id,
        financial_facts_dataset_id=financial_facts_dataset_id,
    )
    bridge_root = Path(bronze_root) / "dart_corp_codes"
    bridge: dict[str, str] | None = None
    bridge_receipt_hash: str | None = None
    if bridge_root.exists():
        bridge, bridge_receipt_hash = load_frozen_dart_ticker_bridge(
            bronze_root=Path(bronze_root), decision_time=decision_time
        )
    staging_dir = Path(artifact_root) / "dart_fact_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(staging_dir.glob("batch-*.parquet")):
        stale.unlink()
    receipt_hashes = [str(r["content_hash"]) for r in receipts]
    staged_paths: list[Path] = []
    quarantined_all: list[QuarantinedFiling] = []
    for batch_idx in range(0, len(receipts), bound):
        batch = receipts[batch_idx : batch_idx + bound]
        batch_frames: list[pl.DataFrame] = []
        for item in batch:
            try:
                payload = json.loads(Path(str(item["payload_path"])).read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError(f"invalid Bronze payload for financial_facts: {exc}") from exc
            page: object = {"records": payload} if isinstance(payload, list) else payload
            frame, batch_quarantine = normalize_dart_financial_facts_with_quarantine(
                pages=[page],
                disclosure_rows=disclosure_rows,
                source_hash=str(item["content_hash"]),
                calendar=calendar,
                decision_time=decision_time,
                ticker_by_corp_code=bridge,
                bridge_receipt_hash=bridge_receipt_hash,
            )
            quarantined_all.extend(batch_quarantine)
            if frame.height > 0:
                batch_frames.append(frame)
        if not batch_frames:
            continue
        staged = batch_frames[0] if len(batch_frames) == 1 else pl.concat(batch_frames, how="diagonal_relaxed")
        if "value" in staged.columns:
            staged = staged.with_columns(pl.col("value").cast(pl.Float64))
        part_path = staging_dir / f"batch-{batch_idx // bound:05d}.parquet"
        try:
            staged.write_parquet(part_path)
        except (OSError, pl.exceptions.PolarsError) as exc:
            raise PITDataError(f"cannot stage financial facts partition: {part_path}") from exc
        staged_paths.append(part_path)
    if staged_paths:
        new_rows = pl.scan_parquet(sorted(staged_paths)).collect()
    else:
        new_rows, empty_quarantine = normalize_dart_financial_facts_with_quarantine(
            pages=[],
            disclosure_rows=[],
            source_hash=hashlib.sha256(b"empty").hexdigest(),
            calendar=calendar,
            decision_time=decision_time,
            ticker_by_corp_code=bridge,
            bridge_receipt_hash=bridge_receipt_hash,
        )
        quarantined_all.extend(empty_quarantine)
    merged = _merge_fact_frames(None, new_rows)
    if merged.height == 0:
        raise PITDataError("DART XBRL facts response is empty; certification blocked")
    _validate_fact_frame(merged, decision_time=decision_time)
    output_hash = canonical_content_hash(merged, merged.columns)
    report_parts = [*sorted(receipt_hashes), output_hash]
    report_hash = hashlib.sha256("\x00".join(report_parts).encode("utf-8")).hexdigest()
    trusted_filings = set(
        zip(
            merged["company_id"].to_list(),
            merged["fiscal_period"].to_list(),
            merged["filing_id"].to_list(),
            strict=True,
        )
    )
    quarantined = [
        record
        for record in _deduplicate_quarantine(quarantined_all)
        if (record.company_id, record.fiscal_period, record.filing_id) not in trusted_filings
    ]
    trusted_kinds = sorted(TRUSTED_FACT_SOURCE_KINDS)
    quarantine_path = _write_quarantine_file(
        artifact_root=Path(artifact_root), output_hash=output_hash, records=quarantined
    )
    calendar_digest = hashlib.sha256(
        "\n".join(session.astimezone(UTC).isoformat() for session in calendar.sessions).encode("utf-8")
    ).hexdigest()
    identity = DatasetIdentity(
        kind="financial_facts",
        layer=DatasetLayer.SILVER,
        policy_version="dart-incremental-v1",
        inputs={
            "bronze_facts": dataset_digest(receipt_hashes),
            "superseded": dataset_digest(sorted(superseded_receipt_hashes)),
            "disclosures": disclosure_digest,
            "ticker_bridge": dataset_digest([bridge_receipt_hash]) if bridge_receipt_hash else dataset_digest([]),
        },
        params={
            "decision_time": decision_time,
            "availability_policy": AVAILABILITY_POLICY,
            "calendar_digest": calendar_digest,
            "ticker_bridge": bridge_receipt_hash,
        },
    )
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions={"part-00000.parquet": merged},
        details={
            "report_hash": report_hash,
            "receipt_hashes": list(receipt_hashes),
            "prior_dataset_hash": prior_hash,
            "availability_policy": AVAILABILITY_POLICY,
            "superseded_receipt_hashes": sorted(superseded_receipt_hashes),
            "quarantined_filings": len(quarantined),
            "trusted_source_kinds": trusted_kinds,
            "content_sha64": output_hash,
        },
    )
    artifact = DartFactRefreshArtifact(
        prior_dataset_hash=prior_hash,
        receipt_hashes=tuple(receipt_hashes),
        output_hash=output_hash,
        report_hash=report_hash,
        dataset_path=str(published.path),
        row_count=merged.height,
        quarantine_path=str(quarantine_path),
        quarantined_filings=len(quarantined),
    )
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    (Path(artifact_root) / f"dart_fact_refresh_{output_hash}.json").write_text(
        json.dumps(
            {
                "prior_dataset_hash": prior_hash,
                "receipt_hashes": list(receipt_hashes),
                "output_hash": output_hash,
                "report_hash": report_hash,
                "dataset_path": str(published.path),
                "row_count": merged.height,
                "quarantine_path": str(quarantine_path),
                "quarantined_filings": len(quarantined),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact
