"""Certified current-state industry-classification Silver snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.data.bronze_aggregation import discover_verified_bronze_receipts
from src.data.schemas import EvidenceKind, PITDataError

POLICY_VERSION = "kis-industry-classification-v1"

_SCHEMA: dict[str, Any] = {
    "ticker": pl.String,
    "instrument_id": pl.String,
    "industry_name": pl.String,
    "market_name": pl.String,
    "available_at": pl.Datetime("us", "UTC"),
    "source_hash": pl.String,
    "policy_version": pl.String,
}


@dataclass(frozen=True, slots=True)
class IndustryClassificationResult:
    dataset_path: Path
    dataset_id: str
    rows: int


def _parse_receipt(payload_path: Path) -> tuple[str, datetime, str, str]:
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    try:
        collected_at = datetime.fromisoformat(str(payload.get("collected_at")))
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}") from exc
    if collected_at.tzinfo is None:
        collected_at = collected_at.replace(tzinfo=UTC)
    records = payload.get("records")
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    industry = records[0].get("industry_name")
    if not isinstance(industry, str) or not industry.strip():
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    return symbol, collected_at.astimezone(UTC), industry, str(records[0].get("market_name") or "")


def materialize_industry_classification_silver(
    *, bronze_root: Path, silver_root: Path, symbols: frozenset[str] | None = None
) -> IndustryClassificationResult:
    """Build a certified ``industry_<hash16>`` Silver dataset from KIS Bronze evidence.

    This holds one row per ticker with its most-recently-collected
    classification, honestly stamped with the ``available_at`` of that
    specific collection (not backdated). The collector behind this evidence
    only observes the *current* classification, so this table is a
    current-state snapshot — it must never be read as history for sessions
    before a ticker's first observed ``available_at``.

    Args:
        bronze_root: Scope Bronze root holding certified ``INDUSTRY`` receipts.
        silver_root: Scope Silver root receiving ``industry_<hash16>/``.
        symbols: Optionally restricts which tickers' Bronze evidence is
            considered (for incremental/targeted rebuilds); ``None`` means
            "use every certified ``INDUSTRY`` receipt found under
            ``bronze_root``."

    Returns:
        The snapshot result; ``rows`` equals the number of tickers covered.

    Raises:
        PITDataError: hash-verification failure of any considered Bronze
            receipt, malformed certified payload, no considered evidence, or
            an existing dataset with different content. ``symbols`` provided
            but empty is also rejected.
    """
    if symbols is not None and not symbols:
        raise PITDataError("industry classification requires a non-empty symbol filter")
    grouped = discover_verified_bronze_receipts(
        bronze_root=Path(bronze_root), kinds=frozenset({EvidenceKind.INDUSTRY})
    )
    receipts = grouped.get(EvidenceKind.INDUSTRY, ())
    parsed = [(receipt, _parse_receipt(receipt.payload_path)) for receipt in receipts]
    if symbols is not None:
        selected = set(symbols)
        filtered = [(receipt, row) for receipt, row in parsed if row[0] in selected]
        if not filtered:
            if parsed:
                raise PITDataError("no certified INDUSTRY Bronze evidence matches the symbol filter")
            raise PITDataError("no certified INDUSTRY Bronze evidence found")
        parsed = filtered
    if not parsed:
        raise PITDataError("no certified INDUSTRY Bronze evidence found")
    considered = sorted(receipt.content_hash for receipt, _ in parsed)
    best: dict[str, tuple[datetime, str, str, str, str]] = {}
    for receipt, (symbol, collected_at, industry, market) in parsed:
        known = best.get(symbol)
        if known is None or (collected_at, receipt.content_hash) > (known[0], known[1]):
            best[symbol] = (collected_at, receipt.content_hash, industry, market, receipt.content_hash)
    frame = (
        pl.DataFrame(
            {
                "ticker": sorted(best),
                "instrument_id": [f"KRX:{ticker}" for ticker in sorted(best)],
                "industry_name": [best[ticker][2] for ticker in sorted(best)],
                "market_name": [best[ticker][3] for ticker in sorted(best)],
                "available_at": [best[ticker][0] for ticker in sorted(best)],
                "source_hash": [best[ticker][4] for ticker in sorted(best)],
                "policy_version": [POLICY_VERSION for _ in best],
            },
            schema=_SCHEMA,
        ).sort("ticker")
    )
    silver_root = Path(silver_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".industry-silver-", dir=silver_root))
    try:
        rel = Path("part.parquet")
        out_path = staging / rel
        frame.write_parquet(out_path)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        dataset_id = "industry_" + hashlib.sha256(
            "\n".join((POLICY_VERSION, *considered)).encode("utf-8")
        ).hexdigest()[:16]
        target_path = silver_root / dataset_id
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "rows": frame.height,
            "partitions": [{"path": str(rel), "row_count": frame.height, "parquet_sha256": digest}],
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target_path.exists():
            try:
                current = (target_path / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(
                    f"existing industry dataset is unreadable: {target_path}"
                ) from exc
            if current != encoded:
                raise PITDataError(f"existing industry dataset differs: {target_path}")
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return IndustryClassificationResult(
        dataset_path=target_path, dataset_id=dataset_id, rows=frame.height
    )
