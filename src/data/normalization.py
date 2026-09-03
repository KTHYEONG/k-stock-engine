"""PIT normalization from Bronze receipts to certified Silver tables."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import polars as pl

from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable


def normalize_stock_evidence(
    receipts: Mapping[EvidenceKind, BronzeReceipt], *, decision_time: datetime
) -> Mapping[SilverTable, pl.DataFrame]:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    missing = [kind for kind in EvidenceKind if kind not in receipts]
    if missing:
        names = sorted(kind.value for kind in missing)
        raise PITDataError(
            f"missing required evidence: {', '.join(names)} (investor_flow, financial_facts)"
        )
    for kind, receipt in receipts.items():
        if not receipt.payload_path.exists() or not receipt.metadata_path.exists():
            raise PITDataError(f"missing Bronze receipt payload for {kind.value}")
        if not receipt.content_hash:
            raise PITDataError(f"empty content hash for {kind.value}")
    raise PITDataError(
        "raw Bronze normalization requires provider-specific parsers; refusing to certify fixture data"
    )
