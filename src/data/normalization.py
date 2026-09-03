"""PIT normalization from Bronze receipts to certified Silver tables."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

import polars as pl

from src.core.time import SessionCalendar
from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, PITDataError, SilverTable


def normalize_stock_evidence(
    receipts: Mapping[EvidenceKind, BronzeReceipt],
    *,
    calendar: SessionCalendar | None = None,
    decision_time: datetime,
) -> tuple[Mapping[SilverTable, pl.DataFrame], CertificationReport]:
    if decision_time.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    if calendar is not None and not calendar.sessions:
        raise PITDataError("calendar must contain sessions")
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
        try:
            raw_text = receipt.payload_path.read_bytes()
            json.loads(raw_text)
        except (OSError, ValueError) as exc:
            raise PITDataError(f"invalid Bronze payload for {kind.value}: {exc}") from exc
    raise PITDataError(
        "raw Bronze normalization requires provider-specific parsers; refusing to certify fixture data"
    )
