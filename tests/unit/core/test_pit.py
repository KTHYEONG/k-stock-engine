from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.core.pit import BronzeReceipt, EvidenceKind, PITDataError


def test_live_core_pit_contracts_preserve_values() -> None:
    moment = datetime(2026, 9, 9, 15, 30, tzinfo=UTC)
    receipt = BronzeReceipt(
        kind=EvidenceKind.DAILY_MARKET,
        content_hash="a" * 64,
        source_path="provider://daily/2026-09-09",
        retrieved_at=moment,
        ingested_at=moment,
        payload_path=Path("data/bronze/payload.json"),
        metadata_path=Path("data/bronze/receipt.json"),
    )

    assert EvidenceKind.DAILY_MARKET.value == "daily_market"
    assert receipt.kind is EvidenceKind.DAILY_MARKET
    assert issubclass(PITDataError, ValueError)

    try:
        receipt.content_hash = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("BronzeReceipt must remain frozen")
