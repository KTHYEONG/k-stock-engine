def test_core_pit_contracts_preserve_values() -> None:
    from datetime import UTC, date, datetime
    from pathlib import Path

    from src.core.datasets import DatasetCertification
    from src.core.pit import BronzeReceipt, CertificationReport, EvidenceKind, PITDataError, PITSnapshotRequest, SilverTable

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
    request = PITSnapshotRequest(
        decision_time=moment,
        required_tables=frozenset({SilverTable.DAILY_MARKET, SilverTable.SECURITY_MASTER}),
    )
    report = CertificationReport(
        certification=DatasetCertification.RESEARCH,
        report_hash="b" * 64,
        coverage_start=date(2016, 1, 4),
        coverage_end=date(2026, 9, 9),
        source_hashes={EvidenceKind.DAILY_MARKET: receipt.content_hash},
    )

    assert EvidenceKind.DAILY_MARKET.value == "daily_market"
    assert SilverTable.DAILY_MARKET.value == "daily_market"
    assert receipt.kind is EvidenceKind.DAILY_MARKET
    assert request.required_tables == frozenset({SilverTable.DAILY_MARKET, SilverTable.SECURITY_MASTER})
    assert report.certification is DatasetCertification.RESEARCH
    assert issubclass(PITDataError, ValueError)

    try:
        receipt.content_hash = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("BronzeReceipt must remain frozen")
