def test_snapshot_returns_only_rows_available_at_decision_time(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    import polars as pl

    from src.data.schemas import PITSnapshotRequest, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    repository = PITSnapshotRepository.from_frames({SilverTable.DISCLOSURES: pl.DataFrame({'company_id': ['DART:1', 'DART:2'], 'filing_id': ['a', 'b'], 'available_at': [decision, datetime(2024, 1, 4, tzinfo=UTC)]})}, root=Path(tmp_path))

    result = repository.snapshot(PITSnapshotRequest(decision_time=decision, required_tables=frozenset({SilverTable.DISCLOSURES})))
    assert result[SilverTable.DISCLOSURES]['filing_id'].to_list() == ['a']


def test_snapshot_rejects_missing_required_table(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError, PITSnapshotRequest, SilverTable
    from src.data.snapshot import PITSnapshotRepository

    repository = PITSnapshotRepository.from_frames(
        {SilverTable.DISCLOSURES: pl.DataFrame({"available_at": [datetime(2024, 1, 3, tzinfo=UTC)]})},
        root=Path(tmp_path),
    )
    request = PITSnapshotRequest(
        decision_time=datetime(2024, 1, 3, tzinfo=UTC),
        required_tables=frozenset({SilverTable.DAILY_MARKET}),
    )

    with pytest.raises(PITDataError, match="required Silver table is missing"):
        repository.snapshot(request)


def test_minimal_complete_fixture_materializes_research_certified_silver(tmp_path) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from src.core.datasets import DatasetCertification
    from src.data.silver import SilverStore, complete_minimal_fixture

    decision = datetime(2024, 1, 3, 1, tzinfo=UTC)
    tables, _, report = complete_minimal_fixture(decision_time=decision)
    output = SilverStore(Path(tmp_path) / 'silver').materialize_all(tables, report=report, decision_time=decision)

    assert report.certification is DatasetCertification.RESEARCH
    assert report.report_hash
    assert set(output) == set(tables)
    assert all(path.exists() for path in output.values())
