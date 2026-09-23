def test_streaming_normalization_module_is_covered() -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    assert StreamingNormalizationCheckpoint is not None


def test_stream_normalize_stock_evidence_requires_core_evidence(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import stream_normalize_stock_evidence

    with pytest.raises(PITDataError, match="missing required evidence"):
        stream_normalize_stock_evidence(
            bronze_root=tmp_path / "bronze",
            silver_root=tmp_path / "silver",
            artifact_root=tmp_path / "artifacts",
            decision_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
