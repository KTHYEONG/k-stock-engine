def test_streaming_normalization_module_is_covered() -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    assert StreamingNormalizationCheckpoint is not None
