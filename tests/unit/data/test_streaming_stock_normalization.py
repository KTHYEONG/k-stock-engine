def test_streaming_normalization_resumes_only_verified_months(tmp_path) -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    store = StreamingNormalizationCheckpoint(tmp_path)
    store.mark_verified(table="daily_market", month="2020-01", source_hashes=("a",), output_hash="b")
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("a",)) is True
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("changed",)) is False


def test_streaming_normalization_accepts_krx_daily_aliases_in_bounded_batches() -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _canonical_daily_row

    row = _canonical_daily_row(
        {
            "price_date": "2016-01-04",
            "instrument_id": "KRX:000020",
            "open": 10,
            "high": 12,
            "low": 9,
            "close": 11,
            "volume": 100,
            "trading_value": 1100,
            "market_cap": 10000,
            "shares_outstanding": 900,
        },
        available_at=datetime(2016, 1, 5, tzinfo=UTC),
        source_hash="a",
    )

    assert row["instrument_id"] == "KRX:000020"
    assert row["session"].year == 2016
    assert row["low"] <= row["open"] <= row["high"]


def test_streaming_writer_rewrites_only_changed_month(tmp_path) -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    checkpoint = StreamingNormalizationCheckpoint(tmp_path)
    checkpoint.mark_verified(table="daily_market", month="2016-01", source_hashes=("jan",), output_hash="digest-jan")
    checkpoint.mark_verified(table="daily_market", month="2016-02", source_hashes=("old-feb",), output_hash="digest-feb")

    assert checkpoint.is_verified(table="daily_market", month="2016-01", source_hashes=("jan",))
    assert not checkpoint.is_verified(table="daily_market", month="2016-02", source_hashes=("new-feb",))


def test_streaming_daily_writer_rejects_duplicate_primary_key() -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import _assert_unique_daily_keys

    row = {"session": datetime(2016, 1, 4, tzinfo=UTC), "instrument_id": "KRX:000020"}
    with pytest.raises(PITDataError, match="duplicate daily_market primary key"):
        _assert_unique_daily_keys([row, dict(row)])


def test_streaming_writer_persists_staging_manifest_after_part_flush(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.schemas import SilverTable
    from src.data.streaming_normalization import StreamingSilverWriter

    writer = StreamingSilverWriter(tmp_path / 'staging', table=SilverTable.DAILY_MARKET, batch_size=1, source_hashes=('source',))
    writer.append(month='2020-01', row={'session': datetime(2020, 1, 2, tzinfo=UTC), 'instrument_id': 'KRX:000020', 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0, 'trading_value': 1.0, 'market_cap': 1.0, 'shares_outstanding': 1.0, 'available_at': datetime(2020, 1, 2, tzinfo=UTC), 'source_hash': 'source'})
    manifest = json.loads((tmp_path / 'staging' / 'daily_market' / 'staging_manifest.json').read_text())
    assert manifest['parts']['2020-01'][0]['row_count'] == 1
