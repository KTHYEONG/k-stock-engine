def test_streaming_normalization_resumes_only_verified_months(tmp_path) -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    store = StreamingNormalizationCheckpoint(tmp_path)
    store.mark_verified(table="daily_market", month="2020-01", source_hashes=("a",), output_hash="b")
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("a",)) is True
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("changed",)) is False


def test_corporate_action_only_refresh_does_not_aggregate_other_bronze_pages(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime

    import pytest

    import src.data.silver as silver
    import src.data.streaming_normalization as streaming
    from src.core.datasets import DatasetCertification
    from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, SilverTable

    decision = datetime(2026, 9, 5, tzinfo=UTC)
    receipts = {
        kind: (
            BronzeReceipt(
                kind=kind,
                content_hash=kind.value.ljust(64, "0"),
                source_path="fixture",
                retrieved_at=decision,
                ingested_at=decision,
                payload_path=tmp_path / f"{kind.value}.json",
                metadata_path=tmp_path / f"{kind.value}.receipt.json",
            ),
        )
        for kind in EvidenceKind
    }
    action_hash = receipts[EvidenceKind.CORPORATE_ACTIONS][0].content_hash
    cache = tmp_path / "artifacts" / "corporate_actions_stream.json"
    cache.parent.mkdir()
    cache.write_text(
        '{"source_hashes": ["' + action_hash + '"], "records": ['
        '{"instrument_id":"KRX:000020","effective_date":"2016-01-04T09:00:00+09:00",'
        '"coverage_end":"2016-01-04T09:00:00+09:00","action_id":"coverage:one",'
        '"type":"no_action","factor":1.0,"cash_amount":0.0,"source":"KRX",'
        '"available_at":"2016-01-04T09:00:00+09:00"}]}',
        encoding="utf-8",
    )
    report = CertificationReport(
        certification=DatasetCertification.RESEARCH,
        report_hash="report",
        coverage_start=date(2016, 1, 4),
        coverage_end=date(2026, 3, 10),
        source_hashes={kind: items[0].content_hash for kind, items in receipts.items()},
    )

    monkeypatch.setattr(streaming, "discover_verified_bronze_receipts", lambda **_: receipts)
    monkeypatch.setattr(streaming, "_aggregate_small", lambda **_: pytest.fail("must not aggregate"))
    monkeypatch.setattr(silver, "certify_corporate_action_refresh", lambda **_: report)

    class FakeStore:
        def __init__(self, root) -> None:
            self.root = root

        def materialize_all(self, tables, **kwargs):
            assert set(tables) == {SilverTable.CORPORATE_ACTIONS}
            return {SilverTable.CORPORATE_ACTIONS: tmp_path / "silver"}

    monkeypatch.setattr(silver, "SilverStore", FakeStore)

    result = streaming.refresh_corporate_action_silver(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts",
        decision_time=decision,
    )

    assert result is report


def test_corporate_action_refresh_certifies_against_immutable_manifests(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.silver import (
        SilverStore,
        certify_corporate_action_refresh,
        complete_minimal_fixture,
    )
    from src.data.schemas import SilverTable

    decision = datetime(2026, 9, 6, tzinfo=UTC)
    tables, receipts, report = complete_minimal_fixture(decision_time=decision)
    silver_root = tmp_path / "silver"
    SilverStore(silver_root).materialize_all(tables, report=report, decision_time=decision)

    refreshed = certify_corporate_action_refresh(
        action_frame=tables[SilverTable.CORPORATE_ACTIONS],
        receipts={kind: (receipt,) for kind, receipt in receipts.items()},
        silver_root=silver_root,
        decision_time=decision,
    )

    assert refreshed.coverage_start == report.coverage_start
    assert refreshed.source_hashes == report.source_hashes


def test_streaming_normalization_accepts_krx_daily_aliases_in_bounded_batches() -> None:
    from datetime import UTC, datetime

    from src.core.time import KRX_TZ, SessionCalendar
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
            "_calendar": SessionCalendar((datetime(2016, 1, 4, 9, tzinfo=KRX_TZ),)),
        },
        available_at=datetime(2016, 1, 5, tzinfo=UTC),
        source_hash="a",
    )

    assert row["instrument_id"] == "KRX:000020"
    assert row["session"].year == 2016
    assert row["low"] <= row["open"] <= row["high"]


def test_streaming_normalization_parses_comma_formatted_krx_capitalisation() -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _canonical_daily_row

    row = _canonical_daily_row({'BAS_DD': '20160104', 'ISU_SRT_CD': '000020', 'TDD_OPNPRC': '10', 'TDD_HGPRC': '12', 'TDD_LWPRC': '9', 'TDD_CLSPRC': '11', 'ACC_TRDVOL': '100', 'ACC_TRDVAL': '1,100', 'MKTCAP': '10,000', 'LIST_SHRS': '900'}, available_at=datetime(2016, 1, 5, tzinfo=UTC), source_hash='a')

    assert row['market_cap'] == 10_000.0
    assert row['shares_outstanding'] == 900.0


def test_streaming_normalization_carries_close_for_untouched_krx_session() -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _canonical_daily_row

    row = _canonical_daily_row(
        {
            'BAS_DD': '20151120', 'ISU_SRT_CD': '037560',
            'TDD_OPNPRC': '0', 'TDD_HGPRC': '0', 'TDD_LWPRC': '0',
            'TDD_CLSPRC': '10900', 'ACC_TRDVOL': '0', 'ACC_TRDVAL': '0',
            'MKTCAP': '844170828500', 'LIST_SHRS': '77446865',
        },
        available_at=datetime(2015, 11, 20, 15, 30, tzinfo=UTC),
        source_hash='a',
    )

    assert row['open'] == row['high'] == row['low'] == row['close'] == 10900.0


def test_streaming_normalization_rejects_invalid_krx_numeric_values() -> None:
    import pytest

    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import _parse_krx_number

    with pytest.raises(PITDataError, match="invalid KRX numeric field"):
        _parse_krx_number(True, field="market_cap")
    with pytest.raises(PITDataError, match="invalid KRX numeric field"):
        _parse_krx_number("not-a-number", field="market_cap")
    with pytest.raises(PITDataError, match="invalid KRX numeric field"):
        _parse_krx_number("NaN", field="market_cap")


def test_streaming_normalization_requires_raw_cap_and_share_fields() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import _canonical_daily_row

    with pytest.raises(PITDataError, match='requires market_cap'):
        _canonical_daily_row(
            {'session': '2016-01-04', 'instrument_id': 'KRX:000020', 'open': 10, 'high': 11, 'low': 9, 'close': 10, 'volume': 1, 'trading_value': 10, 'market_cap': 100},
            available_at=datetime(2016, 1, 4, tzinfo=UTC), source_hash='x',
        )


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


def test_compact_corporate_action_intervals_preserves_gap_and_action_boundaries() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ
    from src.data.streaming_normalization import compact_corporate_action_intervals

    rows = [
        {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-03', 'session': '2016-01-04', 'action_code': 'no_action', 'adjustment_factor': 1.0},
        {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-04', 'session': '2016-01-05', 'action_code': 'no_action', 'adjustment_factor': 1.0},
        {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-05', 'session': '2016-01-06', 'action_code': 'split', 'adjustment_factor': 2.0},
        {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-08', 'session': '2016-01-09', 'action_code': 'no_action', 'adjustment_factor': 1.0},
    ]

    result = compact_corporate_action_intervals(rows, decision_time=datetime(2016, 1, 9, 15, 30, tzinfo=KRX_TZ))

    assert [(row['type'], row['effective_date'].date().isoformat(), row['coverage_end'].date().isoformat()) for row in result] == [('no_action', '2016-01-04', '2016-01-05'), ('split', '2016-01-06', '2016-01-06'), ('no_action', '2016-01-09', '2016-01-09')]


def test_compact_corporate_action_intervals_rejects_invalid_input() -> None:
    from datetime import datetime

    import pytest

    from src.core.time import KRX_TZ
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import compact_corporate_action_intervals

    with pytest.raises(PITDataError, match="decision_time"):
        compact_corporate_action_intervals([], decision_time=datetime(2016, 1, 1))
    with pytest.raises(PITDataError, match="malformed corporate-action"):
        compact_corporate_action_intervals([{}], decision_time=datetime(2016, 1, 1, tzinfo=KRX_TZ))


def test_compact_corporate_action_intervals_excludes_future_records() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ
    from src.data.streaming_normalization import compact_corporate_action_intervals

    result = compact_corporate_action_intervals(
        [
            {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-03', 'session': '2016-01-04', 'action_code': 'no_action'},
            {'instrument_id': 'KRX:000001', 'previous_session': '2016-01-04', 'session': '2017-01-03', 'action_code': 'no_action'},
        ],
        decision_time=datetime(2016, 12, 30, 15, 30, tzinfo=KRX_TZ),
    )
    assert len(result) == 1
    assert result[0]['coverage_end'].date().isoformat() == '2016-01-04'


def test_historical_available_at_separates_market_close_and_next_session_flow() -> None:
    from datetime import datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.schemas import EvidenceKind
    from src.data.streaming_normalization import historical_available_at

    calendar = SessionCalendar((datetime(2016, 1, 4, 9, tzinfo=KRX_TZ), datetime(2016, 1, 5, 9, tzinfo=KRX_TZ)))
    record = {'session': '2016-01-04', 'retrieved_at': '2026-09-05T00:00:00+00:00'}
    market_at = historical_available_at(kind=EvidenceKind.DAILY_MARKET, record=record, calendar=calendar)
    flow_at = historical_available_at(kind=EvidenceKind.INVESTOR_FLOW, record=record, calendar=calendar)

    assert (market_at.hour, market_at.minute, market_at.date().isoformat()) == (15, 30, '2016-01-04')
    assert (flow_at.hour, flow_at.minute, flow_at.date().isoformat()) == (9, 0, '2016-01-05')


def test_historical_available_at_rejects_missing_calendar_or_fields() -> None:
    from datetime import datetime
    import pytest
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.schemas import EvidenceKind, PITDataError
    from src.data.streaming_normalization import historical_available_at

    with pytest.raises(PITDataError, match='mapping'):
        historical_available_at(kind=EvidenceKind.DAILY_MARKET, record=None, calendar=SessionCalendar(()))
    calendar = SessionCalendar((datetime(2016, 1, 4, 9, tzinfo=KRX_TZ),))
    with pytest.raises(PITDataError, match='no sessions'):
        historical_available_at(kind=EvidenceKind.DAILY_MARKET, record={'session': '2016-01-04'}, calendar=SessionCalendar(()))
    with pytest.raises(PITDataError, match='missing session'):
        historical_available_at(kind=EvidenceKind.DAILY_MARKET, record={}, calendar=calendar)
    with pytest.raises(PITDataError, match='next KRX'):
        historical_available_at(kind=EvidenceKind.INVESTOR_FLOW, record={'session': '2016-01-04'}, calendar=calendar)
    with pytest.raises(PITDataError, match='missing session'):
        historical_available_at(kind=EvidenceKind.INVESTOR_FLOW, record={}, calendar=calendar)
    assert historical_available_at(
        kind=EvidenceKind.SECURITY_MASTER, record={'unexpected': 'sentinel'}, calendar=calendar
    ) == calendar.sessions[0]
    assert historical_available_at(
        kind=EvidenceKind.SECURITY_MASTER, record={'session': '2016-01-04'}, calendar=calendar
    ) == calendar.sessions[0]
    assert historical_available_at(
        kind=EvidenceKind.SECURITY_MASTER, record={'session': 'bad'}, calendar=calendar
    ) == calendar.sessions[0]
    assert historical_available_at(
        kind=EvidenceKind.FINANCIAL_FACTS,
        record={'published_at': '2016-01-03T12:00:00+09:00'},
        calendar=calendar,
    ) == calendar.sessions[0]
    with pytest.raises(PITDataError, match='DART record'):
        historical_available_at(kind=EvidenceKind.DISCLOSURES, record={}, calendar=calendar)
    with pytest.raises(PITDataError, match='DART record'):
        historical_available_at(kind=EvidenceKind.DISCLOSURES, record={'published_at': 'bad'}, calendar=calendar)
    with pytest.raises(PITDataError, match='after DART'):
        historical_available_at(kind=EvidenceKind.DISCLOSURES, record={'published_at': '2016-01-05T12:00:00+09:00'}, calendar=calendar)
    from types import SimpleNamespace
    with pytest.raises(PITDataError, match='unsupported'):
        historical_available_at(kind=SimpleNamespace(value='unknown'), record={'session': '2016-01-04'}, calendar=calendar)


def test_corporate_action_interval_parser_streams_top_level_array(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.streaming_normalization import _stream_corporate_action_intervals

    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps({'version': 'v1', 'intervals': [{'instrument_id': 'KRX:000001', 'session': '2016-01-04', 'action_code': 'no_action'}, {'instrument_id': 'KRX:000002', 'session': '2016-01-05', 'action_code': 'split'}]}), encoding='utf-8')
    receipt = BronzeReceipt(kind=EvidenceKind.CORPORATE_ACTIONS, content_hash='a' * 64, source_path='test', retrieved_at=datetime(2016, 1, 1, tzinfo=UTC), ingested_at=datetime(2016, 1, 1, tzinfo=UTC), payload_path=payload_path, metadata_path=tmp_path / 'receipt.json')

    assert list(_stream_corporate_action_intervals((receipt,), read_size=17)) == [{'instrument_id': 'KRX:000001', 'session': '2016-01-04', 'action_code': 'no_action'}, {'instrument_id': 'KRX:000002', 'session': '2016-01-05', 'action_code': 'split'}]


def test_corporate_action_interval_parser_fails_closed_for_invalid_payload(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
    from src.data.streaming_normalization import _stream_corporate_action_intervals

    payload_path = tmp_path / 'payload.json'
    payload_path.write_text('{"records": []}', encoding='utf-8')
    receipt = BronzeReceipt(kind=EvidenceKind.CORPORATE_ACTIONS, content_hash='b' * 64, source_path='test', retrieved_at=datetime(2016, 1, 1, tzinfo=UTC), ingested_at=datetime(2016, 1, 1, tzinfo=UTC), payload_path=payload_path, metadata_path=tmp_path / 'receipt.json')

    with pytest.raises(PITDataError, match='intervals'):
        list(_stream_corporate_action_intervals((receipt,)))
    with pytest.raises(PITDataError, match='read_size'):
        list(_stream_corporate_action_intervals((receipt,), read_size=0))


def test_corporate_action_interval_parser_rejects_truncation_and_unbounded_records(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data import streaming_normalization as module
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

    def receipt_for(name: str, content: str) -> BronzeReceipt:
        path = tmp_path / name
        path.write_text(content, encoding='utf-8')
        return BronzeReceipt(
            kind=EvidenceKind.CORPORATE_ACTIONS, content_hash='c' * 64,
            source_path='test', retrieved_at=datetime(2016, 1, 1, tzinfo=UTC),
            ingested_at=datetime(2016, 1, 1, tzinfo=UTC), payload_path=path,
            metadata_path=tmp_path / 'receipt.json',
        )

    missing = BronzeReceipt(
        kind=EvidenceKind.CORPORATE_ACTIONS, content_hash='d' * 64,
        source_path='test', retrieved_at=datetime(2016, 1, 1, tzinfo=UTC),
        ingested_at=datetime(2016, 1, 1, tzinfo=UTC), payload_path=tmp_path / 'missing.json',
        metadata_path=tmp_path / 'receipt.json',
    )
    with pytest.raises(PITDataError, match='missing corporate-action payload'):
        list(module._stream_corporate_action_intervals((missing,)))
    with pytest.raises(PITDataError, match='intervals'):
        list(module._stream_corporate_action_intervals((receipt_for('no-array.json', '{"intervals"'),)))
    with pytest.raises(PITDataError, match='unterminated'):
        list(module._stream_corporate_action_intervals((receipt_for('truncated.json', '{"intervals": ['),)))
    with pytest.raises(PITDataError, match='malformed corporate-action JSON'):
        list(module._stream_corporate_action_intervals((receipt_for('trailing.json', '{"intervals": []} trailing'),)))
    with pytest.raises(PITDataError, match='malformed corporate-action interval'):
        list(module._stream_corporate_action_intervals((receipt_for('scalar.json', '{"intervals": [1]}'),)))
    with pytest.raises(PITDataError, match='malformed corporate-action JSON'):
        list(module._stream_corporate_action_intervals((receipt_for('broken-object.json', '{"intervals": [{]}}'),)))
    monkeypatch.setattr(module, '_MAX_CORPORATE_ACTION_RECORD_BYTES', 20)
    with pytest.raises(PITDataError, match='bounded parser buffer'):
        list(module._stream_corporate_action_intervals((receipt_for('large.json', '{"intervals": [{"instrument_id": "KRX:000001"}]}'),)))
