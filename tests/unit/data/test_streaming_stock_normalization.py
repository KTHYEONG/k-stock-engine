def test_streaming_normalization_resumes_only_verified_months(tmp_path) -> None:
    from src.data.streaming_normalization import StreamingNormalizationCheckpoint

    store = StreamingNormalizationCheckpoint(tmp_path)
    store.mark_verified(table="daily_market", month="2020-01", source_hashes=("a",), output_hash="b")
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("a",)) is True
    assert store.is_verified(table="daily_market", month="2020-01", source_hashes=("changed",)) is False


def test_corporate_action_only_refresh_does_not_aggregate_other_bronze_pages(monkeypatch, tmp_path) -> None:
    import json
    from datetime import UTC, date, datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    import src.data.silver as silver
    import src.data.streaming_normalization as streaming
    from src.core.datasets import DatasetCertification
    from src.core.time import SessionCalendar
    from src.data.schemas import BronzeReceipt, CertificationReport, EvidenceKind, SilverTable

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2016, 11, 22, 9, tzinfo=tz)
    second = datetime(2016, 11, 23, 9, tzinfo=tz)
    listing = datetime(2016, 12, 14, 9, tzinfo=tz)
    decision = datetime(2026, 9, 5, tzinfo=UTC)
    receipts = {
        kind: (
            BronzeReceipt(
                kind=kind,
                content_hash=kind.value.ljust(64, '0'),
                source_path='fixture',
                retrieved_at=decision,
                ingested_at=decision,
                payload_path=tmp_path / f'{kind.value}.json',
                metadata_path=tmp_path / f'{kind.value}.receipt.json',
            ),
        )
        for kind in EvidenceKind
    }
    payload = {
        'endpoint': 'fricDecsn.json',
        'corp_code': '00219097',
        'status': '000',
        'requested_instrument_id': 'KRX:027410',
        'instrument_mapping_provenance': 'opendart_corp_code_direct',
        'records': [
            {
                'rcept_no': '20161107000214',
                'corp_code': '00219097',
                'bfic_tisstk_ostk': '24773964',
                'nstk_ostk_cnt': '24773661',
                'nstk_ascnt_ps_ostk': '1',
                'nstk_asstd': '2016-11-24',
                'nstk_lstprd': '2016-12-14',
            }
        ],
    }
    (tmp_path / f'{EvidenceKind.CORPORATE_ACTIONS.value}.json').write_text(json.dumps(payload), encoding='utf-8')
    daily = pl.DataFrame(
        {
            'session': [first, second, listing],
            'instrument_id': ['KRX:027410'] * 3,
            'close': [168500.0, 82600.0, 88000.0],
            'shares_outstanding': [24773964.0, 24773964.0, 49547625.0],
            'market_cap': [4174412934000.0, 2046329426400.0, 4360191000000.0],
        }
    ).lazy()
    report = CertificationReport(
        certification=DatasetCertification.RESEARCH,
        report_hash='report',
        coverage_start=date(2016, 11, 22),
        coverage_end=date(2016, 12, 14),
        source_hashes={kind: items[0].content_hash for kind, items in receipts.items()},
    )

    monkeypatch.setattr(streaming, 'discover_verified_bronze_receipts', lambda **_: receipts)
    monkeypatch.setattr(streaming, '_aggregate_small', lambda **_: pytest.fail('must not aggregate'))
    monkeypatch.setattr(silver, 'certify_corporate_action_refresh', lambda **_: report)

    class FakeStore:
        def __init__(self, root) -> None:
            self.root = root

        def materialize_all(self, tables, **kwargs):
            assert set(tables) == {SilverTable.CORPORATE_ACTIONS}
            assert tables[SilverTable.CORPORATE_ACTIONS]['evidence_status'].to_list() == ['verified']
            return {SilverTable.CORPORATE_ACTIONS: tmp_path / 'silver'}

    monkeypatch.setattr(silver, 'SilverStore', FakeStore)

    result = streaming.refresh_corporate_action_silver(
        bronze_root=tmp_path / 'bronze',
        silver_root=tmp_path / 'silver',
        artifact_root=tmp_path / 'artifacts',
        decision_time=listing,
        daily_market=daily,
        calendar=SessionCalendar((first, second, listing)),
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


def test_resolve_opendart_records_rejects_malformed_page_and_receipt() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    import pytest
    from src.core.time import SessionCalendar
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    session = datetime(2024, 1, 2, 9, tzinfo=ZoneInfo("Asia/Seoul"))
    daily = pl.DataFrame({"session": [session], "instrument_id": ["KRX:A"], "close": [100.0]})
    calendar = SessionCalendar((session,))
    with pytest.raises(PITDataError, match="unexpected OpenDART status"):
        resolve_opendart_corporate_action_records(pages=[{"endpoint": "fricDecsn.json", "corp_code": "1", "status": "999", "records": []}], daily_market=daily, calendar=calendar)
    with pytest.raises(PITDataError, match="invalid OpenDART records"):
        resolve_opendart_corporate_action_records(pages=[{"endpoint": "fricDecsn.json", "corp_code": "1", "status": "000", "records": {}}], daily_market=daily, calendar=calendar)
    with pytest.raises(PITDataError, match="invalid OpenDART receipt"):
        resolve_opendart_corporate_action_records(pages=[{"endpoint": "fricDecsn.json", "corp_code": "1", "status": "000", "records": [{"rcept_no": "bad", "corp_code": "1", "bfic_tisstk_ostk": "1", "nstk_ostk_cnt": "1", "nstk_ascnt_ps_ostk": "1", "nstk_asstd": "2024-01-02", "nstk_lstprd": "2024-01-02"}]}], daily_market=daily, calendar=calendar)


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


def test_canonical_master_row_preserves_provider_listing_date_and_unknown_status() -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _canonical_master_row

    row = _canonical_master_row(
        {'ticker': '005930', 'market': 'KOSPI', 'listing_date': '1975-06-11'},
        available_at=datetime(2016, 1, 4, tzinfo=UTC),
        source_hash='a' * 64,
        fallback_session=datetime(2016, 1, 4, tzinfo=UTC),
    )

    assert row['listing_date'].date().isoformat() == '1975-06-11'
    assert row['valid_from'] == datetime(2016, 1, 4, tzinfo=UTC)
    assert row['status'] == '__UNKNOWN__'


def test_resolve_bonus_issue_027410_style_event() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    krx = ZoneInfo('Asia/Seoul')
    sessions = tuple(datetime(2016, 11, day, 9, tzinfo=krx) for day in (8, 22, 23, 24))
    daily = pl.DataFrame({'session': sessions, 'instrument_id': ['KRX:027410'] * 4, 'close': [165000.0, 168500.0, 82600.0, 83000.0], 'shares_outstanding': [24773964.0, 24773964.0, 24773964.0, 49547625.0], 'market_cap': [4087704060000.0, 4174412934000.0, 2046329426400.0, 4112452875000.0]})
    pages = [{'endpoint': 'fricDecsn.json', 'corp_code': '00219097', 'status': '000', 'records': [{'rcept_no': '20161107000214', 'corp_code': '00219097', 'bfic_tisstk_ostk': '24,773,964', 'nstk_ostk_cnt': '24,773,661', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2016년 11월 24일', 'nstk_lstprd': '2016년 11월 24일'}]}]

    records = resolve_opendart_corporate_action_records(pages=pages, daily_market=daily, calendar=SessionCalendar(sessions))

    assert records[0]['instrument_id'] == 'KRX:027410'
    assert records[0]['action_type'] == 'bonus_issue'
    assert records[0]['factor'] == 2.0
    assert records[0]['effective_session'].date().isoformat() == '2016-11-23'
    assert records[0]['available_at'].date().isoformat() == '2016-11-08'
    assert records[0]['evidence_status'] == 'verified'


# test_stream_normalization_rejects_unmodelled_event_and_unexplained_jump
def test_resolve_opendart_records_rejects_unmodelled_merger() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    session = datetime(2024, 1, 2, 9, tzinfo=ZoneInfo('Asia/Seoul'))
    daily = pl.DataFrame({'session': [session], 'instrument_id': ['KRX:A'], 'close': [100.0], 'shares_outstanding': [1.0], 'market_cap': [100.0]})
    resolved = resolve_opendart_corporate_action_records(pages=[{'endpoint': 'cmpMgDecsn.json', 'corp_code': '00123456', 'status': '000', 'records': [{'rcept_no': '20240101000001'}]}], daily_market=daily, calendar=SessionCalendar((session,)))
    assert resolved[0]['evidence_status'] == 'unresolved'
    assert resolved[0]['evidence_reason'] == 'unsupported_merger'
    assert resolved[0]['factor'] == 1.0


def test_resolve_bonus_issue_emits_distinct_price_and_listing_sessions() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    sessions = tuple(datetime(2016, month, day, 9, tzinfo=tz) for month, day in ((11, 8), (11, 22), (11, 23), (11, 24), (12, 13), (12, 14)))
    daily = pl.DataFrame({'session': sessions, 'instrument_id': ['KRX:027410'] * 6, 'close': [165000.0, 168500.0, 82600.0, 79500.0, 86200.0, 88000.0], 'shares_outstanding': [24773964.0, 24773964.0, 24773964.0, 24773964.0, 24773964.0, 49547625.0], 'market_cap': [4087704060000.0, 4174412934000.0, 2046329426400.0, 1969530138000.0, 2135515696800.0, 4360191000000.0]})
    pages = [{'endpoint': 'fricDecsn.json', 'corp_code': '00219097', 'status': '000', 'records': [{'rcept_no': '20161107000214', 'corp_code': '00219097', 'bfic_tisstk_ostk': '24,773,964', 'nstk_ostk_cnt': '24,773,661', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2016년 11월 24일', 'nstk_lstprd': '2016년 12월 14일'}]}]

    action = resolve_opendart_corporate_action_records(pages=pages, daily_market=daily, calendar=SessionCalendar(sessions))[0]

    assert action['effective_session'].date().isoformat() == '2016-11-23'
    assert action['share_listing_date'].date().isoformat() == '2016-12-14'
    assert action['factor'] == 2.0
    assert action['share_delta'] == 24773661


def test_resolve_bonus_issue_requires_listing_date_and_preserves_settlement_fields() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    second = datetime(2024, 1, 3, 9, tzinfo=tz)
    daily = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 49.0], 'shares_outstanding': [10.0, 10.0], 'market_cap': [1000.0, 490.0]})
    record = {'rcept_no': '20240101000001', 'corp_code': '1', 'instrument_id': 'KRX:A', 'bfic_tisstk_ostk': '10', 'nstk_ostk_cnt': '10', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2024-01-03'}

    with pytest.raises(PITDataError, match='nstk_lstprd'):
        resolve_opendart_corporate_action_records(pages=[{'endpoint': 'fricDecsn.json', 'corp_code': '1', 'status': '000', 'records': [record]}], daily_market=daily, calendar=SessionCalendar((first, second)))


def test_corporate_action_silver_schema_requires_settlement_fields() -> None:
    from src.data.schemas import SilverTable
    from src.data.silver import SCHEMA_REGISTRY

    required = SCHEMA_REGISTRY[SilverTable.CORPORATE_ACTIONS]['required_columns']

    assert 'share_listing_date' in required
    assert 'share_delta' in required


def test_resolve_bonus_issue_rejects_out_of_calendar_listing_date() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2024, 1, 2, 9, tzinfo=tz)
    second = datetime(2024, 1, 3, 9, tzinfo=tz)
    daily = pl.DataFrame({'session': [first, second], 'instrument_id': ['KRX:A', 'KRX:A'], 'close': [100.0, 49.0], 'shares_outstanding': [10.0, 10.0], 'market_cap': [1000.0, 490.0]})
    record = {'rcept_no': '20240101000001', 'corp_code': '1', 'instrument_id': 'KRX:A', 'bfic_tisstk_ostk': '10', 'nstk_ostk_cnt': '10', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2024-01-03', 'nstk_lstprd': '2024-02-01'}

    with pytest.raises(PITDataError, match='listing'):
        resolve_opendart_corporate_action_records(pages=[{'endpoint': 'fricDecsn.json', 'corp_code': '1', 'status': '000', 'records': [record]}], daily_market=daily, calendar=SessionCalendar((first, second)))


def test_resolve_opendart_bonus_requires_all_four_krx_proofs() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    sessions = (datetime(2016, 11, 8, 9, tzinfo=tz), datetime(2016, 11, 22, 9, tzinfo=tz), datetime(2016, 11, 23, 9, tzinfo=tz), datetime(2016, 12, 13, 9, tzinfo=tz), datetime(2016, 12, 14, 9, tzinfo=tz))
    daily = pl.DataFrame({'session': sessions, 'instrument_id': ['KRX:027410'] * 5, 'close': [165000.0, 168500.0, 82600.0, 86200.0, 88000.0], 'shares_outstanding': [24773964.0, 24773964.0, 24773964.0, 24773964.0, 49547625.0], 'market_cap': [4087704060000.0, 4174412934000.0, 2046329426400.0, 2135515696800.0, 4360191000000.0]})
    record = {'rcept_no': '20161107000214', 'corp_code': '00219097', 'bfic_tisstk_ostk': '24,773,964', 'nstk_ostk_cnt': '24,773,661', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2016년 11월 24일', 'nstk_lstprd': '2016년 12월 14일'}
    page = {'endpoint': 'fricDecsn.json', 'corp_code': '00219097', 'status': '000', 'requested_instrument_id': 'KRX:027410', 'instrument_mapping_provenance': 'opendart_corp_code_direct', 'records': [record]}
    verified = resolve_opendart_corporate_action_records(pages=[page], daily_market=daily, calendar=SessionCalendar(sessions))[0]
    assert verified['evidence_status'] == 'verified'
    assert verified['factor'] == 2.0
    assert verified['share_listing_date'] == sessions[-1]
    bad_daily = daily.with_columns(pl.when(pl.col('session') == sessions[-1]).then(1.0).otherwise(pl.col('market_cap')).alias('market_cap'))
    unresolved = resolve_opendart_corporate_action_records(pages=[page], daily_market=bad_daily, calendar=SessionCalendar(sessions))[0]
    assert unresolved['evidence_status'] == 'unresolved'
    assert unresolved['action_type'] == 'unresolved'
    assert unresolved['factor'] == 1.0
    assert unresolved['evidence_reason'] == 'krx_listing_market_cap_mismatch'


def test_mapped_action_instruments_rejects_issuer_inference() -> None:
    from src.data.streaming_normalization import mapped_action_instruments

    pages = [
        {'endpoint': 'fricDecsn.json', 'requested_instrument_id': 'KRX:005930', 'instrument_mapping_provenance': 'opendart_corp_code_direct', 'records': []},
        {'endpoint': 'fricDecsn.json', 'corp_code': '00126380', 'records': [{'ticker': '005935'}]},
        {'endpoint': 'fricDecsn.json', 'corp_code': '00126380', 'records': []},
    ]
    assert mapped_action_instruments(pages=pages) == frozenset({'KRX:005930', 'KRX:005935'})


def test_refresh_corporate_action_silver_uses_structured_evidence_not_legacy_intervals(monkeypatch, tmp_path) -> None:
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    import polars as pl
    import src.data.streaming_normalization as module
    from src.core.datasets import DatasetCertification
    from src.core.time import SessionCalendar
    from src.data.schemas import CertificationReport, EvidenceKind

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2016, 11, 22, 9, tzinfo=tz)
    second = datetime(2016, 11, 23, 9, tzinfo=tz)
    listing = datetime(2016, 12, 14, 9, tzinfo=tz)
    daily = pl.DataFrame({'session': [first, second, listing], 'instrument_id': ['KRX:027410'] * 3, 'close': [168500.0, 82600.0, 88000.0], 'shares_outstanding': [24773964.0, 24773964.0, 49547625.0], 'market_cap': [4174412934000.0, 2046329426400.0, 4360191000000.0]}).lazy()
    page = {'endpoint': 'fricDecsn.json', 'corp_code': '00219097', 'status': '000', 'requested_instrument_id': 'KRX:027410', 'instrument_mapping_provenance': 'opendart_corp_code_direct', 'records': [{'rcept_no': '20161107000214', 'corp_code': '00219097', 'bfic_tisstk_ostk': '24773964', 'nstk_ostk_cnt': '24773661', 'nstk_ascnt_ps_ostk': '1', 'nstk_asstd': '2016-11-24', 'nstk_lstprd': '2016-12-14'}]}
    captured = {}
    report = CertificationReport(certification=DatasetCertification.RESEARCH, report_hash='refresh', coverage_start=date(2016, 11, 22), coverage_end=date(2016, 12, 14), source_hashes={EvidenceKind.CORPORATE_ACTIONS: 'h'})
    monkeypatch.setattr(module, 'load_structured_corporate_action_pages', lambda **_: [page])
    monkeypatch.setattr(module, 'compact_corporate_action_intervals', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('legacy interval path')) )
    def persist(**kwargs):
        captured['frame'] = kwargs['action_frame']
        return report
    monkeypatch.setattr(module, '_persist_corporate_action_refresh', persist)
    actual = module.refresh_corporate_action_silver(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts', decision_time=listing, daily_market=daily, calendar=SessionCalendar((first, second, listing)))
    assert actual is report
    assert captured['frame'].select('evidence_status').to_series().to_list() == ['verified']


def test_stream_items_uses_eager_parser_below_bound(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import json
    import src.data.streaming_normalization as mod
    from src.data.schemas import BronzeReceipt, EvidenceKind

    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps({'records': [{'id': 1}, {'id': 2}]}), encoding='utf-8')
    receipt = BronzeReceipt(EvidenceKind.DAILY_MARKET, 'a' * 64, 'krx:daily-market:2016-01-04', datetime(2016, 1, 5, tzinfo=UTC), datetime(2016, 1, 5, tzinfo=UTC), payload, tmp_path / 'receipt.json')
    monkeypatch.setattr(mod.shutil, 'which', lambda _name: None)

    assert list(mod._stream_items_for_kind([receipt], batch_size=50000)) == [{'id': 1}, {'id': 2}]


def test_stream_items_keeps_jq_for_payload_at_bound(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import json
    import src.data.streaming_normalization as mod
    from src.data.schemas import BronzeReceipt, EvidenceKind

    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps({'records': [{'id': 1}], 'padding': 'x' * mod.STREAMING_EAGER_JSON_MAX_BYTES}), encoding='utf-8')
    receipt = BronzeReceipt(EvidenceKind.DAILY_MARKET, 'b' * 64, 'krx:daily-market:2016-01-04', datetime(2016, 1, 5, tzinfo=UTC), datetime(2016, 1, 5, tzinfo=UTC), payload, tmp_path / 'receipt.json')
    monkeypatch.setattr(mod, '_read_doc', lambda _path: (_ for _ in ()).throw(AssertionError('large payload was eagerly read')))

    assert list(mod._stream_items_for_kind([receipt], batch_size=1)) == [{'id': 1}]


def test_order_streaming_receipts_sorts_known_labels_and_rejects_unbounded_fallback(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    import pytest
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable
    from src.data.streaming_normalization import STREAMING_EAGER_JSON_MAX_BYTES, order_streaming_receipts

    def receipt(name: str, label: str, body: dict[str, object]) -> BronzeReceipt:
        path = tmp_path / name
        path.write_text(json.dumps(body), encoding='utf-8')
        return BronzeReceipt(EvidenceKind.DAILY_MARKET, name[0] * 64, label, datetime(2016, 2, 1, tzinfo=UTC), datetime(2016, 2, 1, tzinfo=UTC), path, tmp_path / (name + '.receipt'))

    later = receipt('b.json', 'krx:daily-market:2016-02-01', {'records': []})
    earlier = receipt('a.json', 'krx:daily-market:2016-01-04', {'records': []})
    assert order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(later, earlier)) == (earlier, later)
    large = receipt('c.json', 'unknown', {'padding': 'x' * STREAMING_EAGER_JSON_MAX_BYTES})
    with pytest.raises(PITDataError, match='event date'):
        order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(large,))


def test_streaming_writer_flushes_month_tail_and_rejects_regression(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import PITDataError, SilverTable
    from src.data.streaming_normalization import StreamingSilverWriter

    def row(day: int) -> dict[str, object]:
        stamp = datetime(2020, 1, day, tzinfo=UTC)
        return {'session': stamp, 'instrument_id': f'KRX:{day:06d}', 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0, 'trading_value': 1.0, 'market_cap': 1.0, 'shares_outstanding': 1.0, 'available_at': stamp, 'source_hash': 'source'}

    writer = StreamingSilverWriter(tmp_path / 'staging', table=SilverTable.DAILY_MARKET, batch_size=3, source_hashes=('source',))
    writer.append(month='2020-01', row=row(2))
    writer.append(month='2020-02', row=row(3))
    assert (tmp_path / 'staging' / 'daily_market' / 'year=2020' / 'month=01' / 'part-00000.parquet').exists()
    with pytest.raises(PITDataError, match='month order'):
        writer.append(month='2020-01', row=row(4))


def test_streaming_writer_reuses_digest_checked_sealed_month_only(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.schemas import SilverTable
    from src.data.streaming_normalization import StreamingSilverWriter

    def row(month: int, day: int) -> dict[str, object]:
        stamp = datetime(2020, month, day, tzinfo=UTC)
        return {'session': stamp, 'instrument_id': 'KRX:000001', 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0, 'trading_value': 1.0, 'market_cap': 1.0, 'shares_outstanding': 1.0, 'available_at': stamp, 'source_hash': 'source'}

    root = tmp_path / 'staging'
    first = StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=2, source_hashes=('source',))
    first.append(month='2020-01', row=row(1, 2))
    first.append(month='2020-02', row=row(2, 3))
    manifest = json.loads((root / 'daily_market' / 'staging_manifest.json').read_text())
    assert manifest['verified'] is False
    assert manifest['sealed_months'] == ['2020-01']
    resumed = StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=2, source_hashes=('source',))
    assert resumed.has_reusable_manifest is True
    changed = StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=2, source_hashes=('changed',))
    assert changed.has_reusable_manifest is False


def test_order_streaming_receipts_uses_small_payload_event_date_fallback(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable
    from src.data.streaming_normalization import order_streaming_receipts

    fallback_payload = tmp_path / 'fallback.json'
    fallback_payload.write_text(json.dumps({'session': '2016-01-04'}), encoding='utf-8')
    known_payload = tmp_path / 'known.json'
    known_payload.write_text(json.dumps({'records': []}), encoding='utf-8')
    stamp = datetime(2016, 2, 1, tzinfo=UTC)
    known = BronzeReceipt(EvidenceKind.DAILY_MARKET, 'd' * 64, 'krx:daily-market:2016-01-05', stamp, stamp, known_payload, tmp_path / 'known.receipt')
    fallback = BronzeReceipt(EvidenceKind.DAILY_MARKET, 'e' * 64, 'unlabelled-source', stamp, stamp, fallback_payload, tmp_path / 'fallback.receipt')

    assert order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(known, fallback)) == (fallback, known)
    alternate_payload = tmp_path / 'alternate.json'
    alternate_payload.write_text(json.dumps({'session': 'not-a-date', 'date': '2016-01-03'}), encoding='utf-8')
    alternate = BronzeReceipt(EvidenceKind.DAILY_MARKET, 'f' * 64, 'unlabelled-source', stamp, stamp, alternate_payload, tmp_path / 'alternate.receipt')
    assert order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(fallback, alternate)) == (alternate, fallback)


def test_order_streaming_receipts_rejects_missing_malformed_and_dateless_fallback(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable
    from src.data.streaming_normalization import order_streaming_receipts

    stamp = datetime(2016, 2, 1, tzinfo=UTC)
    def receipt(name: str) -> BronzeReceipt:
        return BronzeReceipt(EvidenceKind.DAILY_MARKET, name[0] * 64, 'unknown', stamp, stamp, tmp_path / name, tmp_path / (name + '.receipt'))

    missing = receipt('missing.json')
    with pytest.raises(PITDataError, match='missing event date'):
        order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(missing,))
    malformed = receipt('malformed.json')
    malformed.payload_path.write_text('{not-json', encoding='utf-8')
    with pytest.raises(PITDataError, match='missing event date'):
        order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(malformed,))
    dateless = receipt('dateless.json')
    dateless.payload_path.write_text('{}', encoding='utf-8')
    with pytest.raises(PITDataError, match='missing event date'):
        order_streaming_receipts(table=SilverTable.DAILY_MARKET, receipts=(dateless,))


def test_streaming_writer_does_not_reuse_active_month_batch_before_boundary(tmp_path) -> None:
    from datetime import UTC, datetime
    import json
    from src.data.schemas import SilverTable
    from src.data.streaming_normalization import StreamingSilverWriter

    def row(month: int, day: int) -> dict[str, object]:
        stamp = datetime(2020, month, day, tzinfo=UTC)
        return {'session': stamp, 'instrument_id': f'KRX:{month:02d}{day:04d}', 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0, 'trading_value': 1.0, 'market_cap': 1.0, 'shares_outstanding': 1.0, 'available_at': stamp, 'source_hash': 'source'}

    root = tmp_path / 'staging'
    writer = StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=1, source_hashes=('source',))
    writer.append(month='2020-01', row=row(1, 2))
    manifest = json.loads((root / 'daily_market' / 'staging_manifest.json').read_text())
    assert manifest['sealed_months'] == []
    assert StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=1, source_hashes=('source',)).has_reusable_manifest is False
    writer.append(month='2020-02', row=row(2, 3))
    manifest = json.loads((root / 'daily_market' / 'staging_manifest.json').read_text())
    assert manifest['sealed_months'] == ['2020-01']
    assert StreamingSilverWriter(root, table=SilverTable.DAILY_MARKET, batch_size=1, source_hashes=('source',)).has_reusable_manifest is True
    writer.close()
    manifest = json.loads((root / 'daily_market' / 'staging_manifest.json').read_text())
    assert manifest['sealed_months'] == ['2020-01', '2020-02']


def test_stream_items_fail_closed_for_missing_malformed_and_scalar_records(tmp_path) -> None:
    from datetime import UTC, datetime
    import pytest
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError
    from src.data.streaming_normalization import _stream_items_for_kind

    stamp = datetime(2016, 1, 5, tzinfo=UTC)
    def receipt(name: str) -> BronzeReceipt:
        payload = tmp_path / name
        return BronzeReceipt(EvidenceKind.DAILY_MARKET, name[0] * 64, 'krx:daily-market:2016-01-04', stamp, stamp, payload, tmp_path / (name + '.receipt'))

    missing = receipt('missing.json')
    with pytest.raises(PITDataError, match='malformed Bronze JSON'):
        list(_stream_items_for_kind([missing], batch_size=1))
    malformed = receipt('malformed.json')
    malformed.payload_path.write_text('{not-json', encoding='utf-8')
    with pytest.raises(PITDataError, match='malformed Bronze JSON'):
        list(_stream_items_for_kind([malformed], batch_size=1))
    scalar = receipt('scalar.json')
    scalar.payload_path.write_text('{"records":[1]}', encoding='utf-8')
    with pytest.raises(PITDataError, match='malformed record'):
        list(_stream_items_for_kind([scalar], batch_size=1))


def test_stream_normalize_wires_ordering_for_both_stream_tables(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime
    import pytest
    import src.data.streaming_normalization as mod
    from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError, SilverTable

    stamp = datetime(2016, 1, 5, tzinfo=UTC)
    grouped: dict[EvidenceKind, tuple[BronzeReceipt, ...]] = {}
    for index, kind in enumerate(EvidenceKind):
        payload = tmp_path / f'{kind.value}.json'
        if kind is EvidenceKind.DAILY_MARKET:
            payload.write_text('{"records": []}', encoding='utf-8')
            label = 'krx:daily-market:2016-01-04'
        elif kind is EvidenceKind.SECURITY_MASTER:
            payload.write_text('{"records": []}', encoding='utf-8')
            label = 'KRX:historical-master:2016-01-04'
        else:
            payload.write_text('{}', encoding='utf-8')
            label = f'fixture:{kind.value}'
        grouped[kind] = (BronzeReceipt(kind, f'{index:064x}', label, stamp, stamp, payload, tmp_path / f'{kind.value}.receipt'),)
    calls: list[SilverTable] = []
    def stop_after_second_order(*, table: SilverTable, receipts: tuple[BronzeReceipt, ...]) -> tuple[BronzeReceipt, ...]:
        calls.append(table)
        if len(calls) == 2:
            raise PITDataError('ordering-wiring-stop')
        return receipts
    monkeypatch.setattr(mod, 'discover_verified_bronze_receipts', lambda **_kwargs: grouped)
    monkeypatch.setattr(mod, 'order_streaming_receipts', stop_after_second_order)

    with pytest.raises(PITDataError, match='ordering-wiring-stop'):
        mod.stream_normalize_stock_evidence(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts', decision_time=stamp)
    assert calls == [SilverTable.DAILY_MARKET, SilverTable.SECURITY_MASTER]
