import pytest


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


def test_streaming_normalization_preserves_krx_isin_as_source_security_id() -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _canonical_daily_row

    row = _canonical_daily_row(
        {
            'BAS_DD': '20160104', 'ISU_CD': 'KR7000020000', 'ISU_SRT_CD': '000020',
            'TDD_OPNPRC': '10', 'TDD_HGPRC': '12', 'TDD_LWPRC': '9',
            'TDD_CLSPRC': '11', 'ACC_TRDVOL': '100', 'ACC_TRDVAL': '1,100',
            'MKTCAP': '10,000', 'LIST_SHRS': '900',
        },
        available_at=datetime(2016, 1, 5, tzinfo=UTC),
        source_hash='a',
    )

    assert row['source_security_id'] == 'KR7000020000'


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


def test_resolve_capital_reduction_verifies_share_basis_and_price_factor() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    sessions = tuple(datetime(2016, 1, day, 9, tzinfo=tz) for day in (4, 5, 6))
    daily = pl.DataFrame(
        {
            'session': sessions,
            'instrument_id': ['KRX:TEST'] * 3,
            'close': [100.0, 500.0, 510.0],
            'shares_outstanding': [100000.0, 20000.0, 20000.0],
            'market_cap': [10000000.0, 10000000.0, 10200000.0],
        }
    )
    pages = [
        {
            'endpoint': 'crDecsn.json',
            'corp_code': '00000001',
            'requested_instrument_id': 'KRX:TEST',
            'instrument_mapping_provenance': 'opendart_corp_code_direct',
            'status': '000',
            'records': [
                {
                    'rcept_no': '20160101000001',
                    'corp_code': '00000001',
                    'bfcr_tisstk_ostk': '100,000',
                    'atcr_tisstk_ostk': '20,000',
                    'crsc_nstklstprd': '2016년 1월 5일',
                }
            ],
        }
    ]

    action = resolve_opendart_corporate_action_records(
        pages=pages, daily_market=daily, calendar=SessionCalendar(sessions)
    )[0]

    assert action['evidence_status'] == 'verified'
    assert action['action_type'] == 'reverse_split'
    assert action['factor'] == 0.2
    assert action['share_delta'] == -80000
    assert action['effective_session'].date().isoformat() == '2016-01-05'


def test_resolve_composite_capital_reduction_and_issuance() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    sessions = tuple(datetime(2016, 2, day, 9, tzinfo=tz) for day in (4, 5, 6))
    daily = pl.DataFrame(
        {
            'session': sessions,
            'instrument_id': ['KRX:TEST'] * 3,
            'close': [100.0, 400.0, 405.0],
            'shares_outstanding': [1000.0, 250.0, 250.0],
            'market_cap': [100000.0, 100000.0, 101250.0],
        }
    )
    pages = [
        {
            'endpoint': 'crDecsn.json',
            'corp_code': '00000001',
            'requested_instrument_id': 'KRX:TEST',
            'instrument_mapping_provenance': 'opendart_corp_code_direct',
            'status': '000',
            'records': [
                {
                    'rcept_no': '20160201000001',
                    'corp_code': '00000001',
                    'bfcr_tisstk_ostk': '1,000',
                    'atcr_tisstk_ostk': '200',
                    'crsc_nstklstprd': '2016년 2월 5일',
                },
                {
                    'rcept_no': '20160201000003',
                    'corp_code': '00000001',
                    'bfcr_tisstk_ostk': '300',
                    'atcr_tisstk_ostk': '200',
                    'crsc_nstklstprd': '2016년 2월 5일',
                },
            ],
        },
        {
            'endpoint': 'piicDecsn.json',
            'corp_code': '00000001',
            'requested_instrument_id': 'KRX:TEST',
            'instrument_mapping_provenance': 'opendart_corp_code_direct',
            'status': '000',
            'records': [
                {
                    'rcept_no': '20160201000002',
                    'corp_code': '00000001',
                    'bfic_tisstk_ostk': '200',
                    'nstk_ostk_cnt': '50',
                }
            ],
        },
    ]

    action = resolve_opendart_corporate_action_records(
        pages=pages, daily_market=daily, calendar=SessionCalendar(sessions)
    )[0]

    assert action['evidence_status'] == 'verified'
    assert action['factor'] == 0.25
    assert action['share_delta'] == -750
    assert action['action_id'] == '20160201000001+20160201000002'


def test_resolve_composite_graph_skips_malformed_candidate_edges() -> None:
    """Malformed composite DART edges are excluded without weakening the later fail-closed resolver."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl
    import pytest

    from src.core.time import SessionCalendar
    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    first = datetime(2016, 2, 4, 9, tzinfo=tz)
    listed = datetime(2016, 2, 5, 9, tzinfo=tz)
    daily = pl.DataFrame(
        {
            'session': [first, listed],
            'instrument_id': ['KRX:TEST', 'KRX:TEST'],
            'close': [100.0, 400.0],
            'shares_outstanding': [1000.0, 250.0],
            'market_cap': [100000.0, 100000.0],
        }
    )
    pages = [
        {
            'endpoint': 'crDecsn.json', 'corp_code': '00000001', 'status': '000',
            'records': [
                object(),
                {'rcept_no': '20160201000000', 'bfcr_tisstk_ostk': '1000', 'atcr_tisstk_ostk': '200', 'crsc_nstklstprd': 'not-a-date'},
                {'rcept_no': '20160201000001', 'bfcr_tisstk_ostk': 'bad', 'atcr_tisstk_ostk': '200', 'crsc_nstklstprd': '2016-02-05'},
                {'rcept_no': '20160201000003', 'bfcr_tisstk_ostk': '1000', 'atcr_tisstk_ostk': '200', 'crsc_nstklstprd': '2016-02-05'},
            ],
        },
        {
            'endpoint': 'piicDecsn.json', 'corp_code': '00000001', 'status': '000',
            'requested_instrument_id': 'KRX:TEST',
            'instrument_mapping_provenance': 'opendart_corp_code_direct',
            'records': [
                object(),
                {'rcept_no': '20150101000001', 'bfic_tisstk_ostk': '200', 'nstk_ostk_cnt': '50'},
                {'rcept_no': '20160201000002', 'bfic_tisstk_ostk': 'bad', 'nstk_ostk_cnt': '50'},
            ],
        },
    ]

    with pytest.raises(PITDataError, match='invalid OpenDART record'):
        resolve_opendart_corporate_action_records(
            pages=pages, daily_market=daily, calendar=SessionCalendar((first, listed))
        )


@pytest.mark.parametrize(
    ("sessions", "closes", "shares", "receipt_no"),
    [
        ((5,), (400.0,), (250.0,), "20160201000001"),
        ((4, 5), (100.0, 120.0), (1000.0, 250.0), "20160201000001"),
        ((4, 5, 8), (100.0, 400.0, 405.0), (1000.0, 250.0, 250.0), "20160205000001"),
    ],
)
def test_resolve_composite_graph_excludes_unexecutable_paths(
    sessions: tuple[int, ...],
    closes: tuple[float, ...],
    shares: tuple[float, ...],
    receipt_no: str,
) -> None:
    """No prior bar, unreconciled return, and late receipt all remain unresolved."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import polars as pl

    from src.core.time import SessionCalendar
    from src.data.streaming_normalization import resolve_opendart_corporate_action_records

    tz = ZoneInfo('Asia/Seoul')
    moments = tuple(datetime(2016, 2, day, 9, tzinfo=tz) for day in sessions)
    daily = pl.DataFrame(
        {
            'session': moments,
            'instrument_id': ['KRX:TEST'] * len(moments),
            'close': closes,
            'shares_outstanding': shares,
            'market_cap': tuple(close * count for close, count in zip(closes, shares, strict=True)),
        }
    )
    page = {
        'endpoint': 'crDecsn.json', 'corp_code': '00000001',
        'requested_instrument_id': 'KRX:TEST',
        'instrument_mapping_provenance': 'opendart_corp_code_direct', 'status': '000',
        'records': [{
            'rcept_no': receipt_no, 'corp_code': '00000001',
            'bfcr_tisstk_ostk': '1000', 'atcr_tisstk_ostk': '250',
            'crsc_nstklstprd': '2016-02-05',
        }],
    }
    resolved = resolve_opendart_corporate_action_records(
        pages=[page], daily_market=daily, calendar=SessionCalendar(moments)
    )
    assert resolved[0]['evidence_status'] == 'unresolved'
    assert resolved[0]['evidence_reason'] == 'unresolved_reverse_split'


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
    compact = receipt('compact.json', 'data/evidence/stocks/master_20160104_20260310_historical_v1.json', {'records': []})
    assert order_streaming_receipts(table=SilverTable.SECURITY_MASTER, receipts=(later, compact)) == (compact, later)
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


def test_streaming_writer_reuses_prior_months_for_append_only_receipts(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.schemas import SilverTable
    from src.data.streaming_normalization import StreamingSilverWriter

    def row(stamp: datetime, symbol: str) -> dict[str, object]:
        return {
            'session': stamp,
            'instrument_id': symbol,
            'open': 1.0,
            'high': 1.0,
            'low': 1.0,
            'close': 1.0,
            'volume': 1.0,
            'trading_value': 1.0,
            'market_cap': 1.0,
            'shares_outstanding': 1.0,
            'available_at': stamp,
            'source_hash': 'source',
        }

    root = tmp_path / 'staging'
    first = StreamingSilverWriter(
        root,
        table=SilverTable.DAILY_MARKET,
        batch_size=1,
        source_hashes=('old',),
        source_paths=('krx:daily-market:2020-01-02',),
    )
    first.append(month='2020-01', row=row(datetime(2020, 1, 2, tzinfo=UTC), 'KRX:000001'))
    first.close()

    resumed = StreamingSilverWriter(
        root,
        table=SilverTable.DAILY_MARKET,
        batch_size=1,
        source_hashes=('old', 'new'),
        source_paths=('krx:daily-market:2020-01-02', 'krx:daily-market:2020-02-03'),
    )
    assert resumed.has_reusable_manifest is True
    assert resumed.pending_source_hashes == frozenset({'new'})
    resumed.append(month='2020-02', row=row(datetime(2020, 2, 3, tzinfo=UTC), 'KRX:000002'))
    manifest = resumed.close()

    assert manifest['months'] == ['2020-01', '2020-02']

    correction = StreamingSilverWriter(
        root,
        table=SilverTable.DAILY_MARKET,
        batch_size=1,
        source_hashes=('old', 'correction'),
        source_paths=('krx:daily-market:2020-01-02', 'krx:daily-market:2020-01-03'),
    )
    assert correction.has_reusable_manifest is False
    assert correction.pending_source_hashes == frozenset({'old', 'correction'})


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
    def preserve_order(*, table: SilverTable, receipts: tuple[BronzeReceipt, ...]) -> tuple[BronzeReceipt, ...]:
        calls.append(table)
        return receipts
    monkeypatch.setattr(mod, 'discover_verified_bronze_receipts', lambda **_kwargs: grouped)
    monkeypatch.setattr(mod, 'order_streaming_receipts', preserve_order)
    monkeypatch.setattr(
        mod, 'normalize_lifecycle_events',
        lambda **_kwargs: (_ for _ in ()).throw(PITDataError('lifecycle-wiring-stop')),
    )

    with pytest.raises(PITDataError, match='lifecycle-wiring-stop'):
        mod.stream_normalize_stock_evidence(bronze_root=tmp_path / 'bronze', silver_root=tmp_path / 'silver', artifact_root=tmp_path / 'artifacts', decision_time=stamp)
    assert calls == [SilverTable.DAILY_MARKET, SilverTable.SECURITY_MASTER]


import polars as pl

from src.core.pit import SilverTable
from src.data.streaming_normalization import normalize_lifecycle_events

def test_stream_normalize_stock_evidence_materializes_empty_or_unresolved_lifecycle_table():
    frame = normalize_lifecycle_events(receipts=(), calendar=__import__('src.core.time', fromlist=['SessionCalendar']).SessionCalendar(()))
    assert frame.schema['instrument_id'] == pl.String
    assert 'resolution_kind' in frame.columns
    assert SilverTable.LIFECYCLE_EVENTS.value == 'lifecycle_events'


def test_normalize_lifecycle_events_preserves_verified_exchange_over_generic_unresolved(tmp_path) -> None:
    from datetime import datetime
    import json
    import polars as pl
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.streaming_normalization import normalize_lifecycle_events

    common = {'lifecycle_event_id': 'evt-003450', 'instrument_id': 'KRX:003450', 'source_security_id': 'KR7003450004', 'delisting_date': '2016-11-01', 'available_at': '2016-10-20T09:00:00+09:00'}
    generic = tmp_path / 'generic.json'
    generic.write_text(json.dumps({**common, 'evidence_status': 'unresolved', 'resolution_kind': 'unresolved'}), encoding='utf-8')
    verified = tmp_path / 'verified.json'
    verified.write_text(json.dumps({**common, 'evidence_status': 'verified', 'resolution_kind': 'merger_or_exchange', 'successor_delivery_date': '2016-11-02', 'successor_allocations_json': '[{"successor_security_id":"KR7105560007","successor_instrument_id":"KRX:105560","ratio":"0.1907312","cost_basis_weight":"1"}]', 'source_provider': 'kind', 'document_receipt_no': 'kind-1', 'document_sha256': 'a' * 64}), encoding='utf-8')
    stamp = datetime(2016, 10, 21, tzinfo=KRX_TZ)
    receipts = (BronzeReceipt(EvidenceKind.LIFECYCLE_EVENTS, '0' * 64, 'generic', stamp, stamp, generic, generic), BronzeReceipt(EvidenceKind.LIFECYCLE_EVENTS, '1' * 64, 'verified', stamp, stamp, verified, verified))
    frame = normalize_lifecycle_events(receipts=receipts, calendar=SessionCalendar((stamp,)))
    assert frame.height == 1
    assert frame.item(0, 'resolution_kind') == 'merger_or_exchange'
    assert frame.item(0, 'successor_delivery_date').isoformat() == '2016-11-02'
    assert 'document_sha256' in frame.columns
    assert frame.schema['successor_allocations_json'] == pl.String


def _flow_receipt(tmp_path, name, source_path, payload_bytes, retrieved_at):
    from src.data.schemas import BronzeReceipt, EvidenceKind

    payload_path = tmp_path / f"{name}.payload.json"
    metadata_path = tmp_path / f"{name}.receipt.json"
    payload_path.write_bytes(payload_bytes)
    metadata_path.write_bytes(b"{}")
    return BronzeReceipt(
        kind=EvidenceKind.INVESTOR_FLOW,
        content_hash=name.ljust(64, "0"),
        source_path=source_path,
        retrieved_at=retrieved_at,
        ingested_at=retrieved_at,
        payload_path=payload_path,
        metadata_path=metadata_path,
    )


def test_dedupe_duplicate_source_paths_keeps_unique_pages_untouched(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    first = _flow_receipt(tmp_path, "a", "LS:frgr-itt:000020:20161229", b"{}", moment)
    second = _flow_receipt(tmp_path, "b", "KIWOOM:ka10059:000020:20161229", b"{}", moment)
    assert _dedupe_duplicate_source_paths((first, second)) == (first, second)


def test_dedupe_duplicate_source_paths_filters_shadowed_warmup_pages(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    normalized = _flow_receipt(
        tmp_path, "n", "normalized-provider-page:daily_market",
        b'{"session": "2024-01-03", "records": []}', moment,
    )
    shadowed = _flow_receipt(tmp_path, "w1", "KRX:warmup:20240105", b"{}", moment)
    earlier = _flow_receipt(tmp_path, "w2", "KRX:warmup:20240101", b"{}", moment)
    result = _dedupe_duplicate_source_paths((normalized, shadowed, earlier))
    assert [item.source_path for item in result] == [
        "normalized-provider-page:daily_market", "KRX:warmup:20240101",
    ]


def test_dedupe_duplicate_source_paths_keeps_widest_retry_page(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    narrow = _flow_receipt(
        tmp_path, "narrow", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-02"}]}', moment,
    )
    wide = _flow_receipt(
        tmp_path, "wide", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-01"}, {"session": "2024-01-03"}]}', moment,
    )
    assert _dedupe_duplicate_source_paths((narrow, wide)) == (wide,)


def test_dedupe_duplicate_source_paths_breaks_ties_by_earliest_retrieval(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    early = _flow_receipt(
        tmp_path, "early", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-02", "note": "a"}]}',
        datetime(2020, 1, 1, tzinfo=UTC),
    )
    late = _flow_receipt(
        tmp_path, "late", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-02", "note": "b"}]}',
        datetime(2020, 1, 2, tzinfo=UTC),
    )
    assert _dedupe_duplicate_source_paths((late, early)) == (early,)


def test_dedupe_duplicate_source_paths_reports_malformed_retry_page(tmp_path) -> None:
    from datetime import UTC, datetime

    import pytest

    from src.data.schemas import PITDataError
    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    good = _flow_receipt(
        tmp_path, "good", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-02"}]}', moment,
    )
    broken = _flow_receipt(tmp_path, "broken", "KIS:flow:20240102", b"{}", moment)
    broken.payload_path.write_bytes(b"not json{")
    with pytest.raises(PITDataError, match="malformed"):
        _dedupe_duplicate_source_paths((good, broken))


def test_source_path_month_helpers_classify_append_only_paths() -> None:
    from src.data.streaming_normalization import _is_append_only_source_path, _source_path_month

    assert _source_path_month("KRX:warmup:20240105") == "2024-01"
    assert _source_path_month("no-date-here") is None
    assert _is_append_only_source_path("KRX:warmup:20240105", "2023-12") is True
    assert _is_append_only_source_path("KRX:warmup:20240105", "2024-01") is False
    assert _is_append_only_source_path("no-date-here", "2024-01") is False


def test_dedupe_duplicate_source_paths_handles_mixed_unique_and_duplicate_keys(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.data.streaming_normalization import _dedupe_duplicate_source_paths

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    solo = _flow_receipt(tmp_path, "solo", "LS:frgr-itt:000020:20161229", b"{}", moment)
    narrow = _flow_receipt(
        tmp_path, "narrow2", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-02"}, {"note": "no-session"}, {"session": "not-a-date"}]}',
        moment,
    )
    wide = _flow_receipt(
        tmp_path, "wide2", "KIS:flow:20240102",
        b'{"records": [{"session": "2024-01-01"}, {"session": "2024-01-03"}]}', moment,
    )
    assert _dedupe_duplicate_source_paths((solo, narrow, wide)) == (solo, wide)


def test_master_available_at_prefers_retained_snapshot_date(tmp_path) -> None:
    from datetime import UTC, datetime

    from src.core.time import KRX_TZ
    from src.data.streaming_normalization import _master_available_at

    moment = datetime(2020, 1, 1, tzinfo=UTC)
    receipt = _flow_receipt(tmp_path, "m", "retained/master_20240102.json", b"{}", moment)
    available = _master_available_at(receipt=receipt, record={"available_time": "2099-01-01"})
    assert available.astimezone(KRX_TZ).date().isoformat() == "2024-01-02"


def test_normalize_lifecycle_events_returns_empty_when_no_delisting_date(tmp_path) -> None:
    from datetime import datetime
    import json

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.schemas import BronzeReceipt, EvidenceKind
    from src.data.streaming_normalization import normalize_lifecycle_events

    stamp = datetime(2016, 10, 21, tzinfo=KRX_TZ)
    path = tmp_path / "nodate.json"
    path.write_text(json.dumps({
        "lifecycle_event_id": "evt-x", "instrument_id": "KRX:000001",
        "source_security_id": "KR7000010000", "available_at": "2016-10-20T09:00:00+09:00",
        "evidence_status": "unresolved", "resolution_kind": "unresolved",
    }), encoding="utf-8")
    receipts = (BronzeReceipt(EvidenceKind.LIFECYCLE_EVENTS, "2" * 64, "nodate", stamp, stamp, path, path),)
    assert normalize_lifecycle_events(receipts=receipts, calendar=SessionCalendar((stamp,))).height == 0


def test_stream_table_worker_materializes_daily_table_in_process(tmp_path) -> None:
    import queue
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable
    from src.data.streaming_normalization import _stream_table_worker

    stamp = datetime(2024, 6, 1, tzinfo=UTC)
    payload_path = tmp_path / "daily.payload.json"
    payload_path.write_bytes(
        b'{"records": [{"session": "2024-01-02", "ticker": "000020", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0, "trading_value": 100500.0, "market_cap": 1000000.0, "shares_outstanding": 10000.0}]}'
    )
    receipt = BronzeReceipt(
        EvidenceKind.DAILY_MARKET, "d" * 64, "KRX:daily-market:2024-01-02",
        stamp, stamp, payload_path, tmp_path / "daily.receipt.json",
    )
    results: queue.Queue = queue.Queue()
    _stream_table_worker(
        table=SilverTable.DAILY_MARKET, receipts=[receipt],
        staging_root=tmp_path / "staging", decision_time=stamp,
        batch_size=10, result_queue=results,
    )
    result = results.get_nowait()
    assert result["ok"] is True
    assert result["count"] == 1


def test_stream_table_worker_materializes_master_table_in_process(tmp_path) -> None:
    import queue
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable
    from src.data.streaming_normalization import _stream_table_worker

    stamp = datetime(2024, 6, 1, tzinfo=UTC)
    payload_path = tmp_path / "master.payload.json"
    payload_path.write_bytes(
        b'{"records": [{"ticker": "000020", "market": "KOSPI", "listing_date": "2024-01-02"}]}'
    )
    receipt = BronzeReceipt(
        EvidenceKind.SECURITY_MASTER, "e" * 64, "KRX:historical-master:2024-01-02",
        stamp, stamp, payload_path, tmp_path / "master.receipt.json",
    )
    results: queue.Queue = queue.Queue()
    _stream_table_worker(
        table=SilverTable.SECURITY_MASTER, receipts=[receipt],
        staging_root=tmp_path / "staging", decision_time=stamp,
        batch_size=10, result_queue=results,
    )
    result = results.get_nowait()
    assert result["ok"] is True
    assert result["count"] == 1


def _mocked_stream_grouped(tmp_path, stamp):
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind

    labels = {
        EvidenceKind.CALENDAR: "calendar:2016-01-04",
        EvidenceKind.SECURITY_MASTER: "KRX:historical-master:2016-01-04",
        EvidenceKind.DAILY_MARKET: "krx:daily-market:2016-01-04",
        EvidenceKind.INVESTOR_FLOW: "LS:frgr-itt:000020:20160104",
        EvidenceKind.FINANCIAL_FACTS: "opendart:fnltt:2016-01-04",
        EvidenceKind.CORPORATE_ACTIONS: "opendart:fricDecsn:2016-01-04",
        EvidenceKind.DISCLOSURES: "opendart:list:2016-01-04",
        EvidenceKind.HISTORICAL_COSTS: "retained:costs:2016-01-04",
        EvidenceKind.LIFECYCLE_EVENTS: "dart:lifecycle:2016-01-04",
    }
    common_envelope = {
        "lifecycle_event_id": "evt-003450", "instrument_id": "KRX:003450",
        "source_security_id": "KR7003450004", "delisting_date": "2016-11-01",
        "available_at": "2016-10-20T09:00:00+09:00",
    }
    generic_envelope = {
        **common_envelope, "evidence_status": "unresolved", "resolution_kind": "unresolved",
    }
    verified_envelope = {
        **common_envelope, "evidence_status": "verified", "resolution_kind": "merger_or_exchange",
        "successor_delivery_date": "2016-11-02",
        "successor_allocations_json": '[{"successor_security_id":"KR7105560007","successor_instrument_id":"KRX:105560","ratio":"0.1907312","cost_basis_weight":"1"}]',
        "source_provider": "kind", "document_receipt_no": "kind-1",
        "document_sha256": "a" * 64,
    }
    grouped = {}
    for index, kind in enumerate(EvidenceKind):
        payload = tmp_path / f"full-{kind.value}.json"
        payload.write_bytes(b'{"records": []}')
        grouped[kind] = (BronzeReceipt(
            kind, f"{index:064x}", labels[kind], stamp, stamp,
            payload, tmp_path / f"full-{kind.value}.receipt.json",
        ),)
    import json as _json

    (tmp_path / "full-lifecycle_events.generic.json").write_bytes(
        _json.dumps(generic_envelope).encode("utf-8")
    )
    (tmp_path / "full-lifecycle_events.verified.json").write_bytes(
        _json.dumps(verified_envelope).encode("utf-8")
    )
    _stamp = datetime(2026, 9, 6, tzinfo=UTC)
    grouped[EvidenceKind.LIFECYCLE_EVENTS] = (
        BronzeReceipt(
            EvidenceKind.LIFECYCLE_EVENTS, "e0" * 32, "dart:lifecycle:2016-11-01",
            _stamp, _stamp, tmp_path / "full-lifecycle_events.generic.json",
            tmp_path / "full-lifecycle_events.generic.json",
        ),
        BronzeReceipt(
            EvidenceKind.LIFECYCLE_EVENTS, "e1" * 32, "dart:lifecycle:2016-11-01",
            _stamp, _stamp, tmp_path / "full-lifecycle_events.verified.json",
            tmp_path / "full-lifecycle_events.verified.json",
        ),
    )
    return grouped


def test_stream_normalize_publishes_small_tables_idempotently(tmp_path, monkeypatch) -> None:
    import json
    from datetime import UTC, datetime

    import src.data.normalization as normalization_module
    import src.data.silver as silver_module
    import src.data.streaming_normalization as streaming
    from src.data.schemas import SilverTable
    from src.data.silver import SilverStore, complete_minimal_fixture
    from src.data.streaming_normalization import _frame_months, stream_normalize_stock_evidence

    decision = datetime(2026, 9, 6, tzinfo=UTC)
    fixture_tables, _receipts, report = complete_minimal_fixture(decision_time=decision)
    small_tables = {
        table: frame for table, frame in fixture_tables.items()
        if table not in streaming._STREAM_TABLES
        and table is not SilverTable.CORPORATE_ACTIONS
        and table is not SilverTable.LIFECYCLE_EVENTS
        and frame.height > 0
    }
    assert SilverTable.CALENDAR in small_tables
    month = next(iter(_frame_months(small_tables[SilverTable.CALENDAR], "session").keys()))
    manifest = {"months": [month], "row_counts": {month: 1}, "parts": {}, "root_hash": "r" * 64}
    grouped = _mocked_stream_grouped(tmp_path, decision)
    action_hash = grouped[streaming.EvidenceKind.CORPORATE_ACTIONS][0].content_hash
    cache_path = tmp_path / "artifacts" / "corporate_actions_stream.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"source_hashes": [action_hash], "records": []}), encoding="utf-8")

    monkeypatch.setattr(streaming, "discover_verified_bronze_receipts", lambda **_kwargs: grouped)
    monkeypatch.setattr(streaming, "select_streaming_receipts", lambda *, kind, receipts: receipts)
    monkeypatch.setattr(streaming, "order_streaming_receipts", lambda *, table, receipts: receipts)
    monkeypatch.setattr(
        streaming, "_stream_table_isolated",
        lambda **_kwargs: {"count": 1, "manifest": dict(manifest)},
    )
    monkeypatch.setattr(
        normalization_module, "normalize_stock_evidence",
        lambda _receipts, **_kwargs: (dict(small_tables), report),
    )
    monkeypatch.setattr(
        SilverStore, "publish_streamed_table",
        lambda self, **_kwargs: tmp_path / "published",
    )

    first = stream_normalize_stock_evidence(
        bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts", decision_time=decision, batch_size=10,
    )
    assert first.report_hash == report.report_hash
    second = stream_normalize_stock_evidence(
        bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts", decision_time=decision, batch_size=10,
    )
    assert second.report_hash == report.report_hash
    manifest_path = next((tmp_path / "silver" / "calendar").rglob("dataset_manifest.json"))
    manifest_path.unlink()
    third = stream_normalize_stock_evidence(
        bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver",
        artifact_root=tmp_path / "artifacts", decision_time=decision, batch_size=10,
    )
    assert third.report_hash == report.report_hash
    assert silver_module.SilverStore is SilverStore


def test_stream_table_worker_skips_already_staged_receipts(tmp_path) -> None:
    import queue
    from datetime import UTC, datetime

    from src.data.schemas import BronzeReceipt, EvidenceKind, SilverTable
    from src.data.streaming_normalization import _stream_table_worker

    stamp = datetime(2024, 6, 1, tzinfo=UTC)

    def daily_receipt(name, session, close, retrieved):
        import json as _json

        payload_path = tmp_path / f"{name}.payload.json"
        payload_path.write_bytes(_json.dumps({"records": [{
            "session": session, "ticker": "000020", "open": 100.0,
            "high": 101.0, "low": 99.0, "close": float(close),
            "volume": 1000.0, "trading_value": 100500.0,
            "market_cap": 1000000.0, "shares_outstanding": 10000.0,
        }]}).encode("utf-8"))
        return BronzeReceipt(
            EvidenceKind.DAILY_MARKET, name.ljust(64, "0"),
            f"KRX:daily-market:{session}", retrieved, retrieved,
            payload_path, tmp_path / f"{name}.receipt.json",
        )

    january = daily_receipt("jan31", "2024-01-02", "100.5", stamp)
    february = daily_receipt("feb31", "2024-02-01", "101.5", stamp)
    staging_root = tmp_path / "staging"
    first_results: queue.Queue = queue.Queue()
    _stream_table_worker(
        table=SilverTable.DAILY_MARKET, receipts=[january],
        staging_root=staging_root, decision_time=stamp,
        batch_size=10, result_queue=first_results,
    )
    assert first_results.get_nowait()["ok"] is True
    # Corrupt the staged receipt's Bronze payload: the resume must skip it
    # without reading, so only the new month is processed.
    january.payload_path.write_bytes(b"corrupted{")
    second_results: queue.Queue = queue.Queue()
    _stream_table_worker(
        table=SilverTable.DAILY_MARKET, receipts=[january, february],
        staging_root=staging_root, decision_time=stamp,
        batch_size=10, result_queue=second_results,
    )
    resumed = second_results.get_nowait()
    assert resumed["ok"] is True
    assert resumed["count"] == 2
