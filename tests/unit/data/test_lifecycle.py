def test_derive_lifecycle_candidates_requires_consecutive_complete_krx_sessions() -> None:
    from datetime import datetime
    import polars as pl
    import pytest
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import derive_lifecycle_candidates
    from src.data.schemas import PITDataError

    first = datetime(2016, 5, 18, tzinfo=KRX_TZ)
    second = datetime(2016, 5, 19, tzinfo=KRX_TZ)
    partial = pl.DataFrame({'instrument_id': ['KRX:008020'], 'valid_from': [first], 'available_at': [first], 'source_hash': ['a']})
    with pytest.raises(PITDataError, match='consecutive'):
        derive_lifecycle_candidates(security_master=partial, calendar=SessionCalendar((first, second)))
    full = pl.concat([partial, pl.DataFrame({'instrument_id': ['KRX:005930'], 'valid_from': [second], 'available_at': [second], 'source_hash': ['b']})])
    out = derive_lifecycle_candidates(security_master=full, calendar=SessionCalendar((first, second)))
    assert len(out) == 1
    assert out[0].instrument_id == 'KRX:008020'
    assert out[0].last_tradable_session == first
    assert out[0].first_absent_session == second


def test_parse_kind_lifecycle_notice_rejects_candidate_date_mismatch() -> None:
    from datetime import UTC, datetime
    import pytest
    from src.core.time import KRX_TZ
    from src.data.lifecycle import LifecycleCandidate, parse_kind_lifecycle_notice
    from src.data.schemas import PITDataError

    candidate = LifecycleCandidate('KRX:008020', '008020', datetime(2016,5,18,tzinfo=KRX_TZ), datetime(2016,5,19,tzinfo=KRX_TZ), ('a','b'))
    html = '<h1>상장폐지결정 및 정리매매</h1><p>결정(확인)일자 2016-05-02</p><p>2016년 5월 10일부터 2016년 5월 18일까지 정리매매</p><p>2016년 5월 19일자로 상장폐지</p>'
    out = parse_kind_lifecycle_notice(candidate=candidate, disclosure_url='https://kind.krx.co.kr/a', html=html, retrieved_at=datetime(2026,9,10,tzinfo=UTC), source_hash='k')
    assert out['evidence_status'] == 'verified'
    assert out['cleanup_end'] == candidate.last_tradable_session
    with pytest.raises(PITDataError, match='does not match'):
        parse_kind_lifecycle_notice(candidate=candidate, disclosure_url='https://kind.krx.co.kr/a', html=html.replace('5월 19일자로', '5월 20일자로'), retrieved_at=datetime(2026,9,10,tzinfo=UTC), source_hash='k2')


def test_parse_kind_lifecycle_notice_accepts_official_dotted_schedule() -> None:
    from datetime import UTC, datetime
    from src.core.time import KRX_TZ
    from src.data.lifecycle import LifecycleCandidate, parse_kind_lifecycle_notice

    candidate = LifecycleCandidate(
        'KRX:019300', '019300', datetime(2016, 8, 10, tzinfo=KRX_TZ),
        datetime(2016, 8, 11, tzinfo=KRX_TZ), ('h',),
    )
    html = (
        '결정(확인)일자 2016-07-27. '
        '2016.08.02 ~ 2016.08.10 정리매매. '
        '2016.08.11자로 상장폐지. 주당 3,600원.'
    )
    out = parse_kind_lifecycle_notice(
        candidate=candidate,
        disclosure_url='https://kind.krx.co.kr/a',
        html=html,
        retrieved_at=datetime(2026, 9, 10, tzinfo=UTC),
        source_hash='dotted',
    )
    assert out['evidence_status'] == 'verified'
    assert out['resolution_kind'] == 'cash_settlement'
    assert out['cash_settlement_per_share'] == 3600.0
    colon = parse_kind_lifecycle_notice(
        candidate=candidate,
        disclosure_url='https://kind.krx.co.kr/a',
        html=html.replace('2016.08.11자로', '상장폐지일자: 2016.08.11'),
        retrieved_at=datetime(2026, 9, 10, tzinfo=UTC),
        source_hash='dotted-colon',
    )
    assert colon['delisting_date'].isoformat() == '2016-08-11'


from datetime import datetime

from src.core.time import KRX_TZ, SessionCalendar
from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind, parse_dart_lifecycle_notice

def test_parse_dart_lifecycle_notice_accepts_real_dotted_dates_without_body_ticker():
    sessions = (datetime(2016, 5, 3, 9, tzinfo=KRX_TZ), datetime(2016, 5, 10, 9, tzinfo=KRX_TZ), datetime(2016, 5, 18, 9, tzinfo=KRX_TZ), datetime(2016, 5, 19, 9, tzinfo=KRX_TZ))
    candidate = LifecycleCandidate('KRX:008020', '008020', sessions[2], sessions[3], ('master-hash',))
    evidence = parse_dart_lifecycle_notice(candidate=candidate, receipt_no='20160502800560', disclosure_url='https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20160502800560', published_at=datetime(2016, 5, 2, tzinfo=KRX_TZ), archive='정리매매 기간은 2016.05.10.부터 2016.05.18.까지이며 2016.05.19.자로 상장폐지. 주당 10,200원. 참조 999999'.encode(), calendar=SessionCalendar(sessions), source_hash='archive-hash')
    assert evidence.resolution_kind is LifecycleResolutionKind.CASH_SETTLEMENT
    assert evidence.cleanup_start == sessions[1]
    assert evidence.cleanup_end == sessions[2]
    assert evidence.available_at == sessions[0]
    assert evidence.cash_settlement_per_share == 10200.0


def test_parse_dart_lifecycle_notice_unpriced_cleanup_is_unsettled():
    from datetime import datetime
    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.lifecycle import LifecycleCandidate, LifecycleResolutionKind, parse_dart_lifecycle_notice
    sessions = (datetime(2016, 8, 25, 9, tzinfo=KRX_TZ), datetime(2016, 9, 5, 9, tzinfo=KRX_TZ), datetime(2016, 9, 2, 9, tzinfo=KRX_TZ))
    ordered = tuple(sorted(sessions))
    candidate = LifecycleCandidate("KRX:074150", "074150", datetime(2016, 9, 2, 9, tzinfo=KRX_TZ), datetime(2016, 9, 5, 9, tzinfo=KRX_TZ), ("h",))
    evidence = parse_dart_lifecycle_notice(candidate=candidate, receipt_no="20160823900419", disclosure_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20160823900419", published_at=datetime(2016, 8, 24, tzinfo=KRX_TZ), archive="정리매매 기간은 2016.08.25.부터 2016.09.02.까지이며 2016.09.05.자로 상장폐지.".encode(), calendar=SessionCalendar(ordered), source_hash="s")
    assert evidence.resolution_kind is LifecycleResolutionKind.UNSETTLED_DELISTING
    assert evidence.cash_settlement_per_share is None


def test_parse_dart_lifecycle_notice_rejects_tilde_date_mismatch_and_late_receipt():
    import pytest
    from src.data.schemas import PITDataError
    sessions = tuple(datetime(2016, 5, day, 9, tzinfo=KRX_TZ) for day in (10, 11, 18, 19))
    candidate = LifecycleCandidate("KRX:008020", "008020", sessions[2], sessions[3], ("h",))
    body = "정리매매기간(2016.05.10~2016.05.18) 후 상장폐지일: 2016.05.19"
    parsed = parse_dart_lifecycle_notice(candidate=candidate, receipt_no="20160502800560", disclosure_url="", published_at=datetime(2016, 5, 2, tzinfo=KRX_TZ), archive=body.encode(), calendar=SessionCalendar(sessions), source_hash="x")
    assert parsed.cleanup_start == sessions[0]
    mismatched = LifecycleCandidate("KRX:008020", "008020", sessions[1], sessions[3], ("h",))
    with pytest.raises(PITDataError, match="last tradable"):
        parse_dart_lifecycle_notice(candidate=mismatched, receipt_no="20160502800560", disclosure_url="", published_at=datetime(2016, 5, 2, tzinfo=KRX_TZ), archive=body.encode(), calendar=SessionCalendar(sessions), source_hash="x")
    delisting_mismatch = LifecycleCandidate("KRX:008020", "008020", sessions[2], datetime(2016, 5, 20, 9, tzinfo=KRX_TZ), ("h",))
    with pytest.raises(PITDataError, match="first absent"):
        parse_dart_lifecycle_notice(candidate=delisting_mismatch, receipt_no="20160502800560", disclosure_url="", published_at=datetime(2016, 5, 2, tzinfo=KRX_TZ), archive=body.encode(), calendar=SessionCalendar(sessions), source_hash="x")
    with pytest.raises(PITDataError, match="available"):
        parse_dart_lifecycle_notice(candidate=candidate, receipt_no="20160502800560", disclosure_url="", published_at=datetime(2016, 5, 10, 12, tzinfo=KRX_TZ), archive=body.encode(), calendar=SessionCalendar(sessions), source_hash="x")


def test_canonicalize_lifecycle_rows_rejects_conflicting_verified_versions() -> None:
    import pytest

    from src.data.lifecycle import canonicalize_lifecycle_event_rows
    from src.data.schemas import PITDataError

    base = {
        'lifecycle_event_id': 'evt-003450',
        'instrument_id': 'KRX:003450',
        'evidence_status': 'verified',
        'resolution_kind': 'merger_or_exchange',
        'source_security_id': 'KR7003450004',
        'successor_allocations_json': '[{"successor_security_id":"KR7105560007"}]',
        'successor_delivery_date': '2016-11-02',
        'delisting_date': '2016-11-01',
    }
    with pytest.raises(PITDataError, match='conflicting'):
        canonicalize_lifecycle_event_rows((base, {**base, 'successor_delivery_date': '2016-11-03'}))
