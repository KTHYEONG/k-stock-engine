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
