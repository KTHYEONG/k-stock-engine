def test_normalize_financial_fact_uses_next_krx_session_when_intraday_unknown() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.silver import next_krx_session_open

    published = datetime(2024, 1, 2, 18, 0, tzinfo=KRX_TZ)
    calendar = SessionCalendar((datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ), datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)))

    assert next_krx_session_open(published, calendar) == datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)


def canonical_page(*, filing_id: str, fiscal_period: str, published_at, value: float) -> dict:
    return {
        "company_id": "001",
        "fiscal_period": fiscal_period,
        "filing_id": filing_id,
        "fact": "sales",
        "published_at": published_at,
        "available_at": published_at,
        "value": value,
        "unit": "KRW",
        "consolidated": True,
        "restatement_id": "r0",
        "source_kind": "legacy_document",
        "mapping_version": "v1",
        "raw_document_hash": "d" * 64,
    }


def test_normalize_dart_facts_preserves_all_filings_and_pit_excludes_later_correction() -> None:
    from datetime import UTC, datetime
    from src.core.time import SessionCalendar
    from src.data.normalization import normalize_dart_financial_facts

    cutoff = datetime(2016, 2, 1, 9, tzinfo=UTC)
    pages = (canonical_page(filing_id="Q1", fiscal_period="2015Q1", published_at=datetime(2015, 5, 15, tzinfo=UTC), value=10.0), canonical_page(filing_id="Q2", fiscal_period="2015Q2", published_at=datetime(2015, 8, 17, tzinfo=UTC), value=20.0), canonical_page(filing_id="CORR", fiscal_period="2015Q1", published_at=datetime(2016, 3, 30, tzinfo=UTC), value=99.0))
    calendar = SessionCalendar((datetime(2015, 5, 18, 9, tzinfo=UTC), datetime(2015, 8, 18, 9, tzinfo=UTC), datetime(2016, 3, 31, 9, tzinfo=UTC)))
    frame = normalize_dart_financial_facts(pages=pages, disclosure_rows=(), source_hash="c" * 64, calendar=calendar, decision_time=cutoff)
    assert frame.height == 2
    assert set(frame["filing_id"].to_list()) == {"Q1", "Q2"}
    assert set(frame["source_kind"].to_list()) == {"legacy_document"}
