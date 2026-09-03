def test_normalize_financial_fact_uses_next_krx_session_when_intraday_unknown() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.silver import next_krx_session_open

    published = datetime(2024, 1, 2, 18, 0, tzinfo=KRX_TZ)
    calendar = SessionCalendar((datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ), datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)))

    assert next_krx_session_open(published, calendar) == datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ)
