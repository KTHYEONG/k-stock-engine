def test_dart_receipt_is_available_next_krx_session() -> None:
    from datetime import datetime

    from src.core.time import KRX_TZ, SessionCalendar
    from src.data.silver import next_krx_session_open

    friday = datetime(2024, 1, 5, tzinfo=KRX_TZ)
    monday = datetime(2024, 1, 8, tzinfo=KRX_TZ)
    calendar = SessionCalendar((friday, monday))

    assert next_krx_session_open(friday, calendar) == datetime(2024, 1, 8, 9, tzinfo=KRX_TZ)


def test_certification_rejects_duplicate_daily_market_primary_key() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import complete_minimal_fixture, validate_table

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, _ = complete_minimal_fixture(decision_time=decision)
    duplicate = pl.concat([tables[SilverTable.DAILY_MARKET], tables[SilverTable.DAILY_MARKET]])
    with pytest.raises(PITDataError, match=r'duplicate.*daily_market'):
        validate_table(SilverTable.DAILY_MARKET, duplicate, decision_time=decision)


def test_certification_rejects_fact_available_after_decision_time() -> None:
    from datetime import UTC, datetime

    import polars as pl
    import pytest

    from src.data.schemas import PITDataError, SilverTable
    from src.data.silver import complete_minimal_fixture, validate_table

    decision = datetime(2024, 1, 3, tzinfo=UTC)
    tables, _, _ = complete_minimal_fixture(decision_time=decision)
    late = tables[SilverTable.DAILY_MARKET].with_columns(pl.lit(datetime(2024, 1, 4, tzinfo=UTC)).alias('available_at'))
    with pytest.raises(PITDataError, match='available_at'):
        validate_table(SilverTable.DAILY_MARKET, late, decision_time=decision)


def test_certification_rejects_incomplete_required_evidence() -> None:
    from datetime import date

    import pytest

    from src.core.datasets import DatasetCertification
    from src.data.schemas import PITDataError
    from src.data.silver import certify_silver

    with pytest.raises(PITDataError, match=r'investor_flow.*financial_facts'):
        certify_silver({}, receipts={}, coverage_start=date(2024, 1, 2), coverage_end=date(2024, 1, 2), certification=DatasetCertification.RESEARCH)
