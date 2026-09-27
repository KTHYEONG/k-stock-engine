"""Invariant guards for the XKRX session calendar."""


def test_xkrx_calendar_excludes_seollal_holiday() -> None:
    from datetime import date

    from src.core.krx_calendar import xkrx_session_calendar
    from src.core.time import KRX_TZ

    calendar = xkrx_session_calendar(start=date(2018, 2, 14), end=date(2018, 2, 19))

    assert [s.astimezone(KRX_TZ).date().isoformat() for s in calendar.sessions] == [
        "2018-02-14",
        "2018-02-19",
    ]


def test_xkrx_calendar_open_instant_is_aware_kst_open() -> None:
    from datetime import date

    from src.core.krx_calendar import xkrx_session_calendar
    from src.core.time import KRX_TZ

    (session,) = xkrx_session_calendar(start=date(2019, 1, 2), end=date(2019, 1, 2)).sessions

    assert session.tzinfo is not None
    assert session.astimezone(KRX_TZ).date() == date(2019, 1, 2)


def test_xkrx_calendar_rejects_inverted_range() -> None:
    from datetime import date

    import pytest

    from src.core.krx_calendar import xkrx_session_calendar

    with pytest.raises(ValueError, match="after"):
        xkrx_session_calendar(start=date(2019, 1, 3), end=date(2019, 1, 2))


def test_xkrx_calendar_rejects_empty_range() -> None:
    from datetime import date

    import pytest

    from src.core.krx_calendar import xkrx_session_calendar

    with pytest.raises(ValueError, match=r"no .* session"):
        xkrx_session_calendar(start=date(2024, 11, 16), end=date(2024, 11, 17))


def test_xkrx_calendar_rejects_out_of_range_bounds() -> None:
    from datetime import date

    import pytest

    from src.core.krx_calendar import xkrx_session_calendar

    with pytest.raises(ValueError, match="outside"):
        xkrx_session_calendar(start=date(1990, 1, 1), end=date(1990, 1, 10))
    with pytest.raises(ValueError, match="outside"):
        xkrx_session_calendar(start=date(2019, 1, 2), end=date(2100, 1, 1))


def test_xkrx_calendar_through_is_stable_within_a_year_and_covers_next_year() -> None:
    from datetime import date

    from src.core.krx_calendar import xkrx_calendar_through

    early = xkrx_calendar_through(date(2026, 1, 2))
    late = xkrx_calendar_through(date(2026, 12, 30))

    assert early.sessions == late.sessions
    assert early.sessions[0].date() == date(2000, 1, 4)
    assert early.sessions[-1].date().year == 2027
    assert xkrx_calendar_through(date(2027, 1, 4)).sessions[-1].date().year == 2028
