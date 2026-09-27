"""KRX session calendar and point-in-time ordering invariants."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.core.time import KRX_DAILY, KRX_TZ, PointInTime, SessionCalendar, TemporalViolationError

class TestPointInTime:
    def test_valid_ordering_accepted(self) -> None:
        pit = PointInTime(
            observation_time=datetime(2024, 1, 2, 14, 0, tzinfo=KRX_TZ),
            available_time=datetime(2024, 1, 2, 15, 30, tzinfo=KRX_TZ),
            decision_time=datetime(2024, 1, 3, 8, 50, tzinfo=KRX_TZ),
            execution_time=datetime(2024, 1, 3, 9, 5, tzinfo=KRX_TZ),
        )
        assert pit.available_time <= pit.decision_time

    def test_available_after_decision_is_rejected(self) -> None:
        with pytest.raises(TemporalViolationError):
            PointInTime(
                observation_time=datetime(2024, 1, 2, 14, 0, tzinfo=KRX_TZ),
                available_time=datetime(2024, 1, 3, 8, 50, tzinfo=KRX_TZ),
                decision_time=datetime(2024, 1, 3, 8, 0, tzinfo=KRX_TZ),
                execution_time=datetime(2024, 1, 3, 9, 5, tzinfo=KRX_TZ),
            )


class TestSession:
    def test_krx_session_hours(self) -> None:
        assert KRX_DAILY.open_time.hour == 9
        assert KRX_DAILY.close_time.hour == 15

    def test_calendar_requires_monotonic_sessions(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            SessionCalendar(
                sessions=(
                    datetime(2024, 1, 2, tzinfo=KRX_TZ),
                    datetime(2024, 1, 2, tzinfo=KRX_TZ),
                )
            )

    def test_calendar_sessions_between(self) -> None:
        cal = SessionCalendar(
            sessions=(
                datetime(2024, 1, 2, tzinfo=KRX_TZ),
                datetime(2024, 1, 3, tzinfo=KRX_TZ),
                datetime(2024, 1, 4, tzinfo=KRX_TZ),
                datetime(2024, 1, 5, tzinfo=KRX_TZ),
            )
        )
        got = cal.sessions_between(
            datetime(2024, 1, 3, tzinfo=KRX_TZ), datetime(2024, 1, 4, tzinfo=KRX_TZ)
        )
        assert len(got) == 1


def test_session_calendar_advance_skips_non_sessions() -> None:
    from datetime import datetime
    import pytest
    from src.core.time import KRX_TZ, SessionCalendar

    friday = datetime(2024, 1, 5, tzinfo=KRX_TZ)
    monday = datetime(2024, 1, 8, tzinfo=KRX_TZ)
    tuesday = datetime(2024, 1, 9, tzinfo=KRX_TZ)
    calendar = SessionCalendar((friday, monday, tuesday))

    assert calendar.advance(datetime(2024, 1, 5, 10, tzinfo=KRX_TZ), 2) == tuesday
    with pytest.raises(ValueError, match="session"):
        calendar.advance(datetime(2024, 1, 6, 10, tzinfo=KRX_TZ), 1)
    with pytest.raises(ValueError, match="coverage"):
        calendar.advance(monday, 2)
