"""SessionIndex bisection invariants."""
from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta, tzinfo

import pytest

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.session_index import SessionIndex


def _random_calendar(rng: random.Random, n: int) -> SessionCalendar:
    base = datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ)
    gaps = [rng.choice([1, 2, 3, 4]) for _ in range(n)]
    sessions: list[datetime] = []
    cursor = base
    for gap in gaps:
        if sessions:
            cursor = sessions[-1] + timedelta(days=gap)
        sessions.append(cursor)
    return SessionCalendar(tuple(sessions))


def test_counts_match_linear_oracle() -> None:
    rng = random.Random(20240102)  # noqa: S311 - deterministic test fixture, not crypto
    for _ in range(25):
        calendar = _random_calendar(rng, rng.randint(2, 12))
        index = SessionIndex.from_calendar(calendar)
        probes: list[date] = [index.dates[0] - timedelta(days=2), index.dates[-1] + timedelta(days=2)]
        for day in index.dates:
            probes.extend([day, day - timedelta(days=1), day + timedelta(days=1)])
        for day in probes:
            assert index.count_on_or_before_date(day) == sum(1 for d in index.dates if d <= day)
        moments = [
            index.sessions[0] - timedelta(hours=1),
            index.sessions[-1] + timedelta(hours=1),
            *index.sessions,
            *(s + timedelta(hours=1) for s in index.sessions),
        ]
        for moment in moments:
            assert index.count_on_or_before_instant(moment) == sum(
                1 for s in index.sessions if s <= moment
            )


def test_duplicate_local_dates_fail_closed() -> None:
    sessions = (
        datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ),
        datetime(2024, 1, 2, 15, 0, tzinfo=KRX_TZ),
    )
    with pytest.raises(PITDataError):
        SessionIndex.from_calendar(SessionCalendar(sessions))


def test_position_of_absent_date() -> None:
    index = SessionIndex.from_calendar(
        SessionCalendar(
            (
                datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ),
                datetime(2024, 1, 4, 9, 0, tzinfo=KRX_TZ),
            )
        )
    )
    assert index.position_of_date(date(2024, 1, 2)) == 0
    assert index.position_of_date(date(2024, 1, 4)) == 1
    with pytest.raises(PITDataError):
        index.position_of_date(date(2024, 1, 3))


def test_naive_instant_rejected() -> None:
    index = SessionIndex.from_calendar(
        SessionCalendar(
            (
                datetime(2024, 1, 2, 9, 0, tzinfo=KRX_TZ),
                datetime(2024, 1, 3, 9, 0, tzinfo=KRX_TZ),
            )
        )
    )
    with pytest.raises(ValueError, match="aware"):
        index.count_on_or_before_instant(datetime(2024, 1, 2, 9, 0))


class _UnawareTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None


@pytest.mark.parametrize("timezone", [None, _UnawareTimezone()])
def test_unaware_session_rejected(timezone: tzinfo | None) -> None:
    calendar = SessionCalendar((datetime(2024, 1, 2, 9, tzinfo=timezone),))
    with pytest.raises(PITDataError, match="aware"):
        SessionIndex.from_calendar(calendar)


def test_unaware_timezone_instant_rejected() -> None:
    index = SessionIndex.from_calendar(SessionCalendar(()))
    with pytest.raises(ValueError, match="aware"):
        index.count_on_or_before_instant(datetime(2024, 1, 2, tzinfo=_UnawareTimezone()))


def test_utc_sessions_use_krx_dates_and_absolute_instants() -> None:
    sessions = (
        datetime(2024, 1, 1, 16, tzinfo=UTC),
        datetime(2024, 1, 2, 16, tzinfo=UTC),
    )
    index = SessionIndex.from_calendar(SessionCalendar(sessions))
    assert index.dates == (date(2024, 1, 2), date(2024, 1, 3))
    assert index.position_of_date(date(2024, 1, 2)) == 0
    assert index.count_on_or_before_date(date(2024, 1, 1)) == 0
    assert index.count_on_or_before_instant(datetime(2024, 1, 2, 1, tzinfo=KRX_TZ)) == 1


def test_empty_index_and_absent_date_boundaries() -> None:
    empty = SessionIndex.from_calendar(SessionCalendar(()))
    assert empty.count_on_or_before_date(date(2024, 1, 2)) == 0
    assert empty.count_on_or_before_instant(datetime(2024, 1, 2, tzinfo=UTC)) == 0
    with pytest.raises(PITDataError, match="no certified session"):
        empty.position_of_date(date(2024, 1, 2))
    index = SessionIndex.from_calendar(
        SessionCalendar((datetime(2024, 1, 2, 9, tzinfo=KRX_TZ),))
    )
    for day in (date(2024, 1, 1), date(2024, 1, 3)):
        with pytest.raises(PITDataError, match="no certified session"):
            index.position_of_date(day)
