"""Precomputed session ordering for morning-build dataset builders."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime

from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar


@dataclass(frozen=True, slots=True)
class SessionIndex:
    """Bisection-ready view of certified session opens."""

    sessions: tuple[datetime, ...]
    dates: tuple[date, ...]

    @classmethod
    def from_calendar(cls, calendar: SessionCalendar) -> SessionIndex:
        """Precompute session instants and their KRX-local dates once."""
        if any(session.utcoffset() is None for session in calendar.sessions):
            raise PITDataError("certified sessions must be aware datetimes")
        dates = tuple(session.astimezone(KRX_TZ).date() for session in calendar.sessions)
        if any(dates[i] >= dates[i + 1] for i in range(len(dates) - 1)):
            raise PITDataError("certified sessions must fall on strictly increasing local dates")
        return cls(sessions=calendar.sessions, dates=dates)

    def count_on_or_before_date(self, day: date) -> int:
        """Number of sessions whose local date is <= day."""
        return bisect_right(self.dates, day)

    def count_on_or_before_instant(self, moment: datetime) -> int:
        """Number of sessions whose instant is <= moment (aware datetime)."""
        if moment.utcoffset() is None:
            raise ValueError("moment must be aware datetime")
        return bisect_right(self.sessions, moment)

    def position_of_date(self, day: date) -> int:
        """Index of the session whose local date equals day; raises PITDataError when absent."""
        position = bisect_left(self.dates, day)
        if position >= len(self.dates) or self.dates[position] != day:
            raise PITDataError(f"no certified session on {day.isoformat()}")
        return position
