"""KRX trading-session calendar sourced from the XKRX exchange calendar."""
from __future__ import annotations

from datetime import date
from typing import Final

import exchange_calendars as xcals

from src.core.time import SessionCalendar


def xkrx_session_calendar(start: date | None = None, end: date | None = None) -> SessionCalendar:
    """Return KRX trading sessions from the ``exchange_calendars`` XKRX calendar.

    The repository has no KRX holiday endpoint. XKRX was verified session-for-session
    against the scope's certified 2019-2025 calendar (1719/1719), so it is the
    session source for availability math that must extend past the last
    collected market day (e.g. filings received after the final Bronze session).

    Each session instant is that session's official open as reported by the
    calendar (late-open days keep their true open), timezone-aware.

    Args:
        start: First date to include; ``None`` means the calendar's first session.
        end: Last date to include; ``None`` means the calendar's last session.

    Returns:
        Strictly increasing session opens.

    Raises:
        ValueError: ``start`` is after ``end``, a bound lies outside the
            calendar's supported range, or the range contains no session.
    """
    calendar = xcals.get_calendar("XKRX")
    labels = list(calendar.sessions)
    first_day = labels[0].date()
    last_day = labels[-1].date()
    if start is not None and end is not None and start > end:
        raise ValueError(f"start {start} is after end {end}")
    if start is not None and (start < first_day or start > last_day):
        raise ValueError(f"start {start} lies outside the XKRX range {first_day}..{last_day}")
    if end is not None and (end < first_day or end > last_day):
        raise ValueError(f"end {end} lies outside the XKRX range {first_day}..{last_day}")
    lo = first_day if start is None else start
    hi = last_day if end is None else end
    return _session_opens(calendar, lo, hi)


def _session_opens(calendar: xcals.ExchangeCalendar, lo: date, hi: date) -> SessionCalendar:
    opens: list[object] = []
    for label in calendar.sessions:
        day = label.date()
        if lo <= day <= hi:
            instant = calendar.session_open(label).to_pydatetime()
            if instant.tzinfo is None:  # pragma: no cover - upstream always aware
                raise ValueError(f"XKRX session open is naive for {day}")
            opens.append(instant)
    if not opens:
        raise ValueError(f"no XKRX session in range {lo}..{hi}")
    return SessionCalendar(tuple(sorted(opens)))  # type: ignore[arg-type]


# 고정 시작일: 라이브러리 기본 범위(오늘 기준 전후 롤링)를 쓰면 같은 입력이라도 날마다 달력이 달라진다.
_STABLE_CALENDAR_START: Final = date(2000, 1, 1)


def xkrx_calendar_through(day: date) -> SessionCalendar:
    """Return XKRX sessions from 2000-01-01 through the end of the year after ``day``.

    Builders record a digest of the calendar they use. ``exchange_calendars``
    defaults to a window rolling with today's date on both ends, so the same
    inputs would get a new dataset identity every day. Both bounds here are
    fixed or year-granular, which keeps identities stable within a year while
    covering every session a decision on ``day`` can reference (next-session
    availability, and pay dates up to the end of the following year).

    Args:
        day: The decision date the build is anchored to (KST calendar date).

    Returns:
        Strictly increasing session opens in ``[2000-01-01, <day.year + 1>-12-31]``.
    """
    horizon = date(day.year + 1, 12, 31)
    calendar = xcals.get_calendar("XKRX", start=_STABLE_CALENDAR_START.isoformat(), end=horizon.isoformat())
    return _session_opens(calendar, _STABLE_CALENDAR_START, horizon)
