"""KRX trading-session calendar sourced from the XKRX exchange calendar."""
from __future__ import annotations

from datetime import date

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
    opens: list[object] = []
    for label in labels:
        day = label.date()
        if lo <= day <= hi:
            instant = calendar.session_open(label).to_pydatetime()
            if instant.tzinfo is None:  # pragma: no cover - upstream always aware
                raise ValueError(f"XKRX session open is naive for {day}")
            opens.append(instant)
    if not opens:
        raise ValueError(f"no XKRX session in range {lo}..{hi}")
    return SessionCalendar(tuple(sorted(opens)))  # type: ignore[arg-type]
