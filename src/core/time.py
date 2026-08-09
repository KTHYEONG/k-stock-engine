"""Market-session and point-in-time contracts.

Every decision has ``observation_time``, ``available_time``, ``decision_time``,
and ``execution_time``. Dataset construction and simulation fail closed on a
temporal violation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, tzinfo
from zoneinfo import ZoneInfo

KRX_TZ = ZoneInfo("Asia/Seoul")


class TemporalViolationError(ValueError):
    """Raised when point-in-time ordering invariants are violated."""


@dataclass(frozen=True, slots=True)
class Session:
    """A named market session with its local open/close wall-clock times."""

    name: str
    open_time: time
    close_time: time
    timezone: tzinfo = KRX_TZ

    def in_session(self, when: datetime) -> bool:
        local = when.astimezone(self.timezone).time()
        return self.open_time <= local <= self.close_time


KRX_DAILY = Session(name="KRX_DAILY", open_time=time(9, 0), close_time=time(15, 30))


@dataclass(frozen=True, slots=True)
class SessionCalendar:
    """Ordered trading sessions used for purged folds and settlement math."""

    sessions: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if any(self.sessions[i] >= self.sessions[i + 1] for i in range(len(self.sessions) - 1)):
            raise ValueError("sessions must be strictly increasing")

    def index_of(self, when: datetime) -> int:
        for i, s in enumerate(self.sessions):
            if s >= when:
                return i
        return len(self.sessions)

    def sessions_between(self, start: datetime, end: datetime) -> tuple[datetime, ...]:
        lo = self.index_of(start)
        hi = self.index_of(end)
        return self.sessions[lo:hi]


@dataclass(frozen=True, slots=True)
class PointInTime:
    """Immutable timestamp bundle enforcing decision-time ordering."""

    observation_time: datetime
    available_time: datetime
    decision_time: datetime
    execution_time: datetime

    def __post_init__(self) -> None:
        if self.observation_time > self.available_time:
            raise TemporalViolationError(
                "observation_time must not be after available_time "
                f"({self.observation_time} > {self.available_time})"
            )
        if self.available_time > self.decision_time:
            raise TemporalViolationError(
                "available_time must not be after decision_time "
                f"({self.available_time} > {self.decision_time})"
            )
        if self.decision_time > self.execution_time:
            raise TemporalViolationError(
                "decision_time must not be after execution_time "
                f"({self.decision_time} > {self.execution_time})"
            )
