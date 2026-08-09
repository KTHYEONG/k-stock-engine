"""Immutable market-data primitives shared by stock and ETF subsystems."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from src.core.instruments import Instrument


class DataQuality(StrEnum):
    OK = "ok"
    STALE = "stale"
    SUSPENDED = "suspended"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class Bar:
    """A single immutable price bar.

    ``source`` records provenance; ``observation_time`` is the market instant
    the bar describes and ``available_time`` is when the bar became available.
    """

    instrument: Instrument
    source: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    observation_time: datetime
    available_time: datetime

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"Bar invariant violated for {self.instrument.instrument_id}: "
                f"open {self.open} outside [low {self.low}, high {self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"Bar invariant violated for {self.instrument.instrument_id}: "
                f"close {self.close} outside [low {self.low}, high {self.high}]"
            )


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A corporate action affecting a canonical instrument."""

    instrument: Instrument
    action_date: datetime
    kind: str
    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError("corporate action factor must be positive")
