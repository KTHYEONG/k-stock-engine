"""Portfolio primitives shared by stock and ETF subsystems.

Fill-side cost contracts live in ``core.costs``.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.instruments import Instrument


@dataclass(frozen=True, slots=True)
class Position:
    """A current holding of one instrument."""

    instrument: Instrument
    quantity: float
    average_cost: float


@dataclass(frozen=True, slots=True)
class Allocation:
    """A target allocation to one instrument."""

    instrument: Instrument
    target_value: float
    reason: str = ""
