"""Fill-side cost contracts shared by stock and ETF simulations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class CostPoint:
    """A single effective-dated cost segment.

    ``effective_from`` is the earliest wall-clock time at which this segment
    applies; every later lookup resolves to this segment until the next one.
    """

    effective_from: datetime
    commission_rate: float
    tax_rate: float
    slippage_bps: float
    settlement_days: int = 2

    def __post_init__(self) -> None:
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        if self.tax_rate < 0:
            raise ValueError("tax_rate must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.settlement_days < 0:
            raise ValueError("settlement_days must be non-negative")


@dataclass(frozen=True, slots=True)
class CostSchedule:
    """Effective-dated fee/tax/slippage/settlement schedule.

    ``points`` must be strictly increasing by ``effective_from`` so every
    lookup is unambiguous. A lookup before the first segment, or a frame with
    gaps or overlaps, fails closed with ``ValueError``.
    """

    name: str
    points: tuple[CostPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("cost schedule must have at least one point")
        prev: datetime | None = None
        for point in self.points:
            when = _as_utc(point.effective_from)
            if prev is not None and when < prev:
                raise ValueError("cost points must be sorted by effective_from")
            if prev is not None and when == prev:
                raise ValueError("cost points must not overlap in effective_from")
            prev = when

    def cost_for(self, effective_time: datetime) -> CostPoint:
        """Resolve the cost segment covering ``effective_time``."""
        when = _as_utc(effective_time)
        chosen: CostPoint | None = None
        for point in self.points:
            if _as_utc(point.effective_from) <= when:
                chosen = point
            else:
                break
        if chosen is None:
            raise ValueError(f"no cost coverage at {when.isoformat()}")
        return chosen


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def default_base_schedule() -> CostSchedule:
    """Reference base cost schedule (explicit input, not a strategy constant)."""
    return CostSchedule(
        name="base",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=0.00015,
                tax_rate=0.0023,
                slippage_bps=5.0,
                settlement_days=2,
            ),
        ),
    )


def default_stress_schedule() -> CostSchedule:
    """Reference stress cost schedule: wider spread and higher tax."""
    return CostSchedule(
        name="stress",
        points=(
            CostPoint(
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                commission_rate=0.0005,
                tax_rate=0.0033,
                slippage_bps=25.0,
                settlement_days=2,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fee/tax/slippage assumptions as typed inputs, not strategy constants.

    ``round_trip_cost`` models one entry and one exit fill: commission on both
    sides, tax once (sell-side only), and slippage as basis points of notional.
    """

    commission_rate: float = 0.0
    tax_rate: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        if self.tax_rate < 0:
            raise ValueError("tax_rate must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

    def round_trip_cost(self, notional: float) -> float:
        return notional * (self.commission_rate * 2 + self.tax_rate + self.slippage_bps / 10_000)
