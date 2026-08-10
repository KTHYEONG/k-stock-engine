"""Fill-side cost contracts shared by stock and ETF simulations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import inf, sqrt


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

@dataclass(frozen=True, slots=True)
class TickSizeRule:
    """One price band of a KRX tick-size schedule.

    ``lower_inclusive`` and ``upper_exclusive`` delimit a half-open price band
    ``[lower_inclusive, upper_exclusive)``. The final band of an effective-dated
    group uses ``float("inf")`` as ``upper_exclusive`` so every positive price
    is covered with no gaps.
    """

    rule_id: str
    effective_from: datetime
    lower_inclusive: float
    upper_exclusive: float
    tick: float
    session: str = "regular"

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("tick rule requires a rule_id")
        if self.lower_inclusive < 0:
            raise ValueError("lower_inclusive must be non-negative")
        if self.upper_exclusive <= self.lower_inclusive:
            raise ValueError("upper_exclusive must exceed lower_inclusive")
        if self.tick <= 0:
            raise ValueError("tick must be positive")
        if not self.session:
            raise ValueError("session must be non-empty")


@dataclass(frozen=True, slots=True)
class TickSizeSchedule:
    """Effective-dated KRX tick-size rules with gapless price coverage.

    For each ``(effective_from, session)`` group the bands must start at zero
    and tile ``[0, inf)`` without overlaps or gaps so any positive reference
    price resolves to exactly one tick.
    """

    rules: tuple[TickSizeRule, ...]

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("tick schedule must have at least one rule")
        groups: dict[tuple[datetime, str], list[TickSizeRule]] = {}
        for rule in self.rules:
            key = (_as_utc(rule.effective_from), rule.session)
            groups.setdefault(key, []).append(rule)
        for key, band_rules in groups.items():
            ordered = sorted(band_rules, key=lambda rule: rule.lower_inclusive)
            if ordered[0].lower_inclusive != 0.0:
                raise ValueError(f"tick bands for {key} do not start at zero")
            for previous, current in pairwise(ordered):
                if previous.upper_exclusive != current.lower_inclusive:
                    raise ValueError(f"tick bands for {key} have a gap or overlap")
            if ordered[-1].upper_exclusive != inf:
                raise ValueError(f"tick bands for {key} do not cover all prices")

    def rule_for(
        self,
        price: float,
        effective_time: datetime,
        session: str = "regular",
    ) -> TickSizeRule:
        """Resolve the tick rule covering ``price`` at ``effective_time``."""
        if price <= 0:
            raise ValueError("price must be positive")
        when = _as_utc(effective_time)
        eligible = [
            rule
            for rule in self.rules
            if _as_utc(rule.effective_from) <= when and rule.session == session
        ]
        chosen = sorted(eligible, key=lambda rule: _as_utc(rule.effective_from))[-1]
        for rule in sorted(
            (rule for rule in eligible if _as_utc(rule.effective_from) == _as_utc(chosen.effective_from)),
            key=lambda rule: rule.lower_inclusive,
        ):
            if rule.lower_inclusive <= price < rule.upper_exclusive:
                return rule
        raise ValueError(
            f"no tick rule covers price {price} at {when.isoformat()} session {session!r}"
        )

    def tick_size(
        self,
        price: float,
        effective_time: datetime,
        session: str = "regular",
    ) -> float:
        """Return the tick for ``price`` at ``effective_time``."""
        return self.rule_for(price, effective_time, session).tick


@dataclass(frozen=True, slots=True)
class LiquiditySlippageModel:
    """Market-impact slippage for one fill.

    One-way slippage in bps is

    ``half_spread_bps + impact_coefficient * stress_multiplier *
    daily_volatility_bps * sqrt(notional / adtv_20d)``

    where ``half_spread_bps`` is derived from the effective-dated KRX tick
    schedule and the reference price. Small retail orders therefore incur
    little impact, while low-liquidity or large orders grow naturally. The
    ``stress_multiplier`` is the explicit artifact stress parameter, never a
    hidden second constant.
    """

    impact_coefficient: float
    tick_schedule: TickSizeSchedule
    stress_multiplier: float = 1.0
    model_id: str = "sqrt_impact_v1"

    def __post_init__(self) -> None:
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")
        if self.stress_multiplier < 0:
            raise ValueError("stress_multiplier must be non-negative")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")

    def half_spread_bps(
        self,
        reference_price: float,
        effective_time: datetime,
        session: str = "regular",
    ) -> float:
        """Half-spread cost in bps from the tick at the reference price."""
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        tick = self.tick_schedule.tick_size(reference_price, effective_time, session)
        return 0.5 * tick / reference_price * 10_000

    def slippage_bps(
        self,
        *,
        notional: float,
        adtv_20d: float,
        daily_volatility: float,
        reference_price: float,
        effective_time: datetime,
        session: str = "regular",
    ) -> float:
        """One-way dynamic slippage in bps for the fill notional."""
        if notional <= 0:
            raise ValueError("notional must be positive")
        if adtv_20d <= 0:
            raise ValueError("adtv_20d must be positive")
        if daily_volatility <= 0:
            raise ValueError("daily_volatility must be positive")
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        half_spread = self.half_spread_bps(reference_price, effective_time, session)
        participation = sqrt(notional / adtv_20d)
        return (
            half_spread
            + self.impact_coefficient
            * self.stress_multiplier
            * daily_volatility
            * 10_000
            * participation
        )

    @property
    def params_hash(self) -> str:
        """Deterministic fingerprint of the explicit model parameters."""
        return sha256(
            f"{self.model_id}|{self.impact_coefficient}|{self.stress_multiplier}".encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class FillCostBreakdown:
    """Resolved per-fill cost components for one order side.

    ``sell_tax_rate`` is the summed statutory sell tax and equals the sum of
    ``securities_transaction_tax_rate`` and ``rural_special_tax_rate``.
    """

    commission_rate: float
    securities_transaction_tax_rate: float
    rural_special_tax_rate: float
    sell_tax_rate: float
    slippage_bps: float
    tick_rule_id: str
    model_id: str
    params_hash: str

    def __post_init__(self) -> None:
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        if self.securities_transaction_tax_rate < 0:
            raise ValueError("securities_transaction_tax_rate must be non-negative")
        if self.rural_special_tax_rate < 0:
            raise ValueError("rural_special_tax_rate must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

    def total_rate(self, *, side: str) -> float:
        """Total cost rate (decimal) applied to the fill notional."""
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        slippage_rate = self.slippage_bps / 10_000
        if side == "BUY":
            return self.commission_rate + slippage_rate
        return self.commission_rate + self.sell_tax_rate + slippage_rate

    def to_dict(self, *, artifact_hash: str) -> dict[str, object]:
        """Deterministic tracing record bound to the cost evidence artifact."""
        return {
            "artifact_hash": artifact_hash,
            "commission_rate": self.commission_rate,
            "securities_transaction_tax_rate": self.securities_transaction_tax_rate,
            "rural_special_tax_rate": self.rural_special_tax_rate,
            "sell_tax_rate": self.sell_tax_rate,
            "slippage_bps": self.slippage_bps,
            "tick_rule_id": self.tick_rule_id,
            "model_id": self.model_id,
            "params_hash": self.params_hash,
        }
