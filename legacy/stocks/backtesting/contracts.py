"""Canonical backtest request/result/ledger/trade contracts.

These dataclasses are the single source of truth for the replay engine;
``backtesting.engine`` re-exports them so every historical import path keeps
resolving. The former duplicate ``BacktestResult`` facade that lived here was
removed: this module owns the real engine contract.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from src.core.costs import CostSchedule
from legacy.stocks.trading.portfolio_constructor import StockRiskPolicy

REQUIRED_BACKTEST_COLUMNS = (
    "instrument_id",
    "session",
    "open",
    "close",
    "volume",
    "trading_value",
)


class BacktestValidationError(ValueError):
    """Raised when a replay request or schedule is invalid."""


@dataclass(frozen=True, slots=True)
class BacktestAttribution:
    """Typed cost attribution for a backtest run.

    Separates base and stress cost components for gross-to-net bridge.
    """

    base_commission: float = 0.0
    base_tax: float = 0.0
    base_spread: float = 0.0
    base_impact: float = 0.0
    base_other: float = 0.0
    base_total: float = 0.0
    stress_commission: float = 0.0
    stress_tax: float = 0.0
    stress_spread: float = 0.0
    stress_impact: float = 0.0
    stress_other: float = 0.0
    stress_total: float = 0.0
    gross_return: float = 0.0
    net_return: float = 0.0
    cost_drag_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class ArtifactSlot:
    """One scheduled artifact eligibility range."""

    eligible_from: datetime
    eligible_to: datetime
    artifact_id: str


@dataclass(frozen=True, slots=True)
class ArtifactSchedule:
    """Immutable, sorted, non-overlapping artifact eligibility schedule."""

    slots: tuple[ArtifactSlot, ...]

    def __post_init__(self) -> None:
        if not self.slots:
            raise BacktestValidationError("artifact schedule must have at least one slot")
        prev_to: datetime | None = None
        for slot in self.slots:
            if slot.eligible_from.tzinfo is None or slot.eligible_to.tzinfo is None:
                raise BacktestValidationError("artifact slots must be timezone-aware")
            if slot.eligible_from > slot.eligible_to:
                raise BacktestValidationError("artifact slot eligible_from after eligible_to")
            if prev_to is not None and slot.eligible_from <= prev_to:
                raise BacktestValidationError("artifact slots must not overlap")
            prev_to = slot.eligible_to

    def artifact_for(self, decision_time: datetime) -> str:
        for slot in self.slots:
            if slot.eligible_from <= decision_time <= slot.eligible_to:
                return slot.artifact_id
        raise BacktestValidationError(
            f"no scheduled artifact eligible at {decision_time.isoformat()}"
        )


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """Input contract for one historical replay run."""

    strategy_id: str
    start_time: datetime
    end_time: datetime
    decision_session_indices: tuple[int, ...]
    cost_schedule: CostSchedule
    stress_cost_schedule: CostSchedule
    risk_policy: StockRiskPolicy
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise BacktestValidationError("strategy_id must be non-empty")
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise BacktestValidationError("start_time and end_time must be timezone-aware")
        if self.start_time >= self.end_time:
            raise BacktestValidationError("start_time must be before end_time")
        if len(set(self.decision_session_indices)) != len(self.decision_session_indices):
            raise BacktestValidationError("decision_session_indices must not repeat")
        if list(self.decision_session_indices) != sorted(self.decision_session_indices):
            raise BacktestValidationError("decision_session_indices must be sorted ascending")
        if self.cost_schedule.name == self.stress_cost_schedule.name:
            raise BacktestValidationError("base and stress cost schedules must differ")
        self.cost_schedule.cost_for(self.start_time)
        self.cost_schedule.cost_for(self.end_time)


@dataclass(frozen=True, slots=True)
class BacktestLedgerRow:
    """One reconciled accounting snapshot at a session close."""

    session: datetime
    settled_cash: float
    unsettled_cash: float
    positions_value: float
    accrued_costs: float
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One attempted fill, filled or unfilled with a reason."""

    session: datetime
    instrument_id: str
    side: str
    quantity: int
    price: float | None
    gross: float | None
    cost: float | None
    reason: str
    cost_breakdown: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of a historical replay: ledger, fills, and derived metrics."""

    ledger: tuple[BacktestLedgerRow, ...]
    trades: tuple[BacktestTrade, ...]
    final_value: float
    total_return: float
    metrics: dict[str, float]
    stress_final_value: float | None = None
    stress_metrics: dict[str, float] | None = None
    stress_ledger: tuple[BacktestLedgerRow, ...] = ()
    data_quality: dict[str, str] = field(default_factory=dict)
    planned_cycles: int = 0
    attempted_orders: int = 0
    filled_orders: int = 0
    no_trade_reasons: tuple[str, ...] = ()
    unfilled_order_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def equity_curve(self) -> list[float]:
        return [row.equity for row in self.ledger]

    @property
    def terminal_equity(self) -> float:
        """Last observed base equity; the canonical terminal value."""
        return self.final_value

    @property
    def turnover_ratio(self) -> float:
        return float(self.metrics.get("turnover", 0.0))

    @property
    def total_cost(self) -> float:
        return float(self.metrics.get("cost_drag", 0.0))


ReplayPlanner = Callable[..., object]


def result_metrics_view(result: BacktestResult) -> Mapping[str, float]:
    """Read-only view of a result's derived metric dictionary."""
    return {key: float(value) for key, value in result.metrics.items()}
