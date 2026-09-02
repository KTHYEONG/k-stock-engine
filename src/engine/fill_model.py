"""Historical fill model with participation, tick, and scenario friction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.instruments import Instrument
from src.core.ledger import LedgerFill, LedgerSide
from src.execution.domain.orders import OrderSide


class BacktestIntegrityError(ValueError):
    """Fatal invariant violation that aborts the backtest without partial result."""


class ExecutionScenario(StrEnum):
    IDEAL = "ideal"
    BASE = "base"
    STRESS = "stress"


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    session_open: datetime
    instrument_id: str
    raw_open: float
    raw_close: float
    adtv_20d: float
    daily_volatility: float

    def __post_init__(self) -> None:
        if self.session_open.tzinfo is None:
            raise ValueError("session_open must be aware")
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    order_id: str
    intent_id: str
    instrument: Instrument
    side: OrderSide
    quantity: int
    decision_time: datetime
    execution_time: datetime

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")
        if not isinstance(self.side, OrderSide):
            raise ValueError("side must be OrderSide")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("quantity must be integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.decision_time.tzinfo is None or self.execution_time.tzinfo is None:
            raise ValueError("decision_time and execution_time must be aware")
        if self.decision_time > self.execution_time:
            raise ValueError("decision_time must not be after execution_time")


@dataclass(frozen=True, slots=True)
class FillOutcome:
    fill: LedgerFill
    requested_quantity: int
    unfilled_quantity: int
    participation: float
    scenario: ExecutionScenario


@dataclass(frozen=True, slots=True)
class BacktestReject:
    reject_id: str
    order_id: str
    reason: str
    rejected_quantity: int
    event_time: datetime

    def __post_init__(self) -> None:
        if not self.reject_id:
            raise ValueError("reject_id must be non-empty")
        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if self.event_time.tzinfo is None:
            raise ValueError("event_time must be aware")


@dataclass(frozen=True, slots=True)
class CapacityDiagnostic:
    order_id: str
    instrument_id: str
    requested_quantity: int
    filled_quantity: int
    participation: float
    adtv_20d: float
    target_cap: float
    hard_cap: float
    scenario: ExecutionScenario


class HistoricalFillModel:
    def __init__(
        self,
        cost_schedule: CostSchedule,
        slippage_model: LiquiditySlippageModel,
        scenario: ExecutionScenario,
        *,
        target_participation_cap: float,
        hard_participation_cap: float,
    ) -> None:
        if not isinstance(scenario, ExecutionScenario):
            raise ValueError("scenario must be ExecutionScenario")
        if not isinstance(target_participation_cap, (int, float)) or not math.isfinite(float(target_participation_cap)):
            raise ValueError("target_participation_cap must be finite")
        if not isinstance(hard_participation_cap, (int, float)) or not math.isfinite(float(hard_participation_cap)):
            raise ValueError("hard_participation_cap must be finite")
        if float(target_participation_cap) <= 0 or float(hard_participation_cap) <= 0:
            raise ValueError("caps must be positive")
        if float(target_participation_cap) > float(hard_participation_cap):
            raise ValueError("target cap must not exceed hard cap")
        self._cost_schedule = cost_schedule
        self._slippage_model = slippage_model
        self._scenario = scenario
        self._target_cap = float(target_participation_cap)
        self._hard_cap = float(hard_participation_cap)
        self.target_cap = self._target_cap
        self.hard_cap = self._hard_cap

    def execute(self, order: BacktestOrder, bar: HistoricalBar) -> FillOutcome | BacktestReject:
        # fatal validations
        if bar.instrument_id != order.instrument.instrument_id:
            raise BacktestIntegrityError(f"bar instrument {bar.instrument_id!r} mismatches order {order.instrument.instrument_id!r}")
        if order.instrument.lot_size < 1:
            raise BacktestIntegrityError("lot_size must be positive")
        if order.quantity % order.instrument.lot_size != 0:
            raise BacktestIntegrityError(f"quantity {order.quantity} not multiple of lot {order.instrument.lot_size}")
        # raw price checks
        for name, val in (("raw_open", bar.raw_open), ("raw_close", bar.raw_close)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0:
                raise BacktestIntegrityError(f"missing/non-positive {name}: {val!r}")
        if isinstance(bar.adtv_20d, bool) or not isinstance(bar.adtv_20d, (int, float)) or not math.isfinite(float(bar.adtv_20d)) or float(bar.adtv_20d) <= 0:
            raise BacktestIntegrityError(f"missing/non-positive adtv_20d: {bar.adtv_20d!r}")
        if isinstance(bar.daily_volatility, bool) or not isinstance(bar.daily_volatility, (int, float)) or not math.isfinite(float(bar.daily_volatility)) or float(bar.daily_volatility) <= 0:
            raise BacktestIntegrityError(f"missing/non-positive daily_volatility: {bar.daily_volatility!r}")
        # cost coverage
        try:
            cost_point = self._cost_schedule.cost_for(bar.session_open)
        except ValueError as exc:
            raise BacktestIntegrityError(str(exc)) from exc
        # tick rule
        try:
            tick = self._slippage_model.tick_schedule.tick_size(float(bar.raw_open), bar.session_open)
        except ValueError as exc:
            raise BacktestIntegrityError(str(exc)) from exc
        if not math.isfinite(float(tick)) or float(tick) <= 0:
            raise BacktestIntegrityError(f"invalid tick {tick!r}")
        # participation capacity
        raw_open = float(bar.raw_open)
        adtv = float(bar.adtv_20d)
        lot = int(order.instrument.lot_size)
        requested = int(order.quantity)
        req_participation = requested * raw_open / adtv
        # hard cap fatal
        if req_participation > self._hard_cap + 1e-12:
            raise BacktestIntegrityError(f"hard participation breach: {req_participation:.6f} > {self._hard_cap}")
        target_qty_max = math.floor((self._target_cap * adtv) / raw_open / lot) * lot
        hard_qty_max = math.floor((self._hard_cap * adtv) / raw_open / lot) * lot
        # determine filled qty
        if requested <= target_qty_max:
            filled_qty = requested
        elif requested <= hard_qty_max:
            if target_qty_max == 0:
                # zero capacity -> reject
                return BacktestReject(
                    reject_id=f"reject:{order.order_id}",
                    order_id=order.order_id,
                    reason="zero capacity at target",
                    rejected_quantity=requested,
                    event_time=bar.session_open,
                )
            filled_qty = int(target_qty_max)
        else:
            # should have been hard breach earlier, but if lot rounding pushed over?
            raise BacktestIntegrityError("hard participation breach after lot rounding")
        if filled_qty == 0:
            return BacktestReject(
                reject_id=f"reject:{order.order_id}",
                order_id=order.order_id,
                reason="zero capacity",
                rejected_quantity=requested,
                event_time=bar.session_open,
            )
        # slippage bps per scenario
        if self._scenario == ExecutionScenario.IDEAL:
            slippage_bps = 0.0
        elif self._scenario == ExecutionScenario.BASE:
            slippage_bps = self._slippage_model.slippage_bps(
                notional=float(filled_qty) * raw_open,
                adtv_20d=adtv,
                daily_volatility=float(bar.daily_volatility),
                reference_price=raw_open,
                effective_time=bar.session_open,
            )
        else:  # STRESS doubles dynamic friction
            base_bps = self._slippage_model.slippage_bps(
                notional=float(filled_qty) * raw_open,
                adtv_20d=adtv,
                daily_volatility=float(bar.daily_volatility),
                reference_price=raw_open,
                effective_time=bar.session_open,
            )
            slippage_bps = base_bps * 2.0
        # adverse price
        if order.side == OrderSide.BUY:
            adj_price = raw_open * (1.0 + slippage_bps / 10_000.0)
            # ceil to tick
            price = math.ceil(adj_price / tick - 1e-9) * tick
            # handle floating tolerance: if adj_price is integer multiple of tick, ceil returns same
            # ensure price is multiple of tick within epsilon
            price = round(price / tick) * tick
        else:
            adj_price = raw_open * (1.0 - slippage_bps / 10_000.0)
            price = math.floor(adj_price / tick + 1e-9) * tick
            price = round(price / tick) * tick
        if not math.isfinite(float(price)) or float(price) <= 0:
            raise BacktestIntegrityError(f"invalid fill price {price!r}")
        # commission and tax per cost_schedule
        commission = filled_qty * float(price) * float(cost_point.commission_rate)
        tax = 0.0
        if order.side == OrderSide.SELL:
            tax = filled_qty * float(price) * float(cost_point.tax_rate)
        slippage_cost = abs(float(price) - raw_open) * filled_qty
        # settlement time
        settlement_time = bar.session_open + timedelta(days=int(cost_point.settlement_days))
        # ensure aware
        if settlement_time.tzinfo is None:
            settlement_time = settlement_time.replace(tzinfo=bar.session_open.tzinfo)
        ledger_side = LedgerSide.BUY if order.side == OrderSide.BUY else LedgerSide.SELL
        fill = LedgerFill(
            fill_id=order.order_id,
            instrument_id=order.instrument.instrument_id,
            side=ledger_side,
            quantity=int(filled_qty),
            price=float(price),
            commission=float(commission),
            tax=float(tax),
            slippage_cost=float(slippage_cost),
            trade_time=bar.session_open,
            settlement_time=settlement_time,
        )
        participation = filled_qty * raw_open / adtv
        unfilled = requested - filled_qty
        return FillOutcome(
            fill=fill,
            requested_quantity=requested,
            unfilled_quantity=int(unfilled),
            participation=float(participation),
            scenario=self._scenario,
        )
