"""Unified event-driven backtester."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime

from src.core.costs import CostSchedule
from src.core.instruments import Instrument
from src.core.ledger import Ledger, LedgerCorporateAction, LedgerJournalEntry, LedgerMark, LedgerNav
from src.core.portfolio import PortfolioSnapshot, Position
from src.core.time import SessionCalendar
from src.engine.decision import DecisionContext, StrategyDecisionPort
from src.engine.fill_model import (
    BacktestIntegrityError,
    BacktestOrder,
    BacktestReject,
    CapacityDiagnostic,
    ExecutionScenario,
    FillOutcome,
    HistoricalBar,
    HistoricalFillModel,
)
from src.execution.domain.intents import TradeIntent
from src.execution.domain.orders import OrderSide

# wiring anchor for spec: build_champion_portfolio
_BUILD_CHAMPION_PORTFOLIO_ANCHOR = "build_champion_portfolio"


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    ledger_id: str
    initial_cash: float
    instruments: Mapping[str, Instrument]
    scenario: ExecutionScenario
    cost_schedule: CostSchedule | None = None
    calendar: SessionCalendar | None = None
    fill_model: HistoricalFillModel | None = None

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise ValueError("ledger_id must be non-empty")
        if not isinstance(self.scenario, ExecutionScenario):
            raise ValueError("scenario must be ExecutionScenario")


@dataclass(frozen=True, slots=True)
class BacktestSession:
    session_open: datetime
    decision_time: datetime
    bars: tuple[HistoricalBar, ...]
    actions: tuple[object, ...]
    market_snapshot: object

    def __post_init__(self) -> None:
        if self.session_open.tzinfo is None or self.decision_time.tzinfo is None:
            raise ValueError("session times must be aware")
        if self.decision_time < self.session_open:
            raise ValueError("decision_time must not be before session_open")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    fills: tuple[object, ...]
    rejects: tuple[BacktestReject, ...]
    journal: tuple[LedgerJournalEntry, ...]
    daily_nav: tuple[LedgerNav, ...]
    capacity_diagnostics: tuple[CapacityDiagnostic, ...]
    scenario: ExecutionScenario


@dataclass(frozen=True, slots=True)
class _PendingTarget:
    intent: TradeIntent
    target_qty: int
    instrument: Instrument
    decision_session_open: datetime


class EventBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def run(self, sessions: tuple[BacktestSession, ...], strategy: StrategyDecisionPort) -> BacktestResult:
        if not sessions:
            raise BacktestIntegrityError("no sessions")
        for i in range(len(sessions) - 1):
            if sessions[i].session_open >= sessions[i + 1].session_open:
                raise BacktestIntegrityError("sessions must be strictly increasing")
        cost_schedule = self._config.cost_schedule
        calendar = self._config.calendar
        fill_model = self._config.fill_model
        if not isinstance(cost_schedule, CostSchedule):
            raise BacktestIntegrityError("cost_schedule must be explicit CostSchedule")
        if not isinstance(calendar, SessionCalendar):
            raise BacktestIntegrityError("calendar must be explicit SessionCalendar")
        if not isinstance(fill_model, HistoricalFillModel):
            raise BacktestIntegrityError("fill_model must be explicit HistoricalFillModel")
        if fill_model.scenario is not self._config.scenario:
            raise BacktestIntegrityError(
                f"fill_model scenario {fill_model.scenario} mismatches config scenario {self._config.scenario}"
            )
        ledger = Ledger(self._config.ledger_id, float(self._config.initial_cash), sessions[0].session_open)
        fills: list[object] = []
        rejects: list[BacktestReject] = []
        daily_nav: list[LedgerNav] = []
        capacity_diagnostics: list[CapacityDiagnostic] = []
        pending_targets: list[_PendingTarget] = []

        def _make_snapshot(as_of: datetime) -> PortfolioSnapshot:
            snap = ledger.snapshot(as_of)
            positions = []
            for pos in snap.positions:
                instr = self._config.instruments.get(pos.instrument_id)
                if instr is None:
                    raise BacktestIntegrityError(f"unknown instrument {pos.instrument_id!r} in ledger snapshot")
                positions.append(Position(instrument=instr, quantity=float(pos.quantity), average_cost=pos.average_cost))
            return PortfolioSnapshot(
                account_snapshot_id=f"{self._config.ledger_id}:{as_of.isoformat()}",
                as_of=as_of,
                settled_cash=snap.settled_cash,
                unsettled_cash=snap.unsettled_cash,
                positions=tuple(positions),
                open_order_ids=tuple(f"order:{p.intent.intent_id}:{p.decision_session_open.isoformat()}" for p in pending_targets),
            )

        try:
            for idx, session in enumerate(sessions):
                seen_ids: set[str] = set()
                for b in session.bars:
                    if b.instrument_id in seen_ids:
                        raise BacktestIntegrityError(f"duplicate bar for {b.instrument_id!r}")
                    seen_ids.add(b.instrument_id)
                    if b.session_open != session.session_open:
                        raise BacktestIntegrityError(f"bar session_open {b.session_open} mismatches session {session.session_open}")
                for act in session.actions:
                    if not isinstance(act, LedgerCorporateAction):
                        raise BacktestIntegrityError(f"unknown action {act!r}")
                try:
                    ledger.settle(session.session_open)
                except ValueError as exc:
                    raise BacktestIntegrityError(str(exc)) from exc
                if session.actions:
                    cash_prices: dict[str, float] = {b.instrument_id: float(b.raw_open) for b in session.bars}
                    try:
                        ledger.apply_corporate_actions(
                            tuple(session.actions),  # type: ignore[arg-type]
                            session_open=session.session_open,
                            cash_in_lieu_prices=cash_prices,
                        )
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                if pending_targets:
                    # determine side for sorting sells before buys using post-action holdings
                    def _delta(pt: _PendingTarget) -> int:
                        return int(pt.target_qty) - int(ledger.quantity_of(pt.instrument.instrument_id))

                    pending_targets = sorted(pending_targets, key=lambda pt: 0 if _delta(pt) < 0 else 1)
                    bar_map: dict[str, HistoricalBar] = {b.instrument_id: b for b in session.bars}
                    next_pending: list[_PendingTarget] = []
                    for pt in pending_targets:
                        bar = bar_map.get(pt.instrument.instrument_id)
                        if bar is None:
                            raise BacktestIntegrityError(f"missing bar for {pt.instrument.instrument_id!r} at {session.session_open}")
                        current_qty = ledger.quantity_of(pt.instrument.instrument_id)
                        delta = int(pt.target_qty) - int(current_qty)
                        if delta == 0:
                            continue
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        qty = abs(delta)
                        if qty % int(pt.instrument.lot_size) != 0:
                            raise BacktestIntegrityError("order quantity not multiple of lot")
                        order = BacktestOrder(
                            order_id=f"order:{pt.intent.intent_id}:{pt.decision_session_open.isoformat()}",
                            intent_id=pt.intent.intent_id,
                            instrument=pt.instrument,
                            side=side,
                            quantity=int(qty),
                            decision_time=pt.intent.decision_time,
                            execution_time=pt.intent.execution_time,
                        )
                        result = fill_model.execute(order, bar)
                        if isinstance(result, FillOutcome):
                            try:
                                settlement_time = calendar.advance(
                                    result.fill.trade_time,
                                    int(cost_schedule.cost_for(result.fill.trade_time).settlement_days),
                                )
                            except ValueError as exc:
                                raise BacktestIntegrityError(str(exc)) from exc
                            fill = replace(result.fill, settlement_time=settlement_time)
                            try:
                                ledger.record_fill(fill)
                            except ValueError as exc:
                                raise BacktestIntegrityError(str(exc)) from exc
                            fills.append(fill)
                            capacity_diagnostics.append(
                                CapacityDiagnostic(
                                    order_id=order.order_id,
                                    instrument_id=order.instrument.instrument_id,
                                    requested_quantity=result.requested_quantity,
                                    filled_quantity=result.fill.quantity,
                                    participation=result.participation,
                                    adtv_20d=float(bar.adtv_20d),
                                    target_cap=fill_model.target_cap,
                                    hard_cap=fill_model.hard_cap,
                                    scenario=self._config.scenario,
                                )
                            )
                            if result.unfilled_quantity > 0:
                                reject = BacktestReject(
                                    reject_id=f"reject:{order.order_id}:remainder",
                                    order_id=order.order_id,
                                    reason="target cap partial fill remainder",
                                    rejected_quantity=int(result.unfilled_quantity),
                                    event_time=bar.session_open,
                                )
                                rejects.append(reject)
                        else:
                            rejects.append(result)
                            capacity_diagnostics.append(
                                CapacityDiagnostic(
                                    order_id=order.order_id,
                                    instrument_id=order.instrument.instrument_id,
                                    requested_quantity=order.quantity,
                                    filled_quantity=0,
                                    participation=0.0,
                                    adtv_20d=float(bar.adtv_20d),
                                    target_cap=fill_model.target_cap,
                                    hard_cap=fill_model.hard_cap,
                                    scenario=self._config.scenario,
                                )
                            )
                    pending_targets = next_pending
                snap_before_mark = ledger.snapshot(session.session_open)
                if snap_before_mark.positions:
                    bar_close_map: dict[str, float] = {b.instrument_id: float(b.raw_close) for b in session.bars}
                    prices: list[tuple[str, float]] = []
                    for pos in snap_before_mark.positions:
                        if pos.instrument_id not in bar_close_map:
                            raise BacktestIntegrityError(f"missing raw close for {pos.instrument_id!r}")
                        price = bar_close_map[pos.instrument_id]
                        if not isinstance(price, float):
                            price = float(price)
                        import math

                        if not math.isfinite(float(price)) or float(price) <= 0:
                            raise BacktestIntegrityError("missing/non-positive raw close")
                        prices.append((pos.instrument_id, price))
                    mark_id = f"mark:{session.session_open.isoformat()}:{idx}"
                    mark = LedgerMark(mark_id, session.session_open, tuple(prices))
                    try:
                        nav = ledger.record_mark(mark)
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                    daily_nav.append(nav)
                else:
                    mark_id = f"mark:{session.session_open.isoformat()}:{idx}"
                    mark = LedgerMark(mark_id, session.session_open, ())
                    try:
                        nav = ledger.record_mark(mark)
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                    daily_nav.append(nav)
                portfolio = _make_snapshot(session.decision_time)
                context = DecisionContext(
                    decision_time=session.decision_time,
                    portfolio=portfolio,
                    market_snapshot=session.market_snapshot,
                )
                intents = strategy.decide(context)
                if intents is None:
                    intents = ()
                next_open = sessions[idx + 1].session_open if idx + 1 < len(sessions) else None
                new_targets: list[_PendingTarget] = []
                for intent in intents:
                    if not isinstance(intent, TradeIntent):
                        raise BacktestIntegrityError("intent must be TradeIntent")
                    if intent.decision_time != session.decision_time:
                        raise BacktestIntegrityError(f"intent decision_time {intent.decision_time} must equal current decision_time {session.decision_time}")
                    if next_open is not None:
                        if intent.execution_time != next_open:
                            raise BacktestIntegrityError(f"T-close intent execution_time must be next session open: {intent.execution_time} != {next_open}")
                    else:
                        continue
                    instr = self._config.instruments.get(intent.instrument_id)
                    if instr is None:
                        raise BacktestIntegrityError(f"unknown instrument {intent.instrument_id!r}")
                    lot = int(instr.lot_size)
                    if lot < 1:
                        raise BacktestIntegrityError("lot_size must be positive")
                    bar_map_ref: dict[str, HistoricalBar] = {b.instrument_id: b for b in session.bars}
                    ref_bar = bar_map_ref.get(intent.instrument_id)
                    if intent.target_quantity is not None:
                        target_qty = int(intent.target_quantity)
                        if target_qty % lot != 0:
                            raise BacktestIntegrityError("target_quantity not multiple of lot")
                    else:
                        if ref_bar is None:
                            raise BacktestIntegrityError(f"missing bar for target_value conversion {intent.instrument_id!r}")
                        ref_price = float(ref_bar.raw_close) if float(ref_bar.raw_close) > 0 else float(ref_bar.raw_open)
                        import math as _math

                        if not _math.isfinite(ref_price) or ref_price <= 0:
                            raise BacktestIntegrityError("missing/non-positive reference price")
                        target_qty = int(intent.target_value / ref_price / lot) * lot
                    # validate target lot (already)
                    new_targets.append(_PendingTarget(intent=intent, target_qty=target_qty, instrument=instr, decision_session_open=session.session_open))
                pending_targets = new_targets
        except BacktestIntegrityError:
            raise
        except ValueError as exc:
            raise BacktestIntegrityError(str(exc)) from exc
        journal = ledger.journal()
        return BacktestResult(
            fills=tuple(fills),
            rejects=tuple(rejects),
            journal=journal,
            daily_nav=tuple(daily_nav),
            capacity_diagnostics=tuple(capacity_diagnostics),
            scenario=self._config.scenario,
        )
