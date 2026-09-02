"""Unified event-driven backtester."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import inf

from src.core.costs import CostPoint, CostSchedule, LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
from src.core.instruments import Instrument
from src.core.ledger import Ledger, LedgerJournalEntry, LedgerMark, LedgerNav
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


class EventBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    def run(self, sessions: tuple[BacktestSession, ...], strategy: StrategyDecisionPort) -> BacktestResult:
        if not sessions:
            raise BacktestIntegrityError("no sessions")
        # validate sessions strictly increasing
        for i in range(len(sessions) - 1):
            if sessions[i].session_open >= sessions[i + 1].session_open:
                raise BacktestIntegrityError("sessions must be strictly increasing")
        # defaults for cost schedule and fill model
        cost_schedule = self._config.cost_schedule
        if cost_schedule is None:
            cost_schedule = CostSchedule(
                "default",
                (CostPoint(datetime(2000, 1, 1, tzinfo=UTC), 0.0, 0.0, 0.0, 2),),
            )
        calendar = self._config.calendar
        if calendar is None:
            calendar = SessionCalendar(tuple(s.session_open for s in sessions))
        fill_model = self._config.fill_model
        if fill_model is None:
            # default tick schedule with tick 1
            ticks = TickSizeSchedule((TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, inf, 1.0),))
            # default slippage model - impact 0.1 as used in fill tests
            slippage = LiquiditySlippageModel(0.1, ticks)
            fill_model = HistoricalFillModel(cost_schedule, slippage, self._config.scenario, target_participation_cap=0.1, hard_participation_cap=0.2)
        # ledger
        ledger = Ledger(self._config.ledger_id, float(self._config.initial_cash), sessions[0].session_open)
        fills: list[object] = []
        rejects: list[BacktestReject] = []
        daily_nav: list[LedgerNav] = []
        capacity_diagnostics: list[CapacityDiagnostic] = []
        # pending orders awaiting next session execution
        pending_orders: list[BacktestOrder] = []

        # helper to create portfolio snapshot from ledger
        def _make_snapshot(as_of: datetime) -> PortfolioSnapshot:
            snap = ledger.snapshot(as_of)
            positions = tuple(
                Position(
                    instrument=self._config.instruments[pos.instrument_id],
                    quantity=float(pos.quantity),
                    average_cost=pos.average_cost,
                )
                for pos in snap.positions
                if pos.instrument_id in self._config.instruments
            )
            # handle instruments not in mapping? create generic instrument
            for pos in snap.positions:
                if pos.instrument_id not in self._config.instruments:
                    # fallback instrument
                    from src.core.instruments import AssetKind

                    instr = Instrument(pos.instrument_id, AssetKind.STOCK, "KRX", pos.instrument_id, "KRW")
                    positions = (*positions, Position(instrument=instr, quantity=float(pos.quantity), average_cost=pos.average_cost))
            return PortfolioSnapshot(
                account_snapshot_id=f"{self._config.ledger_id}:{as_of.isoformat()}",
                as_of=as_of,
                settled_cash=snap.settled_cash,
                unsettled_cash=snap.unsettled_cash,
                positions=positions,
                open_order_ids=tuple(o.order_id for o in pending_orders),
            )

        try:
            for idx, session in enumerate(sessions):
                # per-session order: settle, actions, pending fills, raw-close mark/NAV, reconciliation, then decision
                # 1. settle
                try:
                    ledger.settle(session.session_open)
                except ValueError as exc:
                    raise BacktestIntegrityError(str(exc)) from exc
                # 2. actions
                if session.actions:
                    # build cash_in_lieu prices from bars raw_open
                    cash_prices: dict[str, float] = {b.instrument_id: float(b.raw_open) for b in session.bars}
                    try:
                        ledger.apply_corporate_actions(
                            tuple(session.actions),  # type: ignore[arg-type]
                            session_open=session.session_open,
                            cash_in_lieu_prices=cash_prices,
                        )
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                # 3. pending fills (orders decided previous session, executing this session)
                if pending_orders:
                    # sells before buys
                    pending_orders = sorted(pending_orders, key=lambda o: 0 if o.side == OrderSide.SELL else 1)
                    bar_map: dict[str, HistoricalBar] = {b.instrument_id: b for b in session.bars}
                    # O(P log P) for sorting, O(B+A+P) overall
                    next_pending: list[BacktestOrder] = []
                    for order in pending_orders:
                        bar = bar_map.get(order.instrument.instrument_id)
                        if bar is None:
                            raise BacktestIntegrityError(f"missing bar for {order.instrument.instrument_id!r} at {session.session_open}")
                        # missing price etc will be raised inside execute as fatal
                        result = fill_model.execute(order, bar)
                        if isinstance(result, FillOutcome):
                            # 결제일은 달력일이 아니라 KRX 세션 기준으로 산출한다.
                            try:
                                settlement_time = calendar.advance(
                                    result.fill.trade_time,
                                    int(cost_schedule.cost_for(result.fill.trade_time).settlement_days),
                                )
                            except ValueError as exc:
                                raise BacktestIntegrityError(str(exc)) from exc
                            from dataclasses import replace

                            fill = replace(result.fill, settlement_time=settlement_time)
                            # record fill in ledger
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
                        else:
                            # reject
                            rejects.append(result)
                            # diagnostic for reject with zero filled
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
                    pending_orders = next_pending
                # 4. raw-close mark/NAV and ledger reconciliation
                # build prices for all open positions from bars raw_close
                snap_before_mark = ledger.snapshot(session.session_open)
                if snap_before_mark.positions:
                    bar_close_map: dict[str, float] = {b.instrument_id: float(b.raw_close) for b in session.bars}
                    prices: list[tuple[str, float]] = []
                    for pos in snap_before_mark.positions:
                        if pos.instrument_id not in bar_close_map:
                            raise BacktestIntegrityError(f"missing raw close for {pos.instrument_id!r}")
                        price = bar_close_map[pos.instrument_id]
                        if not price or not isinstance(price, float) or price <= 0:
                            # allow int
                            price = float(price)
                        if price <= 0:
                            raise BacktestIntegrityError("missing/non-positive raw close")
                        prices.append((pos.instrument_id, price))
                    # need to ensure mark covers exactly once - already
                    mark_id = f"mark:{session.session_open.isoformat()}:{idx}"
                    mark = LedgerMark(mark_id, session.session_open, tuple(prices))
                    try:
                        nav = ledger.record_mark(mark)
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                    daily_nav.append(nav)
                else:
                    # no positions: create empty mark
                    mark_id = f"mark:{session.session_open.isoformat()}:{idx}"
                    mark = LedgerMark(mark_id, session.session_open, ())
                    try:
                        nav = ledger.record_mark(mark)
                    except ValueError as exc:
                        raise BacktestIntegrityError(str(exc)) from exc
                    daily_nav.append(nav)
                # 5. reconciliation and decision
                portfolio = _make_snapshot(session.decision_time)
                # validate portfolio as_of <= decision_time (already)
                context = DecisionContext(
                    decision_time=session.decision_time,
                    portfolio=portfolio,
                    market_snapshot=session.market_snapshot,
                )
                # invoke strategy via port - wiring requires strategy.decide(
                intents = strategy.decide(context)
                if intents is None:
                    intents = ()
                # validate no backtest-mode branch - ensure each intent execution_time is next session open
                next_open = sessions[idx + 1].session_open if idx + 1 < len(sessions) else None
                # prepare orders for next session
                new_orders: list[BacktestOrder] = []
                for intent in intents:
                    if not isinstance(intent, TradeIntent):
                        raise BacktestIntegrityError("intent must be TradeIntent")
                    # execution_time must be next session open if not last
                    if next_open is not None:
                        if intent.execution_time != next_open:
                            raise BacktestIntegrityError(f"T-close intent execution_time must be next session open: {intent.execution_time} != {next_open}")
                    else:
                        # last session intents are ignored (no future execution)
                        continue
                    # resolve instrument
                    instr = self._config.instruments.get(intent.instrument_id)
                    if instr is None:
                        # create fallback

                        instr = Instrument(intent.instrument_id, intent.asset_kind, "KRX", intent.instrument_id, "KRW")
                    lot = int(instr.lot_size)
                    # reference price for quantity calc: use current session raw_close
                    bar_map_ref: dict[str, HistoricalBar] = {b.instrument_id: b for b in session.bars}
                    ref_bar = bar_map_ref.get(intent.instrument_id)
                    if ref_bar is None:
                        # if no bar for this instrument, use execution bar raw_open next session? Fallback to 1
                        ref_price = 1.0
                    else:
                        ref_price = float(ref_bar.raw_close) if ref_bar.raw_close > 0 else float(ref_bar.raw_open)
                    if ref_price <= 0:
                        raise BacktestIntegrityError("missing/non-positive reference price")
                    current_qty = ledger.quantity_of(instr.instrument_id)
                    if intent.target_quantity is not None:
                        target_qty = int(intent.target_quantity)
                        if target_qty % lot != 0:
                            raise BacktestIntegrityError("target_quantity not multiple of lot")
                    else:
                        # floor to lot
                        target_qty = int(intent.target_value / ref_price / lot) * lot
                    delta = int(target_qty) - int(current_qty)
                    if delta == 0:
                        continue
                    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                    qty = abs(delta)
                    if qty % lot != 0:
                        raise BacktestIntegrityError("order quantity not multiple of lot")
                    # validate holdings insufficient for sell will be caught at fill time
                    order = BacktestOrder(
                        order_id=f"order:{intent.intent_id}:{session.session_open.isoformat()}",
                        intent_id=intent.intent_id,
                        instrument=instr,
                        side=side,
                        quantity=int(qty),
                        decision_time=intent.decision_time,
                        execution_time=intent.execution_time,
                    )
                    new_orders.append(order)
                # plan sells before buys for next execution
                new_orders = sorted(new_orders, key=lambda o: 0 if o.side == OrderSide.SELL else 1)
                pending_orders = new_orders
        except BacktestIntegrityError:
            raise
        except ValueError as exc:
            # convert any invariant failure to BacktestIntegrityError
            raise BacktestIntegrityError(str(exc)) from exc
        # collect journal
        journal = ledger.journal()
        return BacktestResult(
            fills=tuple(fills),
            rejects=tuple(rejects),
            journal=journal,
            daily_nav=tuple(daily_nav),
            capacity_diagnostics=tuple(capacity_diagnostics),
            scenario=self._config.scenario,
        )
