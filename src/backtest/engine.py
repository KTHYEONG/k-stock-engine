"""Session-timeline backtest engine over dense market arrays."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from src.backtest.costs import CostConfig, Side, fill_cost
from src.backtest.events import EngineEvents, ExitKind
from src.backtest.execution import ExecutionConfig, Fill, Order, Reject, price_orders
from src.backtest.ledger import JournalEntry, Ledger, NavRecord
from src.backtest.market import MarketArrays
from src.core.market_rules import KrxMarketRules


class DelistPolicy(StrEnum):
    LAST_CLOSE = "last_close"
    ZERO = "zero"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    initial_cash: int
    execution: ExecutionConfig
    costs: CostConfig
    halted_exit_policy: DelistPolicy
    cash_buffer: float

    def __post_init__(self) -> None:
        buffer = self.cash_buffer
        if (
            isinstance(buffer, bool)
            or not isinstance(buffer, (int, float))
            or not 0.0 <= float(buffer) < 1.0
        ):
            raise ValueError(f"cash_buffer must be in [0, 1), got {buffer!r}")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    nav: tuple[NavRecord, ...]
    fills: tuple[Fill, ...]
    rejects: tuple[Reject, ...]
    journal: tuple[JournalEntry, ...]
    dividends_integrated: bool
    ledger_hash: str


def _target_orders(
    *,
    weights: Mapping[int, float],
    arrays: MarketArrays,
    t: int,
    holdings: Mapping[int, int],
    nav: int,
    buffer_scale: float,
) -> list[Order]:
    orders: list[Order] = []
    close = arrays.int_fields["close"][t]
    present = arrays.bool_fields["present"][t]
    for instrument_idx, weight in weights.items():
        if not present[instrument_idx] or int(close[instrument_idx]) == 0:
            continue
        target_q = math.floor(weight * float(nav) * buffer_scale / float(close[instrument_idx]))
        delta = target_q - holdings.get(instrument_idx, 0)
        if delta > 0:
            orders.append(Order(instrument_idx, Side.BUY, delta, t))
        elif delta < 0:
            orders.append(Order(instrument_idx, Side.SELL, -delta, t))
    for instrument_idx, held in holdings.items():
        if instrument_idx not in weights:
            orders.append(Order(instrument_idx, Side.SELL, held, t))
    return orders


def _apply_fills(
    *, ledger: Ledger, fills: Sequence[Fill], t: int, arrays: MarketArrays, config: EngineConfig
) -> tuple[list[Fill], list[Reject], list[Order]]:
    sells = [fill for fill in fills if fill.order.side is Side.SELL]
    buys = sorted(
        (fill for fill in fills if fill.order.side is Side.BUY),
        key=lambda fill: fill.quantity * fill.price,
        reverse=True,
    )
    applied: list[Fill] = []
    rejects: list[Reject] = []
    carried: list[Order] = []
    for fill in (*sells, *buys):
        order = fill.order
        n = order.instrument_idx
        if order.side is Side.SELL:
            rate = Decimal(str(float(arrays.float_fields["sell_tax_rate"][t, n])))
            held = ledger.positions().get(n, 0)
            quantity = min(fill.quantity, held)
            shortfall = fill.quantity - quantity
            if quantity > 0:
                cost = fill_cost(
                    side=Side.SELL, quantity=quantity, price=fill.price, sell_tax_rate=rate,
                    config=config.costs,
                )
                ledger.sell(
                    session_idx=t, instrument_idx=n, quantity=quantity, price=fill.price,
                    commission=cost.commission, sell_tax=cost.sell_tax,
                )
                applied.append(Fill(order, quantity, fill.price) if shortfall else fill)
            if shortfall:
                rejects.append(Reject(order, "cash"))
                if config.execution.carry_unfilled:
                    carried.append(Order(n, Side.SELL, shortfall, t))
        else:
            quantity = min(fill.quantity, ledger.cash // fill.price)
            commission = 0
            while quantity > 0:
                commission = fill_cost(
                    side=Side.BUY, quantity=quantity, price=fill.price,
                    sell_tax_rate=Decimal(0), config=config.costs,
                ).commission
                if quantity * fill.price + commission <= ledger.cash:
                    break
                quantity -= 1
            shortfall = fill.quantity - quantity
            if quantity > 0:
                ledger.buy(
                    session_idx=t, instrument_idx=n, quantity=quantity, price=fill.price,
                    commission=commission,
                )
                applied.append(Fill(order, quantity, fill.price) if shortfall else fill)
            if shortfall:
                rejects.append(Reject(order, "cash"))
                if config.execution.carry_unfilled:
                    carried.append(Order(n, Side.BUY, shortfall, t))
    return (applied, rejects, carried)


def _validate_targets(
    targets: Mapping[int, Mapping[int, float]], *, lo: int, hi: int, n_instruments: int,
) -> dict[int, dict[int, float]]:
    cleaned: dict[int, dict[int, float]] = {}
    for row, weights in targets.items():
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError(f"target row must be an int, got {row!r}")
        d = int(row)
        if d < max(0, lo - 1) or d > hi - 1:
            raise ValueError(f"target row {d} is outside the run window")
        if not isinstance(weights, Mapping):
            raise ValueError(f"target weights at row {d} must be a mapping")
        total = 0.0
        row_weights: dict[int, float] = {}
        for instrument_idx, weight in weights.items():
            if isinstance(instrument_idx, bool) or not isinstance(instrument_idx, int):
                raise ValueError(f"target instrument must be an int at row {d}, got {instrument_idx!r}")
            n = int(instrument_idx)
            if not 0 <= n < n_instruments:
                raise ValueError(f"target instrument {n} at row {d} is outside the panel")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"target weight must be a finite number at row {d}, got {weight!r}")
            w = float(weight)
            if not math.isfinite(w) or w < 0.0:
                raise ValueError(f"target weight must be finite and non-negative at row {d}, got {weight!r}")
            total += w
            row_weights[n] = w
        if total > 1.0 + 1e-9:
            raise ValueError(f"target weights at row {d} sum to {total}, above 1")
        cleaned[d] = row_weights
    return cleaned


def run_backtest(
    *,
    arrays: MarketArrays,
    events: EngineEvents,
    targets: Mapping[int, Mapping[int, float]],
    config: EngineConfig,
    deposits: Mapping[date, int],
    rules: KrxMarketRules,
    start: date,
    end: date,
) -> BacktestResult:
    """Replay pre-decided target weights on the fixed session timeline.

    ``targets`` maps a decision session index d to instrument-index weights (fractions of decision-session
    NAV; the remainder is cash). Orders for row d execute at session d+1 (open auction), sells before buys,
    integer shares, no negative cash, T+2-consistent ledger. Per session order is unchanged: share factors,
    dividend entitlements, dividend payments, exits, deposits, execution, close mark, then decision rows.
    Why replay-only: decisions are produced upstream from the causal panel; the ledger's job is accounting
    truth at 10M KRW (integer shares, costs, settlement), not strategy logic.

    Raises: ValueError for a window outside the panel, a target row outside ``[start-1, end)``, or weights that
    are negative, non-finite or sum above 1.
    """
    sessions = list(arrays.sessions)
    session_index = {session: idx for idx, session in enumerate(sessions)}
    if start not in session_index or end not in session_index or session_index[start] > session_index[end]:
        raise ValueError(f"run window [{start}, {end}] is not within the panel sessions")
    lo, hi = session_index[start], session_index[end]
    n_instruments = len(arrays.instrument_ids)
    schedule = _validate_targets(targets, lo=lo, hi=hi, n_instruments=n_instruments)
    deposits_by_session: dict[int, int] = {}
    for pay_date, amount in deposits.items():
        idx = session_index.get(pay_date, bisect.bisect_right(sessions, pay_date))
        if idx < lo:
            idx = lo
        if idx > hi:
            raise ValueError(f"deposit on {pay_date} is after the run end {end}")
        deposits_by_session[idx] = deposits_by_session.get(idx, 0) + amount
    ledger = Ledger(initial_cash=config.initial_cash)
    exited: set[int] = set()
    pending: list[Order] = []
    fills: list[Fill] = []
    rejects: list[Reject] = []
    records: list[NavRecord] = []
    buffer_scale = 1.0 - float(config.cash_buffer)
    if lo - 1 in schedule and lo > 0:
        pending.extend(
            _target_orders(
                weights=schedule[lo - 1], arrays=arrays, t=lo - 1, holdings={},
                nav=config.initial_cash, buffer_scale=buffer_scale,
            )
        )
    for t in range(lo, hi + 1):
        for instrument_idx, factor, base_price in events.share_factor_by_session.get(t, ()):
            ledger.apply_share_factor(
                session_idx=t, instrument_idx=instrument_idx, factor=factor, base_price=base_price
            )
        ledger.record_dividend_entitlements(
            session_idx=t, events=events.dividends_by_ex_session.get(t, ())
        )
        ledger.settle_dividends(session_idx=t, config=config.costs)
        for exit_event in events.exits_by_session.get(t, ()):
            exited.add(exit_event.instrument_idx)
            if exit_event.instrument_idx in ledger.positions():
                if exit_event.kind is ExitKind.TRADED or config.halted_exit_policy is DelistPolicy.LAST_CLOSE:
                    exit_price = exit_event.last_close
                else:
                    exit_price = 0
                ledger.close_exit(
                    session_idx=t, instrument_idx=exit_event.instrument_idx, price=exit_price
                )
        if t in deposits_by_session:
            ledger.deposit(session_idx=t, amount=deposits_by_session[t])
        executable = [order for order in pending if order.instrument_idx not in exited]
        rejects.extend(Reject(order, "delisted") for order in pending if order.instrument_idx in exited)
        order_fills, order_rejects = price_orders(
            orders=executable, arrays=arrays, t=t, config=config.execution, costs=config.costs,
            rules=rules,
        )
        filled_orders = {fill.order for fill in order_fills}
        applied, app_rejects, carried = _apply_fills(
            ledger=ledger, fills=order_fills, t=t, arrays=arrays, config=config
        )
        fills.extend(applied)
        rejects.extend(order_rejects)
        rejects.extend(app_rejects)
        if config.execution.carry_unfilled:
            for reject in order_rejects:
                if reject.reason != "delisted" and reject.order not in filled_orders:
                    carried.append(
                        Order(
                            reject.order.instrument_idx, reject.order.side,
                            reject.order.quantity, t,
                        )
                    )
        record = ledger.mark(
            session_idx=t,
            close=arrays.int_fields["close"][t],
            present=arrays.bool_fields["present"][t],
        )
        records.append(record)
        if t in schedule and t >= lo:
            pending = carried + _target_orders(
                weights=schedule[t], arrays=arrays, t=t, holdings=ledger.positions(),
                nav=record.nav, buffer_scale=buffer_scale,
            )
        else:
            pending = carried
    journal = ledger.journal
    payload = {
        "journal": [
            {
                "session_idx": entry.session_idx,
                "kind": entry.kind.value,
                "instrument_idx": entry.instrument_idx,
                "cash_delta": entry.cash_delta,
                "quantity_delta": entry.quantity_delta,
            }
            for entry in journal
        ],
        "nav": [
            {
                "session_idx": record.session_idx,
                "cash": record.cash,
                "dividend_receivable": record.dividend_receivable,
                "market_value": record.market_value,
                "nav": record.nav,
                "external_flow": record.external_flow,
            }
            for record in records
        ],
    }
    ledger_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BacktestResult(
        nav=tuple(records),
        fills=tuple(fills),
        rejects=tuple(rejects),
        journal=journal,
        dividends_integrated=events.dividends_integrated,
        ledger_hash=ledger_hash,
    )
