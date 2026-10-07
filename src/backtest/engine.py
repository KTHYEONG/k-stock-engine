"""Session-timeline backtest engine over dense market arrays."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from src.backtest.costs import CostConfig, Side, fill_cost
from src.backtest.events import EngineEvents, ExitKind
from src.backtest.execution import ExecutionConfig, Fill, Order, Reject, price_orders
from src.backtest.ledger import JournalEntry, JournalKind, Ledger, LedgerAccount, NavRecord
from src.backtest.market import MarketArrays
from src.backtest.overlay import (
    PRIMARY_LEG,
    DerivativeConfig,
    OverlayMarket,
    OverlayPolicy,
    OverlayState,
    OverlayTarget,
    ceil_amount_krw,
    futures_expiry_rows,
    required_reserve_krw,
)
from src.core.market_rules import KrxMarketRules
from src.core.pit import PITDataError


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
    stock_book_returns: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )


def _target_orders(
    *,
    weights: Mapping[int, float],
    arrays: MarketArrays,
    t: int,
    holdings: Mapping[int, int],
    nav: int,
    buffer_scale: float,
    rebalance_band: float = 0.0,
) -> list[Order]:
    """Orders moving holdings toward ``weights``; resizes of continuing positions inside the band are skipped."""
    orders: list[Order] = []
    close = arrays.int_fields["close"][t]
    present = arrays.bool_fields["present"][t]
    for instrument_idx, weight in weights.items():
        if not present[instrument_idx] or int(close[instrument_idx]) == 0:
            continue
        price = float(close[instrument_idx])
        target_q = math.floor(weight * float(nav) * buffer_scale / price)
        held = holdings.get(instrument_idx, 0)
        delta = target_q - held
        if (
            rebalance_band > 0.0
            and held > 0
            and float(weight) > 0.0
            and target_q > 0
            and abs(delta) * price < rebalance_band * target_q * price
        ):
            continue
        if delta > 0:
            orders.append(Order(instrument_idx, Side.BUY, delta, t))
        elif delta < 0:
            orders.append(Order(instrument_idx, Side.SELL, -delta, t))
    for instrument_idx, held in holdings.items():
        if instrument_idx not in weights:
            orders.append(Order(instrument_idx, Side.SELL, held, t))
    return orders


def _spendable_cash(ledger: Ledger) -> int:
    """Cash an order may commit; an unpaid tax liability is never spent on stocks or on the hedge."""
    return max(ledger.cash - ledger.tax_payable, 0)


def _apply_sell_fills(
    *, ledger: Ledger, fills: Sequence[Fill], t: int, arrays: MarketArrays, config: EngineConfig
) -> tuple[list[Fill], list[Reject], list[Order]]:
    applied: list[Fill] = []
    rejects: list[Reject] = []
    carried: list[Order] = []
    for fill in fills:
        order = fill.order
        n = order.instrument_idx
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
    return (applied, rejects, carried)


def _apply_buy_fills(
    *, ledger: Ledger, fills: Sequence[Fill], t: int, arrays: MarketArrays, config: EngineConfig
) -> tuple[list[Fill], list[Reject], list[Order]]:
    ordered = sorted(fills, key=lambda fill: fill.quantity * fill.price, reverse=True)
    applied: list[Fill] = []
    rejects: list[Reject] = []
    carried: list[Order] = []
    for fill in ordered:
        order = fill.order
        n = order.instrument_idx
        spendable = _spendable_cash(ledger)
        quantity = min(fill.quantity, spendable // fill.price) if fill.price > 0 else 0
        commission = 0
        while quantity > 0:
            commission = fill_cost(
                side=Side.BUY, quantity=quantity, price=fill.price,
                sell_tax_rate=Decimal(0), config=config.costs,
            ).commission
            if quantity * fill.price + commission <= spendable:
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


# KRW movements booked on the cash account that finance the overlay rather than the stock book: they are
# stripped from the stock-book return so hedge financing never registers as stock performance.
_CASH_FLOW_KINDS = frozenset(
    {
        JournalKind.DEPOSIT,
        JournalKind.MARGIN_TRANSFER,
        JournalKind.INVERSE_BUY,
        JournalKind.INVERSE_SELL,
        JournalKind.INVERSE_COMMISSION,
        JournalKind.INVERSE_TAX,
    }
)


def _index_level(values: NDArray[np.float64], t: int) -> float:
    """Index level of an already-validated window; the run rejects a missing level before it starts."""
    return float(values[t])


def _inverse_int(values: NDArray[np.float64], t: int) -> int | None:
    out = float(values[t])
    if not math.isfinite(out) or out <= 0.0:
        return None
    return int(out)


def _cash_flow_krw(ledger: Ledger, since: int) -> int:
    """Net cash-account KRW booked since journal index ``since`` that does not belong to the stock book."""
    return sum(
        entry.cash_delta
        for entry in ledger.journal_since(since)
        if entry.account is LedgerAccount.CASH and entry.kind in _CASH_FLOW_KINDS
    )


def _futures_commission(rate: float, *, delta: int, multiplier: int, level: float) -> int:
    return ceil_amount_krw(rate, abs(delta), multiplier, level)


def _maintenance_floor(contracts: int, *, level: float, config: DerivativeConfig, multiplier: int) -> int:
    """``ceil(trigger · initial_margin_rate · |contracts| · multiplier · level)`` (``0`` when flat)."""
    if contracts == 0:
        return 0
    rate = Decimal(str(float(config.margin_topup_trigger_fraction))) * Decimal(
        str(float(config.initial_margin_rate))
    )
    return ceil_amount_krw(rate, abs(int(contracts)), multiplier, float(level))


def _largest_fundable_contracts(
    *,
    target_contracts: int,
    held: int,
    available: int,
    level: float,
    config: DerivativeConfig,
    multiplier: int,
) -> int:
    """Largest-magnitude ``k`` with the sign of ``target_contracts`` and ``|k| <= |target_contracts|`` such that
    ``required_reserve(k) + commission(k - held) <= available``, or ``0`` when none qualifies.

    Why: choosing the funded contract count must not scale with the contract count. When the account is far larger
    than the hedge target's notional (or a corrupted replay inflates it), a linear scan from the target costs
    seconds per call; the reserve is non-decreasing in ``|k|`` and the commission is never negative, so no ``k``
    whose reserve alone exceeds ``available`` can qualify and the search can start at the reserve bound.

    Args:
        target_contracts: Pending overlay target (signed; negative = net long).
        held: Contracts currently held (signed); only the commission term depends on it.
        available: Margin plus spendable cash in whole KRW (may be <= 0).
        level: Positive finite futures-underlying level.
        config: Derivative terms (margin and buffer rates, cost rate).
        multiplier: KRW per index point per contract.

    Returns:
        The chosen contract count; identical to the first qualifying candidate of a magnitude-descending scan from
        ``target_contracts`` toward zero.

    Raises:
        PITDataError / ValueError: propagated unchanged from ``required_reserve_krw`` for a non-positive or
            non-finite ``level``.
    """
    if target_contracts == 0:
        return 0
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        required_reserve_krw(contracts=target_contracts, level=level, config=config)
    level_f = float(level)
    if not math.isfinite(level_f) or level_f <= 0.0:
        required_reserve_krw(contracts=target_contracts, level=level, config=config)
    if available < 0:
        return 0
    rate = Decimal(str(float(config.initial_margin_rate))) + Decimal(
        str(float(config.margin_buffer_rate))
    )
    capacity = rate * Decimal(int(multiplier)) * Decimal(str(level_f))
    bound = int(
        (Decimal(int(available)) / capacity).to_integral_value(rounding=ROUND_FLOOR)
    )
    target_int = int(target_contracts)
    sign = 1 if target_int > 0 else -1
    start_mag = min(abs(target_int), bound)
    cost_rate = float(config.futures_cost_rate)
    held_int = int(held)
    for magnitude in range(start_mag, -1, -1):
        candidate = sign * magnitude
        reserve = required_reserve_krw(contracts=candidate, level=level, config=config)
        commission = _futures_commission(
            cost_rate, delta=candidate - held_int, multiplier=int(multiplier), level=level,
        )
        if reserve + commission <= available:
            return candidate
    return 0


def _sync_futures(
    ledger: Ledger,
    *,
    session_idx: int,
    target_contracts: int,
    level: float,
    config: DerivativeConfig,
    multiplier: int,
    leg_targets: Mapping[str, int] | None = None,
    leg_markets: Mapping[str, tuple[DerivativeConfig, float]] | None = None,
) -> None:
    """Trade every leg toward its pending target and hold the pooled margin at exactly the summed
    ``required_reserve(k)`` — except inside the band.

    Legs are the primary plus each secondary leg that is targeted or held (``leg_markets`` maps a secondary name to
    its terms and current level). ``h`` held, ``k*`` pending target, ``R`` reserve, ``M`` maintenance floor, all
    summed over legs; ``available`` = margin + free cash - tax_payable. (a) Hold (``k* == h`` on every leg and
    margin >= ``M``): no futures trade and no top-up; excess above ``R`` is still released, but anything inside
    ``[M, R]`` is left alone. (b) Otherwise legs are funded in order (primary, then secondary names ascending): each
    takes the largest same-side ``|k| <= |k*|`` with ``R(k) + commission <= `` what earlier legs left of
    ``available``, then margin syncs to the summed ``R``. While margin >= ``M``, a leg already at its target keeps
    its position. Unaffordable targets shrink (down to flat) instead of borrowing, and a margin the cash cannot
    restore stays negative for a later session to repay.

    Why the band: ordinary daily variation must never cost a trade — the previous always-sync rule liquidated a
    cash-less hedge on any up day (1 contract, cash 0, +1% → margin 1.40M ≥ M 757,500 but < R 1,515,000 → closed).
    For the same reason, opening a secondary leg must never liquidate a primary leg that sits inside the band.
    """
    markets = leg_markets or {}
    targets = dict(leg_targets or {})
    names = sorted(set(targets) | {name for name, _ in ledger.legs})
    legs: list[tuple[str, DerivativeConfig, float, int, int]] = [
        (PRIMARY_LEG, config, level, int(multiplier), target_contracts)
    ]
    for name in names:
        leg_config, leg_level = markets[name]
        legs.append((name, leg_config, leg_level, int(leg_config.contract_multiplier_krw), targets.get(name, 0)))
    held = {name: ledger.leg_contracts(name) for name, *_ in legs}
    floor = sum(
        _maintenance_floor(held[name], level=lvl, config=cfg, multiplier=mult) for name, cfg, lvl, mult, _ in legs
    )
    in_band = ledger.margin >= floor
    if in_band and all(target == held[name] for name, *_, target in legs):
        reserve = sum(required_reserve_krw(contracts=held[name], level=lvl, config=cfg) for name, cfg, lvl, *_ in legs)
        if ledger.margin > reserve:
            ledger.transfer_margin(session_idx=session_idx, amount=reserve - ledger.margin)
        return
    remaining = ledger.margin + _spendable_cash(ledger)
    for name, cfg, lvl, mult, target in legs:
        prior = held[name]
        if in_band and target == prior:
            chosen = prior
        else:
            chosen = _largest_fundable_contracts(
                target_contracts=target, held=prior, available=remaining,
                level=lvl, config=cfg, multiplier=mult,
            )
        if chosen != prior:
            ledger.trade_futures(
                session_idx=session_idx,
                contracts=chosen,
                level=lvl,
                multiplier=mult,
                cost_rate=float(cfg.futures_cost_rate),
                leg=name,
            )
        remaining -= required_reserve_krw(contracts=chosen, level=lvl, config=cfg) + _futures_commission(
            float(cfg.futures_cost_rate), delta=chosen - prior, multiplier=mult, level=lvl,
        )
    need = (
        sum(required_reserve_krw(contracts=ledger.leg_contracts(name), level=lvl, config=cfg) for name, cfg, lvl, *_ in legs)
        - ledger.margin
    )
    if need > 0:
        ledger.transfer_margin(session_idx=session_idx, amount=min(need, _spendable_cash(ledger)))
    elif need < 0:
        ledger.transfer_margin(session_idx=session_idx, amount=need)


def _sync_inverse(
    ledger: Ledger,
    *,
    session_idx: int,
    target_value_krw: int,
    price: int,
    config: DerivativeConfig,
) -> None:
    """Trade the inverse ETF toward ``floor(target_value / price)`` units: sells first, buys within free cash.

    Why sells first: the sale proceeds and the disposal tax are settled together, so a reduction is always
    fundable while a purchase is capped by cash that owes no tax.
    """
    rate = float(config.inverse_cost_rate)
    desired = max(int(target_value_krw) // price, 0)
    held = ledger.inverse_units
    if desired < held:
        ledger.sell_inverse(
            session_idx=session_idx,
            units=held - desired,
            price=price,
            cost_rate=rate,
            tax_rate=config.inverse_tax_rate,
        )
        return
    spendable = _spendable_cash(ledger)
    affordable = min(desired - held, spendable // price)
    while affordable > 0 and affordable * price + ceil_amount_krw(rate, affordable, price) > spendable:
        affordable -= 1
    if affordable > 0:
        ledger.buy_inverse(session_idx=session_idx, units=affordable, price=price, cost_rate=rate)


def _policy_target(policy: OverlayPolicy, state: OverlayState) -> OverlayTarget | None:
    target = policy.target(state)
    if target is not None and not isinstance(target, OverlayTarget):
        raise ValueError("overlay policy must return an OverlayTarget or None")
    return target


def _check_target_legs(
    target: OverlayTarget | None,
    *,
    leg_levels: Mapping[str, NDArray[np.float64]],
    leg_configs: Mapping[str, DerivativeConfig],
    checked: set[str],
    rows: range,
    sessions: Sequence[date],
) -> None:
    """Reject a target leg without terms; verify each targeted leg's levels over the run once (added to ``checked``).

    Why at first targeting rather than up front: a configured leg the policy never trades must not demand history.
    """
    if target is None:
        return
    for leg_name, _ in target.legs:
        if leg_name in checked:
            continue
        if leg_name not in leg_configs:
            raise ValueError(f"unknown target leg {leg_name!r}: no leg_derivatives entry")
        levels = leg_levels[leg_name]
        for row in rows:
            level = float(levels[row])
            if not math.isfinite(level) or level <= 0.0:
                raise PITDataError(f"leg {leg_name!r} level missing at {sessions[row].isoformat()}")
        checked.add(leg_name)


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
    cash_returns: NDArray[np.float64] | None = None,
    overlay: OverlayPolicy | None = None,
    overlay_market: OverlayMarket | None = None,
    derivatives: DerivativeConfig | None = None,
    leg_derivatives: Mapping[str, DerivativeConfig] | None = None,
    rebalance_band: float = 0.0,
) -> BacktestResult:
    """Replay pre-decided target weights on the fixed session timeline.

    ``targets`` maps a decision session index d to instrument-index weights (fractions of decision-session
    NAV; the remainder is cash). Orders for row d execute at session d+1 (open auction), sells before buys,
    integer shares, no negative cash, T+2-consistent ledger. Per session order is: share factors, dividend
    entitlements, dividend payments, exits, deposits, cash yield, sells, tax-payable payment (before any buy),
    buys, then the close phase — variation margin, quarterly roll, overlay execution and margin maintenance,
    inverse ETF, year-end taxes, close mark — and finally the decision rows.
    Why replay-only: decisions are produced upstream from the causal panel; the ledger's job is accounting
    truth at 10M KRW (integer shares, costs, settlement), not strategy logic.
    ``cash_returns`` (aligned to ``arrays.sessions``; element t = cash-ETF close_t /
    close_{t-1} - 1) credits the cash held overnight before session t's orders. ``None`` disables the sweep.
    The cash-yield tax is assessed on the last session of each calendar year and on the run's last session,
    before the close mark, so every ``NavRecord`` already reflects it and ``nav[-1].cash`` equals the journal sum.

    With an overlay, ``targets`` are fractions of the **stock book**; the engine sizes stock orders against
    ``nav - required_reserve(target contracts) - target inverse value`` so the cash the hedge needs at the close
    is left unspent by the open-auction stock trades. The overlay target decided after session d executes in
    session d+1: futures at the index close of d+1, the inverse ETF at its close of d+1. Overlay costs and the
    derivative tax are paid from the margin account; a run never aborts for lack of cash - hedges shrink instead.

    Raises: ValueError for a window outside the panel, a target row outside ``[start-1, end)``, weights that are
    negative, non-finite or sum above 1, or a partial overlay configuration.
    PITDataError: a missing index level in the window, or a missing inverse close while units are held or targeted.
    """
    sessions = list(arrays.sessions)
    session_index = {session: idx for idx, session in enumerate(sessions)}
    if start not in session_index or end not in session_index or session_index[start] > session_index[end]:
        raise ValueError(f"run window [{start}, {end}] is not within the panel sessions")
    lo, hi = session_index[start], session_index[end]
    n_instruments = len(arrays.instrument_ids)
    if isinstance(rebalance_band, bool) or not math.isfinite(float(rebalance_band)) or not 0.0 <= float(rebalance_band) < 1.0:
        raise ValueError(f"rebalance_band must satisfy 0 <= b < 1, got {rebalance_band!r}")
    band = float(rebalance_band)
    schedule = _validate_targets(targets, lo=lo, hi=hi, n_instruments=n_instruments)
    given = (overlay is not None, overlay_market is not None, derivatives is not None)
    if any(given) and not all(given):
        raise ValueError("overlay, overlay_market and derivatives must be given together or not at all")
    if leg_derivatives and overlay is None:
        raise ValueError("leg_derivatives requires an overlay")
    sweep: NDArray[np.float64] | None = None
    if cash_returns is not None:
        sweep = np.asarray(cash_returns, dtype=np.float64)
        if sweep.shape != (len(sessions),):
            raise ValueError(
                f"cash_returns must have shape ({len(sessions)},), got {sweep.shape}"
            )
    index_levels: NDArray[np.float64] = np.empty(0, dtype=np.float64)
    inverse_closes: NDArray[np.float64] = np.empty(0, dtype=np.float64)
    expiry_rows: frozenset[int] = frozenset()
    multiplier = 0
    leg_configs: dict[str, DerivativeConfig] = {}
    leg_levels: dict[str, NDArray[np.float64]] = {}
    checked_legs: set[str] = set()
    if overlay is not None:
        assert overlay_market is not None
        assert derivatives is not None
        index_levels = np.asarray(overlay_market.index_level, dtype=np.float64)
        inverse_closes = np.asarray(overlay_market.inverse_close, dtype=np.float64)
        if index_levels.shape != (len(sessions),) or inverse_closes.shape != (len(sessions),):
            raise ValueError("overlay_market arrays must align to sessions")
        for row in range(max(0, lo - 1), hi + 1):
            level = float(index_levels[row])
            if not math.isfinite(level) or level <= 0.0:
                raise PITDataError(f"index level missing at {sessions[row].isoformat()}")
        expiry_rows = futures_expiry_rows(sessions)
        multiplier = int(derivatives.contract_multiplier_krw)

        for leg_name, leg_config in (leg_derivatives or {}).items():
            if leg_name not in overlay_market.legs:
                raise ValueError(f"leg {leg_name!r} in leg_derivatives is absent from overlay_market.legs")
            if (
                leg_config.futures_tax_rate != derivatives.futures_tax_rate
                or leg_config.futures_annual_deduction_krw != derivatives.futures_annual_deduction_krw
            ):
                raise ValueError(
                    f"leg {leg_name!r} futures_tax_rate / futures_annual_deduction_krw differ from the primary's; "
                    "derivative tax is assessed on one aggregate"
                )
            leg_level = np.asarray(overlay_market.legs[leg_name], dtype=np.float64)
            if leg_level.shape != (len(sessions),):
                raise ValueError(f"leg {leg_name!r} levels must align to sessions")
            leg_configs[leg_name] = leg_config
            leg_levels[leg_name] = leg_level

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
                nav=config.initial_cash, buffer_scale=buffer_scale, rebalance_band=band,
            )
        )
    pending_overlay: OverlayTarget | None = None
    if overlay is not None and lo > 0:
        # The row before the window decides on the first session's hedge; its returns series is still empty.
        pending_overlay = _policy_target(
            overlay,
            OverlayState(
                session_idx=lo - 1,
                nav=int(config.initial_cash),
                stock_book_nav=int(config.initial_cash),
                stock_book_returns=np.zeros(0, dtype=np.float64),
                index_returns=np.zeros(0, dtype=np.float64),
                index_level=_index_level(index_levels, lo - 1),
                contracts=0,
                inverse_units=0,
                leg_contracts=(),
            ),
        )
        _check_target_legs(
            pending_overlay, leg_levels=leg_levels, leg_configs=leg_configs, checked=checked_legs,
            rows=range(max(0, lo - 1), hi + 1), sessions=sessions,
        )
    stock_book_returns: list[float] = []
    stock_book_history: list[float] = []
    index_history: list[float] = []
    stock_book_prev = int(config.initial_cash)

    for t in range(lo, hi + 1):
        journal_start = ledger.journal_size
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
        if sweep is not None:
            rate = float(sweep[t])
            if not math.isfinite(rate):
                raise PITDataError(f"cash return missing at {sessions[t].isoformat()}")
            ledger.accrue_cash_yield(session_idx=t, gross_return=rate)
        executable = [order for order in pending if order.instrument_idx not in exited]
        rejects.extend(Reject(order, "delisted") for order in pending if order.instrument_idx in exited)
        order_fills, order_rejects = price_orders(
            orders=executable, arrays=arrays, t=t, config=config.execution, costs=config.costs,
            rules=rules,
        )
        filled_orders = {fill.order for fill in order_fills}
        sell_fills = [f for f in order_fills if f.order.side is Side.SELL]
        buy_fills = [f for f in order_fills if f.order.side is Side.BUY]
        applied_s, rej_s, carried_s = _apply_sell_fills(
            ledger=ledger, fills=sell_fills, t=t, arrays=arrays, config=config
        )
        fills.extend(applied_s)
        rejects.extend(rej_s)
        ledger.settle_tax_payable(session_idx=t)
        applied_b, rej_b, carried_b = _apply_buy_fills(
            ledger=ledger, fills=buy_fills, t=t, arrays=arrays, config=config
        )
        fills.extend(applied_b)
        rejects.extend(rej_b)
        rejects.extend(order_rejects)
        carried: list[Order] = []
        carried.extend(carried_s)
        carried.extend(carried_b)
        if config.execution.carry_unfilled:
            carried.extend(
                Order(
                    reject.order.instrument_idx, reject.order.side,
                    reject.order.quantity, t,
                )
                for reject in order_rejects
                if reject.reason != "delisted" and reject.order not in filled_orders
            )
        inverse_price: int | None = None
        level_t = math.nan
        if overlay is not None:
            assert derivatives is not None
            level_t = _index_level(index_levels, t)
            inverse_price = _inverse_int(inverse_closes, t)

            if ledger.contracts != 0:
                ledger.settle_variation(
                    session_idx=t,
                    prev_level=_index_level(index_levels, t - 1),
                    level=level_t,
                    multiplier=multiplier,
                )
            for leg_name, _ in ledger.legs:
                ledger.settle_variation(
                    session_idx=t,
                    prev_level=_index_level(leg_levels[leg_name], t - 1),
                    level=_index_level(leg_levels[leg_name], t),
                    multiplier=int(leg_configs[leg_name].contract_multiplier_krw),
                    leg=leg_name,
                )
            if t in expiry_rows:
                if ledger.contracts != 0:
                    ledger.roll_futures(
                        session_idx=t, level=level_t, multiplier=multiplier,
                        cost_rate=float(derivatives.futures_cost_rate),
                    )
                for leg_name, _ in ledger.legs:
                    ledger.roll_futures(
                        session_idx=t,
                        level=_index_level(leg_levels[leg_name], t),
                        multiplier=int(leg_configs[leg_name].contract_multiplier_krw),
                        cost_rate=float(leg_configs[leg_name].futures_cost_rate),
                        leg=leg_name,
                    )
            _sync_futures(
                ledger, session_idx=t,
                target_contracts=(
                    pending_overlay.contracts if pending_overlay is not None else ledger.contracts
                ),
                level=level_t, config=derivatives, multiplier=multiplier,
                leg_targets=dict(pending_overlay.legs if pending_overlay is not None else ledger.legs),
                leg_markets={
                    leg_name: (leg_configs[leg_name], _index_level(leg_levels[leg_name], t))
                    for leg_name in leg_configs
                },
            )
            inverse_value_krw = (
                pending_overlay.inverse_value_krw
                if pending_overlay is not None
                else ledger.inverse_units * (inverse_price or 0)
            )
            if inverse_value_krw > 0 or ledger.inverse_units > 0:
                if inverse_price is None:
                    raise PITDataError(f"inverse close missing at {sessions[t].isoformat()}")
                _sync_inverse(
                    ledger, session_idx=t, target_value_krw=inverse_value_krw,
                    price=inverse_price, config=derivatives,
                )
            pending_overlay = None
        year_end = t == hi or sessions[t].year != sessions[t + 1].year
        if year_end:
            if sweep is not None:
                ledger.settle_cash_yield_tax(session_idx=t, config=config.costs)
            if overlay is not None:
                assert derivatives is not None
                ledger.settle_futures_tax(session_idx=t, config=derivatives)
        record = ledger.mark(
            session_idx=t,
            close=arrays.int_fields["close"][t],
            present=arrays.bool_fields["present"][t],
            inverse_price=inverse_price,
        )
        records.append(record)
        flow = _cash_flow_krw(ledger, journal_start)
        stock_book = (
            record.cash + record.dividend_receivable + record.market_value - record.tax_payable
        )
        session_return = (
            (stock_book - flow) / stock_book_prev - 1.0
            if t > lo and stock_book_prev > 0
            else math.nan
        )
        stock_book_returns.append(session_return)
        stock_book_history.append(session_return)
        if overlay is not None:
            index_history.append(
                math.nan
                if t == lo
                else level_t / _index_level(index_levels, t - 1) - 1.0
            )
        stock_book_prev = stock_book
        sizing_nav = int(record.nav)
        target_contracts = ledger.contracts
        target_inverse_krw = int(record.inverse_value)
        target_legs = ledger.legs
        if overlay is not None:
            assert derivatives is not None
            pending_overlay = _policy_target(
                overlay,
                OverlayState(
                    session_idx=t,
                    nav=int(record.nav),
                    stock_book_nav=(
                        record.cash
                        + record.dividend_receivable
                        + record.market_value
                        - record.tax_payable
                    ),
                    stock_book_returns=np.asarray(stock_book_history, dtype=np.float64),
                    index_returns=np.asarray(index_history, dtype=np.float64),
                    index_level=level_t,
                    contracts=int(ledger.contracts),
                    inverse_units=int(ledger.inverse_units),
                    leg_contracts=ledger.legs,
                ),
            )
            if pending_overlay is not None:
                target_contracts = int(pending_overlay.contracts)
                target_inverse_krw = int(pending_overlay.inverse_value_krw)
                target_legs = pending_overlay.legs
            _check_target_legs(
                pending_overlay, leg_levels=leg_levels, leg_configs=leg_configs, checked=checked_legs,
                rows=range(max(0, lo - 1), hi + 1), sessions=sessions,
            )
            reserve = required_reserve_krw(contracts=target_contracts, level=level_t, config=derivatives) + sum(
                required_reserve_krw(
                    contracts=count, level=_index_level(leg_levels[leg_name], t), config=leg_configs[leg_name]
                )
                for leg_name, count in target_legs
            )
            sizing_nav -= reserve + target_inverse_krw
            sizing_nav = max(sizing_nav, 0)
        if t in schedule and t >= lo:
            pending = carried + _target_orders(
                weights=schedule[t], arrays=arrays, t=t, holdings=ledger.positions(),
                nav=sizing_nav, buffer_scale=buffer_scale, rebalance_band=band,
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
                "account": entry.account.value,
                **({"leg": entry.leg} if entry.leg is not None else {}),
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
                "margin": record.margin,
                "inverse_value": record.inverse_value,
                "tax_payable": record.tax_payable,
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
        stock_book_returns=np.asarray(stock_book_returns, dtype=np.float64),
    )
