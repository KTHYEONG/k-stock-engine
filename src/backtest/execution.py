"""T+1 auction execution pricing with participation caps and tick rounding."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.backtest.costs import CostConfig, Side, impact_fraction
from src.backtest.market import MarketArrays
from src.core.market_rules import KrxMarket, KrxMarketRules
from src.core.pit import PITDataError

_MARKET_BY_CODE: Mapping[int, KrxMarket] = {1: KrxMarket.KOSPI, 2: KrxMarket.KOSDAQ}


class ExecutionScenario(StrEnum):
    OPEN_AUCTION = "open_auction"
    CLOSE_AUCTION = "close_auction"
    VWAP_PROXY = "vwap_proxy"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    scenario: ExecutionScenario
    max_participation: float
    carry_unfilled: bool


@dataclass(frozen=True, slots=True)
class Order:
    instrument_idx: int
    side: Side
    quantity: int
    decision_session_idx: int


@dataclass(frozen=True, slots=True)
class Fill:
    order: Order
    quantity: int
    price: int


@dataclass(frozen=True, slots=True)
class Reject:
    order: Order
    reason: str


def _market_at(arrays: MarketArrays, t: int, n: int) -> KrxMarket:
    try:
        return _MARKET_BY_CODE[int(arrays.market[t, n])]
    except KeyError as exc:
        raise PITDataError(f"unknown market code at session {t}: {arrays.market[t, n]!r}") from exc


def _locked(*, arrays: MarketArrays, scenario: ExecutionScenario, t: int, n: int, is_buy: bool) -> bool:
    if scenario is ExecutionScenario.OPEN_AUCTION:
        if is_buy:
            return bool(arrays.bool_fields["open_at_upper"][t, n])
        return bool(arrays.bool_fields["open_at_lower"][t, n])
    if is_buy:
        return bool(arrays.int_fields["low"][t, n] == arrays.int_fields["upper_limit"][t, n])
    return bool(arrays.int_fields["high"][t, n] == arrays.int_fields["lower_limit"][t, n])


def _base_price(
    *,
    scenario: ExecutionScenario,
    arrays: MarketArrays,
    t: int,
    n: int,
    is_buy: bool,
    session: date,
    market: KrxMarket,
    rules: KrxMarketRules,
) -> float:
    if scenario is ExecutionScenario.OPEN_AUCTION:
        return float(arrays.int_fields["open"][t, n])
    if scenario is ExecutionScenario.CLOSE_AUCTION:
        return float(arrays.int_fields["close"][t, n])
    raw = (
        float(arrays.int_fields["high"][t, n])
        + float(arrays.int_fields["low"][t, n])
        + float(arrays.int_fields["close"][t, n])
    ) / 3.0
    tick = rules.tick_size(session=session, market=market, price=max(1, int(raw)))
    half_tick = tick / 2.0
    return raw + half_tick if is_buy else raw - half_tick


def price_orders(
    *,
    orders: Sequence[Order],
    arrays: MarketArrays,
    t: int,
    config: ExecutionConfig,
    costs: CostConfig,
    rules: KrxMarketRules,
) -> tuple[tuple[Fill, ...], tuple[Reject, ...]]:
    """Price and size orders decided at ``t - 1`` for execution in session ``t``.

    The open auction is the default because a retail account decides after the
    18:00 data release and submits pre-open orders; every participant clears
    at the single auction price, so no spread is paid, but an impact premium is
    charged. Sizing uses only decision-session (``t - 1``) liquidity, because
    the session-``t`` volume is unknown when the order is sent.
    """
    session = arrays.sessions[t]
    prev = t - 1
    fills: list[Fill] = []
    rejects: list[Reject] = []
    for order in orders:
        if order.decision_session_idx != prev:
            raise PITDataError(
                f"order decided at {order.decision_session_idx} cannot execute at {t}"
            )
        if order.side is not Side.BUY and order.side is not Side.SELL:
            raise PITDataError(f"order side must be Side.BUY or Side.SELL, got {order.side!r}")
        n = order.instrument_idx
        is_buy = order.side is Side.BUY
        if not arrays.bool_fields["present"][t, n]:
            rejects.append(Reject(order, "missing_price"))
            continue
        if int(arrays.int_fields["volume"][t, n]) == 0:
            rejects.append(Reject(order, "halted"))
            continue
        market = _market_at(arrays, t, n)
        base = _base_price(
            scenario=config.scenario, arrays=arrays, t=t, n=n, is_buy=is_buy, session=session,
            market=market, rules=rules,
        )
        if base <= 0.0:
            rejects.append(Reject(order, "missing_price"))
            continue
        if _locked(arrays=arrays, scenario=config.scenario, t=t, n=n, is_buy=is_buy):
            rejects.append(Reject(order, "limit_locked"))
            continue
        adtv = float(arrays.float_fields["adtv20"][prev, n])
        decision_price = float(arrays.int_fields["close"][prev, n])
        if not math.isfinite(adtv) or adtv <= 0.0 or decision_price <= 0.0:
            cap_quantity = 0
        else:
            cap_quantity = math.floor(config.max_participation * adtv / decision_price)
        if cap_quantity <= 0:
            rejects.append(Reject(order, "capacity"))
            continue
        fill_quantity = order.quantity if order.quantity < cap_quantity else cap_quantity
        fraction = impact_fraction(
            notional=float(fill_quantity) * base,
            adtv20=adtv,
            vol60=float(arrays.float_fields["ret_vol60"][prev, n]),
            config=costs,
        )
        adverse = base * (1.0 + fraction) if is_buy else base * (1.0 - fraction)
        tick = rules.tick_size(session=session, market=market, price=max(1, int(base)))
        if is_buy:
            price = math.ceil(adverse / tick) * tick
            upper = int(arrays.int_fields["upper_limit"][t, n])
            if price > upper:
                price = upper
        else:
            price = math.floor(adverse / tick) * tick
            lower = int(arrays.int_fields["lower_limit"][t, n])
            if price < lower:
                price = lower
        fills.append(Fill(order, fill_quantity, price))
        if fill_quantity < order.quantity:
            rejects.append(Reject(order, "capacity"))
    return (tuple(fills), tuple(rejects))
