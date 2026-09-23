"""Date-effective KRX equity market rules (tick size, sell tax, price limits)."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

from src.core.pit import PITDataError

VERSION = "krx-market-rules-v1"
_TOP_LEVEL_KEYS = frozenset({"version", "sources", "tick_regimes", "sell_tax_regimes", "price_limit_regimes"})
_MARKETS = ("KOSPI", "KOSDAQ")


class KrxMarket(StrEnum):
    KOSPI = "KOSPI"
    KOSDAQ = "KOSDAQ"


@dataclass(frozen=True, slots=True)
class TickBand:
    lower_price_inclusive: int
    tick: int


@dataclass(frozen=True, slots=True)
class TickRegime:
    effective_from: date
    bands: Mapping[KrxMarket, tuple[TickBand, ...]]


@dataclass(frozen=True, slots=True)
class SellTaxRegime:
    effective_from: date
    rates: Mapping[KrxMarket, Decimal]


@dataclass(frozen=True, slots=True)
class PriceLimitRegime:
    effective_from: date
    ratio: Decimal


@dataclass(frozen=True, slots=True)
class KrxMarketRules:
    """Date-effective KRX equity rules used by data certification and fill simulation.

    Statutory schedules change on known dates; a single constant silently
    misprices every trade outside its era. Lookups are exact by KST session
    date and fail closed outside declared coverage so no caller can extend a
    rule into an undocumented period.
    """

    version: str
    tick_regimes: tuple[TickRegime, ...]
    sell_tax_regimes: tuple[SellTaxRegime, ...]
    price_limit_regimes: tuple[PriceLimitRegime, ...]

    def tick_regime_at(self, session: date) -> TickRegime:
        chosen: TickRegime | None = None
        for regime in self.tick_regimes:
            if regime.effective_from <= session:
                chosen = regime
            else:
                break
        if chosen is None:
            raise PITDataError(f"no tick coverage at {session.isoformat()}")
        return chosen

    def sell_tax_regime_at(self, session: date) -> SellTaxRegime:
        chosen: SellTaxRegime | None = None
        for regime in self.sell_tax_regimes:
            if regime.effective_from <= session:
                chosen = regime
            else:
                break
        if chosen is None:
            raise PITDataError(f"no sell-tax coverage at {session.isoformat()}")
        return chosen

    def price_limit_regime_at(self, session: date) -> PriceLimitRegime:
        chosen: PriceLimitRegime | None = None
        for regime in self.price_limit_regimes:
            if regime.effective_from <= session:
                chosen = regime
            else:
                break
        if chosen is None:
            raise PITDataError(f"no price-limit coverage at {session.isoformat()}")
        return chosen

    def tick_size(self, *, session: date, market: KrxMarket, price: int) -> int:
        """Return the quoting tick for ``price`` on ``session``.

        Raises:
            PITDataError: session precedes coverage or price is not positive.
        """
        if isinstance(price, bool) or not isinstance(price, int) or price <= 0:
            raise PITDataError(f"price must be a positive int, got {price!r}")
        bands = self.tick_regime_at(session).bands[market]
        chosen = bands[0]
        for band in bands:
            if band.lower_price_inclusive <= price:
                chosen = band
            else:
                break
        return chosen.tick

    def sell_tax_rate(self, *, session: date, market: KrxMarket) -> Decimal:
        """Return the total seller-side transaction tax rate for ``session``.

        Raises:
            PITDataError: session precedes coverage.
        """
        return self.sell_tax_regime_at(session).rates[market]

    def price_limits(self, *, session: date, market: KrxMarket, base_price: int) -> tuple[int, int]:
        """Return ``(upper, lower)`` daily limit prices around a KRX base price.

        The limit width is the base price times the regime ratio, truncated to
        the tick of the base price; the resulting bounds are then snapped
        inward to the tick valid at each bound's own price level.

        Raises:
            PITDataError: session precedes coverage or base price is not positive.
        """
        if isinstance(base_price, bool) or not isinstance(base_price, int) or base_price <= 0:
            raise PITDataError(f"base_price must be a positive int, got {base_price!r}")
        ratio = self.price_limit_regime_at(session).ratio
        base_tick = self.tick_size(session=session, market=market, price=base_price)
        width = (
            int((Decimal(base_price) * ratio / Decimal(base_tick)).to_integral_value(rounding=ROUND_FLOOR)) * base_tick
        )
        raw_upper = base_price + width
        raw_lower = base_price - width
        upper_tick = self.tick_size(session=session, market=market, price=raw_upper)
        lower_tick = self.tick_size(session=session, market=market, price=raw_lower)
        upper = (raw_upper // upper_tick) * upper_tick
        snapped_lower = -(-raw_lower // lower_tick) * lower_tick
        lower = snapped_lower if snapped_lower >= lower_tick else lower_tick
        return (upper, lower)


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise PITDataError(f"invalid effective_from {value!r}") from exc
    raise PITDataError(f"invalid effective_from {value!r}")


def _parse_decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise PITDataError(f"invalid {field} {value!r}")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise PITDataError(f"invalid {field} {value!r}") from exc
    raise PITDataError(f"invalid {field} {value!r}")


def _parse_band_entry(entry: Any) -> TickBand:
    lower: Any = None
    tick: Any = None
    if isinstance(entry, Mapping):
        keys = set(entry.keys())
        if keys == {"lower", "tick"}:
            lower, tick = entry["lower"], entry["tick"]
        elif keys == {"lower_price_inclusive", "tick"}:
            lower, tick = entry["lower_price_inclusive"], entry["tick"]
        else:
            raise PITDataError(f"invalid tick band keys {sorted(keys)}")
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        lower, tick = entry[0], entry[1]
    else:
        raise PITDataError(f"invalid tick band entry {entry!r}")
    if isinstance(lower, bool) or not isinstance(lower, int) or lower < 0:
        raise PITDataError(f"invalid band lower {lower!r}")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick <= 0:
        raise PITDataError(f"invalid band tick {tick!r}")
    return TickBand(lower_price_inclusive=lower, tick=tick)


def _parse_tick_bands(value: Any) -> tuple[TickBand, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PITDataError(f"invalid tick bands {value!r}")
    bands = tuple(_parse_band_entry(entry) for entry in value)
    if bands[0].lower_price_inclusive != 0:
        raise PITDataError("tick bands must start at 0")
    for prev, cur in pairwise(bands):
        if cur.lower_price_inclusive <= prev.lower_price_inclusive:
            raise PITDataError("tick bands must be strictly ascending")
    return bands


def _parse_tick_regimes(value: Any) -> tuple[TickRegime, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PITDataError("tick_regimes must be a non-empty list")
    regimes: list[TickRegime] = []
    prev_date: date | None = None
    for raw in value:
        if not isinstance(raw, Mapping):
            raise PITDataError(f"invalid tick regime {raw!r}")
        if set(raw.keys()) != {"effective_from", "KOSPI", "KOSDAQ"}:
            raise PITDataError(f"invalid tick regime keys {sorted(raw.keys())}")
        when = _parse_date(raw["effective_from"])
        if prev_date is not None and when <= prev_date:
            raise PITDataError("tick regimes must be strictly ascending by effective_from")
        prev_date = when
        bands = {KrxMarket(m): _parse_tick_bands(raw[m]) for m in _MARKETS}
        regimes.append(TickRegime(effective_from=when, bands=bands))
    return tuple(regimes)


def _parse_sell_tax_regimes(value: Any) -> tuple[SellTaxRegime, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PITDataError("sell_tax_regimes must be a non-empty list")
    regimes: list[SellTaxRegime] = []
    prev_date: date | None = None
    for raw in value:
        if not isinstance(raw, Mapping):
            raise PITDataError(f"invalid sell-tax regime {raw!r}")
        if set(raw.keys()) != {"effective_from", "KOSPI", "KOSDAQ"}:
            raise PITDataError(f"invalid sell-tax regime keys {sorted(raw.keys())}")
        when = _parse_date(raw["effective_from"])
        if prev_date is not None and when <= prev_date:
            raise PITDataError("sell-tax regimes must be strictly ascending by effective_from")
        prev_date = when
        rates: dict[KrxMarket, Decimal] = {}
        for m in _MARKETS:
            rate = _parse_decimal(raw[m], field=f"sell-tax rate {m}")
            if rate < 0 or rate > Decimal("0.01"):
                raise PITDataError(f"sell-tax rate out of range {rate!s}")
            rates[KrxMarket(m)] = rate
        regimes.append(SellTaxRegime(effective_from=when, rates=rates))
    return tuple(regimes)


def _parse_price_limit_regimes(value: Any) -> tuple[PriceLimitRegime, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PITDataError("price_limit_regimes must be a non-empty list")
    regimes: list[PriceLimitRegime] = []
    prev_date: date | None = None
    for raw in value:
        if not isinstance(raw, Mapping):
            raise PITDataError(f"invalid price-limit regime {raw!r}")
        if set(raw.keys()) != {"effective_from", "ratio"}:
            raise PITDataError(f"invalid price-limit regime keys {sorted(raw.keys())}")
        when = _parse_date(raw["effective_from"])
        if prev_date is not None and when <= prev_date:
            raise PITDataError("price-limit regimes must be strictly ascending by effective_from")
        prev_date = when
        ratio = _parse_decimal(raw["ratio"], field="price-limit ratio")
        if ratio <= 0 or ratio >= 1:
            raise PITDataError(f"price-limit ratio out of range {ratio!s}")
        regimes.append(PriceLimitRegime(effective_from=when, ratio=ratio))
    return tuple(regimes)


def parse_krx_market_rules(document: Mapping[str, Any]) -> KrxMarketRules:
    """Validate a decoded rules document into immutable regimes.

    Raises:
        PITDataError: missing/unknown keys, non-ascending regimes, band lists
            not starting at 0 or not strictly ascending, non-positive ticks,
            rates outside [0, 0.01], ratio outside (0, 1), or a market missing
            from a regime.
    """
    if not isinstance(document, Mapping):
        raise PITDataError(f"rules document must be a mapping, got {type(document).__name__}")
    if set(document.keys()) != set(_TOP_LEVEL_KEYS):
        raise PITDataError(f"invalid top-level keys {sorted(document.keys())}")
    version = document["version"]
    if version != VERSION:
        raise PITDataError(f"unsupported version {version!r}")
    sources = document["sources"]
    if not isinstance(sources, (list, tuple)) or not sources or any(not isinstance(s, str) or not s for s in sources):
        raise PITDataError("sources must be a non-empty list of citations")
    return KrxMarketRules(
        version=version,
        tick_regimes=_parse_tick_regimes(document["tick_regimes"]),
        sell_tax_regimes=_parse_sell_tax_regimes(document["sell_tax_regimes"]),
        price_limit_regimes=_parse_price_limit_regimes(document["price_limit_regimes"]),
    )


def load_krx_market_rules(path: Path) -> KrxMarketRules:
    """Read the versioned TOML rules file with stdlib ``tomllib`` and parse it."""
    with open(path, "rb") as handle:
        document = tomllib.load(handle)
    return parse_krx_market_rules(document)
