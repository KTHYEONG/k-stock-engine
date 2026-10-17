"""Shadow fill slippage against official opens and captured auction paths."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import polars as pl

from src.core.market_rules import KrxMarket, KrxMarketRules
from src.core.time import OPEN_SNAPSHOT_OFFSETS_MIN

__all__ = ["slippage_table"]


def _ticker(name: str) -> str:
    text = str(name).strip()
    return text.split(":")[-1] if ":" in text else text


def _price_of(payload: Any, keys: tuple[str, ...]) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        try:
            parsed = int(str(payload.get(key, "")).strip())
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _expected_block(raw: Any) -> dict[str, Any] | None:
    """The expected-auction mapping from a stored ``output2`` JSON string."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return None


def _expected_price(block: Mapping[str, Any]) -> int | None:
    """Positive integer ``antc_cnpr`` or None."""
    try:
        parsed = int(str(block.get("antc_cnpr", "")).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def slippage_table(
    fills: pl.DataFrame, auction: pl.DataFrame, market: pl.DataFrame, rules: KrxMarketRules
) -> pl.DataFrame:
    """One row per shadow fill joining the model price, the official open and the captured auction path."""
    for name in ("session", "instrument_id", "side", "quantity", "price"):
        if name not in fills.columns:
            raise ValueError(f"fills frame is missing column: {name}")
    for name in ("session", "instrument_id", "open"):
        if name not in market.columns:
            raise ValueError(f"market frame is missing column: {name}")
    has_market = "market" in market.columns
    has_adtv = "adtv20" in market.columns
    opens: dict[tuple[Any, str], int] = {}
    venues: dict[tuple[Any, str], str] = {}
    adtvs: dict[tuple[Any, str], float] = {}
    for row in market.to_dicts():
        try:
            price = int(row["open"])
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        key = (row["session"], _ticker(row["instrument_id"]))
        opens[key] = price
        if has_market:
            venues[key] = str(row.get("market") or "KOSPI")
        if has_adtv:
            raw_adtv = row.get("adtv20")
            if raw_adtv is not None:
                try:
                    adtvs[key] = float(raw_adtv)
                except (TypeError, ValueError):
                    continue
    captured: dict[tuple[Any, str], int] = {}
    expected: dict[tuple[Any, str, int], int] = {}
    if auction.height and {"session", "symbol", "kind", "scheduled_at", "output1"} <= set(auction.columns):
        for row in auction.to_dicts():
            key = (row["session"], str(row["symbol"]))
            if row["kind"] == "open":
                try:
                    payload = json.loads(row["output1"])
                except (TypeError, ValueError):
                    continue
                printed = _price_of(payload, ("stck_oprc",))
                if printed is not None:
                    captured[key] = printed
            elif row["kind"] == "expected":
                block = _expected_block(row.get("output2"))
                if block is None:
                    continue
                exp_price = _expected_price(block)
                if exp_price is None:
                    continue
                try:
                    scheduled = row["scheduled_at"]
                    base = datetime_of(row["session"])
                    offset = round((scheduled - base).total_seconds() / 60.0)
                except (TypeError, ValueError, AttributeError):
                    continue
                if offset in OPEN_SNAPSHOT_OFFSETS_MIN:
                    expected[(row["session"], str(row["symbol"]), offset)] = exp_price
    out: dict[str, list[Any]] = {
        "session": [],
        "instrument_id": [],
        "side": [],
        "quantity": [],
        "model_price": [],
        "official_open": [],
        "captured_open": [],
        "tick": [],
        "model_minus_open_ticks": [],
        "notional_over_adtv20": [],
        "open_mismatch": [],
    }
    for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
        out[f"expected_minus_open_ticks_m{abs(int(offset))}"] = []
    for row in fills.to_dicts():
        session = row["session"]
        inst = str(row["instrument_id"])
        key = (session, _ticker(inst))
        model = int(row["price"])
        qty = int(row["quantity"])
        side = str(row["side"])
        official = opens.get(key)
        printed = captured.get(key)
        if official is None:
            tick = None
            adverse = None
            mismatch = False
        else:
            venue = KrxMarket(venues.get(key, "KOSPI"))
            tick = rules.tick_size(session=session, market=venue, price=official)
            gap = (model - official) if side == "buy" else (official - model)
            adverse = gap / tick
            mismatch = printed is not None and abs(printed - official) >= tick
        anchor = printed if printed is not None else official
        out["session"].append(session)
        out["instrument_id"].append(inst)
        out["side"].append(side)
        out["quantity"].append(qty)
        out["model_price"].append(model)
        out["official_open"].append(official)
        out["captured_open"].append(printed)
        out["tick"].append(tick)
        out["model_minus_open_ticks"].append(adverse)
        for offset in OPEN_SNAPSHOT_OFFSETS_MIN:
            exp = expected.get((session, _ticker(inst), int(offset)))
            if exp is None or anchor is None or tick is None:
                out[f"expected_minus_open_ticks_m{abs(int(offset))}"].append(None)
            else:
                drift = (exp - anchor) if side == "buy" else (anchor - exp)
                out[f"expected_minus_open_ticks_m{abs(int(offset))}"].append(drift / tick)
        adtv = adtvs.get(key)
        if adtv is None or not (adtv > 0):
            out["notional_over_adtv20"].append(None)
        else:
            out["notional_over_adtv20"].append(qty * model / adtv)
        out["open_mismatch"].append(bool(mismatch))
    return pl.DataFrame(out)


def datetime_of(session: Any) -> Any:
    from datetime import datetime

    from src.core.time import KRX_TZ

    return datetime(session.year, session.month, session.day, 9, 0, tzinfo=KRX_TZ)
