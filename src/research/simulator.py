"""Weight-space screening simulator mirroring the ledger engine's execution semantics."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.cube import ResearchCube

__all__ = ["SimConfig", "SimResult", "simulate"]

_LOG = logging.getLogger(__name__)


class SimConfig(BaseModel):
    """Screening execution parameters; defaults come from ``config/backtest/default_engine.toml``.

    Attributes:
        capital_krw: Initial cash.
        commission_rate: Broker commission per side on notional.
        impact_k: Square-root impact coefficient (same meaning as the engine's ``impact_k``).
        max_participation: Cap on one order's notional over decision-session adtv20.
        cash_buffer: Fraction of NAV left uninvested at every rebalance.
        halted_exit_value: Fraction of the last close recovered when a position is delisted after
            a halted final session (1.0 = last close, 0.0 = total loss).
        extra_slippage: Additional adverse price fraction per side (stress scenarios only).
        execution_delay: Extra sessions between decision and execution (0 = next open).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    capital_krw: int
    commission_rate: float
    impact_k: float
    max_participation: float
    cash_buffer: float
    halted_exit_value: float = 1.0
    extra_slippage: float = 0.0
    auction_slippage_ticks: float = 0.0
    execution_delay: int = 0

    @field_validator("capital_krw")
    @classmethod
    def _positive_capital(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"capital_krw must be a positive int, got {value!r}")
        return value

    @field_validator("commission_rate", "impact_k", "extra_slippage", "auction_slippage_ticks")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        out = float(value)
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"value must be finite and >= 0, got {value!r}")
        return out

    @field_validator("max_participation")
    @classmethod
    def _participation(cls, value: float) -> float:
        out = float(value)
        if not math.isfinite(out) or not 0.0 < out <= 1.0:
            raise ValueError(f"max_participation must be in (0, 1], got {value!r}")
        return out

    @field_validator("cash_buffer")
    @classmethod
    def _buffer(cls, value: float) -> float:
        out = float(value)
        if not math.isfinite(out) or not 0.0 <= out < 1.0:
            raise ValueError(f"cash_buffer must be in [0, 1), got {value!r}")
        return out

    @field_validator("halted_exit_value")
    @classmethod
    def _exit_value(cls, value: float) -> float:
        out = float(value)
        if not math.isfinite(out) or not 0.0 <= out <= 1.0:
            raise ValueError(f"halted_exit_value must be in [0, 1], got {value!r}")
        return out

    @field_validator("execution_delay")
    @classmethod
    def _delay(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"execution_delay must be a non-negative int, got {value!r}")
        return value

    @classmethod
    def from_engine_toml(cls, path: Path, *, capital_krw: int) -> SimConfig:
        """Build a screening config from an engine TOML and an explicit capital."""
        import tomllib

        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        sections = [
            section for section in (raw.get("execution"), raw.get("costs")) if isinstance(section, dict)
        ]

        def _pick(key: str) -> object:
            if key in raw:
                return raw[key]
            for section in sections:
                if key in section:
                    return section[key]
            raise ValueError(f"engine config is missing required key: {key}")

        return cls(
            capital_krw=capital_krw,
            commission_rate=float(str(_pick("commission_rate"))),
            impact_k=float(_pick("impact_k")),  # type: ignore[arg-type]
            max_participation=float(_pick("max_participation")),  # type: ignore[arg-type]
            cash_buffer=float(_pick("cash_buffer")),  # type: ignore[arg-type]
            auction_slippage_ticks=float(_pick("auction_slippage_ticks"))  # type: ignore[arg-type]
            if "auction_slippage_ticks" in raw or any("auction_slippage_ticks" in s for s in sections)
            else 0.0,
        )

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class SimResult:
    sessions: tuple[date, ...]
    nav_krw: NDArray[np.float64]
    log_returns: NDArray[np.float64]
    turnover: NDArray[np.float64]
    cost: NDArray[np.float64]
    holdings: NDArray[np.int64]
    blocked_buy_share: float


def _get_float(cube: ResearchCube, name: str) -> NDArray[np.float64]:
    return np.asarray(cube.arrays[name], dtype=np.float64)


def _get_bool(cube: ResearchCube, name: str) -> NDArray[np.bool_]:
    return np.asarray(cube.arrays[name], dtype=bool)


def simulate(
    cube: ResearchCube,
    targets: Mapping[int, NDArray[np.float64]],
    *,
    start: date,
    end: date,
    config: SimConfig,
    authorization: WindowAuthorization,
    rebalance_band: float = 0.0,
) -> SimResult:
    """Simulate target weights on the fixed session timeline in weight space.

    Mirrors the ledger engine's timing and frictions (see module docstring) without integer
    shares, so it is a screening tool; promotion decisions use the ledger engine.

    Raises:
        WindowError: ``[start, end]`` is not inside ``[authorization.start, authorization.end]``.
        ValueError: ``start``/``end`` outside the cube sessions, a target row outside
            ``[lo - 1 - delay, hi - 1 - delay]``, a target vector of the wrong length, negative /
            non-finite weights or a row sum above 1, or ``rebalance_band`` outside ``[0, 1)``.
    """
    if isinstance(rebalance_band, bool) or not math.isfinite(float(rebalance_band)) or not 0.0 <= float(rebalance_band) < 1.0:
        raise ValueError(f"rebalance_band must satisfy 0 <= b < 1, got {rebalance_band!r}")
    band = float(rebalance_band)
    if start < authorization.start or end > authorization.end:
        raise WindowError(f"run window [{start}, {end}] is not inside the authorization")
    sessions = list(cube.sessions)
    index_of = {day: idx for idx, day in enumerate(sessions)}
    if start not in index_of or end not in index_of or index_of[start] > index_of[end]:
        raise ValueError(f"run window [{start}, {end}] is not within the cube sessions")
    lo, hi = index_of[start], index_of[end]
    delay = int(config.execution_delay)
    n_names = len(cube.instrument_ids)
    for row, weights in targets.items():
        if not isinstance(row, int) or isinstance(row, bool):
            raise ValueError(f"target row must be an int, got {row!r}")
        if row < lo - 1 - delay or row > hi - 1 - delay:
            raise ValueError(f"target row {row} is outside the executable window")
        arr = np.asarray(weights, dtype=np.float64)
        if arr.shape != (n_names,):
            raise ValueError(f"target vector has wrong shape at row {row}")
        if not np.all(np.isfinite(arr)) or bool(np.any(arr < 0.0)) or float(np.sum(arr)) > 1.0 + 1e-9:
            raise ValueError(f"invalid target weights at row {row}")
    present = _get_bool(cube, "present")
    volume = _get_float(cube, "volume")
    open_px = _get_float(cube, "open")
    blocked = _get_bool(cube, "entry_blocked")
    upper = _get_bool(cube, "open_at_upper")
    lower = _get_bool(cube, "open_at_lower")
    adtv = _get_float(cube, "adtv20")
    vol60 = _get_float(cube, "ret_vol60")
    tick = _get_float(cube, "tick_at_open")
    r_on = _get_float(cube, "r_on")
    r_id = _get_float(cube, "r_id")
    tax = _get_float(cube, "sell_tax_rate")
    exit_at = np.asarray(cube.exit_at, dtype=np.int64)
    exit_halted = np.asarray(cube.exit_halted, dtype=bool)
    traded: NDArray[np.bool_] = (
        _get_bool(cube, "traded") if "traded" in cube.arrays else present & (volume > 0) & (open_px > 0)
    )
    holdings = np.zeros(n_names, dtype=np.float64)
    cash = float(config.capital_krw)
    capital = float(config.capital_krw)
    width = hi - lo + 1
    nav_out = np.empty(width, dtype=np.float64)
    log_out = np.empty(width, dtype=np.float64)
    turnover_out = np.zeros(width, dtype=np.float64)
    cost_out = np.zeros(width, dtype=np.float64)
    holdings_out = np.zeros(width, dtype=np.int64)
    desired_total = 0.0
    unfilled_total = 0.0
    prev_nav = capital
    buffer_scale = 1.0 - float(config.cash_buffer)
    commission = float(config.commission_rate)
    for pos, t in enumerate(range(lo, hi + 1)):
        matured = np.flatnonzero(exit_at == t)
        for n in matured.tolist():
            value = float(holdings[int(n)])
            if value > 0.0:
                factor = float(config.halted_exit_value) if bool(exit_halted[int(n)]) else 1.0
                cash += value * factor
                holdings[int(n)] = 0.0
        growth_on = np.asarray(r_on[t], dtype=np.float64)
        growth_on = np.where(np.isfinite(growth_on), 1.0 + growth_on, 1.0)
        holdings *= growth_on
        pre_nav = float(np.sum(holdings) + cash)
        traded_notional = 0.0
        session_cost = 0.0
        d = t - 1 - delay
        if d in targets and pre_nav > 0.0 and d >= 0:
            w = np.asarray(targets[d], dtype=np.float64)
            target_value = w * pre_nav * buffer_scale
            delta = target_value - holdings
            if band > 0.0:
                keep = (holdings > 0.0) & (target_value > 0.0) & (np.abs(delta) < band * target_value)
                delta = np.where(keep, 0.0, delta)
            sizable = np.isfinite(adtv[d]) & (adtv[d] > 0) & np.isfinite(vol60[d])
            sell_ok = (
                np.asarray(present[t], dtype=bool)
                & (np.asarray(volume[t], dtype=np.float64) > 0)
                & (np.asarray(open_px[t], dtype=np.float64) > 0)
                & ~np.asarray(lower[t], dtype=bool)
                & np.asarray(sizable, dtype=bool)
            )
            buy_ok = (
                np.asarray(traded[t], dtype=bool)
                & ~np.asarray(blocked[t], dtype=bool)
                & ~np.asarray(upper[t], dtype=bool)
                & np.asarray(sizable, dtype=bool)
            )
            sell_notionals = np.zeros(n_names, dtype=np.float64)
            buy_notionals = np.zeros(n_names, dtype=np.float64)
            for n in range(n_names):
                size = float(delta[n])
                if size < 0.0:
                    if not bool(sell_ok[n]):
                        continue
                    cap = float(config.max_participation) * float(adtv[d, n])
                    notion = min(-size, cap)
                    if notion > 0.0:
                        sell_notionals[n] = notion
                elif size > 0.0:
                    desired_total += size
                    if not bool(buy_ok[n]):
                        unfilled_total += size
                        continue
                    cap = float(config.max_participation) * float(adtv[d, n])
                    notion = min(size, cap)
                    unfilled_total += size - notion
                    if notion > 0.0:
                        buy_notionals[n] = notion
            open_row = np.asarray(open_px[t], dtype=np.float64)
            tick_row = np.asarray(tick[t], dtype=np.float64)
            tax_row = np.asarray(tax[t], dtype=np.float64)
            sell_proceeds = np.zeros(n_names, dtype=np.float64)
            sell_cost_part = 0.0
            for n in np.flatnonzero(sell_notionals > 0).tolist():
                notion = float(sell_notionals[int(n)])
                vol = float(vol60[d, int(n)])
                adv = float(config.impact_k) * vol * math.sqrt(notion / float(adtv[d, int(n)]))
                tick_frac = 0.0
                if math.isfinite(float(open_row[int(n)])) and float(open_row[int(n)]) > 0:
                    tick_val = float(tick_row[int(n)])
                    if math.isfinite(tick_val):
                        tick_frac = tick_val / float(open_row[int(n)])
                frac = adv + tick_frac * float(config.auction_slippage_ticks) + float(config.extra_slippage)
                rate = float(tax_row[int(n)]) if math.isfinite(float(tax_row[int(n)])) else 0.0
                proceeds = notion * (1.0 - frac) * (1.0 - commission - rate)
                sell_proceeds[int(n)] = proceeds
                sell_cost_part += notion - proceeds
                traded_notional += notion
            buy_costs = np.zeros(n_names, dtype=np.float64)
            buy_fracs: dict[int, float] = {}
            for n in np.flatnonzero(buy_notionals > 0).tolist():
                notion = float(buy_notionals[int(n)])
                vol = float(vol60[d, int(n)])
                adv = float(config.impact_k) * vol * math.sqrt(notion / float(adtv[d, int(n)]))
                tick_frac = 0.0
                if math.isfinite(float(open_row[int(n)])) and float(open_row[int(n)]) > 0:
                    tick_val = float(tick_row[int(n)])
                    if math.isfinite(tick_val):
                        tick_frac = tick_val / float(open_row[int(n)])
                frac = adv + tick_frac * float(config.auction_slippage_ticks) + float(config.extra_slippage)
                buy_fracs[int(n)] = frac
                buy_costs[int(n)] = notion * (1.0 + frac) * (1.0 + commission)
            total_buy_cost = float(np.sum(buy_costs))
            scale = 1.0
            if total_buy_cost > cash + float(np.sum(sell_proceeds)) and total_buy_cost > 0.0:
                scale = (cash + float(np.sum(sell_proceeds))) / total_buy_cost
            for n in np.flatnonzero(sell_notionals > 0).tolist():
                notion = float(sell_notionals[int(n)])
                holdings[int(n)] -= notion
                cash += float(sell_proceeds[int(n)])
            for n in np.flatnonzero(buy_notionals > 0).tolist():
                notion = float(buy_notionals[int(n)]) * scale
                frac = float(buy_fracs[int(n)])
                cost = notion * (1.0 + frac) * (1.0 + commission)
                holdings[int(n)] += notion
                cash -= cost
                traded_notional += notion
                session_cost += cost - notion
                if scale < 1.0:
                    unfilled_total += float(buy_notionals[int(n)]) * (1.0 - scale)
            session_cost += sell_cost_part
            if cash < 0.0 and cash > -1e-9 * capital:  # pragma: no cover
                cash = 0.0
            turnover_out[pos] = traded_notional / pre_nav
            cost_out[pos] = session_cost / pre_nav
        growth_id = np.asarray(r_id[t], dtype=np.float64)
        growth_id = np.where(np.isfinite(growth_id), 1.0 + growth_id, 1.0)
        holdings *= growth_id
        nav = float(np.sum(holdings) + cash)
        nav_out[pos] = nav
        log_out[pos] = math.log(nav / prev_nav) if nav > 0 and prev_nav > 0 else 0.0
        prev_nav = nav
        holdings_out[pos] = int(np.count_nonzero(holdings > 0.0))
    blocked_share = float(unfilled_total / desired_total) if desired_total > 0.0 else 0.0
    _LOG.debug("[ALGO] sim window=%s..%s nav=%.0f", str(start), str(end), float(nav_out[-1]))
    return SimResult(
        sessions=tuple(sessions[lo : hi + 1]),
        nav_krw=np.ascontiguousarray(nav_out),
        log_returns=np.ascontiguousarray(log_out),
        turnover=np.ascontiguousarray(turnover_out),
        cost=np.ascontiguousarray(cost_out),
        holdings=np.ascontiguousarray(holdings_out),
        blocked_buy_share=blocked_share,
    )
