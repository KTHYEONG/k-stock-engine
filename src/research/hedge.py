"""NAV-level KOSDAQ 150 beta-neutral hedge overlay simulator."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.core.pit import PITDataError
from src.data.research_protocol import WindowAuthorization, WindowError

if TYPE_CHECKING:
    from src.backtest.overlay import DerivativeConfig, OverlayState, OverlayTarget

__all__ = [
    "BetaNeutralOverlay",
    "HedgeInputs",
    "HedgeResult",
    "HedgeSpec",
    "derivative_config",
    "hedge_inputs_from_frame",
    "rolling_beta",
    "simulate_hedged_book",
]


class HedgeSpec(BaseModel):
    """Frozen identity of the hedge overlay; canonical JSON is part of the strategy identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hedge_ratio: float
    beta_window_sessions: int
    beta_min_sessions: int
    beta_cap: float
    rebalance_every_sessions: int
    use_futures: bool
    contract_multiplier_krw: int
    initial_margin_rate: float
    margin_buffer_rate: float
    margin_topup_trigger_fraction: float
    futures_cost_rate: float
    inverse_cost_rate: float
    resize_sell_cost_rate: float
    resize_buy_cost_rate: float
    futures_tax_rate: float
    futures_annual_deduction_krw: int
    inverse_tax_rate: float

    @field_validator("hedge_ratio")
    @classmethod
    def _hedge_ratio(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"hedge_ratio must be finite and >= 0, got {value!r}")
        return out

    @field_validator("beta_window_sessions")
    @classmethod
    def _beta_window(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"beta_window_sessions must be an int >= 2, got {value!r}")
        return int(value)

    @field_validator("beta_min_sessions")
    @classmethod
    def _beta_min(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"beta_min_sessions must be an int >= 2, got {value!r}")
        return int(value)

    @field_validator("beta_cap")
    @classmethod
    def _beta_cap(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out <= 0.0:
            raise ValueError(f"beta_cap must be finite and > 0, got {value!r}")
        return out

    @field_validator("rebalance_every_sessions")
    @classmethod
    def _rebalance_every(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"rebalance_every_sessions must be an int >= 1, got {value!r}")
        return int(value)

    @field_validator("contract_multiplier_krw")
    @classmethod
    def _multiplier(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"contract_multiplier_krw must be a positive int, got {value!r}")
        return int(value)

    @field_validator("initial_margin_rate")
    @classmethod
    def _margin_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out < 1.0:
            raise ValueError(f"initial_margin_rate must be in (0, 1), got {value!r}")
        return out

    @field_validator("margin_buffer_rate")
    @classmethod
    def _buffer_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"margin_buffer_rate must be finite and >= 0, got {value!r}")
        return out

    @field_validator("margin_topup_trigger_fraction")
    @classmethod
    def _trigger(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 < out <= 1.0:
            raise ValueError(f"margin_topup_trigger_fraction must be in (0, 1], got {value!r}")
        return out

    @field_validator("futures_cost_rate", "inverse_cost_rate", "resize_sell_cost_rate", "resize_buy_cost_rate")
    @classmethod
    def _non_negative_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"cost rate must be finite and >= 0, got {value!r}")
        return out

    @field_validator("futures_tax_rate", "inverse_tax_rate")
    @classmethod
    def _tax_rate(cls, value: object) -> float:
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out) or not 0.0 <= out < 1.0:
            raise ValueError(f"tax rate must satisfy 0 <= rate < 1, got {value!r}")
        return out

    @field_validator("futures_annual_deduction_krw")
    @classmethod
    def _deduction(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"futures_annual_deduction_krw must be an int >= 0, got {value!r}")
        return int(value)

    @model_validator(mode="after")
    def _check_beta_range(self) -> HedgeSpec:
        if self.beta_min_sessions > self.beta_window_sessions:
            raise ValueError("beta_min_sessions must satisfy 2 <= value <= beta_window_sessions")
        if self.initial_margin_rate + self.margin_buffer_rate >= 1.0:
            raise ValueError("initial_margin_rate + margin_buffer_rate must be < 1")
        return self

    def canonical_json(self) -> str:
        """Canonical JSON with sorted keys and compact separators."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class HedgeInputs:
    """Daily hedge series aligned to ``sessions``: futures-underlying index level and inverse-ETF close.

    ``inverse_close`` is NaN where the ETF was not yet listed; both arrays are float64, same length.
    """

    sessions: tuple[date, ...]
    index_level: NDArray[np.float64]
    inverse_close: NDArray[np.float64]


def hedge_inputs_from_frame(frame: pl.DataFrame, *, sessions: Sequence[date]) -> HedgeInputs:
    """Align the Silver ``hedge_series`` frame to ``sessions``.

    Sessions absent from the frame become NaN; the frame must not contain duplicate sessions.

    Raises:
        PITDataError: a duplicate or null session, a missing or non-positive index level or inverse
            close, or missing required columns (session, index_level, inverse_close).
    """
    required = ("session", "index_level", "inverse_close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise PITDataError(f"hedge frame is missing columns: {missing}")
    by_session: dict[date, tuple[float, float]] = {}
    for day, level, inverse in frame.select(required).iter_rows():
        if day is None or day in by_session:
            raise PITDataError(f"hedge frame has a null or duplicate session: {day!r}")
        if level is None or not _valid_price(float(level)):
            raise PITDataError(f"hedge index level is missing or non-positive for {day}")
        if inverse is not None and not _valid_price(float(inverse)):
            raise PITDataError(f"hedge inverse close is non-positive for {day}")
        by_session[day] = (float(level), math.nan if inverse is None else float(inverse))
    days = tuple(sessions)
    nan_pair = (math.nan, math.nan)
    pairs = [by_session.get(day, nan_pair) for day in days]
    return HedgeInputs(
        sessions=days,
        index_level=np.array([pair[0] for pair in pairs], dtype=np.float64),
        inverse_close=np.array([pair[1] for pair in pairs], dtype=np.float64),
    )


def rolling_beta(
    stock_returns: NDArray[np.float64],
    index_returns: NDArray[np.float64],
    *,
    window: int,
    min_sessions: int,
    cap: float,
) -> NDArray[np.float64]:
    """Causal OLS slope of ``stock_returns`` on ``index_returns``, clipped to [0, cap].

    Element ``t`` uses only observations at positions ``[max(0, t - window), t)``
    (strictly before ``t``); 0.0 when fewer than ``min_sessions`` finite pairs exist
    or the index variance is zero.

    Why strictly before: the decision at session ``t`` may not use session ``t``'s own return.

    Raises:
        ValueError: arrays are not 1-D of equal length, ``window`` or ``min_sessions`` < 1, or ``cap`` <= 0.
    """
    stock = np.asarray(stock_returns, dtype=np.float64)
    index = np.asarray(index_returns, dtype=np.float64)
    if stock.shape != index.shape or stock.ndim != 1 or window < 1 or min_sessions < 1 or not cap > 0.0:
        raise ValueError("rolling_beta needs equal-length 1-D arrays, window >= 1, min_sessions >= 1 and cap > 0")
    out = np.zeros(stock.shape[0], dtype=np.float64)
    for t in range(stock.shape[0]):
        xs = index[max(0, t - window) : t]
        ys = stock[max(0, t - window) : t]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if int(np.count_nonzero(mask)) < min_sessions:
            continue
        dx = xs[mask] - xs[mask].mean()
        var = float(np.mean(dx * dx))
        if var > 0.0:
            out[t] = float(np.clip(np.mean(dx * ys[mask]) / var, 0.0, cap))
    return out


@dataclass(frozen=True, slots=True)
class HedgeResult:
    sessions: tuple[date, ...]
    log_returns: NDArray[np.float64]
    nav_krw: NDArray[np.float64]
    beta: NDArray[np.float64]
    futures_contracts: NDArray[np.int64]
    inverse_notional_krw: NDArray[np.float64]
    cost_krw: float
    tax_krw: float
    margin_topups: int


def _valid_price(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def derivative_config(spec: HedgeSpec) -> DerivativeConfig:
    """Map the frozen hedge identity to the engine's derivative terms (multiplier, margin, costs, taxes).

    ``resize_*`` cost rates have no engine counterpart: stock resizing is priced by the engine's own
    execution model.
    """
    from decimal import Decimal

    from src.backtest.overlay import DerivativeConfig

    return DerivativeConfig(
        contract_multiplier_krw=int(spec.contract_multiplier_krw),
        initial_margin_rate=float(spec.initial_margin_rate),
        margin_buffer_rate=float(spec.margin_buffer_rate),
        margin_topup_trigger_fraction=float(spec.margin_topup_trigger_fraction),
        futures_cost_rate=float(spec.futures_cost_rate),
        inverse_cost_rate=float(spec.inverse_cost_rate),
        futures_tax_rate=Decimal(str(float(spec.futures_tax_rate))),
        futures_annual_deduction_krw=int(spec.futures_annual_deduction_krw),
        inverse_tax_rate=Decimal(str(float(spec.inverse_tax_rate))),
    )


def _ols_beta_tail(
    stock_returns: NDArray[np.float64],
    index_returns: NDArray[np.float64],
    *,
    window: int,
    min_sessions: int,
    cap: float,
) -> float:
    """OLS slope over the last ``window`` finite pairs (inclusive), clipped to [0, cap]."""
    stock = np.asarray(stock_returns, dtype=np.float64)
    index = np.asarray(index_returns, dtype=np.float64)
    if stock.ndim != 1 or index.ndim != 1 or stock.shape[0] != index.shape[0]:
        raise ValueError("stock and index returns must be equal-length 1-D arrays")
    tail = slice(max(0, stock.shape[0] - int(window)), stock.shape[0])
    xs = index[tail]
    ys = stock[tail]
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(np.count_nonzero(mask)) < int(min_sessions):
        return 0.0
    dx = xs[mask] - float(xs[mask].mean())
    var = float(np.mean(dx * dx))
    if not var > 0.0:
        return 0.0
    return float(np.clip(float(np.mean(dx * ys[mask]) / var), 0.0, float(cap)))


class BetaNeutralOverlay:
    """``OverlayPolicy`` that shorts ``hedge_ratio · β`` of the stock book with integer futures plus an
    inverse-ETF remainder (or inverse ETF only when ``use_futures`` is false).

    β is the OLS slope of the stock-book returns on the index returns over the last
    ``beta_window_sessions`` finite pairs available in the state (session t inclusive — the decision is made
    after t's close), clipped to ``[0, beta_cap]``; 0.0 with fewer than ``beta_min_sessions`` pairs.
    Rebalances on run rows ``r`` with ``r % rebalance_every_sessions == rebalance_offset`` (r counted from the
    run's first session); returns None on other rows.
    """

    def __init__(self, spec: HedgeSpec, *, rebalance_offset: int, execution_delay: int = 0) -> None:
        every = int(spec.rebalance_every_sessions)
        if isinstance(rebalance_offset, bool) or not isinstance(rebalance_offset, int):
            raise ValueError(f"rebalance_offset must be an int in [0, {every}), got {rebalance_offset!r}")
        if not 0 <= int(rebalance_offset) < every:
            raise ValueError(f"rebalance_offset must satisfy 0 <= value < {every}, got {rebalance_offset!r}")
        if isinstance(execution_delay, bool) or not isinstance(execution_delay, int):
            raise ValueError(f"execution_delay must be an int >= 0, got {execution_delay!r}")
        if int(execution_delay) < 0:
            raise ValueError(f"execution_delay must be >= 0, got {execution_delay!r}")
        self._spec = spec
        self._offset = int(rebalance_offset)
        self._delay = int(execution_delay)
        self._start: int | None = None

    def target(self, state: OverlayState) -> OverlayTarget | None:
        """New overlay target on rebalance rows, else None. Reads only ``state``."""
        from src.backtest.overlay import OverlayTarget

        if self._start is None:
            self._start = int(state.session_idx)
        run_row = int(state.session_idx) - int(self._start)
        if run_row < 0:
            raise ValueError(f"session_idx {state.session_idx} is before the run start {self._start}")
        every = int(self._spec.rebalance_every_sessions)
        if run_row % every != self._offset:
            return None
        if not self._spec.hedge_ratio > 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        stock = np.asarray(state.stock_book_returns, dtype=np.float64)
        index = np.asarray(state.index_returns, dtype=np.float64)
        if self._delay > 0:
            keep = stock.shape[0] - self._delay
            stock = stock[: max(keep, 0)]
            index = index[: max(keep, 0)]
        beta = _ols_beta_tail(
            stock,
            index,
            window=int(self._spec.beta_window_sessions),
            min_sessions=int(self._spec.beta_min_sessions),
            cap=float(self._spec.beta_cap),
        )
        ratio = float(self._spec.hedge_ratio) * float(beta)
        if not ratio > 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        level = float(state.index_level)
        if not math.isfinite(level) or level <= 0.0:
            raise PITDataError(f"index level missing at row {state.session_idx}")
        nav0 = float(state.nav)
        if not math.isfinite(nav0) or nav0 <= 0.0:
            return OverlayTarget(contracts=0, inverse_value_krw=0)
        contract_value = float(int(self._spec.contract_multiplier_krw)) * level
        reserve_rate = float(self._spec.initial_margin_rate) + float(self._spec.margin_buffer_rate)
        _, contracts, inverse_value = _frictionless_split(
            nav0,
            ratio,
            contract_value,
            reserve_rate,
            bool(self._spec.use_futures),
        )
        return OverlayTarget(contracts=int(contracts), inverse_value_krw=math.floor(float(inverse_value)))


def _frictionless_split(
    nav0: float, ratio: float, contract_value: float, reserve_rate: float, use_futures: bool
) -> tuple[float, int, float]:
    """Stock sleeve, short contracts and inverse-ETF value that hedge ``ratio`` of the sleeve at NAV ``nav0``.

    Conservation ``stock + reserve + inverse = nav0`` with hedge notional ``ratio * stock = k * contract_value
    + inverse`` is linear in the stock size for a fixed contract count ``k``; the largest ``k`` whose solution
    falls in its own notional bracket is used (futures need only the margin reserve, so more contracts leave
    more cash for stock). Brackets overlap, so a solution always exists.
    """
    if ratio <= 0.0:
        return nav0, 0, 0.0
    if not use_futures:
        stock = nav0 / (1.0 + ratio)
        return stock, 0, ratio * stock
    best_stock, best_k = 0.0, -1
    for k in range(math.floor(ratio * nav0 / contract_value) + 1):
        stock = (nav0 + (1.0 - reserve_rate) * k * contract_value) / (1.0 + ratio)
        if k * contract_value <= ratio * stock < (k + 1) * contract_value:
            best_stock, best_k = stock, k
    return best_stock, best_k, ratio * best_stock - best_k * contract_value


def simulate_hedged_book(
    stock_returns: NDArray[np.float64],
    sessions: Sequence[date],
    inputs: HedgeInputs,
    spec: HedgeSpec,
    *,
    capital_krw: int,
    rebalance_offset: int = 0,
    execution_delay: int = 0,
    extra_cost_rate: float = 0.0,
    authorization: WindowAuthorization,
) -> HedgeResult:
    """Simulate NAV when the stock sleeve is overlaid with a rolling-beta short hedge.

    ``capital_krw`` is the total NAV at the first session's open (all in stock until the first decision).
    ``rebalance_offset`` shifts the decision grid; ``execution_delay`` makes every beta estimate stale by that
    many sessions; ``extra_cost_rate`` (stress) is added to the futures and inverse per-side cost rates.
    Trading costs, resizing costs and inverse-ETF tax are paid out of the stock sleeve, so NAV is conserved.

    Raises:
        WindowError: ``sessions`` is not inside ``[authorization.start, authorization.end]``.
        PITDataError: an index level or (when the inverse ETF is held) inverse close is missing or
            non-positive for a session of the run or the session before its first; ``sessions`` is not a
            subsequence of ``inputs.sessions``; costs, taxes or a margin call exceed the stock sleeve.
        ValueError: length mismatch, empty run, non-finite stock returns, ``stock_returns <= -1``,
            ``capital_krw < 1``, ``extra_cost_rate`` negative or non-finite, ``rebalance_offset`` or
            ``execution_delay`` out of range.
    """
    days = list(sessions)
    n = len(days)
    rets = np.asarray(stock_returns, dtype=np.float64)
    every = int(spec.rebalance_every_sessions)
    if rets.ndim != 1 or rets.shape[0] != n or n == 0:
        raise ValueError("stock_returns must be 1-D, non-empty and match sessions")
    if not np.all(np.isfinite(rets)) or np.any(rets <= -1.0):
        raise ValueError("stock_returns must be finite and > -1")
    if capital_krw < 1 or not 0 <= rebalance_offset < every or execution_delay < 0:
        raise ValueError("capital_krw >= 1, 0 <= rebalance_offset < every and execution_delay >= 0 are required")
    if not math.isfinite(extra_cost_rate) or extra_cost_rate < 0.0:
        raise ValueError(f"extra_cost_rate must be finite and >= 0, got {extra_cost_rate!r}")
    if days[0] < authorization.start or days[-1] > authorization.end:
        raise WindowError(f"run window [{days[0]}, {days[-1]}] is not inside the authorization")
    position = {day: pos for pos, day in enumerate(inputs.sessions)}
    run_pos = [position.get(day, -1) for day in days]
    if any(a >= b for a, b in pairwise(run_pos)) or run_pos[0] < 1:
        raise PITDataError("sessions must be an increasing subsequence of the hedge inputs with a prior session")
    levels = np.asarray(inputs.index_level, dtype=np.float64)[run_pos]
    prev_levels = np.asarray(inputs.index_level, dtype=np.float64)[[p - 1 for p in run_pos]]
    inverse = np.asarray(inputs.inverse_close, dtype=np.float64)[run_pos]
    prev_inverse = np.asarray(inputs.inverse_close, dtype=np.float64)[[p - 1 for p in run_pos]]
    if not all(_valid_price(float(v)) for v in (*levels, *prev_levels)):
        raise PITDataError("hedge index level is missing or non-positive inside the run")
    beta_full = rolling_beta(
        rets,
        levels / prev_levels - 1.0,
        window=spec.beta_window_sessions,
        min_sessions=spec.beta_min_sessions,
        cap=spec.beta_cap,
    )
    fut_rate = spec.futures_cost_rate + extra_cost_rate
    inv_rate = spec.inverse_cost_rate + extra_cost_rate
    reserve_rate = spec.initial_margin_rate + spec.margin_buffer_rate
    multiplier = float(spec.contract_multiplier_krw)

    stock = float(capital_krw)
    margin = inv_value = basis = ytd_futures = cost_total = tax_total = 0.0
    contracts = topups = 0
    beta_live = 0.0
    nav_prev = float(capital_krw)
    log_out = np.empty(n)
    nav_out = np.empty(n)
    beta_out = np.zeros(n)
    con_out = np.zeros(n, dtype=np.int64)
    inv_out = np.zeros(n)

    for i in range(n):
        level = float(levels[i])
        stock *= 1.0 + float(rets[i])
        pnl = -contracts * multiplier * (level - float(prev_levels[i]))
        margin += pnl
        ytd_futures += pnl
        if inv_value > 0.0:
            if not (_valid_price(float(inverse[i])) and _valid_price(float(prev_inverse[i]))):
                raise PITDataError(f"hedge inverse close is missing or non-positive at {days[i]}")
            inv_value *= float(inverse[i]) / float(prev_inverse[i])
        notional = contracts * multiplier * level
        if contracts > 0 and margin < spec.margin_topup_trigger_fraction * spec.initial_margin_rate * notional:
            need = reserve_rate * notional - margin
            if stock < need * (1.0 + spec.resize_sell_cost_rate):
                raise PITDataError(f"margin shortfall cannot be covered at {days[i]}")
            stock -= need * (1.0 + spec.resize_sell_cost_rate)
            margin += need
            cost_total += need * spec.resize_sell_cost_rate
            topups += 1
        if i >= 1 and i % every == rebalance_offset:
            beta_live = float(beta_full[max(i - execution_delay, 0)])
            nav0 = stock + margin + inv_value
            new_stock, k_new, v_new = _frictionless_split(
                nav0, spec.hedge_ratio * beta_live, multiplier * level, reserve_rate, spec.use_futures
            )
            if v_new > 0.0 and not _valid_price(float(inverse[i])):
                raise PITDataError(f"hedge inverse close is missing or non-positive at {days[i]}")
            tax = 0.0
            if v_new < inv_value:
                sold = (inv_value - v_new) / inv_value
                tax = spec.inverse_tax_rate * max(sold * (inv_value - basis), 0.0)
                basis *= v_new / inv_value
            else:
                basis += v_new - inv_value
            resized = new_stock - stock
            cost = (
                fut_rate * abs(k_new - contracts) * multiplier * level
                + inv_rate * abs(v_new - inv_value)
                + (spec.resize_buy_cost_rate * resized if resized > 0.0 else -spec.resize_sell_cost_rate * resized)
            )
            reserve = reserve_rate * k_new * multiplier * level
            stock = nav0 - reserve - v_new - cost - tax
            cost_total += cost
            tax_total += tax
            margin, inv_value, contracts = reserve, v_new, k_new
        if i == n - 1 or days[i].year != days[i + 1].year:
            due = spec.futures_tax_rate * max(ytd_futures - spec.futures_annual_deduction_krw, 0.0)
            stock -= due
            tax_total += due
            ytd_futures = 0.0
        if stock < 0.0:
            raise PITDataError(f"costs or taxes exceed the stock sleeve at {days[i]}")
        nav = stock + margin + inv_value
        log_out[i] = math.log(nav / nav_prev)
        nav_out[i], beta_out[i], con_out[i], inv_out[i] = nav, beta_live, contracts, inv_value
        nav_prev = nav
    return HedgeResult(
        sessions=tuple(days),
        log_returns=log_out,
        nav_krw=nav_out,
        beta=beta_out,
        futures_contracts=con_out,
        inverse_notional_krw=inv_out,
        cost_krw=cost_total,
        tax_krw=tax_total,
        margin_topups=topups,
    )
