"""Replay screening targets through the integer-ledger engine."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from src.backtest.costs import CostConfig
from src.backtest.engine import DelistPolicy, EngineConfig, run_backtest
from src.backtest.events import build_engine_events
from src.backtest.execution import ExecutionConfig, ExecutionScenario
from src.backtest.market import MarketArrays, load_market_arrays
from src.backtest.overlay import DerivativeConfig, OverlayMarket, OverlayPolicy
from src.core.market_rules import KrxMarketRules
from src.core.pit import PITDataError
from src.data.research_protocol import WindowAuthorization, WindowError
from src.research.cube import ResearchCube
from src.research.hedge import HedgeInputs

__all__ = [
    "LedgerOutcome",
    "cash_returns_from_frame",
    "overlay_market_from_inputs",
    "run_ledger",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LedgerOutcome:
    capital_krw: int
    halted_exit_policy: str
    sessions: tuple[date, ...]
    log_returns: NDArray[np.float64]
    stock_book_returns: NDArray[np.float64] | None = None
    reject_counts: Mapping[str, int] | None = None
    journal_totals_krw: Mapping[str, int] | None = None
    avg_stock_exposure: float = 0.0
    avg_margin_share: float = 0.0
    avg_inverse_share: float = 0.0
    turnover_per_year: float = 0.0
    ledger_hash: str = ""
    nav_krw: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.reject_counts is None:
            object.__setattr__(self, "reject_counts", {})
        if self.journal_totals_krw is None:
            object.__setattr__(self, "journal_totals_krw", {})
        if self.stock_book_returns is None:
            object.__setattr__(self, "stock_book_returns", np.zeros(np.asarray(self.log_returns).shape[0]))
        if self.nav_krw is None:
            logs = np.asarray(self.log_returns, dtype=np.float64)
            object.__setattr__(
                self,
                "nav_krw",
                np.ascontiguousarray(float(self.capital_krw) * np.exp(np.cumsum(logs)), dtype=np.float64),
            )


def overlay_market_from_inputs(inputs: HedgeInputs) -> OverlayMarket:
    """Engine overlay closes from the session-aligned hedge inputs (no fill, NaN preserved)."""
    return OverlayMarket(
        index_level=np.ascontiguousarray(np.asarray(inputs.index_level, dtype=np.float64)),
        inverse_close=np.ascontiguousarray(np.asarray(inputs.inverse_close, dtype=np.float64)),
    )


def cash_returns_from_frame(frame: pl.DataFrame, *, sessions: Sequence[date]) -> NDArray[np.float64]:
    """Element t = cash_close_t / cash_close_{t-1} - 1 over consecutive cube sessions; NaN where either close
    is absent. Raises PITDataError on duplicate sessions or non-positive closes.
    """
    required = ("session", "cash_close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise PITDataError(f"cash frame is missing columns: {missing}")
    by_session: dict[date, float | None] = {}
    for day, close in frame.select(required).iter_rows():
        if day is None or day in by_session:
            raise PITDataError(f"cash frame has a null or duplicate session: {day!r}")
        if close is None:
            by_session[day] = None
            continue
        value = float(close)
        if not math.isfinite(value) or value <= 0.0:
            raise PITDataError(f"cash close is missing or non-positive for {day}")
        by_session[day] = value
    days = list(sessions)
    out = np.empty(len(days), dtype=np.float64)
    for pos, day in enumerate(days):
        if pos == 0:
            out[pos] = math.nan
            continue
        prev = by_session.get(days[pos - 1])
        current = by_session.get(day)
        if prev is None or current is None:
            out[pos] = math.nan
            continue
        out[pos] = current / prev - 1.0
    return np.ascontiguousarray(out, dtype=np.float64)


def _engine_parts(path: Path) -> tuple[ExecutionConfig, CostConfig, float]:
    import tomllib

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    sections: list[dict[str, object]] = [
        section for section in (raw.get("execution"), raw.get("costs")) if isinstance(section, dict)
    ]

    def _pick(key: str) -> object:
        if key in raw:
            return raw[key]
        for section in sections:
            if key in section:
                return section[key]
        raise ValueError(f"engine config is missing required key: {key}")

    scenario = ExecutionScenario(str(_pick("scenario")))
    participation_raw = _pick("max_participation")
    if isinstance(participation_raw, bool) or not isinstance(participation_raw, (int, float)):
        raise ValueError(f"max_participation must be a number, got {participation_raw!r}")
    carry_raw = _pick("carry_unfilled")
    if not isinstance(carry_raw, bool):
        raise ValueError(f"carry_unfilled must be a bool, got {carry_raw!r}")
    try:
        commission_rate = Decimal(str(_pick("commission_rate")))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid commission_rate: {_pick('commission_rate')!r}") from exc
    try:
        withholding_rate = Decimal(str(_pick("dividend_withholding_rate")))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid dividend_withholding_rate: {_pick('dividend_withholding_rate')!r}") from exc
    impact_raw = _pick("impact_k")
    if isinstance(impact_raw, bool) or not isinstance(impact_raw, (int, float)):
        raise ValueError(f"impact_k must be a number, got {impact_raw!r}")
    buffer_raw = _pick("cash_buffer")
    if isinstance(buffer_raw, bool) or not isinstance(buffer_raw, (int, float)):
        raise ValueError(f"cash_buffer must be a number, got {buffer_raw!r}")

    def _optional(key: str) -> object | None:
        if key in raw:
            value: object = raw[key]
            return value
        for section in sections:
            if key in section:
                nested: object = section[key]
                return nested
        return None

    slip_raw = _optional("auction_slippage_ticks")
    if slip_raw is None:
        slip_value = 0.0
    elif isinstance(slip_raw, bool) or not isinstance(slip_raw, (int, float)):
        raise ValueError(f"auction_slippage_ticks must be a number, got {slip_raw!r}")
    else:
        slip_value = float(slip_raw)
    tax_raw = _optional("cash_yield_tax_rate")
    if tax_raw is None:
        yield_tax = Decimal("0.154")
    else:
        try:
            yield_tax = Decimal(str(tax_raw))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid cash_yield_tax_rate: {tax_raw!r}") from exc
    execution = ExecutionConfig(
        scenario=scenario,
        max_participation=float(participation_raw),
        carry_unfilled=carry_raw,
    )
    costs = CostConfig(
        commission_rate=commission_rate,
        impact_k=float(impact_raw),
        dividend_withholding_rate=withholding_rate,
        auction_slippage_ticks=slip_value,
        cash_yield_tax_rate=yield_tax,
    )
    return (execution, costs, float(buffer_raw))


def run_ledger(
    *,
    cube: ResearchCube,
    targets: Mapping[int, NDArray[np.float64]],
    panel_dir: Path,
    dividends: pl.DataFrame,
    engine_config_path: Path,
    rules: KrxMarketRules,
    market_cache_root: Path,
    capital_krw: int,
    halted_exit_policy: DelistPolicy,
    start: date,
    end: date,
    authorization: WindowAuthorization,
    cash_returns: NDArray[np.float64] | None = None,
    overlay: OverlayPolicy | None = None,
    overlay_market: OverlayMarket | None = None,
    derivatives: DerivativeConfig | None = None,
    extra_slippage: float = 0.0,
    auction_slippage_ticks: float | None = None,
    sessions_per_year: int = 252,
    market_arrays: MarketArrays | None = None,
) -> LedgerOutcome:
    """Replay screening targets through the integer-ledger engine.

    ``market_arrays`` short-circuits the panel load so a caller that already holds the dense arrays (the
    evaluation pipeline, once per run) does not rebuild them per scenario, and so the causal perturbation can
    replay against a deliberately corrupted copy. The cube identity check applies either way.

    Raises:
        WindowError: the window is outside the authorization.
        PITDataError: the cube's sessions or instrument ids differ from the panel's arrays.
    """
    if start < authorization.start or end > authorization.end:
        raise WindowError(f"run window [{start}, {end}] is not inside the authorization")
    arrays = (
        market_arrays
        if market_arrays is not None
        else load_market_arrays(panel_dir=Path(panel_dir), cache_root=Path(market_cache_root))
    )
    if tuple(arrays.sessions) != tuple(cube.sessions) or tuple(arrays.instrument_ids) != tuple(cube.instrument_ids):
        raise PITDataError("research cube sessions or instruments differ from the panel")
    execution, costs, cash_buffer = _engine_parts(Path(engine_config_path))
    if auction_slippage_ticks is not None or extra_slippage != 0.0:
        slip = float(costs.auction_slippage_ticks) if auction_slippage_ticks is None else float(auction_slippage_ticks)
        costs = CostConfig(
            commission_rate=costs.commission_rate,
            impact_k=costs.impact_k,
            dividend_withholding_rate=costs.dividend_withholding_rate,
            auction_slippage_ticks=slip,
            extra_slippage=float(extra_slippage),
            cash_yield_tax_rate=costs.cash_yield_tax_rate,
        )
    config = EngineConfig(
        initial_cash=capital_krw,
        execution=execution,
        costs=costs,
        halted_exit_policy=halted_exit_policy,
        cash_buffer=cash_buffer,
    )
    frame = dividends
    dividends_arg: pl.DataFrame | None = frame
    if frame.height:
        dividends_arg = frame.filter(
            (pl.col("ex_session") >= arrays.sessions[0]) & (pl.col("pay_session") <= arrays.sessions[-1])
        )
    else:
        dividends_arg = None
    events = build_engine_events(arrays=arrays, panel_dir=Path(panel_dir), dividends=dividends_arg)
    sparse: dict[int, dict[int, float]] = {}
    for row, weights in targets.items():
        arr = np.asarray(weights, dtype=np.float64)
        sparse[int(row)] = {n: float(arr[n]) for n in np.flatnonzero(arr > 0.0).tolist()}
    result = run_backtest(
        arrays=arrays,
        events=events,
        targets=sparse,
        config=config,
        deposits={},
        rules=rules,
        start=start,
        end=end,
        cash_returns=cash_returns,
        overlay=overlay,
        overlay_market=overlay_market,
        derivatives=derivatives,
    )
    sessions = tuple(arrays.sessions[record.session_idx] for record in result.nav)
    logs = np.empty(len(result.nav), dtype=np.float64)
    navs = np.empty(len(result.nav), dtype=np.float64)
    prev = float(capital_krw)
    for idx, record in enumerate(result.nav):
        nav = float(record.nav)
        logs[idx] = math.log(nav / prev) if nav > 0 and prev > 0 else 0.0
        navs[idx] = nav
        prev = nav
    counts = dict(Counter(reject.reason for reject in result.rejects))
    journal_totals: dict[str, int] = {}
    for entry in result.journal:
        key = entry.kind.value
        journal_totals[key] = journal_totals.get(key, 0) + int(entry.cash_delta)
    exposures: list[float] = []
    margins: list[float] = []
    inverses: list[float] = []
    for record in result.nav:
        nav = float(record.nav)
        scale = 1.0 / nav if nav > 0 else 0.0
        exposures.append(float(record.market_value) * scale)
        margins.append(float(record.margin) * scale)
        inverses.append(float(record.inverse_value) * scale)
    fill_notional = float(sum(int(fill.quantity) * int(fill.price) for fill in result.fills))
    mean_nav = float(np.mean([float(record.nav) for record in result.nav])) if result.nav else 0.0
    n_sessions = len(result.nav)
    turnover = (
        fill_notional / mean_nav * float(sessions_per_year) / float(n_sessions)
        if mean_nav > 0.0 and n_sessions > 0
        else 0.0
    )
    _LOG.info(
        "[EXEC] ledger replay capital=%d policy=%s overlay=%s slip_ticks=%s extra=%s rejects=%s",
        capital_krw,
        halted_exit_policy.value,
        "on" if overlay is not None else "off",
        float(costs.auction_slippage_ticks),
        float(costs.extra_slippage),
        dict(counts),
    )
    return LedgerOutcome(
        capital_krw=capital_krw,
        halted_exit_policy=halted_exit_policy.value,
        sessions=sessions,
        log_returns=np.ascontiguousarray(logs),
        stock_book_returns=np.ascontiguousarray(np.asarray(result.stock_book_returns, dtype=np.float64)),
        reject_counts=counts,
        journal_totals_krw=journal_totals,
        avg_stock_exposure=float(np.mean(exposures)) if exposures else 0.0,
        avg_margin_share=float(np.mean(margins)) if margins else 0.0,
        avg_inverse_share=float(np.mean(inverses)) if inverses else 0.0,
        turnover_per_year=float(turnover),
        ledger_hash=result.ledger_hash,
        nav_krw=np.ascontiguousarray(navs, dtype=np.float64),
    )
