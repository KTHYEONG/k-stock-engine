"""Replay screening targets through the integer-ledger engine."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
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
from src.backtest.market import load_market_arrays
from src.backtest.strategy import PrecomputedTargets
from src.core.market_rules import KrxMarketRules
from src.core.pit import PITDataError
from src.data.research_protocol import LockboxAuthorization, LockboxError
from src.research.cube import ResearchCube

__all__ = ["LedgerOutcome", "run_ledger"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LedgerOutcome:
    capital_krw: int
    halted_exit_policy: str
    sessions: tuple[date, ...]
    log_returns: NDArray[np.float64]
    reject_counts: Mapping[str, int]
    ledger_hash: str


def _engine_parts(path: Path) -> tuple[ExecutionConfig, CostConfig, float, bool]:
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
    static_raw = _pick("allow_static_industry")
    if not isinstance(static_raw, bool):
        raise ValueError(f"allow_static_industry must be a bool, got {static_raw!r}")
    execution = ExecutionConfig(
        scenario=scenario, max_participation=float(participation_raw), carry_unfilled=carry_raw,
    )
    costs = CostConfig(
        commission_rate=commission_rate,
        impact_k=float(impact_raw),
        dividend_withholding_rate=withholding_rate,
    )
    return (execution, costs, float(buffer_raw), static_raw)


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
    authorization: LockboxAuthorization,
) -> LedgerOutcome:
    """Replay screening targets through the integer-ledger engine.

    Raises:
        LockboxError: the window is outside the authorization.
        PITDataError: the cube's sessions or instrument ids differ from the panel's arrays.
    """
    if start < authorization.start or end > authorization.end:
        raise LockboxError(f"run window [{start}, {end}] is not inside the authorization")
    arrays = load_market_arrays(panel_dir=Path(panel_dir), cache_root=Path(market_cache_root))
    if tuple(arrays.sessions) != tuple(cube.sessions) or tuple(arrays.instrument_ids) != tuple(
        cube.instrument_ids
    ):
        raise PITDataError("research cube sessions or instruments differ from the panel")
    execution, costs, cash_buffer, allow_static = _engine_parts(Path(engine_config_path))
    config = EngineConfig(
        initial_cash=capital_krw,
        execution=execution,
        costs=costs,
        halted_exit_policy=halted_exit_policy,
        cash_buffer=cash_buffer,
        allow_static_industry=allow_static,
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
    strategy = PrecomputedTargets(name="precomputed", targets=sparse, params={"capital_krw": capital_krw})
    result = run_backtest(
        arrays=arrays,
        events=events,
        strategy=strategy,
        config=config,
        deposits={},
        asof_tables={},
        rules=rules,
        start=start,
        end=end,
    )
    sessions = tuple(arrays.sessions[record.session_idx] for record in result.nav)
    logs = np.empty(len(result.nav), dtype=np.float64)
    prev = float(capital_krw)
    for idx, record in enumerate(result.nav):
        nav = float(record.nav)
        logs[idx] = math.log(nav / prev) if nav > 0 and prev > 0 else 0.0
        prev = nav
    counts = dict(Counter(reject.reason for reject in result.rejects))
    _LOG.info(
        "[EXEC] ledger replay capital=%d policy=%s rejects=%s",
        capital_krw,
        halted_exit_policy.value,
        dict(counts),
    )
    return LedgerOutcome(
        capital_krw=capital_krw,
        halted_exit_policy=halted_exit_policy.value,
        sessions=sessions,
        log_returns=np.ascontiguousarray(logs),
        reject_counts=counts,
        ledger_hash=result.ledger_hash,
    )
