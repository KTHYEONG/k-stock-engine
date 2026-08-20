"""Execution-equivalent ML evidence: OOF scores replayed through StockBacktester.

The adapter decouples horizon/profile economics and forward-holdout
certification from the residual-vintage proxy. It injects already
causal-calibrated OOF (or frozen-holdout) scores into the same prepared
``StockBacktester`` the operational path uses, so selection evidence is the
actual long-only equity under next-open fills, partial fills, capacity, T+2
settlement, and the base/stress cost schedules.

One immutable ``PreparedReplayMarket`` is built per OOF segment; every segment
starts from the context's initial portfolio and cannot carry positions, pending
orders, settlement, or backtester state across its boundary. ``horizon_sessions``
only bounds the segment decision window; it never drives portfolio capital lock,
target holding period, or rebalance cadence.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import polars as pl

from src.core.costs import CostSchedule, LiquiditySlippageModel
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.backtesting.engine import (
    REQUIRED_BACKTEST_COLUMNS,
    ArtifactSchedule,
    ArtifactSlot,
    BacktestLedgerRow,
    BacktestRequest,
    BacktestResult,
    PreparedReplayDecision,
    PreparedReplayMarket,
    ReplayDecisionProvider,
    ReplayScenarioPlanner,
    StockBacktester,
)
from src.stocks.domain.execution_policy import ExecutionOutcomePolicy
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.trading_cycle import (
    TradingCycleRequest,
    TradingCycleResult,
    plan_prepared_scored_cycle,
)

_SCORE_CANDIDATES = ("predicted_net_alpha", "pred_score")
_ECONOMIC_COLUMNS = (
    "expected_active_alpha",
    "alpha_lower_bound",
    "expected_net_alpha",
    "net_alpha_lower_bound",
    "exit_cost_rate",
)
_ADTV_WINDOW = 20


@dataclass(frozen=True, slots=True)
class ExecutionReplayContext:
    """Immutable full configuration of one execution-equivalent replay."""

    registry: ModelArtifactRegistry
    manifest: DatasetManifest
    instruments: Mapping[str, Instrument]
    artifact_id: str
    strategy_id: str
    initial_portfolio: PortfolioSnapshot
    risk_policy: StockRiskPolicy
    base_cost_schedule: CostSchedule
    stress_cost_schedule: CostSchedule
    liquidity_model: LiquiditySlippageModel | None
    stress_liquidity_model: LiquiditySlippageModel | None
    execution_policy: ExecutionOutcomePolicy
    seed: int


@dataclass(frozen=True, slots=True)
class ExecutionEquivalentReplayRequest:
    """One OOF (or frozen-holdout) execution-equivalent replay request.

    ``market_frame`` is the raw executable panel carrying
    ``REQUIRED_BACKTEST_COLUMNS`` and a timezone-aware ``available_time``.
    ``score_frame`` is the causal-calibrated score panel with a unique
    ``(instrument_id, session)`` key, the declared ``segment_column``, and the
    pre-calibrated production score/economic columns. ``horizon_sessions`` only
    bounds the bootstrap block floor; it is never fed into allocation math.
    """

    context: ExecutionReplayContext
    market_frame: pl.DataFrame
    score_frame: pl.DataFrame
    segment_column: str
    decision_sessions_by_segment: Mapping[int, tuple[datetime, ...]]
    horizon_sessions: int


@dataclass(frozen=True, slots=True)
class ExecutionReplayEvidence:
    """Parallel base/stress daily equity log-growth plus bounded execution sums."""

    base_log_growth: tuple[float, ...]
    stress_log_growth: tuple[float, ...]
    segment_ids: tuple[int, ...]
    planned_cycles: int
    filled_orders: int
    cash_session_fraction: float
    turnover: float
    unfilled_order_reason_counts: tuple[tuple[str, int], ...]
    utility_transition_diagnostics: tuple[tuple[str, float | int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.base_log_growth) != len(self.stress_log_growth):
            raise ValueError("base and stress log growth series must be parallel")
        if len(self.base_log_growth) != len(self.segment_ids):
            raise ValueError("log growth series and segment ids must be parallel")
        if not np.all(np.isfinite(self.base_log_growth)) or not np.all(
            np.isfinite(self.stress_log_growth)
        ):
            raise ValueError("execution log growth must be finite")
        if not 0.0 <= self.cash_session_fraction <= 1.0:
            raise ValueError("cash_session_fraction must be in [0, 1]")
        if not np.isfinite(self.turnover) or self.turnover < 0.0:
            raise ValueError("turnover must be a finite non-negative value")

    def diagnostics(self) -> dict[str, object]:
        """Bounded execution evidence projection; never raw score/price vectors."""
        return {
            "planned_cycles": int(self.planned_cycles),
            "filled_orders": int(self.filled_orders),
            "cash_session_fraction": round(float(self.cash_session_fraction), 12),
            "turnover": round(float(self.turnover), 12),
            "unfilled_order_reason_counts": {
                str(reason): int(count) for reason, count in self.unfilled_order_reason_counts
            },
        }


def replay_execution_equivalent(
    request: ExecutionEquivalentReplayRequest,
) -> ExecutionReplayEvidence:
    """Replay calibrated scores through the prepared engine, segment by segment.

    Each segment builds exactly one ``PreparedReplayMarket`` and starts from the
    context's initial portfolio with fresh pending orders, settlement, and
    backtester state. Score keys are aligned vectorized to the market rows
    (``NaN`` only on unscored market rows); duplicate/unmatched/non-finite score
    keys, missing executable columns, naive timestamps, or an absent score at a
    declared decision fail closed with ``ValueError``.
    """
    context = request.context
    market = request.market_frame
    scores = request.score_frame
    if request.horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive session count")
    _validate_market(market)
    score_column = _resolve_score_column(scores)
    _validate_scores(scores, request.segment_column, score_column)
    _validate_score_market_alignment(market, scores)

    ordered_market = market.sort(["session", "instrument_id"])
    session_to_index = {
        session: index
        for index, session in enumerate(ordered_market["session"].unique().sort().to_list())
    }
    full_sessions = sorted(session_to_index)

    base_growth: list[float] = []
    stress_growth: list[float] = []
    segment_ids: list[int] = []
    total_planned = 0
    total_filled_orders = 0
    total_filled_sessions = 0
    turnover_weighted = 0.0
    unfilled: dict[str, int] = {}

    for segment_id in sorted(request.decision_sessions_by_segment):
        decisions = tuple(request.decision_sessions_by_segment[segment_id])
        if not decisions:
            raise ValueError(f"segment {segment_id} has no declared decision sessions")
        seg_scores = scores.filter(pl.col(request.segment_column) == segment_id)
        if seg_scores.is_empty():
            raise ValueError(f"segment {segment_id} has no score rows")
        _assert_decision_scores(seg_scores, decisions, segment_id)

        first_index = session_to_index.get(min(decisions))
        last_index = session_to_index.get(max(decisions))
        if first_index is None or last_index is None:
            raise ValueError(
                f"segment {segment_id} declares a decision outside the market window"
            )
        segment_sessions = full_sessions[first_index : min(last_index + 2, len(full_sessions))]
        segment_ordered = ordered_market.filter(
            pl.col("session").is_in(segment_sessions)
        )
        decision_indices = tuple(
            session_to_index[decision] - first_index for decision in decisions
        )

        decision_times = _decision_times(segment_ordered, decisions)
        artifacts = ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=min(decision_times),
                    eligible_to=max(decision_times),
                    artifact_id=context.artifact_id,
                ),
            )
        )
        prepared_market = PreparedReplayMarket.build(
            segment_ordered,
            _ADTV_WINDOW,
            instruments=context.instruments,
            artifacts=artifacts,
            initial_portfolio=context.initial_portfolio,
        )

        score_cols = [score_column] + [
            column for column in _ECONOMIC_COLUMNS if column in seg_scores.columns
        ]
        scored_market = segment_ordered.join(
            seg_scores.select(["instrument_id", "session", *score_cols]),
            on=["instrument_id", "session"],
            how="left",
        ).with_columns(pl.Series("adtv", prepared_market.adtv))
        if _can_precompute_volatility(scored_market):
            scored_market = _precompute_volatility(
                scored_market, context.risk_policy.volatility_lookback_sessions
            )
        session_index = _scored_session_index(scored_market)
        overlay = scored_market[score_column].to_numpy().astype(np.float64)
        dataset_hash = _frame_hash(segment_ordered)

        bt_request = BacktestRequest(
            strategy_id=context.strategy_id,
            start_time=min(decision_times),
            end_time=_window_end(min(decision_times), max(decision_times)),
            decision_session_indices=decision_indices,
            cost_schedule=context.base_cost_schedule,
            stress_cost_schedule=context.stress_cost_schedule,
            risk_policy=context.risk_policy,
            seed=context.seed,
        )
        backtester = StockBacktester(
            registry=context.registry,
            instruments=context.instruments,
            manifest=context.manifest,
            cost_schedule=context.base_cost_schedule,
            stress_cost_schedule=context.stress_cost_schedule,
            seed=context.seed,
            policy=context.execution_policy,
            base_liquidity_model=context.liquidity_model,
            stress_liquidity_model=context.stress_liquidity_model,
            decision_provider=_decision_provider(scored_market, session_index),
            scenario_planner=_scenario_planner(context, dataset_hash),
        )
        result = backtester.run_prepared(bt_request, prepared_market, overlay)

        segment_growth = _ledger_log_growth(result.ledger)
        stress_segment_growth = _ledger_log_growth(result.stress_ledger)
        if len(segment_growth) != len(stress_segment_growth):
            raise ValueError(
                f"base and stress ledgers diverged in session count for segment {segment_id}"
            )
        base_growth.extend(segment_growth)
        stress_growth.extend(stress_segment_growth)
        segment_ids.extend([segment_id] * len(segment_growth))

        total_planned += int(result.planned_cycles)
        total_filled_orders += int(result.filled_orders)
        total_filled_sessions += _filled_sessions(result)
        turnover_weighted += float(result.metrics.get("turnover", 0.0)) * max(1, int(result.planned_cycles))
        for reason, count in result.unfilled_order_reason_counts.items():
            unfilled[str(reason)] = unfilled.get(str(reason), 0) + int(count)

    total_planned = max(0, total_planned)
    cash_fraction = (
        1.0
        if total_planned <= 0
        else float(np.clip(1.0 - total_filled_sessions / total_planned, 0.0, 1.0))
    )
    turnover = turnover_weighted / total_planned if total_planned > 0 else 0.0
    return ExecutionReplayEvidence(
        base_log_growth=tuple(base_growth),
        stress_log_growth=tuple(stress_growth),
        segment_ids=tuple(segment_ids),
        planned_cycles=total_planned,
        filled_orders=total_filled_orders,
        cash_session_fraction=cash_fraction,
        turnover=turnover,
        unfilled_order_reason_counts=tuple(sorted(unfilled.items())),
        utility_transition_diagnostics=(),
    )


def _validate_market(market: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_BACKTEST_COLUMNS if column not in market.columns]
    if missing:
        raise ValueError(f"market frame must carry {', '.join(missing)}")
    if "available_time" not in market.columns:
        raise ValueError("market frame must carry an available_time column")
    if market.is_empty():
        raise ValueError("market frame has no rows")
    available_type = market.schema.get("available_time")
    if not isinstance(available_type, pl.Datetime) or available_type.time_zone is None:
        raise ValueError("market available_time must be a timezone-aware datetime column")
    if market["session"].null_count() or market["instrument_id"].null_count():
        raise ValueError("market frame carries null session or instrument_id values")


def _resolve_score_column(scores: pl.DataFrame) -> str:
    for candidate in _SCORE_CANDIDATES:
        if candidate in scores.columns:
            return candidate
    raise ValueError(
        f"score frame must carry one of {', '.join(_SCORE_CANDIDATES)}"
    )


def _validate_scores(scores: pl.DataFrame, segment_column: str, score_column: str) -> None:
    required = ("instrument_id", "session", segment_column, score_column)
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"score frame must carry {', '.join(missing)}")
    if scores.is_empty():
        raise ValueError("score frame has no rows")
    duplicate = (
        scores.group_by(["instrument_id", "session"]).len().filter(pl.col("len") > 1)
    )
    if not duplicate.is_empty():
        raise ValueError("score frame contains duplicate instrument/session keys")
    invalid = scores.filter(
        pl.col(score_column).is_null() | ~pl.col(score_column).is_finite()
    )
    if not invalid.is_empty():
        raise ValueError("score frame contains null or non-finite scores")
    if scores[segment_column].null_count():
        raise ValueError("score frame segment column carries null values")


def _validate_score_market_alignment(
    market: pl.DataFrame, scores: pl.DataFrame
) -> None:
    unmatched = (
        scores.select("instrument_id", "session")
        .join(
            market.select("instrument_id", "session").unique(),
            on=["instrument_id", "session"],
            how="anti",
        )
        .limit(1)
    )
    if not unmatched.is_empty():
        row = unmatched.to_dicts()[0]
        raise ValueError(
            "score key without a market row: "
            f"{row['instrument_id']!r} at {row['session']!r}"
        )


def _assert_decision_scores(
    seg_scores: pl.DataFrame,
    decisions: tuple[datetime, ...],
    segment_id: int,
) -> None:
    for decision in decisions:
        if seg_scores.filter(pl.col("session") == decision).is_empty():
            raise ValueError(
                f"absent score at declared decision {decision.isoformat()} "
                f"(segment {segment_id})"
            )


def _decision_times(
    segment_ordered: pl.DataFrame,
    decisions: tuple[datetime, ...],
) -> tuple[datetime, ...]:
    decision_rows = segment_ordered.filter(pl.col("session").is_in(decisions))
    times: dict[datetime, datetime] = {}
    for session, available in zip(
        decision_rows["session"].to_list(),
        decision_rows["available_time"].to_list(),
        strict=True,
    ):
        if available is None:
            continue
        if session not in times or available > times[session]:
            times[session] = available
    result: list[datetime] = []
    for decision in decisions:
        value = times.get(decision)
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(
                f"missing timezone-aware available_time at decision {decision.isoformat()}"
            )
        result.append(value)
    return tuple(result)


def _window_end(start: datetime, end: datetime) -> datetime:
    if end > start:
        return end
    return start + timedelta(seconds=1)


def _frame_hash(frame: pl.DataFrame) -> str:
    ordered = frame.select(sorted(frame.columns)).sort(["instrument_id", "session"])
    return hashlib.sha256(ordered.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()


def _ledger_log_growth(ledger: tuple[BacktestLedgerRow, ...]) -> tuple[float, ...]:
    equities = [row.equity for row in ledger]
    growth: list[float] = []
    for previous, current in pairwise(equities):
        if previous <= 0.0 or current <= 0.0:
            raise ValueError("non-positive equity in replay ledger")
        if not np.isfinite(previous) or not np.isfinite(current):
            raise ValueError("non-finite equity in replay ledger")
        growth.append(float(np.log(current / previous)))
    return tuple(growth)


def _filled_sessions(result: BacktestResult) -> int:
    return len({trade.session for trade in result.trades if trade.quantity > 0})


@dataclass(frozen=True, slots=True)
class _ScoredSessionIndex:
    """Pre-indexed session boundaries for zero-scan decision slicing."""

    stops: tuple[int, ...]
    available_times: tuple[datetime, ...]
    uniform: bool

    def stop_for(self, decision_time: datetime) -> int | None:
        """Row-prefix stop for every session available at or before decision_time."""
        if not self.uniform:
            return None
        position = bisect_right(self.available_times, decision_time)
        if position == 0:
            return 0
        return int(self.stops[position - 1])


def _scored_session_index(scored_market: pl.DataFrame) -> _ScoredSessionIndex:
    """Build contiguous session row stops from the session-sorted scored panel."""
    agg = (
        scored_market.group_by("session")
        .agg(
            pl.len().alias("__rows"),
            pl.col("available_time").max().alias("__available"),
            pl.col("available_time").n_unique().alias("__variants"),
        )
        .sort("session")
    )
    stops: list[int] = []
    running = 0
    for rows in agg["__rows"].to_list():
        running += int(rows)
        stops.append(running)
    available = tuple(agg["__available"].to_list())
    uniform = bool((agg["__variants"] == 1).all())
    monotonic = all(
        previous <= current for previous, current in pairwise(available)
    )
    return _ScoredSessionIndex(
        stops=tuple(stops),
        available_times=available,
        uniform=uniform and monotonic,
    )


_VOLATILITY_GATE_COLUMNS = frozenset({"data_quality_status", "is_universe", "tradable"})


def _can_precompute_volatility(scored_market: pl.DataFrame) -> bool:
    """Slicing/rolling precompute is exact only when no row-gate filter applies."""
    return not _VOLATILITY_GATE_COLUMNS.intersection(scored_market.columns)


def _precompute_volatility(frame: pl.DataFrame, window: int) -> pl.DataFrame:
    """Causally precompute the constructor's ``__vol`` column once per segment."""
    if "log_return" in frame.columns:
        returns = pl.col("log_return")
    elif "ret" in frame.columns:
        returns = pl.col("ret").log1p()
    else:
        returns = (
            pl.col("close").log() - pl.col("close").log().shift(1).over("instrument_id")
        )
    return frame.with_columns(
        returns.rolling_std(window_size=window, min_samples=2)
        .over("instrument_id")
        .alias("__vol")
    )


def _decision_provider(
    scored_market: pl.DataFrame,
    session_index: _ScoredSessionIndex,
) -> ReplayDecisionProvider:
    def provider(decision_time: datetime, execution_time: datetime) -> PreparedReplayDecision:
        stop = session_index.stop_for(decision_time)
        if stop is None:
            visible = scored_market.filter(pl.col("available_time") <= decision_time)
        else:
            visible = scored_market.slice(0, stop)
        return PreparedReplayDecision(decision_time, execution_time, visible)

    return provider


def _scenario_planner(
    context: ExecutionReplayContext,
    dataset_hash: str,
) -> ReplayScenarioPlanner:
    def planner(
        prepared: PreparedReplayDecision,
        portfolio: PortfolioSnapshot,
        cycle_request: TradingCycleRequest,
    ) -> TradingCycleResult:
        return plan_prepared_scored_cycle(
            prepared, portfolio, cycle_request, context.instruments, dataset_hash
        )

    return planner


def instruments_from_frame(frame: pl.DataFrame) -> dict[str, Instrument]:
    """Canonical instrument mapping for every instrument in an executable panel."""
    instruments: dict[str, Instrument] = {}
    for row in frame.select("instrument_id").unique().iter_rows(named=True):
        instrument_id = str(row["instrument_id"])
        instruments[instrument_id] = Instrument(
            instrument_id=instrument_id,
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=instrument_id.split(":")[-1],
            currency="KRW",
        )
    return instruments
