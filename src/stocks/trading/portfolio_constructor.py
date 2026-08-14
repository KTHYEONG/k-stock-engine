"""Deterministic capped inverse-return-volatility target construction.

The constructor consumes a *scored panel* (one row per instrument per session
carrying ``pred_score``, ``sector``, ``adtv``, and per-session returns) and a
reconciled ``PortfolioSnapshot``. It selects the latest decision cross-section,
sizes inverse-volatility weights under single-name, sector, capacity, gross,
and volatility caps, then applies a convex turnover interpolation when the
current portfolio is feasible or a sell-only ``DE_RISK`` plan when it is not.

All frame-level transforms are vectorized Polars/NumPy; per-row filtering,
``map_rows``, and ``pandas.apply`` are prohibited on this path.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from src.core.instruments import Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot
from src.stocks.trading.allocation_policy import rank_stock_candidate_indices

logger = logging.getLogger("stocks.trading.portfolio_constructor")

REQUIRED_CROSS_SECTION_COLUMNS = ("instrument_id", "pred_score", "sector", "adtv")
_ECONOMIC_COLUMNS = (
    "expected_active_alpha",
    "expected_net_alpha",
    "alpha_lower_bound",
    "exit_cost_rate",
)
_SESSION_COLUMN = "session"
_RETURN_COLUMNS = ("log_return", "ret", "close")
_TOLERANCE = 1e-10


class PortfolioConstraintError(ValueError):
    """Raised when constructed targets violate a hard portfolio constraint."""


@dataclass(frozen=True, slots=True)
class CompoundingPolicyConfig:
    """Immutable lower-confidence compounding overlay configuration.

    ``growth_risk_aversion`` prices horizon-unit variance against the
    confidence edge in the exponential utility overlay; it must be finite and
    strictly positive. ``enabled`` turns the overlay on for economic panels
    that expose ``net_alpha_lower_bound``; legacy score-only panels are
    unaffected either way.
    """

    enabled: bool = True
    growth_risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.growth_risk_aversion) or self.growth_risk_aversion <= 0.0:
            raise ValueError(
                "growth_risk_aversion must be finite and strictly positive"
            )


@dataclass(frozen=True, slots=True)
class PreparedAllocationMarket:
    """Immutable array-backed route market consumed by the prepared allocator.

    Owns the canonical sorted ``(session, instrument_id)`` row index plus the
    candidate-invariant ``close``, ADTV, sector, per-row log-return, and
    causal rolling-volatility arrays for one route OOS interval, so a candidate
    contributes only an aligned ``float64`` score overlay instead of re-joining
    a full market frame. ``session_ranges`` maps each session index to its
    contiguous row range and ``rows_by_key`` resolves ``(instrument_id,
    session)`` rows in ``O(1)``. When the market is dense (every session
    carries the same sorted instrument cross-section) ``returns_matrix`` gives
    the ``(n_sessions, n_instruments)`` return history so the prepared
    allocator builds per-decision windows without per-row Python work;
    ``instrument_position_of`` aligns every row to its ``sorted_instruments``
    column for the non-dense fallback.
    """

    sessions: tuple[datetime, ...]
    session_ranges: Mapping[int, tuple[int, int]]
    instrument_ids: np.ndarray
    row_session_of: np.ndarray
    row_sessions: np.ndarray
    close: np.ndarray
    adtv: np.ndarray
    sector: np.ndarray
    returns: np.ndarray
    volatility_lookback_sessions: int
    vol_series: np.ndarray
    dense: bool
    n_instruments: int
    sorted_instruments: np.ndarray
    instrument_position_of: np.ndarray
    instrument_position_lookup: Mapping[str, int]
    returns_matrix: np.ndarray
    rows_by_key: Mapping[tuple[str, datetime], int]
    cache_bytes: int

    @property
    def row_count(self) -> int:
        return int(self.instrument_ids.size)

    @classmethod
    def build(
        cls,
        frame: pl.DataFrame,
        volatility_lookback_sessions: int = 20,
    ) -> PreparedAllocationMarket:
        """Build the array-backed market from a validated route OOS frame.

        ``volatility_lookback_sessions`` fixes the cached causal rolling-std
        window; the reference path computes the same expression on the bounded
        decision window, so the cached per-row value is bit-identical for every
        decision cross-section whose lookback matches.
        """
        if volatility_lookback_sessions < 1:
            raise ValueError("volatility lookback sessions must be positive")
        missing = [c for c in ("instrument_id", _SESSION_COLUMN, "sector", "adtv", "close") if c not in frame.columns]
        if missing:
            raise ValueError(f"prepared market frame must carry {', '.join(missing)}")
        ordered = frame.sort([_SESSION_COLUMN, "instrument_id"])
        if ordered.is_empty():
            raise ValueError("prepared market frame has no rows")
        sessions = tuple(
            datetime.fromisoformat(str(s))
            if not isinstance(s, datetime)
            else s
            for s in ordered[_SESSION_COLUMN].unique().sort().to_list()
        )
        session_index_of = {session: i for i, session in enumerate(sessions)}
        row_sessions_list = [
            session_index_of[
                datetime.fromisoformat(str(s)) if not isinstance(s, datetime) else s
            ]
            for s in ordered[_SESSION_COLUMN].to_list()
        ]
        ranges: dict[int, tuple[int, int]] = {}
        current = -1
        start = 0
        for i, session_idx in enumerate(row_sessions_list):
            if session_idx != current:
                if current != -1:
                    ranges[current] = (start, i)
                current = session_idx
                start = i
        ranges[current] = (start, len(row_sessions_list))
        row_sessions = np.asarray(
            [sessions[i] for i in row_sessions_list], dtype=object
        )
        instrument_ids = np.asarray(
            [str(i) for i in ordered["instrument_id"].to_list()], dtype=object
        )
        unique_ids = [str(i) for i in ordered["instrument_id"].unique().sort().to_list()]
        n_instruments = len(unique_ids)
        sorted_instruments = np.asarray(unique_ids, dtype=object)
        position_map = {instrument: position for position, instrument in enumerate(unique_ids)}
        instrument_position_of = np.asarray(
            [position_map[str(i)] for i in ordered["instrument_id"].to_list()],
            dtype=np.int64,
        )
        logret = pl.col("close").log() - pl.col("close").log().shift(1).over(
            "instrument_id"
        )
        with_ret = ordered.with_columns(
            logret.alias("__ret"),
            logret.rolling_std(
                window_size=volatility_lookback_sessions, min_samples=2
            )
            .over("instrument_id")
            .alias("__vol"),
        )
        returns = with_ret["__ret"].to_numpy().astype(np.float64)
        vol_series = with_ret["__vol"].to_numpy().astype(np.float64)
        n_sessions = len(sessions)
        dense = ordered.height == n_sessions * n_instruments
        if dense:
            first_session_ids = [
                str(i) for i in ordered["instrument_id"][ranges[0][0]:ranges[0][1]].to_list()
            ]
            dense = first_session_ids == unique_ids
        if dense:
            returns_matrix = returns.reshape(n_sessions, n_instruments)
        else:
            returns_matrix = np.full(
                (n_sessions, n_instruments), np.nan, dtype=np.float64
            )
            for session_index in range(n_sessions):
                lo, hi = ranges[session_index]
                returns_matrix[
                    session_index, instrument_position_of[lo:hi]
                ] = returns[lo:hi]
        rows_by_key: dict[tuple[str, datetime], int] = {}
        market = cls(
            sessions=sessions,
            session_ranges=ranges,
            instrument_ids=instrument_ids,
            row_session_of=np.asarray(row_sessions_list, dtype=np.int64),
            row_sessions=row_sessions,
            close=ordered["close"].to_numpy().astype(np.float64),
            adtv=ordered["adtv"].to_numpy().astype(np.float64),
            sector=np.asarray(ordered["sector"].to_list(), dtype=object),
            returns=returns,
            volatility_lookback_sessions=volatility_lookback_sessions,
            vol_series=vol_series,
            dense=dense,
            n_instruments=n_instruments,
            sorted_instruments=sorted_instruments,
            instrument_position_of=instrument_position_of,
            instrument_position_lookup=position_map,
            returns_matrix=returns_matrix,
            rows_by_key=rows_by_key,
            cache_bytes=int(ordered.estimated_size()),
        )
        for i in range(ordered.height):
            rows_by_key[
                (str(ordered["instrument_id"][i]), sessions[int(row_sessions_list[i])])
            ] = i
        return market


def construct_target_allocations_prepared(
    market: PreparedAllocationMarket,
    decision_index: int,
    score_overlay: np.ndarray,
    calibration_state: Mapping[str, object] | None,
    instruments: Mapping[str, Instrument],
    portfolio: PortfolioSnapshot,
    policy: StockRiskPolicy,
) -> tuple[Allocation, ...]:
    """Construct constrained targets from prepared causal arrays and one overlay.

    The hot path builds the bounded ``(volatility, covariance)`` history window
    directly from the immutable causal return arrays, ranks the current
    cross-section, applies the frozen ``calibration_state`` bucket table when
    supplied, and runs the same constrained target construction as the
    reference :func:`construct_target_allocations` -- with the shrinkage
    covariance computed once per ``(decision, selected-name set)`` and reused
    for volatility scaling, compounding scaling, and post-validation. The
    public :func:`construct_target_allocations` and
    :meth:`PreparedSelectionRoute.window_frame` are never invoked on this path;
    the reference implementation is kept unchanged as the parity oracle.
    ``score_overlay`` must be aligned to ``market`` rows (``NaN`` where the
    candidate has no score). Raises ``ValueError`` on an overlay length
    mismatch or a non-finite market input.
    """
    if score_overlay is None or len(score_overlay) != market.row_count:
        raise ValueError(
            f"score overlay length {0 if score_overlay is None else len(score_overlay)} "
            f"does not match prepared market row count {market.row_count}"
        )
    if not np.all(np.isfinite(market.close)) or not np.all(np.isfinite(market.adtv)):
        raise ValueError("prepared market close/adtv must be finite")
    overlay = np.asarray(score_overlay, dtype=np.float64)
    scored = np.where(~np.isnan(overlay))[0]
    if scored.size and not bool(np.all(np.isfinite(overlay[scored]))):
        raise ValueError("prepared score overlay carries non-finite scored values")
    if decision_index < 0 or decision_index >= len(market.sessions):
        raise ValueError(f"decision_index {decision_index} outside route sessions")
    return _construct_allocations_prepared(
        market, decision_index, overlay, calibration_state,
        instruments, portfolio, policy,
    )


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Average (dense) ranks of ``values``, mirroring Polars ``rank('average')``.

    Ranks are assigned over the non-null values ascending; exact float ties
    receive the mean of their ordinal ranks, matching the reference bucket
    expression exactly.
    """
    n = values.size
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _prepared_economics(
    calibration_state: Mapping[str, object],
    scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """Frozen bucket statistics joined onto the decision cross-section scores.

    Mirrors ``CausalAlphaCalibrator.apply_prepared`` restricted to the decision
    cross-section: scores are average-ranked within the session, bucketed, and
    joined to the frozen bucket table, then the net columns and exit cost are
    derived. Unscored rows and rows without bucket statistics stay ``NaN`` so
    the economic gates exclude them exactly as the reference null semantics do.
    """
    bucket_count = int(cast(int, calibration_state.get("bucket_count", 0)))
    round_trip_cost = float(
        cast(float, calibration_state.get("round_trip_cost", 0.0))
    )
    exit_cost_rate = float(
        cast(float, calibration_state.get("exit_cost_rate", 0.0))
    )
    bucket_stats = {
        int(cast(int, bucket["bucket"])): (
            bucket.get("expected_active_alpha"),
            bucket.get("alpha_lower_bound"),
        )
        for bucket in cast(
            Sequence[Mapping[str, object]],
            calibration_state.get("buckets") or [],
        )
    }
    n = scores.size
    active = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    scored_positions = np.where(~np.isnan(scores))[0]
    if scored_positions.size:
        values = scores[scored_positions]
        within = int(scored_positions.size)
        if within > 1:
            pct_rank = (_average_rank(values) - 1.0) / (within - 1.0)
        else:
            pct_rank = np.full(within, 0.5, dtype=np.float64)
        buckets = (
            np.floor(pct_rank * bucket_count).clip(0, bucket_count - 1).astype(np.int64)
        )
        for position, bucket in zip(scored_positions, buckets, strict=True):
            stats = bucket_stats.get(int(bucket))
            if stats is None:
                continue
            active_alpha, alpha_lower = stats
            if active_alpha is not None:
                active[position] = float(cast(float, active_alpha))
            if alpha_lower is not None:
                lower[position] = float(cast(float, alpha_lower))
    net_alpha = np.where(np.isnan(active), np.nan, active - round_trip_cost)
    net_lower = np.where(np.isnan(lower), np.nan, lower - round_trip_cost)
    return {
        "expected_active_alpha": active,
        "alpha_lower_bound": lower,
        "expected_net_alpha": net_alpha,
        "net_alpha_lower_bound": net_lower,
        "exit_cost_rate": np.full(n, exit_cost_rate, dtype=np.float64),
    }


def _window_returns(
    market: PreparedAllocationMarket,
    start: int,
    decision_index: int,
) -> np.ndarray:
    """Bounded ``(window_sessions, n_instruments)`` return matrix.

    The matrix carries within-window log returns: the first window appearance
    of every instrument is ``NaN`` exactly as the reference window-frame diff
    produces, and later appearances use the causal full-market diff. Dense
    markets are served by the precomputed reshape; the scatter path handles
    any cross-section gap identically.
    """
    if market.dense:
        result = market.returns_matrix[start:decision_index + 1].copy()
        result[0, :] = np.nan
        return result
    result = np.full(
        (decision_index - start + 1, market.n_instruments), np.nan, dtype=np.float64
    )
    seen = np.zeros(market.n_instruments, dtype=bool)
    for offset, session_index in enumerate(range(start, decision_index + 1)):
        lo, hi = market.session_ranges[session_index]
        cols = market.instrument_position_of[lo:hi]
        first = ~seen[cols]
        values = market.returns[lo:hi]
        result[offset, cols] = np.where(first, np.nan, values)
        seen[cols] = True
    return result


def _prepared_return_matrix(
    returns: np.ndarray,
    market: PreparedAllocationMarket,
    selected_ids: Sequence[str],
) -> np.ndarray | None:
    """Causal raw return window for ``selected_ids`` mirroring the reference pivot.

    Columns follow ``selected_ids`` order and rows are the full decision
    window sessions, carrying ``NaN`` for any unobserved return, exactly like
    the reference :func:`_return_matrix`. ``None`` reproduces the reference
    ``insufficient covariance data`` structural failure when a selected name
    has no column in the market.
    """
    cols = [
        market.instrument_position_lookup.get(instrument_id)
        for instrument_id in selected_ids
    ]
    if any(col is None for col in cols):
        return None
    selected = returns[:, cast(list[int], cols)]
    if selected.ndim != 2 or selected.shape[1] != len(selected_ids):
        return None
    return selected


def _prepared_vol_fallback(
    market: PreparedAllocationMarket,
    returns_window: np.ndarray,
    decision_index: int,
    lookback: int,
) -> np.ndarray:
    """Causal rolling volatility for a mismatched lookback via the exact Polars expression."""
    cs_lo, cs_hi = market.session_ranges[decision_index]
    cols = market.instrument_position_of[cs_lo:cs_hi]
    out = np.full(cs_hi - cs_lo, np.nan, dtype=np.float64)
    window_rows = returns_window[-lookback:, :]
    for offset, col in enumerate(cols):
        rolled = pl.Series("__ret", window_rows[:, col]).rolling_std(
            window_size=lookback, min_samples=2
        )
        tail_value = rolled[-1]
        if tail_value is not None:
            out[offset] = float(tail_value)
    return out


def _prepared_equity_and_prices(
    portfolio: PortfolioSnapshot,
    cs_ids: np.ndarray,
    close: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Mark-to-market equity and close price map for the decision cross-section."""
    price_map = {
        str(cs_ids[position]): float(close[position])
        for position in range(cs_ids.size)
        if not np.isnan(close[position])
    }
    equity = portfolio.equity(price_map)
    if not math.isfinite(equity) or equity <= 0:
        raise PortfolioConstraintError(f"portfolio equity must be finite and positive, got {equity}")
    return equity, price_map


def _prepared_priority_alpha_of(
    selected_ids: Sequence[str],
    incumbent_ids: set[str],
    econ: Mapping[str, np.ndarray],
    cs_ids: np.ndarray,
) -> dict[str, float]:
    """Per-name priority alpha mirroring :func:`_priority_alpha_of`."""
    active = econ["expected_active_alpha"]
    net = econ["expected_net_alpha"]
    exit_rate = econ["exit_cost_rate"]
    net_alpha: dict[str, float] = {}
    active_alpha: dict[str, float] = {}
    exit_cost: dict[str, float] = {}
    for position in range(cs_ids.size):
        instrument_id = str(cs_ids[position])
        if not np.isnan(net[position]):
            net_alpha[instrument_id] = float(net[position])
        if not np.isnan(active[position]):
            active_alpha[instrument_id] = float(active[position])
        exit_cost[instrument_id] = float(exit_rate[position])
    priority: dict[str, float] = {}
    for instrument_id in selected_ids:
        if instrument_id in incumbent_ids:
            priority[instrument_id] = max(
                active_alpha.get(instrument_id, 0.0)
                - exit_cost.get(instrument_id, 0.0),
                0.0,
            )
        else:
            priority[instrument_id] = max(net_alpha.get(instrument_id, 0.0), 0.0)
    return priority


def _construct_allocations_prepared(
    market: PreparedAllocationMarket,
    decision_index: int,
    overlay: np.ndarray,
    calibration_state: Mapping[str, object] | None,
    instruments: Mapping[str, Instrument],
    portfolio: PortfolioSnapshot,
    policy: StockRiskPolicy,
) -> tuple[Allocation, ...]:
    """Array-backed constrained target construction for one prepared decision."""
    window_len = (
        max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
        + 1
    )
    start = max(0, decision_index - window_len + 1)
    cs_lo, cs_hi = market.session_ranges[decision_index]
    cs_slice = slice(cs_lo, cs_hi)
    scores = overlay[cs_slice]
    close = market.close[cs_slice]
    adtv = market.adtv[cs_slice]
    sector = market.sector[cs_slice]
    cs_ids = market.instrument_ids[cs_slice]
    returns_window = _window_returns(market, start, decision_index)

    if policy.volatility_lookback_sessions == market.volatility_lookback_sessions:
        vol = market.vol_series[cs_slice].astype(np.float64)
    else:
        vol = _prepared_vol_fallback(
            market, returns_window, decision_index, policy.volatility_lookback_sessions
        )

    eligible = (
        ~np.isnan(scores)
        & ~np.isnan(vol)
        & (vol > 0.0)
        & (adtv > 0.0)
        & ~np.isnan(close)
    )
    if not bool(np.any(eligible)):
        return ()
    equity, price_map = _prepared_equity_and_prices(portfolio, cs_ids, close)
    current_weights = _current_weights(portfolio, price_map, equity)
    incumbent_ids = set(current_weights)

    econ: dict[str, np.ndarray] | None = None
    if calibration_state is not None:
        econ = _prepared_economics(calibration_state, scores)
        active = econ["expected_active_alpha"]
        net_alpha = econ["expected_net_alpha"]
        lower = econ["alpha_lower_bound"]
        net_lower = econ["net_alpha_lower_bound"]
        exit_rate = econ["exit_cost_rate"]
        incumbent_mask = np.asarray(
            [str(instrument_id) in incumbent_ids for instrument_id in cs_ids], dtype=bool
        )
        keep_ok = (
            (active - exit_rate > 0.0)
            & (active > 0.0)
            & (lower > 0.0)
        )
        enter_ok = (
            (net_alpha > 0.0)
            & (active > 0.0)
            & (lower > 0.0)
            & (net_lower > 0.0)
        )
        gate = np.where(incumbent_mask, keep_ok, enter_ok)
        eligible = eligible & np.where(np.isnan(active), False, gate)
        if not bool(np.any(eligible)):
            return ()

    positions = np.where(eligible)[0]
    preds = scores[positions]
    id_strs = np.asarray([str(cs_ids[position]) for position in positions], dtype=object)
    order = rank_stock_candidate_indices(preds, id_strs)
    positions_sorted = positions[order]
    ranks = np.arange(1, positions_sorted.size + 1, dtype=np.int64)
    incumbent_of = np.asarray(
        [str(cs_ids[position]) in incumbent_ids for position in positions_sorted],
        dtype=bool,
    )
    keep = (ranks <= policy.enter_rank) | (
        incumbent_of & (ranks <= policy.keep_rank)
    )
    ranked_positions = positions_sorted[keep]
    ranked_count = int(ranked_positions.size)
    ranked_positions = ranked_positions[: policy.top_k]
    if ranked_positions.size == 0:
        return ()
    selected_ids = [str(cs_ids[position]) for position in ranked_positions]
    selected_count = len(selected_ids)
    candidate_count = int(eligible.sum())
    decision_session = market.sessions[decision_index]

    net_lower_bound_of: dict[str, float] = {}
    if policy.compounding.enabled and econ is not None:
        net_lower_all = econ["net_alpha_lower_bound"]
        for position in range(cs_ids.size):
            value = net_lower_all[position]
            if not np.isnan(value):
                net_lower_bound_of[str(cs_ids[position])] = float(value)
        for instrument_id in selected_ids:
            if (
                instrument_id not in net_lower_bound_of
                or not math.isfinite(net_lower_bound_of[instrument_id])
            ):
                _record_compounding_decision(
                    policy,
                    decision_session=decision_session,
                    candidate_count=candidate_count,
                    ranked_count=ranked_count,
                    selected_count=selected_count,
                    confidence_edge_h=None,
                    confidence_variance_h=None,
                    confidence_scale=None,
                    gross_before_compounding=0.0,
                    gross_after_compounding=0.0,
                    turnover_lambda=0.0,
                    cash_reason="invalid-confidence-variance",
                )
                return ()

    sector_of = {str(cs_ids[position]): sector[position] for position in range(cs_ids.size)}
    adtv_of = {str(cs_ids[position]): float(adtv[position]) for position in range(cs_ids.size)}
    vol_of = {str(cs_ids[position]): float(vol[position]) for position in ranked_positions}
    priority_alpha_of: dict[str, float] = {}
    if econ is not None:
        priority_alpha_of = _prepared_priority_alpha_of(
            selected_ids, incumbent_ids, econ, cs_ids
        )

    feasible = _portfolio_is_feasible(current_weights, sector_of, equity, policy)
    if feasible:
        ids_matrix = _prepared_return_matrix(
            returns_window, market, selected_ids
        )
        if ids_matrix is None:
            raise PortfolioConstraintError("insufficient covariance data")
        covariance, covariance_source = causal_covariance_or_fallback(
            ids_matrix,
            volatility_lookback_sessions=policy.volatility_lookback_sessions,
            covariance_lookback_sessions=policy.covariance_lookback_sessions,
        )
        allocations = _build_allocations(
            selected_ids,
            sector_of,
            adtv_of,
            vol_of,
            current_weights,
            equity,
            pl.DataFrame(),
            instruments,
            policy,
            priority_alpha_of=priority_alpha_of,
            net_lower_bound_of=net_lower_bound_of,
            decision_session=decision_session,
            candidate_count=candidate_count,
            ranked_count=ranked_count,
            selected_count=selected_count,
            covariance=covariance,
            covariance_source=covariance_source,
        )
    else:
        allocations = _de_risk_allocations(
            current_weights, sector_of, equity, instruments, policy
        )

    post_weights = {
        allocation.instrument.instrument_id: allocation.target_value / equity
        for allocation in allocations
        if allocation.target_value > 0
    }
    post_covariance: np.ndarray | None = None
    if post_weights:
        post_matrix = _prepared_return_matrix(
            returns_window, market, sorted(post_weights)
        )
        if post_matrix is None:
            raise PortfolioConstraintError("insufficient covariance data")
        post_covariance, _ = causal_covariance_or_fallback(
            post_matrix,
            volatility_lookback_sessions=policy.volatility_lookback_sessions,
            covariance_lookback_sessions=policy.covariance_lookback_sessions,
        )
    _post_validate(
        allocations, equity, policy, sector_of, adtv_of, pl.DataFrame(),
        covariance=post_covariance,
        current_weights=current_weights,
    )
    return tuple(sorted(allocations, key=lambda a: a.instrument.instrument_id))


@dataclass(frozen=True, slots=True)
class StockRiskPolicy:
    """Frozen, versioned risk profile for target construction.

    The seed profile is a configuration artifact frozen before holdout; live
    mode must load an explicitly identified policy artifact rather than
    silently instantiate these defaults.
    """

    top_k: int = 20
    target_count: int | None = None
    enter_rank: int = 15
    keep_rank: int = 30
    gross_cap: float = 0.90
    single_name_cap: float = 0.08
    sector_cap: float = 0.25
    participation_limit: float = 0.005
    target_annual_volatility: float = 0.12
    turnover_budget: float = 0.20
    volatility_lookback_sessions: int = 20
    covariance_lookback_sessions: int = 60
    rebalance_frequency_sessions: int = 5
    annualization_sessions: int = 252
    economic_hysteresis: bool = True
    compounding: CompoundingPolicyConfig = field(default_factory=CompoundingPolicyConfig)
    compounding_evidence: list[dict[str, object]] = field(
        default_factory=list, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.target_count is not None:
            if self.target_count <= 0:
                raise ValueError("target_count must be positive")
            if self.target_count != self.top_k or self.target_count != self.enter_rank:
                raise ValueError(
                    "target_count, top_k, and enter_rank must agree for v3 policy"
                )
        if not (0 < self.enter_rank <= self.keep_rank):
            raise ValueError("ranks must satisfy 0 < enter_rank <= keep_rank")
        if not (0.0 < self.single_name_cap <= self.sector_cap <= self.gross_cap <= 1.0):
            raise ValueError(
                "caps must satisfy 0 < single_name_cap <= sector_cap <= gross_cap <= 1"
            )
        if not (0.0 <= self.participation_limit <= 1.0):
            raise ValueError("participation_limit must be in [0, 1]")
        if self.target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be positive")
        if self.turnover_budget < 0:
            raise ValueError("turnover_budget must be non-negative")
        if (
            self.volatility_lookback_sessions <= 0
            or self.covariance_lookback_sessions <= 0
            or self.rebalance_frequency_sessions <= 0
            or self.annualization_sessions <= 0
        ):
            raise ValueError("lookbacks, frequency, and annualization must be positive")


def construct_target_allocations(
    scores: pl.DataFrame,
    instruments: Mapping[str, Instrument],
    portfolio: PortfolioSnapshot,
    policy: StockRiskPolicy,
) -> tuple[Allocation, ...]:
    """Build constrained long-only target allocations from a scored panel.

    Returns allocations sorted by ``instrument_id``. When no input is eligible,
    returns an empty tuple rather than synthetic weights.
    """
    _validate_scores_frame(scores, instruments)
    panel = scores.sort([_SESSION_COLUMN, "instrument_id"])
    returns = _returns_column(panel)
    panel = panel.with_columns(
        returns.rolling_std(
            window_size=policy.volatility_lookback_sessions,
            min_samples=2,
        )
        .over("instrument_id")
        .alias("__vol"),
    )

    cross_section = _latest_cross_section(panel)
    eligible = cross_section.filter(
        pl.col("pred_score").is_not_null()
        & pl.col("__vol").is_not_null()
        & (pl.col("__vol") > 0)
        & (pl.col("adtv") > 0)
        & (pl.col("sector").is_not_null())
    )
    if eligible.is_empty():
        return ()

    equity = _portfolio_equity(portfolio, cross_section)
    price_map = _price_map(cross_section)
    current_weights = _current_weights(portfolio, price_map, equity)
    incumbent_ids = set(current_weights)

    if all(column in cross_section.columns for column in _ECONOMIC_COLUMNS):
        eligible = _economically_eligible(cross_section, eligible, incumbent_ids)

    order = rank_stock_candidate_indices(
        np.asarray(eligible["pred_score"].to_list(), dtype=np.float64),
        np.asarray(eligible["instrument_id"].to_list(), dtype=object),
    )
    ranked = (
        eligible.gather(order)
        .with_row_index("__rank", offset=1)
        .filter(
            (pl.col("__rank") <= policy.enter_rank)
            | (
                pl.col("instrument_id").is_in(incumbent_ids)
                & (pl.col("__rank") <= policy.keep_rank)
            )
        )
    )
    ranked_count = ranked.height
    ranked = ranked.head(policy.top_k).drop("__rank")
    if ranked.is_empty():
        return ()

    ids = [str(r) for r in ranked["instrument_id"].to_list()]
    decision_session = cross_section[_SESSION_COLUMN][0]
    candidate_count = eligible.height
    selected_count = len(ids)
    net_lower_bound_of: dict[str, float] = {}
    if policy.compounding.enabled and "net_alpha_lower_bound" in cross_section.columns:
        net_lower_bound_of = {
            str(r["instrument_id"]): float(r["net_alpha_lower_bound"])
            for r in cross_section.to_dicts()
            if r["net_alpha_lower_bound"] is not None
        }
        for instrument_id in ids:
            if instrument_id not in net_lower_bound_of or not math.isfinite(
                net_lower_bound_of[instrument_id]
            ):
                _record_compounding_decision(
                    policy,
                    decision_session=decision_session,
                    candidate_count=candidate_count,
                    ranked_count=ranked_count,
                    selected_count=selected_count,
                    confidence_edge_h=None,
                    confidence_variance_h=None,
                    confidence_scale=None,
                    gross_before_compounding=0.0,
                    gross_after_compounding=0.0,
                    turnover_lambda=0.0,
                    cash_reason="invalid-confidence-variance",
                )
                return ()
    sector_of = {str(r["instrument_id"]): r["sector"] for r in cross_section.to_dicts()}
    adtv_of = {str(r["instrument_id"]): float(r["adtv"]) for r in cross_section.to_dicts()}
    vol_of = {str(r["instrument_id"]): float(r["__vol"]) for r in ranked.to_dicts()}
    priority_alpha_of = _priority_alpha_of(
        ranked, incumbent_ids, cross_section
    )

    feasible = _portfolio_is_feasible(current_weights, sector_of, equity, policy)
    if feasible:
        allocations = _build_allocations(
            ids,
            sector_of,
            adtv_of,
            vol_of,
            current_weights,
            equity,
            panel,
            instruments,
            policy,
            priority_alpha_of=priority_alpha_of,
            net_lower_bound_of=net_lower_bound_of,
            decision_session=decision_session,
            candidate_count=candidate_count,
            ranked_count=ranked_count,
            selected_count=selected_count,
        )
    else:
        allocations = _de_risk_allocations(
            current_weights, sector_of, equity, instruments, policy
        )

    _post_validate(
        allocations, equity, policy, sector_of, adtv_of, panel,
        current_weights=current_weights,
    )
    return tuple(sorted(allocations, key=lambda a: a.instrument.instrument_id))


def _validate_scores_frame(scores: pl.DataFrame, instruments: Mapping[str, Instrument]) -> None:
    missing = [c for c in REQUIRED_CROSS_SECTION_COLUMNS if c not in scores.columns]
    if missing:
        raise ValueError(f"scores panel must carry {', '.join(missing)}")
    if _SESSION_COLUMN not in scores.columns:
        raise ValueError(f"scores panel must carry {_SESSION_COLUMN!r}")
    if not any(c in scores.columns for c in _RETURN_COLUMNS):
        raise ValueError(f"scores panel must carry one of {_RETURN_COLUMNS}")
    if "close" not in scores.columns:
        raise ValueError("scores panel must carry a close column for valuation")
    known = set(instruments)
    present = {str(i) for i in scores["instrument_id"].unique().to_list()}
    unknown = present - known
    if unknown:
        raise PortfolioConstraintError(f"unknown instruments in scores panel: {sorted(unknown)}")


def _returns_column(panel: pl.DataFrame) -> pl.Expr:
    """Return the log-return expression, converting simple returns via log1p."""
    if "log_return" in panel.columns:
        return pl.col("log_return")
    if "ret" in panel.columns:
        return pl.col("ret").log1p()
    return pl.col("close").log() - pl.col("close").log().shift(1).over("instrument_id")


def _latest_cross_section(panel: pl.DataFrame) -> pl.DataFrame:
    """Select exactly the latest decision session per instrument."""
    if panel.is_empty():
        return panel
    latest = panel.select(pl.col(_SESSION_COLUMN).max()).to_series()[0]
    return panel.filter(pl.col(_SESSION_COLUMN) == latest)


def _price_map(cross_section: pl.DataFrame) -> dict[str, float]:
    return {
        str(row["instrument_id"]): float(row["close"])
        for row in cross_section.select(["instrument_id", "close"]).to_dicts()
        if row["close"] is not None
    }


def _portfolio_equity(portfolio: PortfolioSnapshot, cross_section: pl.DataFrame) -> float:
    equity = portfolio.equity(_price_map(cross_section))
    if not math.isfinite(equity) or equity <= 0:
        raise PortfolioConstraintError(f"portfolio equity must be finite and positive, got {equity}")
    return equity


def _current_weights(
    portfolio: PortfolioSnapshot,
    price_map: dict[str, float],
    equity: float,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for position in portfolio.positions:
        instrument_id = position.instrument.instrument_id
        if instrument_id in price_map:
            weights[instrument_id] = position.quantity * price_map[instrument_id] / equity
    return weights


def _economically_eligible(
    cross_section: pl.DataFrame,
    eligible: pl.DataFrame,
    incumbent_ids: set[str],
) -> pl.DataFrame:
    """Gate entries and holdings on cost-adjusted expected net alpha.

    New entrants require a positive ``expected_net_alpha`` (cost-adjusted
    expected active return), a positive ``alpha_lower_bound`` confidence bound,
    and a positive ``net_alpha_lower_bound`` (the bootstrap lower bound net of
    the full round-trip cost) when the economic panel exposes it. Existing
    holdings are retained only while the keep-versus-exit net benefit
    (``expected_active_alpha`` minus the one-way sell cost) is positive with a
    positive confidence bound. Null/non-finite alpha is never a buy signal:
    those rows fail the gate and fall through to cash or sell-only.
    """
    if "expected_active_alpha" not in cross_section.columns:
        return eligible
    incumbent = pl.col("instrument_id").is_in(incumbent_ids)
    keep_ok = (
        (pl.col("expected_active_alpha") - pl.col("exit_cost_rate") > 0.0)
        & (pl.col("expected_active_alpha") > 0.0)
        & (pl.col("alpha_lower_bound") > 0.0)
    )
    net_lower_bound_ok = (
        (pl.col("net_alpha_lower_bound") > 0.0)
        if "net_alpha_lower_bound" in cross_section.columns
        else pl.lit(True)
    )
    enter_ok = (
        (pl.col("expected_net_alpha") > 0.0)
        & (pl.col("expected_active_alpha") > 0.0)
        & (pl.col("alpha_lower_bound") > 0.0)
        & net_lower_bound_ok
    )
    gate = pl.when(incumbent).then(keep_ok).otherwise(enter_ok)
    return eligible.filter(gate.fill_null(False))


def _priority_alpha_of(
    ranked: pl.DataFrame,
    incumbent_ids: set[str],
    cross_section: pl.DataFrame,
) -> dict[str, float]:
    """Per-name positive expected alpha used for relative sizing priority.

    New names size by ``max(expected_net_alpha, 0)``; incumbents size by the
    keep-versus-exit net benefit ``max(expected_active_alpha - exit_cost, 0)``.
    """
    if "expected_net_alpha" not in ranked.columns:
        return {}
    ids = [str(r) for r in ranked["instrument_id"].to_list()]
    net_alpha = {
        str(r["instrument_id"]): float(r["expected_net_alpha"])
        for r in cross_section.to_dicts()
        if r["expected_net_alpha"] is not None
    }
    active_alpha = {
        str(r["instrument_id"]): float(r["expected_active_alpha"])
        for r in cross_section.to_dicts()
        if r["expected_active_alpha"] is not None
    }
    exit_cost = {
        str(r["instrument_id"]): float(r["exit_cost_rate"])
        for r in cross_section.to_dicts()
        if r["exit_cost_rate"] is not None
    }
    priority: dict[str, float] = {}
    for instrument_id in ids:
        if instrument_id in incumbent_ids:
            priority[instrument_id] = max(
                active_alpha.get(instrument_id, 0.0)
                - exit_cost.get(instrument_id, 0.0),
                0.0,
            )
        else:
            priority[instrument_id] = max(net_alpha.get(instrument_id, 0.0), 0.0)
    return priority


def _build_allocations(
    ids: list[str],
    sector_of: dict[str, object],
    adtv_of: dict[str, float],
    vol_of: dict[str, float],
    current_weights: dict[str, float],
    equity: float,
    panel: pl.DataFrame,
    instruments: Mapping[str, Instrument],
    policy: StockRiskPolicy,
    *,
    priority_alpha_of: dict[str, float] | None = None,
    net_lower_bound_of: dict[str, float] | None = None,
    decision_session: object = "",
    candidate_count: int = 0,
    ranked_count: int = 0,
    selected_count: int = 0,
    covariance: np.ndarray | None = None,
    covariance_source: str = "",
) -> tuple[Allocation, ...]:
    if covariance is None:
        covariance, covariance_source = _covariance(panel, ids, policy)
    if priority_alpha_of:
        raw_scores = {
            instrument_id: priority_alpha_of.get(instrument_id, 0.0)
            / max(vol_of[instrument_id] ** 2, _TOLERANCE)
            for instrument_id in ids
        }
    else:
        raw_scores = {
            instrument_id: 1.0 / max(vol_of[instrument_id], _TOLERANCE)
            for instrument_id in ids
        }
    total = sum(raw_scores.values()) or 1.0

    weights: dict[str, float] = {}
    for instrument_id in ids:
        raw = raw_scores[instrument_id] / total * policy.gross_cap
        weights[instrument_id] = min(raw, policy.single_name_cap)

    weights = _scale_sectors(weights, sector_of, policy.sector_cap)
    weights = _scale_gross(weights, policy.gross_cap)
    weights = _scale_volatility(
        weights, panel, ids, policy, covariance=covariance
    )

    gross_before_compounding = sum(weights.values())
    compounding_applied = False
    confidence_edge_h: float | None = None
    confidence_variance_h: float | None = None
    confidence_scale: float | None = None
    if policy.compounding.enabled and net_lower_bound_of:
        confidence_scale, confidence_edge_h, confidence_variance_h, cash_reason = (
            _compounding_scale(
                weights, net_lower_bound_of, panel, ids, policy,
                covariance=covariance,
            )
        )
        if cash_reason is not None:
            _record_compounding_decision(
                policy,
                decision_session=decision_session,
                candidate_count=candidate_count,
                ranked_count=ranked_count,
                selected_count=selected_count,
                confidence_edge_h=confidence_edge_h,
                confidence_variance_h=confidence_variance_h,
                confidence_scale=confidence_scale,
                gross_before_compounding=gross_before_compounding,
                gross_after_compounding=0.0,
                turnover_lambda=0.0,
                cash_reason=cash_reason,
                covariance_source=covariance_source,
            )
            return ()
        weights = {
            instrument_id: weight * confidence_scale
            for instrument_id, weight in weights.items()
        }
        compounding_applied = True
    gross_after_compounding = sum(weights.values())

    if policy.turnover_budget > 0.0:
        target_full = dict.fromkeys(current_weights, 0.0)
        for instrument_id, weight in weights.items():
            target_full[instrument_id] = weight
        lambda_ = _turnover_lambda(target_full, current_weights, policy.turnover_budget)
        for instrument_id in target_full:
            current = current_weights.get(instrument_id, 0.0)
            target_full[instrument_id] = current + lambda_ * (
                target_full[instrument_id] - current
            )
    else:
        target_full = dict(weights)
        lambda_ = 1.0

    if policy.participation_limit > 0.0:
        for instrument_id in current_weights:
            target_full.setdefault(instrument_id, 0.0)
        for instrument_id, target in target_full.items():
            current = current_weights.get(instrument_id, 0.0)
            delta_cap = policy.participation_limit * adtv_of.get(
                instrument_id, 0.0
            ) / equity
            if delta_cap <= 0.0:
                raise PortfolioConstraintError(
                    f"missing capacity for {instrument_id}"
                )
            target_full[instrument_id] = min(
                max(target, current - delta_cap), current + delta_cap
            )

    if compounding_applied:
        _record_compounding_decision(
            policy,
            decision_session=decision_session,
            candidate_count=candidate_count,
            ranked_count=ranked_count,
            selected_count=selected_count,
            confidence_edge_h=confidence_edge_h,
            confidence_variance_h=confidence_variance_h,
            confidence_scale=confidence_scale,
            gross_before_compounding=gross_before_compounding,
            gross_after_compounding=gross_after_compounding,
            turnover_lambda=lambda_,
            cash_reason=None,
            covariance_source=covariance_source,
        )

    allocations: list[Allocation] = []
    for instrument_id, weight in target_full.items():
        if weight <= 0.0:
            continue
        if instrument_id in current_weights and instrument_id not in weights:
            reason = "turnover-reduction"
        elif instrument_id in ids:
            reason = "inverse-vol-constrained"
        else:
            continue
        allocations.append(
            Allocation(
                instrument=_instrument(instrument_id, instruments),
                target_value=max(0.0, weight) * equity,
                reason=reason,
            )
        )
    return tuple(allocations)


def _scale_sectors(
    weights: dict[str, float],
    sector_of: dict[str, object],
    sector_cap: float,
) -> dict[str, float]:
    sector_total: dict[object, float] = {}
    for instrument_id, weight in weights.items():
        sector = sector_of.get(instrument_id)
        sector_total[sector] = sector_total.get(sector, 0.0) + weight
    scaled = dict(weights)
    for sector, total in sector_total.items():
        if total > sector_cap:
            factor = sector_cap / total
            for instrument_id in weights:
                if sector_of.get(instrument_id) == sector:
                    scaled[instrument_id] = weights[instrument_id] * factor
    return scaled


def _scale_gross(weights: dict[str, float], gross_cap: float) -> dict[str, float]:
    total = sum(weights.values())
    if total > gross_cap:
        factor = gross_cap / total
        return {instrument_id: weight * factor for instrument_id, weight in weights.items()}
    return dict(weights)


def _scale_volatility(
    weights: dict[str, float],
    panel: pl.DataFrame,
    ids: list[str],
    policy: StockRiskPolicy,
    *,
    covariance: np.ndarray | None = None,
) -> dict[str, float]:
    if covariance is None:
        covariance, _ = _covariance(panel, ids, policy)
    vector = np.asarray([weights[instrument_id] for instrument_id in ids], dtype=float)
    portfolio_variance = float(vector @ covariance @ vector)
    if portfolio_variance <= 0.0 or not math.isfinite(portfolio_variance):
        return dict(weights)
    forecast_vol = math.sqrt(portfolio_variance) * math.sqrt(policy.annualization_sessions)
    scalar = min(1.0, policy.target_annual_volatility / forecast_vol)
    return {instrument_id: weight * scalar for instrument_id, weight in weights.items()}


def _compounding_scale(
    weights: dict[str, float],
    net_lower_bound_of: dict[str, float],
    panel: pl.DataFrame,
    ids: list[str],
    policy: StockRiskPolicy,
    *,
    covariance: np.ndarray | None = None,
) -> tuple[float, float, float, str | None]:
    """Return ``(scale, confidence_edge_h, confidence_variance_h, cash_reason)``.

    The risky target scale ``s*`` prices the horizon-unit lower-confidence
    edge ``A_h = w.T @ net_alpha_lower_bound`` against the horizon-scaled
    portfolio variance ``V_h = h * w.T @ Sigma_daily @ w`` with the policy's
    ``growth_risk_aversion``. ``net_alpha_lower_bound`` already nets the
    calibrated route-level round-trip cost, so it is never cost-subtracted
    again. ``s*`` only reduces a constrained target, never increases it. A
    non-positive edge, a non-finite/negative variance, or unavailable
    covariance yields ``cash_reason`` and a zero risky scale.
    """
    horizon = max(1, int(policy.rebalance_frequency_sessions))
    risk_aversion = policy.compounding.growth_risk_aversion
    vector = np.asarray([weights[instrument_id] for instrument_id in ids], dtype=float)
    lower = np.asarray(
        [net_lower_bound_of.get(instrument_id, 0.0) for instrument_id in ids],
        dtype=float,
    )
    confidence_edge_h = float(vector @ lower)
    if not math.isfinite(confidence_edge_h) or confidence_edge_h <= 0.0:
        return 0.0, confidence_edge_h, 0.0, "non-positive-confidence-edge"
    if covariance is None:
        covariance, _ = _covariance(panel, ids, policy)
    confidence_variance_h = horizon * float(vector @ covariance @ vector)
    if not math.isfinite(confidence_variance_h) or confidence_variance_h < 0.0:
        return 0.0, confidence_edge_h, confidence_variance_h, "invalid-confidence-variance"
    if confidence_variance_h == 0.0:
        return 1.0, confidence_edge_h, confidence_variance_h, None
    scale = min(1.0, max(0.0, confidence_edge_h / (risk_aversion * confidence_variance_h)))
    return scale, confidence_edge_h, confidence_variance_h, None


def _record_compounding_decision(
    policy: StockRiskPolicy,
    *,
    decision_session: object,
    candidate_count: int,
    ranked_count: int,
    selected_count: int,
    confidence_edge_h: float | None,
    confidence_variance_h: float | None,
    confidence_scale: float | None,
    gross_before_compounding: float,
    gross_after_compounding: float,
    turnover_lambda: float,
    cash_reason: str | None,
    covariance_source: str = "",
) -> None:
    """Append one deterministic JSON-safe per-decision compounding record.

    The record captures the decision inputs and outcome without reading any
    future returns or labels. Numeric diagnostics that are not finite are
    stored as ``None`` and the decision is fail-closed through ``cash_reason``.
    ``covariance_source`` records whether the covariance matrix used for this
    decision came from the complete-matrix shrinkage path (``full``) or the
    conservative common-correlation fallback (``fallback``); it is ``""`` for
    decisions that never reach covariance construction. The same record is
    emitted only through the ``stocks.trading.portfolio_constructor``
    DEBUG logger; default INFO runs keep allocation values, messages, and
    behavior unchanged.
    """

    def _finite_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    if isinstance(decision_session, datetime):
        decision_session = decision_session.isoformat()
    record: dict[str, object] = {
        "decision_session": str(decision_session),
        "candidate_count": int(candidate_count),
        "ranked_count": int(ranked_count),
        "selected_count": int(selected_count),
        "confidence_edge_h": _finite_or_none(confidence_edge_h),
        "confidence_variance_h": _finite_or_none(confidence_variance_h),
        "confidence_scale": _finite_or_none(confidence_scale),
        "gross_before_compounding": float(gross_before_compounding),
        "gross_after_compounding": float(gross_after_compounding),
        "turnover_lambda": float(turnover_lambda),
        "cash_reason": cash_reason,
        "covariance_source": str(covariance_source),
    }
    policy.compounding_evidence.append(record)
    logger.debug(
        "compounding_decision session=%s record=%s",
        record["decision_session"],
        record,
    )


def _return_matrix(
    panel: pl.DataFrame,
    ids: list[str],
) -> np.ndarray | None:
    """Causal raw per-session return matrix for ``ids`` (``NaN`` where missing).

    Pivots the selected names to one row per session and returns the raw
    window without dropping incomplete sessions, so the caller can run the
    complete-matrix path or the conservative fallback on exactly the same
    input. Returns ``None`` only when no selected name matches the panel.
    """
    with_return = panel.filter(pl.col("instrument_id").is_in(ids)).with_columns(
        _returns_column(panel).alias("__logret")
    )
    pivoted = (
        with_return.select(["instrument_id", _SESSION_COLUMN, "__logret"])
        .sort([_SESSION_COLUMN, "instrument_id"])
        .pivot(
            on="instrument_id",
            index=_SESSION_COLUMN,
            values="__logret",
            aggregate_function="first",
        )
    )
    columns = [c for c in pivoted.columns if c != _SESSION_COLUMN]
    if not columns:
        return None
    arr = pivoted.select(columns).to_numpy()
    if arr.ndim != 2 or arr.shape[1] != len(ids):
        return None
    order = [columns.index(instrument_id) for instrument_id in ids]
    return np.asarray(arr[:, order], dtype=float)


def _covariance(
    panel: pl.DataFrame,
    ids: list[str],
    policy: StockRiskPolicy,
) -> tuple[np.ndarray, str]:
    """Full-or-fallback causal covariance for ``ids`` on one decision window.

    Both the prepared and reference construction paths must agree on the exact
    covariance result and its ``full``/``fallback`` source so the same matrix
    flows into scaling, compounding, and post-validation.
    """
    matrix = _return_matrix(panel, ids)
    if matrix is None:
        raise PortfolioConstraintError("insufficient covariance data")
    return causal_covariance_or_fallback(
        matrix,
        volatility_lookback_sessions=policy.volatility_lookback_sessions,
        covariance_lookback_sessions=policy.covariance_lookback_sessions,
    )


def _shrinkage_covariance(returns: np.ndarray) -> np.ndarray:
    sample = np.cov(returns, rowvar=False, ddof=0)
    n_assets = returns.shape[1]
    if n_assets == 1:
        return np.asarray([[float(np.var(returns, ddof=0))]], dtype=float)
    diagonal = np.diag(sample)
    target = np.diag(np.full(n_assets, float(np.mean(diagonal))))
    return 0.5 * sample + 0.5 * target


def causal_covariance_or_fallback(
    returns: np.ndarray,
    *,
    volatility_lookback_sessions: int,
    covariance_lookback_sessions: int,
) -> tuple[np.ndarray, str]:
    """Causal covariance with a conservative common-correlation fallback.

    ``returns`` is the causal raw return window (rows are sessions, columns are
    selected assets, ``NaN`` marks an unobserved return at that point in time);
    it must contain no future rows. When a complete trailing matrix of at least
    two sessions exists within ``covariance_lookback_sessions`` the existing
    shrinkage covariance is returned unchanged with ``source='full'``.
    Otherwise every selected asset must expose at least
    ``volatility_lookback_sessions`` finite own returns with finite positive
    variance; pair correlations are estimated only from overlapping rows and a
    PSD common-correlation matrix built from the upper quartile of the observed
    positive pair correlations (perfect correlation when no positive pair is
    observable) is combined with the diagonal own variances. A missing pair is
    never treated as zero correlation, so the fallback overstates risk when
    data is sparse instead of understating it. Any asset without sufficient
    finite volatility history fails closed as
    :class:`PortfolioConstraintError`. Returns ``(covariance, source)`` where
    ``source`` is ``'full'`` or ``'fallback'``.
    """
    if volatility_lookback_sessions < 1 or covariance_lookback_sessions < 1:
        raise ValueError("lookback sessions must be positive")
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise PortfolioConstraintError("insufficient covariance data")
    n_assets = int(matrix.shape[1])

    complete = matrix[np.all(np.isfinite(matrix), axis=1)]
    complete_tail = complete[-covariance_lookback_sessions:]
    if complete_tail.shape[0] >= 2:
        return _shrinkage_covariance(complete_tail), "full"

    variances = np.empty(n_assets, dtype=float)
    for asset in range(n_assets):
        own = matrix[:, asset]
        own = own[np.isfinite(own)]
        if own.size < volatility_lookback_sessions:
            raise PortfolioConstraintError("insufficient covariance data")
        variance = float(np.var(own, ddof=0))
        if not math.isfinite(variance) or variance <= 0.0:
            raise PortfolioConstraintError("insufficient covariance data")
        variances[asset] = variance
    observed: list[float] = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            both = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            if np.count_nonzero(both) >= 2:
                left = matrix[both, i]
                right = matrix[both, j]
                if np.std(left) > 0.0 and np.std(right) > 0.0:
                    rho = float(np.corrcoef(left, right)[0, 1])
                    if math.isfinite(rho) and rho > 0.0:
                        observed.append(rho)
    common = float(np.quantile(observed, 0.75)) if observed else 1.0
    correlation = np.full((n_assets, n_assets), common, dtype=float)
    np.fill_diagonal(correlation, 1.0)
    std = np.sqrt(variances)
    return (std[:, None] * correlation) * std[None, :], "fallback"


def _turnover_lambda(
    target: dict[str, float],
    current: dict[str, float],
    budget: float,
) -> float:
    union = set(target) | set(current)
    turnover = sum(abs(target.get(i, 0.0) - current.get(i, 0.0)) for i in union)
    if turnover <= 0.0:
        return 1.0
    return min(1.0, budget / turnover)


def _portfolio_is_feasible(
    current_weights: dict[str, float],
    sector_of: dict[str, object],
    equity: float,
    policy: StockRiskPolicy,
) -> bool:
    if not current_weights:
        return True
    if any(weight > policy.single_name_cap + _TOLERANCE for weight in current_weights.values()):
        return False
    sector_total: dict[object, float] = {}
    for instrument_id, weight in current_weights.items():
        sector = sector_of.get(instrument_id)
        sector_total[sector] = sector_total.get(sector, 0.0) + weight
    for total in sector_total.values():
        if total > policy.sector_cap + _TOLERANCE:
            return False
    if sum(current_weights.values()) > policy.gross_cap + _TOLERANCE:
        return False
    del equity
    return True


def _de_risk_allocations(
    current_weights: dict[str, float],
    sector_of: dict[str, object],
    equity: float,
    instruments: Mapping[str, Instrument],
    policy: StockRiskPolicy,
) -> tuple[Allocation, ...]:
    """Sell-only plan: reduce every over-cap position toward its cap.

    Holdings are scaled down proportionally to respect the gross cap; no name
    or sector weight is increased and no new position is introduced.
    """
    scaled = dict(current_weights)
    for instrument_id in scaled:
        scaled[instrument_id] = min(scaled[instrument_id], policy.single_name_cap)
    scaled = _scale_sectors(scaled, sector_of, policy.sector_cap)
    scaled = _scale_gross(scaled, policy.gross_cap)
    return tuple(
        Allocation(
            instrument=_instrument(instrument_id, instruments),
            target_value=max(0.0, weight) * equity,
            reason="de-risk-sell-only",
        )
        for instrument_id, weight in scaled.items()
        if weight > 0.0 and weight < current_weights[instrument_id] - _TOLERANCE
    )


def _instrument(instrument_id: str, instruments: Mapping[str, Instrument]) -> Instrument:
    return instruments[instrument_id]


def _post_validate(
    allocations: tuple[Allocation, ...],
    equity: float,
    policy: StockRiskPolicy,
    sector_of: Mapping[str, object],
    adtv_of: Mapping[str, float],
    panel: pl.DataFrame,
    *,
    covariance: np.ndarray | None = None,
    current_weights: Mapping[str, float] | None = None,
) -> None:
    total = sum(a.target_value for a in allocations)
    if total > equity * policy.gross_cap + _TOLERANCE:
        raise PortfolioConstraintError("gross target exceeds gross_cap")
    sector_totals: dict[object, float] = {}
    for allocation in allocations:
        instrument_id = allocation.instrument.instrument_id
        weight = allocation.target_value / equity
        if weight < -_TOLERANCE or weight > policy.single_name_cap + _TOLERANCE:
            raise PortfolioConstraintError(f"name cap exceeded for {instrument_id}")
        sector = sector_of.get(instrument_id)
        if sector is None:
            raise PortfolioConstraintError(f"missing sector for {instrument_id}")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight
        adtv = adtv_of.get(instrument_id)
        if adtv is None or adtv <= 0:
            raise PortfolioConstraintError(f"missing capacity for {instrument_id}")
        if policy.participation_limit > 0:
            current = (current_weights or {}).get(instrument_id, 0.0)
            delta_cap = policy.participation_limit * adtv / equity
            if abs(weight - current) > delta_cap + _TOLERANCE:
                raise PortfolioConstraintError(
                    f"capacity delta cap exceeded for {instrument_id}"
                )
        if allocation.target_value < -_TOLERANCE:
            raise PortfolioConstraintError("target value must be non-negative")
    if any(total > policy.sector_cap + _TOLERANCE for total in sector_totals.values()):
        raise PortfolioConstraintError("sector cap exceeded")
    weights = {
        a.instrument.instrument_id: a.target_value / equity
        for a in allocations
        if a.target_value > 0
    }
    if weights:
        if covariance is None:
            covariance, _ = _covariance(panel, sorted(weights), policy)
        vector = np.asarray([weights[i] for i in sorted(weights)], dtype=float)
        variance = float(vector @ covariance @ vector)
        forecast_vol = math.sqrt(max(variance, 0.0)) * math.sqrt(policy.annualization_sessions)
        if not math.isfinite(forecast_vol) or forecast_vol > policy.target_annual_volatility + _TOLERANCE:
            raise PortfolioConstraintError("portfolio volatility cap exceeded")
