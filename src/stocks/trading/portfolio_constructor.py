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

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.core.instruments import Instrument
from src.core.portfolio import Allocation, PortfolioSnapshot

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
class StockRiskPolicy:
    """Frozen, versioned risk profile for target construction.

    The seed profile is a configuration artifact frozen before holdout; live
    mode must load an explicitly identified policy artifact rather than
    silently instantiate these defaults.
    """

    top_k: int = 20
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

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
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

    ranked = (
        eligible.sort(
            [pl.col("pred_score").sort(descending=True), pl.col("instrument_id").sort()]
        )
        .with_row_index("__rank", offset=1)
        .filter(
            (pl.col("__rank") <= policy.enter_rank)
            | (
                pl.col("instrument_id").is_in(incumbent_ids)
                & (pl.col("__rank") <= policy.keep_rank)
            )
        )
        .head(policy.top_k)
        .drop("__rank")
    )
    if ranked.is_empty():
        return ()

    ids = [str(r) for r in ranked["instrument_id"].to_list()]
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
        )
    else:
        allocations = _de_risk_allocations(
            current_weights, sector_of, equity, instruments, policy
        )

    _post_validate(allocations, equity, policy, sector_of, adtv_of, panel)
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
    expected active return) and a positive ``alpha_lower_bound`` confidence
    bound. Existing holdings are retained only while the keep-versus-exit net
    benefit (``expected_active_alpha`` minus the one-way sell cost) is positive
    with a positive confidence bound. Null/non-finite alpha is never a buy
    signal: those rows fail the gate and fall through to cash or sell-only.
    """
    if "expected_active_alpha" not in cross_section.columns:
        return eligible
    incumbent = pl.col("instrument_id").is_in(incumbent_ids)
    keep_ok = (
        (pl.col("expected_active_alpha") - pl.col("exit_cost_rate") > 0.0)
        & (pl.col("expected_active_alpha") > 0.0)
        & (pl.col("alpha_lower_bound") > 0.0)
    )
    enter_ok = (
        (pl.col("expected_net_alpha") > 0.0)
        & (pl.col("expected_active_alpha") > 0.0)
        & (pl.col("alpha_lower_bound") > 0.0)
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
) -> tuple[Allocation, ...]:
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
        capacity = policy.participation_limit * adtv_of[instrument_id] / equity
        weights[instrument_id] = min(raw, policy.single_name_cap, capacity)

    weights = _scale_sectors(weights, sector_of, policy.sector_cap)
    weights = _scale_gross(weights, policy.gross_cap)
    weights = _scale_volatility(weights, panel, ids, policy)

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
) -> dict[str, float]:
    return_matrix = _return_matrix(panel, ids, policy.covariance_lookback_sessions)
    if return_matrix is None:
        raise PortfolioConstraintError("insufficient covariance data")
    vector = np.asarray([weights[instrument_id] for instrument_id in ids], dtype=float)
    covariance = _shrinkage_covariance(return_matrix)
    portfolio_variance = float(vector @ covariance @ vector)
    if portfolio_variance <= 0.0 or not math.isfinite(portfolio_variance):
        return dict(weights)
    forecast_vol = math.sqrt(portfolio_variance) * math.sqrt(policy.annualization_sessions)
    scalar = min(1.0, policy.target_annual_volatility / forecast_vol)
    return {instrument_id: weight * scalar for instrument_id, weight in weights.items()}


def _return_matrix(
    panel: pl.DataFrame,
    ids: list[str],
    lookback: int,
) -> np.ndarray | None:
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
    pivoted = pivoted.drop_nulls()
    columns = [c for c in pivoted.columns if c != _SESSION_COLUMN]
    arr = pivoted.tail(lookback).select(columns).to_numpy()
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != len(ids):
        return None
    order = [columns.index(instrument_id) for instrument_id in ids]
    return np.asarray(arr[:, order], dtype=float)


def _shrinkage_covariance(returns: np.ndarray) -> np.ndarray:
    sample = np.cov(returns, rowvar=False, ddof=0)
    n_assets = returns.shape[1]
    if n_assets == 1:
        return np.asarray([[float(np.var(returns, ddof=0))]], dtype=float)
    diagonal = np.diag(sample)
    target = np.diag(np.full(n_assets, float(np.mean(diagonal))))
    return 0.5 * sample + 0.5 * target


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
        if policy.participation_limit > 0 and weight > policy.participation_limit * adtv / equity + _TOLERANCE:
            raise PortfolioConstraintError(f"capacity cap exceeded for {instrument_id}")
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
        matrix = _return_matrix(panel, sorted(weights), policy.covariance_lookback_sessions)
        if matrix is None:
            raise PortfolioConstraintError("insufficient covariance data")
        covariance = _shrinkage_covariance(matrix)
        vector = np.asarray([weights[i] for i in sorted(weights)], dtype=float)
        variance = float(vector @ covariance @ vector)
        forecast_vol = math.sqrt(max(variance, 0.0)) * math.sqrt(policy.annualization_sessions)
        if not math.isfinite(forecast_vol) or forecast_vol > policy.target_annual_volatility + _TOLERANCE:
            raise PortfolioConstraintError("portfolio volatility cap exceeded")
