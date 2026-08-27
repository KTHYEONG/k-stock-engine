"""Ranking-quality and economic evaluation metrics.

Evaluation records ranking quality and economic metrics separately, including
costs, turnover, exposure, drawdown, and coverage; it never chooses a model
solely by NDCG or one backtest metric. Forward-holdout compound growth is
certified only geometrically, over complete observed cohorts, and with a
seeded vectorized moving-block bootstrap lower bound.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.stocks.ml.contracts import CompoundingCertificationSettings
from src.stocks.ml.hedge_sleeve import project_hedge_sleeve

if TYPE_CHECKING:
    from src.stocks.ml.horizons import GrowthRouteEvidence


def ndcg_at_k(scores: pl.Series, labels: pl.Series, k: int | None = None) -> float:
    """Normalized discounted cumulative gain at ``k``."""
    s = scores.to_numpy()
    rel = labels.to_numpy()
    if len(s) == 0:
        return 0.0
    order = np.argsort(s)[::-1]
    rel_sorted = rel[order]
    if k is not None:
        rel_sorted = rel_sorted[:k]
    if len(rel_sorted) == 0:
        return 0.0
    dcg = float(np.sum((2.0**rel_sorted - 1.0) / np.log2(np.arange(2, len(rel_sorted) + 2))))
    best = float(np.sum((2.0 ** np.sort(rel)[::-1] - 1.0) / np.log2(np.arange(2, len(rel) + 2))))
    return dcg / best if best > 0 else 0.0


def rank_ic(scores: pl.Series, labels: pl.Series) -> float:
    """Spearman rank correlation between scores and labels."""
    s = scores.to_numpy()
    rel = labels.to_numpy()
    if len(s) < 2 or np.std(s) == 0 or np.std(rel) == 0:
        return 0.0
    rs = rankdata(s)
    rr = rankdata(rel)
    return float(np.corrcoef(rs, rr)[0, 1])


def coverage(rows_with_label: int, rows_total: int) -> float:
    if rows_total == 0:
        return 0.0
    return rows_with_label / rows_total


def max_drawdown(equity_curve: list[float] | np.ndarray) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq) / np.where(peaks > 0, peaks, 1.0)
    return float(np.max(dd)) if dd.size else 0.0


def compounded_growth_metrics(
    returns: Sequence[float],
    annualization_sessions: int,
) -> dict[str, float]:
    """Annualized compound-growth and drawdown metrics for one return series.

    Only finite realized returns strictly greater than ``-1`` are accepted;
    an empty, non-finite, or ``<= -1`` series is evidence-incomplete and is
    never zero-filled, so ``evidence_complete`` is ``0.0`` and the metrics are
    reported as zero. Otherwise ``cagr`` annualizes the exact geometric-mean
    log growth over the observed return-interval count, ``mdd`` is the peak
    drawdown of the cumulative equity curve, and ``calmar`` is ``cagr / mdd``
    with ``mdd == 0`` mapping to ``+inf`` for a positive CAGR and ``0``
    otherwise. The result is deterministic and JSON-safe for finite inputs.
    """
    if annualization_sessions <= 0:
        raise ValueError("annualization_sessions must be positive")
    arr = np.asarray(returns, dtype=float)
    if (
        arr.size == 0
        or not bool(np.all(np.isfinite(arr)))
        or bool(np.any(arr <= -1.0))
    ):
        return {
            "evidence_complete": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
        }
    log_growth = float(np.sum(np.log1p(arr)))
    cagr = float(
        math.expm1(log_growth * annualization_sessions / arr.size)
    )
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / np.where(peaks > 0, peaks, 1.0)))
    calmar = (
        (float("inf") if cagr > 0.0 else 0.0)
        if mdd == 0.0
        else cagr / mdd
    )
    return {
        "evidence_complete": 1.0,
        "cagr": cagr,
        "mdd": mdd,
        "calmar": calmar,
    }


def economic_transfer_attribution(
    scored: pl.DataFrame,
    label_column: str,
    top_k: int,
) -> dict[str, float | int]:
    """Cross-sectional ranking-to-selected-tail attribution per retained session.

    For every retained session the score-to-label ordering is measured with the
    cross-sectional Spearman Rank-IC, and the label mean of the top ``top_k``
    names is compared against the session universe mean so the report can see
    whether the ordering concentrates in the ownable tail before any costs or
    allocation. Top-k membership turnover versus the preceding retained session
    quantifies how much of the tail is rebuilt each decision (``1.0`` for the
    first retained session). Rows with a missing label are excluded, never
    zero-filled, and a non-finite score or label entering a retained session
    raises ``ValueError``. An empty valid frame returns zero-valued aggregates.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    required = ("session", "instrument_id", "pred_score", label_column)
    missing = [column for column in required if column not in scored.columns]
    if missing:
        raise ValueError(
            "economic transfer attribution requires " + ", ".join(missing)
        )
    valid = scored.filter(
        pl.col("pred_score").is_not_null() & pl.col(label_column).is_not_null()
    )
    non_finite = valid.filter(
        ~pl.col("pred_score").is_finite() | ~pl.col(label_column).is_finite()
    )
    if not non_finite.is_empty():
        raise ValueError("non-finite score or label enters a retained session")
    if valid.is_empty():
        return {
            "decision_count": 0,
            "retained_session_count": 0,
            "session_coverage": 0.0,
            "positive_rank_ic_session_count": 0,
            "mean_rank_ic": 0.0,
            "mean_top_k_label": 0.0,
            "mean_universe_label": 0.0,
            "mean_top_k_active_label": 0.0,
            "mean_membership_turnover": 0.0,
        }
    total_sessions = int(
        scored.select(pl.col("session").n_unique()).to_series()[0]
    )
    rank_ics: list[float] = []
    top_k_labels: list[float] = []
    universe_labels: list[float] = []
    active_labels: list[float] = []
    turnovers: list[float] = []
    previous_members: set[str] = set()
    previous_size = 0
    for cross in valid.sort("session").partition_by("session"):
        k = min(top_k, int(cross.height))
        top = cross.sort("pred_score", descending=True).head(k)
        members = set(top["instrument_id"].to_list())
        turnover = (
            1.0
            - len(previous_members & members) / max(previous_size, len(members))
            if previous_size
            else 1.0
        )
        universe = float(np.mean(cross[label_column].to_numpy()))
        top_mean = float(np.mean(top[label_column].to_numpy()))
        rank_ics.append(rank_ic(cross["pred_score"], cross[label_column]))
        universe_labels.append(universe)
        top_k_labels.append(top_mean)
        active_labels.append(top_mean - universe)
        turnovers.append(turnover)
        previous_members, previous_size = members, len(members)
    retained = len(rank_ics)
    return {
        "decision_count": retained,
        "retained_session_count": retained,
        "session_coverage": retained / max(total_sessions, 1),
        "positive_rank_ic_session_count": sum(1 for value in rank_ics if value > 0.0),
        "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "mean_top_k_label": float(np.mean(top_k_labels)) if top_k_labels else 0.0,
        "mean_universe_label": float(np.mean(universe_labels)) if universe_labels else 0.0,
        "mean_top_k_active_label": float(np.mean(active_labels)) if active_labels else 0.0,
        "mean_membership_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }

def _bootstrap_lower_mean_log_growth(
    log_growth: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> float:
    """Seeded vectorized moving-block bootstrap alpha-quantile of period means."""
    arr = np.asarray(log_growth, dtype=float)
    n = arr.size
    block = min(max(block_length, 1), n)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block)
    index = (
        starts[:, :, None] + offsets[None, None, :]
    ).reshape(n_bootstrap, n_blocks * block)[:, :n]
    means = arr[index].mean(axis=1)
    return float(np.quantile(means, alpha))


def _round_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return round(float(value), 12)


def annualize_bootstrap_lower_cagr(
    lower_mean: float,
    *,
    annualization_sessions: int,
    period_count: int,
    observed_sessions: int,
) -> float:
    """Annualized lower-bound CAGR from a bootstrap mean log growth.

    The annualization factor is ``annualization_sessions * period_count /
    observed_sessions`` so that daily-ledger returns are correctly scaled
    regardless of the horizon block length.
    """
    if observed_sessions <= 0:
        return 0.0
    if period_count <= 0:
        return 0.0
    if not math.isfinite(lower_mean):
        return 0.0
    factor = annualization_sessions * period_count / observed_sessions
    return float(math.expm1(lower_mean * factor))


def _certify_path(
    period_returns: Sequence[float],
    horizon_sessions: int,
    observed_sessions: int,
    active_cohort_count: int,
    settings: CompoundingCertificationSettings,
) -> dict[str, object]:
    """Compute one base/stress path's geometric certificate and fail-closed gates.

    The arithmetic period returns are transformed with ``log1p`` and never
    zero-filled; a non-finite or ``<= -1`` period series, an insufficient
    observed-session window, zero/insufficient active coverage, a non-positive
    CAGR or bootstrap lower CAGR, or a drawdown beyond the immutable policy all
    fail closed with normalized reasons while still reporting the aggregate
    values for diagnosis.
    """
    arr = np.asarray(list(period_returns), dtype=float)
    reasons: list[str] = []
    if (
        arr.size == 0
        or not bool(np.all(np.isfinite(arr)))
        or bool(np.any(arr <= -1.0))
    ):
        return {
            "passed": False,
            "reasons": ["period-series-incomplete"],
            "period_count": int(arr.size),
            "observed_sessions": int(observed_sessions),
            "active_cohort_count": int(active_cohort_count),
            "cagr": 0.0,
            "lower_cagr": 0.0,
            "mdd": 0.0,
            "calmar": None,
        }
    if observed_sessions <= 0:
        return {
            "passed": False,
            "reasons": ["invalid-observed-sessions"],
            "period_count": int(arr.size),
            "observed_sessions": int(observed_sessions),
            "active_cohort_count": int(active_cohort_count),
            "cagr": 0.0,
            "lower_cagr": 0.0,
            "mdd": 0.0,
            "calmar": None,
        }
    period_count = int(arr.size)
    annualization = settings.annualization_sessions
    log_growth = np.log1p(arr)
    total_log = float(np.sum(log_growth))
    cagr = float(np.expm1(total_log * annualization / observed_sessions))
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / np.where(peaks > 0, peaks, 1.0)))
    calmar = (float("inf") if cagr > 0.0 else 0.0) if mdd == 0.0 else cagr / mdd
    lower_mean = _bootstrap_lower_mean_log_growth(
        log_growth,
        horizon_sessions,
        settings.bootstrap_resamples,
        settings.seed,
        settings.bootstrap_alpha,
    )
    lower_cagr = annualize_bootstrap_lower_cagr(
        lower_mean,
        annualization_sessions=annualization,
        period_count=period_count,
        observed_sessions=observed_sessions,
    )
    effective_min_observed = settings.min_observed_sessions
    if settings.allowed_tail_censoring_sessions > 0:
        # Structurally censored tail sessions (vintages maturing past the
        # dataset end) lower the requirement; genuine shortfalls below the
        # 95% floor of the reduced requirement still fail closed.
        effective_min_observed = math.ceil(
            (settings.min_observed_sessions - settings.allowed_tail_censoring_sessions)
            * 0.95
        )
    if observed_sessions < effective_min_observed:
        reasons.append("insufficient-observed-sessions")
    active_fraction = (
        active_cohort_count / period_count if period_count > 0 else 0.0
    )
    if (
        active_cohort_count <= 0
        or active_fraction < settings.min_active_cohort_fraction
    ):
        reasons.append("active-coverage-insufficient")
    if not math.isfinite(cagr) or cagr <= 0.0:
        reasons.append("non-positive-cagr")
    if not math.isfinite(lower_cagr) or lower_cagr <= 0.0:
        reasons.append("non-positive-lower-cagr")
    if mdd > settings.max_drawdown:
        reasons.append("max-drawdown-exceeded")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "period_count": period_count,
        "observed_sessions": int(observed_sessions),
        "active_cohort_count": int(active_cohort_count),
        "cagr": _round_or_none(cagr),
        "lower_cagr": _round_or_none(lower_cagr),
        "mdd": _round_or_none(mdd),
        "calmar": _round_or_none(calmar),
    }


@dataclass(frozen=True, slots=True)
class CompoundingCertificationEvidence:
    """Immutable compound-growth certificate for one untouched forward holdout.

    ``base`` and ``stress`` are bounded per-path summaries (counts, CAGR,
    bootstrap lower CAGR, MDD, Calmar, pass flags and normalized reasons). The
    certificate holds only aggregate values and never score, label, or return
    vectors.
    """

    passed: bool
    reasons: tuple[str, ...]
    base: dict[str, object]
    stress: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "base": dict(self.base),
            "stress": dict(self.stress),
        }


def certify_compounded_holdout(
    base_period_returns: Sequence[float],
    stress_period_returns: Sequence[float],
    horizon_sessions: int,
    observed_sessions: int,
    active_cohort_count: int,
    settings: CompoundingCertificationSettings,
) -> CompoundingCertificationEvidence:
    """Certify the untouched forward holdout under base and stress cost paths.

    Both paths consume the identical frozen scores, sessions, portfolio
    constraints and realized liquidity rows; only the effective cost/liquidity
    schedule changes. The certificate passes only when every quantitative gate
    passes on both paths, and is deterministic for fixed inputs.
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    if observed_sessions < 0:
        raise ValueError("observed_sessions must be non-negative")
    if active_cohort_count < 0:
        raise ValueError("active_cohort_count must be non-negative")
    base = _certify_path(
        base_period_returns, horizon_sessions, observed_sessions,
        active_cohort_count, settings,
    )
    stress = _certify_path(
        stress_period_returns, horizon_sessions, observed_sessions,
        active_cohort_count, settings,
    )
    passed = bool(base["passed"]) and bool(stress["passed"])
    base_reasons = base["reasons"]
    stress_reasons = stress["reasons"]
    combined = (
        list(base_reasons) if isinstance(base_reasons, (list, tuple)) else []
    ) + (
        list(stress_reasons) if isinstance(stress_reasons, (list, tuple)) else []
    )
    reasons = list(dict.fromkeys(combined))
    return CompoundingCertificationEvidence(
        passed=passed,
        reasons=tuple(reasons),
        base=base,
        stress=stress,
    )


@dataclass(frozen=True, slots=True)
class CompoundingCertificate:
    """Immutable compound-growth certificate with absolute and relative gates.

    ``passed`` is True only when every absolute base/stress gate and the
    exposure-matched lower excess CAGR > 0 both pass.  ``promoted`` requires
    ``passed`` and a positive matched lower excess; an absolute pass with
    non-positive matched lower excess yields
    ``RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN`` and ``promoted=False``.
    """

    passed: bool
    reasons: tuple[str, ...]
    promoted: bool
    absolute_base: dict[str, object]
    absolute_stress: dict[str, object]
    matched_excess_lower_cagr: float

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe representation for persisted evaluation evidence."""
        return {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "promoted": bool(self.promoted),
            "absolute_base": self.absolute_base,
            "absolute_stress": self.absolute_stress,
            "matched_excess_lower_cagr": float(self.matched_excess_lower_cagr),
        }


_RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN = "RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN"  # noqa: S105


def certify_exposure_matched_excess(
    strategy_log_growth: Sequence[float],
    benchmark_log_growth: Sequence[float],
    horizon_sessions: int,
    active_cohort_count: int,
    settings: CompoundingCertificationSettings,
) -> CompoundingCertificate:
    """Certify absolute and exposure-matched relative growth evidence.

    The absolute base/stress gates are evaluated via
    ``certify_compounded_holdout``.  The matched lower excess CAGR is the
    bootstrap lower bound of the per-interval strategy-minus-benchmark log
    growth, annualized over the observed sessions.  Promotion requires both
    absolute gates and a positive matched lower excess CAGR.  An absolute
    pass with non-positive matched lower excess returns
    ``RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN`` and no artifact promotion.
    """
    arr_strategy = np.asarray(list(strategy_log_growth), dtype=float)
    arr_benchmark = np.asarray(list(benchmark_log_growth), dtype=float)
    if arr_strategy.size == 0 or arr_benchmark.size == 0:
        return CompoundingCertificate(
            passed=False,
            reasons=("empty-growth-series",),
            promoted=False,
            absolute_base={},
            absolute_stress={},
            matched_excess_lower_cagr=0.0,
        )
    n = min(arr_strategy.size, arr_benchmark.size)
    arr_strategy = arr_strategy[:n]
    arr_benchmark = arr_benchmark[:n]
    finite_mask = np.isfinite(arr_strategy) & np.isfinite(arr_benchmark)
    arr_strategy = arr_strategy[finite_mask]
    arr_benchmark = arr_benchmark[finite_mask]
    if arr_strategy.size == 0:
        return CompoundingCertificate(
            passed=False,
            reasons=("no-finite-growth-values",),
            promoted=False,
            absolute_base={},
            absolute_stress={},
            matched_excess_lower_cagr=0.0,
        )

    absolute = certify_compounded_holdout(
        tuple(np.expm1(arr_strategy.tolist())),
        tuple(np.expm1(arr_strategy.tolist())),
        horizon_sessions, int(arr_strategy.size),
        active_cohort_count, settings,
    )
    stress_absolute = certify_compounded_holdout(
        tuple(np.expm1(arr_strategy.tolist())),
        tuple(np.expm1(arr_strategy.tolist())),
        horizon_sessions, int(arr_strategy.size),
        active_cohort_count, settings,
    )

    excess = arr_strategy - arr_benchmark
    if excess.size < 2:
        matched_lower_cagr = 0.0
    else:
        block = max(1, horizon_sessions)
        n_blocks = math.ceil(excess.size / block)
        if n_blocks < 2:
            matched_lower_cagr = 0.0
        else:
            max_start = max(1, excess.size - block + 1)
            rng = np.random.default_rng(settings.seed)
            starts = rng.integers(0, max_start, size=(settings.bootstrap_resamples, n_blocks))
            offsets = np.arange(block)
            index = (
                starts[:, :, None] + offsets[None, None, :]
            ).reshape(settings.bootstrap_resamples, n_blocks * block)[:, :excess.size]
            boot_means = excess[index].mean(axis=1)
            lower_mean = float(np.quantile(boot_means, settings.bootstrap_alpha))
            factor = settings.annualization_sessions * excess.size / excess.size
            matched_lower_cagr = float(math.expm1(lower_mean * factor))

    abs_passed = bool(absolute.passed) and bool(stress_absolute.passed)
    abs_reasons = list(absolute.reasons) + list(stress_absolute.reasons)
    matched_positive = matched_lower_cagr > 0.0
    reasons: list[str] = list(dict.fromkeys(abs_reasons))
    if not matched_positive:
        reasons.append(_RESEARCH_ABSOLUTE_PASS_RELATIVE_UNPROVEN)
    passed = abs_passed and matched_positive
    promoted = passed
    return CompoundingCertificate(
        passed=passed,
        reasons=tuple(reasons),
        promoted=promoted,
        absolute_base=absolute.base,
        absolute_stress=stress_absolute.stress,
        matched_excess_lower_cagr=round(matched_lower_cagr, 12),
    )


def certify_growth_route(
    route: GrowthRouteEvidence,
    horizon_sessions: int,
    settings: CompoundingCertificationSettings,
    *,
    minimum_lower_cagr: float = 0.0,
    max_drawdown: float | None = None,
) -> dict[str, object]:
    """Certify a stitched growth route under absolute and relative growth gates.

    The route is one strategy-level hypothesis: its base/stress per-interval
    log-growth series are bootstrapped with the shared moving-block kernel at
    ``settings.bootstrap_alpha`` and annualized over the observed intervals.
    The certificate passes only when the base and stress compound lower CAGR
    are both strictly positive, the exposure-matched lower excess CAGR over
    the parallel benchmark series is strictly positive, observed sessions and
    invested coverage satisfy ``settings``, MDD is within the cap, and at
    least one order filled. Sparse-minus-dense and turnover diagnostics never
    enter the decision. Every failure emits a normalized predicate while still
    reporting the aggregate values for diagnosis.

    Raises:
        ValueError: when ``bootstrap_resamples`` cannot resolve the alpha
            quantile (``resamples < ceil(1 / alpha)``).
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    minimum_resamples = math.ceil(1.0 / settings.bootstrap_alpha)
    if settings.bootstrap_resamples < minimum_resamples:
        raise ValueError(
            f"bootstrap_resamples={settings.bootstrap_resamples} is below the "
            f"resolvable minimum {minimum_resamples} for "
            f"alpha={settings.bootstrap_alpha}"
        )

    base_log = np.asarray(route.base_log_growth, dtype=float)
    stress_log = np.asarray(route.stress_log_growth, dtype=float)
    benchmark_log = np.asarray(route.benchmark_log_growth, dtype=float)
    observed = int(
        route.observed_interval_count
        if route.observed_interval_count > 0
        else base_log.size
    )
    invested = int(route.invested_interval_count)

    def _result(reasons: list[str], **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "passed": not reasons,
            "reasons": sorted(dict.fromkeys(reasons)),
            "cagr_base": None,
            "cagr_stress": None,
            "base_lower_cagr": None,
            "stress_lower_cagr": None,
            "matched_lower_excess_cagr": None,
            "mdd": None,
            "observed_intervals": observed,
            "invested_intervals": invested,
            "filled_orders": int(route.filled_orders),
        }
        payload.update(overrides)
        return payload

    if (
        base_log.size == 0
        or not bool(np.all(np.isfinite(base_log)))
        or not bool(np.all(np.isfinite(stress_log)))
        or stress_log.size != base_log.size
    ):
        return _result(["period-series-incomplete"])
    reasons = ["no-filled-orders"] if route.filled_orders <= 0 else []

    equity = np.exp(np.cumsum(base_log))
    peaks = np.maximum.accumulate(equity)
    mdd = float(np.max(1.0 - equity / np.where(peaks > 0, peaks, 1.0)))
    cagr_base = float(np.expm1(float(np.sum(base_log)) * settings.annualization_sessions / observed))
    cagr_stress = float(
        np.expm1(float(np.sum(stress_log)) * settings.annualization_sessions / observed)
    )
    block_length = max(1, min(horizon_sessions, base_log.size))
    base_lower_mean = _bootstrap_lower_mean_log_growth(
        base_log,
        block_length,
        settings.bootstrap_resamples,
        settings.seed,
        settings.bootstrap_alpha,
    )
    stress_lower_mean = _bootstrap_lower_mean_log_growth(
        stress_log,
        block_length,
        settings.bootstrap_resamples,
        settings.seed + horizon_sessions,
        settings.bootstrap_alpha,
    )
    base_lower_cagr = annualize_bootstrap_lower_cagr(
        base_lower_mean,
        annualization_sessions=settings.annualization_sessions,
        period_count=int(base_log.size),
        observed_sessions=observed,
    )
    stress_lower_cagr = annualize_bootstrap_lower_cagr(
        stress_lower_mean,
        annualization_sessions=settings.annualization_sessions,
        period_count=int(base_log.size),
        observed_sessions=observed,
    )

    if observed < settings.min_observed_sessions:
        reasons.append("insufficient-observed-sessions")
    invested_fraction = invested / observed if observed > 0 else 0.0
    if invested <= 0 or invested_fraction < settings.min_active_cohort_fraction:
        reasons.append("invested-coverage-insufficient")
    if not math.isfinite(base_lower_cagr) or base_lower_cagr <= 0.0:
        reasons.append("non-positive-base-lower-cagr")
    if not math.isfinite(stress_lower_cagr) or stress_lower_cagr <= 0.0:
        reasons.append("non-positive-stress-lower-cagr")
    effective_max_drawdown = max_drawdown if max_drawdown is not None else settings.max_drawdown
    if mdd > effective_max_drawdown:
        reasons.append("max-drawdown-exceeded")
    if math.isfinite(minimum_lower_cagr) and minimum_lower_cagr > 0.0:
        if not math.isfinite(base_lower_cagr) or base_lower_cagr < minimum_lower_cagr:
            reasons.append("base-lower-cagr-below-target")
        if not math.isfinite(stress_lower_cagr) or stress_lower_cagr < minimum_lower_cagr:
            reasons.append("stress-lower-cagr-below-target")

    matched_lower_excess: float | None = None
    if benchmark_log.size != base_log.size or benchmark_log.size == 0:
        reasons.append("matched-benchmark-missing")
    elif not bool(np.all(np.isfinite(benchmark_log))):
        reasons.append("period-series-incomplete")
    else:
        excess = base_log - benchmark_log
        excess_lower_mean = _bootstrap_lower_mean_log_growth(
            excess,
            block_length,
            settings.bootstrap_resamples,
            settings.seed + 2 * horizon_sessions,
            settings.bootstrap_alpha,
        )
        matched_lower_excess = annualize_bootstrap_lower_cagr(
            excess_lower_mean,
            annualization_sessions=settings.annualization_sessions,
            period_count=int(excess.size),
            observed_sessions=observed,
        )
        if not math.isfinite(matched_lower_excess) or matched_lower_excess <= 0.0:
            reasons.append("non-positive-matched-lower-excess")

    return _result(
        reasons,
        cagr_base=_round_or_none(cagr_base),
        cagr_stress=_round_or_none(cagr_stress),
        base_lower_cagr=_round_or_none(base_lower_cagr),
        stress_lower_cagr=_round_or_none(stress_lower_cagr),
        matched_lower_excess_cagr=(
            None if matched_lower_excess is None else _round_or_none(matched_lower_excess)
        ),
        mdd=_round_or_none(mdd),
    )


def certify_hedged_excess_route(
    route: GrowthRouteEvidence,
    horizon_sessions: int,
    settings: CompoundingCertificationSettings,
) -> dict[str, object]:
    """Certify the exposure-matched excess stream for hedge-sleeve promotion.

    The excess series (``base - benchmark``) is bootstrapped with the shared
    moving-block kernel; the certificate passes only when the annualized
    excess lower CAGR is strictly positive, a leverage rung from
    :func:`project_hedge_sleeve` is admissible under the pre-registered
    ``settings.max_drawdown`` cap, and the variance-drag-adjusted growth of
    that rung evaluated **on the lower-bound scenario** stays strictly
    positive. Point estimates never gate. Every failure emits a normalized
    predicate while still reporting the bounded scalars for diagnosis.

    Raises:
        ValueError: when ``bootstrap_resamples`` cannot resolve the alpha
            quantile (``resamples < ceil(1 / alpha)``).
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    minimum_resamples = math.ceil(1.0 / settings.bootstrap_alpha)
    if settings.bootstrap_resamples < minimum_resamples:
        raise ValueError(
            f"bootstrap_resamples={settings.bootstrap_resamples} is below the "
            f"resolvable minimum {minimum_resamples} for "
            f"alpha={settings.bootstrap_alpha}"
        )

    base_log = np.asarray(route.base_log_growth, dtype=float)
    stress_log = np.asarray(route.stress_log_growth, dtype=float)
    benchmark_log = np.asarray(route.benchmark_log_growth, dtype=float)
    observed = int(
        route.observed_interval_count
        if route.observed_interval_count > 0
        else base_log.size
    )
    invested = int(route.invested_interval_count)

    def _result(reasons: list[str], **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "passed": not reasons,
            "reasons": sorted(dict.fromkeys(reasons)),
            "excess_lower_cagr": None,
            "sleeve_lower_stress_cagr": None,
            "hedge_variant": "",
            "hedge_leverage": None,
            "hedge_point_cagr": None,
            "hedge_stress_cagr": None,
            "hedge_projected_mdd": None,
            "hedge_margin_buffer": None,
            "observed_intervals": observed,
            "invested_intervals": invested,
            "filled_orders": int(route.filled_orders),
        }
        payload.update(overrides)
        return payload

    if (
        base_log.size == 0
        or not bool(np.all(np.isfinite(base_log)))
        or not bool(np.all(np.isfinite(stress_log)))
        or stress_log.size != base_log.size
    ):
        return _result(["period-series-incomplete"])
    reasons = ["no-filled-orders"] if route.filled_orders <= 0 else []
    if observed < settings.min_observed_sessions:
        reasons.append("insufficient-observed-sessions")
    invested_fraction = invested / observed if observed > 0 else 0.0
    if invested <= 0 or invested_fraction < settings.min_active_cohort_fraction:
        reasons.append("invested-coverage-insufficient")
    if benchmark_log.size != base_log.size or benchmark_log.size == 0:
        reasons.append("matched-benchmark-missing")
        return _result(reasons)
    if not bool(np.all(np.isfinite(benchmark_log))):
        reasons.append("period-series-incomplete")
        return _result(reasons)

    excess = base_log - benchmark_log
    block_length = max(1, min(horizon_sessions, int(excess.size)))
    excess_lower_mean = _bootstrap_lower_mean_log_growth(
        excess,
        block_length,
        settings.bootstrap_resamples,
        settings.seed + 3 * horizon_sessions,
        settings.bootstrap_alpha,
    )
    excess_lower_cagr = annualize_bootstrap_lower_cagr(
        excess_lower_mean,
        annualization_sessions=settings.annualization_sessions,
        period_count=int(excess.size),
        observed_sessions=observed,
    )
    if not math.isfinite(excess_lower_cagr) or excess_lower_cagr <= 0.0:
        reasons.append("non-positive-excess-lower-cagr")

    projection = project_hedge_sleeve(
        excess.tolist(),
        leverage_grid=(
            settings.hedge_leverage_grid
            if settings.hedge_leverage_grid is not None
            else (1.0, 1.5, 2.0)
        ),
        annualization_sessions=settings.annualization_sessions,
        max_projected_mdd=settings.max_drawdown,
        vol_managed_lookback=26,
        vol_managed_target_annualized_vol=0.10,
    )
    def _scalar(row: dict[str, object], key: str) -> float:
        return float(cast("float", row[key]))

    ladder_rows = [
        row
        for row in cast("list[dict[str, object]]", projection["leverage_ladder"])
        if bool(row.get("admissible"))
    ]
    if not ladder_rows:
        reasons.append("no-admissible-hedge-rung")
        return _result(
            reasons, excess_lower_cagr=_round_or_none(excess_lower_cagr)
        )

    variant_rank = {"vol_managed": 0, "static": 1}
    best = min(
        ladder_rows,
        key=lambda row: (
            -_scalar(row, "stress_cagr"),
            variant_rank[str(row["variant"])],
            _scalar(row, "leverage"),
        ),
    )
    leverage = _scalar(best, "leverage")
    projected_vol = _scalar(best, "projected_vol")
    lower_stress_log = (
        leverage * excess_lower_mean * settings.annualization_sessions
        - 0.5 * projected_vol**2
    )
    sleeve_lower_stress_cagr = math.expm1(
        max(min(lower_stress_log, 50.0), -50.0)
    )
    if (
        not math.isfinite(sleeve_lower_stress_cagr)
        or sleeve_lower_stress_cagr <= 0.0
    ):
        reasons.append("non-positive-sleeve-lower-stress-cagr")

    return _result(
        reasons,
        excess_lower_cagr=_round_or_none(excess_lower_cagr),
        sleeve_lower_stress_cagr=_round_or_none(sleeve_lower_stress_cagr),
        hedge_variant=str(best["variant"]),
        hedge_leverage=_round_or_none(leverage),
        hedge_point_cagr=_round_or_none(_scalar(best, "point_cagr")),
        hedge_stress_cagr=_round_or_none(_scalar(best, "stress_cagr")),
        hedge_projected_mdd=_round_or_none(_scalar(best, "projected_mdd")),
        hedge_margin_buffer=_round_or_none(_scalar(best, "margin_buffer")),
    )
