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

import numpy as np
import polars as pl
from scipy.stats import rankdata

from src.stocks.ml.contracts import CompoundingCertificationSettings


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
    period_count = int(arr.size)
    annualization = settings.annualization_sessions
    log_growth = np.log1p(arr)
    total_log = float(np.sum(log_growth))
    cagr = (
        float(np.expm1(total_log * annualization / observed_sessions))
        if observed_sessions > 0
        else 0.0
    )
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
    lower_cagr = float(
        np.expm1(lower_mean * annualization / horizon_sessions)
    )
    if observed_sessions < settings.min_observed_sessions:
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
