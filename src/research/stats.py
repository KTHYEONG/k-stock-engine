"""Multiplicity statistics over active daily log returns."""

# ruff: noqa: RUF002
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = [
    "EULER_GAMMA",
    "BootstrapProfile",
    "PointMetrics",
    "annualized_log_growth",
    "block_bootstrap_annualized_means",
    "bootstrap_profile",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_dispersion",
    "effective_trial_count",
    "max_drawdown",
    "point_metrics",
]

EULER_GAMMA: Final = 0.5772156649015329


def annualized_log_growth(log_returns: NDArray[np.float64], *, sessions_per_year: int) -> float:
    """Mean daily log return scaled to an annual horizon."""
    x = np.asarray(log_returns, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("log_returns must be a non-empty 1-D array")
    return float(np.mean(x) * sessions_per_year)


def max_drawdown(log_returns: NDArray[np.float64]) -> float:
    """Minimum peak-to-trough decline of the wealth path ``W = exp(cumsum)``."""
    x = np.asarray(log_returns, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("log_returns must be a non-empty 1-D array")
    wealth = np.concatenate(([1.0], np.exp(np.cumsum(x))))
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(np.min(drawdown))


def block_bootstrap_annualized_means(
    x: NDArray[np.float64],
    *,
    block: int,
    draws: int,
    seed: int,
    sessions_per_year: int,
    horizon: int | None = None,
) -> NDArray[np.float64]:
    """Circular-free block bootstrap of annualized means with a seeded RNG."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    if block <= 0 or block > values.size:
        raise ValueError("block must satisfy 0 < block <= len(x)")
    if draws <= 0:
        raise ValueError("draws must be positive")
    target = values.size if horizon is None else horizon
    if target <= 0:
        raise ValueError("horizon must be positive")
    rng = np.random.default_rng(seed)
    upper = values.size - block
    out = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        pieces: list[NDArray[np.float64]] = []
        filled = 0
        while filled < target:
            start = int(rng.integers(0, upper + 1))
            chunk = values[start : start + block][: target - filled]
            pieces.append(chunk)
            filled += chunk.size
        out[i] = float(np.mean(np.concatenate(pieces)) * sessions_per_year)
    return out


def effective_trial_count(returns: NDArray[np.float64]) -> float:
    """Effective independent trial count from the column correlation spectrum."""
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2-D array")
    n_rows, n_cols = matrix.shape
    if n_cols < 1 or n_rows < 2:
        raise ValueError("returns must have at least 2 rows and 1 column")
    stds = np.std(matrix, axis=0, ddof=1)
    if np.any(stds == 0.0):
        raise ValueError("returns columns must have positive variance")
    centered = matrix - np.mean(matrix, axis=0)
    corr = np.corrcoef(centered, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64).reshape(n_cols, n_cols)
    eigenvalues = np.linalg.eigvalsh(corr)
    total = float(np.sum(eigenvalues))
    squares = float(np.sum(eigenvalues**2))
    if squares == 0.0:  # pragma: no cover
        raise ValueError("degenerate correlation spectrum")
    return float(total**2 / squares)


def deflated_sharpe_ratio(
    candidate: NDArray[np.float64], *, trial_sharpes: NDArray[np.float64], n_trials: float
) -> float:
    """Deflated Sharpe ratio of Bailey & López de Prado (2014).

    Units: daily log returns in, annualized by ``sessions_per_year`` only through the
    per-period Sharpe below; ``candidate`` holds per-period log returns.

    ``SR0 = sqrt(Var[trial_sharpes]) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))``
    ``DSR = Φ((SR - SR0)·sqrt(T-1) / sqrt(1 - skew·SR + (kurt-1)/4·SR²))``
    SR, skew and non-excess kurtosis are per-period values of ``candidate``.
    """
    x = np.asarray(candidate, dtype=np.float64)
    trials = np.asarray(trial_sharpes, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("candidate must be a 1-D array with at least 2 sessions")
    if trials.ndim != 1 or trials.size == 0:
        raise ValueError("trial_sharpes must be a non-empty 1-D array")
    if not n_trials >= 1:
        raise ValueError("n_trials must be >= 1")
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        raise ValueError("candidate must have positive variance")
    sharpe = float(np.mean(x) / std)
    centered = x - np.mean(x)
    moment2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / moment2**1.5) if moment2 > 0 else 0.0
    kurt = float(np.mean(centered**4) / moment2**2) if moment2 > 0 else 3.0
    if n_trials < 2 or float(np.var(trials)) == 0.0:
        expected_best = 0.0
    else:
        dispersion = math.sqrt(float(np.var(trials)))
        expected_best = dispersion * (
            (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    denominator = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denominator <= 0.0:  # pragma: no cover
        raise ValueError("non-positive DSR denominator")
    return float(norm.cdf((sharpe - expected_best) * math.sqrt(x.size - 1) / math.sqrt(denominator)))


@dataclass(frozen=True, slots=True)
class PointMetrics:
    """Point path statistics of one log-return stream."""

    cagr: float
    log_growth: float
    max_drawdown: float
    calmar: float
    sharpe: float
    underwater_sessions: int


def _longest_underwater_run(wealth: NDArray[np.float64]) -> int:
    peak = np.maximum.accumulate(wealth)
    below = wealth[1:] < peak[1:]
    longest = 0
    run = 0
    for flag in below:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def point_metrics(log_returns: NDArray[np.float64], *, sessions_per_year: int) -> PointMetrics:
    """CAGR = exp(mean*spy)-1; max_drawdown <= 0 on the wealth path; calmar = cagr/|max_drawdown|
    (+inf when the drawdown is 0); sharpe annualized from daily log returns; underwater_sessions = longest
    run of sessions with wealth below its running peak. Raises ValueError for empty/non-1-D input."""
    x = np.asarray(log_returns, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("log_returns must be a non-empty 1-D array")
    growth = float(np.mean(x) * sessions_per_year)
    cagr = float(math.exp(growth) - 1.0)
    drawdown = float(max_drawdown(x))
    calmar = float(cagr / abs(drawdown)) if drawdown != 0.0 else math.inf
    std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
    sharpe = float(np.mean(x) / std * math.sqrt(sessions_per_year)) if std > 0.0 else 0.0
    wealth = np.concatenate(([1.0], np.exp(np.cumsum(x))))
    return PointMetrics(
        cagr=cagr,
        log_growth=growth,
        max_drawdown=drawdown,
        calmar=calmar,
        sharpe=sharpe,
        underwater_sessions=_longest_underwater_run(wealth),
    )


@dataclass(frozen=True, slots=True)
class BootstrapProfile:
    """Horizon path-statistic profile from resampled blocks."""

    p_cagr_ge_abs_mdd: float
    p_cagr_le_zero: float
    cagr_p5: float
    mdd_p5: float
    p_mdd_below_limit: float
    underwater_median_sessions: float
    underwater_p95_sessions: float


def bootstrap_profile(
    log_returns: NDArray[np.float64],
    *,
    block: int,
    draws: int,
    seed: int,
    horizon: int,
    mdd_limit: float,
    sessions_per_year: int,
) -> BootstrapProfile:
    """Resample ``horizon`` sessions from contiguous blocks (uniform block starts, seeded) and summarize
    path statistics of each draw: P(CAGR >= |MDD|), P(CAGR <= 0), 5th percentile CAGR, 5th percentile
    (worst) MDD, P(MDD < mdd_limit), median/95th percentile longest underwater run.
    Why: a single historical path's Calmar is one draw; the question is how often the compounding
    condition holds over a 5-year horizon. Raises ValueError for block>len, draws<=0, horizon<=0."""
    values = np.asarray(log_returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_returns must be a non-empty 1-D array")
    if block <= 0 or block > values.size or draws <= 0 or horizon <= 0:
        raise ValueError("bootstrap requires 0 < block <= len(x), draws > 0 and horizon > 0")
    rng = np.random.default_rng(seed)
    upper = values.size - block
    cagrs = np.empty(draws, dtype=np.float64)
    mdds = np.empty(draws, dtype=np.float64)
    underwaters = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        pieces: list[NDArray[np.float64]] = []
        filled = 0
        while filled < horizon:
            start = int(rng.integers(0, upper + 1))
            chunk = values[start : start + block][: horizon - filled]
            pieces.append(chunk)
            filled += chunk.size
        path = np.concatenate(pieces)
        growth = float(np.mean(path) * sessions_per_year)
        cagrs[i] = float(math.exp(growth) - 1.0)
        wealth = np.concatenate(([1.0], np.exp(np.cumsum(path))))
        peak = np.maximum.accumulate(wealth)
        mdds[i] = float(np.min(wealth / peak - 1.0))
        underwaters[i] = float(_longest_underwater_run(wealth))
    return BootstrapProfile(
        p_cagr_ge_abs_mdd=float(np.mean(cagrs >= np.abs(mdds))),
        p_cagr_le_zero=float(np.mean(cagrs <= 0.0)),
        cagr_p5=float(np.percentile(cagrs, 5.0)),
        mdd_p5=float(np.percentile(mdds, 5.0)),
        p_mdd_below_limit=float(np.mean(mdds < mdd_limit)),
        underwater_median_sessions=float(np.median(underwaters)),
        underwater_p95_sessions=float(np.percentile(underwaters, 95.0)),
    )


def deflated_sharpe_ratio_from_dispersion(
    candidate: NDArray[np.float64], *, sharpe_std_per_period: float, n_trials: float
) -> float:
    """DSR with an externally supplied dispersion of trial Sharpes (per-period units), for multiplicity
    that predates the trial registry. Raises ValueError for n_trials < 1 or non-positive variance."""
    x = np.asarray(candidate, dtype=np.float64)
    dispersion = float(sharpe_std_per_period)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("candidate must be a 1-D array with at least 2 sessions")
    if not n_trials >= 1 or not math.isfinite(dispersion) or dispersion <= 0.0:
        raise ValueError("n_trials must be >= 1 with positive finite dispersion")
    std = float(np.std(x, ddof=1))
    if not std > 1e-12:
        raise ValueError("candidate must have positive variance")
    sharpe = float(np.mean(x) / std)
    centered = x - np.mean(x)
    moment2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / moment2**1.5) if moment2 > 0 else 0.0
    kurt = float(np.mean(centered**4) / moment2**2) if moment2 > 0 else 3.0
    if n_trials < 2:
        expected_best = 0.0
    else:
        expected_best = dispersion * (
            (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    denominator = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denominator <= 0.0:  # pragma: no cover
        raise ValueError("non-positive DSR denominator")
    return float(norm.cdf((sharpe - expected_best) * math.sqrt(x.size - 1) / math.sqrt(denominator)))
