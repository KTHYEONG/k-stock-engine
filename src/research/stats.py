"""Multiplicity statistics over active daily log returns."""

# ruff: noqa: RUF002
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

__all__ = [
    "EULER_GAMMA",
    "PboResult",
    "active_t_stat",
    "annualized_log_growth",
    "block_bootstrap_annualized_means",
    "cscv_pbo",
    "deflated_sharpe_ratio",
    "effective_trial_count",
    "information_ratio",
    "max_drawdown",
    "yearly_active_log",
]

EULER_GAMMA: Final = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class PboResult:
    pbo: float
    logits: NDArray[np.float64]


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


def information_ratio(active: NDArray[np.float64], *, sessions_per_year: int) -> float:
    """Annualized mean active log return per unit of tracking volatility."""
    x = np.asarray(active, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("active must be a 1-D array with at least 2 sessions")
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(x) / std * math.sqrt(sessions_per_year))


def active_t_stat(active: NDArray[np.float64]) -> float:
    """t-statistic of the mean active log return."""
    x = np.asarray(active, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("active must be a 1-D array with at least 2 sessions")
    std = float(np.std(x, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(x) / std * math.sqrt(x.size))


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


def _sharpe_of(block_values: NDArray[np.float64]) -> float:
    if block_values.size < 2:
        return 0.0
    std = float(np.std(block_values, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(block_values) / std)


def cscv_pbo(returns: NDArray[np.float64], *, blocks: int) -> PboResult:
    """Combinatorial symmetric cross-validation PBO over contiguous blocks.

    Units: daily log returns in, annualized by ``sessions_per_year`` only through the
    per-period Sharpe below.

    CSCV splits into ``blocks`` contiguous equal-as-possible blocks. Over all C(S, S/2) splits
    it picks the in-sample best Sharpe (ties go to the lowest column), ranks it out of sample
    as ``ω = rank/(K+1)`` with ``rank = 1 + #{OOS Sharpe < chosen}``, and sets
    ``λ = ln(ω/(1-ω))``. PBO is the share of splits with ``λ ≤ 0``.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2-D array")
    n_rows, n_cols = matrix.shape
    if blocks < 2 or blocks % 2 != 0:
        raise ValueError("blocks must be an even integer >= 2")
    if n_cols < 2:
        raise ValueError("returns must have at least 2 columns")
    if blocks > n_rows:
        raise ValueError("blocks must not exceed the session count")
    base, extra = divmod(n_rows, blocks)
    boundaries = [0]
    for i in range(blocks):
        boundaries.append(boundaries[-1] + base + (1 if i < extra else 0))
    chunks = [np.arange(boundaries[i], boundaries[i + 1]) for i in range(blocks)]
    half = blocks // 2
    logits: list[float] = []
    for in_sample in itertools.combinations(range(blocks), half):
        in_set = set(in_sample)
        in_idx = np.concatenate([chunks[b] for b in sorted(in_set)])
        out_idx = np.concatenate([chunks[b] for b in range(blocks) if b not in in_set])
        is_sharpes = np.array([_sharpe_of(matrix[in_idx, k]) for k in range(n_cols)], dtype=np.float64)
        oos_sharpes = np.array([_sharpe_of(matrix[out_idx, k]) for k in range(n_cols)], dtype=np.float64)
        chosen = int(np.argmax(is_sharpes))
        rank = 1 + int(np.sum(oos_sharpes < oos_sharpes[chosen]))
        omega = rank / (n_cols + 1)
        logits.append(math.log(omega / (1.0 - omega)))
    logits_array = np.array(logits, dtype=np.float64)
    return PboResult(pbo=float(np.mean(logits_array <= 0.0)), logits=logits_array)


def yearly_active_log(sessions: Sequence[date], active: NDArray[np.float64]) -> dict[int, float]:
    """Per-calendar-year sums of active log returns."""
    values = np.asarray(active, dtype=np.float64)
    days = list(sessions)
    if len(days) != values.size:
        raise ValueError("sessions and active must have equal length")
    totals: dict[int, float] = {}
    for day, value in zip(days, values, strict=True):
        totals[day.year] = totals.get(day.year, 0.0) + float(value)
    return totals
