"""Multiplicity statistics over active daily log returns."""

# ruff: noqa: RUF002
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GrowthProfile",
    "PairedDelta",
    "PointMetrics",
    "annualized_log_growth",
    "block_bootstrap_annualized_means",
    "breakeven_slippage_ticks",
    "growth_profile",
    "max_drawdown",
    "paired_growth_delta",
    "point_metrics",
]


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
class GrowthProfile:
    """Point and block-bootstrap profile of annualized log growth over a fixed horizon."""

    g_point: float
    g_quantile: float
    g_median: float
    p_growth_le_zero: float
    p_mdd_below: Mapping[float, float]
    mdd_median: float


def _check_bootstrap_args(
    n: int, *, block: int, draws: int, horizon: int, sessions_per_year: int
) -> None:
    if block <= 0 or block > n:
        raise ValueError(f"block must satisfy 0 < block <= len(x) ({n}), got {block}")
    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if sessions_per_year <= 0:
        raise ValueError(f"sessions_per_year must be positive, got {sessions_per_year}")


def _path_mdd(path: NDArray[np.float64]) -> float:
    wealth = np.concatenate(([1.0], np.exp(np.cumsum(path))))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def growth_profile(
    log_returns: NDArray[np.float64],
    *,
    block: int,
    draws: int,
    seed: int,
    horizon: int,
    sessions_per_year: int,
    quantile: float,
    mdd_limits: Sequence[float],
) -> GrowthProfile:
    """Resample ``horizon`` sessions from contiguous blocks with uniform seeded starts; growth of a path is
    ``mean(path)·sessions_per_year``. Raises ValueError for an empty/non-1-D series, block outside
    [1, len], draws ≤ 0, horizon ≤ 0, quantile outside (0, 1) or a limit outside (−1, 0)."""
    x = np.asarray(log_returns, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("log_returns must be a non-empty 1-D array")
    _check_bootstrap_args(x.size, block=block, draws=draws, horizon=horizon, sessions_per_year=sessions_per_year)
    q = float(quantile)
    if not math.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError(f"quantile must satisfy 0 < quantile < 1, got {quantile!r}")
    limits = [float(v) for v in mdd_limits]
    for lim in limits:
        if not math.isfinite(lim) or not -1.0 < lim < 0.0:
            raise ValueError(f"mdd limit must satisfy -1 < limit < 0, got {lim!r}")
    rng = np.random.default_rng(seed)
    upper = x.size - block
    growths = np.empty(draws, dtype=np.float64)
    mdds = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        pieces: list[NDArray[np.float64]] = []
        filled = 0
        while filled < horizon:
            start = int(rng.integers(0, upper + 1))
            chunk = x[start : start + block][: horizon - filled]
            pieces.append(chunk)
            filled += chunk.size
        path = np.concatenate(pieces)
        growths[i] = float(np.mean(path) * sessions_per_year)
        mdds[i] = _path_mdd(path)
    return GrowthProfile(
        g_point=float(np.mean(x) * sessions_per_year),
        g_quantile=float(np.quantile(growths, q)),
        g_median=float(np.median(growths)),
        p_growth_le_zero=float(np.mean(growths <= 0.0)),
        p_mdd_below={lim: float(np.mean(mdds < lim)) for lim in limits},
        mdd_median=float(np.median(mdds)),
    )


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """Bootstrap distribution summary of annualized growth(a) − growth(b) on shared sessions."""

    mean: float
    lower: float
    upper: float
    p_positive: float
    sessions: int


def paired_growth_delta(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    block: int,
    draws: int,
    seed: int,
    horizon: int,
    sessions_per_year: int,
    alpha: float,
) -> PairedDelta:
    """Apply the SAME block starts to both streams in every draw, so shared market noise cancels.
    Raises ValueError for unequal lengths or alpha outside (0, 0.5)."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.ndim != 1 or xa.size == 0 or xb.ndim != 1 or xb.size == 0:
        raise ValueError("inputs must be non-empty 1-D arrays")
    if xa.size != xb.size:
        raise ValueError(f"paired streams must share length, got {xa.size} vs {xb.size}")
    _check_bootstrap_args(xa.size, block=block, draws=draws, horizon=horizon, sessions_per_year=sessions_per_year)
    level = float(alpha)
    if not math.isfinite(level) or not 0.0 < level < 0.5:
        raise ValueError(f"alpha must satisfy 0 < alpha < 0.5, got {alpha!r}")
    rng = np.random.default_rng(seed)
    upper = xa.size - block
    deltas = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        parts_a: list[NDArray[np.float64]] = []
        parts_b: list[NDArray[np.float64]] = []
        filled = 0
        while filled < horizon:
            start = int(rng.integers(0, upper + 1))
            need = horizon - filled
            parts_a.append(xa[start : start + block][:need])
            parts_b.append(xb[start : start + block][:need])
            filled += parts_a[-1].size
        path_a = np.concatenate(parts_a)
        path_b = np.concatenate(parts_b)
        deltas[i] = float((np.mean(path_a) - np.mean(path_b)) * sessions_per_year)
    return PairedDelta(
        mean=float(np.mean(deltas)),
        lower=float(np.quantile(deltas, level)),
        upper=float(np.quantile(deltas, 1.0 - level)),
        p_positive=float(np.mean(deltas > 0.0)),
        sessions=int(xa.size),
    )


def breakeven_slippage_ticks(growth_by_ticks: Mapping[float, float]) -> float:
    """Slippage (ticks/side) at which growth crosses 0 by piecewise-linear interpolation of the sorted grid;
    ``inf`` when growth stays > 0 on the whole grid (extrapolation is not done), 0.0 when ≤ 0 at the first
    grid point. Raises ValueError for fewer than two points or non-finite values."""
    items = [(float(k), float(v)) for k, v in growth_by_ticks.items()]
    if len(items) < 2:
        raise ValueError(f"growth grid must hold at least two points, got {len(items)}")
    for tick, growth in items:
        if not math.isfinite(tick) or not math.isfinite(growth):
            raise ValueError("growth grid ticks and values must be finite")
    items.sort(key=lambda kv: kv[0])
    if items[0][1] <= 0.0:
        return 0.0
    for (t0, g0), (t1, g1) in pairwise(items):
        if g0 > 0.0 and g1 <= 0.0:
            if g1 == g0:  # pragma: no cover - guarded by the sign change
                return float(t1)
            frac = (0.0 - g0) / (g1 - g0)
            return float(t0 + frac * (t1 - t0))
    return math.inf
