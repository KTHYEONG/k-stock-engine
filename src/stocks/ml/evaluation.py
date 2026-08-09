"""Ranking-quality and economic evaluation metrics.

Evaluation records ranking quality and economic metrics separately, including
costs, turnover, exposure, drawdown, and coverage; it never chooses a model
solely by NDCG or one backtest metric.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata


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
