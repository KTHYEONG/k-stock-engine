"""Deterministic tail-objective primitives: exact-K relevance and tail capture.

``build_tail_relevance`` derives the fold-local ordering target from decimal
``risk_residual - reference_cost`` per decision session; ties break by
ascending ``instrument_id`` and any missing, non-finite, or undersized
cross-section fails closed. ``measure_tail_capture`` compares model-selected
top-K arithmetic mean residual utility with the same-session oracle top-K and
the full universe, emitting bounded scalar evidence only. The residual is a
cross-sectional ordering/utility proxy, never a portfolio return: no
``log1p`` growth operation exists here, and exact execution-ledger equity is
the only compound-growth evidence. Relevance is an ordering target, never
realised PnL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.stocks.ml.labels import (
    GROSS_COLUMN,
    ID_COLUMN,
    REFERENCE_COST_COLUMN,
    RISK_RESIDUAL_COLUMN,
    SESSION_COLUMN,
)
from src.stocks.ml.models import SCORE_COLUMN

__all__ = [
    "InvalidOofEconomicUtilityError",
    "SegmentTailEvidence",
    "TailCaptureEvidence",
    "build_tail_relevance",
    "measure_tail_capture",
    "project_route_utility",
    "route_labels_for_capture",
]


class InvalidOofEconomicUtilityError(ValueError):
    """A score-label join carried null or non-finite residual/cost utility."""

_RELEVANCE_COLUMN = "relevance"
_SEGMENT_COLUMN_CANDIDATES = ("oof_segment_id",)


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], role: str) -> None:
    if frame.is_empty():
        raise ValueError(f"{role} frame is empty")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{role} frame is missing required columns {missing}")


def _decimal_utility(frame: pl.DataFrame) -> pl.Series:
    """Arithmetic decimal residual utility; rankable on the whole real line."""
    values = (
        pl.col(RISK_RESIDUAL_COLUMN) - pl.col(REFERENCE_COST_COLUMN)
    ).alias("__utility")
    series = frame.select(values)["__utility"].cast(pl.Float64)
    if series.null_count() > 0:
        raise InvalidOofEconomicUtilityError(
            "economic utility has null risk_residual/reference_cost rows; "
            "a missing realized outcome must fail closed"
        )
    array = series.to_numpy()
    if not np.all(np.isfinite(array)):
        raise InvalidOofEconomicUtilityError("economic utility must be finite")
    return series


def project_route_utility(
    frame: pl.DataFrame,
    route: object,
) -> pl.Series:
    """Project the route-aligned learning target without stateful cost.

    For ``unhedged_absolute`` the gross absolute outcome is used; for
    ``hedged_residual`` the risk residual is used only when hedge evidence
    is certified. The reusable label never bakes an account-specific static
    cost into the learning target — execution cost stays in the portfolio
    replay stage. Missing hedge evidence raises before any fit.
    """
    kind_value = getattr(route, "kind", route)
    kind_str = str(getattr(kind_value, "value", kind_value)).lower()

    if kind_str == "hedged_residual":
        # Must have certified hedge evidence
        instrument = getattr(route, "hedge_instrument", None)
        evidence = getattr(route, "hedge_evidence_hash", None)
        if not instrument or not evidence:
            raise ValueError(
                "hedged_residual route requires hedge_instrument and hedge_evidence_hash"
            )
        if RISK_RESIDUAL_COLUMN not in frame.columns:
            raise ValueError(f"frame missing required column {RISK_RESIDUAL_COLUMN!r}")
        series = frame[RISK_RESIDUAL_COLUMN].cast(pl.Float64)
        if series.null_count() > 0:
            raise InvalidOofEconomicUtilityError("hedged residual has null rows")
        arr = series.to_numpy()
        if not np.all(np.isfinite(arr)):
            raise InvalidOofEconomicUtilityError("hedged residual must be finite")
        return series
    # Unhedged absolute requires gross_return; never silently substitute residual.
    if kind_str != "hedged_residual":
        if GROSS_COLUMN not in frame.columns:
            raise ValueError(f"unhedged_absolute route requires {GROSS_COLUMN!r} column (gross missing)")
        series = frame[GROSS_COLUMN].cast(pl.Float64)
    elif RISK_RESIDUAL_COLUMN in frame.columns:
        series = frame[RISK_RESIDUAL_COLUMN].cast(pl.Float64)
    else:
        raise ValueError(f"frame missing {GROSS_COLUMN!r} or {RISK_RESIDUAL_COLUMN!r}")
    if series.null_count() > 0:
        raise InvalidOofEconomicUtilityError("route utility has null rows")
    arr = series.to_numpy()
    if not np.all(np.isfinite(arr)):
        raise InvalidOofEconomicUtilityError("route utility must be finite")
    return series


def route_labels_for_capture(frame: pl.DataFrame, route: object) -> pl.DataFrame:
    """Build capture labels whose utility matches the pre-bound route."""
    utility = project_route_utility(frame, route)
    return frame.select(ID_COLUMN, SESSION_COLUMN).with_columns(
        utility.alias(RISK_RESIDUAL_COLUMN),
        pl.lit(0.0).alias(REFERENCE_COST_COLUMN),
    )


def build_tail_relevance(labels: pl.DataFrame, *, top_k: int) -> pl.DataFrame:
    """Exact-K cross-sectional indicator of ``risk_residual - reference_cost``.

    Rows rank by descending decimal utility inside each session with ascending
    ``instrument_id`` as the deterministic tie-breaker; exactly ``top_k`` rows
    per session carry ``relevance=1``. Sessions holding fewer than ``top_k``
    labelled names raise ``ValueError`` instead of degrading silently.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    _require_columns(
        labels,
        (ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN),
        "labels",
    )
    utility = _decimal_utility(labels).alias("__utility")
    work = labels.with_columns(utility)
    sizes = work.group_by(SESSION_COLUMN).len()
    undersized = sizes.filter(pl.col("len") < top_k)
    if not undersized.is_empty():
        raise ValueError(
            f"undersized cross-sections: {undersized.height} session(s) hold "
            f"fewer than top_k={top_k} labelled names"
        )
    return work.sort(
        [SESSION_COLUMN, "__utility", ID_COLUMN],
        descending=[False, True, False],
        maintain_order=True,
    ).with_columns(
        (pl.int_range(pl.len()).over(SESSION_COLUMN) < top_k)
        .cast(pl.Int8)
        .alias(_RELEVANCE_COLUMN)
    ).sort([SESSION_COLUMN, ID_COLUMN], maintain_order=True)


@dataclass(frozen=True, slots=True)
class SegmentTailEvidence:
    """Bounded per-cluster tail-capture scalars."""

    segment_id: int
    session_count: int
    model_excess_utility: float
    oracle_excess_utility: float
    tail_capture_ratio: float | None
    positive_session_fraction: float


@dataclass(frozen=True, slots=True)
class TailCaptureEvidence:
    """Bounded tail-capture comparison of model top-K versus oracle and universe.

    Excesses are the mean per-session ``(top-K mean decimal residual utility)
    - (universe mean residual utility)`` computed arithmetically; they are
    never compound-growth quantities. ``tail_excess_lower_bound`` is the seeded
    cluster-bootstrap lower quantile of the model excess utility; Rank IC is
    deliberately absent because it is observability only.
    """

    top_k: int
    session_count: int
    model_excess_utility: float
    oracle_excess_utility: float
    tail_capture_ratio: float | None
    positive_session_fraction: float
    tail_excess_lower_bound: float
    segments: tuple[SegmentTailEvidence, ...]
    bootstrap_alpha: float
    bootstrap_resamples: int
    seed: int

    @property
    def oracle_capacity_ok(self) -> bool:
        return self.oracle_excess_utility > 0.0

    @property
    def tail_gate_ok(self) -> bool:
        return self.tail_excess_lower_bound > 0.0


def measure_tail_capture(
    scored: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    top_k: int,
    bootstrap_alpha: float,
    bootstrap_resamples: int,
    seed: int,
) -> TailCaptureEvidence:
    """Compare model-selected top-K against same-session oracle and universe.

    The scored frame supplies ``instrument_id``, ``session``, scores, and an
    optional ``oof_segment_id`` cluster column; labels supply the decimal
    cost-adjusted utility. Bootstrap clusters are OOF segments when present
    and individual sessions otherwise. No raw rows or scores are retained.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not 0.0 < bootstrap_alpha < 1.0:
        raise ValueError("bootstrap_alpha must be in (0, 1)")
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    _require_columns(scored, (ID_COLUMN, SESSION_COLUMN, SCORE_COLUMN), "scored")
    _require_columns(
        labels,
        (ID_COLUMN, SESSION_COLUMN, RISK_RESIDUAL_COLUMN, REFERENCE_COST_COLUMN),
        "labels",
    )
    segment_column = next(
        (c for c in _SEGMENT_COLUMN_CANDIDATES if c in scored.columns), None
    )
    # Validate utility only on rows the join actually produces: locked-holdout
    # labels never influence runtime research evidence.
    joined = scored.join(
        labels.select(
            pl.col(ID_COLUMN),
            pl.col(SESSION_COLUMN),
            pl.col(RISK_RESIDUAL_COLUMN),
            pl.col(REFERENCE_COST_COLUMN),
        ),
        on=[ID_COLUMN, SESSION_COLUMN],
        how="inner",
    )
    if joined.is_empty():
        raise ValueError("scored-label join is empty")

    utility = _decimal_utility(joined).to_numpy()
    scores = joined[SCORE_COLUMN].cast(pl.Float64).to_numpy()
    ids = joined[ID_COLUMN].to_numpy()
    sessions = joined[SESSION_COLUMN].to_numpy()
    clusters = (
        joined[segment_column].cast(pl.Int64).to_numpy()
        if segment_column is not None
        else np.zeros(joined.height, dtype=np.int64)
    )
    order = np.argsort(sessions, kind="stable")
    utility = utility[order]
    scores = scores[order]
    ids = ids[order]
    clusters = clusters[order]
    sessions = sessions[order]

    boundaries = np.flatnonzero(np.diff(sessions)) + 1
    session_slices = np.split(np.arange(sessions.size), boundaries)

    model_excess = np.empty(session_slices.__len__(), dtype=np.float64)
    oracle_excess = np.empty_like(model_excess)
    positive = np.zeros_like(model_excess, dtype=bool)
    cluster_of_session = np.empty_like(model_excess, dtype=np.int64)
    for index, slice_rows in enumerate(session_slices):
        u = utility[slice_rows]
        k = min(top_k, u.size)
        universe_mean = math.fsum(u.tolist()) / u.size
        oracle_pick = np.lexsort((ids[slice_rows], -u))[:k]
        score_pick = np.lexsort((ids[slice_rows], -scores[slice_rows]))[:k]
        model_mean = math.fsum(u[score_pick].tolist()) / k
        oracle_mean = math.fsum(u[oracle_pick].tolist()) / k
        model_excess[index] = model_mean - universe_mean
        oracle_excess[index] = oracle_mean - universe_mean
        positive[index] = model_mean > 0.0
        cluster_of_session[index] = int(clusters[slice_rows[0]])

    unique_clusters = np.unique(cluster_of_session)
    cluster_index = {int(c): i for i, c in enumerate(unique_clusters)}
    mapped = np.asarray(
        [cluster_index[int(c)] for c in cluster_of_session], dtype=np.int64
    )
    cluster_sums = np.bincount(mapped, weights=model_excess)
    cluster_counts = np.bincount(mapped).astype(np.float64)
    cluster_positives = np.bincount(mapped, weights=positive.astype(np.float64))

    segment_evidence: list[SegmentTailEvidence] = []
    for local_index, cluster in enumerate(unique_clusters):
        model_c = float(cluster_sums[local_index] / cluster_counts[local_index])
        oracle_slice = oracle_excess[mapped == local_index]
        oracle_c = math.fsum(oracle_slice.tolist()) / oracle_slice.size if oracle_slice.size else 0.0
        segment_evidence.append(
            SegmentTailEvidence(
                segment_id=int(cluster),
                session_count=int(cluster_counts[local_index]),
                model_excess_utility=model_c,
                oracle_excess_utility=oracle_c,
                tail_capture_ratio=(
                    model_c / oracle_c if oracle_c > 0.0 else None
                ),
                positive_session_fraction=float(
                    cluster_positives[local_index] / cluster_counts[local_index]
                ),
            )
        )

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique_clusters.size, size=(bootstrap_resamples, unique_clusters.size))
    bootstrap_means = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    lower_bound = float(np.quantile(bootstrap_means, bootstrap_alpha))

    total_oracle = math.fsum(oracle_excess.tolist()) / oracle_excess.size if oracle_excess.size else 0.0
    total_model = math.fsum(model_excess.tolist()) / model_excess.size if model_excess.size else 0.0
    return TailCaptureEvidence(
        top_k=int(top_k),
        session_count=int(model_excess.size),
        model_excess_utility=total_model,
        oracle_excess_utility=total_oracle,
        tail_capture_ratio=(
            total_model / total_oracle if total_oracle > 0.0 else None
        ),
        positive_session_fraction=float(positive.mean()),
        tail_excess_lower_bound=lower_bound,
        segments=tuple(segment_evidence),
        bootstrap_alpha=float(bootstrap_alpha),
        bootstrap_resamples=int(bootstrap_resamples),
        seed=int(seed),
    )
