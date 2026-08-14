"""Horizon discovery: evidence-driven primary and conditional secondary selection.

Horizon selection is the single gate between the pre-registered candidate
horizon grid and the learner: a horizon is kept only when its cost-after-risk
exact-policy block-bootstrap lower bound is strictly positive, and a secondary
horizon survives only when it adds paired incremental utility after the primary
prediction is cross-sectionally residualized. No correlation threshold or magic
number selects or drops a horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np


@dataclass(frozen=True, slots=True)
class HorizonOOFEvidence:
    """One horizon's inner OOF net log-growth block evidence.

    ``block_log_excess`` is the per-block cost-after-risk net log-growth series
    produced by the same exact policy kernel for the given horizon; block
    length is at least the horizon so overlapping labels stay within one block.
    """

    horizon: int
    block_log_excess: tuple[float, ...]
    label_correlation: dict[int, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be a positive session count")
        if not self.block_log_excess:
            raise ValueError("horizon evidence requires a non-empty block series")


@dataclass(frozen=True, slots=True)
class HorizonSelectionEvidence:
    """Immutable outcome of one horizon-discovery run.

    ``primary_horizon`` is ``None`` when no candidate clears the lower-bound
    gate (a normal ``NO_TRADE`` outcome). ``secondary_horizon`` is at most one
    and only present when its paired incremental lower bound over the
    residualized primary prediction is strictly positive.
    """

    primary_horizon: int | None
    secondary_horizon: int | None
    lower_bounds: dict[int, float]
    effective_horizon_count: float
    selection_reasons: tuple[str, ...]
    correlation_matrix: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def selected_horizons(self) -> tuple[int, ...]:
        return tuple(
            h
            for h in (self.primary_horizon, self.secondary_horizon)
            if h is not None
        )

    def to_json(self) -> dict[str, object]:
        return {
            "primary_horizon": self.primary_horizon,
            "secondary_horizon": self.secondary_horizon,
            "lower_bounds": dict(self.lower_bounds),
            "effective_horizon_count": self.effective_horizon_count,
            "selection_reasons": list(self.selection_reasons),
            "correlation_matrix": {
                f"{i}x{j}": value
                for (i, j), value in sorted(self.correlation_matrix.items())
            },
        }

    @property
    def evidence_hash(self) -> str:
        payload = sha256()
        payload.update(f"{self.primary_horizon}:{self.secondary_horizon}".encode())
        for h, bound in sorted(self.lower_bounds.items()):
            payload.update(f"{h}:{bound:.17g};".encode())
        return payload.hexdigest()


def _moving_block_bootstrap_lower_bound(
    values: tuple[float, ...],
    block_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> float:
    """Seeded moving-block bootstrap alpha-quantile of block means."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        return 0.0
    block = min(max(block_length, 1), n)
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block + 1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(
        n_bootstrap, n_blocks * block
    )[:, :n]
    means = arr[index].mean(axis=1)
    return float(np.quantile(means, alpha))


def _effective_horizon_count(correlation: np.ndarray) -> float:
    """Effective independent horizon count ``tr(C)^2 / tr(C^2)``.

    A single-horizon system yields ``1.0``; perfectly correlated horizons
    approach ``1.0`` from above while independent horizons approach ``k``.
    """
    if correlation.size == 0:
        return 0.0
    trace = float(np.trace(correlation))
    trace_sq = float(np.trace(correlation @ correlation))
    if trace_sq <= 0.0:
        return 0.0
    return trace * trace / trace_sq


def select_horizons(
    candidates: tuple[HorizonOOFEvidence, ...],
    alpha: float,
    seed: int,
) -> HorizonSelectionEvidence:
    """Select at most one primary and one conditional secondary horizon.

    Selection is evidence-only: every candidate's ``block_log_excess`` is
    resampled with a moving block whose length is at least the horizon, and only
    strictly positive lower bounds survive. Ties within bootstrap resolution
    prefer the shorter horizon, then the smaller integer. A secondary is
    allowed only when, after residualizing the primary prediction, its paired
    incremental policy-utility lower bound is strictly positive.

    Args:
        candidates: pre-registered candidate horizons with their OOF block
            series, in ascending horizon order.
        alpha: bootstrap alpha quantile for the lower bound.
        seed: deterministic bootstrap seed.

    Returns:
        ``HorizonSelectionEvidence``; ``primary_horizon`` is ``None`` when every
        lower bound is non-positive (the ``NO_TRADE`` outcome).
    """
    if not candidates:
        raise ValueError("select_horizons requires at least one candidate")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    ordered = tuple(sorted(candidates, key=lambda c: c.horizon))
    correlation = _horizon_correlation_matrix(ordered)
    lower_bounds: dict[int, float] = {}
    for candidate in ordered:
        block_length = max(candidate.horizon, 1)
        lower_bounds[candidate.horizon] = _moving_block_bootstrap_lower_bound(
            candidate.block_log_excess,
            block_length,
            n_bootstrap=200,
            seed=seed,
            alpha=alpha,
        )

    eligible = [
        candidate
        for candidate in ordered
        if lower_bounds[candidate.horizon] > 0.0
    ]
    reasons: list[str] = []
    if not eligible:
        reasons.append("no candidate has a positive lower bound")
        return HorizonSelectionEvidence(
            primary_horizon=None,
            secondary_horizon=None,
            lower_bounds=lower_bounds,
            effective_horizon_count=_effective_horizon_count(correlation),
            selection_reasons=tuple(reasons),
            correlation_matrix=_correlation_pairs(ordered, correlation),
        )

    primary = _tiebreak_primary(eligible, lower_bounds)
    reasons.append(
        f"primary={primary.horizon} from lower bounds "
        f"{{{', '.join(f'{h}:{lower_bounds[h]:.6g}' for h in lower_bounds)}}}"
    )

    secondary_horizon = _conditional_secondary(
        primary, ordered, lower_bounds, alpha=alpha, seed=seed
    )
    if secondary_horizon is not None:
        reasons.append(
            f"secondary={secondary_horizon}: paired incremental lower bound > 0"
        )
    else:
        reasons.append("no secondary clears the incremental lower bound")

    return HorizonSelectionEvidence(
        primary_horizon=primary.horizon,
        secondary_horizon=secondary_horizon,
        lower_bounds=lower_bounds,
        effective_horizon_count=_effective_horizon_count(correlation),
        selection_reasons=tuple(reasons),
        correlation_matrix=_correlation_pairs(ordered, correlation),
    )


def _tiebreak_primary(
    eligible: list[HorizonOOFEvidence],
    lower_bounds: dict[int, float],
) -> HorizonOOFEvidence:
    """Within bootstrap resolution prefer shorter, then smaller, horizon."""
    best = eligible[0]
    for candidate in eligible[1:]:
        current_bound = lower_bounds[candidate.horizon]
        best_bound = lower_bounds[best.horizon]
        if (
            current_bound > best_bound + 1e-12
            or (
                abs(current_bound - best_bound) <= 1e-12
                and candidate.horizon < best.horizon
            )
        ):
            best = candidate
    return best


def _conditional_secondary(
    primary: HorizonOOFEvidence,
    ordered: tuple[HorizonOOFEvidence, ...],
    lower_bounds: dict[int, float],
    *,
    alpha: float,
    seed: int,
) -> int | None:
    """Conditional secondary: paired incremental lower bound over residualized primary.

    The secondary candidate's block series is cross-sectionally residualized
    against the primary's (orthogonal linear projection, so a perfectly
    correlated candidate contributes nothing), then resampled; a strictly
    positive lower bound on that residualized incremental series is the only
    admission gate.
    """
    primary_blocks = np.asarray(primary.block_log_excess, dtype=float)
    best_secondary: int | None = None
    best_bound = 0.0
    for candidate in ordered:
        if candidate.horizon == primary.horizon:
            continue
        if lower_bounds.get(candidate.horizon, 0.0) <= 0.0:
            continue
        secondary_blocks = np.asarray(candidate.block_log_excess, dtype=float)
        length = min(primary_blocks.size, secondary_blocks.size)
        if length < 2:
            continue
        residual = _orthogonal_residual(secondary_blocks[:length], primary_blocks[:length])
        if residual is None:
            continue
        block_length = max(candidate.horizon, 1)
        bound = _moving_block_bootstrap_lower_bound(
            tuple(float(v) for v in residual),
            block_length,
            n_bootstrap=200,
            seed=seed + candidate.horizon,
            alpha=alpha,
        )
        if bound > best_bound:
            best_bound = bound
            best_secondary = candidate.horizon
    if best_secondary is not None and best_bound > 0.0:
        return best_secondary
    return None


def _orthogonal_residual(
    secondary: np.ndarray, primary: np.ndarray
) -> np.ndarray | None:
    """Cross-sectional residual of ``secondary`` on ``primary``.

    Fits ``secondary = beta * primary`` by slope-only least squares and returns
    ``secondary - beta * primary``, so a candidate that is purely a scaled copy
    of the primary contributes zero incremental utility while an independent
    positive drift survives. A degenerate primary series returns ``None`` so a
    candidate contributes nothing.
    """
    denom = float(np.dot(primary, primary))
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    beta = float(np.dot(primary, secondary)) / denom
    residual = secondary - beta * primary
    if not np.all(np.isfinite(residual)):
        return None
    return residual


def _horizon_correlation_matrix(
    candidates: tuple[HorizonOOFEvidence, ...],
) -> np.ndarray:
    """Pairwise Pearson correlation of horizon block series."""
    count = len(candidates)
    correlation = np.ones((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            left = np.asarray(candidates[i].block_log_excess, dtype=float)
            right = np.asarray(candidates[j].block_log_excess, dtype=float)
            length = min(left.size, right.size)
            if length < 2:
                value = 0.0
            else:
                value = float(np.corrcoef(left[:length], right[:length])[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            correlation[i, j] = value
            correlation[j, i] = value
    return correlation


def _correlation_pairs(
    candidates: tuple[HorizonOOFEvidence, ...],
    correlation: np.ndarray,
) -> dict[tuple[int, int], float]:
    return {
        (candidates[i].horizon, candidates[j].horizon): float(correlation[i, j])
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
    }
