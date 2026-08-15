"""Horizon discovery: cohort-unit bootstrap, Holm admission, one primary horizon.

Selection is the single gate between the pre-registered candidate horizon grid
and the learner. Every candidate carries base and stress per-session log-growth
cohort evidence over the common, segment-identity-preserving OOF calendar.
Admission applies a one-sided centered moving-block bootstrap in cohort units
(resampling never crosses a segment boundary) and Holm-Bonferroni multiplicity
control across all pre-registered horizons; a horizon is economically admissible
only when both its base and stress adjusted lower growth bounds are strictly
positive. At most one primary horizon is selected (the maximum stress-cost
adjusted lower growth, ties preferring the shorter horizon). There is no
secondary horizon and no effective-horizon heuristic here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 200
_BOUND_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class HorizonOOFEvidence:
    """One horizon's base/stress cohort log-growth evidence.

    ``base_log_growth`` and ``stress_log_growth`` are parallel per-session log
    growth series ``log1p(r) / horizon_sessions`` over the complete cohorts of
    the common OOF calendar; ``cohort_segment_ids`` is the segment identity of
    each cohort so resampling never crosses a segment boundary. ``model_family``
    records which family produced the OOF scores. The cohort counts publish the
    complete/active/partial/missing decomposition and ``fold_rank_ics`` the
    session-mean Rank-IC per usable fold.
    """

    horizon_sessions: int
    model_family: str
    base_log_growth: tuple[float, ...]
    stress_log_growth: tuple[float, ...]
    cohort_segment_ids: tuple[int, ...]
    complete_cohort_count: int
    active_cohort_count: int
    partial_cohort_count: int
    missing_cohort_count: int
    segment_count: int
    fold_rank_ics: tuple[float, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be a positive session count")
        if len(self.base_log_growth) != len(self.stress_log_growth):
            raise ValueError("base and stress log growth series must be parallel")
        if len(self.base_log_growth) != len(self.cohort_segment_ids):
            raise ValueError(
                "cohort_segment_ids must be parallel to the log growth series"
            )
        if not self.base_log_growth:
            raise ValueError("horizon evidence requires a non-empty cohort series")
        if not np.all(np.isfinite(self.base_log_growth)) or not np.all(
            np.isfinite(self.stress_log_growth)
        ):
            raise ValueError("cohort log growth must be finite")
        if any(segment < 0 for segment in self.cohort_segment_ids):
            raise ValueError("cohort segment identity must be a non-negative integer")
        if not self.model_family:
            raise ValueError("model_family must be non-empty")


@dataclass(frozen=True, slots=True)
class HorizonSelectionEvidence:
    """Immutable outcome of one Holm-adjusted horizon-discovery run.

    ``primary_horizon_sessions`` is ``None`` when no candidate is economically
    admissible (a normal ``NO_TRADE`` outcome). ``adjusted_lower_growth`` maps a
    candidate horizon to its per-session log-growth lower means at the
    candidate's Holm threshold for the ``base`` and ``stress`` cost paths; the
    annualized ``LB_CAGR`` is derived by the caller as
    ``expm1(annualization_sessions * lower_growth)``. ``rankability_reason``
    records why the linear screen may not run a nonlinear challenger.
    """

    primary_horizon_sessions: int | None
    adjusted_lower_growth: dict[int, dict[str, float]]
    base_p_values: dict[int, float]
    stress_p_values: dict[int, float]
    base_holm_thresholds: dict[int, float]
    stress_holm_thresholds: dict[int, float]
    selection_reasons: tuple[str, ...]
    rankability_reason: str = ""
    rank_ic_lower_bound: float = 0.0

    @property
    def selected_horizons(self) -> tuple[int, ...]:
        return (
            ()
            if self.primary_horizon_sessions is None
            else (self.primary_horizon_sessions,)
        )

    def to_json(self) -> dict[str, object]:
        return {
            "primary_horizon_sessions": self.primary_horizon_sessions,
            "adjusted_lower_growth": {
                str(horizon): dict(path)
                for horizon, path in sorted(self.adjusted_lower_growth.items())
            },
            "base_p_values": dict(self.base_p_values),
            "stress_p_values": dict(self.stress_p_values),
            "base_holm_thresholds": dict(self.base_holm_thresholds),
            "stress_holm_thresholds": dict(self.stress_holm_thresholds),
            "selection_reasons": list(self.selection_reasons),
            "rankability_reason": self.rankability_reason,
            "rank_ic_lower_bound": self.rank_ic_lower_bound,
        }

    @property
    def evidence_hash(self) -> str:
        payload = sha256()
        payload.update(f"{self.primary_horizon_sessions}".encode())
        for horizon, path in sorted(self.adjusted_lower_growth.items()):
            payload.update(
                f"{horizon}:{path.get('base', 0.0):.17g}:"
                f"{path.get('stress', 0.0):.17g};".encode()
            )
        return payload.hexdigest()


def _segment_block_length(n_cohorts: int) -> int:
    """Deterministic moving-block length ``ceil(n ** (1/3))`` in cohort units."""
    if n_cohorts < 1:
        raise ValueError("segment must contain at least one cohort")
    block: int = ceil(float(n_cohorts) ** (1.0 / 3.0))
    return max(1, block)


@dataclass(frozen=True, slots=True)
class _CohortBootstrap:
    """One candidate cost path's cohort-unit bootstrap summary."""

    observed_mean: float
    boot_means: np.ndarray
    p_value: float
    n_blocks_total: int

    def lower_mean(self, quantile_level: float) -> float:
        return float(np.quantile(self.boot_means, quantile_level))


def _cohort_bootstrap(
    log_growth: tuple[float, ...],
    cohort_segment_ids: tuple[int, ...],
    n_bootstrap: int,
    seed: int,
) -> _CohortBootstrap | None:
    """One-sided centered moving-block bootstrap in non-overlapping cohort units.

    Resampling is performed independently within each segment with block length
    ``ceil(n_s ** (1/3))`` cohorts and never crosses a segment boundary; the
    per-segment resample means are pooled weighted by cohort count. Returns
    ``None`` (inadmissible) when any segment has fewer than two resampling
    blocks.
    """
    by_segment: dict[int, list[float]] = {}
    for segment, value in zip(cohort_segment_ids, log_growth, strict=True):
        by_segment.setdefault(int(segment), []).append(float(value))
    distributions: list[np.ndarray] = []
    weights: list[float] = []
    n_blocks_total = 0
    for segment in sorted(by_segment):
        values = np.asarray(by_segment[segment], dtype=float)
        block = _segment_block_length(values.size)
        n_blocks = int(np.ceil(values.size / block))
        if n_blocks < 2:
            return None
        n_blocks_total += n_blocks
        rng = np.random.default_rng(seed + segment)
        starts = rng.integers(0, max(1, values.size - block + 1), size=(n_bootstrap, n_blocks))
        offsets = np.arange(block)
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            n_bootstrap, n_blocks * block
        )[:, : values.size]
        distributions.append(values[index].mean(axis=1))
        weights.append(float(values.size))
    total = sum(weights)
    if total <= 0.0:
        return None
    pooled = np.zeros(n_bootstrap, dtype=np.float64)
    for weight, distribution in zip(weights, distributions, strict=True):
        pooled += weight * distribution
    pooled /= total
    observed = float(sum(log_growth) / len(log_growth))
    centered_p_value = float(np.mean(pooled >= 2.0 * observed))
    return _CohortBootstrap(
        observed_mean=observed,
        boot_means=pooled,
        p_value=centered_p_value,
        n_blocks_total=n_blocks_total,
    )


def _holm_admission(
    candidates: tuple[HorizonOOFEvidence, ...],
    bootstrap: dict[int, dict[str, _CohortBootstrap | None]],
    bootstrap_alpha: float,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float],
    list[str],
]:
    """Holm-Bonferroni control across all candidates on the least favorable path.

    The family is the candidate horizon set; each candidate's combined p-value is
    the maximum of its base and stress centered p-values (the least favorable
    path). The hypothesis at sorted rank ``j`` is rejected when
    ``combined_p <= alpha / (m - j + 1)``. Returns the per-candidate combined
    p-values, the base and stress Holm thresholds, and rejection reasons.
    """
    combined: dict[int, float] = {}
    for candidate in candidates:
        horizon = candidate.horizon_sessions
        base = bootstrap[horizon].get("base")
        stress = bootstrap[horizon].get("stress")
        if base is None or stress is None:
            combined[horizon] = 1.0
            continue
        combined[horizon] = max(base.p_value, stress.p_value)
    m = len(candidates)
    ordered = sorted(candidates, key=lambda candidate: combined[candidate.horizon_sessions])
    base_thresholds: dict[int, float] = {}
    stress_thresholds: dict[int, float] = {}
    reasons: list[str] = []
    for rank, candidate in enumerate(ordered, start=1):
        horizon = candidate.horizon_sessions
        threshold = bootstrap_alpha / (m - rank + 1)
        base_thresholds[horizon] = threshold
        stress_thresholds[horizon] = threshold
        base = bootstrap[horizon]["base"]
        stress = bootstrap[horizon]["stress"]
        if base is None or stress is None:
            reasons.append(
                f"h{horizon}: inadmissible (fewer than two resampling blocks)"
            )
        elif combined[horizon] > threshold:
            reasons.append(
                f"h{horizon}: Holm p {combined[horizon]:.6g} > {threshold:.6g}"
            )
    return combined, base_thresholds, stress_thresholds, reasons


def select_horizons(
    evidence: tuple[HorizonOOFEvidence, ...],
    bootstrap_alpha: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> HorizonSelectionEvidence:
    """Select at most one economically admissible primary horizon.

    Selection is evidence-only: every candidate's base and stress per-session
    log-growth series are resampled in cohort units (segment-local, never across
    segment boundaries) and one-sided centered p-values are computed for the
    null ``mean(g) <= 0``. Holm-Bonferroni is applied across every pre-registered
    candidate; a horizon is admissible only when both its base and stress
    adjusted lower growth are strictly positive. The primary is the admissible
    horizon with the maximum stress-cost adjusted lower growth (ties prefer the
    shorter horizon). ``primary_horizon_sessions`` is ``None`` when no candidate
    is admissible (the ``NO_TRADE`` outcome).

    Args:
        evidence: pre-registered candidate horizons with their base/stress cohort
            evidence, in ascending horizon order.
        bootstrap_alpha: bootstrap alpha quantile for the lower bound and Holm
            family-wise control.
        seed: deterministic bootstrap seed.
        n_bootstrap: request-controlled moving-block bootstrap resample count;
            values below two are rejected.

    Returns:
        ``HorizonSelectionEvidence``; ``primary_horizon_sessions`` is ``None``
        when every candidate is rejected.
    """
    if not evidence:
        raise ValueError("select_horizons requires at least one candidate")
    if not 0.0 < bootstrap_alpha < 1.0:
        raise ValueError("bootstrap_alpha must be in (0, 1)")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    ordered = tuple(sorted(evidence, key=lambda candidate: candidate.horizon_sessions))
    bootstrap: dict[int, dict[str, _CohortBootstrap | None]] = {}
    for candidate in ordered:
        horizon = candidate.horizon_sessions
        bootstrap[horizon] = {
            "base": _cohort_bootstrap(
                candidate.base_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed,
            ),
            "stress": _cohort_bootstrap(
                candidate.stress_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed + horizon,
            ),
        }

    _combined, base_thresholds, stress_thresholds, reasons = _holm_admission(
        ordered, bootstrap, bootstrap_alpha
    )

    adjusted_lower_growth: dict[int, dict[str, float]] = {}
    admissible: list[HorizonOOFEvidence] = []
    for candidate in ordered:
        horizon = candidate.horizon_sessions
        threshold = base_thresholds[horizon]
        base = bootstrap[horizon]["base"]
        stress = bootstrap[horizon]["stress"]
        base_lower = 0.0
        stress_lower = 0.0
        if base is not None:
            base_lower = base.lower_mean(threshold)
        if stress is not None:
            stress_lower = stress.lower_mean(threshold)
        adjusted_lower_growth[horizon] = {
            "base": base_lower,
            "stress": stress_lower,
        }
        if base_lower > 0.0 and stress_lower > 0.0:
            admissible.append(candidate)
            reasons.append(
                f"h{horizon}: admissible base={base_lower:.6g} stress={stress_lower:.6g}"
            )
        else:
            reasons.append(
                f"h{horizon}: adjusted lower growth base={base_lower:.6g} "
                f"stress={stress_lower:.6g} not strictly positive"
            )

    primary: int | None = None
    if admissible:
        primary = _best_primary(admissible, adjusted_lower_growth)
        reasons.append(f"primary={primary} (max stress adjusted lower growth)")
    else:
        reasons.append("no candidate is economically admissible")

    base_p_values: dict[int, float] = {}
    stress_p_values: dict[int, float] = {}
    for horizon in bootstrap:
        base = bootstrap[horizon]["base"]
        stress = bootstrap[horizon]["stress"]
        base_p_values[horizon] = base.p_value if base is not None else 1.0
        stress_p_values[horizon] = stress.p_value if stress is not None else 1.0

    return HorizonSelectionEvidence(
        primary_horizon_sessions=primary,
        adjusted_lower_growth=adjusted_lower_growth,
        base_p_values=base_p_values,
        stress_p_values=stress_p_values,
        base_holm_thresholds=base_thresholds,
        stress_holm_thresholds=stress_thresholds,
        selection_reasons=tuple(reasons),
    )


def _best_primary(
    admissible: list[HorizonOOFEvidence],
    adjusted_lower_growth: dict[int, dict[str, float]],
) -> int:
    """Maximum stress adjusted lower growth; ties prefer the shorter horizon."""
    best = admissible[0].horizon_sessions
    for candidate in admissible[1:]:
        horizon = candidate.horizon_sessions
        current = adjusted_lower_growth[horizon]["stress"]
        best_value = adjusted_lower_growth[best]["stress"]
        if (
            current > best_value + _BOUND_TOLERANCE
            or (abs(current - best_value) <= _BOUND_TOLERANCE and horizon < best)
        ):
            best = horizon
    return best
