"""Horizon/profile discovery: vintage bootstrap, Holm admission, one primary.

Selection is the single gate between the pre-registered candidate
``(horizon, profile)`` frontier and the learner. Every candidate carries base
and stress per-session log-growth evidence over the common,
segment-identity-preserving OOF calendar; ``profile_id`` records which policy
profile produced the evidence. Admission applies a one-sided centered
moving-block bootstrap in per-vintage session units with a block length of at
least the candidate horizon (preserving the dependency of overlapping h-day
returns; resampling never crosses a segment boundary) and Holm-Bonferroni
multiplicity control across the whole frontier. A ``(horizon, profile)``
candidate is economically admissible only when both its base and stress
adjusted lower growth bounds are strictly positive. At most one primary
``(horizon, profile)`` pair is selected (maximum stress-cost adjusted lower
growth, ties preferring the shorter horizon then the lexicographically
smaller profile id).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 200
_BOUND_TOLERANCE = 1e-12


def _frontier_key(horizon_sessions: int, profile_id: str) -> tuple[int, str]:
    return (horizon_sessions, profile_id)


@dataclass(frozen=True, slots=True)
class HorizonOOFEvidence:
    """One ``(horizon, profile)`` candidate's base/stress vintage log-growth evidence.

    ``base_log_growth`` and ``stress_log_growth`` are parallel per-vintage log
    growth series ``log1p(r)`` over the evaluated vintages of the common OOF
    calendar (each decision session is one overlapping holding vintage);
    ``cohort_segment_ids`` is the segment identity of each vintage so
    resampling never crosses a segment boundary. ``profile_id`` records which
    pre-registered policy profile produced the evidence and
    ``model_family`` which family produced the OOF scores. The vintage counts
    publish the evaluated/active/partial/missing decomposition and
    ``fold_rank_ics`` the session-mean Rank-IC per usable fold.
    """

    horizon_sessions: int
    profile_id: str
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
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
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
    """Immutable outcome of one Holm-adjusted frontier-discovery run.

    ``primary_horizon_sessions`` and ``primary_profile_id`` are ``None`` when
    no ``(horizon, profile)`` candidate is economically admissible (a normal
    ``NO_TRADE`` outcome). ``adjusted_lower_growth`` maps a candidate keyed by
    ``(horizon_sessions, profile_id)`` to its per-vintage log-growth lower
    means at the candidate's Holm threshold for the ``base`` and ``stress``
    cost paths; the annualized ``LB_CAGR`` is derived by the caller as
    ``expm1(annualization_sessions * lower_growth)``. ``rankability_reason``
    records why the linear screen may not run a nonlinear challenger.
    """

    primary_horizon_sessions: int | None
    primary_profile_id: str | None
    adjusted_lower_growth: dict[tuple[int, str], dict[str, float]]
    base_p_values: dict[tuple[int, str], float]
    stress_p_values: dict[tuple[int, str], float]
    base_holm_thresholds: dict[tuple[int, str], float]
    stress_holm_thresholds: dict[tuple[int, str], float]
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

    @property
    def selected_profile_id(self) -> str | None:
        return self.primary_profile_id

    def to_json(self) -> dict[str, object]:
        def keyed(mapping: dict[tuple[int, str], float]) -> dict[str, float]:
            return {
                f"{horizon}:{profile}": float(value)
                for (horizon, profile), value in sorted(mapping.items())
            }

        return {
            "primary_horizon_sessions": self.primary_horizon_sessions,
            "primary_profile_id": self.primary_profile_id,
            "adjusted_lower_growth": {
                f"{horizon}:{profile}": dict(path)
                for (horizon, profile), path in sorted(
                    self.adjusted_lower_growth.items()
                )
            },
            "base_p_values": keyed(self.base_p_values),
            "stress_p_values": keyed(self.stress_p_values),
            "base_holm_thresholds": keyed(self.base_holm_thresholds),
            "stress_holm_thresholds": keyed(self.stress_holm_thresholds),
            "selection_reasons": list(self.selection_reasons),
            "rankability_reason": self.rankability_reason,
            "rank_ic_lower_bound": self.rank_ic_lower_bound,
        }

    @property
    def evidence_hash(self) -> str:
        payload = sha256()
        payload.update(f"{self.primary_horizon_sessions}".encode())
        for (horizon, profile), path in sorted(self.adjusted_lower_growth.items()):
            payload.update(
                f"{horizon}:{profile}:{path.get('base', 0.0):.17g}:"
                f"{path.get('stress', 0.0):.17g};".encode()
            )
        return payload.hexdigest()


def _segment_block_length(n_cohorts: int) -> int:
    """Deterministic moving-block length ``ceil(n ** (1/3))`` in session units."""
    if n_cohorts < 1:
        raise ValueError("segment must contain at least one cohort")
    block: int = ceil(float(n_cohorts) ** (1.0 / 3.0))
    return max(1, block)


@dataclass(frozen=True, slots=True)
class _CohortBootstrap:
    """One candidate cost path's session-unit bootstrap summary."""

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
    min_block_length: int = 1,
) -> _CohortBootstrap | None:
    """One-sided centered moving-block bootstrap in per-vintage session units.

    Resampling is performed independently within each segment with block length
    ``max(ceil(n_s ** (1/3)), min_block_length)`` sessions and never crosses a
    segment boundary; the block length is at least ``min_block_length`` (the
    candidate horizon) so the dependency of overlapping h-day returns is
    preserved. The per-segment resample means are pooled weighted by vintage
    count. Returns ``None`` (inadmissible) when any segment has fewer than two
    resampling blocks.
    """
    by_segment: dict[int, list[float]] = {}
    for segment, value in zip(cohort_segment_ids, log_growth, strict=True):
        by_segment.setdefault(int(segment), []).append(float(value))
    distributions: list[np.ndarray] = []
    weights: list[float] = []
    n_blocks_total = 0
    for segment in sorted(by_segment):
        values = np.asarray(by_segment[segment], dtype=float)
        block = max(_segment_block_length(values.size), min_block_length)
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
    bootstrap: dict[tuple[int, str], dict[str, _CohortBootstrap | None]],
    bootstrap_alpha: float,
) -> tuple[
    dict[tuple[int, str], float],
    dict[tuple[int, str], float],
    dict[tuple[int, str], float],
    list[str],
]:
    """Holm-Bonferroni control across all candidates on the least favorable path.

    The family is the candidate ``(horizon, profile)`` set; each candidate's
    combined p-value is the maximum of its base and stress centered p-values
    (the least favorable path). The hypothesis at sorted rank ``j`` is rejected
    when ``combined_p <= alpha / (m - j + 1)``. Returns the per-candidate
    combined p-values, the base and stress Holm thresholds, and rejection
    reasons.
    """
    combined: dict[tuple[int, str], float] = {}
    for candidate in candidates:
        key = _frontier_key(candidate.horizon_sessions, candidate.profile_id)
        base = bootstrap[key].get("base")
        stress = bootstrap[key].get("stress")
        if base is None or stress is None:
            combined[key] = 1.0
            continue
        combined[key] = max(base.p_value, stress.p_value)
    m = len(candidates)
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            combined[_frontier_key(candidate.horizon_sessions, candidate.profile_id)],
            candidate.horizon_sessions,
            candidate.profile_id,
        ),
    )
    base_thresholds: dict[tuple[int, str], float] = {}
    stress_thresholds: dict[tuple[int, str], float] = {}
    reasons: list[str] = []
    for rank, candidate in enumerate(ordered, start=1):
        key = _frontier_key(candidate.horizon_sessions, candidate.profile_id)
        threshold = bootstrap_alpha / (m - rank + 1)
        base_thresholds[key] = threshold
        stress_thresholds[key] = threshold
        base = bootstrap[key]["base"]
        stress = bootstrap[key]["stress"]
        if base is None or stress is None:
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                "inadmissible (fewer than two resampling blocks)"
            )
        elif combined[key] > threshold:
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"Holm p {combined[key]:.6g} > {threshold:.6g}"
            )
    return combined, base_thresholds, stress_thresholds, reasons


def select_horizons(
    evidence: tuple[HorizonOOFEvidence, ...],
    bootstrap_alpha: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> HorizonSelectionEvidence:
    """Select at most one economically admissible primary ``(horizon, profile)``.

    Selection is evidence-only: every candidate's base and stress per-vintage
    log-growth series are resampled in session units (segment-local, block
    length at least the candidate horizon, never across segment boundaries) and
    one-sided centered p-values are computed for the null ``mean(g) <= 0``.
    Holm-Bonferroni is applied across every pre-registered candidate
    ``(horizon, profile)`` pair; a pair is admissible only when both its base
    and stress adjusted lower growth are strictly positive. The primary is the
    admissible pair with the maximum stress-cost adjusted lower growth (ties
    prefer the shorter horizon then the lexicographically smaller profile id).
    ``primary_horizon_sessions``/``primary_profile_id`` are ``None`` when no
    pair is admissible (the ``NO_TRADE`` outcome).

    Args:
        evidence: pre-registered candidate ``(horizon, profile)`` pairs with
            their base/stress vintage evidence.
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

    ordered = tuple(
        sorted(evidence, key=lambda candidate: (candidate.horizon_sessions, candidate.profile_id))
    )
    bootstrap: dict[tuple[int, str], dict[str, _CohortBootstrap | None]] = {}
    for candidate in ordered:
        key = _frontier_key(candidate.horizon_sessions, candidate.profile_id)
        bootstrap[key] = {
            "base": _cohort_bootstrap(
                candidate.base_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed,
                min_block_length=candidate.horizon_sessions,
            ),
            "stress": _cohort_bootstrap(
                candidate.stress_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed + candidate.horizon_sessions,
                min_block_length=candidate.horizon_sessions,
            ),
        }

    _combined, base_thresholds, stress_thresholds, reasons = _holm_admission(
        ordered, bootstrap, bootstrap_alpha
    )

    adjusted_lower_growth: dict[tuple[int, str], dict[str, float]] = {}
    admissible: list[HorizonOOFEvidence] = []
    for candidate in ordered:
        key = _frontier_key(candidate.horizon_sessions, candidate.profile_id)
        threshold = base_thresholds[key]
        base = bootstrap[key]["base"]
        stress = bootstrap[key]["stress"]
        base_lower = 0.0
        stress_lower = 0.0
        if base is not None:
            base_lower = base.lower_mean(threshold)
        if stress is not None:
            stress_lower = stress.lower_mean(threshold)
        adjusted_lower_growth[key] = {
            "base": base_lower,
            "stress": stress_lower,
        }
        if base_lower > 0.0 and stress_lower > 0.0:
            admissible.append(candidate)
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"admissible base={base_lower:.6g} stress={stress_lower:.6g}"
            )
        else:
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"adjusted lower growth base={base_lower:.6g} "
                f"stress={stress_lower:.6g} not strictly positive"
            )

    primary_horizon: int | None = None
    primary_profile: str | None = None
    if admissible:
        primary_horizon, primary_profile = _best_primary(
            admissible, adjusted_lower_growth
        )
        reasons.append(
            f"primary=h{primary_horizon}:{primary_profile} "
            "(max stress adjusted lower growth)"
        )
    else:
        reasons.append("no candidate is economically admissible")

    base_p_values: dict[tuple[int, str], float] = {}
    stress_p_values: dict[tuple[int, str], float] = {}
    for key, path in bootstrap.items():
        base = path["base"]
        stress = path["stress"]
        base_p_values[key] = base.p_value if base is not None else 1.0
        stress_p_values[key] = stress.p_value if stress is not None else 1.0

    return HorizonSelectionEvidence(
        primary_horizon_sessions=primary_horizon,
        primary_profile_id=primary_profile,
        adjusted_lower_growth=adjusted_lower_growth,
        base_p_values=base_p_values,
        stress_p_values=stress_p_values,
        base_holm_thresholds=base_thresholds,
        stress_holm_thresholds=stress_thresholds,
        selection_reasons=tuple(reasons),
    )


def _best_primary(
    admissible: list[HorizonOOFEvidence],
    adjusted_lower_growth: dict[tuple[int, str], dict[str, float]],
) -> tuple[int, str]:
    """Maximum stress adjusted lower growth; ties prefer shorter horizon, then id."""
    best = admissible[0]
    best_stress = adjusted_lower_growth[
        _frontier_key(best.horizon_sessions, best.profile_id)
    ]["stress"]
    for candidate in admissible[1:]:
        key = _frontier_key(candidate.horizon_sessions, candidate.profile_id)
        current = adjusted_lower_growth[key]["stress"]
        if (
            current > best_stress + _BOUND_TOLERANCE
            or (
                abs(current - best_stress) <= _BOUND_TOLERANCE
                and (
                    candidate.horizon_sessions < best.horizon_sessions
                    or (
                        candidate.horizon_sessions == best.horizon_sessions
                        and candidate.profile_id < best.profile_id
                    )
                )
            )
        ):
            best = candidate
            best_stress = current
    return best.horizon_sessions, best.profile_id
