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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil

import numpy as np

from src.stocks.research.bootstrap import pooled_segment_bootstrap_means

DEFAULT_BOOTSTRAP_RESAMPLES = 200
_BOUND_TOLERANCE = 1e-12


def _frontier_key(
    horizon_sessions: int,
    rebalance_frequency_sessions: int,
    top_k: int,
    profile_id: str,
) -> tuple[int, int, int, str]:
    return (horizon_sessions, rebalance_frequency_sessions, top_k, profile_id)


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
    rebalance_frequency_sessions: int = 5
    top_k: int = 20
    paired_stress_log_growth: tuple[float, ...] = ()
    sparse_turnover: float = 0.0
    shadow_turnover: float = 0.0
    reasons: tuple[str, ...] = ()
    unresolved_outcome_counts: tuple[tuple[str, int], ...] = ()
    blocked_vintage_count: int = 0

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
        if self.paired_stress_log_growth and len(self.paired_stress_log_growth) != len(
            self.base_log_growth
        ):
            raise ValueError(
                "paired_stress_log_growth must be parallel to the log growth series"
            )
        if self.paired_stress_log_growth and not np.all(
            np.isfinite(self.paired_stress_log_growth)
        ):
            raise ValueError("paired stress log growth must be finite")
        if any(segment < 0 for segment in self.cohort_segment_ids):
            raise ValueError("cohort segment identity must be a non-negative integer")
        if not np.isfinite(self.sparse_turnover) or self.sparse_turnover < 0.0:
            raise ValueError("sparse_turnover must be a finite non-negative value")
        if not np.isfinite(self.shadow_turnover) or self.shadow_turnover < 0.0:
            raise ValueError("shadow_turnover must be a finite non-negative value")
        if not self.model_family:
            raise ValueError("model_family must be non-empty")
        if self.blocked_vintage_count < 0:
            raise ValueError("blocked_vintage_count must be non-negative")
        if self.rebalance_frequency_sessions < 1:
            raise ValueError("rebalance_frequency_sessions must be a positive session count")
        if self.top_k < 1:
            raise ValueError("top_k must be a positive session count")
        seen: set[str] = set()
        for state, count in self.unresolved_outcome_counts:
            if not state:
                raise ValueError("unresolved outcome state must be non-empty")
            if count < 0:
                raise ValueError("unresolved outcome count must be non-negative")
            if state in seen:
                raise ValueError(f"duplicate unresolved outcome state {state!r}")
            seen.add(state)


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
    adjusted_lower_growth: dict[tuple[int, int, int, str], dict[str, float]]
    base_p_values: dict[tuple[int, int, int, str], float]
    stress_p_values: dict[tuple[int, int, int, str], float]
    base_holm_thresholds: dict[tuple[int, int, int, str], float]
    stress_holm_thresholds: dict[tuple[int, int, int, str], float]
    primary_rebalance_frequency_sessions: int | None = None
    primary_top_k: int | None = None
    paired_lower_bounds: dict[tuple[int, int, int, str], float] = field(default_factory=dict)
    paired_p_values: dict[tuple[int, int, int, str], float] = field(default_factory=dict)
    paired_holm_thresholds: dict[tuple[int, int, int, str], float] = field(default_factory=dict)
    shadow_turnover: dict[tuple[int, int, int, str], float] = field(default_factory=dict)
    turnover_ratio: dict[tuple[int, int, int, str], float] = field(default_factory=dict)
    selection_reasons: tuple[str, ...] = ()
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
        def keyed(mapping: dict[tuple[int, int, int, str], float]) -> dict[str, float]:
            return {
                f"{horizon}:{cadence}:{top_k}:{profile}": float(value)
                for (horizon, cadence, top_k, profile), value in sorted(
                    mapping.items()
                )
            }

        return {
            "primary_horizon_sessions": self.primary_horizon_sessions,
            "primary_profile_id": self.primary_profile_id,
            "primary_rebalance_frequency_sessions": self.primary_rebalance_frequency_sessions,
            "primary_top_k": self.primary_top_k,
            "adjusted_lower_growth": {
                f"{horizon}:{cadence}:{top_k}:{profile}": dict(path)
                for (horizon, cadence, top_k, profile), path in sorted(
                    self.adjusted_lower_growth.items()
                )
            },
            "base_p_values": keyed(self.base_p_values),
            "stress_p_values": keyed(self.stress_p_values),
            "base_holm_thresholds": keyed(self.base_holm_thresholds),
            "stress_holm_thresholds": keyed(self.stress_holm_thresholds),
            "paired_lower_bounds": keyed(self.paired_lower_bounds),
            "paired_p_values": keyed(self.paired_p_values),
            "paired_holm_thresholds": keyed(self.paired_holm_thresholds),
            "shadow_turnover": keyed(self.shadow_turnover),
            "turnover_ratio": keyed(self.turnover_ratio),
            "selection_reasons": list(self.selection_reasons),
            "rankability_reason": self.rankability_reason,
            "rank_ic_lower_bound": self.rank_ic_lower_bound,
        }

    @property
    def evidence_hash(self) -> str:
        payload = sha256()
        payload.update(f"{self.primary_horizon_sessions}".encode())
        for (horizon, cadence, top_k, profile), path in sorted(
            self.adjusted_lower_growth.items()
        ):
            payload.update(
                f"{horizon}:{cadence}:{top_k}:{profile}:"
                f"{path.get('base', 0.0):.17g}:"
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
    count through the shared bounded-workspace primitive. Returns ``None``
    (inadmissible) when any segment has fewer than two resampling blocks.
    """
    by_segment: dict[int, list[float]] = {}
    for segment, value in zip(cohort_segment_ids, log_growth, strict=True):
        by_segment.setdefault(int(segment), []).append(float(value))
    ordered_ids = sorted(by_segment)
    segments = tuple(
        np.asarray(by_segment[segment], dtype=float) for segment in ordered_ids
    )
    n_blocks_total = 0
    for values in segments:
        block = max(_segment_block_length(values.size), min_block_length)
        if int(np.ceil(values.size / block)) < 2:
            return None
        n_blocks_total += int(np.ceil(values.size / block))
    if not segments:
        return None
    boot_means = pooled_segment_bootstrap_means(
        segments, min_block_length, n_bootstrap, seed
    )
    observed = float(sum(log_growth) / len(log_growth))
    centered_p_value = float(np.mean(boot_means >= 2.0 * observed))
    return _CohortBootstrap(
        observed_mean=observed,
        boot_means=boot_means,
        p_value=centered_p_value,
        n_blocks_total=n_blocks_total,
    )


def _holm_admission(
    candidates: tuple[HorizonOOFEvidence, ...],
    bootstrap: dict[tuple[int, int, int, str], dict[str, _CohortBootstrap | None]],
    bootstrap_alpha: float,
) -> tuple[
    dict[tuple[int, int, int, str], float],
    dict[tuple[int, int, int, str], float],
    dict[tuple[int, int, int, str], float],
    dict[tuple[int, int, int, str], float],
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
    combined: dict[tuple[int, int, int, str], float] = {}
    has_paired = any("paired" in path for path in bootstrap.values())
    hypotheses: list[tuple[float, tuple[int, int, int, str], str]] = []
    for candidate in candidates:
        key = _frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    )
        base = bootstrap[key].get("base")
        stress = bootstrap[key].get("stress")
        paired = bootstrap[key].get("paired")
        if base is None or stress is None:
            combined[key] = 1.0
            hypotheses.extend(((1.0, key, "base"), (1.0, key, "stress")))
            if paired is not None:
                hypotheses.append((1.0, key, "paired"))
            continue
        combined[key] = max(base.p_value, stress.p_value)
        hypotheses.extend(((base.p_value, key, "base"), (stress.p_value, key, "stress")))
        if paired is not None:
            combined[key] = max(combined[key], paired.p_value)
            hypotheses.append((paired.p_value, key, "paired"))
    if not has_paired:
        hypotheses = [
            (combined[_frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    )],
             _frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    ), "combined")
            for candidate in candidates
        ]
    hypotheses.sort(
        key=lambda item: (
            item[0], item[1][0], item[1][1], item[1][2], item[1][3], item[2]
        )
    )
    m = len(hypotheses)
    base_thresholds: dict[tuple[int, int, int, str], float] = {}
    stress_thresholds: dict[tuple[int, int, int, str], float] = {}
    paired_thresholds: dict[tuple[int, int, int, str], float] = {}
    reasons: list[str] = []
    for rank, (p_value, key, path_name) in enumerate(hypotheses, start=1):
        threshold = bootstrap_alpha / (m - rank + 1)
        if path_name in ("combined", "base"):
            base_thresholds[key] = threshold
        if path_name in ("combined", "stress"):
            stress_thresholds[key] = threshold
        if path_name == "paired":
            paired_thresholds[key] = threshold
        if p_value > threshold:
            reasons.append(
                f"h{key[0]}:c{key[1]}:k{key[2]}:{key[3]} {path_name} "
                f"Holm p {p_value:.6g} > {threshold:.6g}"
            )
    return combined, base_thresholds, stress_thresholds, paired_thresholds, reasons


def select_horizons(
    evidence: tuple[HorizonOOFEvidence, ...],
    bootstrap_alpha: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> HorizonSelectionEvidence:
    """Select at most one economically admissible primary ``(H, C, K, profile)``.

    Selection is evidence-only: every candidate's base and stress per-vintage
    log-growth series are resampled in session units (segment-local, block
    length at least ``max(horizon, rebalance_frequency_sessions)`` from the
    candidate's own cadence, never across segment boundaries) and one-sided
    centered p-values are computed for the null ``mean(g) <= 0``. Holm-Bonferroni
    is applied across every pre-registered candidate ``(horizon,
    rebalance_frequency_sessions, top_k, profile_id)`` and a candidate is
    admissible only when its base, stress, and paired lower growth are strictly
    positive and its sparse/shadow turnover ratio is at most 0.60. The primary
    is the admissible candidate with the maximum stress-cost adjusted lower
    growth (ties prefer the shorter horizon, then cadence, then top-k, then the
    lexicographically smaller profile id). ``primary_horizon_sessions``/
    ``primary_profile_id`` are ``None`` when no candidate is admissible (the
    ``NO_TRADE`` outcome).

    Args:
        evidence: pre-registered candidate ``(horizon, rebalance_frequency_sessions,
            top_k, profile_id)`` tuples with their base/stress vintage evidence.
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
        sorted(
            evidence,
            key=lambda candidate: (
                candidate.horizon_sessions,
                candidate.rebalance_frequency_sessions,
                candidate.top_k,
                candidate.profile_id,
            ),
        )
    )
    bootstrap: dict[tuple[int, int, int, str], dict[str, _CohortBootstrap | None]] = {}
    for candidate in ordered:
        key = _frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    )
        block_floor = max(
            candidate.horizon_sessions, candidate.rebalance_frequency_sessions
        )
        path: dict[str, _CohortBootstrap | None] = {
            "base": _cohort_bootstrap(
                candidate.base_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed,
                min_block_length=block_floor,
            ),
            "stress": _cohort_bootstrap(
                candidate.stress_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed + candidate.horizon_sessions,
                min_block_length=block_floor,
            ),
        }
        if candidate.paired_stress_log_growth:
            path["paired"] = _cohort_bootstrap(
                candidate.paired_stress_log_growth,
                candidate.cohort_segment_ids,
                n_bootstrap,
                seed + 2 * candidate.horizon_sessions,
                min_block_length=block_floor,
            )
        bootstrap[key] = path

    _combined, base_thresholds, stress_thresholds, paired_thresholds, reasons = _holm_admission(
        ordered, bootstrap, bootstrap_alpha
    )

    adjusted_lower_growth: dict[tuple[int, int, int, str], dict[str, float]] = {}
    paired_lower_bounds: dict[tuple[int, int, int, str], float] = {}
    paired_p_values: dict[tuple[int, int, int, str], float] = {}
    paired_holm_thresholds: dict[tuple[int, int, int, str], float] = {}
    shadow_turnover: dict[tuple[int, int, int, str], float] = {}
    turnover_ratio: dict[tuple[int, int, int, str], float] = {}
    admissible: list[HorizonOOFEvidence] = []
    for candidate in ordered:
        key = _frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    )
        base_threshold = base_thresholds[key]
        stress_threshold = stress_thresholds[key]
        paired_threshold = paired_thresholds.get(key, stress_threshold)
        base = bootstrap[key]["base"]
        stress = bootstrap[key]["stress"]
        base_lower = base.lower_mean(base_threshold) if base is not None else 0.0
        stress_lower = stress.lower_mean(stress_threshold) if stress is not None else 0.0
        adjusted_lower_growth[key] = {
            "base": base_lower,
            "stress": stress_lower,
        }
        paired_lower = 0.0
        has_paired = "paired" in bootstrap[key]
        if has_paired:
            paired_boot = bootstrap[key]["paired"]
            paired_lower = paired_boot.lower_mean(paired_threshold) if paired_boot is not None else 0.0
            paired_lower_bounds[key] = paired_lower
            paired_p_values[key] = paired_boot.p_value if paired_boot is not None else 1.0
            paired_holm_thresholds[key] = paired_threshold
        sh_turnover = max(candidate.shadow_turnover, 1e-12)
        shadow_turnover[key] = float(candidate.shadow_turnover)
        ratio = float(candidate.sparse_turnover) / sh_turnover
        turnover_ratio[key] = ratio
        base_p = base.p_value if base is not None else 1.0
        stress_p = stress.p_value if stress is not None else 1.0
        paired_boot = bootstrap[key].get("paired") if has_paired else None
        paired_p = paired_boot.p_value if paired_boot is not None else 1.0
        base_ok = (
            base is not None
            and base_p <= base_threshold
            and base_lower > 0.0
        )
        stress_ok = (
            stress is not None
            and stress_p <= stress_threshold
            and stress_lower > 0.0
        )
        paired_ok = (
            (paired_lower > 0.0 and paired_p <= paired_threshold)
            if has_paired
            else True
        )
        if base_ok and stress_ok and paired_ok and ratio <= 0.60:
            admissible.append(candidate)
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"admissible base={base_lower:.6g} stress={stress_lower:.6g} "
                f"paired={paired_lower:.6g} turnover_ratio={ratio:.6g}"
            )
        else:
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"rejected base={base_lower:.6g} stress={stress_lower:.6g} "
                f"paired={paired_lower:.6g} turnover_ratio={ratio:.6g} "
                f"base_p={base_p:.6g} stress_p={stress_p:.6g}"
            )

    primary_horizon: int | None = None
    primary_profile: str | None = None
    primary_cadence: int | None = None
    primary_topk: int | None = None
    if admissible:
        primary_horizon, primary_cadence, primary_topk, primary_profile = _best_primary(
            admissible, adjusted_lower_growth
        )
        selected_candidate = next(
            c for c in admissible
            if (
                c.horizon_sessions == primary_horizon
                and c.rebalance_frequency_sessions == primary_cadence
                and c.top_k == primary_topk
                and c.profile_id == primary_profile
            )
        )
        primary_cadence = selected_candidate.rebalance_frequency_sessions
        primary_topk = selected_candidate.top_k
        reasons.append(
            f"primary=h{primary_horizon}:{primary_profile} "
            f"c={primary_cadence} k={primary_topk} "
            "(max stress adjusted lower growth)"
        )
    else:
        reasons.append("no candidate is economically admissible")

    base_p_values: dict[tuple[int, int, int, str], float] = {}
    stress_p_values: dict[tuple[int, int, int, str], float] = {}
    for key, path in bootstrap.items():
        base = path["base"]
        stress = path["stress"]
        base_p_values[key] = base.p_value if base is not None else 1.0
        stress_p_values[key] = stress.p_value if stress is not None else 1.0

    return HorizonSelectionEvidence(
        primary_horizon_sessions=primary_horizon,
        primary_profile_id=primary_profile,
        primary_rebalance_frequency_sessions=primary_cadence,
        primary_top_k=primary_topk,
        adjusted_lower_growth=adjusted_lower_growth,
        base_p_values=base_p_values,
        stress_p_values=stress_p_values,
        base_holm_thresholds=base_thresholds,
        stress_holm_thresholds=stress_thresholds,
        paired_lower_bounds=paired_lower_bounds,
        paired_p_values=paired_p_values,
        paired_holm_thresholds=paired_holm_thresholds,
        shadow_turnover=shadow_turnover,
        turnover_ratio=turnover_ratio,
        selection_reasons=tuple(reasons),
    )


def _best_primary(
    admissible: list[HorizonOOFEvidence],
    adjusted_lower_growth: dict[tuple[int, int, int, str], dict[str, float]],
) -> tuple[int, int, int, str]:
    """Maximum stress adjusted lower growth; ties prefer shorter horizon, then id."""
    best = admissible[0]
    best_stress = adjusted_lower_growth[
        _frontier_key(
            best.horizon_sessions,
            best.rebalance_frequency_sessions,
            best.top_k,
            best.profile_id,
        )
    ]["stress"]
    for candidate in admissible[1:]:
        key = _frontier_key(
        candidate.horizon_sessions,
        candidate.rebalance_frequency_sessions,
        candidate.top_k,
        candidate.profile_id,
    )
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
    return (
        best.horizon_sessions,
        best.rebalance_frequency_sessions,
        best.top_k,
        best.profile_id,
    )


@dataclass(frozen=True, slots=True)
class PrequentialEconomicEvidence:
    """Outcome of prequential (causal) policy selection across segments.

    ``segment_policies`` maps each segment index to the evidence that
    determined its policy (``None`` means cash).  Policies are selected only
    from earlier segments; segment 0 always uses a predeclared seed policy
    (or cash when no earlier evidence is admissible).
    """

    segment_policies: dict[int, HorizonOOFEvidence | None]


def minimum_resolvable_bootstrap_count(
    path_hypothesis_count: int, alpha: float
) -> int:
    """Minimum bootstrap resample count to resolve the smallest Holm threshold.

    The smallest threshold is ``alpha / path_hypothesis_count``.  A bootstrap
    with ``n`` draws has p-value resolution ``1/n``.  Returns
    ``ceil(path_hypothesis_count / alpha)`` so that the smallest drawable
    p-value is at most the smallest Holm threshold.
    """
    if path_hypothesis_count < 1:
        raise ValueError("path_hypothesis_count must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    return ceil(path_hypothesis_count / alpha)


def select_prequential_execution_policy(
    evidence_by_segment: Mapping[int, Sequence[HorizonOOFEvidence]],
    bootstrap_alpha: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> PrequentialEconomicEvidence:
    """Prequential causal policy selection: outer segment uses only earlier segments.

    For each outer segment ``s``, evaluate candidate policies from segments
    ``< s`` only.  Choose by deterministic admissibility and lower excess
    growth, then replay the locked choice on ``s``.  When no earlier policy is
    admissible, segment ``s`` is cash (``None``).  The stitched prequential
    path is the single strategy-level hypothesis.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    required_bootstrap = minimum_resolvable_bootstrap_count(
        max(1, sum(len(values) for values in evidence_by_segment.values())),
        bootstrap_alpha,
    )
    if n_bootstrap < required_bootstrap:
        raise ValueError(
            f"n_bootstrap={n_bootstrap} is below minimum resolvable "
            f"count {required_bootstrap}"
        )

    sorted_segments = sorted(evidence_by_segment.keys())
    segment_policies: dict[int, HorizonOOFEvidence | None] = {}
    accumulated_evidence: list[HorizonOOFEvidence] = []

    for segment in sorted_segments:
        candidates = list(accumulated_evidence)
        if not candidates:
            segment_policies[segment] = None
        else:
            admissible = [
                c
                for c in candidates
                if all(
                    v > 0.0
                    for v in (
                        _candidate_lower_bound(c, bootstrap_alpha, n_bootstrap, seed)
                    )
                )
            ]
            if admissible:
                best = min(
                    admissible,
                    key=lambda c: (
                        -_candidate_lower_bound(c, bootstrap_alpha, n_bootstrap, seed)[1],
                        c.horizon_sessions,
                        c.profile_id,
                    ),
                )
                segment_policies[segment] = best
            else:
                segment_policies[segment] = None

        accumulated_evidence.extend(evidence_by_segment.get(segment, []))

    return PrequentialEconomicEvidence(segment_policies=segment_policies)


def _candidate_lower_bound(
    evidence: HorizonOOFEvidence,
    bootstrap_alpha: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Return (base_lower, stress_lower) for a candidate's log-growth series."""
    block_floor = max(evidence.horizon_sessions, evidence.rebalance_frequency_sessions)
    base_boot = _cohort_bootstrap(
        evidence.base_log_growth,
        evidence.cohort_segment_ids,
        n_bootstrap,
        seed,
        min_block_length=block_floor,
    )
    stress_boot = _cohort_bootstrap(
        evidence.stress_log_growth,
        evidence.cohort_segment_ids,
        n_bootstrap,
        seed + evidence.horizon_sessions,
        min_block_length=block_floor,
    )
    base_lower = base_boot.lower_mean(bootstrap_alpha) if base_boot is not None else 0.0
    stress_lower = stress_boot.lower_mean(bootstrap_alpha) if stress_boot is not None else 0.0
    return base_lower, stress_lower
