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

DEFAULT_BOOTSTRAP_RESAMPLES = 2000
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
    turnover_ratio: dict[tuple[int, int, int, str], float | None] = field(
        default_factory=dict
    )
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
        def keyed(
            mapping: Mapping[tuple[int, int, int, str], float | None],
        ) -> dict[str, float | None]:
            return {
                f"{horizon}:{cadence}:{top_k}:{profile}": (
                    None if value is None else float(value)
                )
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
    n_bootstrap: int,
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

    Raises:
        ValueError: when ``n_bootstrap * bootstrap_alpha`` is below the family
            size ``m``, because then the smallest rank-1 threshold falls under
            the k/B grid resolution of the discrete bootstrap p-value and no
            hypothesis except an exactly-zero draw could ever be admitted.
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
    minimum_resamples = ceil(m / bootstrap_alpha)
    if n_bootstrap < minimum_resamples:
        raise ValueError(
            f"n_bootstrap={n_bootstrap} is below the resolvable minimum "
            f"{minimum_resamples} for alpha={bootstrap_alpha} and family size {m}"
        )
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
    *,
    family_scope: str = "frontier",
) -> HorizonSelectionEvidence:
    """Select at most one economically admissible primary ``(H, C, K, profile)``.

    Selection is evidence-only: every candidate's base and stress per-vintage
    log-growth series are resampled in session units (segment-local, block
    length at least ``max(horizon, rebalance_frequency_sessions)`` from the
    candidate's own cadence, never across segment boundaries) and one-sided
    centered p-values are computed for the null ``mean(g) <= 0``. Holm-Bonferroni
    is applied across every pre-registered candidate ``(horizon,
    rebalance_frequency_sessions, top_k, profile_id)``; ``n_bootstrap`` must
    satisfy ``n_bootstrap >= ceil(family_size / bootstrap_alpha)`` so the rank-1
    threshold stays measurable on the discrete k/B p-value grid, otherwise a
    ``ValueError`` is raised before any bootstrap work. A candidate is
    admissible only when its base, stress, and paired lower growth are strictly
    positive and its sparse/shadow turnover ratio is at most 0.60. The primary
    is the admissible candidate with the maximum stress-cost adjusted lower
    growth (ties prefer the shorter horizon, then cadence, then top-k, then the
    lexicographically smaller profile id). ``primary_horizon_sessions``/
    ``primary_profile_id`` are ``None`` when no candidate is admissible (the
    ``NO_TRADE`` outcome).

    ``family_scope='frontier'`` (the pre-registered default) keeps the Holm
    p-value gate active. ``family_scope='route_gatekeeping'`` demotes that
    gate to published diagnostics: combined p-values and thresholds are
    computed and reported unchanged while admission requires only positive
    lower bounds and turnover discipline — the statistical burden rests on
    the strategy-level growth-route certificate and holdout certification.

    Args:
        evidence: pre-registered candidate ``(horizon, rebalance_frequency_sessions,
            top_k, profile_id)`` tuples with their base/stress vintage evidence.
        bootstrap_alpha: bootstrap alpha quantile for the lower bound and Holm
            family-wise control.
        seed: deterministic bootstrap seed.
        n_bootstrap: request-controlled moving-block bootstrap resample count;
            values below two are rejected.
        family_scope: multiplicity scope selector.

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
    if family_scope not in ("frontier", "route_gatekeeping"):
        raise ValueError(
            "family_scope must be 'frontier' or 'route_gatekeeping', "
            f"got {family_scope!r}"
        )

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
        ordered, bootstrap, bootstrap_alpha, n_bootstrap
    )

    adjusted_lower_growth: dict[tuple[int, int, int, str], dict[str, float]] = {}
    paired_lower_bounds: dict[tuple[int, int, int, str], float] = {}
    paired_p_values: dict[tuple[int, int, int, str], float] = {}
    paired_holm_thresholds: dict[tuple[int, int, int, str], float] = {}
    shadow_turnover: dict[tuple[int, int, int, str], float] = {}
    turnover_ratio: dict[tuple[int, int, int, str], float | None] = {}
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
        shadow_turnover[key] = float(candidate.shadow_turnover)
        # A missing dense shadow publishes an exempt ratio instead of a
        # fabricated denominator; the ratio gate then cannot auto-reject.
        has_shadow = candidate.shadow_turnover > 0.0
        ratio: float | None = (
            float(candidate.sparse_turnover) / float(candidate.shadow_turnover)
            if has_shadow
            else None
        )
        turnover_ratio[key] = ratio
        base_p = base.p_value if base is not None else 1.0
        stress_p = stress.p_value if stress is not None else 1.0
        paired_boot = bootstrap[key].get("paired") if has_paired else None
        paired_p = paired_boot.p_value if paired_boot is not None else 1.0
        base_ok = base is not None and base_lower > 0.0
        stress_ok = stress is not None and stress_lower > 0.0
        if family_scope == "frontier":
            # Pre-registered per-cell Holm gate; route_gatekeeping demotes it
            # to published diagnostics while lower bounds still gate.
            base_ok = base_ok and base_p <= base_threshold
            stress_ok = stress_ok and stress_p <= stress_threshold
        paired_ok = (
            (paired_lower > 0.0 and paired_p <= paired_threshold)
            if has_paired
            else True
        )
        ratio_ok = ratio is None or ratio <= 0.60
        if base_ok and stress_ok and paired_ok and ratio_ok:
            admissible.append(candidate)
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"admissible base={base_lower:.6g} stress={stress_lower:.6g} "
                f"paired={paired_lower:.6g} turnover_ratio="
                + ("exempt" if ratio is None else f"{ratio:.6g}")
            )
        else:
            reasons.append(
                f"h{candidate.horizon_sessions}:{candidate.profile_id} "
                f"rejected base={base_lower:.6g} stress={stress_lower:.6g} "
                f"paired={paired_lower:.6g} turnover_ratio="
                + ("exempt" if ratio is None else f"{ratio:.6g}")
                + f" base_p={base_p:.6g} stress_p={stress_p:.6g}"
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


GROWTH_ROUTE_VERSION = "v1"

PolicyKey = tuple[int, int, int, str]


@dataclass(frozen=True, slots=True)
class GrowthRouteEvidence:
    """Immutable stitched prequential growth route.

    ``base_log_growth``/``stress_log_growth`` are the aligned per-interval log
    growth series appended one evaluated OOF segment at a time in
    chronological order; ``segment_ids`` carries each interval's segment
    identity. ``selected_policies`` records, per segment, the
    ``(H, C, K, profile)`` policy causally selected from strictly earlier
    segments (``None`` marks a cash segment); ``interval_policies`` repeats the
    per-interval source policy (``None`` for cash intervals).
    ``benchmark_log_growth`` is an optional parallel exposure-matched
    benchmark series attached by the caller. Coverage and fills are bounded
    scalars; sparse-minus-dense growth and turnover ratio are diagnostics only.
    """

    base_log_growth: tuple[float, ...]
    stress_log_growth: tuple[float, ...]
    segment_ids: tuple[int, ...]
    selected_policies: tuple[PolicyKey | None, ...]
    interval_policies: tuple[PolicyKey | None, ...] = ()
    benchmark_log_growth: tuple[float, ...] = ()
    candidate_count: int = 0
    observed_interval_count: int = 0
    invested_interval_count: int = 0
    filled_orders: int = 0
    filled_cycle_count: int = 0
    sparse_minus_dense_lower_growth: float = 0.0
    turnover_ratio: float | None = None
    route_version: str = GROWTH_ROUTE_VERSION
    seed_policy: PolicyKey | None = None
    benchmark_reconcile_failure: str = ""

    def __post_init__(self) -> None:
        count = len(self.base_log_growth)
        if len(self.stress_log_growth) != count or len(self.segment_ids) != count:
            raise ValueError(
                "route base/stress log growth and segment ids must be parallel"
            )
        if self.interval_policies and len(self.interval_policies) != count:
            raise ValueError("interval_policies must be parallel to the route series")
        if self.benchmark_log_growth and len(self.benchmark_log_growth) != count:
            raise ValueError(
                "benchmark_log_growth must be parallel to the route series"
            )
        if not np.all(np.isfinite(self.base_log_growth)) or not np.all(
            np.isfinite(self.stress_log_growth)
        ):
            raise ValueError("route log growth must be finite")
        if self.benchmark_log_growth and not np.all(
            np.isfinite(self.benchmark_log_growth)
        ):
            raise ValueError("route benchmark log growth must be finite")
        if any(segment < 0 for segment in self.segment_ids):
            raise ValueError("route segment identity must be non-negative")
        for key in (*self.selected_policies, *self.interval_policies, self.seed_policy):
            if key is None:
                continue
            horizon, cadence, top_k, profile_id = key
            if horizon < 1 or cadence < 1 or top_k < 1 or not profile_id:
                raise ValueError(
                    "route policy keys must carry positive H/C/K and a profile id"
                )
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if self.observed_interval_count < 0 or self.invested_interval_count < 0:
            raise ValueError("route coverage counts must be non-negative")
        if self.invested_interval_count > self.observed_interval_count:
            raise ValueError(
                "invested_interval_count cannot exceed observed_interval_count"
            )
        if self.filled_orders < 0 or self.filled_cycle_count < 0:
            raise ValueError("route fill counts must be non-negative")
        if not np.isfinite(self.sparse_minus_dense_lower_growth):
            raise ValueError("sparse_minus_dense_lower_growth must be finite")
        if self.turnover_ratio is not None and (
            not np.isfinite(self.turnover_ratio) or self.turnover_ratio < 0.0
        ):
            raise ValueError("turnover_ratio must be None or a finite non-negative value")

    @property
    def invested_interval_fraction(self) -> float:
        if self.observed_interval_count <= 0:
            return 0.0
        return self.invested_interval_count / self.observed_interval_count

    @property
    def base_log_growth_minus_benchmark(self) -> tuple[float, ...]:
        """Parallel matched-excess series; empty when no benchmark is attached."""
        if not self.benchmark_log_growth:
            return ()
        return tuple(
            float(base) - float(benchmark)
            for base, benchmark in zip(
                self.base_log_growth, self.benchmark_log_growth, strict=True
            )
        )


def stitch_prequential_growth_route(
    candidates: tuple[HorizonOOFEvidence, ...],
    bootstrap_alpha: float,
    seed: int,
    n_bootstrap: int,
    *,
    seed_policy: PolicyKey | None = None,
    benchmarks_by_key: Mapping[PolicyKey, tuple[float, ...]] | None = None,
) -> GrowthRouteEvidence:
    """Causally stitch one strategy-level prequential growth route.

    For every outer OOF segment ``s`` (the sorted union of candidate segment
    identities) each candidate is bootstrapped on its slice of complete
    segments strictly below ``s`` with the shared moving-block kernel. A
    candidate is admissible for ``s`` only when both its sliced base and
    stress lower growth bounds are strictly positive; sparse-minus-dense and
    turnover diagnostics never gate selection. The admissible candidate with
    the maximum stress lower growth wins (deterministic ``(H, C, K, profile)``
    tie-breaks), and exactly that policy's current-segment values are appended.
    When no earlier candidate qualifies, ``seed_policy`` — declared from the
    request contract alone, never from outcomes — invests the segment through
    its matching candidate instead of forcing cash; an admissible selection
    always outranks the seed. Segment ``s``'s own returns never influence
    which policy serves it.

    When ``benchmarks_by_key`` is supplied, selection and admissibility
    operate on the exposure-matched excess series (base minus benchmark,
    stress minus benchmark) so the route certifies relative performance;
    candidates without a parallel benchmark series are skipped. The chosen
    candidate's benchmark slice is appended in parallel to base/stress and
    the route is tagged ``v1-excess``/``v2-excess``.

    Args:
        candidates: the discovery frontier's per-candidate vintage evidence.
        bootstrap_alpha: one-sided bootstrap quantile for the lower bounds.
        seed: deterministic bootstrap seed (per-path offsets follow the
            existing horizon-selection convention).
        n_bootstrap: resample count; the route is one strategy-level
            hypothesis, so at least ``ceil(1 / bootstrap_alpha)`` resamples
            are required to resolve its significance.
        seed_policy: optional ex-ante ``(H, C, K, profile)`` policy used only
            where no earlier evidence is admissible; must match a candidate.
        benchmarks_by_key: optional per-candidate exposure-matched benchmark
            series parallel to each candidate's ``base_log_growth``; enables
            excess-scoped selection when supplied.

    Returns:
        The immutable :class:`GrowthRouteEvidence` stitched route, tagged
        version ``v2`` (or ``v2-excess`` in excess scope) when a seed was
        spliced at least once, else ``v1`` (or ``v1-excess``).

    Raises:
        ValueError: when ``candidates`` is empty, ``bootstrap_alpha`` or
            ``n_bootstrap`` violates resolvability, or ``seed_policy``
            matches no discovery candidate.
    """
    if not candidates:
        raise ValueError(
            "stitch_prequential_growth_route requires at least one candidate"
        )
    if not 0.0 < bootstrap_alpha < 1.0:
        raise ValueError("bootstrap_alpha must be in (0, 1)")
    minimum_resamples = ceil(1.0 / bootstrap_alpha)
    if n_bootstrap < max(2, minimum_resamples):
        raise ValueError(
            f"n_bootstrap={n_bootstrap} is below the resolvable minimum "
            f"{minimum_resamples} for alpha={bootstrap_alpha}"
        )

    ordered = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.horizon_sessions,
                candidate.rebalance_frequency_sessions,
                candidate.top_k,
                candidate.profile_id,
            ),
        )
    )
    seed_by_key: dict[PolicyKey, HorizonOOFEvidence] = {
        (
            int(candidate.horizon_sessions),
            int(candidate.rebalance_frequency_sessions),
            int(candidate.top_k),
            str(candidate.profile_id),
        ): candidate
        for candidate in ordered
    }
    if seed_policy is not None and seed_policy not in seed_by_key:
        raise ValueError(
            f"seed policy {seed_policy!r} matches no discovery candidate; "
            "refusing to splice an uninvestable seed"
        )
    segments = sorted(
        {int(seg) for candidate in ordered for seg in candidate.cohort_segment_ids}
    )

    benchmarks: dict[PolicyKey, tuple[float, ...]] | None = None
    if benchmarks_by_key is not None:
        benchmarks = {}
        for candidate in ordered:
            candidate_key = _frontier_key(
                candidate.horizon_sessions,
                candidate.rebalance_frequency_sessions,
                candidate.top_k,
                candidate.profile_id,
            )
            series = benchmarks_by_key.get(candidate_key)
            if series is None:
                continue
            if len(series) != len(candidate.base_log_growth):
                continue
            benchmarks[candidate_key] = tuple(float(value) for value in series)

    base_out: list[float] = []
    stress_out: list[float] = []
    benchmark_out: list[float] = []
    segment_out: list[int] = []
    interval_policy_out: list[PolicyKey | None] = []
    selected_policies: list[PolicyKey | None] = []
    paired_deltas: list[float] = []
    turnover_ratios: list[float] = []
    seed_spliced = False

    def _slice_indices(candidate: HorizonOOFEvidence, upper: int) -> list[int]:
        return [
            index
            for index, segment in enumerate(candidate.cohort_segment_ids)
            if segment < upper
        ]

    for segment in segments:
        chosen: HorizonOOFEvidence | None = None
        chosen_stress_lower = -float("inf")
        for candidate in ordered:
            prior_indices = _slice_indices(candidate, segment)
            if not prior_indices:
                continue
            candidate_key = _frontier_key(
                candidate.horizon_sessions,
                candidate.rebalance_frequency_sessions,
                candidate.top_k,
                candidate.profile_id,
            )
            if benchmarks is not None:
                bench = benchmarks.get(candidate_key)
                if bench is None:
                    continue
                base_values = tuple(
                    float(candidate.base_log_growth[index]) - bench[index]
                    for index in prior_indices
                )
                stress_values = tuple(
                    float(candidate.stress_log_growth[index]) - bench[index]
                    for index in prior_indices
                )
            else:
                base_values = tuple(
                    float(candidate.base_log_growth[index])
                    for index in prior_indices
                )
                stress_values = tuple(
                    float(candidate.stress_log_growth[index])
                    for index in prior_indices
                )
            block_floor = max(
                candidate.horizon_sessions, candidate.rebalance_frequency_sessions
            )
            base_boot = _cohort_bootstrap(
                base_values,
                tuple(
                    int(candidate.cohort_segment_ids[index]) for index in prior_indices
                ),
                n_bootstrap,
                seed + candidate.horizon_sessions,
                min_block_length=block_floor,
            )
            stress_boot = _cohort_bootstrap(
                stress_values,
                tuple(
                    int(candidate.cohort_segment_ids[index]) for index in prior_indices
                ),
                n_bootstrap,
                seed + 2 * candidate.horizon_sessions,
                min_block_length=block_floor,
            )
            if base_boot is None or stress_boot is None:
                continue
            base_lower = base_boot.lower_mean(bootstrap_alpha)
            stress_lower = stress_boot.lower_mean(bootstrap_alpha)
            if base_lower <= 0.0 or stress_lower <= 0.0:
                continue
            if (
                stress_lower > chosen_stress_lower + _BOUND_TOLERANCE
                or chosen is None
            ):
                chosen = candidate
                chosen_stress_lower = stress_lower

        if chosen is None and seed_policy is not None:
            # Ex-ante seed: declared from the request contract alone, it only
            # fills segments no earlier evidence admits and never overrides
            # an admissible selection.
            chosen = seed_by_key[seed_policy]
            seed_spliced = True

        selected_policies.append(
            None
            if chosen is None
            else (
                chosen.horizon_sessions,
                chosen.rebalance_frequency_sessions,
                chosen.top_k,
                chosen.profile_id,
            )
        )
        if chosen is None:
            # Every segment id originates from some candidate's calendar, so
            # at least one candidate covers a cash segment.
            cash_length = max(
                (
                    sum(
                        1
                        for seg in candidate.cohort_segment_ids
                        if int(seg) == segment
                    )
                    for candidate in ordered
                    if any(int(seg) == segment for seg in candidate.cohort_segment_ids)
                ),
                default=0,
            )
            base_out.extend(0.0 for _ in range(cash_length))
            stress_out.extend(0.0 for _ in range(cash_length))
            if benchmarks is not None:
                benchmark_out.extend(0.0 for _ in range(cash_length))
            segment_out.extend([segment] * cash_length)
            interval_policy_out.extend([None] * cash_length)
            continue

        key: PolicyKey = (
            chosen.horizon_sessions,
            chosen.rebalance_frequency_sessions,
            chosen.top_k,
            chosen.profile_id,
        )
        current_indices = [
            index
            for index, seg in enumerate(chosen.cohort_segment_ids)
            if int(seg) == segment
        ]
        base_out.extend(float(chosen.base_log_growth[index]) for index in current_indices)
        stress_out.extend(
            float(chosen.stress_log_growth[index]) for index in current_indices
        )
        if benchmarks is not None:
            bench = benchmarks.get(key, tuple(0.0 for _ in chosen.base_log_growth))
            benchmark_out.extend(bench[index] for index in current_indices)
        segment_out.extend([segment] * len(current_indices))
        interval_policy_out.extend([key] * len(current_indices))

        if chosen.paired_stress_log_growth:
            paired_boot = _cohort_bootstrap(
                tuple(
                    float(chosen.paired_stress_log_growth[index])
                    for index in current_indices
                ),
                tuple(int(chosen.cohort_segment_ids[index]) for index in current_indices),
                n_bootstrap,
                seed + 3 * chosen.horizon_sessions,
                min_block_length=max(
                    chosen.horizon_sessions, chosen.rebalance_frequency_sessions
                ),
            )
            if paired_boot is not None:
                paired_deltas.append(paired_boot.lower_mean(bootstrap_alpha))
        if chosen.shadow_turnover > 0.0:
            turnover_ratios.append(
                float(chosen.sparse_turnover) / float(chosen.shadow_turnover)
            )

    if benchmarks is not None:
        route_version = "v2-excess" if seed_spliced else "v1-excess"
    else:
        route_version = "v2" if seed_spliced else GROWTH_ROUTE_VERSION
    return GrowthRouteEvidence(
        base_log_growth=tuple(base_out),
        stress_log_growth=tuple(stress_out),
        segment_ids=tuple(segment_out),
        selected_policies=tuple(selected_policies),
        interval_policies=tuple(interval_policy_out),
        benchmark_log_growth=(
            tuple(benchmark_out) if benchmarks is not None else ()
        ),
        candidate_count=len(ordered),
        observed_interval_count=len(base_out),
        invested_interval_count=sum(
            1
            for base_value, stress_value in zip(
                base_out, stress_out, strict=True
            )
            if base_value != 0.0 or stress_value != 0.0
        ),
        sparse_minus_dense_lower_growth=(
            float(np.mean(paired_deltas)) if paired_deltas else 0.0
        ),
        turnover_ratio=(
            float(np.mean(turnover_ratios)) if turnover_ratios else None
        ),
        route_version=route_version,
        seed_policy=seed_policy,
    )
