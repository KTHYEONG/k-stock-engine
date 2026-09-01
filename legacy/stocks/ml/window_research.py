"""Read-only temporal-window study over one common causal OOS calendar.

``evaluate_temporal_window_study`` derives one shared validation calendar
(single first-validation boundary plus segment identities), replays every
pre-registered rolling/expanding fit-window candidate sequentially through
the existing single-window evaluator, applies family-adjusted certificate
alpha across windows, and classifies exactly one deterministic next action.
The study is read-only: it never publishes an artifact, writes a ledger,
or inspects the locked forward holdout, and its payload carries bounded
scalars and normalized reason counts only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from legacy.stocks.ml.contracts import NetAlphaResearchData, NetAlphaTrainingRequest
from legacy.stocks.research.artifacts import ModelArtifactRegistry

__all__ = [
    "TemporalWindowStudySettings",
    "classify_temporal_study",
    "derive_study_fold_count",
    "evaluate_temporal_window_study",
]

from legacy.stocks.ml.training import evaluate_growth_route_research

_REPAIR_ECONOMIC_EVIDENCE = "repair-economic-evidence"
_RERUN_QUALIFIED_WINDOW = "rerun-qualified-window"
_RESEARCH_SIGNAL_OBJECTIVE = "research-signal-objective"
_RESEARCH_EXECUTION_ECONOMICS = "research-execution-economics"

_INSUFFICIENT_COMMON_WINDOW = "insufficient-common-window-calendar"
_PERIOD_SERIES_INCOMPLETE = "period-series-incomplete"
_MIN_STUDY_FOLDS = 3


@dataclass(frozen=True, slots=True)
class TemporalWindowStudySettings:
    """Immutable pre-registered candidate order and common-calendar rules.

    ``candidate_lookback_sessions`` holds strictly ascending finite session
    caps with at most one trailing ``None`` representing the expanding-window
    control. ``common_min_train_sessions`` must cover the maximum finite
    candidate so every candidate shares one first validation boundary.
    """

    candidate_lookback_sessions: tuple[int | None, ...] = (504, 756, 1260, None)
    common_min_train_sessions: int = 1260
    min_validation_segment_sessions: int = 126

    def __post_init__(self) -> None:
        if not self.candidate_lookback_sessions:
            raise ValueError("candidate_lookback_sessions must be non-empty")
        finite = [v for v in self.candidate_lookback_sessions if v is not None]
        if any(v < 1 for v in finite):
            raise ValueError("finite candidate lookbacks must be positive sessions")
        if len(set(finite)) != len(finite) or list(finite) != sorted(finite):
            raise ValueError(
                "finite candidate lookbacks must be strictly ascending and unique"
            )
        if any(
            v is None for v in tuple(self.candidate_lookback_sessions)[:-1]
        ):
            raise ValueError("expanding (None) is only permitted in the final position")
        if self.common_min_train_sessions < 1:
            raise ValueError("common_min_train_sessions must be positive")
        if self.min_validation_segment_sessions < 1:
            raise ValueError("min_validation_segment_sessions must be positive")
        if finite and self.common_min_train_sessions < max(finite):
            raise ValueError(
                "common_min_train_sessions must be at least the maximum "
                "finite candidate lookback"
            )


def derive_study_fold_count(
    *,
    total_sessions: int,
    forward_holdout_sessions: int,
    common_min_train_sessions: int,
    label_horizon_sessions: int,
    embargo_sessions: int,
    annualization_sessions: int,
    min_validation_segment_sessions: int,
) -> int:
    """Floor count of common validation segments after the shared warm-up.

    The first validation decision lands after
    ``common_min_train_sessions + label_horizon_sessions + embargo_sessions``
    source sessions; every remaining pre-holdout session is split into
    segments of at least ``max(min_validation_segment_sessions,
    4 * label_horizon_sessions)`` sessions. Fewer than three segments means
    the study fails closed before any candidate executes.
    """
    if total_sessions < 0:
        raise ValueError("total_sessions must be non-negative")
    if forward_holdout_sessions < 0:
        raise ValueError("forward_holdout_sessions must be non-negative")
    if label_horizon_sessions < 1:
        raise ValueError("label_horizon_sessions must be positive")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative")
    if annualization_sessions < 1:
        raise ValueError("annualization_sessions must be positive")
    pre_holdout_sessions = total_sessions - forward_holdout_sessions
    first_validation_start = (
        common_min_train_sessions + label_horizon_sessions + embargo_sessions
    )
    validation_sessions = max(0, pre_holdout_sessions - first_validation_start)
    segment_floor = max(min_validation_segment_sessions, 4 * label_horizon_sessions)
    return validation_sessions // segment_floor


def classify_temporal_study(
    candidate_results: tuple[Mapping[str, object], ...],
    *,
    study_complete: bool,
    recommended_lookback_sessions: int | None,
    recommended_is_expanding: bool,
) -> str:
    """Map the bounded study outcome onto exactly one deterministic action."""
    if not study_complete:
        return _REPAIR_ECONOMIC_EVIDENCE
    if recommended_lookback_sessions is not None or recommended_is_expanding:
        return _RERUN_QUALIFIED_WINDOW
    completed = [
        result for result in candidate_results if _candidate_completed(result)
    ]
    if not completed or any(
        _PERIOD_SERIES_INCOMPLETE in _reason_tokens(result)
        for result in candidate_results
    ):
        return _REPAIR_ECONOMIC_EVIDENCE
    if all(_invested_intervals(result) == 0 for result in completed):
        return _RESEARCH_SIGNAL_OBJECTIVE
    return _RESEARCH_EXECUTION_ECONOMICS


def evaluate_temporal_window_study(
    data: NetAlphaResearchData,
    request: NetAlphaTrainingRequest,
    settings: TemporalWindowStudySettings,
    *,
    registry: ModelArtifactRegistry,
) -> dict[str, object]:
    """Compare declared fit windows on one common causal OOS calendar.

    Sequential per-candidate evaluation keeps peak memory equal to one
    candidate run; each bounded projection retains certificate scalars and
    route summaries only. A recommendation requires a corrected certificate
    passing every economic predicate; otherwise the study classifies the
    next research action without ever relaxing an existing gate.
    """
    annualization = request.compounding.annualization_sessions
    candidates = settings.candidate_lookback_sessions
    finite = [v for v in candidates if v is not None]
    if any(v < annualization for v in finite):
        raise ValueError(
            "every finite candidate lookback must be at least "
            f"annualization_sessions={annualization}"
        )
    alpha_window = request.compounding.bootstrap_alpha / len(candidates)
    bootstrap_resamples = max(
        request.compounding.bootstrap_resamples, math.ceil(1.0 / alpha_window)
    )
    total_sessions = int(data.feature_frame["session"].n_unique())
    label_horizon_sessions = max(request.candidate_horizon_sessions) + 1
    fold_count = derive_study_fold_count(
        total_sessions=total_sessions,
        forward_holdout_sessions=request.forward_holdout_sessions,
        common_min_train_sessions=settings.common_min_train_sessions,
        label_horizon_sessions=label_horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        annualization_sessions=annualization,
        min_validation_segment_sessions=settings.min_validation_segment_sessions,
    )
    header: dict[str, object] = {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "adjusted_bootstrap_alpha": round(alpha_window, 12),
        "bootstrap_resamples": int(bootstrap_resamples),
        "common_fold_count": int(fold_count),
        "recommended_lookback_sessions": None,
        "recommended_is_expanding": False,
    }
    if fold_count < _MIN_STUDY_FOLDS:
        return {
            **header,
            "study_complete": False,
            "next_action": _REPAIR_ECONOMIC_EVIDENCE,
            "rejection_reason_counts": {_INSUFFICIENT_COMMON_WINDOW: 1},
            "candidates": [],
        }

    candidate_payloads: list[dict[str, object]] = []
    for lookback in candidates:
        candidate_request = replace(
            request,
            max_training_lookback_sessions=lookback,
            fold_count=fold_count,
            compounding=replace(
                request.compounding,
                bootstrap_alpha=alpha_window,
                bootstrap_resamples=bootstrap_resamples,
            ),
        )
        result = evaluate_growth_route_research(
            data,
            candidate_request,
            registry=registry,
            min_oof_train_sessions=settings.common_min_train_sessions,
        )
        candidate_payloads.append(
            {
                "training_lookback_sessions": lookback,
                "is_expanding": lookback is None,
                **dict(result),
            }
        )

    study_complete = all(
        _candidate_completed(result) for result in candidate_payloads
    )
    best_index: int | None = None
    for index, result in enumerate(candidate_payloads):
        if not _certificate_qualifies(result.get("certificate"), request):
            continue
        if best_index is None or _selection_key(result) > _selection_key(
            candidate_payloads[best_index]
        ):
            best_index = index
    recommended = (
        candidate_payloads[best_index] if best_index is not None else None
    )
    recommended_lookback = (
        recommended["training_lookback_sessions"] if recommended else None
    )
    assert recommended_lookback is None or isinstance(recommended_lookback, int)
    recommended_is_expanding = bool(recommended["is_expanding"]) if recommended else False
    next_action = classify_temporal_study(
        tuple(candidate_payloads),
        study_complete=study_complete,
        recommended_lookback_sessions=recommended_lookback,
        recommended_is_expanding=recommended_is_expanding,
    )
    if not study_complete or recommended is None:
        recommended_lookback = None
        recommended_is_expanding = False
    return {
        **header,
        "study_complete": study_complete,
        "next_action": next_action,
        "recommended_lookback_sessions": recommended_lookback,
        "recommended_is_expanding": recommended_is_expanding,
        "rejection_reason_counts": _aggregate_reasons(candidate_payloads),
        "candidates": candidate_payloads,
    }


def _certificate_qualifies(
    certificate: object, request: NetAlphaTrainingRequest
) -> bool:
    """Re-check every pre-registered economic predicate on one certificate."""
    if not isinstance(certificate, Mapping) or not bool(certificate.get("passed")):
        return False

    def _positive(name: str) -> bool:
        value = certificate.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    if not (
        _positive("base_lower_cagr")
        and _positive("stress_lower_cagr")
        and _positive("matched_lower_excess_cagr")
    ):
        return False
    filled = certificate.get("filled_orders")
    observed = _int_or_zero(certificate.get("observed_intervals"))
    invested = _int_or_zero(certificate.get("invested_intervals"))
    mdd = certificate.get("mdd")
    compounding = request.compounding
    if not isinstance(filled, int) or isinstance(filled, bool) or filled <= 0:
        return False
    if observed <= 0 or observed < compounding.min_observed_sessions:
        return False
    if invested / observed < compounding.min_active_cohort_fraction:
        return False
    mdd_ok = (
        isinstance(mdd, (int, float))
        and not isinstance(mdd, bool)
        and float(mdd) <= compounding.max_drawdown
    )
    return mdd_ok


def _selection_key(result: Mapping[str, object]) -> tuple[float, float]:
    certificate = result.get("certificate")
    stress = matched = -math.inf
    if isinstance(certificate, Mapping):
        stress = _float_or_minus_inf(certificate.get("stress_lower_cagr"))
        matched = _float_or_minus_inf(certificate.get("matched_lower_excess_cagr"))
    return (stress, matched)


def _float_or_minus_inf(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -math.inf


def _int_or_zero(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _candidate_completed(result: Mapping[str, object]) -> bool:
    route = result.get("growth_route")
    if not isinstance(route, Mapping):
        return False
    return _int_or_zero(route.get("candidate_count")) > 0


def _invested_intervals(result: Mapping[str, object]) -> int:
    for holder in ("growth_route", "certificate"):
        section = result.get(holder)
        if isinstance(section, Mapping):
            value = section.get("invested_intervals")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    return 0


def _reason_tokens(result: Mapping[str, object]) -> set[str]:
    tokens: set[str] = set()
    certificate = result.get("certificate")
    if isinstance(certificate, Mapping):
        reasons = certificate.get("reasons")
        if isinstance(reasons, (list, tuple)):
            tokens.update(str(reason) for reason in reasons)
    route = result.get("growth_route")
    if isinstance(route, Mapping):
        counts = route.get("rejection_reason_counts")
        if isinstance(counts, Mapping):
            tokens.update(str(key) for key in counts)
    return tokens


def _aggregate_reasons(
    results: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for token in sorted(_reason_tokens(result)):
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))
