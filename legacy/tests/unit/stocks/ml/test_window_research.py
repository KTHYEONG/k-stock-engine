"""Temporal-window study contract: common calendar, strict selection, read-only."""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from legacy.stocks.ml import window_research
from legacy.stocks.ml.contracts import NetAlphaTrainingRequest
from legacy.stocks.ml.window_research import (
    TemporalWindowStudySettings,
    classify_temporal_study,
    derive_study_fold_count,
    evaluate_temporal_window_study,
)

TEMPORAL_WINDOW_01_COMMON_CALENDAR = "TEMPORAL_WINDOW_01_COMMON_CALENDAR"
TEMPORAL_WINDOW_02_INSUFFICIENT_HISTORY = "TEMPORAL_WINDOW_02_INSUFFICIENT_HISTORY"
TEMPORAL_WINDOW_03_STRICT_RECOMMENDATION = "TEMPORAL_WINDOW_03_STRICT_RECOMMENDATION"
TEMPORAL_WINDOW_04_ALL_CASH_CLASSIFICATION = "TEMPORAL_WINDOW_04_ALL_CASH_CLASSIFICATION"

_DEFAULT_SETTINGS = TemporalWindowStudySettings()
_REGISTRY_SENTINEL = SimpleNamespace(name="registry")


def _request(**overrides) -> NetAlphaTrainingRequest:
    return NetAlphaTrainingRequest(
        artifact_id="tw_study",
        candidate_horizon_sessions=(10,),
        forward_holdout_sessions=252,
        **overrides,
    )


def _data(total_sessions: int) -> SimpleNamespace:
    frame = pl.DataFrame({"session": list(range(total_sessions))})
    return SimpleNamespace(feature_frame=frame)


def _certificate(**overrides) -> dict[str, object]:
    certificate: dict[str, object] = {
        "passed": False,
        "reasons": [],
        "base_lower_cagr": None,
        "stress_lower_cagr": None,
        "matched_lower_excess_cagr": None,
        "mdd": None,
        "observed_intervals": 720,
        "invested_intervals": 0,
        "filled_orders": 0,
    }
    certificate.update(overrides)
    return certificate


def _passing_certificate(
    stress: float, matched: float, base: float = 0.08
) -> dict[str, object]:
    return _certificate(
        passed=True,
        base_lower_cagr=base,
        stress_lower_cagr=stress,
        matched_lower_excess_cagr=matched,
        mdd=0.21,
        observed_intervals=720,
        invested_intervals=650,
        filled_orders=189,
    )


def _complete_result(certificate: dict[str, object]) -> dict[str, object]:
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": certificate,
        "growth_route": {
            "version": "v1",
            "candidate_count": 24,
            "selected_policy": None,
            "rejection_reason_counts": {},
        },
    }


def _rejected_result(reason: str) -> dict[str, object]:
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "certificate": {
            "passed": False,
            "reasons": [reason],
            "base_lower_cagr": None,
            "stress_lower_cagr": None,
            "matched_lower_excess_cagr": None,
        },
        "growth_route": {
            "version": "v1",
            "candidate_count": 0,
            "selected_policy": None,
            "rejection_reason_counts": {reason: 1},
        },
    }


def _install_evaluator(monkeypatch, responder) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake(data, request, *, registry, min_oof_train_sessions=None):
        del data
        calls.append(
            {
                "lookback": request.max_training_lookback_sessions,
                "min_oof_train_sessions": min_oof_train_sessions,
                "compounding_alpha": request.compounding.bootstrap_alpha,
                "compounding_resamples": request.compounding.bootstrap_resamples,
                "top_alpha": request.bootstrap_alpha,
                "top_resamples": request.bootstrap_resamples,
                "fold_count": request.fold_count,
                "registry": registry,
            }
        )
        return responder(request.max_training_lookback_sessions)

    monkeypatch.setattr(window_research, "evaluate_growth_route_research", fake)
    return calls


def test_derive_study_fold_count_arithmetic() -> None:
    folds = derive_study_fold_count(
        total_sessions=1952,
        forward_holdout_sessions=252,
        common_min_train_sessions=1260,
        label_horizon_sessions=11,
        embargo_sessions=5,
        annualization_sessions=252,
        min_validation_segment_sessions=126,
    )
    assert folds == 3


def test_temporal_window_01_common_calendar(monkeypatch) -> None:
    """TEMPORAL_WINDOW_01_COMMON_CALENDAR.

    All four declared candidates share one first validation boundary through
    min_oof_train_sessions=1260, receive the family-adjusted certificate alpha
    (0.05/4), and execute in declared order on an identical fold count >= 3.
    """
    calls = _install_evaluator(
        monkeypatch,
        lambda lookback: _complete_result(
            _certificate(reasons=["invested-coverage-insufficient"])
        ),
    )
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        _DEFAULT_SETTINGS,
        registry=_REGISTRY_SENTINEL,
    )

    assert [call["lookback"] for call in calls] == [504, 756, 1260, None]
    for call in calls:
        assert call["min_oof_train_sessions"] == 1260
        assert call["compounding_alpha"] == pytest.approx(0.05 / 4)
        assert call["compounding_resamples"] >= 80
        assert call["top_alpha"] == pytest.approx(0.05)
        assert call["top_resamples"] == 2000
        assert call["fold_count"] == 3
        assert call["registry"] is _REGISTRY_SENTINEL
    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["study_complete"] is True
    assert payload["common_fold_count"] == 3
    assert payload["recommended_lookback_sessions"] is None
    assert payload["recommended_is_expanding"] is False
    assert payload["next_action"] == "research-signal-objective"


def test_temporal_window_02_insufficient_history(monkeypatch) -> None:
    """TEMPORAL_WINDOW_02_INSUFFICIENT_HISTORY.

    1258 total sessions cannot afford the shared warm-up plus holdout, so the
    fold derivation fails closed before any candidate evaluator executes.
    """
    calls = _install_evaluator(
        monkeypatch, lambda lookback: pytest.fail("evaluator must not run")
    )
    derived = derive_study_fold_count(
        total_sessions=1258,
        forward_holdout_sessions=252,
        common_min_train_sessions=1260,
        label_horizon_sessions=11,
        embargo_sessions=5,
        annualization_sessions=252,
        min_validation_segment_sessions=126,
    )
    assert derived < 3

    payload = evaluate_temporal_window_study(
        _data(1258),
        _request(),
        _DEFAULT_SETTINGS,
        registry=_REGISTRY_SENTINEL,
    )

    assert calls == []
    assert payload["study_complete"] is False
    assert payload["recommended_lookback_sessions"] is None
    assert payload["recommended_is_expanding"] is False
    assert (
        payload["rejection_reason_counts"].get("insufficient-common-window-calendar")
        == 1
    )


def test_temporal_window_03_strict_recommendation_stress_winner(monkeypatch) -> None:
    """TEMPORAL_WINDOW_03_STRICT_RECOMMENDATION.

    Among fully passing candidates the larger stress lower CAGR wins; a
    candidate with any non-positive lower CAGR is excluded even with a larger
    nominal stress value.
    """
    responses = {
        504: _complete_result(_passing_certificate(0.02, 0.03)),
        756: _complete_result(_passing_certificate(0.05, 0.01)),
    }
    _install_evaluator(monkeypatch, lambda lookback: responses[lookback])
    settings = TemporalWindowStudySettings(candidate_lookback_sessions=(504, 756))

    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        settings,
        registry=_REGISTRY_SENTINEL,
    )

    assert payload["study_complete"] is True
    assert payload["recommended_lookback_sessions"] == 756
    assert payload["next_action"] == "rerun-qualified-window"

    responses = {
        504: _complete_result(_passing_certificate(-0.001, 0.09)),
        756: _complete_result(_passing_certificate(-0.002, 0.01)),
    }
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        settings,
        registry=_REGISTRY_SENTINEL,
    )
    assert payload["recommended_lookback_sessions"] is None
    assert payload["next_action"] == "research-execution-economics"


def test_temporal_window_03_tie_break_matched_then_declared(monkeypatch) -> None:
    responses = {
        504: _complete_result(_passing_certificate(0.04, 0.01)),
        756: _complete_result(_passing_certificate(0.04, 0.02)),
    }
    _install_evaluator(monkeypatch, lambda lookback: responses[lookback])
    settings = TemporalWindowStudySettings(candidate_lookback_sessions=(504, 756))
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        settings,
        registry=_REGISTRY_SENTINEL,
    )
    assert payload["recommended_lookback_sessions"] == 756

    shared = _passing_certificate(0.04, 0.02)
    responses = {504: _complete_result(shared), 756: _complete_result(shared)}
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        settings,
        registry=_REGISTRY_SENTINEL,
    )
    assert payload["recommended_lookback_sessions"] == 504


def test_temporal_window_04_all_cash_classification(monkeypatch) -> None:
    """TEMPORAL_WINDOW_04_ALL_CASH_CLASSIFICATION.

    All-cash complete routes classify as research-signal-objective even when
    matched-benchmark-missing appears; an incomplete candidate forces
    repair-economic-evidence with no recommendation.
    """
    all_cash = _certificate(
        reasons=[
            "invested-coverage-insufficient",
            "no-filled-orders",
            "non-positive-base-lower-cagr",
            "matched-benchmark-missing",
        ]
    )
    _install_evaluator(monkeypatch, lambda lookback: _complete_result(all_cash))
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        _DEFAULT_SETTINGS,
        registry=_REGISTRY_SENTINEL,
    )
    assert payload["study_complete"] is True
    assert payload["recommended_lookback_sessions"] is None
    assert payload["next_action"] == "research-signal-objective"

    classification = classify_temporal_study(
        (_complete_result(all_cash),),
        study_complete=True,
        recommended_lookback_sessions=None,
        recommended_is_expanding=False,
    )
    assert classification == "research-signal-objective"


def test_temporal_window_04_incomplete_forces_repair(monkeypatch) -> None:
    responses = {
        504: _complete_result(_certificate()),
        756: _rejected_result("insufficient-oof-calendar"),
    }
    _install_evaluator(monkeypatch, lambda lookback: responses[lookback])
    settings = TemporalWindowStudySettings(candidate_lookback_sessions=(504, 756))
    payload = evaluate_temporal_window_study(
        _data(1952),
        _request(),
        settings,
        registry=_REGISTRY_SENTINEL,
    )
    assert payload["study_complete"] is False
    assert payload["recommended_lookback_sessions"] is None
    assert payload["recommended_is_expanding"] is False
    assert payload["next_action"] == "repair-economic-evidence"


def test_settings_reject_invalid_candidates() -> None:
    for kwargs, fragment in (
        ({"candidate_lookback_sessions": ()}, "non-empty"),
        ({"candidate_lookback_sessions": (504, 504)}, "ascending"),
        ({"candidate_lookback_sessions": (756, 504)}, "ascending"),
        ({"candidate_lookback_sessions": (None, 504)}, "final position"),
        ({"common_min_train_sessions": 503}, "maximum"),
    ):
        with pytest.raises(ValueError, match=fragment):
            TemporalWindowStudySettings(**kwargs)


def test_evaluate_rejects_sub_annual_candidates(monkeypatch) -> None:
    _install_evaluator(monkeypatch, lambda lookback: pytest.fail("must not run"))
    settings = TemporalWindowStudySettings(
        candidate_lookback_sessions=(126, None),
        common_min_train_sessions=126,
    )
    with pytest.raises(ValueError, match="annualization_sessions"):
        evaluate_temporal_window_study(
            _data(4000),
            _request(),
            settings,
            registry=_REGISTRY_SENTINEL,
        )
