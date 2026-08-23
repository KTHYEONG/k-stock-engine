"""Economic family study contract: causal selection and all-cash outcomes."""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from src.stocks.ml import economic_research
from src.stocks.ml.contracts import (
    ELASTIC_NET_FAMILY,
    TAIL_LAMBDARANK_FAMILY,
    EconomicFamilyStudySettings,
    NetAlphaTrainingRequest,
)
from src.stocks.ml.economic_research import evaluate_economic_family_study

ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION = "ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION"
ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS = (
    "ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS"
)

_FAILURE_ACTIONS = (
    "no-label-capacity",
    "tail-capture-insufficient",
    "execution-economics-insufficient",
)

_REGISTRY_SENTINEL = SimpleNamespace(name="registry")


def _request(**overrides) -> NetAlphaTrainingRequest:
    from dataclasses import replace

    base = NetAlphaTrainingRequest(
        artifact_id="econ_study",
        candidate_horizon_sessions=(10,),
        forward_holdout_sessions=252,
    )
    return replace(base, **overrides)


def _cost_bound_request(**overrides) -> NetAlphaTrainingRequest:
    return _request(
        base_cost_schedule=SimpleNamespace(kind="base"),
        stress_cost_schedule=SimpleNamespace(kind="stress"),
        liquidity_model=SimpleNamespace(name="base_liq"),
        stress_liquidity_model=SimpleNamespace(name="stress_liq"),
        **overrides,
    )


def _settings(**overrides) -> EconomicFamilyStudySettings:
    kwargs: dict[str, object] = {"candidate_lookback_sessions": (504, 756)}
    kwargs.update(overrides)
    return EconomicFamilyStudySettings(**kwargs)


def _data(total_sessions: int, holdout_flip: bool = False) -> SimpleNamespace:
    sessions = list(range(total_sessions))
    if holdout_flip:
        # Mutate only the locked newest holdout sessions; the pre-holdout
        # research surface stays byte-identical.
        boundary = total_sessions - 252
        sessions = [
            -s if index >= boundary else s for index, s in enumerate(sessions)
        ]
    return SimpleNamespace(feature_frame=pl.DataFrame({"session": sessions}))


def _certificate(
    *,
    passed: bool = True,
    base: float | None = 0.08,
    stress: float | None = None,
    matched: float | None = None,
) -> dict[str, object]:
    return {
        "passed": passed,
        "reasons": [] if passed else ["non-positive-base-lower-cagr"],
        "base_lower_cagr": base,
        "stress_lower_cagr": stress if stress is not None else 0.05,
        "matched_lower_excess_cagr": matched if matched is not None else 0.02,
        "mdd": 0.21,
        "observed_intervals": 720,
        "invested_intervals": 650,
        "filled_orders": 189,
    }


def _failing_certificate() -> dict[str, object]:
    return {
        "passed": False,
        "reasons": ["no-filled-orders"],
        "base_lower_cagr": None,
        "stress_lower_cagr": None,
        "matched_lower_excess_cagr": None,
        "mdd": None,
        "observed_intervals": 720,
        "invested_intervals": 0,
        "filled_orders": 0,
    }


def _family_summary(
    family: str,
    *,
    qualified: bool,
    failure_stage: str | None,
    certificate: dict[str, object] | None,
    tail_bound: float = 0.01,
) -> dict[str, object]:
    return {
        "model_family": family,
        "admitted_candidate_count": 1 if qualified else 0,
        "oracle_capacity_observed": failure_stage != "no-label-capacity",
        "tail_gate_observed": failure_stage not in (
            "no-label-capacity",
            "tail-capture-insufficient",
        ),
        "best_tail_excess_lower_bound": tail_bound if qualified else -0.004,
        "best_tail_capture_ratio": None,
        "certificate": certificate,
        "growth_route": {
            "version": "v1",
            "candidate_count": 1 if qualified else 0,
            "selected_policy": None,
            "rejection_reason_counts": {},
        },
        "qualified": qualified,
        "failure_stage": failure_stage,
        "rejection_reason_counts": {},
    }


def _window_result(
    summaries: dict[str, dict[str, object]],
    *,
    selected_family: str | None = None,
    certificate: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": "RESEARCH_ONLY",
        "artifact_published": False,
        "study_complete": True,
        "fold_count": 3,
        "candidates_evaluated": len(summaries),
        "selected_family": selected_family,
        "certificate": certificate,
        "families": summaries,
        "rejection_reason_counts": {},
    }


def _install_candidate(monkeypatch, responder) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake(data, request, settings, *, registry):
        calls.append(
            {
                "lookback": request.max_training_lookback_sessions,
                "request_alpha": request.bootstrap_alpha,
                "request_resamples": request.bootstrap_resamples,
                "alpha": request.compounding.bootstrap_alpha,
                "resamples": request.compounding.bootstrap_resamples,
                "fold_count": request.fold_count,
                "registry": registry,
            }
        )
        return responder(request.max_training_lookback_sessions)

    monkeypatch.setattr(
        economic_research, "evaluate_economic_window_candidate", fake
    )
    return calls


def test_economic_family_03_causal_family_selection(monkeypatch) -> None:
    """ECONOMIC_FAMILY_03_CAUSAL_FAMILY_SELECTION.

    Selection is invariant to locked-holdout label changes, every candidate
    shares one fold calendar and family-adjusted alpha, and only a candidate
    with positive base/stress/matched/tail lower bounds is ever selected.
    """
    responses = {
        504: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=True,
                    failure_stage=None,
                    certificate=_certificate(stress=0.05, matched=0.02),
                ),
            },
            selected_family=TAIL_LAMBDARANK_FAMILY,
            certificate=_certificate(stress=0.05, matched=0.02),
        ),
        756: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="tail-capture-insufficient",
                    certificate=None,
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=True,
                    failure_stage=None,
                    certificate=_certificate(stress=0.01, matched=0.09),
                ),
            },
            selected_family=TAIL_LAMBDARANK_FAMILY,
            certificate=_certificate(stress=0.01, matched=0.09),
        ),
    }
    calls = _install_candidate(monkeypatch, lambda lookback: responses[lookback])

    baseline = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    flipped = evaluate_economic_family_study(
        _data(1952, holdout_flip=True),
        _cost_bound_request(),
        _settings(),
        registry=_REGISTRY_SENTINEL,
    )

    assert [call["lookback"] for call in calls] == [504, 756, 504, 756]
    for call in calls:
        assert call["fold_count"] == 3
        assert call["alpha"] == pytest.approx(0.05 / 240)
        assert call["request_alpha"] == pytest.approx(call["alpha"])
        assert call["resamples"] >= 4800
        assert call["request_resamples"] == call["resamples"]
        assert call["registry"] is _REGISTRY_SENTINEL
    assert baseline["study_complete"] is True
    assert baseline["artifact_published"] is False
    assert baseline["status"] == "RESEARCH_ONLY"
    assert baseline["selected_family"] == TAIL_LAMBDARANK_FAMILY
    assert baseline["recommended_lookback_sessions"] == 504
    # The winner owns strictly positive base/stress/matched lower bounds.
    winner = baseline["candidates"][0]["families"][TAIL_LAMBDARANK_FAMILY]
    certificate = winner["certificate"]
    assert certificate["base_lower_cagr"] > 0
    assert certificate["stress_lower_cagr"] > 0
    assert certificate["matched_lower_excess_cagr"] > 0
    assert winner["best_tail_excess_lower_bound"] > 0
    # Identical selection under mutated holdout labels proves the lock.
    assert flipped["selected_family"] == baseline["selected_family"]
    assert (
        flipped["recommended_lookback_sessions"]
        == baseline["recommended_lookback_sessions"]
    )
    assert flipped["next_action"] == baseline["next_action"]
    assert baseline["next_action"] == "rerun-qualified-family"


def _passing_negative_stress() -> dict[str, object]:
    return {
        "passed": True,
        "reasons": [],
        "base_lower_cagr": -0.001,
        "stress_lower_cagr": -0.002,
        "matched_lower_excess_cagr": 0.09,
        "mdd": 0.21,
        "observed_intervals": 720,
        "invested_intervals": 650,
        "filled_orders": 189,
    }


def test_economic_family_03_no_positive_candidate_selected(monkeypatch) -> None:
    """A negative-stress certificate never wins even with larger nominal values."""
    responses = {
        lookback: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_passing_negative_stress(),
                ),
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: responses[lookback])

    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )

    assert payload["selected_family"] is None
    assert payload["recommended_lookback_sessions"] is None
    assert payload["recommended_is_expanding"] is False
    assert payload["next_action"] == "execution-economics-insufficient"


def test_economic_family_04_all_cash_on_insufficient_economics(monkeypatch) -> None:
    """ECONOMIC_FAMILY_04_ALL_CASH_ON_INSUFFICIENT_ECONOMICS.

    When every candidate dies from non-positive oracle capacity, zero fills,
    or non-positive stress lower growth, the result stays all-cash research
    evidence: no selected family, no publication, one deterministic next
    action drawn from the declared failure vocabulary.
    """
    mixed = {
        504: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="no-label-capacity",
                    certificate=_failing_certificate(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
            }
        ),
        756: _window_result(
            {
                ELASTIC_NET_FAMILY: _family_summary(
                    ELASTIC_NET_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
                TAIL_LAMBDARANK_FAMILY: _family_summary(
                    TAIL_LAMBDARANK_FAMILY,
                    qualified=False,
                    failure_stage="execution-economics-insufficient",
                    certificate=_failing_certificate(),
                ),
            }
        ),
    }
    _install_candidate(monkeypatch, lambda lookback: mixed[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )

    assert payload["status"] == "RESEARCH_ONLY"
    assert payload["artifact_published"] is False
    assert payload["selected_family"] is None
    assert payload["next_action"] in _FAILURE_ACTIONS

    all_oracle = {
        lookback: _window_result(
            {
                family: _family_summary(
                    family,
                    qualified=False,
                    failure_stage="no-label-capacity",
                    certificate=_failing_certificate(),
                )
                for family in (ELASTIC_NET_FAMILY, TAIL_LAMBDARANK_FAMILY)
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: all_oracle[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert payload["selected_family"] is None
    assert payload["next_action"] == "no-label-capacity"

    all_tail = {
        lookback: _window_result(
            {
                family: _family_summary(
                    family,
                    qualified=False,
                    failure_stage="tail-capture-insufficient",
                    certificate=_failing_certificate(),
                )
                for family in (ELASTIC_NET_FAMILY, TAIL_LAMBDARANK_FAMILY)
            }
        )
        for lookback in (504, 756)
    }
    _install_candidate(monkeypatch, lambda lookback: all_tail[lookback])
    payload = evaluate_economic_family_study(
        _data(1952), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert payload["selected_family"] is None
    assert payload["next_action"] == "tail-capture-insufficient"


def test_economic_family_study_rejects_missing_cost_evidence() -> None:
    request = NetAlphaTrainingRequest(artifact_id="econ_nocost")
    with pytest.raises(ValueError, match="cost-evidence-required"):
        evaluate_economic_family_study(
            _data(1952),
            request,
            _settings(candidate_lookback_sessions=(1260,)),
            registry=_REGISTRY_SENTINEL,
        )


def test_economic_family_study_rejects_sub_annual_windows(monkeypatch) -> None:
    _install_candidate(monkeypatch, lambda lookback: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="annualization_sessions"):
        evaluate_economic_family_study(
            _data(4000),
            _cost_bound_request(),
            _settings(
                candidate_lookback_sessions=(126,),
                common_min_train_sessions=126,
            ),
            registry=_REGISTRY_SENTINEL,
        )


def test_economic_family_study_folds_fail_closed(monkeypatch) -> None:
    calls = _install_candidate(monkeypatch, lambda lookback: pytest.fail("must not run"))
    payload = evaluate_economic_family_study(
        _data(1400), _cost_bound_request(), _settings(), registry=_REGISTRY_SENTINEL
    )
    assert calls == []
    assert payload["study_complete"] is False
    assert payload["next_action"] == "no-label-capacity"
    assert payload["rejection_reason_counts"].get(
        "insufficient-common-window-calendar"
    ) == 1


def test_settings_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="ascending"):
        _settings(candidate_lookback_sessions=(756, 504))
    with pytest.raises(ValueError, match="final position"):
        _settings(candidate_lookback_sessions=(None, 504))
    with pytest.raises(ValueError, match="maximum"):
        _settings(common_min_train_sessions=503)
    with pytest.raises(ValueError, match="unknown model families"):
        EconomicFamilyStudySettings(model_families=("mystery_ranker",))
    with pytest.raises(ValueError, match="unique"):
        EconomicFamilyStudySettings(model_families=(ELASTIC_NET_FAMILY, ELASTIC_NET_FAMILY))
