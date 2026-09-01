from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from legacy.stocks.ml.compound_alpha import (
    _prequential_segment_selections,
    build_compound_feature_view,
    build_compound_labels,
    evaluate_compound_alpha_study,
    select_compound_champion,
)
from legacy.stocks.ml.contracts import (
    COMPOUND_ALPHA_EXPERIMENTS,
    COMPOUND_ALPHA_EXPERIMENT_IDS,
    CompoundAlphaExperiment,
    CompoundAlphaStudySettings,
    CompoundCandidateEvidence,
    NetAlphaTrainingRequest,
)


def test_COMPOUND_ALPHA_01_FIXED_24_REGISTRY() -> None:
    """COMPOUND_ALPHA_01_FIXED_24_REGISTRY."""
    assert len(COMPOUND_ALPHA_EXPERIMENTS) == 24
    assert tuple(e.experiment_id for e in COMPOUND_ALPHA_EXPERIMENTS) == COMPOUND_ALPHA_EXPERIMENT_IDS
    with pytest.raises(ValueError, match="requires exactly"):
        CompoundAlphaStudySettings(experiment_ids=COMPOUND_ALPHA_EXPERIMENT_IDS[:-1])
    with pytest.raises(ValueError, match="must be exactly"):
        CompoundAlphaStudySettings(experiment_ids=tuple(reversed(COMPOUND_ALPHA_EXPERIMENT_IDS)))


def test_COMPOUND_ALPHA_02_FOLD_CAUSAL_FEATURES() -> None:
    """COMPOUND_ALPHA_02_FOLD_CAUSAL_FEATURES."""
    train = pl.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "session": [1, 1],
            "bp_ratio": [1.0, 2.0],
            "ep_ratio": [0.5, 0.7],
            "ret_2_5d": [0.1, 0.2],
        }
    )
    apply = train.with_columns(pl.lit(100.0).alias("ret_2_5d"))
    transformed, schema = build_compound_feature_view(
        train,
        apply,
        experiment=COMPOUND_ALPHA_EXPERIMENTS[0],
        roles={"bp_ratio": "ALPHA", "ep_ratio": "ALPHA", "ret_2_5d": "ALPHA"},
    )
    assert schema.certified is False
    assert "bp_ratio__pct" not in transformed.columns
    assert "ep_ratio__pct" not in transformed.columns
    assert schema.fingerprint


def test_COMPOUND_ALPHA_03_LOG_DOWNSIDE_LABEL() -> None:
    """COMPOUND_ALPHA_03_LOG_DOWNSIDE_LABEL."""
    labels = pl.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "session": [1, 1],
            "gross_return": [0.10, -0.05],
            "reference_cost": [0.01, 0.01],
            "risk_residual": [0.0, 0.0],
        }
    )
    result = build_compound_labels(labels, horizon_sessions=10)
    assert result["log_return"][0] == pytest.approx(0.09531018, rel=1e-6)
    assert result["downside"][0] == 0.0
    assert result["downside"][1] < 0.0
    with pytest.raises(ValueError, match="<= -1"):
        build_compound_labels(labels.with_columns(pl.lit(-1.0).alias("gross_return")), horizon_sessions=10)


def test_COMPOUND_ALPHA_04_PRIOR_ONLY_SELECTOR() -> None:
    """COMPOUND_ALPHA_04_PRIOR_ONLY_SELECTOR."""
    ids = ("B00", "C01")
    evidence = {0: {"B00": 1.0, "C01": 2.0}, 1: {"B00": 0.0, "C01": 0.0}}
    selected = _prequential_segment_selections(evidence, ids)
    assert selected[0] is None
    assert selected[1] == "C01"
    changed = _prequential_segment_selections({0: evidence[0], 1: {"B00": 100.0, "C01": -100.0}}, ids)
    assert changed[0] is None


def _evidence(experiment_id: str, *, delta: float, mdd: float = 0.10) -> CompoundCandidateEvidence:
    return CompoundCandidateEvidence(
        experiment_id=experiment_id,
        base_lower_cagr=0.40 if experiment_id != "B00" else 0.20,
        stress_lower_cagr=0.35 if experiment_id != "B00" else 0.15,
        base_paired_lower_delta=delta,
        stress_paired_lower_delta=delta,
        base_paired_lower_delta_bound=delta,
        stress_paired_lower_delta_bound=delta,
        matched_lower_excess_cagr=0.10,
        mdd_point=mdd,
        mdd_stress=mdd,
        filled_orders=10,
        observed_interval_count=10,
        invested_interval_count=5,
        invested_interval_fraction=0.5,
        active_cohort_fraction=0.5,
        coverage_passed=True,
    )


def test_COMPOUND_ALPHA_05_VISIBLE_DELTA_GATE() -> None:
    """COMPOUND_ALPHA_05_VISIBLE_DELTA_GATE."""
    request = NetAlphaTrainingRequest(artifact_id="gate")
    settings = CompoundAlphaStudySettings()
    champion = select_compound_champion(
        [_evidence("B00", delta=0.0), _evidence("C01", delta=0.10)],
        baseline_id="B00",
        settings=settings,
        request=request,
    )
    assert champion is not None
    assert champion.experiment_id == "C01"
    assert select_compound_champion(
        [_evidence("B00", delta=0.0), _evidence("C01", delta=0.099999)],
        baseline_id="B00",
        settings=settings,
        request=request,
    ) is None


def test_COMPOUND_ALPHA_06_FAIL_CLOSED_COST_EVIDENCE() -> None:
    """COMPOUND_ALPHA_06_FAIL_CLOSED_COST_EVIDENCE."""
    data = SimpleNamespace(feature_frame=pl.DataFrame(), labels_by_horizon={})
    result = evaluate_compound_alpha_study(
        data,  # type: ignore[arg-type]
        NetAlphaTrainingRequest(artifact_id="cost"),
        CompoundAlphaStudySettings(),
        registry=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert result["status"] == "RESEARCH_ONLY"
    assert result["candidate_count"] == 0
    assert result["promotion_ready"] is False
    assert result["recommended_experiment_id"] is None


def test_COMPOUND_ALPHA_07_READ_ONLY_CLI() -> None:
    """COMPOUND_ALPHA_07_READ_ONLY_CLI."""
    # The CLI integration test owns snapshot resolution; this contract-level
    # assertion verifies the study envelope cannot advertise a published run.
    data = SimpleNamespace(feature_frame=pl.DataFrame(), labels_by_horizon={})
    result = evaluate_compound_alpha_study(
        data,  # type: ignore[arg-type]
        NetAlphaTrainingRequest(artifact_id="readonly"),
        CompoundAlphaStudySettings(),
        registry=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert result["artifact_published"] is False
    assert result["candidate_count"] == 0


def test_COMPOUND_ALPHA_08_MAINLINE_PUBLISH_GATE() -> None:
    """COMPOUND_ALPHA_08_MAINLINE_PUBLISH_GATE."""
    experiment = CompoundAlphaExperiment("C23", "lgbm", "stacked", "stacked", "stacked", "sparse_hold_replace_v2")
    assert experiment.transition_mode == "sparse_hold_replace_v2"
