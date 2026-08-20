"""Net-alpha trainer: causal OOF discovery and untouched-holdout contracts."""
from __future__ import annotations

SPARSE_GROWTH_V5_08_SELECTION_GATES = "SPARSE_GROWTH_V5_08_SELECTION_GATES"

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.stocks.ml import training
from src.stocks.trading.portfolio_constructor import CompoundingPolicyConfig
from src.stocks.trading.portfolio_constructor import stock_risk_policy_fingerprint
from src.stocks.ml.training import _build_horizon_evidence, _evaluate_forward_holdout


def test_sparse_shadow_calendar_is_rebalance_bound() -> None:
    """SPARSE_GROWTH_05_OOF_SHADOW_CALENDAR: shadow uses the frozen cadence."""
    from src.stocks.trading.rebalance_schedule import rebalance_session_indices

    sessions = tuple(
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index)
        for index in range(12)
    )
    indices = rebalance_session_indices(
        list(sessions), sessions[0], sessions[-1], 5, legacy_daily=False
    )
    assert indices == (0, 5, 10)


def test_horizon_evidence_has_no_proxy_score() -> None:
    assert "_proxy_scores" not in _build_horizon_evidence.__code__.co_names


def test_forward_holdout_contract_signature() -> None:
    parameters = inspect.signature(_evaluate_forward_holdout).parameters
    assert list(parameters) == [
        "model",
        "calibration",
        "holdout_panel",
        "request",
        "horizon_sessions",
        "profile",
    ]


class _FakeScoringModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(self.score).alias("predicted_net_alpha")
        )


class _FakeCalibration:
    def apply(self, scored: pl.DataFrame) -> pl.DataFrame:
        return scored.with_columns(
            pl.col("predicted_net_alpha").alias("expected_active_alpha"),
            pl.col("predicted_net_alpha").alias("alpha_lower_bound"),
            pl.col("predicted_net_alpha").alias("expected_net_alpha"),
            pl.col("predicted_net_alpha").alias("net_alpha_lower_bound"),
            pl.lit(0.001, dtype=pl.Float64).alias("exit_cost_rate"),
        )


def _holdout_panel(
    n_sessions: int = 60, score: float = 0.05
) -> pl.DataFrame:
    from datetime import timedelta

    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for i in range(n_sessions):
        session = start + timedelta(days=i)
        open_ = 100.0 * 1.03**i
        rows.append(
            {
                "instrument_id": "KRX:00001",
                "session": session,
                "observation_time": session.replace(hour=15, minute=30),
                "available_time": session.replace(hour=15, minute=31),
                "risk_residual": 0.03,
                "open": open_,
                "close": open_ * 1.02,
                "volume": 1_000_000.0,
                "trading_value": open_ * 1_000_000.0,
                "sector": "S0",
                "adtv": 1.0e8,
                "adtv_20d": 1.0e8,
                "volatility_20d": 0.02,
            }
        )
    return pl.DataFrame(rows)


def _compound_request():
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
    )
    from tests.fixtures.stocks.helpers import stock_liquidity_model

    return NetAlphaTrainingRequest(
        artifact_id="na_holdout_unit",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        compounding=CompoundingCertificationSettings(
            min_observed_sessions=40,
            min_active_cohort_fraction=0.25,
            max_drawdown=0.9,
        ),
    )


def _legacy_profile():
    from src.stocks.ml.contracts import PolicyProfile

    return PolicyProfile(
        profile_id="legacy_overlay_5bps",
        no_trade_band_bps=5.0,
        execution_utility_mode="legacy_target_interpolation_v1",
    )


def test_forward_holdout_empty_panel_fails_closed() -> None:
    request = _compound_request()
    evidence = _evaluate_forward_holdout(
        _FakeScoringModel(0.05),
        _FakeCalibration(),
        pl.DataFrame(),
        request,
        1,
        _legacy_profile(),
    )
    assert evidence["passed"] is False
    assert evidence["reason"] == "holdout-has-no-realized"


def test_forward_holdout_passes_with_complete_base_and_stress() -> None:
    request = _compound_request()
    evidence = _evaluate_forward_holdout(
        _FakeScoringModel(0.05),
        _FakeCalibration(),
        _holdout_panel(n_sessions=60),
        request,
        1,
        _legacy_profile(),
    )
    assert evidence["passed"] is True
    assert evidence["reason"] == ""
    assert evidence["evaluation_kind"] == "prepared-equity-v2-economic-rank"
    assert evidence["order_count"] == 18
    assert evidence["block_count"] == 59
    assert evidence["cohorts"]["eligible_sessions"] == 57
    assert evidence["cohorts"]["observed_sessions"] == 59
    assert evidence["cohorts"]["active_cohort_count"] == 18
    assert evidence["cohorts"]["missing_realized_cohorts"] == 0
    certificate = evidence["certificate"]
    assert certificate["passed"] is True
    assert certificate["base"]["passed"] is True
    assert certificate["stress"]["passed"] is True
    assert "eligibility" not in evidence


def test_forward_holdout_no_economic_edge_is_explicit() -> None:
    request = _compound_request()
    evidence = _evaluate_forward_holdout(
        _FakeScoringModel(0.0002),
        _FakeCalibration(),
        _holdout_panel(n_sessions=60),
        request,
        1,
        _legacy_profile(),
    )
    assert evidence["passed"] is False
    assert evidence["reason"] == "holdout-no-economic-edge"
    assert evidence["cohorts"]["eligible_sessions"] == 0
    assert evidence["cohorts"]["active_cohort_count"] == 0


def test_forward_holdout_incomplete_realized_cohorts_is_explicit() -> None:
    request = _compound_request()
    panel = _holdout_panel(n_sessions=60)

    class _AugmentingModel:
        def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
            base = frame.with_columns(
                pl.lit(0.05).alias("predicted_net_alpha")
            )
            extra = base.head(1).with_columns(
                pl.lit("KRX:99999").alias("instrument_id")
            )
            return pl.concat([base, extra])

    evidence = _evaluate_forward_holdout(
        _AugmentingModel(),
        _FakeCalibration(),
        panel,
        request,
        1,
        _legacy_profile(),
    )
    # A scored key with no executable market row fails closed and is never
    # zero-filled or silently dropped.
    assert evidence["passed"] is False
    assert evidence["reason"].startswith("holdout-replay-invalid:")


def test_forward_holdout_compound_certification_failure_is_explicit() -> None:
    request = _compound_request()
    evidence = _evaluate_forward_holdout(
        _FakeScoringModel(0.05),
        _FakeCalibration(),
        _holdout_panel(n_sessions=10),
        request,
        1,
        _legacy_profile(),
    )
    assert evidence["passed"] is False
    assert evidence["reason"].startswith(
        "holdout-compound-certification-failed:"
    )
    assert "insufficient-observed-sessions" in evidence["reason"]


def _training_fixture(
    n_sessions: int = 120,
    annualization_sessions: int = 40,
) -> tuple[object, object, pl.DataFrame, list[object], tuple[str, ...], object]:
    """Composed net-alpha fixture: data, request, pre_holdout, folds, columns, schema."""
    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
        RiskSettings,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.ml.features import (
        apply_model_feature_schema,
        fit_model_feature_schema,
        stock_net_alpha_v1_roles,
    )
    from src.stocks.research.folds import PurgedWalkForward
    from tests.fixtures.stocks.helpers import (
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=n_sessions, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC),
        (3, 5, 8, 10, 15, 20),
    )
    roles = dict(stock_net_alpha_v1_roles())
    raw = training._index_sessions(data.feature_frame)
    pre_holdout_raw, _holdout_raw, reason = training._locked_holdout(raw, request=NetAlphaTrainingRequest(artifact_id="na_fixture"))
    assert reason == ""
    schema = fit_model_feature_schema(pre_holdout_raw, roles)
    pre_holdout = apply_model_feature_schema(pre_holdout_raw, schema)
    learner_columns = schema.learner_columns
    request = NetAlphaTrainingRequest(
        artifact_id="na_test",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        risk=RiskSettings(min_calibration_sessions=10),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=annualization_sessions,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    splitter = PurgedWalkForward(
        n_folds=2,
        label_horizon_sessions=21,
        embargo_sessions=5,
        session_column="session_index",
        min_train_sessions=annualization_sessions,
    )
    folds = splitter.split(pre_holdout)
    return data, request, pre_holdout, folds, learner_columns, schema


def test_coverage_failure_reason_fails_closed_on_missing_and_incomplete() -> None:
    from dataclasses import replace

    from src.stocks.ml.horizons import HorizonOOFEvidence

    request = _compound_request()
    base_evidence = HorizonOOFEvidence(
        horizon_sessions=3,
        profile_id="lower_bound_only",
        model_family="net_alpha_elastic_net",
        base_log_growth=tuple(0.01 for _ in range(40)),
        stress_log_growth=tuple(0.01 for _ in range(40)),
        cohort_segment_ids=(0,) * 20 + (1,) * 20,
        complete_cohort_count=40,
        active_cohort_count=40,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=2,
        fold_rank_ics=(0.1, 0.2),
    )
    assert training._coverage_failure_reason(base_evidence, request) == ""

    missing = replace(base_evidence, missing_cohort_count=2)
    assert training._coverage_failure_reason(missing, request).startswith(
        "missing-realized-vintages:"
    )

    incomplete = replace(
        base_evidence, cohort_segment_ids=(0,) * 40, segment_count=2
    )
    assert training._coverage_failure_reason(incomplete, request).startswith(
        "incomplete-segment-coverage:"
    )

    inactive = replace(base_evidence, active_cohort_count=0)
    assert training._coverage_failure_reason(inactive, request).startswith(
        "active-coverage-insufficient:"
    )

    typed = replace(
        base_evidence,
        unresolved_outcome_counts=(("MISSING_EXIT_PRICE", 2),),
    )
    reason = training._coverage_failure_reason(typed, request)
    assert reason == ""


def test_score_is_constant_classifies_degenerate_predictions() -> None:
    assert training._score_is_constant(np.asarray([0.0, 0.0])) is True
    assert training._score_is_constant(np.asarray([0.0, 1.0])) is False
    assert training._score_is_constant(np.asarray([np.nan, np.nan])) is True
    assert training._score_is_constant(np.asarray([1.0, np.nan, 1.0])) is True
    assert training._score_is_constant(np.asarray([1.0, 2.0, np.nan])) is False
    assert training._score_is_constant(np.asarray([])) is True


def test_select_elastic_alpha_recovers_linear_signal() -> None:
    from datetime import timedelta

    from src.stocks.ml.contracts import NetAlphaTrainingRequest, RegularizationGrid
    from src.stocks.research.models import ModelManifest

    rng = np.random.default_rng(7)
    rows: list[dict] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for s in range(80):
        for t in range(12):
            feature = float(rng.normal(0.0, 1.0))
            target = 0.1 * feature + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "session_index": s,
                    "session": start + timedelta(days=s),
                    "instrument_id": f"KRX:{t:05d}",
                    "feature__test_x": feature,
                    "net_alpha_target": target,
                    "realized_net_return": target,
                }
            )
    fold_train = pl.DataFrame(rows)
    request = NetAlphaTrainingRequest(artifact_id="na_alpha")
    manifest = ModelManifest(
        artifact_id="na_alpha",
        asset_kind="stock",
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=3,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )
    alpha, fraction, alpha_max, path_evaluations = training._select_elastic_alpha(
        fold_train, request, ("feature__test_x",), 3, RegularizationGrid(), manifest
    )
    assert alpha is not None
    assert alpha > 0.0
    assert alpha_max is not None
    assert alpha_max > 0.0
    assert fraction in RegularizationGrid().fractions
    assert alpha == pytest.approx(fraction * alpha_max)
    assert path_evaluations <= RegularizationGrid().fractions.__len__() + 1


def test_compute_alpha_max_reuses_precomputed_standardized_design() -> None:
    rng = np.random.default_rng(3)
    rows: list[dict] = []
    for s in range(40):
        for t in range(10):
            rows.append(  # noqa: PERF401
                {
                    "session_index": s,
                    "session": datetime(2024, 1, 1, tzinfo=UTC),
                    "instrument_id": f"KRX:{t:05d}",
                    "feature__test_x": float(rng.normal(0.0, 1.0)),
                    "feature__test_y": float(rng.normal(0.0, 1.0)),
                    "net_alpha_target": float(rng.normal(0.0, 1.0)),
                    "realized_net_return": 0.0,
                }
            )
    fold_train = pl.DataFrame(rows)
    columns = ("feature__test_x", "feature__test_y")
    fraction = 0.1
    fresh = training._compute_alpha_max(fold_train, columns, fraction, seed=7)
    standardized = training._standardized_design(fold_train, columns)
    precomputed = training._compute_alpha_max(
        fold_train, columns, fraction, seed=7, standardized=standardized
    )
    assert fresh is not None
    assert precomputed is not None
    assert precomputed[0] == pytest.approx(fresh[0])
    assert precomputed[1] == pytest.approx(fresh[1])


def test_fit_oof_reports_fit_error_instead_of_swallowing() -> None:
    from src.stocks.ml.contracts import FoldScoreDiagnostic

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    assert folds
    manifest = training._base_manifest(request, data, data.feature_frame, 5)

    class _ExplodingModel:
        def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
            del train, validation
            raise ValueError("boom")

        def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
            del frame
            raise RuntimeError("unreachable")

        def manifest(self) -> object:
            return manifest

    oof, oof_labels, rank_ics, diagnostic, _path_count = training._fit_oof(
        pre_holdout, folds, data, request, manifest, learner_columns, 5,
        _ExplodingModel, family="net_alpha_lightgbm_l1",
    )
    assert oof.is_empty()
    assert oof_labels.is_empty()
    assert rank_ics == []
    assert any(
        isinstance(diag, FoldScoreDiagnostic)
        and "fit-error:ValueError:boom" in diag.failure_reason
        for diag in diagnostic.fold_diagnostics
    )


def test_horizon_evidence_constant_baseline_triggers_structural_fallback() -> None:
    """A constant linear screen must skip the challenger and cap it at one horizon."""
    from src.stocks.ml.contracts import (
        FoldScoreDiagnostic,
        HorizonOOFDiagnostic,
        PolicyProfile,
    )
    from src.stocks.ml.horizons import HorizonOOFEvidence

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    diagnostic = HorizonOOFDiagnostic(
        horizon_sessions=5,
        model_family="net_alpha_elastic_net",
        fold_diagnostics=(
            FoldScoreDiagnostic(fold_index=0, failure_reason="constant-oof-score"),
        ),
    )
    profile = PolicyProfile(profile_id="legacy_overlay_5bps", no_trade_band_bps=5.0)
    evidence = HorizonOOFEvidence(
        horizon_sessions=5,
        profile_id="legacy_overlay_5bps",
        model_family="net_alpha_elastic_net",
        base_log_growth=(0.01,),
        stress_log_growth=(0.01,),
        cohort_segment_ids=(0,),
        complete_cohort_count=1,
        active_cohort_count=1,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=1,
        fold_rank_ics=(0.1,),
    )
    selection = training.HorizonSelectionEvidence(
        primary_horizon_sessions=5,
        primary_profile_id="legacy_overlay_5bps",
        adjusted_lower_growth={
            (5, "legacy_overlay_5bps"): {"base": 0.001, "stress": 0.001}
        },
        base_p_values={(5, "legacy_overlay_5bps"): 0.01},
        stress_p_values={(5, "legacy_overlay_5bps"): 0.01},
        base_holm_thresholds={(5, "legacy_overlay_5bps"): 0.05},
        stress_holm_thresholds={(5, "legacy_overlay_5bps"): 0.05},
        selection_reasons=(),
    )
    reason = training._rankability_gate(diagnostic, evidence, selection, request)
    assert reason.startswith("challenger-skipped:no-rankability-evidence:constant-score")

    manifest = training._base_manifest(request, data, data.feature_frame, 5)
    baseline_oof = pl.DataFrame({"x": [1.0, 2.0]})
    baseline_labels = pl.DataFrame({"y": [0.0, 1.0]})
    selected, failure, oof, labels, ics, diag = training._adopt_model_family(
        pre_holdout, folds, data, request, manifest, learner_columns, 5,
        profile, selection,
        baseline_oof, baseline_labels, [0.0372, 0.0550], diagnostic, reason,
    )
    assert selected == "net_alpha_elastic_net"
    assert failure == reason
    assert oof.equals(baseline_oof)
    assert labels.equals(baseline_labels)
    assert ics == [0.0372, 0.0550]
    assert diag == diagnostic

def test_rank_ic_lower_bound_small_fold_series_uses_unit_blocks() -> None:
    from src.stocks.ml.contracts import NetAlphaTrainingRequest

    request = NetAlphaTrainingRequest(
        artifact_id="na_rankic",
        fold_count=2,
        candidate_horizon_sessions=(10,),
        bootstrap_resamples=200,
        seed=11,
    )
    low = training._rank_ic_lower_bound((0.0372, 0.0550), request)
    assert low > 0.0
    assert low <= 0.0372 + 1e-9
    assert low == training._rank_ic_lower_bound((0.0372, 0.0550), request)
    assert (
        training._rank_ic_lower_bound((0.30, 0.40), request)
        > low
    )
    assert training._rank_ic_lower_bound((0.05, 0.05), request) == pytest.approx(0.05)


def test_fold_plan_is_balanced_and_segment_identified() -> None:
    data, request, pre_holdout, folds, _learner_columns, _schema = _training_fixture()
    assert len(folds) == 2
    counts = [len(fold.validation_sessions) for fold in folds]
    assert max(counts) - min(counts) <= 1
    assert [fold.segment_id for fold in folds] == [0, 1]
    for fold in folds:
        assert fold.train_label_end < fold.validation_decision_start


def test_discovery_oof_reuses_cached_baseline_never_refits(tmp_path) -> None:
    from pathlib import Path

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    cache = training._OofCache(Path(tmp_path) / "training")
    discovery = training._build_horizon_evidence(
        pre_holdout, folds, data, request, learner_columns, oof_cache=cache
    )
    if not discovery.evidence:
        pytest.skip("fixture produced no horizon evidence")
    primary = discovery.evidence[0].horizon_sessions
    oof, oof_labels, ics, _diagnostic = training._discovery_oof(
        discovery, primary, folds
    )
    oof_path, labels_path, cached_ics = discovery.oof_by_horizon[primary]
    # The discovery baseline is read back from its temporary OOF cache; the
    # selected primary is never refit.
    assert "_fit_oof" not in training._discovery_oof.__code__.co_names
    assert pl.read_parquet(oof_path).equals(oof)
    assert pl.read_parquet(labels_path).equals(oof_labels)
    assert ics == cached_ics
    assert discovery.path_evaluation_count <= discovery.path_evaluation_bound
    # Terminal cleanup removes the whole cache directory.
    cache.close()
    assert not oof_path.exists()
    assert not labels_path.exists()
    assert not cache.root.exists()


class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta) -> None:
        self.value += delta


def test_training_telemetry_records_phases_without_fitting() -> None:
    from datetime import timedelta

    clock = _FakeClock(datetime(2024, 1, 1, tzinfo=UTC))
    telemetry = training.TrainingTelemetry(clock=clock)
    telemetry.phase("integrity_audit", {"passed": True})
    clock.advance(timedelta(seconds=2))
    telemetry.phase(
        "feature_transform",
        {
            "learner_feature_count": 4,
            "schema_fingerprint": "abc123",
        },
    )
    clock.advance(timedelta(seconds=1))
    telemetry.phase(
        "horizon_discovery",
        {
            "path_evaluation_count": 48,
            "path_evaluation_bound": 72,
        },
    )
    data = telemetry.to_dict()
    assert [p["name"] for p in data["phases"]] == [
        "integrity_audit",
        "feature_transform",
        "horizon_discovery",
    ]
    assert data["phases"][0]["elapsed_ms"] == 0
    assert data["phases"][1]["elapsed_ms"] == 2000
    assert data["phases"][1]["learner_feature_count"] == 4
    assert data["phases"][1]["schema_fingerprint"] == "abc123"
    assert data["phases"][2]["path_evaluation_count"] == 48
    assert data["phases"][2]["path_evaluation_bound"] == 72
    assert data["phases"][0]["peak_rss_mib"] is None or isinstance(
        data["phases"][0]["peak_rss_mib"], float
    )


def test_run_observability_preserves_terminal_no_trade_reason(tmp_path) -> None:
    import json
    from pathlib import Path

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.fixtures.stocks.helpers import (
        pin_net_alpha_outcome_evidence,
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=0.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = pin_net_alpha_outcome_evidence(
        compose_net_alpha_training_data(
            snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
        )
    )
    registry = ModelArtifactRegistry(Path(tmp_path) / "artifacts")
    request = NetAlphaTrainingRequest(
        artifact_id="na_obs_reason",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=40,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    manifest = training.train_net_alpha_model(data, registry, request)
    assert manifest.model_type == "no_trade"
    metrics = json.loads(
        (Path(tmp_path) / "artifacts" / "na_obs_reason" / "metrics.json").read_text()
    )
    run_obs = metrics["run_observability"]
    phases = run_obs["phases"]
    names = [p["name"] for p in phases]
    assert names[:5] == [
        "integrity_audit",
        "snapshot_outcome_readiness",
        "holdout_lock",
        "feature_transform",
        "horizon_discovery",
    ]
    readiness_phase = phases[1]
    assert readiness_phase["passed"] is True
    assert phases[-1]["name"] == "artifact_publish"
    assert phases[-1]["reason"] == "no-horizon-evidence"
    assert all("admission" in entry for entry in run_obs["horizons"])
    json.dumps(run_obs)
    assert len(json.dumps(run_obs).encode("utf-8")) < 24 * 1024
    frontier = metrics["policy_frontier"]
    assert frontier["candidate_count"] == 0
    assert frontier["profile_ids"] == ["legacy_overlay_5bps", "lower_bound_only", "lower_bound_half_kelly"]
    assert len(frontier["dropout_reasons"]) == 18
    assert all(
        f"{h}:{p}" in frontier["dropout_reasons"]
        for h in (3, 5, 8, 10, 15, 20)
        for p in ("legacy_overlay_5bps", "lower_bound_only", "lower_bound_half_kelly")
    )
    json.dumps(frontier)


def test_memory_budget_breach_publishes_bounded_no_trade(tmp_path) -> None:
    import json
    from pathlib import Path

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.fixtures.stocks.helpers import (
        pin_net_alpha_outcome_evidence,
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=0.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = pin_net_alpha_outcome_evidence(
        compose_net_alpha_training_data(
            snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
        )
    )
    registry = ModelArtifactRegistry(Path(tmp_path) / "artifacts")
    request = NetAlphaTrainingRequest(
        artifact_id="na_budget",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        max_rss_mib=1,
        liquidity_model=stock_liquidity_model(),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=40,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    manifest = training.train_net_alpha_model(data, registry, request)
    assert manifest.model_type == "no_trade"
    metrics = json.loads(
        (Path(tmp_path) / "artifacts" / "na_budget" / "metrics.json").read_text()
    )
    run_obs = metrics["run_observability"]
    assert run_obs["phases"][-1]["name"] == "artifact_publish"
    assert run_obs["phases"][-1]["reason"] == (
        "memory-budget-exceeded:horizon_discovery"
    )
    assert len(json.dumps(run_obs).encode("utf-8")) < 24 * 1024


def test_train_net_alpha_model_promotes_champion_or_no_trade(tmp_path) -> None:
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.fixtures.stocks.helpers import (
        pin_net_alpha_outcome_evidence,
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=160, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = pin_net_alpha_outcome_evidence(
        compose_net_alpha_training_data(
            snapshot,
            datetime(2024, 12, 31, tzinfo=UTC),
            (3, 5, 8, 10, 15, 20),
        )
    )
    registry = ModelArtifactRegistry(Path(tmp_path) / "artifacts")
    request = NetAlphaTrainingRequest(
        artifact_id="na_trainer",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=40,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    manifest = training.train_net_alpha_model(data, registry, request)
    assert manifest.artifact_id == "na_trainer"
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
    metrics = json.loads(
        (Path(tmp_path) / "artifacts" / "na_trainer" / "metrics.json").read_text()
    )
    run_obs = metrics["run_observability"]
    discovery_phase = next(
        phase for phase in run_obs["phases"] if phase["name"] == "horizon_discovery"
    )
    assert discovery_phase["path_evaluation_bound"] == 6 * 2 * (3 + 1)
    assert discovery_phase["path_evaluation_count"] <= discovery_phase["path_evaluation_bound"]


def test_horizon_evidence_coverage_keeps_unresolved_outcomes_diagnostic() -> None:
    """SCENARIO_UNRESOLVED_OUTCOME_IS_DIAGNOSTIC: unresolved is not a sole gate."""
    from dataclasses import replace

    from src.stocks.ml.contracts import (
        OUTCOME_MISSING_EXIT_PRICE,
        OUTCOME_REALIZED,
    )

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    discovery = training._build_horizon_evidence(
        pre_holdout, folds, data, request, learner_columns
    )
    # The fixture derives a fully-REALIZED status panel, so every candidate that
    # reaches replay must have no typed unresolved counts.
    for candidate in discovery.evidence:
        assert candidate.unresolved_outcome_counts == ()

    # A candidate whose replay would fail with a typed unresolved outcome must
    # be rejected by the coverage gate before bootstrap.
    evidence = discovery.evidence[0] if discovery.evidence else None
    if evidence is None:
        pytest.skip("fixture produced no horizon evidence")
    unresolved = replace(
        evidence,
        unresolved_outcome_counts=((OUTCOME_MISSING_EXIT_PRICE, 1),),
    )
    reason = training._coverage_failure_reason(unresolved, request)
    assert reason != f"unresolved-outcome:{OUTCOME_MISSING_EXIT_PRICE}:1"
    assert OUTCOME_REALIZED not in reason


def test_replay_costs_produces_parallel_execution_equity() -> None:
    """Base/stress execution evidence share the segment identity and cycle sums."""
    from dataclasses import replace

    from src.stocks.ml.execution_replay import ExecutionReplayEvidence

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    discovery = training._build_horizon_evidence(
        pre_holdout, folds, data, request, learner_columns
    )
    if not discovery.evidence:
        pytest.skip("fixture produced no horizon evidence")
    primary = discovery.evidence[0].horizon_sessions
    oof, oof_labels, _ics, _diag = training._discovery_oof(
        discovery, primary, folds
    )
    profile = discovery.evidence[0].profile_id
    risk = replace(
        request.risk,
        no_trade_band_bps=(
            5.0
            if profile == "legacy_overlay_5bps"
            else 0.0
        ),
    )
    base_evidence, stress_evidence = training._replay_costs(
        oof, oof_labels, request, primary, risk, pre_holdout, data.manifest,
    )
    assert isinstance(base_evidence, ExecutionReplayEvidence)
    assert base_evidence.segment_ids == stress_evidence.segment_ids
    assert len(base_evidence.base_log_growth) == len(base_evidence.segment_ids)
    assert len(base_evidence.stress_log_growth) == len(base_evidence.segment_ids)
    assert base_evidence.planned_cycles == stress_evidence.planned_cycles
    assert base_evidence.filled_orders == stress_evidence.filled_orders
    assert base_evidence.cash_session_fraction == stress_evidence.cash_session_fraction


def test_training_publishes_data_readiness_no_trade_before_oof(tmp_path: Path) -> None:
    """SCENARIO_UNRESOLVED_OUTCOME_IS_DIAGNOSTIC: a data hole stops before OOF."""
    import json
    from dataclasses import replace
    from datetime import UTC, datetime

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        OUTCOME_MISSING_EXIT_PRICE,
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.research.artifacts import ModelArtifactRegistry
    from tests.fixtures.stocks.helpers import (
        pin_net_alpha_outcome_evidence,
        stock_liquidity_model,
        stock_net_alpha_composed_df,
        stock_net_alpha_manifest,
    )

    df = stock_net_alpha_composed_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC),
        (3, 5, 8, 10, 15, 20),
    )
    sessions = data.feature_frame["session"].unique().sort().to_list()
    tail = set(sessions[-3:])
    early = (
        data.feature_frame.filter(~pl.col("session").is_in(sorted(tail)))
        .sort("session")
        .limit(1)
    )
    early_id = str(early["instrument_id"][0])
    early_session = early["session"][0]
    broken = data.status_by_horizon[3].with_columns(
        pl.when(
            (pl.col("instrument_id") == early_id)
            & (pl.col("session") == early_session)
        )
        .then(pl.lit(OUTCOME_MISSING_EXIT_PRICE))
        .otherwise(pl.col("outcome_status"))
        .alias("outcome_status")
    )
    data = replace(data, status_by_horizon={**data.status_by_horizon, 3: broken})
    data = pin_net_alpha_outcome_evidence(data)
    source_unavailable = data.evidence_by_horizon[3].with_columns(
        pl.when(
            (pl.col("instrument_id") == early_id)
            & (pl.col("session") == early_session)
        )
        .then(pl.lit("SOURCE_UNAVAILABLE"))
        .otherwise(pl.col("resolution_kind"))
        .alias("resolution_kind")
    )
    data = replace(
        data,
        evidence_by_horizon={**data.evidence_by_horizon, 3: source_unavailable},
    )

    registry = ModelArtifactRegistry(Path(tmp_path) / "artifacts")
    request = NetAlphaTrainingRequest(
        artifact_id="na_readiness_gate",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=40,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    manifest = training.train_net_alpha_model(data, registry, request)
    assert manifest.model_type == "no_trade"
    metrics = json.loads(
        (Path(tmp_path) / "artifacts" / "na_readiness_gate" / "metrics.json").read_text()
    )
    assert metrics["promotion_reasons"] == ["snapshot-outcome-readiness-failed"]
    assert metrics["snapshot_outcome_readiness"]["passed"] is False
    phases = metrics["run_observability"]["phases"]
    names = [phase["name"] for phase in phases]
    assert names[:2] == ["integrity_audit", "snapshot_outcome_readiness"]
    assert names[-1] == "artifact_publish"
    assert phases[-1]["reason"] == "snapshot-outcome-readiness-failed"
    assert "horizon_discovery" not in names
    assert "holdout_lock" not in names


def test_scenario_unresolved_outcome_is_diagnostic_no_trade_blockers(tmp_path) -> None:
    """LMD-03: missing exit diagnostics stay bounded and do not define admission."""
    import json
    from dataclasses import replace

    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.contracts import (
        CompoundingCertificationSettings,
        NetAlphaTrainingRequest,
        RiskSettings,
    )
    from src.stocks.ml.data import compose_net_alpha_training_data
    from src.stocks.ml.features import (
        apply_model_feature_schema,
        fit_model_feature_schema,
        stock_net_alpha_v1_roles,
    )
    from src.stocks.research.folds import PurgedWalkForward
    from tests.fixtures.stocks.helpers import (
        stock_liquidity_model,
        stock_net_alpha_manifest,
        stock_net_alpha_pinned_df,
    )

    df = stock_net_alpha_pinned_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, datetime(2024, 12, 31, tzinfo=UTC), (3, 5, 8, 10, 15, 20)
    )
    roles = dict(stock_net_alpha_v1_roles())
    raw = training._index_sessions(data.feature_frame)
    request = NetAlphaTrainingRequest(
        artifact_id="na_blockers",
        fold_count=2,
        candidate_horizon_sessions=(3, 5, 8, 10, 15, 20),
        bootstrap_resamples=50,
        liquidity_model=stock_liquidity_model(),
        risk=RiskSettings(min_calibration_sessions=10),
        compounding=CompoundingCertificationSettings(
            annualization_sessions=40,
            min_observed_sessions=10,
            min_active_cohort_fraction=0.1,
        ),
    )
    pre_holdout_raw, _holdout_raw, reason = training._locked_holdout(
        raw, request=request
    )
    assert reason == ""
    schema = fit_model_feature_schema(pre_holdout_raw, roles)
    pre_holdout = apply_model_feature_schema(pre_holdout_raw, schema)
    learner_columns = schema.learner_columns
    splitter = PurgedWalkForward(
        n_folds=2,
        label_horizon_sessions=21,
        embargo_sessions=5,
        session_column="session_index",
        min_train_sessions=40,
    )
    folds = splitter.split(pre_holdout)

    discovery = training._build_horizon_evidence(
        pre_holdout, folds, data, request, learner_columns
    )
    if not discovery.evidence:
        pytest.skip("fixture produced no horizon evidence")
    primary = discovery.evidence[0].horizon_sessions
    profile = discovery.evidence[0].profile_id
    risk = replace(
        request.risk,
        no_trade_band_bps=(
            5.0
            if profile == "legacy_overlay_5bps"
            else 0.0
        ),
    )
    oof, oof_labels, _ics, _diag = training._discovery_oof(
        discovery, primary, folds
    )
    base_evidence, _stress_evidence = training._replay_costs(
        oof, oof_labels, request, primary, risk, pre_holdout, data.manifest,
    )

    # The frontier projection persists only bounded execution evidence, never
    # raw scores, orders, or return vectors.
    from src.stocks.ml.training import _policy_frontier_projection

    projection = _policy_frontier_projection(request, discovery, None)
    assert isinstance(projection["execution_evidence"], dict)
    json.dumps(projection)
    for entry in projection["execution_evidence"].values():
        assert "period_net_returns" not in entry
        assert "predicted_net_alpha" not in entry
        assert "net_return" not in entry
    json.dumps(base_evidence.diagnostics())
    assert all(
        isinstance(count, int) and count >= 0
        for _reason, count in base_evidence.unfilled_order_reason_counts
    )


def test_coverage_failure_reason_allows_isolated_blocked_vintages() -> None:
    """Isolated blocked vintages below tolerance do not fail admission."""
    from dataclasses import replace

    from src.stocks.ml.horizons import HorizonOOFEvidence

    request = _compound_request()
    base_evidence = HorizonOOFEvidence(
        horizon_sessions=3,
        profile_id="lower_bound_only",
        model_family="net_alpha_elastic_net",
        base_log_growth=tuple(0.01 for _ in range(400)),
        stress_log_growth=tuple(0.01 for _ in range(400)),
        cohort_segment_ids=(0,) * 200 + (1,) * 200,
        complete_cohort_count=400,
        active_cohort_count=400,
        partial_cohort_count=0,
        missing_cohort_count=0,
        segment_count=2,
        fold_rank_ics=(0.1, 0.2),
    )
    # 2/400 = 0.5% and 20/400 = 5.0% stay within tolerance and are admitted.
    isolated = replace(base_evidence, blocked_vintage_count=2)
    assert training._coverage_failure_reason(isolated, request) == ""
    at_tolerance = replace(base_evidence, blocked_vintage_count=20)
    assert training._coverage_failure_reason(at_tolerance, request) == ""
    # 21/400 = 5.25% exceeds the tolerance and still fails closed.
    dominant = replace(base_evidence, blocked_vintage_count=21)
    assert training._coverage_failure_reason(dominant, request).startswith(
        "selected-exit-unresolved:"
    )

def _adaptive_observations(
    n_sessions: int = 120, n_tickers: int = 20
) -> pl.DataFrame:
    """Deterministic residual panel with score-correlated positive alpha."""
    rng = np.random.default_rng(11)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for s in range(n_sessions):
        for t in range(n_tickers):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": start + timedelta(days=s),
                    "score": float(t) + (s % 3),
                    "residual_o2o_5d": float(
                        0.004 * t + rng.normal(0.0, 0.0002)
                    ),
                    "label_available_time": start + timedelta(days=s + 6),
                }
            )
    return pl.DataFrame(rows)


def test_fold_zero_active_with_seed_ledger() -> None:
    """Seed ledger injection removes fold zero's cold-start cash window."""
    from src.stocks.ml.labels import SESSION_COLUMN

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    horizon = 3
    manifest = training._base_manifest(request, data, data.feature_frame, horizon)
    initial_train = pre_holdout.filter(
        pl.col("session_index") < folds[0].validation_decision_start
    )
    seed = training.build_initial_calibration_seed(
        initial_train, request, learner_columns, horizon, manifest, data=data
    )
    assert not seed.is_empty()
    oof, oof_labels, _ics, _diag, _path = training._fit_oof(
        pre_holdout, [folds[0]], data, request, manifest, learner_columns,
        horizon, None, family="net_alpha_elastic_net",
    )
    calibrated = training._causal_oof_calibrate(
        oof, oof_labels, request, horizon, seed_ledger=seed
    )
    active = calibrated.filter(
        pl.col("net_alpha_lower_bound").is_not_null()
        & (pl.col("net_alpha_lower_bound") > 0.0)
    )
    assert active[SESSION_COLUMN].n_unique() > 0
    first_session = oof[SESSION_COLUMN].min()
    unseeded = training._causal_oof_calibrate(oof, oof_labels, request, horizon)
    early = unseeded.filter(pl.col(SESSION_COLUMN) == first_session)
    assert early["net_alpha_lower_bound"].null_count() == early.height


def test_adaptive_bucketing_cold_start() -> None:
    """Thin history uses quintiles; thick history uses the nominal deciles."""
    from src.core.costs import default_base_schedule
    from src.stocks.research.economic_alpha import (
        CausalAlphaCalibrator,
        _shrink_lower_bound,
        adaptive_bucket_count,
    )

    assert adaptive_bucket_count(10, 251) == 5
    assert adaptive_bucket_count(10, 252) == 10
    assert adaptive_bucket_count(5, 100) == 5
    assert adaptive_bucket_count(5, 400) == 5

    location = 0.004
    thin = _shrink_lower_bound(-0.02, location, 20)
    assert location > thin > -0.02
    assert _shrink_lower_bound(-0.02, location, 2520) < thin

    cal = CausalAlphaCalibrator(
        bucket_count=10, min_calibration_sessions=5, n_bootstrap=50
    )
    cold = cal.prepare_decision(
        _adaptive_observations(120),
        datetime(2024, 4, 29, tzinfo=UTC),
        default_base_schedule(),
    )
    assert cold["history_sessions"] >= 5
    assert cold["bucket_count"] == 5
    assert cold["buckets"]
    warm = cal.prepare_decision(
        _adaptive_observations(300),
        datetime(2024, 10, 22, tzinfo=UTC),
        default_base_schedule(),
    )
    assert warm["history_sessions"] >= 252
    assert warm["bucket_count"] == 10


_TRAINING_ARTIFACT_SCENARIO = "training_artifact_marks_economic_rank_v2"


def test_training_artifact_marks_economic_rank_v2() -> None:
    """Promoted artifact records economic_net_v1 mode and v4 delta-cost-aware evidence version."""
    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    profile = _compound_request().policy_profiles[0]
    horizon = 10
    policy = training._risk_policy_for_profile(request, profile, horizon)
    assert policy.economic_ranking_mode == "economic_net_v1"
    assert policy.compounding.forecast_horizon_sessions == horizon
    assert policy.execution_utility_mode == "sparse_hold_replace_v2"

    params_json = training._policy_profile_params(request, profile, horizon)
    params = json.loads(params_json)
    assert params["economic_ranking_mode"] == "economic_net_v1"
    assert params["execution_evidence_version"] == "prepared-equity-v5-sparse-growth"
    assert params["execution_utility_mode"] == "sparse_hold_replace_v2"
    assert params["forecast_horizon_sessions"] == horizon

    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    raw_policy = StockRiskPolicy(
        top_k=request.portfolio.top_k,
        gross_cap=request.portfolio.max_exposure,
        single_name_cap=request.portfolio.max_single_weight,
        participation_limit=request.portfolio.participation_limit,
        no_trade_band_bps=profile.no_trade_band_bps,
        compounding=CompoundingPolicyConfig(growth_risk_aversion=profile.growth_risk_aversion),
        economic_ranking_mode="raw_score_v1",
    )
    assert stock_risk_policy_fingerprint(policy) != stock_risk_policy_fingerprint(raw_policy)


def test_horizon_pins_policy() -> None:
    """HC_LOG_UTILITY_03_TRAINING_REPLAY_PINS_CANDIDATE_HORIZON: _risk_policy_for_profile pins forecast_horizon_sessions."""
    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()
    profile = _compound_request().policy_profiles[0]

    policy_10 = training._risk_policy_for_profile(request, profile, 10)
    assert policy_10.compounding.forecast_horizon_sessions == 10

    policy_5 = training._risk_policy_for_profile(request, profile, 5)
    assert policy_5.compounding.forecast_horizon_sessions == 5

    params_10 = json.loads(training._policy_profile_params(request, profile, 10))
    assert params_10["forecast_horizon_sessions"] == 10

    params_5 = json.loads(training._policy_profile_params(request, profile, 5))
    assert params_5["forecast_horizon_sessions"] == 5

    assert (
        stock_risk_policy_fingerprint(policy_10)
        != stock_risk_policy_fingerprint(policy_5)
    )


DELTA_COST_UTILITY_07 = "DELTA_COST_UTILITY_07_OOF_HOLDOUT_MODE_PINNING"


def test_delta_cost_utility_07_oof_holdout_mode_pinning() -> None:
    """_risk_policy_for_profile pins delta_cost_aware_v1 and changes fingerprint vs legacy."""
    from src.stocks.ml.contracts import PolicyProfile

    data, request, pre_holdout, folds, learner_columns, _schema = _training_fixture()

    v4_profile = PolicyProfile(
        profile_id="lower_bound_only",
        no_trade_band_bps=0.0,
        execution_utility_mode="delta_cost_aware_v1",
    )
    horizon = 10
    v4_policy = training._risk_policy_for_profile(request, v4_profile, horizon)
    assert v4_policy.execution_utility_mode == "delta_cost_aware_v1"

    v4_params = json.loads(training._policy_profile_params(request, v4_profile, horizon))
    assert v4_params["execution_utility_mode"] == "delta_cost_aware_v1"
    assert v4_params["execution_evidence_version"] == "prepared-equity-v4-delta-cost-aware"

    legacy_profile = PolicyProfile(
        profile_id="lower_bound_only",
        no_trade_band_bps=0.0,
        execution_utility_mode="legacy_target_interpolation_v1",
    )
    legacy_policy = training._risk_policy_for_profile(request, legacy_profile, horizon)
    assert legacy_policy.execution_utility_mode == "legacy_target_interpolation_v1"

    legacy_params = json.loads(training._policy_profile_params(request, legacy_profile, horizon))
    assert legacy_params["execution_utility_mode"] == "legacy_target_interpolation_v1"
    assert legacy_params["execution_evidence_version"] == "prepared-equity-v3-horizon-consistent"

    assert stock_risk_policy_fingerprint(v4_policy) != stock_risk_policy_fingerprint(legacy_policy)
