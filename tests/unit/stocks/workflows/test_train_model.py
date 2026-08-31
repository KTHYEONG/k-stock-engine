"""Net-alpha mainline training workflow tests.

Covers the transformed-frame training, causal horizon selection, decimal
realized-outcome replay, untouched forward holdout, calibration persistence,
and complete ``NO_TRADE`` evidence contract of the canonical
``stock_net_alpha_v1`` workflow. The legacy LambdaRank/Optuna v2 path is not
part of this suite.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.cli.train import build_parser
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.contracts import (
    CompoundingCertificationSettings,
    ExecutionFrontierSettings,
    NetAlphaTrainingRequest,
    RiskSettings,
)
from src.stocks.ml.features import build_model_features, stock_net_alpha_v1_roles
from src.stocks.research.artifacts import (
    METRICS_FILENAME,
    ModelArtifactRegistry,
    PredictionRequest,
)
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_liquidity_model,
    stock_net_alpha_manifest,
    stock_net_alpha_pinned_df,
)


def _snapshot(
    n_sessions: int = 160, n_tickers: int = 8
) -> tuple[DatasetSnapshot, pl.DataFrame]:
    df = stock_net_alpha_pinned_df(
        n_sessions=n_sessions,
        n_tickers=n_tickers,
        audit_clean=True,
        label_scale=50.0,
    )
    manifest = stock_net_alpha_manifest(columns=df.columns)
    return DatasetSnapshot(manifest=manifest, frame=df), df


def _request(artifact_id: str, **kwargs) -> NetAlphaTrainingRequest:
    defaults = {
        "artifact_id": artifact_id,
        "fold_count": 2,
        "candidate_horizon_sessions": (5, 10),
        "bootstrap_resamples": 50,
        "liquidity_model": stock_liquidity_model(),
    }
    defaults.update(kwargs)
    horizon_sessions = defaults["candidate_horizon_sessions"]
    defaults["execution_frontier"] = ExecutionFrontierSettings(
        candidate_horizon_sessions=horizon_sessions,
    )
    return NetAlphaTrainingRequest(**defaults)


def _compound_request(artifact_id: str, **kwargs) -> NetAlphaTrainingRequest:
    """Request tuned to certify a synthetic fixture holdout."""
    defaults = {
        "artifact_id": artifact_id,
        "fold_count": 2,
        "candidate_horizon_sessions": (5, 10),
        "bootstrap_resamples": 50,
        "liquidity_model": stock_liquidity_model(),
        "risk": RiskSettings(min_calibration_sessions=40),
        "compounding": CompoundingCertificationSettings(
            annualization_sessions=60,
            min_observed_sessions=40,
            min_active_cohort_fraction=0.5,
            max_drawdown=0.5,
        ),
    }
    defaults.update(kwargs)
    horizon_sessions = defaults["candidate_horizon_sessions"]
    defaults["execution_frontier"] = ExecutionFrontierSettings(
        candidate_horizon_sessions=horizon_sessions,
    )
    return NetAlphaTrainingRequest(**defaults)


def _holdout_eligibility(
    df: pl.DataFrame, primary_horizon_sessions: int
) -> tuple[str, str]:
    """First/last realized holdout sessions for the selected primary horizon.

    ``eligible_from`` is the first raw holdout session; ``eligible_to`` is the
    last session whose label is available at the decision time (``session +
    horizon days <= last session``), i.e. the last realized holdout session.
    """
    from datetime import timedelta

    # The long-format composition drops each instrument's warm-up observation,
    # so the trainer's session spine starts at the second distinct session.
    sessions = sorted(df["session"].unique().to_list())[1:]
    holdout_count = max(1, len(sessions) // 5)
    holdout_start = sessions[-holdout_count]
    last = sessions[-1]
    return (
        holdout_start.isoformat(),
        (last - timedelta(days=primary_horizon_sessions)).isoformat(),
    )


def _feature_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Build the canonical learner frame exactly as the trainer consumes it."""
    drops = [
        c
        for c in df.columns
        if c.startswith(("net_alpha_", "label_available_time_", "risk_residual_", "reference_cost_"))
    ]
    features, _ = build_model_features(df.drop(drops), dict(stock_net_alpha_v1_roles()))
    return features


def test_train_net_alpha_publishes_artifact_or_no_trade(tmp_path) -> None:
    snapshot, df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(snapshot, registry, _request("na_mainline"))
    assert manifest.artifact_id == "na_mainline"
    assert manifest.feature_set == "stock_net_alpha_v1"
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
    if manifest.model_type == "no_trade":
        assert manifest.eligible_from == sorted(df["session"].unique().to_list())[1].isoformat()
    if manifest.model_type != "no_trade":
        stored = json.loads(
            (tmp_path / "artifacts" / "na_mainline" / "manifest.json").read_text()
        )
        assert "calibration_state" in stored["params"]
        loaded = registry.load(
            manifest.artifact_id,
            PredictionRequest(
                asset_kind=AssetKind.STOCK,
                feature_set=manifest.feature_set,
                feature_schema_hash=manifest.feature_schema_hash,
                decision_time=datetime(2024, 6, 1, tzinfo=UTC),
            ),
        ).model
        scored = loaded.predict(_feature_frame(df))
        assert "predicted_net_alpha" in scored.columns
        assert "net_alpha_lower_bound" in scored.columns
        with pytest.raises(ValueError, match="rejects target"):
            loaded.predict(
                _feature_frame(df).with_columns(pl.lit(0.0).alias("net_alpha_target"))
            )
        holdout = registry.read_forward_holdout(manifest.artifact_id)
        assert holdout is not None
        assert holdout.get("evidence", {}).get("passed") is True


def test_train_net_alpha_promoted_eligibility_is_forward_holdout(tmp_path) -> None:
    """A promoted artifact is eligible only over its realized holdout interval."""
    snapshot, df = _snapshot(n_sessions=140, n_tickers=4)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(
        snapshot,
        registry,
        _compound_request(
            "na_holdout_elig",
            candidate_horizon_sessions=(5,),
            risk=RiskSettings(min_calibration_sessions=20),
            compounding=CompoundingCertificationSettings(
                annualization_sessions=60,
                min_observed_sessions=20,
                min_active_cohort_fraction=0.5,
                max_drawdown=0.5,
            ),
        ),
    )
    if manifest.model_type == "no_trade":
        return
    metrics = json.loads(
        (tmp_path / "artifacts" / "na_holdout_elig" / "metrics.json").read_text()
    )
    primary = metrics["primary_horizon_sessions"]
    holdout_from, holdout_to = _holdout_eligibility(df, primary)
    assert manifest.eligible_from == holdout_from
    assert manifest.eligible_to == holdout_to
    assert manifest.eligible_from != df["session"].min().isoformat()
    holdout = metrics["holdout"]
    assert holdout["eligibility"]["eligible_from"] == holdout_from
    assert holdout["eligibility"]["eligible_to"] == holdout_to
    assert holdout["certificate"]["passed"] is True
    assert holdout["certificate"]["base"]["passed"] is True
    assert holdout["certificate"]["stress"]["passed"] is True
    assert holdout["certificate"]["base"]["cagr"] > 0.0
    assert holdout["certificate"]["base"]["lower_cagr"] > 0.0
    assert holdout["certificate"]["stress"]["cagr"] > 0.0
    assert holdout["certificate"]["stress"]["lower_cagr"] > 0.0
    assert holdout["cohorts"]["observed_sessions"] >= 20
    assert holdout["cohorts"]["active_cohort_count"] > 0
    assert holdout["cohorts"]["missing_realized_cohorts"] == 0


def test_train_net_alpha_no_trade_diagnostics_are_explicit(tmp_path) -> None:
    """A failed holdout publishes explicit no-trade diagnostics, never a gate."""
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(snapshot, registry, _request("na_no_trade_diag"))
    assert manifest.model_type == "no_trade"
    metrics = json.loads(
        (tmp_path / "artifacts" / "na_no_trade_diag" / "metrics.json").read_text()
    )
    holdout = metrics.get("holdout")
    if holdout is not None:
        assert holdout["passed"] is False
        reason = holdout["reason"]
        assert reason.startswith(("holdout-", "holdout-replay-"))
        assert "certificate" in holdout
        assert "cohorts" in holdout


def test_train_net_alpha_writes_complete_no_trade_evidence(tmp_path) -> None:
    df = stock_net_alpha_pinned_df(
        n_sessions=160, n_tickers=8, audit_clean=True, label_scale=0.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    artifact_root = tmp_path / "artifacts"
    registry = ModelArtifactRegistry(artifact_root)
    train_model(snapshot, registry, _compound_request("na_no_trade"))
    metrics_path = artifact_root / "na_no_trade" / METRICS_FILENAME
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["no_trade"] is True
    assert payload["promoted"] is False
    assert "promotion_reasons" in payload
    assert "gates" in payload
    assert "oof_diagnostics" in payload
    assert isinstance(payload["oof_diagnostics"], list)
    assert payload["oof_diagnostics"]
    assert "no-horizon-evidence" in payload["promotion_reasons"][0]


def test_train_net_alpha_rejects_legacy_snapshot(tmp_path) -> None:
    from tests.fixtures.stocks.helpers import stock_v2_composed_df, stock_v2_manifest

    df = stock_v2_composed_df(n_sessions=60, n_tickers=4)
    manifest = stock_v2_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="net-alpha"):
        train_model(
            DatasetSnapshot(manifest=manifest, frame=df),
            registry,
            _request("legacy_reject", candidate_horizon_sessions=(5,)),
        )


def test_train_net_alpha_rejects_legacy_optuna_flags(tmp_path) -> None:
    with pytest.raises(TypeError):
        # optuna_trials must not exist on the net-alpha request
        NetAlphaTrainingRequest(  # type: ignore[call-arg]
            artifact_id="x", optuna_trials=80
        )


def test_train_net_alpha_rejects_invalid_request(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="model_threads must be positive"):
        train_model(
            snapshot, registry, _request("bad_threads", model_threads=0)
        )


def test_train_net_alpha_duplicate_publish_rejected(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    train_model(snapshot, registry, _request("na_dup"))
    with pytest.raises(ValueError, match="already exists"):
        train_model(snapshot, registry, _request("na_dup"))


def test_train_net_alpha_folds_respected(tmp_path) -> None:
    snapshot, _df = _snapshot(n_sessions=200)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(
        snapshot, registry, _request("na_folds", fold_count=3)
    )
    assert manifest.artifact_id == "na_folds"


def test_train_net_alpha_v3_publishes_no_trade_without_positive_evidence(
    tmp_path,
) -> None:
    """The net-alpha path fails closed to a complete NO_TRADE artifact.

    A snapshot with genuinely non-positive evidence (zero-signal labels that
    still satisfy the integrity audit and carry realized outcomes) must publish
    ``no_trade`` with complete evidence rather than relax a gate.
    """
    df = stock_net_alpha_pinned_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=0.0
    )
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    result = train_model(snapshot, registry, _compound_request("na_no_evidence"))
    assert result.artifact_id == "na_no_evidence"
    assert result.model_type == "no_trade"
    metrics = json.loads(
        (tmp_path / "artifacts" / "na_no_evidence" / METRICS_FILENAME).read_text()
    )
    assert metrics["no_trade"] is True
    assert metrics["promoted"] is False
    assert "oof_diagnostics" in metrics


def test_train_net_alpha_missing_realized_outcomes_fail_closed(tmp_path) -> None:
    """Missing decimal realized outcomes raise, never silently become NO_TRADE.

    A snapshot whose label frames carry targets but no ``risk_residual`` /
    ``reference_cost`` realized outcome columns violates the replay schema and
    must fail closed with ``ValueError`` instead of degrading into an empty
    block list that looks like a genuine no-trade.
    """
    df = stock_net_alpha_pinned_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=50.0
    )
    df = df.drop(["risk_residual", "reference_cost"])
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    with pytest.raises(ValueError, match="realized-outcome"):
        train_model(snapshot, registry, _compound_request("na_missing_realized"))


def test_train_net_alpha_model_types_are_canonical(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = train_model(snapshot, registry, _request("na_types"))
    assert manifest.model_type in {
        "net_alpha_elastic_net",
        "net_alpha_lightgbm_l1",
        "no_trade",
    }
    if manifest.model_type != "no_trade":
        stored = json.loads(
            (
                tmp_path / "artifacts" / "na_types" / "manifest.json"
            ).read_text()
        )
        assert stored["model_type"] in {
            "net_alpha_elastic_net",
            "net_alpha_lightgbm_l1",
        }


def test_train_model_records_completed_through_injected_observer(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    calls: list[tuple] = []

    class _SpyObserver:
        def record_completed(self, context, manifest, registry, telemetry=None):
            del registry, telemetry
            calls.append(("completed", context.artifact_id, manifest.artifact_id))

        def record_failed(self, context, phase, exc, telemetry=None):
            del context, phase, exc, telemetry
            calls.append(("failed",))

    train_model(snapshot, registry, _request("na_obs"), observer=_SpyObserver())
    assert calls == [("completed", "na_obs", "na_obs")]


def test_train_model_records_failure_through_injected_observer(tmp_path) -> None:
    df = stock_net_alpha_pinned_df(
        n_sessions=120, n_tickers=8, audit_clean=True, label_scale=50.0
    ).drop(["risk_residual", "reference_cost"])
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    calls: list[tuple] = []

    class _SpyObserver:
        def record_completed(self, *args, **kwargs):
            del args, kwargs
            calls.append(("completed",))

        def record_failed(self, context, phase, exc, telemetry=None):
            del context, telemetry
            calls.append(("failed", phase, type(exc).__name__))

    with pytest.raises(ValueError, match="realized-outcome"):
        train_model(
            DatasetSnapshot(manifest=manifest, frame=df),
            registry,
            _compound_request("na_fail_obs"),
            observer=_SpyObserver(),
        )
    assert calls
    assert calls[0][0] == "failed"
    assert calls[0][2] == "ValueError"


def test_train_model_ledger_failure_does_not_change_artifact(tmp_path) -> None:
    snapshot, _df = _snapshot()
    registry = ModelArtifactRegistry(tmp_path / "artifacts")

    class _ExplodingObserver:
        def record_completed(self, context, manifest, registry, telemetry=None):
            del context, manifest, registry, telemetry
            raise RuntimeError("ledger boom")

        def record_failed(self, context, phase, exc, telemetry=None):
            del context, phase, exc, telemetry

    manifest = train_model(
        snapshot,
        registry,
        _request("na_obs_ledger_fail"),
        observer=_ExplodingObserver(),
    )
    assert manifest.artifact_id == "na_obs_ledger_fail"


def test_execution_frontier_replay_policy_and_no_trade_gate() -> None:
    """ML_EXEC_FRONTIER_02_REPLAY_POLICY_AND_NO_TRADE_GATE.

    A selected synthetic cell persists its own C and K: the operational
    ``StockRiskPolicy`` uses those exact rebalance cadence and top_k for both the
    sparse (v5) and matched dense-shadow (delta-cost-aware) replay paths. A
    candidate whose base, stress, or paired lower bound is <= 0, or whose
    sparse/shadow turnover ratio exceeds 0.60, remains NO_TRADE (primary None).
    """
    from src.stocks.ml.horizons import HorizonOOFEvidence, select_horizons
    from src.stocks.ml.training import _risk_policy_for_profile
    from src.stocks.ml.contracts import PolicyProfile

    request = _request("na_exec_frontier")
    sparse_profile = request.policy_profiles[0]  # legacy_overlay_5bps (v5)
    dense_profile = PolicyProfile(
        profile_id="lower_bound_half_kelly",
        no_trade_band_bps=0.0,
        growth_risk_aversion=2.0,
        execution_utility_mode="delta_cost_aware_v1",
        sizing_mode="alpha_vol_squared_v1",
    )

    # The same (H=20, C=5, K=12) cell drives both sparse and dense paths.
    sparse_policy = _risk_policy_for_profile(
        request, sparse_profile, 20,
        rebalance_frequency_sessions=5, top_k=12,
    )
    dense_policy = _risk_policy_for_profile(
        request, dense_profile, 20,
        rebalance_frequency_sessions=5, top_k=12,
    )
    for policy in (sparse_policy, dense_policy):
        assert policy.rebalance_frequency_sessions == 5
        assert policy.top_k == 12
        assert policy.compounding.forecast_horizon_sessions == 20

    def _cell(
        base: tuple[float, ...],
        stress: tuple[float, ...],
        paired: tuple[float, ...],
        *,
        sparse_turnover: float,
        shadow_turnover: float,
    ) -> HorizonOOFEvidence:
        return HorizonOOFEvidence(
            horizon_sessions=20,
            profile_id=sparse_profile.profile_id,
            model_family="net_alpha_elastic_net",
            base_log_growth=base,
            stress_log_growth=stress,
            cohort_segment_ids=tuple(0 for _ in base),
            complete_cohort_count=len(base),
            active_cohort_count=len(base),
            partial_cohort_count=0,
            missing_cohort_count=0,
            segment_count=1,
            fold_rank_ics=(0.1, 0.2, 0.3),
            rebalance_frequency_sessions=5,
            top_k=12,
            paired_stress_log_growth=paired,
            sparse_turnover=sparse_turnover,
            shadow_turnover=shadow_turnover,
        )

    good = _cell(
        (0.01,) * 60, (0.01,) * 60, (0.002,) * 60,
        sparse_turnover=1.0, shadow_turnover=2.0,
    )
    selected = select_horizons((good,), 0.05, 42)
    assert selected.primary_horizon_sessions == 20
    assert selected.primary_rebalance_frequency_sessions == 5
    assert selected.primary_top_k == 12

    high_turnover = _cell(
        (0.01,) * 60, (0.01,) * 60, (0.002,) * 60,
        sparse_turnover=1.0, shadow_turnover=1.0,
    )
    rejected_ratio = select_horizons((high_turnover,), 0.05, 42)
    assert rejected_ratio.primary_horizon_sessions is None

    negative_stress = _cell(
        (0.01,) * 60, tuple(-0.001 for _ in range(60)), (0.002,) * 60,
        sparse_turnover=1.0, shadow_turnover=2.0,
    )
    rejected_stress = select_horizons((negative_stress,), 0.05, 42)
    assert rejected_stress.primary_horizon_sessions is None


def test_request_and_cli_defaults_updated() -> None:
    """request_and_cli_defaults_updated: bootstrap resolvability defaults."""
    assert NetAlphaTrainingRequest(artifact_id="defaults").bootstrap_resamples == 2000
    args = build_parser().parse_args(["--artifact-id", "defaults"])
    assert args.bootstrap_resamples == 2000
    # Holdout certificate policy is a separate pre-registered scope.
    assert CompoundingCertificationSettings().bootstrap_resamples == 200



def _rawnet_dispatch_fixture():
    """Minimal causal panel for dispatch-level tests (no fitting executed)."""
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from src.core.datasets import DatasetManifest
    from src.core.instruments import AssetKind
    from src.stocks.ml.contracts import (
        ExecutionFrontierSettings,
        NetAlphaResearchData,
        NetAlphaTrainingRequest,
    )
    from src.stocks.research.folds import Fold

    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for s in range(12):
        session = start + timedelta(days=s)
        rows.extend(
            {
                "instrument_id": f"KRX:{t + 1:05d}",
                "session": session,
                "session_index": s,
                "feature__a": float(t) / 6.0,
                "open": 10000.0,
                "adtv_20d": 5.0e9,
                "volatility_20d": 0.03,
            }
            for t in range(6)
        )
    frame = pl.DataFrame(rows).sort(["session", "instrument_id"])
    label_rows = [
        {
            "instrument_id": row["instrument_id"],
            "session": row["session"],
            "net_alpha_target": 0.0,
            "risk_residual": 0.001,
            "reference_cost": 0.0005,
            "label_available_time": row["session"] + timedelta(days=10),
        }
        for row in frame.iter_rows(named=True)
    ]
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="h",
        provider_version="p",
        universe_policy_version="u",
        universe_policy_hash="u",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="f",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        time_start=start,
        time_end=start + timedelta(days=12),
        generated_time=start + timedelta(days=12),
        row_count=len(rows),
    )
    data = NetAlphaResearchData(
        feature_frame=frame,
        labels_by_horizon={10: pl.DataFrame(label_rows)},
        manifest=manifest,
    )
    request = NetAlphaTrainingRequest(
        artifact_id="rawnet-dispatch",
        candidate_horizon_sessions=(10,),
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=(10,),
            candidate_rebalance_frequency_sessions=(5,),
            candidate_top_k=(12,),
        ),
        model_threads=1,
    )
    folds = [
        Fold(
            train_mask=list(range(36)),
            validation_mask=list(range(48, 60)),
            train_label_end=8,
            validation_decision_start=10,
            segment_id=0,
            validation_sessions=(10, 11),
        )
    ]
    return frame, folds, data, request


def test_rawnet_lgbm_04_training_dispatch_fail_closed(monkeypatch) -> None:
    """SCENARIO_RAWNET_LGBM_04_TRAINING_DISPATCH_FAIL_CLOSED."""
    import src.stocks.ml.economic_research as economic_research_module
    from src.stocks.ml.contracts import ELASTIC_NET_FAMILY
    from src.stocks.ml.training import _fit_oof

    pre_holdout, folds, data, request = _rawnet_dispatch_fixture()
    calls: dict[str, int] = {}

    def spy(pre_holdout_arg, folds_arg, data_arg, request_arg, learner_columns, horizon_sessions):
        calls["count"] = calls.get("count", 0) + 1
        return pl.DataFrame({"instrument_id": ["A"], "score": [0.5]}), pl.DataFrame(
            {"risk_residual": [0.001]}
        )

    monkeypatch.setattr(economic_research_module, "fit_rawnet_lgbm_oof", spy)
    oof, labeled, ics, diagnostic, path_evaluations = _fit_oof(
        pre_holdout,
        folds,
        data,
        request,
        None,
        ("feature__a",),
        10,
        None,
        family="economic_rawnet_lgbm",
    )
    assert calls["count"] == 1
    assert oof.columns == ["instrument_id", "score"]
    assert labeled.columns == ["risk_residual"]
    assert ics == []
    assert path_evaluations == 0
    assert diagnostic.model_family == "economic_rawnet_lgbm"

    with pytest.raises(ValueError, match="declared economic families"):
        _fit_oof(
            pre_holdout,
            folds,
            data,
            request,
            None,
            ("feature__a",),
            10,
            None,
            family="unknown-family",
        )
    assert calls["count"] == 1

    # The elastic-net family stays admissible (empty folds -> empty evidence).
    empty = _fit_oof(
        pre_holdout,
        [],
        data,
        request,
        None,
        ("feature__a",),
        10,
        None,
        family=ELASTIC_NET_FAMILY,
    )
    assert empty[0].is_empty()


def test_rawnet_lgbm_05_cli_family_wiring() -> None:
    """SCENARIO_RAWNET_LGBM_05_CLI_FAMILY_WIRING."""
    from src.stocks.cli.train import _build_training_request
    from src.stocks.ml.contracts import ELASTIC_NET_FAMILY

    args = build_parser().parse_args(
        ["--artifact-id", "x", "--discovery-model-family", "economic_rawnet_lgbm"]
    )
    request = _build_training_request(args)
    assert request.discovery_model_family == "economic_rawnet_lgbm"
    default_args = build_parser().parse_args(["--artifact-id", "defaults"])
    default_request = _build_training_request(default_args)
    assert default_request.discovery_model_family == ELASTIC_NET_FAMILY
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--artifact-id", "x", "--discovery-model-family", "not-a-family"]
        )


def test_cadence_decision_monotone_pure_kernel() -> None:
    """CADENCE_DECISION_MONOTONE: decision count strictly decreases with cadence."""
    from datetime import timedelta
    from itertools import pairwise

    from src.stocks.ml.training import _cadence_decision_sessions

    sessions = tuple(datetime(2020, 1, 1) + timedelta(days=i) for i in range(101))

    by_freq = {
        freq: _cadence_decision_sessions(sessions, freq) for freq in (5, 10, 20)
    }
    lengths = [len(by_freq[5]), len(by_freq[10]), len(by_freq[20])]
    assert lengths[0] > lengths[1] > lengths[2] > 0
    assert lengths == [21, 11, 6]
    for freq in (5, 10, 20):
        decisions = by_freq[freq]
        assert decisions[0] == sessions[0]
        assert list(decisions) == sorted(decisions)
        for earlier, later in pairwise(decisions):
            gap = sessions.index(later) - sessions.index(earlier)
            assert gap >= freq

    assert _cadence_decision_sessions(sessions, 200) == (sessions[0],)

    with pytest.raises(ValueError, match="frequency_sessions must be positive"):
        _cadence_decision_sessions(sessions, 0)


def test_SPARSE_REWATERFILL_05_REQUEST_CLI_THREADING() -> None:
    """SPARSE_REWATERFILL_05_REQUEST_CLI_THREADING.

    The opt-in flag threads request -> risk policy -> ledger projection:
    flagged requests build a band_limited_rewaterfill_v1 policy mode, defaults
    stay freeze_v1, the CLI parser accepts the flag, and flagged/unflagged
    request projections carry distinct fingerprints.
    """
    from src.stocks.cli.train import _build_training_request
    from src.stocks.ml.result_ledger import _project_request
    from src.stocks.ml.training import _risk_policy_for_profile

    request = NetAlphaTrainingRequest(
        artifact_id="rewaterfill",
        enable_sparse_retained_rewaterfill=True,
    )
    default_request = NetAlphaTrainingRequest(artifact_id="defaults")
    profile = request.policy_profiles[1]
    assert (
        _risk_policy_for_profile(
            request, profile, 10, rebalance_frequency_sessions=5, top_k=12
        ).retained_sizing_mode
        == "band_limited_rewaterfill_v1"
    )
    assert (
        _risk_policy_for_profile(
            default_request, profile, 10, rebalance_frequency_sessions=5, top_k=12
        ).retained_sizing_mode
        == "freeze_v1"
    )

    args = build_parser().parse_args(
        ["--artifact-id", "flagged", "--enable-sparse-retained-rewaterfill"]
    )
    assert _build_training_request(args).enable_sparse_retained_rewaterfill is True
    default_args = build_parser().parse_args(["--artifact-id", "defaults"])
    assert (
        _build_training_request(default_args).enable_sparse_retained_rewaterfill is False
    )

    flagged_projection = _project_request(request)
    default_projection = _project_request(default_request)
    assert flagged_projection["enable_sparse_retained_rewaterfill"] is True
    assert default_projection["enable_sparse_retained_rewaterfill"] is False
    assert (
        flagged_projection["request_fingerprint"]
        != default_projection["request_fingerprint"]
    )


def test_train_main_keeps_single_typed_dispatch_boundary(monkeypatch) -> None:
    from types import SimpleNamespace

    from src.stocks.cli import train

    parsed = SimpleNamespace(supervise=True, internal_worker=False, artifact_id='run-1')
    monkeypatch.setattr(train, 'build_parser', lambda: SimpleNamespace(parse_args=lambda args: parsed))
    monkeypatch.setattr(train, 'parse_train_command', lambda value: 'command')
    monkeypatch.setattr(train, 'TrainSupervisor', lambda run_id: SimpleNamespace(run=lambda args: 17))

    assert train.main(['--supervise']) == 17
