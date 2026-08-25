"""Portfolio simulation workflow wiring tests."""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.domain.execution_policy import (
    SCHEDULED_OPEN_POLICY_ID,
    SCHEDULED_OPEN_V1,
)
from src.stocks.ml.contracts import NetAlphaTrainingRequest, policy_portfolio_fingerprint
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.trading.portfolio_constructor import CompoundingPolicyConfig
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import (
    artifact_policy_profile,
    simulate_portfolio,
)
from src.stocks.workflows.train_model import train_model
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def test_simulate_portfolio_reconciles_and_returns_metrics(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    train_model(
        snapshot,
        registry,
        NetAlphaTrainingRequest(
            artifact_id="stock_net_alpha_20240101",
            fold_count=2,
            candidate_horizon_sessions=(5,),
            bootstrap_resamples=50,
        ),
    )
    decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
    result = simulate_portfolio(
        snapshot,
        registry,
        SimulationRequest(
            artifact_id="stock_net_alpha_20240101", decision_time=decision
        ),
    )
    assert result.final_value > 0
    assert result.total_return is not None
    assert "cagr" in result.metrics


class _DummyModel:
    def __init__(self, manifest: ModelManifest) -> None:
        self._manifest = manifest

    def fit(self, train, validation) -> None:
        del train, validation

    def predict(self, frame):
        return frame

    def manifest(self) -> ModelManifest:
        return self._manifest


def _publish_policy_artifact(registry: ModelArtifactRegistry) -> str:
    payload = json.dumps(
        {
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
        },
        sort_keys=True,
    )
    manifest = ModelManifest(
        artifact_id="na_policy_artifact",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": payload},
    )
    registry.publish(_DummyModel(manifest), manifest)
    return manifest.artifact_id


def test_simulate_portfolio_rejects_divergent_policy_request(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=40, n_tickers=4)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    _publish_policy_artifact(registry)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
    # Default SimulationRequest caps (top_k=5) diverge from the artifact policy.
    with pytest.raises(ValueError, match="diverge from the artifact policy profile"):
        simulate_portfolio(
            snapshot,
            registry,
            SimulationRequest(artifact_id="na_policy_artifact", decision_time=decision),
        )
    # An explicit divergent band is rejected too.
    with pytest.raises(ValueError, match="no_trade_band_bps"):
        simulate_portfolio(
            snapshot,
            registry,
            SimulationRequest(
                artifact_id="na_policy_artifact",
                decision_time=decision,
                top_k=20,
                max_single_weight=0.08,
                max_exposure=0.9,
                participation_limit=0.005,
                no_trade_band_bps=5.0,
            ),
        )
    # An explicit divergent profile id is rejected too.
    with pytest.raises(ValueError, match="policy_profile_id"):
        simulate_portfolio(
            snapshot,
            registry,
            SimulationRequest(
                artifact_id="na_policy_artifact",
                decision_time=decision,
                top_k=20,
                max_single_weight=0.08,
                max_exposure=0.9,
                participation_limit=0.005,
                policy_profile_id="legacy_overlay_5bps",
            ),
        )


def test_artifact_policy_profile_matches_manifest(tmp_path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = _publish_policy_artifact(registry)
    profile = artifact_policy_profile(registry, artifact_id)
    assert profile is not None
    assert profile["profile_id"] == "lower_bound_only"
    assert profile["no_trade_band_bps"] == 0.0
    assert profile["top_k"] == 20
    assert profile["portfolio_fingerprint"] == policy_portfolio_fingerprint(
        20, 0.08, 0.9, 0.005
    )


def _prepared_equity_payload(
    *,
    execution_policy_hash: str = SCHEDULED_OPEN_V1.canonical_hash,
) -> str:
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    policy = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
    )
    return json.dumps(
        {
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v1",
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
            "execution_policy_hash": execution_policy_hash,
        },
        sort_keys=True,
    )


def _publish_prepared_equity_artifact(registry: ModelArtifactRegistry) -> str:
    manifest = ModelManifest(
        artifact_id="na_prepared_artifact",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": _prepared_equity_payload()},
    )
    registry.publish(_DummyModel(manifest), manifest)
    return manifest.artifact_id


def test_simulate_portfolio_rejects_divergent_prepared_equity_policy(tmp_path) -> None:
    """prepared-equity-v1 artifacts validate risk/execution fingerprints."""
    df = stock_net_alpha_composed_df(n_sessions=40, n_tickers=4)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    _publish_prepared_equity_artifact(registry)
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
    # Default SimulationRequest caps diverge from the artifact risk policy.
    with pytest.raises(ValueError, match="risk-policy fingerprint diverges"):
        simulate_portfolio(
            snapshot,
            registry,
            SimulationRequest(artifact_id="na_prepared_artifact", decision_time=decision),
        )
    # Matching caps but a foreign execution-policy hash still fails closed.
    divergent = ModelArtifactRegistry(tmp_path / "divergent")
    divergent_manifest = ModelManifest(
        artifact_id="na_divergent",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": _prepared_equity_payload(execution_policy_hash="deadbeef")},
    )
    divergent.publish(_DummyModel(divergent_manifest), divergent_manifest)
    with pytest.raises(ValueError, match="execution-policy hash diverges"):
        simulate_portfolio(
            snapshot,
            divergent,
            SimulationRequest(
                artifact_id="na_divergent",
                decision_time=decision,
                top_k=20,
                max_single_weight=0.08,
                max_exposure=0.9,
                participation_limit=0.005,
            ),
        )


def test_half_kelly_artifact_parity(tmp_path) -> None:
    """CGRA-05-half-kelly-artifact-parity"""
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    from src.stocks.workflows.simulate_portfolio import (
        _profile_growth_risk_aversion,
        _validate_prepared_equity_policy,
    )

    policy_aversion_2 = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        compounding=CompoundingPolicyConfig(growth_risk_aversion=2.0),
    )
    fp2 = stock_risk_policy_fingerprint(policy_aversion_2)

    policy_aversion_1 = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        compounding=CompoundingPolicyConfig(growth_risk_aversion=1.0),
    )
    fp1 = stock_risk_policy_fingerprint(policy_aversion_1)

    assert fp2 != fp1

    profile_v2 = {
        "profile_id": "lower_bound_half_kelly",
        "no_trade_band_bps": 0.0,
        "growth_risk_aversion": 2.0,
        "top_k": 20,
        "max_single_weight": 0.08,
        "max_exposure": 0.9,
        "participation_limit": 0.005,
        "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
        "execution_evidence_version": "prepared-equity-v1",
        "risk_policy_fingerprint": fp2,
        "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
    }
    aversion = _profile_growth_risk_aversion(profile_v2)
    assert aversion == 2.0

    reconstructed = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        compounding=CompoundingPolicyConfig(growth_risk_aversion=aversion),
    )
    assert stock_risk_policy_fingerprint(reconstructed) == fp2

    mismatched_profile = dict(profile_v2)
    mismatched_profile["risk_policy_fingerprint"] = fp1
    with pytest.raises(ValueError, match="risk-policy fingerprint diverges"):
        _validate_prepared_equity_policy(
            SimulationRequest(
                artifact_id="na_half_kelly",
                decision_time=datetime(2024, 4, 29, tzinfo=UTC),
                top_k=20,
                max_single_weight=0.08,
                max_exposure=0.9,
                participation_limit=0.005,
            ),
            mismatched_profile,
            0.0,
        )

    profile_missing = {
        "profile_id": "lower_bound_only",
        "no_trade_band_bps": 0.0,
        "top_k": 20,
        "max_single_weight": 0.08,
        "max_exposure": 0.9,
        "participation_limit": 0.005,
        "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
        "execution_evidence_version": "prepared-equity-v1",
        "risk_policy_fingerprint": fp1,
        "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
    }
    assert _profile_growth_risk_aversion(profile_missing) == 1.0


_SIMULATION_RANK_MODE_SCENARIO = "simulation_preserves_artifact_rank_mode"


def test_simulation_preserves_artifact_rank_mode() -> None:
    """v1 profile defaults to raw_score_v1; v2 profile uses economic_net_v1."""
    from src.stocks.workflows.simulate_portfolio import (
        _profile_economic_ranking_mode,
    )

    v1_profile = {
        "profile_id": "lower_bound_only",
        "no_trade_band_bps": 0.0,
        "top_k": 20,
        "max_single_weight": 0.08,
        "max_exposure": 0.9,
        "participation_limit": 0.005,
        "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
        "execution_evidence_version": "prepared-equity-v1",
        "risk_policy_fingerprint": "dummy",
        "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
    }
    assert _profile_economic_ranking_mode(v1_profile) == "raw_score_v1"

    v2_profile = {
        **v1_profile,
        "execution_evidence_version": "prepared-equity-v2-economic-rank",
        "economic_ranking_mode": "economic_net_v1",
        "risk_policy_fingerprint": "dummy_v2",
    }
    assert _profile_economic_ranking_mode(v2_profile) == "economic_net_v1"

    with pytest.raises(ValueError, match="economic_ranking_mode must be"):
        _profile_economic_ranking_mode({**v2_profile, "economic_ranking_mode": 42})

    with pytest.raises(ValueError, match="economic_ranking_mode must be"):
        _profile_economic_ranking_mode({**v2_profile, "economic_ranking_mode": "unknown"})

    with pytest.raises(ValueError, match="economic_ranking_mode is required"):
        _profile_economic_ranking_mode(
            {key: value for key, value in v2_profile.items() if key != "economic_ranking_mode"}
        )


def _v3_payload(horizon: int = 10) -> str:
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    policy = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=1.0,
            forecast_horizon_sessions=horizon,
        ),
        economic_ranking_mode="economic_net_v1",
    )
    return json.dumps(
        {
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "growth_risk_aversion": 1.0,
            "forecast_horizon_sessions": horizon,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v3-horizon-consistent",
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
            "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
            "economic_ranking_mode": "economic_net_v1",
        },
        sort_keys=True,
    )


def _publish_v3_artifact(
    registry: ModelArtifactRegistry, horizon: int = 10
) -> str:
    manifest = ModelManifest(
        artifact_id="na_v3_artifact",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=horizon,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": _v3_payload(horizon)},
    )
    registry.publish(_DummyModel(manifest), manifest)
    return manifest.artifact_id


def test_horizon_policy_v3_reconstructs_fingerprint(tmp_path) -> None:
    """HC_LOG_UTILITY_04_V3_SIMULATION_RECONSTRUCTS_OR_REJECTS: v3 artifact reconstructs the same fingerprint."""
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    from src.stocks.workflows.simulate_portfolio import (
        _policy_from_artifact,
        _profile_forecast_horizon_sessions,
    )

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = _publish_v3_artifact(registry, horizon=10)
    artifact_manifest = registry.read_manifest(artifact_id)
    request = SimulationRequest(
        artifact_id=artifact_id,
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    policy = _policy_from_artifact(artifact_manifest, request)
    assert policy.compounding.forecast_horizon_sessions == 10

    profile = json.loads(artifact_manifest.params["policy_profile"])
    assert _profile_forecast_horizon_sessions(profile) == 10

    expected = StockRiskPolicy(
        top_k=20,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=1.0,
            forecast_horizon_sessions=10,
        ),
        economic_ranking_mode="economic_net_v1",
    )
    assert stock_risk_policy_fingerprint(policy) == stock_risk_policy_fingerprint(expected)


def test_v3_manifest_without_horizon_raises(tmp_path) -> None:
    """HC_LOG_UTILITY_04_V3_SIMULATION_RECONSTRUCTS_OR_REJECTS: v3 manifest missing horizon raises ValueError."""
    from src.stocks.workflows.simulate_portfolio import _policy_from_artifact

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = _publish_v3_artifact(registry, horizon=10)
    artifact_manifest = registry.read_manifest(artifact_id)

    broken_profile = json.loads(artifact_manifest.params["policy_profile"])
    del broken_profile["forecast_horizon_sessions"]
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    broken_profile["risk_policy_fingerprint"] = stock_risk_policy_fingerprint(
        StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            participation_limit=0.005,
            no_trade_band_bps=0.0,
            compounding=CompoundingPolicyConfig(growth_risk_aversion=1.0),
            economic_ranking_mode="economic_net_v1",
        )
    )
    broken_manifest = ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": json.dumps(broken_profile, sort_keys=True)},
    )
    request = SimulationRequest(
        artifact_id=artifact_id,
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    with pytest.raises(ValueError, match="requires forecast_horizon_sessions"):
        _policy_from_artifact(broken_manifest, request)


def test_v3_manifest_non_integer_horizon_raises(tmp_path) -> None:
    """HC_LOG_UTILITY_04_V3_SIMULATION_RECONSTRUCTS_OR_REJECTS: v3 manifest with non-integer horizon raises ValueError."""
    from src.stocks.workflows.simulate_portfolio import (
        _profile_forecast_horizon_sessions,
    )

    with pytest.raises(ValueError, match="forecast_horizon_sessions must be a positive integer"):
        _profile_forecast_horizon_sessions({"forecast_horizon_sessions": "abc"})
    with pytest.raises(ValueError, match="forecast_horizon_sessions must be a positive integer"):
        _profile_forecast_horizon_sessions({"forecast_horizon_sessions": 0})
    with pytest.raises(ValueError, match="forecast_horizon_sessions must be a positive integer"):
        _profile_forecast_horizon_sessions({"forecast_horizon_sessions": -1})


def test_legacy_v2_replay_is_stable(tmp_path) -> None:
    """HC_LOG_UTILITY_05_LEGACY_V2_REPLAY_IS_STABLE: v2 artifact without forecast_horizon_sessions still works."""
    from src.stocks.workflows.simulate_portfolio import (
        _policy_from_artifact,
        _profile_forecast_horizon_sessions,
    )

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = _publish_prepared_equity_artifact(registry)
    artifact_manifest = registry.read_manifest(artifact_id)
    request = SimulationRequest(
        artifact_id=artifact_id,
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    policy = _policy_from_artifact(artifact_manifest, request)
    assert policy.compounding.forecast_horizon_sessions is None

    profile = json.loads(artifact_manifest.params["policy_profile"])
    assert _profile_forecast_horizon_sessions(profile) is None


DELTA_COST_UTILITY_06 = "DELTA_COST_UTILITY_06_ARTIFACT_PARITY_AND_LEGACY"


def test_delta_cost_utility_06_artifact_parity_and_legacy(tmp_path) -> None:
    """v4 invalid mode raises ValueError; v1-v3 reconstructs legacy; valid v4 reconstructs matching fingerprint."""
    from src.stocks.workflows.simulate_portfolio import (
        _policy_from_artifact,
        _profile_execution_utility_mode,
    )

    registry = ModelArtifactRegistry(tmp_path / "artifacts")

    # v1-v3 profile without execution_utility_mode -> legacy
    v1_profile = {
        "profile_id": "lower_bound_only",
        "no_trade_band_bps": 0.0,
        "top_k": 20,
        "max_single_weight": 0.08,
        "max_exposure": 0.9,
        "participation_limit": 0.005,
        "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
    }
    assert _profile_execution_utility_mode(v1_profile) == "legacy_target_interpolation_v1"

    # v4 with invalid mode
    v4_invalid = {**v1_profile, "execution_utility_mode": "unknown"}
    with pytest.raises(ValueError, match="execution_utility_mode must be"):
        _profile_execution_utility_mode(v4_invalid)

    # v4 with valid mode
    v4_valid = {**v1_profile, "execution_utility_mode": "delta_cost_aware_v1"}
    assert _profile_execution_utility_mode(v4_valid) == "delta_cost_aware_v1"

    # v4 requiring mode but missing it
    v4_missing = {
        **v1_profile,
        "execution_evidence_version": "prepared-equity-v4-delta-cost-aware",
    }
    with pytest.raises(ValueError, match="execution_utility_mode is required"):
        _profile_execution_utility_mode(v4_missing)

    # Publish a v4 artifact and verify fingerprint matches
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )

    v4_policy = StockRiskPolicy(
        top_k=20, gross_cap=0.9, single_name_cap=0.08,
        participation_limit=0.005, no_trade_band_bps=0.0,
        execution_utility_mode="delta_cost_aware_v1",
    )
    v4_payload = json.dumps({
        **v1_profile,
        "execution_utility_mode": "delta_cost_aware_v1",
        "execution_evidence_version": "prepared-equity-v4-delta-cost-aware",
        "risk_policy_fingerprint": stock_risk_policy_fingerprint(v4_policy),
        "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
    }, sort_keys=True)
    v4_manifest = ModelManifest(
        artifact_id="na_v4_artifact",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": v4_payload},
    )
    registry.publish(_DummyModel(v4_manifest), v4_manifest)
    artifact_manifest = registry.read_manifest("na_v4_artifact")
    request = SimulationRequest(
        artifact_id="na_v4_artifact",
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    policy = _policy_from_artifact(artifact_manifest, request)
    assert policy.execution_utility_mode == "delta_cost_aware_v1"
    assert stock_risk_policy_fingerprint(policy) == stock_risk_policy_fingerprint(v4_policy)

    # Mismatched fingerprint fails closed
    bad_payload = json.dumps({
        **v1_profile,
        "execution_utility_mode": "delta_cost_aware_v1",
        "execution_evidence_version": "prepared-equity-v4-delta-cost-aware",
        "risk_policy_fingerprint": "deadbeef",
        "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
    }, sort_keys=True)
    bad_manifest = ModelManifest(
        artifact_id="na_v4_bad",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": bad_payload},
    )
    registry.publish(_DummyModel(bad_manifest), bad_manifest)
    with pytest.raises(ValueError, match="risk-policy fingerprint diverges"):
        _policy_from_artifact(
            registry.read_manifest("na_v4_bad"),
            SimulationRequest(
                artifact_id="na_v4_bad",
                decision_time=datetime(2024, 4, 29, tzinfo=UTC),
                top_k=20,
                max_single_weight=0.08,
                max_exposure=0.9,
                participation_limit=0.005,
            ),
        )


def test_simulation_parity_growth_recovery(tmp_path) -> None:
    """GROWTH_RECOVERY_SIMULATION_PARITY_06.

    An independent simulation reconstructs the persisted v7 horizon-locked
    policy (active count, candidate pool, cadence, sizing mode, fingerprint)
    from the artifact payload, and a divergent stored v7 fingerprint fails
    closed with ValueError before the backtester runs.
    """
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )
    from src.stocks.workflows.simulate_portfolio import _policy_from_artifact

    def _v7_payload(horizon: int, fingerprint: str | None = None) -> str:
        policy = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            participation_limit=0.005,
            no_trade_band_bps=0.0,
            rebalance_frequency_sessions=horizon,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=1.0,
                forecast_horizon_sessions=horizon,
            ),
            economic_ranking_mode="economic_net_v1",
            execution_utility_mode="delta_cost_aware_v1",
            sizing_mode="confidence_mean_variance_v1",
        )
        effective = math.ceil(0.9 / 0.08)
        payload = {
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "growth_risk_aversion": 1.0,
            "forecast_horizon_sessions": horizon,
            "rebalance_frequency_sessions": horizon,
            "effective_active_count": effective,
            "candidate_pool_count": 2 * effective,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v7-horizon-locked",
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "v7_risk_policy_fingerprint": (
                fingerprint
                if fingerprint is not None
                else stock_risk_policy_fingerprint(policy)
            ),
            "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
            "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
            "economic_ranking_mode": "economic_net_v1",
            "execution_utility_mode": "delta_cost_aware_v1",
            "sizing_mode": "confidence_mean_variance_v1",
        }
        return json.dumps(payload, sort_keys=True)

    def _publish(registry: ModelArtifactRegistry, artifact_id: str, payload: str) -> str:
        manifest = ModelManifest(
            artifact_id=artifact_id,
            asset_kind=AssetKind.STOCK,
            feature_set="stock_net_alpha_v1",
            feature_schema_hash="h",
            universe_policy_hash="u",
            label_definition="net_alpha_o2o",
            label_horizon_sessions=10,
            eligible_from="2024-01-01T00:00:00+00:00",
            eligible_to="2024-04-29T00:00:00+00:00",
            model_type="net_alpha_elastic_net",
            params={"policy_profile": payload},
        )
        registry.publish(_DummyModel(manifest), manifest)
        return artifact_id

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    artifact_id = _publish(registry, "na_v7_artifact", _v7_payload(10))
    request = SimulationRequest(
        artifact_id=artifact_id,
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    policy = _policy_from_artifact(registry.read_manifest(artifact_id), request)
    assert policy.rebalance_frequency_sessions == 10
    assert policy.compounding.forecast_horizon_sessions == 10
    assert policy.sizing_mode == "confidence_mean_variance_v1"
    assert math.ceil(policy.gross_cap / policy.single_name_cap) == 12

    registry_divergent = ModelArtifactRegistry(tmp_path / "divergent")
    _publish(
        registry_divergent,
        "na_v7_bad",
        _v7_payload(10, fingerprint="deadbeef"),
    )
    with pytest.raises(ValueError, match="fingerprint diverges"):
        _policy_from_artifact(
            registry_divergent.read_manifest("na_v7_bad"),
            request,
        )


ML_COMPOUNDING_01_V5_ARTIFACT_CADENCE_PARITY = (
    "ML_COMPOUNDING_01_V5_ARTIFACT_CADENCE_PARITY"
)


def test_v5_artifact_cadence_parity(tmp_path) -> None:
    """ML_COMPOUNDING_01_V5_ARTIFACT_CADENCE_PARITY.

    A v5 sparse profile persisted with H=10 and C=10 reconstructs a policy
    whose rebalance_frequency_sessions is 10 and whose fingerprint equals the
    stored risk_policy_fingerprint; a missing/non-integral/mismatched cadence
    raises ValueError before replay.
    """
    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )
    from src.stocks.workflows.simulate_portfolio import (
        _policy_from_artifact,
        _profile_rebalance_frequency_sessions,
    )

    def _v5_payload(cadence: int = 10) -> str:
        policy = StockRiskPolicy(
            top_k=20,
            gross_cap=0.9,
            single_name_cap=0.08,
            participation_limit=0.005,
            no_trade_band_bps=0.0,
            rebalance_frequency_sessions=cadence,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=1.0,
                forecast_horizon_sessions=10,
            ),
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
        )
        return json.dumps({
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v5-sparse-growth",
            "rebalance_frequency_sessions": cadence,
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
            "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
            "execution_utility_mode": "sparse_hold_replace_v2",
            "sizing_mode": "risk_balanced_waterfill_v2",
            "growth_risk_aversion": 1.0,
            "forecast_horizon_sessions": 10,
        }, sort_keys=True)

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    manifest = ModelManifest(
        artifact_id="na_v5_cadence",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": _v5_payload(10)},
    )
    registry.publish(_DummyModel(manifest), manifest)
    request = SimulationRequest(
        artifact_id="na_v5_cadence",
        decision_time=datetime(2024, 4, 29, tzinfo=UTC),
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
    )
    policy = _policy_from_artifact(registry.read_manifest("na_v5_cadence"), request)
    assert policy.rebalance_frequency_sessions == 10
    expected = StockRiskPolicy(
        top_k=20, gross_cap=0.9, single_name_cap=0.08,
        participation_limit=0.005, no_trade_band_bps=0.0,
        rebalance_frequency_sessions=10,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=1.0, forecast_horizon_sessions=10,
        ),
        execution_utility_mode="sparse_hold_replace_v2",
        sizing_mode="risk_balanced_waterfill_v2",
    )
    assert stock_risk_policy_fingerprint(policy) == stock_risk_policy_fingerprint(expected)

    assert _profile_rebalance_frequency_sessions({"rebalance_frequency_sessions": 10}) == 10
    assert _profile_rebalance_frequency_sessions({}) == 5
    with pytest.raises(ValueError, match="positive integer"):
        _profile_rebalance_frequency_sessions({"rebalance_frequency_sessions": "abc"})
    with pytest.raises(ValueError, match="positive integer"):
        _profile_rebalance_frequency_sessions({"rebalance_frequency_sessions": 0})

    bad_manifest = ModelManifest(
        artifact_id="na_v5_bad_cadence",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
        params={"policy_profile": json.dumps({
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "top_k": 20,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(20, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v5-sparse-growth",
            "rebalance_frequency_sessions": "invalid",
            "risk_policy_fingerprint": "deadbeef",
            "execution_policy_id": SCHEDULED_OPEN_POLICY_ID,
            "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
            "execution_utility_mode": "sparse_hold_replace_v2",
            "sizing_mode": "risk_balanced_waterfill_v2",
            "growth_risk_aversion": 1.0,
            "forecast_horizon_sessions": 10,
        }, sort_keys=True)},
    )
    registry_bad = ModelArtifactRegistry(tmp_path / "bad")
    registry_bad.publish(_DummyModel(bad_manifest), bad_manifest)
    with pytest.raises(ValueError, match="positive integer"):
        _policy_from_artifact(
            registry_bad.read_manifest("na_v5_bad_cadence"), request,
        )


def test_execution_frontier_artifact_simulation_parity() -> None:
    """ML_EXEC_FRONTIER_03_ARTIFACT_SIMULATION_PARITY.

    An artifact selected with H=20, C=5, K=12 reconstructs a simulator policy
    with forecast_horizon_sessions=20, rebalance_frequency_sessions=5, top_k=12,
    and an identical risk-policy fingerprint; a K mismatch raises ValueError
    before replay.
    """
    from dataclasses import replace

    from src.stocks.ml.contracts import policy_portfolio_fingerprint
    from src.stocks.trading.portfolio_constructor import (
        CompoundingPolicyConfig,
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )
    from src.stocks.workflows.simulate_portfolio import _policy_from_artifact

    request = SimulationRequest(
        artifact_id="exec_frontier_artifact",
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=12,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
        portfolio_value=100_000_000.0,
        initial_cash=100_000_000.0,
        policy_profile_id="exec_frontier_profile",
        no_trade_band_bps=0.0,
    )
    policy = StockRiskPolicy(
        top_k=12,
        gross_cap=0.9,
        single_name_cap=0.08,
        participation_limit=0.005,
        no_trade_band_bps=0.0,
        rebalance_frequency_sessions=5,
        compounding=CompoundingPolicyConfig(
            growth_risk_aversion=1.0, forecast_horizon_sessions=20
        ),
        economic_ranking_mode="economic_net_v1",
        execution_utility_mode="sparse_hold_replace_v2",
        sizing_mode="risk_balanced_waterfill_v2",
    )
    fingerprint = stock_risk_policy_fingerprint(policy)
    profile_payload = {
        "profile_id": "exec_frontier_profile",
        "no_trade_band_bps": 0.0,
        "growth_risk_aversion": 1.0,
        "forecast_horizon_sessions": 20,
        "rebalance_frequency_sessions": 5,
        "top_k": 12,
        "max_single_weight": 0.08,
        "max_exposure": 0.9,
        "participation_limit": 0.005,
        "portfolio_fingerprint": policy_portfolio_fingerprint(12, 0.08, 0.9, 0.005),
        "execution_evidence_version": "prepared-equity-v5-sparse-growth",
        "risk_policy_fingerprint": fingerprint,
        "v7_risk_policy_fingerprint": fingerprint,
        "execution_policy_id": SCHEDULED_OPEN_V1.policy_id,
        "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
        "economic_ranking_mode": "economic_net_v1",
        "execution_utility_mode": "sparse_hold_replace_v2",
        "sizing_mode": "risk_balanced_waterfill_v2",
    }
    base_manifest = ModelManifest(
        artifact_id="exec_frontier_artifact",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=20,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
    )
    manifest = replace(
        base_manifest,
        params={
            "policy_profile": json.dumps(profile_payload),
            "holm_gate_version": "v6",
        },
    )
    reconstructed = _policy_from_artifact(manifest, request)
    assert reconstructed.compounding.forecast_horizon_sessions == 20
    assert reconstructed.rebalance_frequency_sessions == 5
    assert reconstructed.top_k == 12
    assert stock_risk_policy_fingerprint(reconstructed) == fingerprint

    mismatched = replace(request, top_k=20)
    with pytest.raises(ValueError, match="risk-policy fingerprint"):
        _policy_from_artifact(manifest, mismatched)


def test_SPARSE_REWATERFILL_06_SIM_RECONSTRUCTION_ROUNDTRIP() -> None:
    """SPARSE_REWATERFILL_06_SIM_RECONSTRUCTION_ROUNDTRIP.

    persisted retained_sizing_mode round-trips through _policy_from_artifact
    and binds the stored fingerprint; an absent key reconstructs freeze_v1.
    """
    from dataclasses import replace

    from src.stocks.trading.portfolio_constructor import (
        StockRiskPolicy,
        stock_risk_policy_fingerprint,
    )
    from src.stocks.workflows.simulate_portfolio import _policy_from_artifact

    request = SimulationRequest(
        artifact_id="rewaterfill_rt",
        decision_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=12,
        max_single_weight=0.08,
        max_exposure=0.9,
        participation_limit=0.005,
        portfolio_value=100_000_000.0,
        initial_cash=100_000_000.0,
        no_trade_band_bps=0.0,
    )
    base_manifest = ModelManifest(
        artifact_id="rewaterfill_rt",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="h",
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-04-29T00:00:00+00:00",
        model_type="net_alpha_elastic_net",
    )

    def manifest_with_mode(mode: str | None) -> ModelManifest:
        policy = StockRiskPolicy(
            top_k=12,
            gross_cap=0.9,
            single_name_cap=0.08,
            participation_limit=0.005,
            no_trade_band_bps=0.0,
            rebalance_frequency_sessions=5,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=1.0, forecast_horizon_sessions=10
            ),
            economic_ranking_mode="economic_net_v1",
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
            **({} if mode is None else {"retained_sizing_mode": mode}),
        )
        payload: dict[str, object] = {
            "profile_id": "lower_bound_only",
            "no_trade_band_bps": 0.0,
            "growth_risk_aversion": 1.0,
            "forecast_horizon_sessions": 10,
            "rebalance_frequency_sessions": 5,
            "top_k": 12,
            "max_single_weight": 0.08,
            "max_exposure": 0.9,
            "participation_limit": 0.005,
            "portfolio_fingerprint": policy_portfolio_fingerprint(12, 0.08, 0.9, 0.005),
            "execution_evidence_version": "prepared-equity-v5-sparse-growth",
            "risk_policy_fingerprint": stock_risk_policy_fingerprint(policy),
            "execution_policy_id": SCHEDULED_OPEN_V1.policy_id,
            "execution_policy_hash": SCHEDULED_OPEN_V1.canonical_hash,
            "economic_ranking_mode": "economic_net_v1",
            "execution_utility_mode": "sparse_hold_replace_v2",
            "sizing_mode": "risk_balanced_waterfill_v2",
        }
        if mode is not None:
            payload["retained_sizing_mode"] = mode
        return replace(
            base_manifest,
            params={"policy_profile": json.dumps(payload)},
        )

    flagged = _policy_from_artifact(manifest_with_mode("band_limited_rewaterfill_v1"), request)
    assert flagged.retained_sizing_mode == "band_limited_rewaterfill_v1"
    assert stock_risk_policy_fingerprint(flagged) == stock_risk_policy_fingerprint(
        StockRiskPolicy(
            top_k=12,
            gross_cap=0.9,
            single_name_cap=0.08,
            participation_limit=0.005,
            no_trade_band_bps=0.0,
            rebalance_frequency_sessions=5,
            compounding=CompoundingPolicyConfig(
                growth_risk_aversion=1.0, forecast_horizon_sessions=10
            ),
            economic_ranking_mode="economic_net_v1",
            execution_utility_mode="sparse_hold_replace_v2",
            sizing_mode="risk_balanced_waterfill_v2",
            retained_sizing_mode="band_limited_rewaterfill_v1",
        )
    )
    legacy = _policy_from_artifact(manifest_with_mode(None), request)
    assert legacy.retained_sizing_mode == "freeze_v1"
