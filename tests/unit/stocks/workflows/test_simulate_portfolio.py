"""Portfolio simulation workflow wiring tests."""
from __future__ import annotations

import json
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
