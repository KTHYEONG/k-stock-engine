"""Stock simulation application: artifact -> score -> portfolio -> simulator."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.core.instruments import AssetKind
from src.core.portfolio import CostModel
from src.stocks.ml.artifacts import ModelArtifactRegistry, PredictionRequest
from src.stocks.ml.dataset import DatasetManifest
from src.stocks.simulation.runner import StockSimulator
from src.stocks.strategies.portfolio_policy import PortfolioPolicy


def run_simulation(
    registry: ModelArtifactRegistry,
    artifact_id: str,
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    decision_time: datetime,
) -> pl.DataFrame:
    """Load a stock artifact and score ``frame`` with it.

    The ``migration_wiring`` contract requires this application to load an
    artifact via ``registry.load``.
    """
    request = PredictionRequest(
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        decision_time=decision_time,
    )
    loaded = registry.load(artifact_id, request)
    scored = loaded.model.predict(frame)
    if scored.is_empty():
        raise ValueError("no rows scored")
    return scored


def run_simulation_with_policy(
    registry: ModelArtifactRegistry,
    artifact_id: str,
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    decision_time: datetime,
) -> object:
    """End-to-end simulation returning portfolio allocations."""
    scored = run_simulation(registry, artifact_id, frame, manifest, decision_time)
    policy = PortfolioPolicy(top_k=5, max_single_weight=0.2, max_exposure=1.0)
    simulator = StockSimulator(CostModel(commission_rate=0.00015, tax_rate=0.0023))
    return simulator.simulate(scored, policy, AssetKind.STOCK, price_frame=scored)


def main(args: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a stock simulation from an artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parsed = parser.parse_args(args)

    registry = ModelArtifactRegistry(parsed.registry)
    frame = pl.read_parquet(parsed.frame)
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="fixture",
        provider_version="fixture",
        universe_policy_version="v1",
        universe_policy_hash="universe-v1",
        feature_set="stock_alpha_v1",
        feature_set_hash="features-v1",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 12, 31, tzinfo=UTC),
        generated_time=datetime.now(UTC),
        row_count=frame.height,
    )
    run_simulation_with_policy(registry, parsed.artifact_id, frame, manifest, datetime.now(UTC))
    return 0
