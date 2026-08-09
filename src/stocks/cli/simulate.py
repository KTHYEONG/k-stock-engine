"""Stock simulate CLI: parse args, load snapshot, invoke simulation workflow."""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.paths import STOCK_ARTIFACT_ROOT, STOCK_DATASET_ROOT
from src.stocks.data.repositories import StockDatasetRepository
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import DEFAULT_STOCK_ALPHA
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.cli.simulate")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a stock simulation from an artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=STOCK_DATASET_ROOT)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    parsed = parser.parse_args(args)

    settings = DEFAULT_STOCK_ALPHA
    decision_time = parsed.decision_time or datetime.now(UTC)
    repository = StockDatasetRepository(ParquetDatasetStore(parsed.dataset_root))
    snapshot = repository.read(parsed.dataset_id, settings.feature_set, decision_time)

    registry = ModelArtifactRegistry(parsed.registry)
    request = SimulationRequest(
        artifact_id=parsed.artifact_id,
        decision_time=decision_time,
        top_k=settings.top_k,
        max_single_weight=settings.max_single_weight,
        max_exposure=settings.max_exposure,
    )
    result = simulate_portfolio(snapshot, registry, request)
    logger.info(
        "final_value=%.2f total_return=%.4f", result.final_value, result.total_return
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
