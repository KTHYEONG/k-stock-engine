"""Stock train CLI: parse args, load snapshot, invoke train_model workflow."""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.paths import STOCK_ARTIFACT_ROOT, STOCK_DATASET_ROOT
from src.stocks.data.repositories import StockDatasetRepository
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import DEFAULT_STOCK_ALPHA
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.cli.train")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a stock baseline model artifact")
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
    request = TrainingRequest(
        artifact_id=parsed.artifact_id,
        n_folds=settings.n_folds,
        embargo_sessions=settings.embargo_sessions,
    )
    manifest = train_model(snapshot, registry, request)
    logger.info("published artifact %s", manifest.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
