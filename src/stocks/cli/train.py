"""Stock train CLI: resolve a research snapshot, compose, invoke train_model.

Training requires an explicit ``--snapshot-id``; there is no implicit newest
selection. Provisional snapshots are rejected for paper/live modes; evidence
incomplete snapshots are rejected by the snapshot resolver.
"""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_CANONICAL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_DERIVED_ROOT,
)
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import DEFAULT_STOCK_ALPHA
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.train_model import train_model

logger = logging.getLogger("stocks.cli.train")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a stock baseline model artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--snapshot-id", required=True, help="immutable research snapshot id")
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_CANONICAL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_DERIVED_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_CANONICAL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("research", "paper", "live"),
        default="research",
        help="paper/live modes reject provisional snapshots",
    )
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    parsed = parser.parse_args(args)

    settings = DEFAULT_STOCK_ALPHA
    decision_time = parsed.decision_time or datetime.now(UTC)
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    composed = repository.compose_training_snapshot(
        snapshot,
        feature_set=settings.feature_set,
        decision_time=decision_time,
    )

    registry = ModelArtifactRegistry(parsed.registry)
    request = TrainingRequest(
        artifact_id=parsed.artifact_id,
        n_folds=settings.n_folds,
        embargo_sessions=settings.embargo_sessions,
    )
    manifest = train_model(composed, registry, request)
    logger.info("published artifact %s", manifest.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
