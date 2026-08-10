"""Stock simulate CLI: resolve a research snapshot, compose, invoke simulation.

Simulation requires an explicit ``--snapshot-id``; there is no implicit newest
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
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import simulate_portfolio

logger = logging.getLogger("stocks.cli.simulate")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a stock simulation from an artifact")
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
    request = SimulationRequest(
        artifact_id=parsed.artifact_id,
        decision_time=decision_time,
        top_k=settings.top_k,
        max_single_weight=settings.max_single_weight,
        max_exposure=settings.max_exposure,
    )
    result = simulate_portfolio(composed, registry, request)
    logger.info(
        "final_value=%.2f total_return=%.4f", result.final_value, result.total_return
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
