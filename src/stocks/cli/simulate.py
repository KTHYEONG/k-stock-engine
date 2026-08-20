"""Stock simulate CLI: resolve a research snapshot, compose, invoke simulation.

Simulation requires an explicit ``--snapshot-id``; there is no implicit newest
selection. Provisional snapshots are rejected for paper/live modes; evidence
incomplete snapshots are rejected by the snapshot resolver.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.data.costs import load_cost_evidence
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import DEFAULT_STOCK_ALPHA, REFERENCE_DATETIME
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import (
    artifact_policy_profile,
    simulate_portfolio,
)

logger = logging.getLogger("stocks.cli.simulate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stock simulation from an artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--snapshot-id", required=True, help="immutable research snapshot id")
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("research", "paper", "live"),
        default="research",
        help="paper/live modes reject provisional snapshots",
    )
    parser.add_argument("--feature-set", default=None, help="feature set identifier")
    parser.add_argument(
        "--decision-time",
        type=datetime.fromisoformat,
        default=REFERENCE_DATETIME,
        help="decision timestamp (default: 2026-03-10T06:30:00+00:00)",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    settings = DEFAULT_STOCK_ALPHA
    decision_time = parsed.decision_time or REFERENCE_DATETIME
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    feature_set = parsed.feature_set or "stock_net_alpha_v1"
    composed = repository.compose_training_snapshot(
        snapshot,
        feature_set=feature_set,
        decision_time=decision_time,
    )

    registry = ModelArtifactRegistry(parsed.registry)
    cost_evidence = None
    if snapshot.costs is not None:
        cost_evidence = load_cost_evidence(
            Path(snapshot.costs.path), snapshot.execution_range
        )
    policy_profile = artifact_policy_profile(registry, parsed.artifact_id)
    if policy_profile is not None:
        request = SimulationRequest(
            artifact_id=parsed.artifact_id,
            decision_time=decision_time,
            top_k=cast(int, policy_profile["top_k"]),
            max_single_weight=cast(float, policy_profile["max_single_weight"]),
            max_exposure=cast(float, policy_profile["max_exposure"]),
            participation_limit=cast(float, policy_profile["participation_limit"]),
            policy_profile_id=cast(str, policy_profile["profile_id"]),
            no_trade_band_bps=cast(float, policy_profile["no_trade_band_bps"]),
        )
    else:
        request = SimulationRequest(
            artifact_id=parsed.artifact_id,
            decision_time=decision_time,
            top_k=settings.top_k,
            max_single_weight=settings.max_single_weight,
            max_exposure=settings.max_exposure,
        )
    result = simulate_portfolio(composed, registry, request, cost_evidence=cost_evidence)
    logger.info(
        "final_value=%.2f total_return=%.4f", result.final_value, result.total_return
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
