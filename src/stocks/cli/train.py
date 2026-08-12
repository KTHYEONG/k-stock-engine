"""Stock train CLI: resolve a research snapshot, compose, invoke train_model.

Training requires an explicit ``--snapshot-id``; there is no implicit newest
selection. Provisional snapshots are rejected for paper/live modes; evidence
incomplete snapshots are rejected by the snapshot resolver. Training is v2-only:
a snapshot that does not satisfy the ``stock_alpha_v2`` feature/label contract
raises one actionable error naming the materialization CLI instead of silently
downgrading to v1 or recomputing labels.
"""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
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

STOCK_ALPHA_V2_FEATURE_SET = "stock_alpha_v2"
_V2_CONTRACT_MISMATCH_MARKERS = (
    "feature_set mismatch",
    "has feature_set",
    "label_definition",
    "residual_o2o_5d",
    "relevance",
    "v2 composition",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a stock baseline model artifact")
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
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=80,
        help="number of serial LambdaRank search trials (default 80)",
    )
    parser.add_argument(
        "--max-rss-mib",
        type=int,
        default=None,
        help="explicit RSS budget in MiB; a breach raises TrainingCapacityError",
    )
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
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
    try:
        composed = repository.compose_labeled_training_snapshot(
            snapshot,
            feature_set=STOCK_ALPHA_V2_FEATURE_SET,
            decision_time=decision_time,
        )
    except ValueError as exc:
        message = str(exc)
        if any(marker in message for marker in _V2_CONTRACT_MISMATCH_MARKERS):
            raise ValueError(
                f"snapshot {parsed.snapshot_id} does not satisfy the "
                f"{STOCK_ALPHA_V2_FEATURE_SET} contract ({message}). "
                f"Materialize a v2 snapshot first via "
                f"`python -m src.stocks.cli.build_research_v2 "
                f"--source-snapshot-id {parsed.snapshot_id} --snapshot-id <id> "
                f"--feature-dataset-id <id> --label-dataset-id <id>`."
            ) from exc
        raise

    registry = ModelArtifactRegistry(parsed.registry)
    request = TrainingRequest(
        artifact_id=parsed.artifact_id,
        n_folds=settings.n_folds,
        embargo_sessions=settings.embargo_sessions,
        top_k=20,
        max_single_weight=0.08,
        max_exposure=0.90,
        participation_limit=0.005,
        optuna_trials=parsed.optuna_trials,
        max_rss_mib=parsed.max_rss_mib,
    )
    logger.info(
        "frozen candidate route set: holding horizons %s, per-horizon trial "
        "budget %s",
        list(request.candidate_horizons),
        request.optuna_trials // len(request.candidate_horizons),
    )
    manifest = train_model(composed, registry, request)
    logger.info("published artifact %s", manifest.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
