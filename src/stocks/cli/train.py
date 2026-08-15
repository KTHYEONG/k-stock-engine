"""Stock train CLI: resolve a net-alpha snapshot, compose, and train the mainline.

The only supported training path is the canonical ``stock_net_alpha_v1``
mainline. A snapshot or artifact that does not satisfy the net-alpha contract
raises one actionable ``ValueError`` naming the materialization CLI; there is
no legacy LambdaRank/Optuna flag, no implicit fallback, and no fixed 5/10/15
route.
"""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.costs import (
    CostSchedule,
    LiquiditySlippageModel,
    default_base_schedule,
    default_stress_schedule,
)
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
from src.stocks.ml.contracts import (
    DEFAULT_CANDIDATE_HORIZON_SESSIONS,
    NetAlphaTrainingRequest,
    PortfolioSettings,
    RiskSettings,
)
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.training import train_net_alpha_model
from src.stocks.research.artifacts import ModelArtifactRegistry

logger = logging.getLogger("stocks.cli.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a net-alpha model artifact")
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
        "--candidate-horizon-sessions",
        type=str,
        default=",".join(str(h) for h in DEFAULT_CANDIDATE_HORIZON_SESSIONS),
        help=(
            "pre-registered discovery grid of horizon session counts "
            f"(default {DEFAULT_CANDIDATE_HORIZON_SESSIONS})"
        ),
    )
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--embargo-sessions", type=int, default=5)
    parser.add_argument("--forward-holdout-sessions", type=int, default=0)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument(
        "--max-rss-mib",
        type=int,
        default=None,
        help="explicit RSS budget in MiB; a breach publishes complete NO_TRADE evidence",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cost-schedule",
        choices=("base", "stress"),
        default="base",
        help=(
            "effective-dated cost schedule resolved from the snapshot cost "
            "evidence when present, else the canonical base/stress schedule"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-single-weight", type=float, default=0.08)
    parser.add_argument("--max-exposure", type=float, default=0.90)
    parser.add_argument("--participation-limit", type=float, default=0.005)
    parser.add_argument("--portfolio-value", type=float, default=100_000_000.0)
    parser.add_argument("--reference-notional", type=float, default=100_000_000.0)
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    return parser


def _parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            "candidate-horizon-sessions must be comma-separated integers"
        ) from exc
    if not values:
        raise ValueError("candidate-horizon-sessions must be non-empty")
    return values


def _resolve_cost_context(
    snapshot: object, cost_schedule: str
) -> tuple[CostSchedule, LiquiditySlippageModel | None]:
    """Resolve one effective-dated cost schedule plus liquidity model.

    The snapshot's hash-bound cost evidence is the preferred source for both;
    without it the canonical base/stress schedules are used and the liquidity
    model is left unset (replay then fails closed on realized outcomes).
    """
    costs = getattr(snapshot, "costs", None)
    if costs is not None:
        research_range = getattr(snapshot, "research_range", None)
        if research_range is None:
            raise ValueError("snapshot cost evidence requires a research_range")
        evidence = load_cost_evidence(Path(costs.path), research_range)
        if cost_schedule == "stress":
            return evidence.stress_schedule(), evidence.stress_liquidity_model
        return evidence.base_schedule(), evidence.base_liquidity_model
    if cost_schedule == "stress":
        return default_stress_schedule(), None
    return default_base_schedule(), None


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

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
            feature_set="stock_net_alpha_v1",
            decision_time=decision_time,
        )
    except ValueError as exc:
        message = str(exc)
        if "feature_set mismatch" in message or "net-alpha" in message:
            raise ValueError(
                f"snapshot {parsed.snapshot_id} does not satisfy the "
                f"stock_net_alpha_v1 contract ({message}). Materialize a "
                "net-alpha snapshot first via "
                "`python -m src.stocks.cli.build_research --pipeline net-alpha "
                f"--source-snapshot-id {parsed.snapshot_id} --snapshot-id <id> "
                "--feature-dataset-id <id> --label-dataset-id <id>`."
            ) from exc
        raise
    data = compose_net_alpha_training_data(
        composed,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )

    base_cost_schedule, liquidity_model = _resolve_cost_context(
        snapshot, parsed.cost_schedule
    )
    registry = ModelArtifactRegistry(parsed.registry)
    request = NetAlphaTrainingRequest(
        artifact_id=parsed.artifact_id,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
        fold_count=parsed.fold_count,
        embargo_sessions=parsed.embargo_sessions,
        forward_holdout_sessions=parsed.forward_holdout_sessions,
        bootstrap_alpha=parsed.bootstrap_alpha,
        bootstrap_resamples=parsed.bootstrap_resamples,
        model_threads=parsed.model_threads,
        max_rss_mib=parsed.max_rss_mib,
        seed=parsed.seed,
        portfolio=PortfolioSettings(
            top_k=parsed.top_k,
            max_single_weight=parsed.max_single_weight,
            max_exposure=parsed.max_exposure,
            participation_limit=parsed.participation_limit,
            portfolio_value=parsed.portfolio_value,
            initial_cash=parsed.portfolio_value,
            reference_notional=parsed.reference_notional,
        ),
        risk=RiskSettings(),
        base_cost_schedule=base_cost_schedule,
        liquidity_model=liquidity_model,
    )
    logger.info(
        "training net-alpha mainline artifact %s over candidate horizons %s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )
    manifest = train_net_alpha_model(data, registry, request)
    logger.info("published artifact %s (%s)", manifest.artifact_id, manifest.model_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
