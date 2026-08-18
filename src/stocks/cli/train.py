"""Stock train CLI: resolve a net-alpha snapshot, compose, and train the mainline.

The only supported training path is the canonical ``stock_net_alpha_v1``
mainline. A snapshot or artifact that does not satisfy the net-alpha contract
raises one actionable ``ValueError`` naming the materialization CLI; there is
no legacy LambdaRank/Optuna flag, no implicit fallback, and no fixed 5/10/15
route.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from src.core.costs import (
    CostSchedule,
    LiquiditySlippageModel,
    default_base_schedule,
    default_stress_schedule,
)
from src.core.datasets import DatasetCertification
from src.core.paths import (
    PROJECT_ROOT,
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.data.contracts import CoverageRange, DatasetSnapshot
from src.stocks.data.costs import load_cost_evidence
from src.stocks.data.lineage import ResearchDataBundle, ResolvedDataLineage
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
from src.stocks.ml.data import compose_net_alpha_training_data, validate_ml_market_data
from src.stocks.ml.result_ledger import (
    CostRunContext,
    MlResultLedger,
    MlRunContext,
)
from src.stocks.ml.training import train_net_alpha_model
from src.stocks.research.artifacts import ModelArtifactRegistry

logger = logging.getLogger("stocks.cli.train")

STOCK_RESULTS_DOC_ROOT = PROJECT_ROOT / "docs" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a net-alpha model artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="immutable research snapshot id (legacy path; prefer --as-of)",
    )
    parser.add_argument(
        "--as-of",
        type=datetime.fromisoformat,
        default=None,
        help="resolve datasets at this UTC timestamp (direct catalog selection)",
    )
    parser.add_argument(
        "--research-start",
        type=date.fromisoformat,
        default=date(2016, 1, 4),
        help="inclusive research data start date for direct selection",
    )
    parser.add_argument(
        "--research-end",
        type=date.fromisoformat,
        default=None,
        help="inclusive research data end date for direct selection",
    )
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=STOCK_RESULTS_DOC_ROOT,
        help="directory owning the generated result ledger (default docs/results)",
    )
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
    parser.add_argument("--model-threads", type=int, default=4)
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
            "labels the run's reference schedule kind in the result ledger; "
            "both base and stress schedules (with matching liquidity models) "
            "are always resolved from the same snapshot cost evidence, and the "
            "base schedule remains the only schedule used for fitting and "
            "calibration"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-single-weight", type=float, default=0.08)
    parser.add_argument("--max-exposure", type=float, default=0.90)
    parser.add_argument("--participation-limit", type=float, default=0.005)
    parser.add_argument("--portfolio-value", type=float, default=100_000_000.0)
    parser.add_argument("--reference-notional", type=float, default=100_000_000.0)
    parser.add_argument("--decision-time", type=datetime.fromisoformat, default=None)
    parser.add_argument(
        "--base-dataset-id",
        default=None,
        help="direct base dataset ID (bypasses snapshot resolution)",
    )
    parser.add_argument(
        "--feature-dataset-id",
        default=None,
        help="direct feature dataset ID (bypasses snapshot resolution)",
    )
    parser.add_argument(
        "--label-dataset-id",
        default=None,
        help="direct label dataset ID (bypasses snapshot resolution)",
    )
    parser.add_argument(
        "--research-start-direct",
        type=date.fromisoformat,
        default=None,
        help="start date for direct dataset loading (requires --base-dataset-id)",
    )
    parser.add_argument(
        "--research-end-direct",
        type=date.fromisoformat,
        default=None,
        help="end date for direct dataset loading (requires --base-dataset-id)",
    )
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


def _resolve_cost_contexts(
    snapshot: object,
) -> tuple[
    CostSchedule,
    LiquiditySlippageModel | None,
    CostSchedule,
    LiquiditySlippageModel | None,
]:
    """Resolve the base and stress cost schedules plus their liquidity models.

    Both schedules are resolved from the same hash-bound cost evidence when the
    snapshot provides it, so base/stress certification shares one evidence
    identity. Without evidence the canonical base/stress schedules are used and
    the liquidity models are left unset (replay then fails closed on realized
    outcomes). The base schedule remains the only schedule permitted for fitting
    and calibration.
    """
    costs = getattr(snapshot, "costs", None)
    if costs is not None:
        execution_range = getattr(snapshot, "execution_range", None)
        if execution_range is None:
            raise ValueError("snapshot cost evidence requires an execution_range")
        evidence = load_cost_evidence(Path(costs.path), execution_range)
        return (
            evidence.base_schedule(),
            evidence.base_liquidity_model,
            evidence.stress_schedule(),
            evidence.stress_liquidity_model,
        )
    return (
        default_base_schedule(),
        None,
        default_stress_schedule(),
        None,
    )


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    # Direct dataset loading path
    if parsed.base_dataset_id and parsed.feature_dataset_id and parsed.label_dataset_id:
        return _run_direct_training(parsed, parser)
    
    # Legacy snapshot/as-of path
    if not parsed.snapshot_id and not parsed.as_of:
        parser.error("either --snapshot-id or --as-of is required")

    decision_time = parsed.decision_time or datetime.now(UTC)
    started_at = datetime.now(UTC)
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )

    data_lineage_json: dict[str, object] | None = None
    resolved_lineage: ResolvedDataLineage | None = None
    composed: DatasetSnapshot | ResearchDataBundle
    if parsed.as_of:
        from src.stocks.data.catalog import CatalogStore
        from src.stocks.data.lineage import (
            CatalogCompatibilityResolver,
            DataSelectionRequest,
        )

        store = CatalogStore(parsed.catalog_root)
        resolver = CatalogCompatibilityResolver(store)
        as_of = parsed.as_of
        if not isinstance(as_of, datetime):
            raise ValueError("--as-of must be a datetime")
        selection = DataSelectionRequest(
            asset_kind="stock",
            feature_set="stock_net_alpha_v1",
            label_definition="net_alpha_o2o",
            candidate_horizons=tuple(
                int(h)
                for h in parsed.candidate_horizon_sessions.split(",")
                if h.strip()
            ),
            as_of=as_of,
            research_range=CoverageRange(
                start=parsed.research_start,
                end=parsed.research_end or as_of.date(),
            ),
            minimum_outcome_coverage=0.0,
            required_certification=DatasetCertification.PROVISIONAL,
        )
        lineage = resolver.resolve(selection)
        resolved_lineage = lineage
        data_lineage_json = lineage.to_json()
        bundle = repository.compose_labeled_training_data(
            lineage,
            feature_set="stock_net_alpha_v1",
            decision_time=decision_time,
        )
        if not isinstance(bundle, ResearchDataBundle):
            raise TypeError("direct selection must return a ResearchDataBundle")
        composed = bundle
        snapshot = None
    else:
        snapshot = resolve_snapshot_for_mode(
            parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
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
    logger.info(
        "[DATA] stage=compose feature_rows=%d instruments=%d sessions=%d columns=%d",
        data.feature_frame.height,
        data.feature_frame["instrument_id"].n_unique(),
        data.feature_frame["session"].n_unique(),
        len(data.feature_frame.columns),
    )

    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(snapshot)
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
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
    )
    costs = getattr(snapshot, "costs", None) if snapshot is not None else None
    cost_context = CostRunContext(
        cost_schedule_kind=parsed.cost_schedule,
        cost_evidence_path=Path(costs.path).name if costs is not None else None,
        cost_evidence_hash=getattr(costs, "content_hash", None),
        has_liquidity_model=liquidity_model is not None,
    )
    snapshot_id_or_lineage = (
        json.dumps(data_lineage_json, sort_keys=True)
        if data_lineage_json is not None
        else (parsed.snapshot_id or "n/a")
    )
    context = MlRunContext.from_cli(
        request=request,
        snapshot_id=snapshot_id_or_lineage,
        data=data,
        cost_context=cost_context,
        started_at=started_at,
        data_lineage=resolved_lineage,
    )
    ledger = MlResultLedger(parsed.results_root)
    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )
    try:
        manifest = train_net_alpha_model(data, registry, request)
    except Exception as exc:
        try:
            ledger.record_failed(context, "train_net_alpha_model", exc)
        except Exception as ledger_exc:
            logger.error(
                "[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc
            )
        raise
    logger.info(
        "[ALGO] stage=train selected_family=%s artifact=%s",
        manifest.model_type,
        manifest.artifact_id,
    )
    logger.info(
        "[EVAL] stage=promotion artifact=%s promoted=%s no_trade=%s",
        manifest.artifact_id,
        manifest.model_type != "no_trade",
        manifest.model_type == "no_trade",
    )
    try:
        ledger.record_completed(context, manifest, registry)
    except Exception as exc:
        logger.error(
            "[SYS] stage=result_ledger status=write_failed error=%s", exc
        )
    else:
        logger.info(
            "[SYS] stage=result_ledger status=written artifact=%s",
            manifest.artifact_id,
        )
    return 0


def _run_direct_training(
    parsed: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """Run training using direct dataset IDs instead of snapshot resolution."""
    from src.stocks.data.direct import DirectDataRequest, DirectMarketDataLoader

    if not parsed.research_start_direct or not parsed.research_end_direct:
        parser.error(
            "direct dataset loading requires --research-start-direct and --research-end-direct"
        )

    decision_time = parsed.decision_time or datetime.now(UTC)
    started_at = datetime.now(UTC)

    loader = DirectMarketDataLoader(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )

    request_data = DirectDataRequest(
        base_dataset_id=parsed.base_dataset_id,
        feature_dataset_id=parsed.feature_dataset_id,
        label_dataset_id=parsed.label_dataset_id,
        start=parsed.research_start_direct,
        end=parsed.research_end_direct,
        feature_set="stock_net_alpha_v1",
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )

    market_data = loader.load(request_data)
    validate_ml_market_data(market_data, request_data.candidate_horizon_sessions)
    logger.info(
        "[DATA] stage=direct_load feature_rows=%d instruments=%d sessions=%d columns=%d",
        market_data.frame.height,
        market_data.frame["instrument_id"].n_unique(),
        market_data.frame["session"].n_unique(),
        len(market_data.frame.columns),
    )

    # Convert to NetAlphaResearchData for compatibility with existing training
    from src.core.datasets import make_manifest
    from src.core.instruments import AssetKind
    from src.stocks.data.contracts import DatasetSnapshot
    from src.stocks.ml.data import compose_net_alpha_training_data

    # Create a minimal snapshot for compatibility
    data_manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=market_data.frame.columns,
        feature_set="stock_net_alpha_v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=max(_parse_horizons(parsed.candidate_horizon_sessions)),
        time_start=datetime.combine(parsed.research_start_direct, datetime.min.time()),
        time_end=datetime.combine(parsed.research_end_direct, datetime.min.time()),
        generated_time=datetime.now(UTC),
        row_count=market_data.frame.height,
        provider_version="direct-loader",
        universe_policy_version="direct-loader",
    )
    snapshot = DatasetSnapshot(manifest=data_manifest, frame=market_data.frame)

    data = compose_net_alpha_training_data(
        snapshot,
        decision_time,
        candidate_horizon_sessions=_parse_horizons(parsed.candidate_horizon_sessions),
    )

    # Build training request
    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_cost_contexts(None)

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
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
        enforce_snapshot_outcome_readiness=False,
    )

    cost_context = CostRunContext(
        cost_schedule_kind=parsed.cost_schedule,
        cost_evidence_path=None,
        cost_evidence_hash=None,
        has_liquidity_model=liquidity_model is not None,
    )

    context = MlRunContext.from_cli(
        request=request,
        snapshot_id=f"direct:{parsed.base_dataset_id}:{parsed.feature_dataset_id}:{parsed.label_dataset_id}",
        data=data,
        cost_context=cost_context,
        started_at=started_at,
        input_ids=market_data.input_ids,
    )

    ledger = MlResultLedger(parsed.results_root)
    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )

    try:
        model_manifest = train_net_alpha_model(data, registry, request)
    except Exception as exc:
        try:
            ledger.record_failed(context, "train_net_alpha_model", exc)
        except Exception as ledger_exc:
            logger.error(
                "[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc
            )
        raise

    logger.info(
        "[ALGO] stage=train selected_family=%s artifact=%s",
        model_manifest.model_type,
        model_manifest.artifact_id,
    )

    try:
        ledger.record_completed(context, model_manifest, registry)
    except Exception as exc:
        logger.error(
            "[SYS] stage=result_ledger status=write_failed error=%s", exc
        )
    else:
        logger.info(
            "[SYS] stage=result_ledger status=written artifact=%s",
            model_manifest.artifact_id,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
