"""Stock train CLI: resolve a net-alpha snapshot, compose, and train the mainline.

The only supported training path is the canonical ``stock_net_alpha_v1``
mainline. A snapshot or artifact that does not satisfy the net-alpha contract
raises one actionable ``ValueError`` naming the materialization CLI; there is
no legacy LambdaRank/Optuna flag, no implicit fallback, and no fixed 5/10/15
route.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import replace
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
from src.stocks.config.research import resolve_training_request
from src.stocks.config.runtime import StockRuntimeSettings
from src.stocks.data.contracts import CoverageRange, DatasetSnapshot
from src.stocks.data.costs import load_cost_evidence
from src.stocks.data.lineage import ResearchDataBundle, ResolvedDataLineage
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.ml.contracts import (
    DEFAULT_CANDIDATE_HORIZON_SESSIONS,
    DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS,
    DEFAULT_CANDIDATE_TOP_K,
    ExecutionFrontierSettings,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
    PortfolioSettings,
    RiskSettings,
)
from src.stocks.ml.data import (
    compose_direct_net_alpha_training_data,
    compose_net_alpha_training_data,
    validate_ml_market_data,
)
from src.stocks.ml.result_ledger import (
    CostRunContext,
    MlResultLedger,
    MlRunContext,
)
from src.stocks.ml.training import TrainingOrchestrator, train_net_alpha_model
from src.stocks.observability.contracts import RunDiagnostics, RunIdentity
from src.stocks.observability.recorder import open_run_diagnostics
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.models import ModelManifest
from src.stocks.settings import REFERENCE_DATE, REFERENCE_DATETIME

logger = logging.getLogger("stocks.cli.train")


def _invoke_training(
    data: NetAlphaResearchData,
    registry: ModelArtifactRegistry,
    request: NetAlphaTrainingRequest,
    diagnostics: RunDiagnostics,
) -> ModelManifest:
    """Drive the training orchestrator, honoring legacy callable signatures."""
    parameters = inspect.signature(train_net_alpha_model).parameters
    if "diagnostics" in parameters:
        orchestrator = TrainingOrchestrator(
            data, registry, request, diagnostics=diagnostics
        )
        return orchestrator.run()
    return train_net_alpha_model(data, registry, request)

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
        default=REFERENCE_DATE,
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
    parser.add_argument(
        "--candidate-rebalance-frequency-sessions",
        type=str,
        default=",".join(
            str(value) for value in DEFAULT_CANDIDATE_REBALANCE_FREQUENCY_SESSIONS
        ),
        help="pre-registered execution cadence grid in sessions",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CANDIDATE_TOP_K),
        help="pre-registered execution maximum active-name grid",
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
    parser.add_argument(
        "--memory-reserve-mib",
        type=int,
        default=0,
        help=(
            "measured concurrent-workload memory reserve in MiB subtracted "
            "from cgroup/system headroom during pre-allocation planning"
        ),
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
    parser.add_argument(
        "--decision-time",
        type=datetime.fromisoformat,
        default=REFERENCE_DATETIME,
        help="decision timestamp (default: 2026-03-10T06:30:00+00:00)",
    )
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
        default=date(2016, 1, 4),
        help="start date for direct dataset loading (requires --base-dataset-id)",
    )
    parser.add_argument(
        "--research-end-direct",
        type=date.fromisoformat,
        default=REFERENCE_DATE,
        help="end date for direct dataset loading (requires --base-dataset-id)",
    )
    parser.add_argument(
        "--cost-snapshot-id",
        default=None,
        help=(
            "hash-bound cost snapshot id for direct runs; "
            "without it, the run is research-only"
        ),
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


def _build_training_request(args: argparse.Namespace) -> NetAlphaTrainingRequest:
    """Build the typed training request from parsed CLI arguments.

    Cost schedules default to the canonical base/stress pair here so the
    request exists before any repository/catalog access; callers holding a
    hash-bound cost snapshot replace the schedules afterwards without altering
    any validated field.
    """
    horizons = _parse_horizons(args.candidate_horizon_sessions)
    cadences = _parse_horizons(args.candidate_rebalance_frequency_sessions)
    top_k = _parse_horizons(args.candidate_top_k)
    return NetAlphaTrainingRequest(
        artifact_id=args.artifact_id,
        candidate_horizon_sessions=horizons,
        execution_frontier=ExecutionFrontierSettings(
            candidate_horizon_sessions=horizons,
            candidate_rebalance_frequency_sessions=cadences,
            candidate_top_k=top_k,
        ),
        fold_count=args.fold_count,
        embargo_sessions=args.embargo_sessions,
        forward_holdout_sessions=args.forward_holdout_sessions,
        bootstrap_alpha=args.bootstrap_alpha,
        bootstrap_resamples=args.bootstrap_resamples,
        model_threads=args.model_threads,
        max_rss_mib=args.max_rss_mib,
        memory_reserve_mib=args.memory_reserve_mib,
        seed=args.seed,
        portfolio=PortfolioSettings(
            top_k=args.top_k,
            max_single_weight=args.max_single_weight,
            max_exposure=args.max_exposure,
            participation_limit=args.participation_limit,
            portfolio_value=args.portfolio_value,
            initial_cash=args.portfolio_value,
            reference_notional=args.reference_notional,
        ),
        risk=RiskSettings(),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=None,
        stress_liquidity_model=None,
        enforce_snapshot_outcome_readiness=False,
    )


def _validate_static_training_request(request: NetAlphaTrainingRequest) -> None:
    """Fail closed on an infeasible request before any data allocation.

    Validates execution-frontier feasibility and the static discovery-grid
    identity purely from the request contract: no repository, catalog,
    Parquet, or loader is touched.
    """
    if tuple(request.execution_frontier.candidate_horizon_sessions) != tuple(
        request.candidate_horizon_sessions
    ):
        raise ValueError(
            "execution_frontier.candidate_horizon_sessions must equal "
            "candidate_horizon_sessions"
        )
    if request.fold_count < 1:
        raise ValueError("fold-count must be a positive session count")
    if request.embargo_sessions < 0:
        raise ValueError("embargo-sessions must be non-negative")
    request.execution_frontier.require_feasible_horizons(
        request.portfolio.max_exposure, request.portfolio.max_single_weight
    )


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


def _resolve_direct_cost_context(
    cost_snapshot_id: str | None,
    parsed: argparse.Namespace,
    market_data: object,
) -> tuple[
    CostSchedule,
    LiquiditySlippageModel | None,
    CostSchedule,
    LiquiditySlippageModel | None,
]:
    """Resolve cost schedules for direct dataset runs with hash-bound provenance.

    A direct production run requires a hash-bound cost snapshot whose execution
    coverage contains the direct data range and whose stock universe identity
    is compatible.  Without a ``cost_snapshot_id``, canonical defaults are used
    but the run is research-only (no artifact publication).
    """
    if cost_snapshot_id is None:
        return (
            default_base_schedule(),
            None,
            default_stress_schedule(),
            None,
        )
    from src.core.paths import STOCK_CATALOG_ROOT
    from src.stocks.data.costs import load_cost_evidence

    cost_path = STOCK_CATALOG_ROOT / "costs" / f"{cost_snapshot_id}.json"
    if not cost_path.exists():
        raise ValueError(
            f"cost snapshot {cost_snapshot_id!r} not found at {cost_path}"
        )
    required_range = CoverageRange(
        start=parsed.research_start_direct,
        end=parsed.research_end_direct,
    )
    evidence = load_cost_evidence(cost_path, required_range)
    return (
        evidence.base_schedule(),
        evidence.base_liquidity_model,
        evidence.stress_schedule(),
        evidence.stress_liquidity_model,
    )


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    # Build and statically validate the request BEFORE any repository,
    # catalog, Parquet, or loader access so an infeasible frontier fails
    # closed without data allocation.
    request = _build_training_request(parsed)
    _validate_static_training_request(request)

    # Direct dataset loading path
    if parsed.base_dataset_id and parsed.feature_dataset_id and parsed.label_dataset_id:
        return _run_direct_training(parsed, parser, request)

    # Legacy snapshot/as-of path
    if not parsed.snapshot_id and not parsed.as_of:
        parser.error("either --snapshot-id or --as-of is required")

    decision_time = parsed.decision_time or REFERENCE_DATETIME
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
    request = replace(
        request,
        base_cost_schedule=base_cost_schedule,
        stress_cost_schedule=stress_cost_schedule,
        liquidity_model=liquidity_model,
        stress_liquidity_model=stress_liquidity_model,
        enforce_snapshot_outcome_readiness=True,
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
    identity = RunIdentity(run_id=request.artifact_id, project="stocks")
    runtime_settings = StockRuntimeSettings(diagnostics_enabled=True).model_dump()
    resolve_training_request(request.artifact_id, overrides={})
    diagnostics = open_run_diagnostics(identity, runtime_settings)
    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )
    try:
        manifest = _invoke_training(data, registry, request, diagnostics)
    except Exception as exc:
        diagnostics.close("FAIL")
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
    diagnostics.close("PASS")
    return 0


def _run_direct_training(
    parsed: argparse.Namespace,
    parser: argparse.ArgumentParser,
    request: NetAlphaTrainingRequest,
) -> int:
    """Run training using direct dataset IDs instead of snapshot resolution.

    The statically validated request arrives prebuilt; this path only resolves
    hash-bound cost schedules, performs the bounded separated load, and
    composes without any repeated-label snapshot.
    """
    from src.stocks.data.direct import DirectDataRequest, DirectMarketDataLoader

    if not parsed.research_start_direct or not parsed.research_end_direct:
        parser.error(
            "direct dataset loading requires --research-start-direct and --research-end-direct"
        )

    decision_time = parsed.decision_time or REFERENCE_DATETIME
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

    data = compose_direct_net_alpha_training_data(market_data, decision_time)
    logger.info(
        "[DATA] stage=compose_direct decision_rows=%d horizons=%s",
        data.feature_frame.height,
        sorted(data.labels_by_horizon),
    )

    # Bind the immutable feature dataset schema identity so the published
    # artifact's feature_schema_hash equals the selected feature dataset
    # manifest.schema_hash and the feature content hash is preserved exactly.
    feature_manifest = market_data.feature_manifest
    if feature_manifest is not None:
        data = replace(
            data,
            manifest=replace(
                data.manifest,
                schema_hash=feature_manifest.schema_hash,
                feature_set_hash=(
                    feature_manifest.feature_set_hash or feature_manifest.schema_hash
                ),
                content_hash=feature_manifest.content_hash
                or feature_manifest.schema_hash,
            ),
        )

    (
        base_cost_schedule,
        liquidity_model,
        stress_cost_schedule,
        stress_liquidity_model,
    ) = _resolve_direct_cost_context(parsed.cost_snapshot_id, parsed, market_data)

    # Retain only provenance metadata and release the raw market container:
    # NetAlphaResearchData stays the live owner of feature/label frames.
    input_ids = dict(market_data.input_ids)
    input_content_hashes = dict(market_data.input_content_hashes)
    logger.info(
        "[DATA] stage=provenance_retained datasets=%d content_hashes=%d",
        len(input_ids),
        len(input_content_hashes),
    )
    del market_data

    registry = ModelArtifactRegistry(parsed.registry)
    request = replace(
        request,
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
        input_ids=input_ids,
    )

    ledger = MlResultLedger(parsed.results_root)
    identity = RunIdentity(run_id=request.artifact_id, project="stocks")
    runtime_settings = StockRuntimeSettings(diagnostics_enabled=True).model_dump()
    resolve_training_request(request.artifact_id, overrides={})
    diagnostics = open_run_diagnostics(identity, runtime_settings)
    logger.info(
        "[ALGO] stage=train artifact=%s candidate_horizons=%s",
        request.artifact_id,
        list(request.candidate_horizon_sessions),
    )

    try:
        model_manifest = _invoke_training(data, registry, request, diagnostics)
    except Exception as exc:
        diagnostics.close("FAIL")
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

    diagnostics.close("PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
