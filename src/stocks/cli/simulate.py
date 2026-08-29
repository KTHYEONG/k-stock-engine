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
from src.stocks.config.research import resolve_simulation_request
from src.stocks.data.costs import load_cost_evidence
from src.stocks.data.repositories import (
    ResearchDataRepository,
    resolve_snapshot_for_mode,
)
from src.stocks.observability.contracts import RunIdentity
from src.stocks.observability.recorder import open_run_diagnostics
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
    parser.add_argument("--snapshot-id", required=False, default=None, help="immutable research snapshot id (legacy)")
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("docs/results"), help="results root for ledger")
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
    parser.add_argument("--base-dataset-id", default=None, help="direct base dataset ID")
    parser.add_argument("--feature-dataset-id", default=None, help="direct feature dataset ID")
    parser.add_argument("--label-dataset-id", default=None, help="direct label dataset ID")
    parser.add_argument("--data-start", type=lambda s: __import__("datetime").date.fromisoformat(s), default=None, help="inclusive data start")
    parser.add_argument("--data-end", type=lambda s: __import__("datetime").date.fromisoformat(s), default=None, help="inclusive data end")
    parser.add_argument("--cost-evidence-path", type=Path, default=None, help="optional cost evidence path")
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    # Direct-input validation: partial group fails via parser
    direct_ids = [parsed.base_dataset_id, parsed.feature_dataset_id, parsed.label_dataset_id]
    if any(direct_ids) and not all(direct_ids):
        parser.error("direct simulation requires --base-dataset-id, --feature-dataset-id, --label-dataset-id together")
    if any(direct_ids) and (parsed.data_start is None or parsed.data_end is None):
        parser.error("direct simulation requires --data-start and --data-end")

    settings = DEFAULT_STOCK_ALPHA
    decision_time = parsed.decision_time or REFERENCE_DATETIME

    # Direct path: use DirectMarketDataLoader without snapshot resolution
    if all(direct_ids):
        from datetime import UTC

        from src.stocks.data.direct import DirectDataRequest, DirectMarketDataLoader
        from src.stocks.ml.result_ledger import MlResultLedger

        loader = DirectMarketDataLoader(
            base_root=parsed.base_root,
            feature_root=parsed.feature_root,
            label_root=parsed.label_root,
        )
        direct_request = DirectDataRequest(
            base_dataset_id=str(parsed.base_dataset_id),
            feature_dataset_id=str(parsed.feature_dataset_id),
            label_dataset_id=str(parsed.label_dataset_id),
            start=parsed.data_start,
            end=parsed.data_end,
            candidate_horizon_sessions=(10,),
        )
        # readiness assessment
        readiness = loader.assess_readiness(direct_request, decision_time, cost_evidence_path=parsed.cost_evidence_path)
        data_inputs = {
            "base_dataset_id": direct_request.base_dataset_id,
            "feature_dataset_id": direct_request.feature_dataset_id,
            "label_dataset_id": direct_request.label_dataset_id,
            "start": direct_request.start.isoformat(),
            "end": direct_request.end.isoformat(),
            "feature_schema_hash": readiness.input_reference.feature_schema_hash,
            "feature_content_hash": readiness.input_reference.feature_content_hash,
            "cost_evidence_path": readiness.input_reference.cost_evidence_path,
            "cost_evidence_hash": readiness.input_reference.cost_evidence_hash,
        }
        readiness_map = {
            "errors": [e.code for e in readiness.errors],
            "warnings": [w.code for w in readiness.warnings],
            "passed": readiness.passed,
        }
        ledger = MlResultLedger(parsed.results_root)
        # load snapshot via direct loader (bounded scans)
        try:
            composed_snapshot = loader.load_backtest_snapshot(direct_request, decision_time, readiness=readiness)
        except Exception as exc:
            try:
                ledger.record_research_outcome(
                    run_id=f"backtest-{parsed.artifact_id}",
                    status="failed",
                    data_inputs=data_inputs,
                    readiness=readiness_map,
                    outcome={},
                    started_at=datetime.now(UTC),
                    failure=exc,
                )
            except Exception as ledger_exc:  # noqa: BLE001
                logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
            raise
        registry = ModelArtifactRegistry(parsed.registry)
        policy_profile = artifact_policy_profile(registry, parsed.artifact_id)
        resolve_simulation_request(policy_profile or {}, overrides={})
        identity = RunIdentity(run_id=f"backtest-{parsed.artifact_id}", project="stocks")
        diagnostics = open_run_diagnostics(identity, {"diagnostics_enabled": True})
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
        cost_evidence = None
        if parsed.cost_evidence_path is not None and parsed.cost_evidence_path.exists():
            # direct cost evidence path supplied; use it if possible, else warning already recorded
            try:
                from src.stocks.data.contracts import CoverageRange
                from src.stocks.data.costs import load_cost_evidence as _load

                cost_evidence = _load(parsed.cost_evidence_path, CoverageRange(start=parsed.data_start, end=parsed.data_end))
            except Exception:
                cost_evidence = None
        try:
            result = simulate_portfolio(
                composed_snapshot,
                registry,
                request,
                cost_evidence,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            diagnostics.close("FAIL")
            try:
                ledger.record_research_outcome(
                    run_id=f"backtest-{parsed.artifact_id}",
                    status="failed",
                    data_inputs=data_inputs,
                    readiness=readiness_map,
                    outcome={},
                    started_at=datetime.now(UTC),
                    failure=exc,
                )
            except Exception as ledger_exc:  # noqa: BLE001
                logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
            raise
        diagnostics.close("PASS")
        # bounded terminal record
        try:
            ledger.record_research_outcome(
                run_id=f"backtest-{parsed.artifact_id}",
                status="completed",
                data_inputs=data_inputs,
                readiness=readiness_map,
                outcome={"final_value": float(result.final_value), "total_return": float(result.total_return) if result.total_return is not None else None},
                started_at=datetime.now(UTC),
            )
        except Exception as ledger_exc:  # noqa: BLE001
            logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
        logger.info(
            "final_value=%.2f total_return=%.4f", result.final_value, result.total_return
        )
        return 0

    # Legacy snapshot path
    if not parsed.snapshot_id:
        parser.error("either --snapshot-id or direct dataset IDs are required")
    snapshot = resolve_snapshot_for_mode(
        parsed.catalog_root, parsed.snapshot_id, mode=parsed.mode
    )
    repository = ResearchDataRepository(
        base_root=parsed.base_root,
        feature_root=parsed.feature_root,
        label_root=parsed.label_root,
    )
    feature_set = parsed.feature_set or "stock_net_alpha_v1"
    composed_snapshot = repository.compose_training_snapshot(
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
    resolve_simulation_request(policy_profile or {}, overrides={})
    identity = RunIdentity(run_id=f"backtest-{parsed.artifact_id}", project="stocks")
    diagnostics = open_run_diagnostics(identity, {"diagnostics_enabled": True})
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
    try:
        result = simulate_portfolio(
            composed_snapshot,
            registry,
            request,
            cost_evidence,
            diagnostics=diagnostics,
        )
    except Exception:
        diagnostics.close("FAIL")
        raise
    diagnostics.close("PASS")
    logger.info(
        "final_value=%.2f total_return=%.4f", result.final_value, result.total_return
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
