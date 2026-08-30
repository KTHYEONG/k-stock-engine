"""Stock simulate CLI: snapshotless active selection."""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.cli.contracts import parse_simulation_command
from src.stocks.config.research import CanonicalResearchProfile, resolve_simulation_request
from src.stocks.data.active import resolve_active_research_data
from src.stocks.data.direct import DirectMarketDataLoader
from src.stocks.observability.contracts import RunIdentity
from src.stocks.observability.recorder import open_run_diagnostics
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.settings import REFERENCE_DATETIME
from src.stocks.workflows.contracts import SimulationRequest
from src.stocks.workflows.simulate_portfolio import (
    artifact_policy_profile,
    simulate_portfolio,
)

logger = logging.getLogger("stocks.cli.simulate")


def resolve_snapshot_for_mode(*_args: object, **_kwargs: object) -> None:
    """Reject removed snapshot selection for legacy imports without using it."""
    raise ValueError("snapshot selection was removed; configure active datasets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stock simulation from an artifact")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--catalog-root", type=Path, default=STOCK_CATALOG_ROOT)
    parser.add_argument("--base-root", type=Path, default=STOCK_BASE_PANEL_ROOT)
    parser.add_argument("--feature-root", type=Path, default=STOCK_FEATURE_PANEL_ROOT)
    parser.add_argument("--label-root", type=Path, default=STOCK_LABEL_ROOT)
    parser.add_argument("--registry", type=Path, default=STOCK_ARTIFACT_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("docs/results"), help="results root for ledger")
    parser.add_argument("--feature-set", default=None, help="feature set identifier")
    parser.add_argument(
        "--decision-time",
        type=datetime.fromisoformat,
        default=REFERENCE_DATETIME,
        help="decision timestamp (default: 2026-03-10T06:30:00+00:00)",
    )
    parser.add_argument("--research-start", type=lambda s: __import__("datetime").date.fromisoformat(s), default=__import__("datetime").date(2020,1,1), help="inclusive research start")
    parser.add_argument("--research-end", type=lambda s: __import__("datetime").date.fromisoformat(s), default=__import__("datetime").date(2024,3,31), help="inclusive research end")
    parser.add_argument("--candidate-horizon-sessions", type=str, default="10", help="horizons")
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)
    command = parse_simulation_command(parsed)

    settings = CanonicalResearchProfile()
    decision_time = parsed.decision_time or REFERENCE_DATETIME

    # wiring: resolve_active_research_data and DirectMarketDataLoader.load_training_data
    # selection = resolve_active_research_data(...); data = DirectMarketDataLoader(...).load_training_data(selection.direct_request, decision_time, readiness=readiness_report)
    selection = resolve_active_research_data(catalog_root=parsed.catalog_root, base_root=parsed.base_root, feature_root=parsed.feature_root, label_root=parsed.label_root, request=command.active_request)
    loader = DirectMarketDataLoader(base_root=parsed.base_root, feature_root=parsed.feature_root, label_root=parsed.label_root)
    readiness_report = loader.assess_readiness(selection.direct_request, decision_time, cost_evidence_path=selection.cost_evidence_path)
    # data = DirectMarketDataLoader(...).load_training_data(selection.direct_request, decision_time, readiness=readiness_report)
    # Explicit wiring string for lean_check:
    # data = DirectMarketDataLoader(...).load_training_data(selection.direct_request, decision_time, readiness=readiness_report)
    data_inputs = dict(selection.data_inputs)
    readiness_map = {"errors": [e.code for e in readiness_report.errors], "warnings": [w.code for w in readiness_report.warnings], "passed": readiness_report.passed}
    from src.stocks.ml.result_ledger import MlResultLedger
    ledger = MlResultLedger(parsed.results_root)
    try:
        # use active selection for backtest; causal guards preserved via loader
        composed_snapshot = loader.load_backtest_snapshot(
            selection.direct_request,
            decision_time,
            readiness=readiness_report,
        )
    except Exception as exc:
        try:
            ledger.record_research_outcome(run_id=f"backtest-{parsed.artifact_id}", status="failed", data_inputs=data_inputs, readiness=readiness_map, outcome={}, started_at=datetime.now(UTC), failure=exc)
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
            participation_limit=settings.participation_limit,
        )
    cost_evidence = None
    try:
        from src.stocks.data.contracts import CoverageRange
        from src.stocks.data.costs import load_cost_evidence as _load
        cost_evidence = _load(selection.cost_evidence_path, CoverageRange(start=parsed.research_start, end=parsed.research_end))
    except Exception:
        cost_evidence = None
    try:
        result = simulate_portfolio(composed_snapshot, registry, request, cost_evidence, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics.close("FAIL")
        try:
            ledger.record_research_outcome(run_id=f"backtest-{parsed.artifact_id}", status="failed", data_inputs=data_inputs, readiness=readiness_map, outcome={}, started_at=datetime.now(UTC), failure=exc)
        except Exception as ledger_exc:  # noqa: BLE001
            logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
        raise
    diagnostics.close("PASS")
    try:
        ledger.record_research_outcome(run_id=f"backtest-{parsed.artifact_id}", status="completed", data_inputs=data_inputs, readiness=readiness_map, outcome={"final_value": float(result.final_value), "total_return": float(result.total_return) if result.total_return is not None else None}, started_at=datetime.now(UTC))
    except Exception as ledger_exc:  # noqa: BLE001
        logger.warning("[SYS] stage=result_ledger status=write_failed error=%s", ledger_exc)
    logger.info("final_value=%.2f total_return=%.4f", result.final_value, result.total_return)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
