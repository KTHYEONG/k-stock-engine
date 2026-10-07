"""Build-area commands: Silver/Gold materialization, fact refresh and scope refresh."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from src.data.cli.common import (
    CommandFailed,
    add_scoped_args,
    emit,
    register_dataset,
    registered_id,
    resolve_input_id,
    scoped_catalog,
    scoped_runtime,
)
from src.data.cli.registry import Command

__all__ = ["BUILD_COMMANDS"]

_LOG = logging.getLogger(__name__)


def _parse_decision_time(value: str) -> datetime:
    from src.core.pit import PITDataError

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise PITDataError("decision_time must be timezone-aware")
    return parsed


def _add_normalize(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bronze-root", type=Path, required=False, default=None)
    parser.add_argument("--silver-root", type=Path, required=False, default=None)
    parser.add_argument("--artifact-root", type=Path, required=False, default=None)
    parser.add_argument("--scope-config", type=Path, required=False, default=None)
    parser.add_argument("--data-root", type=Path, required=False, default=None)
    parser.add_argument("--decision-time", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--superseded-receipts", type=Path, default=None,
        help="JSON list of Bronze receipt hashes to exclude as superseded",
    )


def _add_flow_silver(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--universe-dataset-id", default=None)
    parser.add_argument("--workers", type=int, default=4)


def _add_daily_silver(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--universe-dataset-id", default=None)


def _add_ordinary(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--sessions", type=Path, default=None, help="Optional JSON list of requested session dates")


def _add_panel(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--daily-market-dataset-id", default=None)
    parser.add_argument("--universe-dataset-id", default=None)
    parser.add_argument("--market-actions-dataset-id", default=None)
    parser.add_argument("--rules", type=Path, required=False, default=None)
    parser.add_argument("--instrument-buckets", type=int, default=16)


def _add_benchmarks(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--market-panel-dataset-id", default=None)
    parser.add_argument("--definitions", type=Path, required=False, default=None)


def _add_refresh(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--collect", action="store_true", help="Run collection jobs before building")
    parser.add_argument("--dry-run", action="store_true", help="Report stale nodes and pending counts only")


def _add_kis_supplement(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--ls-flow-dataset-id", default=None)


def _add_flow_union(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--ls-flow-dataset-id", default=None)
    parser.add_argument("--kis-supplement-dataset-id", default=None)


def _add_quality(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)
    parser.add_argument("--facts-dataset-id", default=None)
    parser.add_argument("--quarantine-file", type=Path, default=None)
    parser.add_argument("--unresolved-events-file", type=Path, default=None)
    parser.add_argument("--decision-time", required=True)


def _add_scoped_only(parser: argparse.ArgumentParser) -> None:
    add_scoped_args(parser)


def _run_normalize(args: argparse.Namespace) -> Mapping[str, object]:
    from src.core.pit import PITDataError
    from src.data.dataset_registry import DatasetRegistry
    from src.data.incremental_normalization import normalize_dart_facts

    normalize_runtime = None
    if args.scope_config is not None or args.data_root is not None:
        if args.scope_config is None or args.data_root is None:
            raise PITDataError("normalize-dart-facts scope-config and data-root must be supplied together")
        normalize_runtime = scoped_runtime(args)
    if normalize_runtime is None:
        if any(value is None for value in (args.bronze_root, args.silver_root, args.artifact_root)):
            raise PITDataError("normalize-dart-facts requires explicit roots or scoped runtime arguments")
        bronze_root, silver_root, artifact_root = Path(args.bronze_root), Path(args.silver_root), Path(args.artifact_root)
    else:
        bronze_root = Path(args.bronze_root) if args.bronze_root is not None else normalize_runtime.workspace.bronze_root
        silver_root = Path(args.silver_root) if args.silver_root is not None else normalize_runtime.workspace.silver_root
        # 격리 목록은 품질 빌드가 스코프 state에서 해시 검증 후 읽으므로 state에 둔다.
        artifact_root = (
            Path(args.artifact_root) if args.artifact_root is not None else normalize_runtime.workspace.state_root
        )
    normalize_registry = DatasetRegistry(normalize_runtime.workspace.state_root) if normalize_runtime is not None else None
    superseded = getattr(args, "superseded_receipts", None)
    if superseded is None and normalize_runtime is not None:
        from src.data.fact_state import SUPERSEDED_RECEIPTS_FILE

        candidate = normalize_runtime.workspace.state_root / SUPERSEDED_RECEIPTS_FILE
        superseded = candidate if candidate.exists() else None
    payload = normalize_dart_facts(
        bronze_root=bronze_root, silver_root=silver_root, artifact_root=artifact_root,
        decision_time=_parse_decision_time(args.decision_time), batch_size=int(getattr(args, "batch_size", 500)),
        superseded_receipts=superseded,
        disclosures_dataset_id=normalize_registry.current("disclosures") if normalize_registry is not None else None,
        financial_facts_dataset_id=normalize_registry.current("financial_facts") if normalize_registry is not None else None,
    )
    dataset_id = Path(str(payload["dataset_path"])).name
    if normalize_runtime is not None:
        DatasetRegistry(normalize_runtime.workspace.state_root).register("financial_facts", dataset_id)
    else:
        resolved_silver = silver_root.resolve()
        if resolved_silver.parent.name == "silver":
            DatasetRegistry(resolved_silver.parent.parent / "state" / resolved_silver.name).register("financial_facts", dataset_id)
        else:
            DatasetRegistry(resolved_silver.parent / "state" / "unscoped",
                            data_root=resolved_silver.parent, scope_id="").register("financial_facts", dataset_id)
    return {"output_hash": payload["output_hash"], "report_hash": payload["report_hash"], "row_count": payload["row_count"],
            "quarantined_filings": payload["quarantined_filings"], "quarantine_path": payload["quarantine_path"],
            "dataset_id": dataset_id, "dataset_path": payload["dataset_path"]}


def _run_flow_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.cli.common import scoped_catalog
    from src.data.dataset_registry import DatasetRegistry
    from src.data.investor_flow_silver import materialize_investor_flow_silver

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    result = materialize_investor_flow_silver(
        catalog=scoped_catalog(runtime), universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root, workers=args.workers, universe_dataset_id=universe_id,
    )
    register_dataset(runtime, "investor_flow_ls", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path)}


def _run_daily_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.daily_market_silver import materialize_daily_market_silver
    from src.data.dataset_registry import DatasetRegistry

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    result = materialize_daily_market_silver(
        catalog=scoped_catalog(runtime), universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root, universe_dataset_id=universe_id,
    )
    register_dataset(runtime, "daily_market", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path)}


def _run_ordinary(args: argparse.Namespace) -> Mapping[str, object]:
    from src.core.pit import PITDataError
    from src.data.ordinary_universe import catalog_master_sessions, materialize_ordinary_universe_from_catalog

    runtime = scoped_runtime(args)
    if args.sessions is None:
        sessions = catalog_master_sessions(scoped_catalog(runtime))
    else:
        raw_sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
        if not isinstance(raw_sessions, list):
            raise PITDataError("ordinary universe sessions must be a JSON list")
        sessions = tuple(date.fromisoformat(str(value)[:10]) for value in raw_sessions)
    path = materialize_ordinary_universe_from_catalog(
        catalog=scoped_catalog(runtime), sessions=sessions, silver_root=runtime.workspace.silver_root,
    )
    register_dataset(runtime, "ordinary_universe", path.name)
    return {"dataset_id": path.name, "dataset_path": str(path), "sessions": len(sessions)}


def _run_panel(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.dataset_registry import DatasetRegistry
    from src.data.market_panel import materialize_market_panel

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    daily_id = registered_id(registry, "daily_market", args.daily_market_dataset_id)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    actions_id = args.market_actions_dataset_id
    if actions_id is None:
        actions_id = registry.current("market_actions")
    result = materialize_market_panel(
        daily_market_path=runtime.workspace.silver_root / daily_id,
        universe_path=runtime.workspace.silver_root / universe_id,
        rules=load_krx_market_rules(Path(args.rules) if args.rules is not None else load_runtime_config().market_rules),
        gold_root=runtime.workspace.gold_root, instrument_buckets=args.instrument_buckets,
        market_actions_path=runtime.workspace.silver_root / actions_id if actions_id is not None else None,
    )
    register_dataset(runtime, "market_panel", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path),
                             "daily_market_dataset_id": daily_id, "universe_dataset_id": universe_id,
                             "market_actions_dataset_id": actions_id}


def _run_benchmarks(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.config.runtime import load_runtime_config
    from src.data.dataset_registry import DatasetRegistry
    from src.data.reference_benchmarks import load_benchmark_definitions, materialize_reference_benchmarks

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    panel_id = registered_id(registry, "market_panel", args.market_panel_dataset_id)
    definitions_path = Path(args.definitions) if args.definitions is not None else load_runtime_config().reference_benchmarks
    version, definitions = load_benchmark_definitions(definitions_path)
    result = materialize_reference_benchmarks(
        market_panel_path=runtime.workspace.gold_root / panel_id, definitions=definitions,
        definitions_version=version, gold_root=runtime.workspace.gold_root,
    )
    register_dataset(runtime, "reference_benchmarks", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path), "market_panel_dataset_id": panel_id}


def _run_refresh(args: argparse.Namespace) -> Mapping[str, object]:
    from src.data.pipeline_graph import build_refresh_context, refresh_scope

    runtime = scoped_runtime(args)
    ctx = build_refresh_context(runtime)
    report = refresh_scope(
        ctx, collect=bool(getattr(args, "collect", False)), dry_run=bool(getattr(args, "dry_run", False)),
        emit=lambda payload: emit(dict(payload)),
    )
    summary = {
        "type": "summary", "status": report.status, "planned": list(report.planned), "built": list(report.built),
        "datasets": dict(report.datasets), "blocking_job": report.blocking_job,
        "decision_time": report.decision_time.isoformat(),
        "collection": [dict(step) for step in report.collection],
    }
    _LOG.info("[DATA] command=refresh_scope status=%s planned=%d built=%d",
              report.status, len(report.planned), len(report.built))
    if report.status == "blocked":
        raise CommandFailed(1, summary)
    return summary


def _run_kis_supplement(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.cli.common import scoped_catalog
    from src.data.dataset_registry import DatasetRegistry
    from src.data.flow_targets import investor_flow_targets
    from src.data.investor_flow_kis_supplement import materialize_investor_flow_kis_supplement

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    ls_id = registered_id(registry, "investor_flow_ls", args.ls_flow_dataset_id)
    targets = investor_flow_targets(runtime)
    result = materialize_investor_flow_kis_supplement(
        catalog=scoped_catalog(runtime), targets=targets,
        ls_flow_silver_path=runtime.workspace.silver_root / ls_id, silver_root=runtime.workspace.silver_root,
    )
    register_dataset(runtime, "investor_flow_kis_supplement", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path),
                             "ls_dataset_id": ls_id}


def _run_flow_union(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.dataset_registry import DatasetRegistry
    from src.data.investor_flow_union import materialize_investor_flow_union

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    ls_id = registered_id(registry, "investor_flow_ls", args.ls_flow_dataset_id)
    kis_id = registered_id(registry, "investor_flow_kis_supplement", args.kis_supplement_dataset_id)
    result = materialize_investor_flow_union(
        ls_flow_silver_path=runtime.workspace.silver_root / ls_id,
        kis_supplement_silver_path=runtime.workspace.silver_root / kis_id,
        silver_root=runtime.workspace.silver_root,
    )
    register_dataset(runtime, "investor_flow", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path)}


def _run_quality(args: argparse.Namespace) -> Mapping[str, object]:
    from src.data.dataset_registry import DatasetRegistry
    from src.data.financial_quality import build_financial_quality_dataset

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    from src.data.fact_state import fact_quarantine_file, unresolved_events_file

    facts_id = registered_id(registry, "financial_facts", args.facts_dataset_id)
    state_root = runtime.workspace.state_root
    return build_financial_quality_dataset(
        runtime, facts_dataset_id=facts_id, decision_time=str(args.decision_time),
        quarantine_file=(
            args.quarantine_file
            if args.quarantine_file is not None
            else fact_quarantine_file(state_root, runtime.workspace.silver_root / facts_id)
        ),
        unresolved_events_file=(
            args.unresolved_events_file if args.unresolved_events_file is not None else unresolved_events_file(state_root)
        ),
    )


def _run_dividends(args: argparse.Namespace) -> Mapping[str, object]:
    from datetime import datetime

    from src.config import load_provider_policy, load_runtime_config
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ
    from src.data.dataset_registry import DatasetRegistry
    from src.data.dividend_events import materialize_dividend_events

    runtime = scoped_runtime(args)
    # 배당 타당성 게이트는 일별 시세가 있어야 동작하므로 CLI 경로도 현재 시세 데이터셋을 반드시 넘긴다.
    daily_id = registered_id(DatasetRegistry(runtime.workspace.state_root), "daily_market", None)
    path = materialize_dividend_events(
        catalog=scoped_catalog(runtime),
        silver_root=runtime.workspace.silver_root, calendar=xkrx_calendar_through(datetime.now(KRX_TZ).date()),
        daily_market_path=runtime.workspace.silver_root / daily_id,
        policy=load_provider_policy(load_runtime_config()).dividends,
    )
    register_dataset(runtime, "dividend_events", path.name)
    return {"dataset_id": path.name, "dataset_path": str(path)}


def _run_earnings_releases(args: argparse.Namespace) -> Mapping[str, object]:
    from datetime import datetime

    from src.config import load_provider_policy, load_runtime_config
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ
    from src.data.earnings_releases import materialize_earnings_releases

    runtime = scoped_runtime(args)
    path = materialize_earnings_releases(
        catalog=scoped_catalog(runtime),
        silver_root=runtime.workspace.silver_root, calendar=xkrx_calendar_through(datetime.now(KRX_TZ).date()),
        policy=load_provider_policy(load_runtime_config()).earnings_releases,
    )
    register_dataset(runtime, "earnings_releases", path.name)
    return {"dataset_id": path.name, "dataset_path": str(path)}


def _run_earnings_benchmark(args: argparse.Namespace) -> Mapping[str, object]:
    from src.data.dataset_registry import DatasetRegistry
    from src.data.earnings_releases import benchmark_earnings_releases

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    releases_id = registered_id(registry, "earnings_releases", None)
    facts_id = registered_id(registry, "financial_facts", None)
    report = benchmark_earnings_releases(
        releases_path=runtime.workspace.silver_root / releases_id,
        facts_path=runtime.workspace.silver_root / facts_id,
    )
    return {
        "releases": releases_id,
        "facts": facts_id,
        "benchmarks": [
            {
                "metric": entry.metric,
                "matched": entry.matched,
                "within_1pct": entry.within_1pct,
                "within_5pct": entry.within_5pct,
                "median_abs_rel_error": entry.median_abs_rel_error,
            }
            for entry in report
        ],
    }


def _run_market_actions(args: argparse.Namespace) -> Mapping[str, object]:
    from datetime import datetime

    from src.config import load_provider_policy, load_runtime_config
    from src.core.krx_calendar import xkrx_calendar_through
    from src.core.time import KRX_TZ
    from src.data.jobs.universe import read_corp_code_bridge
    from src.data.market_actions import materialize_market_actions

    runtime = scoped_runtime(args)
    mapping, _ = read_corp_code_bridge(scoped_catalog(runtime))
    provider = load_provider_policy(load_runtime_config())
    path = materialize_market_actions(
        catalog=scoped_catalog(runtime),
        silver_root=runtime.workspace.silver_root,
        calendar=xkrx_calendar_through(datetime.now(KRX_TZ).date()),
        bridge=dict(mapping),
        kind_keywords=tuple(provider.kind.search_keywords),
        kind_coverage_start=runtime.scope.evidence_start,
    )
    register_dataset(runtime, "market_actions", path.name)
    return {"dataset_id": path.name, "dataset_path": str(path)}


def _run_industry_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.data.cli.common import scoped_catalog
    from src.data.industry_silver import materialize_industry_classification_silver

    runtime = scoped_runtime(args)
    result = materialize_industry_classification_silver(
        catalog=scoped_catalog(runtime), silver_root=runtime.workspace.silver_root,
    )
    register_dataset(runtime, "industry", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path)}


def _run_hedge_series_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.config import load_runtime_config
    from src.data.dataset_registry import DatasetRegistry
    from src.data.hedge_series_silver import load_hedge_series_config, materialize_hedge_series_silver

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    hedge = load_hedge_series_config(load_runtime_config().hedge_series)
    result = materialize_hedge_series_silver(
        catalog=scoped_catalog(runtime), universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root, config=hedge, universe_dataset_id=universe_id,
    )
    register_dataset(runtime, "hedge_series", result.dataset_id)
    return (
        asdict(result)
        | {"dataset_path": str(result.dataset_path),
           "inverse_listing_session": result.inverse_listing_session.isoformat()}
    )


def _run_trend_series_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.config import load_runtime_config
    from src.data.dataset_registry import DatasetRegistry
    from src.data.hedge_series_silver import load_hedge_series_config, materialize_hedge_series_silver

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    trend = load_hedge_series_config(load_runtime_config().trend_series)
    result = materialize_hedge_series_silver(
        catalog=scoped_catalog(runtime), universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root, config=trend, universe_dataset_id=universe_id,
    )
    register_dataset(runtime, "trend_series", result.dataset_id)
    return (
        asdict(result)
        | {"dataset_path": str(result.dataset_path),
           "inverse_listing_session": result.inverse_listing_session.isoformat()}
    )


def _run_cash_series_silver(args: argparse.Namespace) -> Mapping[str, object]:
    from dataclasses import asdict

    from src.config import load_runtime_config
    from src.data.cash_series_silver import load_cash_series_config, materialize_cash_series_silver
    from src.data.dataset_registry import DatasetRegistry

    runtime = scoped_runtime(args)
    registry = DatasetRegistry(runtime.workspace.state_root)
    universe_id = resolve_input_id(runtime, registry, "ordinary_universe", args.universe_dataset_id)
    cash = load_cash_series_config(load_runtime_config().cash_series)
    result = materialize_cash_series_silver(
        catalog=scoped_catalog(runtime), universe_root=runtime.workspace.silver_root,
        silver_root=runtime.workspace.silver_root, config=cash, universe_dataset_id=universe_id,
    )
    register_dataset(runtime, "cash_series", result.dataset_id)
    return asdict(result) | {"dataset_path": str(result.dataset_path)}


BUILD_COMMANDS: tuple[Command, ...] = (
    Command("normalize-dart-facts", "Incremental DART fact refresh", _add_normalize, _run_normalize),
    Command("build-investor-flow-silver", "Build Silver LS investor-flow dataset from raw rows", _add_flow_silver, _run_flow_silver),
    Command("build-daily-market-silver", "Build Silver daily-market dataset with KRX base prices", _add_daily_silver, _run_daily_silver),
    Command("build-hedge-series-silver", "Build Silver hedge-series dataset", _add_daily_silver, _run_hedge_series_silver),
    Command("build-trend-series-silver", "Build Silver KOSPI 200 trend-series dataset", _add_daily_silver, _run_trend_series_silver),
    Command("build-cash-series-silver", "Build Silver cash-series dataset", _add_daily_silver, _run_cash_series_silver),
    Command("build-ordinary-universe", "Build the point-in-time ordinary-share universe", _add_ordinary, _run_ordinary),
    Command("build-market-panel", "Build decision-safe Gold market panel", _add_panel, _run_panel),
    Command("build-reference-benchmarks", "Build frictionless Gold reference benchmarks", _add_benchmarks, _run_benchmarks),
    Command("refresh-scope", "Refresh the whole scope (collect, build, verify, register)", _add_refresh, _run_refresh),
    Command("build-investor-flow-kis-supplement", "Build the provider-tagged KIS supplement for the LS gap", _add_kis_supplement, _run_kis_supplement),
    Command("build-investor-flow-union", "Union certified LS flow and its KIS supplement into one dataset", _add_flow_union, _run_flow_union),
    Command("build-financial-quality", "Build certified financial-quality evidence from facts and quarantine", _add_quality, _run_quality),
    Command("build-dividend-events", "Build Silver cash-dividend events from dated decision filings", _add_scoped_only, _run_dividends),
    Command("build-earnings-releases", "Build Silver early-earnings releases from preliminary filings", _add_scoped_only, _run_earnings_releases),
    Command("benchmark-earnings-releases", "Report preliminary-vs-filed agreement", _add_scoped_only, _run_earnings_benchmark),
    Command("build-market-actions", "Build Silver exchange market actions from DART, KIND and daily flags", _add_scoped_only, _run_market_actions),
    Command("build-industry-classification-silver", "Build the certified industry classification Silver snapshot", _add_scoped_only, _run_industry_silver),
)
