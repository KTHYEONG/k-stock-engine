"""Research CLI: cube builds, ML trend-cash backtests, registration and holdout."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from src.core.pit import PITDataError
from src.data.research_protocol import LockboxError

__all__ = ["main"]

_LOG = logging.getLogger(__name__)


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope-config", type=Path, required=False, default=None)
    parser.add_argument("--data-root", type=Path, required=False, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.research")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cube = sub.add_parser("build-cube")
    _common_args(build_cube)

    backtest = sub.add_parser("backtest")
    _common_args(backtest)
    backtest.add_argument("--spec", type=Path, required=True)

    register = sub.add_parser("register")
    _common_args(register)
    register.add_argument("--spec", type=Path, required=True)

    holdout = sub.add_parser("holdout")
    _common_args(holdout)
    holdout.add_argument("--spec", type=Path, required=True)

    trials = sub.add_parser("trials")
    _common_args(trials)
    return parser


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _pipeline_for(args: argparse.Namespace) -> Any:  # pragma: no cover - needs real datasets
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.core.time import KRX_TZ
    from src.data.dataset_registry import DatasetRegistry
    from src.data.research_protocol import LockboxLedger, load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.cube import CubeInputs, load_research_cube
    from src.research.ledger_bridge import run_ledger
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import TrialRegistry

    runtime_config = load_runtime_config()
    runtime = resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
    protocol = load_research_protocol(runtime_config.research_protocol, runtime.scope)
    registry = DatasetRegistry(runtime.workspace.state_root)
    state_root = runtime.workspace.state_root
    market_id = registry.require("market_panel")
    dividend_id = registry.require("dividend_events")
    facts_id = registry.require("financial_facts")
    flow_id = registry.require("investor_flow")
    releases_id = registry.require("earnings_releases")
    inputs = CubeInputs(
        market_panel=runtime.workspace.gold_root / market_id,
        dividend_events=runtime.workspace.silver_root / dividend_id,
        financial_facts=runtime.workspace.silver_root / facts_id,
        investor_flow=runtime.workspace.silver_root / flow_id,
        earnings_releases=runtime.workspace.silver_root / releases_id,
        market_rules=load_krx_market_rules(runtime_config.market_rules),
        dividend_withholding_rate=Decimal("0.154"),
    )
    cube = load_research_cube(inputs, cache_root=state_root / "research" / "cube")
    trial_registry = TrialRegistry(state_root / "research" / "trials_ml")
    lockbox = LockboxLedger(state_root=state_root, protocol=protocol, now=lambda: datetime.now(KRX_TZ))
    from src.data.datasets import read_dataset

    dividends = read_dataset(runtime.workspace.silver_root / dividend_id).collect()
    context = PipelineContext(
        protocol=protocol,
        cube=cube,
        registry=trial_registry,
        lockbox=lockbox,
        panel_dir=runtime.workspace.gold_root / market_id,
        dividends=dividends,
        rules=load_krx_market_rules(runtime_config.market_rules),
        engine_config_path=runtime_config.engine,
        market_cache_root=state_root / "research" / "market_cache",
        reports_root=state_root / "research" / "reports",
        scores_root=state_root / "research" / "scores",
        ledger_runner=run_ledger,
        now=lambda: datetime.now(KRX_TZ),
    )
    return Pipeline(context)


def _load_spec(path: Path) -> Any:
    from src.research.pipeline import load_strategy_spec

    return load_strategy_spec(Path(path))


def _run_build_cube(args: argparse.Namespace) -> Mapping[str, object]:  # pragma: no cover - needs real datasets
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.data.dataset_registry import DatasetRegistry
    from src.data.research_protocol import load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.cube import CubeInputs, load_research_cube

    runtime_config = load_runtime_config()
    runtime = resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
    load_research_protocol(runtime_config.research_protocol, runtime.scope)
    registry = DatasetRegistry(runtime.workspace.state_root)
    state_root = runtime.workspace.state_root
    inputs = CubeInputs(
        market_panel=runtime.workspace.gold_root / registry.require("market_panel"),
        dividend_events=runtime.workspace.silver_root / registry.require("dividend_events"),
        financial_facts=runtime.workspace.silver_root / registry.require("financial_facts"),
        investor_flow=runtime.workspace.silver_root / registry.require("investor_flow"),
        earnings_releases=runtime.workspace.silver_root / registry.require("earnings_releases"),
        market_rules=load_krx_market_rules(runtime_config.market_rules),
        dividend_withholding_rate=Decimal("0.154"),
    )
    cube = load_research_cube(inputs, cache_root=state_root / "research" / "cube")
    shape = (len(cube.sessions), len(cube.instrument_ids))
    present = cube.arrays.get("present")
    coverage = float(present.mean()) if present is not None else 0.0
    return {"cube_id": cube.cube_id, "sessions": shape[0], "instruments": shape[1], "coverage": coverage}


def _run_backtest(args: argparse.Namespace) -> Mapping[str, object]:
    pipeline = _pipeline_for(args)
    spec = _load_spec(args.spec)
    report = pipeline.evaluate_discovery(spec)
    return {
        "spec_hash": report.spec_hash,
        "passed": bool(report.passed),
        "digest": report.digest,
    }


def _run_register(args: argparse.Namespace) -> Mapping[str, object]:
    pipeline = _pipeline_for(args)
    spec = _load_spec(args.spec)
    pipeline.register_finalist(spec)
    return {"registered": spec.spec_hash}


def _run_holdout(args: argparse.Namespace) -> Mapping[str, object]:
    pipeline = _pipeline_for(args)
    spec = _load_spec(args.spec)
    report = pipeline.holdout(spec)
    return {"spec_hash": report.spec_hash, "passed": bool(report.passed), "digest": report.digest}


def _run_trials(args: argparse.Namespace) -> Mapping[str, object]:
    from src.config.runtime import load_runtime_config
    from src.data.research_protocol import load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.registry import TrialRegistry
    from src.research.stats import effective_trial_count

    runtime_config = load_runtime_config()
    runtime = resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
    protocol = load_research_protocol(runtime_config.research_protocol, runtime.scope)
    registry = TrialRegistry(runtime.workspace.state_root / "research" / "trials_ml")
    trials = registry.trials()
    by_sessions: dict[tuple[str, ...], list[str]] = {}
    for trial in trials:
        try:
            stored = registry.returns(trial.trial_id)
        except PITDataError:
            continue
        key = tuple(day.isoformat() for day in stored.sessions)
        by_sessions.setdefault(key, []).append(trial.trial_id)
    effective = 1.0
    if by_sessions:
        largest = max(by_sessions.values(), key=len)
        columns = []
        for trial_id in largest:
            try:
                stored = registry.returns(trial_id)
            except PITDataError:  # pragma: no cover - intact after the first pass
                continue
            columns.append(np.asarray(stored.net, dtype=float))
        if columns:
            if len(columns) == 1:
                try:
                    matrix = np.column_stack(columns)
                    effective = float(effective_trial_count(matrix))
                except ValueError:
                    effective = 1.0
            elif columns and len(columns[0]) >= 2:
                matrix = np.column_stack(columns)
                try:
                    effective = float(effective_trial_count(matrix))
                except ValueError:
                    effective = float(len(columns))
    prior_trials = int(protocol.criteria.prior_trials)
    prior_effective = float(protocol.criteria.prior_effective_trials)
    return {
        "raw_count": len(trials),
        "effective_count": float(effective),
        "prior_trials": prior_trials,
        "prior_effective_trials": float(prior_effective),
        "n_for_dsr": float(effective) + float(prior_effective),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one research command with a JSON-line envelope."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build-cube":
            _emit(_run_build_cube(args))
        elif args.command == "backtest":
            _emit(_run_backtest(args))
        elif args.command == "register":
            _emit(_run_register(args))
        elif args.command == "holdout":
            _emit(_run_holdout(args))
        elif args.command == "trials":
            _emit(_run_trials(args))
    except LockboxError as exc:
        _LOG.error("[RISK] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    return 0
