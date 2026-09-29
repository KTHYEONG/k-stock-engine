"""Research CLI: cube builds, screens, gate validations, and holdout execution."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import PITDataError

__all__ = ["expand_grid", "main"]

_LOG = logging.getLogger(__name__)


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def expand_grid(grid_path: Path) -> list[Any]:
    """Expand a grid TOML into validated strategy specs."""
    from src.research.strategy import StrategySpec

    raw = _read_toml(Path(grid_path))
    base_ref = raw.get("base")
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError("grid TOML must declare base = \"<spec path>\"")
    grid_file = Path(grid_path)
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        candidate = grid_file.parent / base_path
        base_path = candidate if candidate.exists() else Path.cwd() / base_path
    base_raw = _read_toml(base_path)
    axes = raw.get("axes", {})
    if not isinstance(axes, dict) or not axes:
        raise ValueError("grid TOML must declare a non-empty [axes] table")
    keys = sorted(axes)
    value_lists: list[list[Any]] = []
    for key in keys:
        values = axes[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid axis {key!r} must be a non-empty list")
        value_lists.append(list(values))
    specs: list[Any] = []
    for combo in itertools.product(*value_lists):
        merged = copy.deepcopy(base_raw)
        for key, value in zip(keys, combo, strict=True):
            _set_dotted(merged, key, value)
        try:
            specs.append(StrategySpec.model_validate(merged))
        except ValueError as exc:
            raise ValueError(f"invalid grid member {dict(zip(keys, combo, strict=True))}: {exc}") from exc
    hashes = {spec.spec_hash for spec in specs}
    if len(hashes) != len(specs):
        raise ValueError("grid expansion produced duplicate specs")
    return specs


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope-config", type=Path, required=False, default=None)
    parser.add_argument("--data-root", type=Path, required=False, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.research")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cube = sub.add_parser("build-cube")
    _common_args(build_cube)

    screen = sub.add_parser("screen")
    _common_args(screen)
    screen.add_argument("--spec", type=Path, required=True)

    family = sub.add_parser("family")
    _common_args(family)
    family.add_argument("--grid", type=Path, required=True)

    validate = sub.add_parser("validate")
    _common_args(validate)
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--grid", type=Path, required=True)

    register = sub.add_parser("register-finalists")
    _common_args(register)
    register.add_argument("--spec", type=Path, required=True, action="append")

    holdout = sub.add_parser("holdout")
    _common_args(holdout)
    holdout.add_argument("--spec", type=Path, required=True)

    forward = sub.add_parser("forward")
    _common_args(forward)
    forward.add_argument("--spec", type=Path, required=True)

    trials = sub.add_parser("trials")
    _common_args(trials)
    return parser


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _evaluator_for(args: argparse.Namespace) -> Any:  # pragma: no cover - needs real datasets
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.core.time import KRX_TZ
    from src.data.dataset_registry import DatasetRegistry
    from src.data.research_protocol import LockboxLedger, load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.cube import CubeInputs, load_research_cube
    from src.research.evaluator import EvaluationContext, Evaluator
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
        dividend_withholding_rate=__import__("decimal").Decimal("0.154"),
    )
    cube = load_research_cube(inputs, cache_root=state_root / "research" / "cube")
    trial_registry = TrialRegistry(state_root / "research" / "trials")
    lockbox = LockboxLedger(state_root=state_root, protocol=protocol, now=lambda: datetime.now(KRX_TZ))
    try:
        from src.data.datasets import read_dataset

        dividends = read_dataset(state_root.parent.parent / "silver" / runtime.scope.scope_id / dividend_id).collect()
    except Exception:
        dividends = pl.DataFrame()
    context = EvaluationContext(
        protocol=protocol, cube=cube, cube_inputs=inputs, registry=trial_registry, lockbox=lockbox,
        engine_config_path=runtime_config.engine, market_cache_root=state_root / "research" / "market_cache",
        reports_root=state_root / "research" / "reports", dividends=dividends,
        now=lambda: datetime.now(KRX_TZ),
    )
    return Evaluator(context)


def _load_spec(path: Path) -> Any:
    from src.research.strategy import load_strategy_spec

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
        dividend_withholding_rate=__import__("decimal").Decimal("0.154"),
    )
    cube = load_research_cube(inputs, cache_root=state_root / "research" / "cube")
    shape = (len(cube.sessions), len(cube.instrument_ids))
    present = cube.arrays.get("present")
    coverage = float(present.mean()) if present is not None else 0.0
    return {"cube_id": cube.cube_id, "sessions": shape[0], "instruments": shape[1], "coverage": coverage}


def _run_screen(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    spec = _load_spec(args.spec)
    record = evaluator.screen(spec)
    return {"trial_id": record.trial_id, "spec_hash": record.spec_hash,
            "metrics": dict(record.metrics), "segment": record.segment.value}


def _run_family(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    specs = expand_grid(args.grid)
    records = evaluator.screen_family(specs)
    return {"trials": [{"trial_id": record.trial_id, "spec_hash": record.spec_hash} for record in records],
            "count": len(records)}


def _run_validate(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    spec = _load_spec(args.spec)
    family = expand_grid(args.grid)
    report = evaluator.validate(spec, family)
    return {"spec_hash": report.spec_hash, "trial_id": report.trial_id, "passed": report.passed,
            "digest": report.digest,
            "checks": [{"name": check.name, "passed": check.passed, "value": check.value} for check in report.checks]}


def _run_register(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    specs = [_load_spec(path) for path in (args.spec or [])]
    evaluator.register_finalists(specs)
    return {"registered": [spec.spec_hash for spec in specs]}


def _run_holdout(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    spec = _load_spec(args.spec)
    report = evaluator.holdout(spec)
    return {"spec_hash": report.spec_hash, "passed": report.passed, "digest": report.digest}


def _run_forward(args: argparse.Namespace) -> Mapping[str, object]:
    evaluator = _evaluator_for(args)
    spec = _load_spec(args.spec)
    record = evaluator.forward(spec)
    return {"trial_id": record.trial_id, "spec_hash": record.spec_hash}


def _run_trials(args: argparse.Namespace) -> Mapping[str, object]:
    from src.config.runtime import load_runtime_config
    from src.data.research_protocol import load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.registry import TrialRegistry
    from src.research.stats import effective_trial_count

    runtime_config = load_runtime_config()
    runtime = resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
    protocol = load_research_protocol(runtime_config.research_protocol, runtime.scope)
    registry = TrialRegistry(runtime.workspace.state_root / "research" / "trials")
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
            stored = registry.returns(trial_id)
            columns.append(__import__("numpy").asarray(stored.net, dtype=float))
        if columns and len(columns[0]) >= 2:
            import numpy as np

            matrix = np.column_stack(columns)
            try:
                effective = float(effective_trial_count(matrix))
            except ValueError:
                effective = float(len(columns))
    return {"raw_count": len(trials), "effective_count": effective,
            "prior_trials": protocol.prior_trials, "n_for_dsr": effective + float(protocol.prior_trials)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one research command with a JSON-line envelope."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build-cube":
            _emit(_run_build_cube(args))
        elif args.command == "screen":
            _emit(_run_screen(args))
        elif args.command == "family":
            _emit(_run_family(args))
        elif args.command == "validate":
            _emit(_run_validate(args))
        elif args.command == "register-finalists":
            _emit(_run_register(args))
        elif args.command == "holdout":
            _emit(_run_holdout(args))
        elif args.command == "forward":
            _emit(_run_forward(args))
        elif args.command == "trials":
            _emit(_run_trials(args))
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    return 0
