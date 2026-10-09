"""Research CLI: cube builds, account-engine evaluations and champion promotion."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.pit import PITDataError
from src.data.research_protocol import WindowError

if TYPE_CHECKING:
    from src.research.champion import ChampionRecord

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

    evaluate = sub.add_parser("evaluate")
    _common_args(evaluate)
    evaluate.add_argument("--spec", type=Path, required=True)
    evaluate.add_argument("--capital", type=int, required=False, default=None)

    champion = sub.add_parser("champion")
    _common_args(champion)
    champion.add_argument("--sync-file", action="store_true")

    challenge = sub.add_parser("challenge")
    _common_args(challenge)
    challenge.add_argument("--spec", type=Path, required=True)
    challenge.add_argument("--neighbors", type=Path, nargs="*", default=[])

    promote = sub.add_parser("promote")
    _common_args(promote)
    promote.add_argument("--spec", type=Path, required=True)
    promote.add_argument("--bootstrap", action="store_true")
    promote.add_argument(
        "--policy-override",
        metavar="RATIONALE",
        default=None,
        help="adopt a saved non-promotable decision as a risk policy when only the growth-margin tests failed",
    )
    return parser


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _pipeline_for(args: argparse.Namespace) -> Any:  # pragma: no cover - needs real datasets
    from src.config.runtime import load_runtime_config
    from src.core.market_rules import load_krx_market_rules
    from src.core.time import KRX_TZ
    from src.data.dataset_registry import DatasetRegistry
    from src.data.research_protocol import WindowGuard, load_research_protocol
    from src.data.runtime import resolve_data_runtime
    from src.research.cube import CubeInputs, load_research_cube
    from src.research.ledger_bridge import run_ledger
    from src.research.pipeline import Pipeline, PipelineContext
    from src.research.registry import RunRegistry

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
    run_registry = RunRegistry(state_root / "research" / "runs")
    guard = WindowGuard(protocol=protocol, last_session=cube.sessions[-1])
    from src.data.datasets import read_dataset

    dividends = read_dataset(runtime.workspace.silver_root / dividend_id).collect()
    hedge_id = registry.require("hedge_series")
    hedge_frame = read_dataset(runtime.workspace.silver_root / hedge_id).collect()
    from src.research.hedge import hedge_inputs_from_frame
    from src.research.ledger_bridge import cash_returns_from_frame

    hedge_inputs = hedge_inputs_from_frame(hedge_frame, sessions=list(cube.sessions))
    cash_id = registry.require("cash_series")
    cash_frame = read_dataset(runtime.workspace.silver_root / cash_id).collect()
    cash_returns = cash_returns_from_frame(cash_frame, sessions=list(cube.sessions))
    trend_id = registry.current("trend_series")
    if trend_id is not None:
        trend_frame = read_dataset(runtime.workspace.silver_root / trend_id).collect()
        trend_inputs = hedge_inputs_from_frame(trend_frame, sessions=list(cube.sessions))
    else:
        trend_inputs = None
    dataset_ids = {
        "market_panel": market_id,
        "dividend_events": dividend_id,
        "hedge_series": hedge_id,
        "cash_series": cash_id,
    }
    if trend_id is not None:
        dataset_ids["trend_series"] = trend_id
    context = PipelineContext(
        protocol=protocol,
        cube=cube,
        registry=run_registry,
        guard=guard,
        panel_dir=runtime.workspace.gold_root / market_id,
        dividends=dividends,
        hedge_inputs=hedge_inputs,
        rules=load_krx_market_rules(runtime_config.market_rules),
        engine_config_path=runtime_config.engine,
        market_cache_root=state_root / "research" / "market_cache",
        reports_root=state_root / "research" / "reports",
        scores_root=state_root / "research" / "scores",
        ledger_runner=run_ledger,
        now=lambda: datetime.now(KRX_TZ),
        dataset_ids=dataset_ids,
        cash_returns=cash_returns,
        trend_inputs=trend_inputs,
    )
    return Pipeline(context)


def _load_spec(path: Path) -> Any:
    from src.config.runtime import load_runtime_config
    from src.research.pipeline import load_strategy_spec

    runtime_config = load_runtime_config()
    return load_strategy_spec(Path(path), futures_constants=runtime_config.futures_constants)


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


def _run_evaluate(args: argparse.Namespace) -> Mapping[str, object]:
    pipeline = _pipeline_for(args)
    spec = _load_spec(args.spec)
    capital = int(args.capital) if args.capital is not None else None
    run = pipeline.evaluate(spec, capital_krw=capital)
    report = run.report
    report_path = Path(pipeline._ctx.reports_root) / f"{report.spec_hash}_{report.run_id}.json"
    return {
        "spec_hash": report.spec_hash,
        "run_id": report.run_id,
        "passed": bool(report.passed),
        "objective_j": float(report.objective_j),
        "g": float(report.metrics["g"]),
        "mdd": float(report.metrics["mdd"]),
        "report_path": str(report_path),
    }


def _champion_store(args: argparse.Namespace) -> Any:
    from src.data.runtime import resolve_data_runtime
    from src.research.champion import ChampionStore

    runtime = resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
    return ChampionStore(runtime.workspace.state_root / "research" / "champion")


def _champion_file_path(args: argparse.Namespace) -> Path:
    from src.config.runtime import load_runtime_config

    return load_runtime_config().champion_file


def _champion_futures_constants(args: argparse.Namespace) -> Path:
    from src.config.runtime import load_runtime_config

    return load_runtime_config().futures_constants


def _repo_relative_champion(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _champion_file_state(current: ChampionRecord, path: Path, futures_constants: Path) -> str:
    """``"missing"``, ``"drift"`` or ``"synced"``: whether ``path`` resolves to the stored champion's ``spec_hash``."""
    target = Path(path)
    if not target.exists():
        return "missing"
    if not target.is_file():
        return "drift"
    from src.research.pipeline import load_strategy_spec

    try:
        file_spec = load_strategy_spec(target, futures_constants=Path(futures_constants))
    except (OSError, ValueError):
        return "drift"
    return "synced" if file_spec.spec_hash == current.spec_hash else "drift"


def _write_champion_file(current: ChampionRecord, path: Path, futures_constants: Path) -> None:
    """Atomically replace ``path`` with the rendered stored champion spec."""
    from src.research.pipeline import strategy_spec_from_canonical_json
    from src.research.strategy_file import render_strategy_toml

    spec = strategy_spec_from_canonical_json(current.spec_json)
    text = render_strategy_toml(spec, futures_constants=Path(futures_constants))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def _file_spec_hash(path: Path, futures_constants: Path) -> str:
    from src.research.pipeline import load_strategy_spec

    try:
        return load_strategy_spec(Path(path), futures_constants=Path(futures_constants)).spec_hash
    except (OSError, ValueError):
        return "unreadable"


def _require_champion_file_synced(current: ChampionRecord, path: Path, futures_constants: Path) -> None:
    state = _champion_file_state(current, path, futures_constants)
    if state == "synced":
        return
    file_hash = "missing" if state == "missing" else _file_spec_hash(path, futures_constants)
    raise ValueError(
        f"champion file {path} is {state}: file spec {file_hash} != champion {current.spec_hash}; "
        "run 'champion --sync-file' or restore the file"
    )


def _rewrite_champion_file(record: ChampionRecord, args: argparse.Namespace) -> None:
    _write_champion_file(record, _champion_file_path(args), _champion_futures_constants(args))


def _run_champion(args: argparse.Namespace) -> Mapping[str, object]:
    from src.research.champion import champion_record_fields

    store = _champion_store(args)
    current = store.current()
    history = store.history()[-5:]
    if getattr(args, "sync_file", False):
        if current is None:
            raise ValueError("no champion: run 'promote --spec PATH --bootstrap' first")
        path = _champion_file_path(args)
        _write_champion_file(current, path, _champion_futures_constants(args))
        return {"champion_file": _repo_relative_champion(Path(path)), "state": "synced"}
    if current is None:
        state = "missing"
    else:
        state = _champion_file_state(current, _champion_file_path(args), _champion_futures_constants(args))
    return {
        "champion": None if current is None else champion_record_fields(current),
        "history": [champion_record_fields(record) for record in history],
        "champion_file_state": state,
    }


def _run_challenge(args: argparse.Namespace) -> Mapping[str, object]:
    from src.research.champion import decide_challenge, with_seeds
    from src.research.evaluation import EvaluationPolicy
    from src.research.pipeline import strategy_spec_from_canonical_json

    store = _champion_store(args)
    current = store.current()
    if current is None:
        raise ValueError("no champion: run 'promote --spec PATH --bootstrap' first")
    _require_champion_file_synced(current, _champion_file_path(args), _champion_futures_constants(args))
    pipeline = _pipeline_for(args)
    protocol = pipeline._ctx.protocol
    policy = EvaluationPolicy.model_validate(protocol.evaluation.model_dump())
    challenger_spec = _load_spec(args.spec)
    champion_spec = strategy_spec_from_canonical_json(current.spec_json)
    challenger = pipeline.evaluate(challenger_spec)
    champion = pipeline.evaluate(with_seeds(champion_spec, protocol.champion.champion_seeds))
    neighbors = [_load_spec(path) for path in (args.neighbors or [])]
    neighbor_runs = tuple(pipeline.evaluate(neighbor) for neighbor in neighbors)
    prior_decisions = sum(1 for d in store.decisions() if d.champion_hash == current.spec_hash)
    decision = decide_challenge(
        challenger=challenger,
        champion=champion,
        neighbors=neighbor_runs,
        challenger_spec=challenger_spec,
        champion_spec=champion_spec,
        protocol=protocol,
        policy=policy,
        prior_decisions=prior_decisions,
    )
    path = store.save_decision(decision)
    return {
        "promotable": decision.promotable,
        "path": decision.path,
        "reasons": list(decision.reasons),
        "baseline_hash": decision.baseline_hash,
        "delta_mean": decision.paired.mean,
        "delta_lower": decision.paired.lower,
        "tail_mean": decision.tail.mean if decision.tail is not None else None,
        "tail_lower": decision.tail.lower if decision.tail is not None else None,
        "noninferiority_margin": decision.noninferiority_margin,
        "challenger_j": decision.challenger_j,
        "champion_j": decision.champion_j,
        "alpha_effective": decision.alpha_effective,
        "paired_horizon_sessions": decision.paired_horizon_sessions,
        "decision_path": str(path),
    }


def _run_promote(args: argparse.Namespace) -> Mapping[str, object]:
    store = _champion_store(args)
    if not args.bootstrap:
        current = store.current()
        if current is None:
            raise ValueError("no champion to promote over: pass --bootstrap")
        _require_champion_file_synced(current, _champion_file_path(args), _champion_futures_constants(args))
    spec = _load_spec(args.spec)
    pipeline = _pipeline_for(args)
    run = pipeline.evaluate(spec)
    now = pipeline._ctx.now()
    if args.bootstrap:
        record = store.bootstrap(run=run, spec=spec, spec_path=Path(args.spec), now=now)
    else:
        current = store.current()
        if current is None:
            raise ValueError("no champion to promote over: pass --bootstrap")
        override = args.policy_override
        _require_champion_file_synced(current, _champion_file_path(args), _champion_futures_constants(args))
        decision = _saved_decision(
            store, spec_hash=spec.spec_hash, run=run, champion=current, promotable=override is None
        )
        if override is None:
            record = store.promote(decision=decision, run=run, spec=spec, spec_path=Path(args.spec), now=now)
        else:
            record = store.adopt_policy(
                decision=decision, run=run, spec=spec, spec_path=Path(args.spec), now=now, rationale=override
            )
    _LOG.info(
        "[PORTFOLIO] champion promoted spec_hash=%s reason=%s J=%.6f",
        record.spec_hash,
        record.reason,
        record.objective_j,
    )
    _rewrite_champion_file(record, args)
    return {
        "spec_hash": record.spec_hash,
        "reason": record.reason,
        "objective_j": record.objective_j,
        "run_id": record.run_id,
        "decision_digest": record.decision_digest,
    }


def _saved_decision(store: Any, *, spec_hash: str, run: Any, champion: Any, promotable: bool = True) -> Any:
    """The one saved decision (promotable, or non-promotable for a policy override) for this spec vs the current
    champion on the current window.

    Re-evaluating the challenger first turns the stored ``run_id`` into a freshness check: a decision made
    before the cube grew names a different run and is therefore refused.
    """
    window = (run.evidence.sessions[0], run.evidence.sessions[-1])
    matches = [
        decision
        for decision in store.decisions()
        if decision.promotable == promotable
        and decision.challenger_hash == spec_hash
        and decision.champion_hash == champion.spec_hash
        and decision.window == window
        and decision.run_ids[0] == run.report.run_id
    ]
    if not matches:
        raise ValueError(
            f"no saved {'promotable ' if promotable else 'non-promotable '}decision for this spec against the "
            "current champion on the current window: "
            "run 'challenge --spec PATH' first"
        )
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} saved decisions match this spec and window; run 'challenge' again")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one research command with a JSON-line envelope."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build-cube":
            _emit(_run_build_cube(args))
        elif args.command == "evaluate":
            _emit(_run_evaluate(args))
        elif args.command == "champion":
            _emit(_run_champion(args))
        elif args.command == "challenge":
            _emit(_run_challenge(args))
        elif args.command == "promote":
            _emit(_run_promote(args))
    except WindowError as exc:
        _LOG.error("[RISK] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    except (PITDataError, ValueError, OSError) as exc:
        _LOG.error("[DATA] command=%s status=failed error=%s", args.command, exc)
        _emit({"error": str(exc)})
        return 1
    return 0
