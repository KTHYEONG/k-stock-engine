"""Backtest-run CLI over one strategy, a capital grid, and halted-exit policies."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one strategy across a capital grid")
    parser.add_argument("--scope-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--panel-dataset-id", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--capital", dest="capitals", action="append", type=int, required=True)
    parser.add_argument("--deposits", type=Path, required=False, default=None)
    parser.add_argument("--dividends-dataset-id", type=str, required=False, default=None)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--allow-forward", action="store_true", default=False)
    return parser.parse_args(argv)


def _read_toml(path: Path) -> dict[str, object]:
    import tomllib

    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError:
        raise
    except ValueError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"invalid TOML in {path}: top level must be a table")
    return raw


def _require_engine_key(raw: dict[str, object], sections: list[dict[str, object]], key: str) -> object:
    if key in raw:
        return raw[key]
    for section in sections:
        if key in section:
            return section[key]
    raise ValueError(f"engine config is missing required key: {key}")


def _load_deposits(path: Path | None) -> dict[date, int]:
    import csv
    from datetime import date

    deposits: dict[date, int] = {}
    if path is None:
        return deposits
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != {"date", "amount_krw"}:
                raise ValueError(f"deposits CSV must have header date,amount_krw: {path}")
            for row in reader:
                day = date.fromisoformat(str(row["date"]).strip())
                amount = int(str(row["amount_krw"]).strip())
                if amount <= 0:
                    raise ValueError(f"deposit amount must be a positive KRW int, got {amount!r}")
                deposits[day] = deposits.get(day, 0) + amount
    except (OSError, ValueError):
        raise
    return deposits


def _resolve_rules_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "config" / "market" / "krx_market_rules.toml"
    if candidate.is_file():
        return candidate
    return Path("config/market/krx_market_rules.toml")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one strategy across a capital grid and both halted-exit scenarios.

    Flags: ``--scope-config``, ``--data-root``, ``--panel-dataset-id``,
    ``--strategy``, ``--strategy-config`` (TOML of constructor params),
    ``--engine-config`` (TOML: execution, costs, cash_buffer), ``--capital``
    (repeatable, KRW int), ``--deposits`` (CSV ``date,amount_krw``, optional),
    ``--dividends-dataset-id`` (optional), ``--start``, ``--end``.

    Returns 0 on success; 1 with a JSON ``error`` on stdout for PITDataError,
    ValueError, or OSError.
    """
    from datetime import date
    from decimal import Decimal, InvalidOperation

    args = _parse_args(argv)
    try:
        from src.data.research_scope import load_research_scope

        scope = load_research_scope(Path(args.scope_config))
        data_root = Path(args.data_root)
        if not data_root.is_absolute():
            data_root = (Path.cwd() / data_root).resolve()
        start = date.fromisoformat(str(args.start))
        end = date.fromisoformat(str(args.end))
        if start > end:
            raise ValueError(f"run window [{start}, {end}] must satisfy start <= end")
        if (end > scope.holdout_end or start > scope.holdout_end) and not bool(args.allow_forward):
            raise ValueError(
                f"end {end.isoformat()} is beyond holdout_end {scope.holdout_end.isoformat()}; "
                "pass --allow-forward to run on sealed forward dates"
            )
        capitals: list[int] = list(args.capitals or [])
        if not capitals:
            raise ValueError("at least one --capital is required")
        for capital in capitals:
            if isinstance(capital, bool) or not isinstance(capital, int) or capital <= 0:
                raise ValueError(f"capital must be a positive KRW int, got {capital!r}")

        from src.backtest.strategy import STRATEGIES

        strategy_name = str(args.strategy)
        if strategy_name not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy_name!r}")
        strategy_params_raw = _read_toml(Path(args.strategy_config))
        strategy_params: dict[str, object] = dict(strategy_params_raw)

        engine_raw = _read_toml(Path(args.engine_config))
        execution_section = engine_raw.get("execution")
        costs_section = engine_raw.get("costs")
        sections: list[dict[str, object]] = [
            section for section in (execution_section, costs_section) if isinstance(section, dict)
        ]

        from src.backtest.costs import CostConfig
        from src.backtest.engine import DelistPolicy, EngineConfig, run_backtest
        from src.backtest.execution import ExecutionConfig, ExecutionScenario
        from src.backtest.manifest import write_run
        from src.backtest.metrics import summarize
        from src.core.market_rules import load_krx_market_rules

        scenario_raw = _require_engine_key(engine_raw, sections, "scenario")
        participation_raw = _require_engine_key(engine_raw, sections, "max_participation")
        carry_raw = _require_engine_key(engine_raw, sections, "carry_unfilled")
        commission_raw = _require_engine_key(engine_raw, sections, "commission_rate")
        impact_raw = _require_engine_key(engine_raw, sections, "impact_k")
        buffer_raw = _require_engine_key(engine_raw, sections, "cash_buffer")

        try:
            scenario = ExecutionScenario(str(scenario_raw))
        except ValueError as exc:
            raise ValueError(f"invalid execution scenario: {scenario_raw!r}") from exc
        if isinstance(participation_raw, bool) or not isinstance(participation_raw, (int, float)):
            raise ValueError(f"max_participation must be a number, got {participation_raw!r}")
        max_participation = float(participation_raw)
        if not 0.0 < max_participation <= 1.0:
            raise ValueError(f"max_participation must be in (0, 1], got {participation_raw!r}")
        if not isinstance(carry_raw, bool):
            raise ValueError(f"carry_unfilled must be a bool, got {carry_raw!r}")
        try:
            commission_rate = Decimal(str(commission_raw))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid commission_rate: {commission_raw!r}") from exc
        if commission_rate < 0:
            raise ValueError(f"commission_rate must be >= 0, got {commission_raw!r}")
        if isinstance(impact_raw, bool) or not isinstance(impact_raw, (int, float)):
            raise ValueError(f"impact_k must be a number, got {impact_raw!r}")
        impact_k = float(impact_raw)
        if isinstance(buffer_raw, bool) or not isinstance(buffer_raw, (int, float)):
            raise ValueError(f"cash_buffer must be a number, got {buffer_raw!r}")
        cash_buffer = float(buffer_raw)

        execution = ExecutionConfig(
            scenario=scenario,
            max_participation=max_participation,
            carry_unfilled=carry_raw,
        )
        costs = CostConfig(commission_rate=commission_rate, impact_k=impact_k)

        deposits = _load_deposits(Path(args.deposits) if args.deposits is not None else None)

        gold_root = data_root / "gold" / scope.scope_id
        panel_dir = gold_root / str(args.panel_dataset_id)
        state_root = data_root / "state" / scope.scope_id
        run_root = state_root / "backtests"
        run_root.mkdir(parents=True, exist_ok=True)
        cache_root = state_root / "market_cache"

        from src.backtest.events import build_engine_events
        from src.backtest.market import load_market_arrays

        arrays = load_market_arrays(panel_dir=panel_dir, cache_root=cache_root)

        dividends = None
        dividends_id = args.dividends_dataset_id
        if dividends_id is not None:
            import polars as pl

            div_path = gold_root / str(dividends_id)
            if div_path.is_dir():
                files = sorted(str(p) for p in div_path.rglob("*.parquet"))
                if not files:
                    raise ValueError(f"dividends dataset has no parquet files: {div_path}")
                dividends = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
            elif div_path.is_file():
                dividends = pl.read_parquet(str(div_path))
            else:
                raise OSError(f"dividends dataset not found: {div_path}")

        events = build_engine_events(arrays=arrays, panel_dir=panel_dir, dividends=dividends)
        rules = load_krx_market_rules(_resolve_rules_path())

        inputs_base = {
            "market_panel": str(args.panel_dataset_id),
            "scope": scope.scope_id,
            "scope_hash": scope.content_hash,
            "dividends": str(dividends_id) if dividends_id is not None else "none",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        for capital in capitals:
            for policy in (DelistPolicy.LAST_CLOSE, DelistPolicy.ZERO):
                strategy = STRATEGIES[strategy_name](**strategy_params)
                config = EngineConfig(
                    initial_cash=capital,
                    execution=execution,
                    costs=costs,
                    halted_exit_policy=policy,
                    cash_buffer=cash_buffer,
                )
                result = run_backtest(
                    arrays=arrays,
                    events=events,
                    strategy=strategy,
                    config=config,
                    deposits=deposits,
                    asof_tables={},
                    rules=rules,
                    start=start,
                    end=end,
                )
                summary = summarize(result=result, arrays=arrays, sessions_per_year=252)
                run_path = write_run(
                    run_root=run_root,
                    result=result,
                    summary=summary,
                    inputs=inputs_base,
                    config=config,
                    strategy=strategy,
                    deposits=deposits,
                )
                sys.stdout.write(
                    json.dumps(
                        {
                            "run_id": run_path.name,
                            "capital": capital,
                            "policy": policy.value,
                            "log_growth_annualized": summary.log_growth_annualized,
                            "final_nav": summary.final_nav,
                            "price_return_only": summary.price_return_only,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    except (ValueError, OSError) as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}, sort_keys=True) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
