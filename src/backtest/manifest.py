"""Run persistence with reproducible manifests and a trial registry."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from src.backtest.engine import BacktestResult, EngineConfig
from src.backtest.metrics import PerformanceSummary
from src.backtest.strategy import Strategy

_NAV_SCHEMA: dict[str, Any] = {
    "session_idx": pl.Int64,
    "cash": pl.Int64,
    "dividend_receivable": pl.Int64,
    "market_value": pl.Int64,
    "nav": pl.Int64,
    "external_flow": pl.Int64,
}
_FILLS_SCHEMA: dict[str, Any] = {
    "decision_session_idx": pl.Int64,
    "instrument_idx": pl.Int64,
    "side": pl.String,
    "quantity": pl.Int64,
    "price": pl.Int64,
}
_REJECTS_SCHEMA: dict[str, Any] = {
    "decision_session_idx": pl.Int64,
    "instrument_idx": pl.Int64,
    "side": pl.String,
    "quantity": pl.Int64,
    "reason": pl.String,
}
_JOURNAL_SCHEMA: dict[str, Any] = {
    "session_idx": pl.Int64,
    "kind": pl.String,
    "instrument_idx": pl.Int64,
    "cash_delta": pl.Int64,
    "quantity_delta": pl.Int64,
}


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _frame_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _code_version() -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _engine_config_payload(config: EngineConfig) -> dict[str, Any]:
    return {
        "initial_cash": config.initial_cash,
        "execution": {
            "scenario": config.execution.scenario.value,
            "max_participation": config.execution.max_participation,
            "carry_unfilled": config.execution.carry_unfilled,
        },
        "costs": {
            "commission_rate": str(config.costs.commission_rate),
            "impact_k": config.costs.impact_k,
        },
        "halted_exit_policy": config.halted_exit_policy.value,
        "cash_buffer": config.cash_buffer,
    }


def _summary_payload(summary: PerformanceSummary) -> dict[str, Any]:
    return {
        "log_growth_annualized": summary.log_growth_annualized,
        "log_growth_by_year": {
            str(year): value for year, value in sorted(summary.log_growth_by_year.items())
        },
        "twr_total": summary.twr_total,
        "mwr_annualized": summary.mwr_annualized,
        "final_nav": summary.final_nav,
        "total_external_flow": summary.total_external_flow,
        "turnover_annualized": summary.turnover_annualized,
        "cost_drag_annualized": summary.cost_drag_annualized,
        "reject_counts": dict(sorted(summary.reject_counts.items())),
        "max_participation": summary.max_participation,
        "price_return_only": summary.price_return_only,
    }


def write_run(
    *,
    run_root: Path,
    result: BacktestResult,
    summary: PerformanceSummary,
    inputs: Mapping[str, str],
    config: EngineConfig,
    strategy: Strategy,
    deposits: Mapping[date, int],
) -> Path:
    """Persist one run under ``run_root/<run_id>/`` with a manifest sufficient to reproduce it.

    ``run_id`` hashes the input dataset ids, engine config, strategy name and
    params, deposit schedule, and code version (git HEAD); rewriting an
    identical run is idempotent, and a differing run with the same id fails.
    """
    run_root = Path(run_root)
    strategy_params = dict(strategy.params())
    params_hash = hashlib.sha256(_canonical(strategy_params).encode("utf-8")).hexdigest()
    config_payload = _engine_config_payload(config)
    config_hash = hashlib.sha256(_canonical(config_payload).encode("utf-8")).hexdigest()
    body: dict[str, Any] = {
        "inputs": dict(sorted(inputs.items())),
        "strategy": {
            "name": strategy.name,
            "params": strategy_params,
            "params_hash": params_hash,
        },
        "config": config_payload,
        "config_hash": config_hash,
        "deposits": {day.isoformat(): amount for day, amount in sorted(deposits.items())},
        "code_version": _code_version(),
        "dividends_integrated": result.dividends_integrated,
        "ledger_hash": result.ledger_hash,
    }
    run_id = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()[:16]
    target = run_root / run_id
    nav_frame = pl.DataFrame(
        [
            {
                "session_idx": record.session_idx,
                "cash": record.cash,
                "dividend_receivable": record.dividend_receivable,
                "market_value": record.market_value,
                "nav": record.nav,
                "external_flow": record.external_flow,
            }
            for record in result.nav
        ],
        schema=_NAV_SCHEMA,
    )
    fills_frame = pl.DataFrame(
        [
            {
                "decision_session_idx": fill.order.decision_session_idx,
                "instrument_idx": fill.order.instrument_idx,
                "side": fill.order.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
            }
            for fill in result.fills
        ],
        schema=_FILLS_SCHEMA,
    )
    rejects_frame = pl.DataFrame(
        [
            {
                "decision_session_idx": reject.order.decision_session_idx,
                "instrument_idx": reject.order.instrument_idx,
                "side": reject.order.side.value,
                "quantity": reject.order.quantity,
                "reason": reject.reason,
            }
            for reject in result.rejects
        ],
        schema=_REJECTS_SCHEMA,
    )
    journal_frame = pl.DataFrame(
        [
            {
                "session_idx": entry.session_idx,
                "kind": entry.kind.value,
                "instrument_idx": entry.instrument_idx,
                "cash_delta": entry.cash_delta,
                "quantity_delta": entry.quantity_delta,
            }
            for entry in result.journal
        ],
        schema=_JOURNAL_SCHEMA,
    )
    summary_payload = _summary_payload(summary)
    files = {
        "fills.parquet": _frame_bytes(fills_frame),
        "journal.parquet": _frame_bytes(journal_frame),
        "nav.parquet": _frame_bytes(nav_frame),
        "rejects.parquet": _frame_bytes(rejects_frame),
        "summary.json": (json.dumps(summary_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    manifest: dict[str, Any] = {
        **body,
        "run_id": run_id,
        "trial": {
            "strategy": strategy.name,
            "params_hash": params_hash,
            "config_hash": config_hash,
        },
        "files": {
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists():
        try:
            current = (target / "manifest.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"existing run is unreadable: {target}") from exc
        if current != manifest_text:
            raise ValueError(f"run id collision with different content: {target}")
        return target
    staging = run_root / f".{run_id}.tmp"
    staging.mkdir(parents=True)
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)
        (staging / "manifest.json").write_text(manifest_text, encoding="utf-8")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    with (run_root / "trials.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_canonical(manifest["trial"]) + "\n")
    return target
