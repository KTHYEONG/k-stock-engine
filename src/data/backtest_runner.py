"""Executable Champion backtest with replayable result artifact."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC
from pathlib import Path
from typing import Any

from src.core.ledger import LedgerNav
from src.data.pipeline import BacktestDataArtifact
from src.engine.backtest import BacktestConfig, BacktestResult, BacktestSession
from src.engine.decision import StrategyDecisionPort


def compute_backtest_performance(daily_nav: tuple[LedgerNav, ...]) -> dict[str, float]:
    """Calculate CAGR, MDD, Sharpe ratio, and return metrics from daily NAV series."""
    if not daily_nav:
        return {"initial_nav": 0.0, "final_nav": 0.0, "total_return": 0.0, "cagr": 0.0, "mdd": 0.0, "sharpe": 0.0}
    initial = float(daily_nav[0].nav)
    final = float(daily_nav[-1].nav)
    total_return = (final - initial) / initial if initial > 0 else 0.0

    peak = initial
    max_dd = 0.0
    for entry in daily_nav:
        val = float(entry.nav)
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    n = len(daily_nav)
    if n > 1:
        returns = []
        for i in range(1, n):
            prev = float(daily_nav[i - 1].nav)
            curr = float(daily_nav[i].nav)
            r = (curr - prev) / prev if prev > 0 else 0.0
            returns.append(r)
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns) if len(returns) > 1 else 0.0
        std = math.sqrt(var)
        sharpe = (mean_r / std * math.sqrt(252.0)) if std > 1e-12 else 0.0
        years = (n - 1) / 252.0
        cagr = ((final / initial) ** (1.0 / years) - 1.0) if years > 0 and final > 0 and initial > 0 else 0.0
    else:
        sharpe = 0.0
        cagr = 0.0

    return {
        "initial_nav": round(initial, 2),
        "final_nav": round(final, 2),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "mdd": round(max_dd, 6),
        "sharpe": round(sharpe, 4),
    }


def verify_accounting_identity(daily_nav: tuple[LedgerNav, ...], tolerance: float = 1e-4) -> bool:
    """Verify accounting identity (NAV == settled_cash + unsettled_cash + marked_value) for all sessions."""
    for entry in daily_nav:
        components = float(entry.settled_cash) + float(entry.unsettled_cash) + float(entry.marked_value)
        if abs(float(entry.nav) - components) > tolerance:
            return False
    return True


def run_champion_backtest(
    *,
    artifact: BacktestDataArtifact,
    sessions: tuple[BacktestSession, ...],
    config: BacktestConfig,
    strategy: StrategyDecisionPort,
    artifact_root: Path,
) -> BacktestResult:
    from src.engine.runner import run_backtest

    if not sessions:
        raise ValueError("sessions must be non-empty")
    if not artifact.content_hash:
        raise ValueError("artifact content_hash must be non-empty")
    result = run_backtest(config, sessions, strategy)
    digest = hashlib.sha256()
    for part in (
        artifact.content_hash,
        artifact.universe_hash,
        artifact.qvef_hash,
        artifact.champion_scores_hash,
        config.ledger_id,
        config.scenario.value,
        str(len(sessions)),
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()
    out_dir = Path(artifact_root) / "backtests" / content_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    perf = compute_backtest_performance(result.daily_nav)
    accounting_ok = verify_accounting_identity(result.daily_nav)
    payload = {
        "content_hash": content_hash,
        "artifact_content_hash": artifact.content_hash,
        "universe_hash": artifact.universe_hash,
        "qvef_hash": artifact.qvef_hash,
        "champion_scores_hash": artifact.champion_scores_hash,
        "benchmark_cap_hash": artifact.benchmark_cap_hash,
        "benchmark_equal_hash": artifact.benchmark_equal_hash,
        "silver_report_hash": artifact.silver_report_hash,
        "ledger_id": config.ledger_id,
        "scenario": config.scenario.value,
        "session_count": len(sessions),
        "fill_count": len(result.fills),
        "reject_count": len(result.rejects),
        "nav_points": len(result.daily_nav),
        "performance": perf,
        "accounting_reconciled": accounting_ok,
        "generated_at": __import__("datetime").datetime.now(UTC).isoformat(),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_managed_backtest(
    *,
    sessions: tuple[BacktestSession, ...],
    config: BacktestConfig,
    strategy: StrategyDecisionPort,
    artifact_root: Path,
    dataset_hash: str = "custom",
    manifest_hash: str | None = None,
    smoke_symbol: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[BacktestResult, dict[str, Any]]:
    """Execute backtest and record deterministic run manifest artifact with performance and accounting proof."""
    from src.engine.runner import run_backtest

    if not sessions:
        raise ValueError("sessions must be non-empty")

    result = run_backtest(config, sessions, strategy)

    digest = hashlib.sha256()
    for part in (
        dataset_hash,
        manifest_hash or "none",
        config.ledger_id,
        config.scenario.value,
        str(config.initial_cash),
        str(smoke_symbol or "full_universe"),
        str(len(sessions)),
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    content_hash = digest.hexdigest()

    out_dir = Path(artifact_root) / "backtests" / content_hash
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = compute_backtest_performance(result.daily_nav)
    accounting_ok = verify_accounting_identity(result.daily_nav)

    fills_summary: list[dict[str, Any]] = []
    for i, f in enumerate(result.fills):
        t_time = getattr(f, "trade_time", None)
        s_time = getattr(f, "settlement_time", None)
        fills_summary.append(
            {
                "fill_id": getattr(f, "fill_id", str(i)),
                "instrument_id": getattr(f, "instrument_id", ""),
                "side": getattr(getattr(f, "side", None), "value", str(getattr(f, "side", ""))),
                "quantity": getattr(f, "quantity", 0),
                "price": getattr(f, "price", 0.0),
                "commission": getattr(f, "commission", 0.0),
                "tax": getattr(f, "tax", 0.0),
                "slippage_cost": getattr(f, "slippage_cost", 0.0),
                "trade_time": t_time.isoformat() if t_time is not None else None,
                "settlement_time": s_time.isoformat() if s_time is not None else None,
            }
        )

    payload: dict[str, Any] = {
        "content_hash": content_hash,
        "dataset_hash": dataset_hash,
        "manifest_hash": manifest_hash,
        "smoke_symbol": smoke_symbol,
        "ledger_id": config.ledger_id,
        "scenario": config.scenario.value,
        "initial_cash": config.initial_cash,
        "session_count": len(sessions),
        "fill_count": len(result.fills),
        "reject_count": len(result.rejects),
        "nav_points": len(result.daily_nav),
        "performance": perf,
        "accounting_reconciled": accounting_ok,
        "fills": fills_summary,
        "metadata": extra_metadata or {},
        "generated_at": __import__("datetime").datetime.now(UTC).isoformat(),
    }

    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result, payload

