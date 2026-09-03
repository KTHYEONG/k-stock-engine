"""Executable Champion backtest with replayable result artifact."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC
from pathlib import Path

from src.data.pipeline import BacktestDataArtifact
from src.engine.backtest import BacktestConfig, BacktestResult, BacktestSession
from src.engine.decision import StrategyDecisionPort


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
    nav_tail = result.daily_nav[-1] if result.daily_nav else None
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
        "final_nav": float(nav_tail.nav) if nav_tail is not None else 0.0,
        "generated_at": __import__("datetime").datetime.now(UTC).isoformat(),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result
