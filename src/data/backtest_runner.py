"""Executable Champion backtest with replayable result artifact."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from src.core.ledger import LedgerNav
from src.data.backtest_run_manifest import BacktestRunManifest
from src.data.gold import GOLD_RELEASE_FILE, GOLD_RELEASES_DIR
from src.data.pipeline import BacktestDataArtifact
from src.data.runtime import DataRuntime
from src.data.schemas import PITDataError
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
    from src.data.research_period import summarize_research_segments

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
        "research_segments": summarize_research_segments(result.daily_nav),
        "accounting_reconciled": accounting_ok,
        "fills": fills_summary,
        "metadata": extra_metadata or {},
        "generated_at": __import__("datetime").datetime.now(UTC).isoformat(),
    }

    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return result, payload


@dataclass(frozen=True, slots=True)
class StaticTransformRow:
    """One dated observation a static transform proposes to fit on."""

    as_of: date
    value: float


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    """One manifest-bound universe row offered to the strategy for a session."""

    instrument_id: str
    close: float | None
    eligible: bool
    verified: bool
    common: bool


@dataclass(frozen=True, slots=True)
class SwingSignal:
    """Target weight requested by the strategy for one instrument."""

    instrument_id: str
    target_weight: float


class DailySwingStrategy(Protocol):
    """Daily-bar swing strategy evaluated only on manifest-bound inputs."""

    @property
    def strategy_id(self) -> str: ...
    def propose_fit_rows(self) -> Sequence[StaticTransformRow]: ...
    def apply_fit(self, rows: Sequence[StaticTransformRow]) -> None: ...
    def decide(self, *, session: date, candidates: Sequence[UniverseCandidate]) -> Sequence[SwingSignal]: ...


class DailyExecutionModel(Protocol):
    """Next-session execution convention with an explicit transaction-cost policy."""

    @property
    def commission_rate(self) -> float: ...
    @property
    def tax_rate(self) -> float: ...
    def next_session(self, current: date, sessions: Sequence[date]) -> date | None: ...


@dataclass
class EqualWeightSwingStrategy:
    """Minimal deterministic strategy for the scope-bound backtest command."""

    strategy_id: str = "equal-weight"
    target_weight: float = 1.0
    max_positions: int = 5
    fit_rows: tuple[StaticTransformRow, ...] = ()

    def propose_fit_rows(self) -> Sequence[StaticTransformRow]:
        return self.fit_rows

    def apply_fit(self, rows: Sequence[StaticTransformRow]) -> None:
        self.fit_rows = tuple(rows)

    def decide(self, *, session: date, candidates: Sequence[UniverseCandidate]) -> Sequence[SwingSignal]:
        ranked = sorted({item.instrument_id for item in candidates})
        return tuple(SwingSignal(instrument_id=iid, target_weight=self.target_weight) for iid in ranked[: max(0, self.max_positions)])


@dataclass(frozen=True, slots=True)
class NextSessionExecutionModel:
    """Fill every signal at the next session close with explicit cost rates."""

    commission_rate: float
    tax_rate: float

    def next_session(self, current: date, sessions: Sequence[date]) -> date | None:
        ordered = sorted(set(sessions))
        if current not in ordered:
            raise PITDataError(f"execution session {current.isoformat()} is not a trade session")
        position = ordered.index(current)
        return ordered[position + 1] if position + 1 < len(ordered) else None


@dataclass(frozen=True, slots=True)
class ScopeBoundBacktestResult:
    """Result whose inputs and evaluation period are recoverable from its manifest."""

    manifest_hash: str
    result_path: Path
    metrics: Mapping[str, float]


_INITIAL_CASH = 1_000_000.0


def _as_day(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _load_scope_bars(*, silver_root: Path, dataset_id: str) -> dict[date, dict[str, float]]:
    dataset_dir = Path(silver_root) / "daily_market" / dataset_id
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise PITDataError("manifest-bound daily_market bars are missing")
    frame = pl.concat([pl.read_parquet(path) for path in files], how="diagonal_relaxed")
    if not {"session", "instrument_id", "close"}.issubset(frame.columns):
        raise PITDataError("manifest-bound daily_market bars have invalid schema")
    bars: dict[date, dict[str, float]] = {}
    for row in frame.to_dicts():
        instrument = str(row.get("instrument_id") or "")
        if not instrument:
            continue
        try:
            close = float(str(row.get("close")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(close) or close <= 0:
            raise PITDataError(f"manifest-bound bar has non-positive close for {instrument!r}")
        bars.setdefault(_as_day(row.get("session")), {})[instrument] = close
    return bars


def _load_scope_universe(*, release_dir: Path) -> list[dict[str, Any]]:
    universe_path = release_dir / "universe.parquet"
    if not universe_path.is_file():
        raise PITDataError("manifest-bound gold universe is missing")
    frame = pl.read_parquet(universe_path)
    if not {"session", "instrument_id", "eligible", "verified", "common"}.issubset(frame.columns):
        raise PITDataError("manifest-bound gold universe has invalid schema")
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        instrument = str(row.get("instrument_id") or "")
        if not instrument:
            continue
        rows.append(
            {
                "session": _as_day(row.get("session")),
                "instrument_id": instrument,
                "eligible": bool(row.get("eligible")),
                "verified": bool(row.get("verified")),
                "common": bool(row.get("common")),
            }
        )
    return rows


def _check_fit_rows(*, strategy: DailySwingStrategy, scope: Any) -> None:
    rows = strategy.propose_fit_rows()
    for row in rows:
        if row.as_of < scope.development_start or row.as_of > scope.development_end:
            raise PITDataError("static fit consumes rows outside the development segment")
    strategy.apply_fit(rows)


def _tradable(candidate: UniverseCandidate) -> str | None:
    if not candidate.common:
        return "preferred"
    if not candidate.verified:
        return "unverified"
    if not candidate.eligible:
        return "ineligible"
    if candidate.close is None:
        return "no_bar"
    return None


def run_scope_bound_backtest(
    *,
    runtime: DataRuntime,
    manifest: BacktestRunManifest,
    strategy: DailySwingStrategy,
    execution_model: DailyExecutionModel,
) -> ScopeBoundBacktestResult:
    """Run exactly the manifest segment using only manifest-bound Silver and Gold releases."""
    scope = runtime.scope
    if manifest.scope_hash != scope.content_hash:
        raise PITDataError("backtest manifest scope hash does not match the active scope")
    if strategy.strategy_id != manifest.strategy_id:
        raise PITDataError("backtest strategy does not match the manifest")
    run_dir = runtime.workspace.runs_root / "backtests" / manifest.content_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and existing.get("manifest_hash") == manifest.content_hash:
            metrics = existing.get("metrics")
            if isinstance(metrics, dict):
                return ScopeBoundBacktestResult(
                    manifest_hash=manifest.content_hash,
                    result_path=result_path,
                    metrics={str(key): float(value) for key, value in metrics.items()},
                )
    _check_fit_rows(strategy=strategy, scope=scope)
    bars_id = manifest.silver_dataset_ids.get("daily_market")
    if not bars_id:
        raise PITDataError("backtest manifest has no daily_market dataset binding")
    bars = _load_scope_bars(silver_root=runtime.workspace.silver_root, dataset_id=bars_id)
    release_dir = runtime.workspace.gold_root / GOLD_RELEASES_DIR / manifest.gold_dataset_id
    if not (release_dir / GOLD_RELEASE_FILE).is_file():
        raise PITDataError("manifest-bound gold release is missing")
    universe = _load_scope_universe(release_dir=release_dir)
    sessions = sorted({row["session"] for row in universe if manifest.period_start <= row["session"] <= manifest.period_end})
    if not sessions:
        raise PITDataError("no manifest-bound universe sessions in the manifest period")
    if any(session not in bars for session in sessions):
        raise PITDataError("manifest-bound bars miss a required completed session")
    by_session: dict[date, list[dict[str, Any]]] = {}
    for row in universe:
        if manifest.period_start <= row["session"] <= manifest.period_end:
            by_session.setdefault(row["session"], []).append(row)
    cash = _INITIAL_CASH
    positions: dict[str, float] = {}
    total_cost = 0.0
    trade_count = 0
    exclusions = {"unverified": 0, "ineligible": 0, "preferred": 0, "no_bar": 0, "no_execution": 0, "invalid_weight": 0}
    ledger_tmp = run_dir / "ledger.tmp.json"
    fills: list[dict[str, Any]] = []
    for session in sessions:
        day_bars = bars.get(session, {})
        candidates = tuple(
            UniverseCandidate(
                instrument_id=row["instrument_id"],
                close=day_bars.get(row["instrument_id"]),
                eligible=row["eligible"],
                verified=row["verified"],
                common=row["common"],
            )
            for row in by_session.get(session, [])
        )
        for signal in strategy.decide(session=session, candidates=candidates):
            if not math.isfinite(signal.target_weight) or signal.target_weight <= 0:
                exclusions["invalid_weight"] += 1
                continue
            execution_day = execution_model.next_session(session, sessions)
            if execution_day is None:
                exclusions["no_execution"] += 1
                continue
            wanted = next((item for item in candidates if item.instrument_id == signal.instrument_id), None)
            cause = _tradable(wanted) if wanted is not None else "no_bar"
            if cause is not None:
                exclusions[cause] += 1
                continue
            assert wanted is not None
            assert wanted.close is not None
            price = float(bars[execution_day][signal.instrument_id]) if signal.instrument_id in bars.get(execution_day, {}) else None
            if price is None:
                exclusions["no_bar"] += 1
                continue
            nav = cash + sum(shares * day_bars.get(iid, 0.0) for iid, shares in positions.items())
            order_value = signal.target_weight * nav
            cost_rate = execution_model.commission_rate
            shares = order_value / price
            if order_value * (1.0 + cost_rate) > cash:
                shares = cash / (price * (1.0 + cost_rate))
            fill_value = shares * price
            cost = fill_value * cost_rate
            cash -= fill_value + cost
            total_cost += cost
            positions[signal.instrument_id] = positions.get(signal.instrument_id, 0.0) + shares
            trade_count += 1
            fills.append({"session": execution_day.isoformat(), "instrument_id": signal.instrument_id, "shares": shares, "price": price})
    ledger_tmp.write_text(json.dumps({"fills": fills, "cash": cash}, indent=2, sort_keys=True), encoding="utf-8")
    last_bars = bars[sessions[-1]]
    final_nav = cash + sum(shares * last_bars.get(iid, 0.0) for iid, shares in positions.items())
    metrics = {
        "initial_nav": float(_INITIAL_CASH),
        "final_nav": float(final_nav),
        "total_return": float((final_nav - _INITIAL_CASH) / _INITIAL_CASH),
        "trade_count": float(trade_count),
    }
    payload = {
        "manifest_hash": manifest.content_hash,
        "segment": manifest.segment,
        "period_start": manifest.period_start.isoformat(),
        "period_end": manifest.period_end.isoformat(),
        "silver_dataset_ids": dict(sorted(manifest.silver_dataset_ids.items())),
        "gold_dataset_id": manifest.gold_dataset_id,
        "strategy_id": manifest.strategy_id,
        "strategy_policy_hash": manifest.strategy_policy_hash,
        "execution_policy_hash": manifest.execution_policy_hash,
        "universe_policy_hash": manifest.universe_policy_hash,
        "costs": {
            "commission_rate": float(execution_model.commission_rate),
            "tax_rate": float(execution_model.tax_rate),
            "total_cost": float(total_cost),
        },
        "eligible_session_count": len(sessions),
        "exclusions": dict(exclusions),
        "fills": fills,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
    }
    ledger_tmp.unlink(missing_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return ScopeBoundBacktestResult(manifest_hash=manifest.content_hash, result_path=result_path, metrics=metrics)

