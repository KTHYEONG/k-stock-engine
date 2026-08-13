"""Canonical execution-matched selection/replay kernel (v5 architecture).

The v5 research redesign replaces the equal-weight proxy screen with a single
candidate-invariant prepared base-ledger replay kernel. The kernel owns only
candidate-invariant objects (prepared market rows, allocation history, decision
schedule, cost schedule, and the frozen ``StockRiskPolicy``); a candidate
supplies one aligned score overlay and, when enabled, one precomputed causal
calibration ledger. ``run_base`` executes exactly the decision provider,
allocation constructor, order planner, fill calculation, settlement state, and
metrics used by the base leg of ``StockBacktester.run_prepared`` so selection
and promotion optimize the same feasible, costed, stateful portfolio outcome.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np
import polars as pl

from src.core.costs import CostSchedule
from src.core.datasets import DatasetManifest
from src.core.instruments import Instrument
from src.core.portfolio import PortfolioSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.calibration_schedule import (
    CausalCalibrationSchedule,
    SessionClusterCalibrationSchedule,
)
from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.trading.portfolio_constructor import (
    StockRiskPolicy,
    construct_target_allocations_prepared,
)

logger = logging.getLogger("stocks.workflows.execution_matched_replay")

KERNEL_PARITY_VERSION = "execution-matched-v1"

INNER_SELECTION_BASE_ONLY = "INNER_SELECTION_BASE_ONLY"
FINAL_PROMOTION_BASE_AND_STRESS = "FINAL_PROMOTION_BASE_AND_STRESS"

_BOOTSTRAP_BYTES_PER_ROW = 24


def _bootstrap_workspace_bytes(n_bootstrap: int, history_rows: int) -> int:
    return n_bootstrap * history_rows * _BOOTSTRAP_BYTES_PER_ROW


def _benchmark_return_series(panel: pl.DataFrame) -> list[float]:
    if "session" not in panel.columns or "close" not in panel.columns:
        return []
    with_return = panel.sort("session").with_columns(
        (pl.col("close").log().diff().over("instrument_id")).alias("_logret")
    )
    daily = (
        with_return.group_by("session")
        .agg(pl.col("_logret").mean().alias("bench"))
        .sort("session")
    )
    returns: list[float] = []
    for row in daily.to_dicts():
        value = row["bench"]
        if value is not None:
            returns.append(float(value))
    return returns


def _strategy_return_series(ledger: list[Any]) -> list[float]:
    returns: list[float] = []
    for i in range(1, len(ledger)):
        prev = float(ledger[i - 1].equity)
        current = float(ledger[i].equity)
        if prev > 0:
            returns.append(math.log(current / prev) if current > 0 else 0.0)
        else:
            returns.append(0.0)
    return returns


def _aligned_excess(
    strategy_returns: list[float],
    benchmark_returns: list[float],
) -> list[float]:
    common = min(len(strategy_returns), len(benchmark_returns))
    return [
        strategy_returns[i] - benchmark_returns[i] for i in range(common)
    ]


@dataclass(frozen=True, slots=True)
class ExecutionMatchedEvidence:
    """Compact JSON-safe base replay evidence (no raw ledger/score arrays).

    The evidence carries the canonical base ledger and metrics for parity
    checks plus scalar telemetry; raw ledgers, score arrays, and return arrays
    are never persisted into metrics. ``to_json_safe`` emits only finite
    scalars and small maps.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    strategy_returns: list[float] = field(default_factory=list)
    benchmark_returns: list[float] = field(default_factory=list)
    excess_returns: list[float] = field(default_factory=list)
    decision_boundaries: list[int] = field(default_factory=list)
    planned_cycles: int = 0
    attempted_orders: int = 0
    filled_orders: int = 0
    no_trade_reason_counts: dict[str, int] = field(default_factory=dict)
    unfilled_order_reason_counts: dict[str, int] = field(default_factory=dict)
    prepared_decision_count: int = 0
    base_total_return: float = 0.0
    benchmark_total_return: float = 0.0
    holding_horizon_sessions: int = 5
    final_value: float = 0.0
    stress_final_value: float | None = None
    stress_metrics: dict[str, float] | None = None
    stress_total_return: float | None = None
    calibration_evidence: dict[str, object] = field(default_factory=dict)
    replay_resource: dict[str, object] = field(default_factory=dict)
    compounding_overlay: dict[str, object] = field(default_factory=dict)
    replay_stage_seconds: dict[str, float] = field(default_factory=dict)
    kernel_parity_version: str = KERNEL_PARITY_VERSION
    replay_mode: str | None = None
    # In-memory parity reference; never persisted to metrics.
    ledger: tuple[Any, ...] = ()
    trades: tuple[Any, ...] = ()

    def to_json_safe(self) -> dict[str, object]:
        return {
            "kernel_parity_version": self.kernel_parity_version,
            "metrics": dict(sorted(self.metrics.items())),
            "planned_cycles": int(self.planned_cycles),
            "attempted_orders": int(self.attempted_orders),
            "filled_orders": int(self.filled_orders),
            "no_trade_reason_counts": dict(
                sorted(self.no_trade_reason_counts.items())
            ),
            "unfilled_order_reason_counts": dict(
                sorted(self.unfilled_order_reason_counts.items())
            ),
            "prepared_decision_count": int(self.prepared_decision_count),
            "base_total_return": round(self.base_total_return, 8),
            "benchmark_total_return": round(self.benchmark_total_return, 8),
            "holding_horizon_sessions": int(self.holding_horizon_sessions),
            "stress_total_return": (
                round(self.stress_total_return, 8)
                if self.stress_total_return is not None
                else None
            ),
            "replay_mode": self.replay_mode,
        }


class ExecutionMatchedReplayKernel:
    """Candidate-invariant prepared base-ledger replay kernel.

    Owns the prepared market rows, allocation history, decision schedule,
    initial portfolio, cost schedule, instrument map, and frozen risk policy
    for one contiguous proxy route. A candidate contributes only an aligned
    ``float64`` execution overlay plus a precomputed causal calibration ledger.
    ``run_base`` reproduces the canonical base leg of
    ``StockBacktester.run_prepared`` and returns compact
    :class:`ExecutionMatchedEvidence`.
    """

    def __init__(
        self,
        *,
        panel: pl.DataFrame,
        prepared_route: Any,
        instruments: Mapping[str, Instrument],
        policy: StockRiskPolicy,
        request: Any,
        dataset_manifest: DatasetManifest,
        registry: ModelArtifactRegistry,
        base_schedule: CostSchedule,
        stress_schedule: CostSchedule,
        holding_horizon_sessions: int,
        label_column: str,
        label_available_column: str,
        replay_guard: Any | None = None,
    ) -> None:
        self._panel = panel
        self._prepared_route = prepared_route
        self._instruments = instruments
        self._policy = policy
        self._request = request
        self._dataset_manifest = dataset_manifest
        self._registry = registry
        self._base_schedule = base_schedule
        self._stress_schedule = stress_schedule
        self._holding_horizon_sessions = int(holding_horizon_sessions)
        self._label_column = label_column
        self._label_available_column = label_available_column
        self._replay_guard = replay_guard
        self._cache_bytes = int(prepared_route.cache_bytes)
        first, last = prepared_route.sessions[0], prepared_route.sessions[-1]
        self._replay_frame = panel.filter(
            (pl.col("session") >= first) & (pl.col("session") <= last)
        )
        self._benchmark_returns = _benchmark_return_series(self._replay_frame)

    @classmethod
    def build(
        cls,
        panel: pl.DataFrame,
        oos_sessions: Sequence[datetime],
        request: Any,
        route: Any,
        *,
        dataset_manifest: DatasetManifest,
        registry: ModelArtifactRegistry,
        base_schedule: CostSchedule,
        stress_schedule: CostSchedule,
        guard: Any | None = None,
    ) -> ExecutionMatchedReplayKernel:
        """Build the candidate-invariant kernel for one contiguous OOS interval.

        The prepared route owns the bounded execution market, the instrument
        map, and the frozen risk policy; no redundant full-panel replay static
        context is constructed. Benchmark returns are computed once from the
        bounded route interval and cached for every candidate replay.
        """
        from src.stocks.workflows.train_model import PreparedSelectionRoute

        prepared_route = PreparedSelectionRoute.build(
            panel, oos_sessions, request, route, guard=guard
        )
        return cls(
            panel=panel,
            prepared_route=prepared_route,
            instruments=prepared_route.instruments,
            policy=prepared_route.policy,
            request=request,
            dataset_manifest=dataset_manifest,
            registry=registry,
            base_schedule=base_schedule,
            stress_schedule=stress_schedule,
            holding_horizon_sessions=int(route.horizon),
            label_column=route.label_column,
            label_available_column=route.label_available_column,
            replay_guard=guard,
        )

    @property
    def prepared_route(self) -> Any:
        return self._prepared_route

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def label_column(self) -> str:
        return self._label_column

    @property
    def label_available_column(self) -> str:
        return self._label_available_column

    def run_base(
        self,
        oos_scored: pl.DataFrame,
        calibration_ledger: pl.DataFrame | None = None,
        *,
        replay_mode: str = INNER_SELECTION_BASE_ONLY,
        replay_guard: Any | None = None,
    ) -> ExecutionMatchedEvidence:
        """Run the canonical base replay for one candidate score overlay."""
        from src.stocks.backtesting.engine import (
            BacktestRequest,
            PreparedReplayDecision,
            StockBacktester,
        )
        from src.stocks.workflows.trading_cycle import (
            CycleStatus,
            TradingCycleRequest,
            TradingCycleResult,
            _build_intents,
        )

        guard = replay_guard if replay_guard is not None else self._replay_guard
        route = self._prepared_route
        policy = self._policy
        request = self._request
        policy.compounding_evidence.clear()

        stage_seconds: dict[str, float] = {
            "overlay": 0.0,
            "window_frame": 0.0,
            "allocator": 0.0,
            "calibration_state": 0.0,
        }
        overlay_started = time.perf_counter()
        overlay, allocation_overlay = route.scatter_overlays(oos_scored)
        stage_seconds["overlay"] += time.perf_counter() - overlay_started
        if guard is not None:
            guard.admit(int(oos_scored.estimated_size()), stage="candidate_overlay")
        sessions = route.sessions
        decision_indices = route.decision_indices
        decision_times = tuple(sessions[i] for i in decision_indices)
        start_time = sessions[0]
        end_time = sessions[-1]

        replay_frame = self._replay_frame
        if guard is not None:
            guard.admit(int(replay_frame.estimated_size()), stage="replay_adtv")
        market = route.market

        calibrator = (
            CausalAlphaCalibrator(
                bucket_count=request.calibration_bucket_count,
                min_calibration_sessions=request.min_calibration_sessions,
                seed=request.seed,
                n_bootstrap=request.n_bootstrap,
                bootstrap_alpha=request.bootstrap_alpha,
                label_column=self._label_column,
                label_available_column=self._label_available_column,
            )
            if calibration_ledger is not None and not calibration_ledger.is_empty()
            else None
        )
        calibration_schedule: CausalCalibrationSchedule | None = None
        session_cluster_schedule: SessionClusterCalibrationSchedule | None = None
        if calibrator is not None:
            assert calibration_ledger is not None
            if replay_mode == INNER_SELECTION_BASE_ONLY:
                session_cluster_schedule = SessionClusterCalibrationSchedule.build(
                    calibration_ledger,
                    decision_times,
                    calibrator,
                    self._base_schedule,
                    block_length=self._holding_horizon_sessions,
                    max_workspace_bytes=_bootstrap_workspace_bytes(
                        request.n_bootstrap, int(calibration_ledger.height)
                    ),
                )
            else:
                calibration_schedule = CausalCalibrationSchedule.build(
                    calibration_ledger,
                    decision_times,
                    calibrator,
                    self._base_schedule,
                    max_workspace_bytes=_bootstrap_workspace_bytes(
                        request.n_bootstrap, int(calibration_ledger.height)
                    ),
                )
        calibration_tracker: dict[str, object] = {}

        def _scored_no_trade(
            portfolio: PortfolioSnapshot,
            cycle_request: TradingCycleRequest,
            reason: str,
        ) -> TradingCycleResult:
            return TradingCycleResult(
                status=CycleStatus.NO_TRADE,
                cycle_id="stub",
                decision_time=cycle_request.decision_time,
                dataset_hash="d",
                artifact_id=cycle_request.artifact_id,
                account_snapshot_id=portfolio.account_snapshot_id,
                allocations=(),
                intents=(),
                selected_instruments=(),
                reasons=(reason,),
            )

        def _state_at(
            decision_time: datetime,
            visible: pl.DataFrame,
        ) -> dict[str, object] | None:
            if calibrator is None:
                return None
            assert calibration_ledger is not None
            workspace_cap: int | None = None
            active_schedule = (
                session_cluster_schedule
                if session_cluster_schedule is not None
                else calibration_schedule
            )
            if guard is not None:
                assert active_schedule is not None
                prefix_rows = active_schedule.eligible_prefix_rows(decision_time)
                if prefix_rows > 0:
                    workspace_cap = guard.bootstrap_workspace_cap(
                        history_rows=prefix_rows,
                        projected_output_bytes=int(visible.estimated_size()),
                        n_bootstrap=request.n_bootstrap,
                    )
            if active_schedule is not None:
                state = active_schedule.state_at(
                    decision_time,
                    max_bootstrap_workspace_bytes=workspace_cap,
                )
            else:
                state = calibrator.prepare_decision(
                    calibration_ledger,
                    decision_time,
                    self._base_schedule,
                    max_bootstrap_workspace_bytes=workspace_cap,
                )
            calibration_tracker["state"] = calibrator.calibration_state()
            return state

        def _prepare_decision(
            decision_time: datetime,
            execution_time: datetime,
        ) -> PreparedReplayDecision:
            if guard is not None:
                guard.record_prepared_decision()
            try:
                allocation_decision_index = route.allocation_decision_index_for(
                    decision_time
                )
                if allocation_decision_index is None:
                    return PreparedReplayDecision(
                        decision_time, execution_time, pl.DataFrame(),
                        calibration_state=None, reason="no-decision-session",
                    )
                window_started = time.perf_counter()
                visible = route.window_frame(
                    allocation_decision_index, allocation_overlay
                )
                stage_seconds["window_frame"] += time.perf_counter() - window_started
            except ValueError as exc:
                return PreparedReplayDecision(
                    decision_time, execution_time, pl.DataFrame(),
                    calibration_state=None, reason=f"constraint:{exc}",
                )
            if visible.is_empty():
                return PreparedReplayDecision(
                    decision_time, execution_time, pl.DataFrame(),
                    calibration_state=None, reason="empty-scored-cross-section",
                )
            try:
                calibration_started = time.perf_counter()
                calibration_state = _state_at(decision_time, visible)
                stage_seconds["calibration_state"] += (
                    time.perf_counter() - calibration_started
                )
            except ValueError as exc:
                return PreparedReplayDecision(
                    decision_time, execution_time, pl.DataFrame(),
                    calibration_state=None, reason=f"constraint:{exc}",
                )
            return PreparedReplayDecision(
                decision_time, execution_time, visible,
                calibration_state=calibration_state,
            )

        def _scenario_planner(
            prepared: PreparedReplayDecision,
            portfolio: PortfolioSnapshot,
            cycle_request: TradingCycleRequest,
        ) -> TradingCycleResult:
            if prepared.reason is not None:
                return _scored_no_trade(portfolio, cycle_request, prepared.reason)
            try:
                allocation_decision_index = route.allocation_decision_index_for(
                    prepared.decision_time
                )
                if allocation_decision_index is None:
                    return _scored_no_trade(
                        portfolio, cycle_request, "no-decision-session"
                    )
                allocator_started = time.perf_counter()
                allocations = construct_target_allocations_prepared(
                    route.allocation_market,
                    allocation_decision_index,
                    allocation_overlay,
                    prepared.calibration_state,
                    self._instruments,
                    portfolio,
                    policy,
                )
                stage_seconds["allocator"] += time.perf_counter() - allocator_started
            except ValueError as exc:
                return _scored_no_trade(
                    portfolio, cycle_request, f"constraint:{exc}"
                )
            if not allocations:
                return _scored_no_trade(
                    portfolio, cycle_request, "no-feasible-allocation"
                )
            intents = _build_intents(tuple(allocations), portfolio, cycle_request)
            return TradingCycleResult(
                status=CycleStatus.PLANNED,
                cycle_id="stub",
                decision_time=cycle_request.decision_time,
                dataset_hash="d",
                artifact_id=cycle_request.artifact_id,
                account_snapshot_id=portfolio.account_snapshot_id,
                allocations=tuple(allocations),
                intents=intents,
                selected_instruments=tuple(
                    sorted({a.instrument.instrument_id for a in allocations})
                ),
                reasons=("scored-plan",),
            )

        backtest_request = BacktestRequest(
            strategy_id=request.artifact_id,
            start_time=start_time,
            end_time=end_time,
            decision_session_indices=decision_indices,
            cost_schedule=self._base_schedule,
            stress_cost_schedule=self._stress_schedule,
            risk_policy=policy,
            seed=request.seed,
        )
        from src.stocks.data.costs import CostEvidence

        evidence: CostEvidence | None = getattr(request, "cost_evidence", None)
        if replay_mode == INNER_SELECTION_BASE_ONLY:
            backtester = StockBacktester(
                registry=self._registry,
                instruments=self._instruments,
                manifest=self._dataset_manifest,
                cost_schedule=self._base_schedule,
                stress_cost_schedule=None,
                cost_evidence=evidence,
                seed=request.seed,
                decision_provider=_prepare_decision,
                scenario_planner=_scenario_planner,
            )
        else:
            backtester = StockBacktester(
                registry=self._registry,
                instruments=self._instruments,
                manifest=self._dataset_manifest,
                cost_schedule=self._base_schedule,
                stress_cost_schedule=self._stress_schedule,
                cost_evidence=evidence,
                seed=request.seed,
                decision_provider=_prepare_decision,
                scenario_planner=_scenario_planner,
            )
        result = backtester.run_prepared(backtest_request, market, overlay)
        if guard is not None:
            guard.check_after(stage="replay")
            guard.record_stage("replay", 0.0)

        benchmark = self._benchmark_returns
        strategy_returns = _strategy_return_series(list(result.ledger))
        excess = _aligned_excess(strategy_returns, benchmark)
        initial_cash = request.initial_cash
        stress_total: float | None = None
        if result.stress_final_value is not None and initial_cash > 0:
            stress_total = (result.stress_final_value - initial_cash) / initial_cash
        if not benchmark or not all(math.isfinite(value) for value in benchmark):
            benchmark_total = float("nan")
        else:
            benchmark_total = float(np.expm1(np.sum(benchmark)))
        no_trade_reason_counts = dict(
            Counter(
                reason
                for cycle in backtester._last_cycles.values()
                if cycle.status is not CycleStatus.PLANNED
                for reason in cycle.reasons
            )
        )
        calibration_state = calibration_tracker.get("state")
        calibration_evidence: dict[str, object] = {}
        if calibration_state is not None and isinstance(calibration_state, dict):
            buckets = calibration_state.get("buckets") or []
            net_alphas = [
                float(row["expected_active_alpha"])
                - float(calibration_state["round_trip_cost"])
                for row in buckets
                if row.get("expected_active_alpha") is not None
            ]
            calibration_evidence = {
                "history_sessions": int(calibration_state.get("history_sessions", 0)),
                "eligible_bucket_count": len(net_alphas),
                "average_expected_net_alpha": (
                    float(np.mean(net_alphas)) if net_alphas else 0.0
                ),
                "round_trip_cost": float(calibration_state.get("round_trip_cost", 0.0)),
                "exit_cost_rate": float(calibration_state.get("exit_cost_rate", 0.0)),
                "calibration_state": calibration_state,
            }
        if session_cluster_schedule is not None:
            calibration_evidence["bootstrap_unit"] = "session_cluster"
            calibration_evidence.update(session_cluster_schedule.telemetry())
        replay_resource: dict[str, object] = {}
        if guard is not None:
            replay_resource = guard.telemetry()
            if session_cluster_schedule is not None:
                replay_resource.update(session_cluster_schedule.telemetry())
            stage_seconds_payload = dict(
                cast(dict[str, object], replay_resource.get("replay_stage_seconds") or {})
            )
            stage_seconds_payload.update(
                {
                    stage: round(seconds, 3)
                    for stage, seconds in stage_seconds.items()
                }
            )
            replay_resource["replay_stage_seconds"] = stage_seconds_payload
        from src.stocks.workflows.train_model import _compounding_overlay_summary

        compounding_overlay = _compounding_overlay_summary(
            list(policy.compounding_evidence),
            include_records=replay_mode != INNER_SELECTION_BASE_ONLY,
        )
        return ExecutionMatchedEvidence(
            metrics=result.metrics,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark,
            excess_returns=excess,
            decision_boundaries=list(decision_indices),
            planned_cycles=result.planned_cycles,
            attempted_orders=result.attempted_orders,
            filled_orders=result.filled_orders,
            no_trade_reason_counts=no_trade_reason_counts,
            unfilled_order_reason_counts=dict(result.unfilled_order_reason_counts),
            prepared_decision_count=backtester.prepared_decision_count,
            base_total_return=(
                (result.final_value - initial_cash) / initial_cash
                if initial_cash > 0
                else 0.0
            ),
            benchmark_total_return=benchmark_total,
            holding_horizon_sessions=self._holding_horizon_sessions,
            final_value=result.final_value,
            stress_final_value=result.stress_final_value,
            stress_metrics=result.stress_metrics,
            stress_total_return=stress_total,
            calibration_evidence=calibration_evidence,
            replay_resource=replay_resource,
            compounding_overlay=compounding_overlay,
            replay_mode=replay_mode,
            replay_stage_seconds={
                stage: round(seconds, 3) for stage, seconds in stage_seconds.items()
            },
            ledger=tuple(result.ledger),
            trades=tuple(result.trades),
        )
