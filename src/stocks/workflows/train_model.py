"""Stock model-training workflow: v2 snapshot -> nested walk-forward -> champion/NO_TRADE artifact.

The workflow consumes the composed labeled v2 snapshot (base OHLCV + ``feature__*``
stock_alpha_v2 columns + canonical residual labels), runs a purged/embargoed
expanding walk-forward with quarterly refits, tunes the ``LambdaRankBlendModel``
with seeded serial Optuna TPE trials on a prequential panel derived exclusively
from the first outer fold's training mask, evaluates every fold through the
event-driven ``StockBacktester`` under base and stress costs, and publishes
either an immutable champion or an immutable ``NO_TRADE`` artifact. The trial
search is bounded by an explicit or effective RSS budget and replays scored
scores with bounded point-in-time market history. Promotion is lexicographic
and fail-closed; a failing gate never relaxes parameters. Promotion remains
false until a frozen candidate passes one new 252-session forward holdout
starting after 2026-03-10.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import polars as pl
import psutil

from src.core.costs import CostSchedule, default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.calibration_schedule import (
    CausalCalibrationSchedule,
    SessionClusterCalibrationSchedule,
)
from src.stocks.research.datasets import (
    research_eligible_frame,
    validate_stock_rows_available,
)
from src.stocks.research.economic_alpha import CausalAlphaCalibrator
from src.stocks.research.features import (
    apply_v2_transforms,
    fit_v2_winsor_quantiles,
    stock_alpha_v2_allowlist,
    v2_feature_columns,
)
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.labels import (
    LABEL_AVAILABLE_COLUMN,
    RELEVANCE_COLUMN,
)
from src.stocks.research.lambdarank import (
    FitTrialOutcome,
    LambdaRankBlendModel,
    LambdaRankConfig,
    PreparedLambdaRankFold,
    adaptive_refit_rounds,
    resolve_lgb_num_threads,
)
from src.stocks.research.models import ModelManifest, StableRankComposite
from src.stocks.trading.portfolio_constructor import (
    CompoundingPolicyConfig,
    PreparedAllocationMarket,
    StockRiskPolicy,
    construct_target_allocations_prepared,
)
from src.stocks.workflows.contracts import (
    COMPUTE_PLAN_VERSION,
    SELECTION_MULTIPLICITY_VERSION,
    TrainingRequest,
)
from src.stocks.workflows.economic_selection import (
    SELECTION_POLICY_VERSION,
    ScreenFidelityPolicy,
)
from src.stocks.workflows.training_run_store import TrainingRunStore, content_hash

if TYPE_CHECKING:
    from src.core.portfolio import PortfolioSnapshot
    from src.stocks.backtesting.engine import (
        ArtifactSchedule,
        BacktestLedgerRow,
        PreparedReplayMarket,
    )

logger = logging.getLogger("stocks.workflows.train_model")

_ECONOMIC_COLUMNS = ("open", "high", "low", "close", "volume", "trading_value", "market_cap")

_REPLAY_MARKET_INDEX_REQUIRED = (
    "instrument_id",
    "session",
    "available_time",
    "open",
    "close",
    "volume",
    "trading_value",
    "sector",
    "adtv",
)
_REPLAY_MARKET_INDEX_OPTIONAL = (
    "high",
    "low",
    "limit_locked",
    "action_interval_covered",
    "feature__volatility_20d",
    "data_quality_status",
)
_BOOTSTRAP_START_BYTES = 8
_BOOTSTRAP_INDEX_BYTES = 8
_BOOTSTRAP_VALUE_BYTES = 8
_BOOTSTRAP_BYTES_PER_ROW = (
    _BOOTSTRAP_START_BYTES + _BOOTSTRAP_INDEX_BYTES + _BOOTSTRAP_VALUE_BYTES
)


class ReplayMode(StrEnum):
    """Replay evidence scope distinguishing selection from final promotion."""

    INNER_SELECTION_BASE_ONLY = "INNER_SELECTION_BASE_ONLY"
    FINAL_PROMOTION_BASE_AND_STRESS = "FINAL_PROMOTION_BASE_AND_STRESS"

_MIN_TRAIN_SESSIONS = 756
_VALIDATION_BLOCK_SESSIONS = 252
_REFIT_EVERY_SESSIONS = 63
_FORWARD_HOLDOUT_START = date(2026, 3, 10)
_FORWARD_HOLDOUT_SESSIONS = 252
_MIN_GROUP_SIZE = 20
_BYTES_PER_CELL = 4
_ALLOCATION_MULTIPLE = 3
_SCREEN_BOOSTING_ROUNDS = 800
_SCREEN_EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True, slots=True)
class PromotionRiskBudget:
    """Versioned risk budget enforced by the promotion gates."""

    min_positive_refit_fraction: float = 0.75
    bootstrap_alpha: float = 0.05
    deflated_sharpe_probability: float = 0.95
    max_benchmark_drawdown_ratio: float = 1.10


@dataclass(frozen=True, slots=True)
class CompoundingEvidence:
    """Holding-period-consistent compounding evidence for one replay.

    ``block_log_excess`` is the ordered series of complete route-length
    block log-excess wealth differences
    ``sum(log1p(strategy)) - sum(log1p(benchmark))``. ``bootstrap_lower_bound``
    is the seeded moving-block bootstrap (block length two rebalances) lower
    confidence bound of that series; ``dsr_probability`` is the Deflated Sharpe
    probability of the same series. ``complete_block_count`` and
    ``rejected_block_count`` make the fail-closed boundary visible.
    """

    block_log_excess: list[float]
    bootstrap_lower_bound: float
    dsr_probability: float
    complete_block_count: int
    rejected_block_count: int


_DEFAULT_COMPOUNDING_POLICY_ID = "default:neutral"
_DEFAULT_GROWTH_RISK_AVERSION = 1.0
_DEFAULT_TURNOVER_BUDGET = 0.20


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Event-ledger outcome used by the promotion gates."""

    ledger: tuple[object, ...] = ()
    trades: tuple[object, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    stress_metrics: dict[str, float] | None = None
    final_value: float = 0.0
    excess_returns: list[float] = field(default_factory=list)
    benchmark_returns: list[float] = field(default_factory=list)
    strategy_returns: list[float] = field(default_factory=list)
    base_total_return: float = 0.0
    stress_total_return: float | None = None
    benchmark_total_return: float = 0.0
    planned_cycles: int = 0
    attempted_orders: int = 0
    filled_orders: int = 0
    no_trade_reason_counts: dict[str, int] = field(default_factory=dict)
    unfilled_order_reason_counts: dict[str, int] = field(default_factory=dict)
    calibration_evidence: dict[str, object] = field(default_factory=dict)
    replay_mode: str | None = None
    replay_resource: dict[str, object] = field(default_factory=dict)
    prepared_decision_count: int = 0
    decision_boundaries: list[int] = field(default_factory=list)
    holding_horizon_sessions: int = 5
    compounding_overlay: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingCapacityError(RuntimeError):
    """Raised when the trial search cannot fit its allocation under the RSS budget."""

    message: str = ""

    def __str__(self) -> str:
        return self.message


class TrialResourceGuard:
    """Admit each fold's allocation against an explicit or effective RSS budget.

    The effective limit is the explicit ``request.max_rss_mib``; absent one it
    is the process memory ceiling less the memory already unavailable according
    to ``psutil.virtual_memory()``. The per-fold estimate is the conservative
    design-matrix lower bound ``rows * predictor_count * 4 * 3`` bytes and is
    intentionally not a full peak bound. Baseline, peak RSS, estimates, and
    trial/fold timings are recorded for Optuna study user attributes and the
    published artifact metrics.
    """

    def __init__(self, request: TrainingRequest, predictor_count: int) -> None:
        self._predictor_count = predictor_count
        self._limit_mib = (
            float(request.max_rss_mib)
            if request.max_rss_mib is not None
            else self._effective_limit_mib()
        )
        self.baseline_rss_mib = self._rss_mib()
        self.peak_rss_mib = self.baseline_rss_mib
        self.fold_timings: dict[str, float] = {}
        self.estimates_mib: dict[str, float] = {}

    @staticmethod
    def _rss_mib() -> float:
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)

    @staticmethod
    def _effective_limit_mib() -> float:
        vm = psutil.virtual_memory()
        available = float(vm.available)
        if available <= 0.0:
            available = float(vm.total)
        return available / (1024 * 1024)

    def estimate_mib(self, rows: int) -> float:
        """Conservative per-fold design-matrix lower bound in MiB."""
        return rows * self._predictor_count * _BYTES_PER_CELL * _ALLOCATION_MULTIPLE / (1024 * 1024)

    def admit(self, rows: int, *, extra_bytes: int = 0) -> None:
        """Reject the run before allocation when the increment cannot fit.

        ``extra_bytes`` accounts for immutable prepared-fold cache arrays held
        across candidates so a cache never silently pushes the process past the
        RSS budget.
        """
        estimate = self.estimate_mib(rows)
        if extra_bytes > 0:
            estimate += extra_bytes / (1024 * 1024)
        current = self._rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, current)
        if current + estimate > self._limit_mib:
            raise TrainingCapacityError(
                f"training capacity breach: current RSS {current:.1f} MiB plus "
                f"estimated design matrix {estimate:.1f} MiB exceeds the "
                f"{self._limit_mib:.1f} MiB limit"
            )

    def check_after(self) -> None:
        """Reject when the sampled process RSS exceeds the limit after a fold."""
        current = self._rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, current)
        if current > self._limit_mib:
            raise TrainingCapacityError(
                f"training capacity breach: measured RSS {current:.1f} MiB exceeds "
                f"the {self._limit_mib:.1f} MiB limit"
            )

    def record_fold(self, key: str, elapsed_seconds: float, estimate_mib: float) -> None:
        self.fold_timings[key] = elapsed_seconds
        self.estimates_mib[key] = estimate_mib

    def telemetry(self) -> dict[str, object]:
        return {
            "baseline_rss_mib": round(self.baseline_rss_mib, 3),
            "peak_rss_mib": round(self.peak_rss_mib, 3),
            "limit_mib": round(self._limit_mib, 3),
            "trial_fold_timings_seconds": dict(self.fold_timings),
            "estimates_mib": dict(self.estimates_mib),
        }


class ReplayResourceGuard:
    """Admit replay materializations against a hard process memory ceiling.

    The hard limit is the minimum of the explicit ``request.max_rss_mib``, a
    finite readable cgroup v2 ``memory.max``, and available host memory plus
    the current process RSS. Replay work is admitted against a more
    conservative *operational* ceiling that reserves one eighth of the hard
    ceiling for native LightGBM/Polars allocator and OS transient allocations:
    ``operational_limit = hard_limit - hard_limit / 8`` (7,000 MiB for an
    8,000 MiB ceiling). Before each replay materialization the concrete
    ``DataFrame.estimated_size()``, the known NumPy bootstrap workspaces, the
    candidate score overlay, and the projected output arrays are admitted; the
    observed RSS is recorded after the stage. An unsafe admission or observed
    breach raises ``TrainingCapacityError`` with the deterministic
    ``replay_capacity_exceeded`` reason so callers can publish a traceable
    ``NO_TRADE`` artifact instead of leaving a missing result directory.
    """

    def __init__(
        self,
        request: TrainingRequest,
        *,
        replay_mode: ReplayMode = ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
    ) -> None:
        self.replay_mode = replay_mode
        self._hard_limit_mib = self._resolve_limit_mib(request)
        self._operational_limit_mib = self._hard_limit_mib - self._hard_limit_mib / 8.0
        self.baseline_rss_mib = self._rss_mib()
        self.peak_rss_mib = self.baseline_rss_mib
        self.stage_seconds: dict[str, float] = {}
        self.stage_estimated_bytes: dict[str, int] = {}
        self.stage_peak_rss_mib: dict[str, float] = {}
        self.prepared_decision_count = 0
        self.capacity_failure_reason: str | None = None
        self.bootstrap_batch_size = 0
        self.bootstrap_workspace_bytes = 0

    @staticmethod
    def _rss_mib() -> float:
        return float(psutil.Process().memory_info().rss) / (1024 * 1024)

    @staticmethod
    def _cgroup_v2_memory_max_mib() -> float | None:
        """Read the finite cgroup v2 ``memory.max`` ceiling when available."""
        try:
            with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
                raw = handle.read().strip()
        except OSError:
            return None
        if not raw or raw == "max":
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value / (1024 * 1024)

    @classmethod
    def _resolve_limit_mib(cls, request: TrainingRequest) -> float:
        candidates: list[float] = []
        if request.max_rss_mib is not None:
            candidates.append(float(request.max_rss_mib))
        cgroup_mib = cls._cgroup_v2_memory_max_mib()
        if cgroup_mib is not None:
            candidates.append(cgroup_mib)
        vm = psutil.virtual_memory()
        available = float(vm.available)
        if available <= 0.0:
            available = float(vm.total)
        candidates.append(available / (1024 * 1024) + cls._rss_mib())
        return min(candidates)

    def _fail(self, stage: str) -> None:
        self.capacity_failure_reason = "replay_capacity_exceeded"
        raise TrainingCapacityError(
            f"replay_capacity_exceeded:{stage} exceeds the "
            f"{self._operational_limit_mib:.1f} MiB operational ceiling "
            f"(hard {self._hard_limit_mib:.1f} MiB)"
        )

    def admit(self, estimated_bytes: int, *, stage: str) -> None:
        """Reject the run before allocation when the increment cannot fit."""
        estimate_mib = estimated_bytes / (1024 * 1024)
        current = self._rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, current)
        self.stage_estimated_bytes[stage] = int(estimated_bytes)
        if current + estimate_mib > self._operational_limit_mib:
            self._fail(stage)

    def bootstrap_workspace_cap(
        self,
        *,
        history_rows: int,
        projected_output_bytes: int,
        n_bootstrap: int,
    ) -> int:
        """Admit decision preparation and return the bounded bootstrap workspace.

        Samples current RSS once, converts the remaining operational-budget
        capacity to bytes, and reserves the projected calibrated output plus one
        conservative draw workspace (``history_rows * 24``) so allocator
        overlap is modeled instead of assuming an input frame is freed before
        its output exists. The selected batch cap is limited to the full
        ``n_bootstrap * history_rows * 24`` upper bound and admitted; when one
        draw plus the projected output cannot fit, the run stays fail-closed
        with ``replay_capacity_exceeded:decision_preparation``.
        """
        current_mib = self._rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, current_mib)
        remaining_bytes = int(
            max(0.0, self._operational_limit_mib - current_mib) * (1024 * 1024)
        )
        per_draw = history_rows * _BOOTSTRAP_BYTES_PER_ROW
        reserve = projected_output_bytes + per_draw
        max_batch = min(
            remaining_bytes - reserve,
            _bootstrap_workspace_bytes(n_bootstrap, history_rows),
        )
        if max_batch < per_draw:
            self._fail("decision_preparation")
        batch_draws = max_batch // per_draw
        workspace_cap = batch_draws * per_draw
        self.admit(workspace_cap + reserve, stage="decision_preparation")
        self.bootstrap_batch_size = batch_draws
        self.bootstrap_workspace_bytes = workspace_cap
        return workspace_cap

    def check_after(self, *, stage: str) -> None:
        """Record the observed RSS after a materialization and fail on breach."""
        current = self._rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, current)
        self.stage_peak_rss_mib[stage] = current
        if current > self._operational_limit_mib:
            self._fail(stage)

    def record_stage(self, stage: str, elapsed_seconds: float) -> None:
        self.stage_seconds[stage] = elapsed_seconds

    def record_prepared_decision(self) -> None:
        self.prepared_decision_count += 1

    def telemetry(self) -> dict[str, object]:
        return {
            "replay_stage_seconds": dict(self.stage_seconds),
            "replay_stage_estimated_bytes": dict(self.stage_estimated_bytes),
            "replay_stage_peak_rss_mib": dict(self.stage_peak_rss_mib),
            "replay_mode": self.replay_mode.value,
            "prepared_decision_count": int(self.prepared_decision_count),
            "capacity_failure_reason": self.capacity_failure_reason,
            "baseline_rss_mib": round(self.baseline_rss_mib, 3),
            "replay_peak_rss_mib": round(self.peak_rss_mib, 3),
            "replay_limit_mib": round(self._hard_limit_mib, 3),
            "replay_operational_limit_mib": round(self._operational_limit_mib, 3),
            "bootstrap_batch_size": int(self.bootstrap_batch_size),
            "bootstrap_workspace_bytes": int(self.bootstrap_workspace_bytes),
        }


def _build_replay_market_index(
    panel: pl.DataFrame,
    *,
    guard: ReplayResourceGuard | None = None,
) -> pl.DataFrame:
    """Drop every non-execution column from the replay market frame.

    Only the smallest column set needed by execution and allocation survives:
    the explicit point-in-time market columns plus the execution volatility
    field. ``feature__*`` (except ``feature__volatility_20d``),
    ``residual_o2o_*``, ``relevance*``, and ``label_available_time*`` columns
    never enter replay. Raises ``ValueError`` for a missing required column.
    """
    missing = [c for c in _REPLAY_MARKET_INDEX_REQUIRED if c not in panel.columns]
    if missing:
        raise ValueError(f"replay index must carry {', '.join(missing)}")
    columns = list(_REPLAY_MARKET_INDEX_REQUIRED) + [
        c for c in _REPLAY_MARKET_INDEX_OPTIONAL if c in panel.columns
    ]
    if guard is not None:
        guard.admit(int(panel.estimated_size()), stage="replay_market_index")
    return panel.select(columns)


def _bootstrap_workspace_bytes(
    n_bootstrap: int,
    history_rows: int,
) -> int:
    """Conservative NumPy moving-block bootstrap workspace bytes for all draws.

    The materialization retains the start-index array, the expanded indices,
    and the sampled values: three ``int64``/``float64`` row-sized work arrays,
    i.e. 24 bytes per history row per draw.
    """
    return n_bootstrap * history_rows * _BOOTSTRAP_BYTES_PER_ROW


@dataclass(frozen=True, slots=True)
class _StableTrialContext:
    """Transient per-fold inputs retained only until preparation completes.

    After :func:`_prepare_candidate_context` runs, the rich transformed
    train/validation Polars frames are released and the slim
    :class:`PreparedCandidateContext` becomes the only retained fold state.
    """

    train_processed: pl.DataFrame
    validation_processed: pl.DataFrame
    validation_frame: pl.DataFrame
    stable_scores: pl.DataFrame
    prepared: PreparedLambdaRankFold | None = None


@dataclass(frozen=True, slots=True)
class PreparedCandidateContext:
    """Slim per-fold candidate inputs retained after preparation.

    Carries the immutable prepared fold matrices plus the validation keys and
    stable scores aligned row-for-row with ``prepared.validation_matrix``, the
    labels needed for Rank-IC, and the row counts. The rich transformed
    train/validation frames are released after preparation, so no per-candidate
    fold retains predictor columns in OOS candidate frames.
    """

    prepared: PreparedLambdaRankFold
    validation_index: pl.DataFrame
    stable_scores: pl.DataFrame
    labels: pl.DataFrame
    train_rows: int
    validation_rows: int


@dataclass(slots=True)
class _SelectionTimings:
    """Exclusive route-qualified stage timings for one selection funnel."""

    context_prepare_seconds: float = 0.0
    refit_train_seconds: float = 0.0
    refit_predict_seconds: float = 0.0
    replay_prepare_seconds: float = 0.0
    economic_replay_seconds: float = 0.0
    actual_refit_rounds: dict[str, int] = field(default_factory=dict)
    actual_best_iterations: dict[str, int | None] = field(default_factory=dict)
    fold_telemetry: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Immutable route definition binding one holding horizon to its labels.

    A route is valid only when the composed snapshot carries its residual,
    relevance, and availability columns; every purge/embargo, rebalance cadence,
    calibration window, and cost amortization for the route uses
    ``horizon`` sessions.
    """

    horizon: int
    label_column: str
    relevance_column: str
    label_available_column: str

    @property
    def label_span_sessions(self) -> int:
        return self.horizon + 1


@dataclass(frozen=True, slots=True)
class EconomicCandidateEvidence:
    """Immutable per-candidate economic evidence for one shortlisted trial.

    All fields are frozen selection inputs or full-refit/replay diagnostics so
    the exact failed predicate for every shortlisted trial stays recoverable
    from the published artifact.
    """

    trial_number: int
    screen_economic_lower_bound: float
    fold_rank_ic: list[float]
    median_rank_ic: float
    attempted_orders: int
    filled_orders: int
    planned_cycles: int
    no_trade_reason_counts: dict[str, int]
    unfilled_order_reason_counts: dict[str, int]
    replay_finite: bool
    bootstrap_lower_bound: float
    strategy_ir: float
    max_drawdown: float
    turnover: float
    failure_reasons: tuple[str, ...]
    eligible: bool
    calibration_history_sessions: int = 0
    eligible_bucket_count: int = 0
    average_expected_net_alpha: float = 0.0
    cash_cycles: int = 0
    cost_drag: float = 0.0
    calibration_state: dict[str, object] | None = None
    holding_horizon_sessions: int = 5
    label_column: str = ""
    relevance_column: str = ""
    label_available_column: str = ""
    terminal_trial_count: int = 0
    policy_id: str = ""
    dsr_probability: float = 0.0
    geometric_excess_growth: float = 0.0
    compounding_block_count: int = 0
    legacy_daily_excess_lower_bound: float = 0.0
    replay_resource: dict[str, object] = field(default_factory=dict)
    compounding_overlay: dict[str, object] = field(default_factory=dict)
    total_terminal_screen_trials: int = 0
    route_terminal_screen_trials: int = 0
    exact_compounding_policy_replays: int = 0
    configured_compounding_policy_cells: int = 0
    selection_multiplicity_version: str = ""

    def to_json_safe(self) -> dict[str, object]:
        """JSON-serializable evidence row with deterministic failure reasons."""
        return {
            "trial_number": int(self.trial_number),
            "screen_economic_lower_bound": round(
                self.screen_economic_lower_bound, 8
            ),
            "fold_rank_ic": [round(value, 8) for value in self.fold_rank_ic],
            "median_rank_ic": round(self.median_rank_ic, 8),
            "attempted_orders": int(self.attempted_orders),
            "filled_orders": int(self.filled_orders),
            "planned_cycles": int(self.planned_cycles),
            "no_trade_reason_counts": dict(sorted(self.no_trade_reason_counts.items())),
            "unfilled_order_reason_counts": dict(
                sorted(self.unfilled_order_reason_counts.items())
            ),
            "replay_finite": bool(self.replay_finite),
            "bootstrap_lower_bound": round(self.bootstrap_lower_bound, 8),
            "strategy_ir": round(self.strategy_ir, 8),
            "max_drawdown": round(self.max_drawdown, 8),
            "turnover": round(self.turnover, 8),
            "failure_reasons": list(self.failure_reasons),
            "eligible": bool(self.eligible),
            "calibration_history_sessions": int(self.calibration_history_sessions),
            "eligible_bucket_count": int(self.eligible_bucket_count),
            "average_expected_net_alpha": round(self.average_expected_net_alpha, 10),
            "cash_cycles": int(self.cash_cycles),
            "cost_drag": round(self.cost_drag, 10),
            "calibration_state": self.calibration_state,
            "holding_horizon_sessions": int(self.holding_horizon_sessions),
            "label_column": str(self.label_column),
            "relevance_column": str(self.relevance_column),
            "label_available_column": str(self.label_available_column),
            "terminal_trial_count": int(self.terminal_trial_count),
            "policy_id": str(self.policy_id),
            "dsr_probability": round(self.dsr_probability, 8),
            "geometric_excess_growth": round(self.geometric_excess_growth, 8),
            "compounding_block_count": int(self.compounding_block_count),
            "legacy_daily_excess_lower_bound": round(
                self.legacy_daily_excess_lower_bound, 8
            ),
            "replay_resource": dict(self.replay_resource or {}),
            "compounding_overlay": dict(self.compounding_overlay or {}),
            "total_terminal_screen_trials": int(self.total_terminal_screen_trials),
            "route_terminal_screen_trials": int(self.route_terminal_screen_trials),
            "exact_compounding_policy_replays": int(
                self.exact_compounding_policy_replays
            ),
            "configured_compounding_policy_cells": int(
                self.configured_compounding_policy_cells
            ),
            "selection_multiplicity_version": str(self.selection_multiplicity_version),
        }


@dataclass(frozen=True, slots=True)
class ReplayStaticContext:
    """Immutable point-in-time market inputs shared across economic candidates.

    The compact market index, instrument map, and risk policy do not depend on
    any candidate's scores, so they are built once per selection panel and
    reused by every shortlist replay; only ``pred_score`` is candidate-specific
    and joined per replay. ``cache_bytes`` is the estimated resident size of
    the cached market inputs and participates in every resource-guard admit.
    """

    market_index: pl.DataFrame
    instruments: Mapping[str, Instrument]
    policy: StockRiskPolicy
    cache_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedSelectionRoute:
    """Candidate-invariant route OOS replay market and decision schedule.

    Built once per route after the inner folds are known. Owns one
    array-backed ``PreparedReplayMarket`` for the route OOS interval (execution
    only) and the aligned ``PreparedAllocationMarket`` built from the complete
    panel so the allocator sees the same pre-OOS volatility/covariance/ADTV
    warm-up as the reference replay, plus the canonical decision and execution
    session indexes, the OOS-decision-to-full-panel allocation-session index
    map, and the artifact schedule / initial portfolio the engine requires. A
    candidate contributes only a NaN ``float64`` execution overlay and an
    aligned allocation overlay scattered by :meth:`scatter_overlays`; no full
    market-score join, ``partition_by``, ``to_dicts``,
    ``_bounded_replay_history``, or new ``PreparedReplayMarket`` is created in
    the candidate replay loop.
    """

    market: PreparedReplayMarket
    allocation_market: PreparedAllocationMarket
    sessions: tuple[datetime, ...]
    decision_indices: tuple[int, ...]
    decision_index_by_time: Mapping[datetime, int]
    allocation_decision_index_by_time: Mapping[datetime, int]
    execution_index_by_decision: Mapping[int, int]
    allocation_window: int
    artifacts: ArtifactSchedule
    initial_portfolio: PortfolioSnapshot
    cache_bytes: int

    @classmethod
    def build(
        cls,
        panel: pl.DataFrame,
        oos_sessions: Sequence[datetime],
        request: TrainingRequest,
        route: RouteSpec,
        guard: ReplayResourceGuard | None = None,
    ) -> PreparedSelectionRoute:
        """Build the immutable route market, allocation market, and decision map.

        ``panel`` is the full pre-OOS training panel; ``oos_sessions`` are the
        chronological scored sessions that begin the route OOS interval. The
        execution ``PreparedReplayMarket`` covers only OOS sessions while the
        ``PreparedAllocationMarket`` is built from the complete panel with a
        causal rolling ADTV so the allocator sees the same pre-OOS warm-up as
        the reference ``_bounded_replay_history`` path. Each OOS decision
        timestamp maps to its full-panel allocation session index. The route
        cadence is the holding horizon in sessions, matching the replay
        policy's rebalance frequency. Raises ``ValueError`` for a panel missing
        the required replay columns or for an empty OOS interval.
        """
        from src.core.portfolio import PortfolioSnapshot
        from src.stocks.backtesting.engine import (
            REQUIRED_BACKTEST_COLUMNS,
            ArtifactSchedule,
            ArtifactSlot,
            PreparedReplayMarket,
        )

        if not oos_sessions:
            raise ValueError("PreparedSelectionRoute requires a non-empty OOS interval")
        replay_frame = panel.filter(pl.col("session") >= oos_sessions[0])
        if replay_frame.is_empty():
            raise ValueError("PreparedSelectionRoute OOS interval exposes no rows")
        missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in replay_frame.columns]
        if missing:
            raise ValueError(f"prepared route panel must carry {', '.join(missing)}")
        instruments = _instruments_from_frame(replay_frame)
        sessions = tuple(
            _session_as_datetime(s)
            for s in replay_frame["session"].unique().sort().to_list()
        )
        artifacts = ArtifactSchedule(
            slots=(
                ArtifactSlot(
                    eligible_from=sessions[0],
                    eligible_to=sessions[-1],
                    artifact_id=request.artifact_id,
                ),
            )
        )
        initial_portfolio = PortfolioSnapshot(
            account_snapshot_id="promotion",
            as_of=datetime(2000, 1, 1, tzinfo=UTC),
            settled_cash=request.initial_cash,
            unsettled_cash=0.0,
            positions=(),
        )
        if guard is not None:
            guard.admit(int(replay_frame.estimated_size()), stage="prepared_route")
        market = PreparedReplayMarket.build(
            replay_frame,
            adtv_window=20,
            instruments=instruments,
            artifacts=artifacts,
            initial_portfolio=initial_portfolio,
        )
        allocation_frame = panel.with_columns(
            pl.col("trading_value")
            .rolling_mean(20, min_samples=1)
            .over("instrument_id")
            .alias("adtv")
        )
        allocation_market = PreparedAllocationMarket.build(allocation_frame)
        cadence = max(1, int(route.horizon))
        decision_indices = tuple(
            i for i in range(len(sessions)) if i % cadence == 0
        )
        decision_times = tuple(
            _decision_time_at(market, i) for i in decision_indices
        )
        decision_index_by_time = {
            decision_times[position]: position
            for position in range(len(decision_indices))
        }
        policy = StockRiskPolicy(
            top_k=request.top_k,
            gross_cap=request.max_exposure,
            single_name_cap=request.max_single_weight,
            participation_limit=request.participation_limit,
            rebalance_frequency_sessions=cadence,
        )
        allocation_window = (
            max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
            + 1
        )
        allocation_session_index_of = {
            session: i for i, session in enumerate(allocation_market.sessions)
        }
        allocation_decision_index_by_time = {
            decision_times[position]: allocation_session_index_of[
                sessions[decision_indices[position]]
            ]
            for position in range(len(decision_indices))
        }
        execution_index_by_decision = {
            position: (decision_indices[position] + 1)
            for position in range(len(decision_indices))
            if decision_indices[position] + 1 < len(sessions)
        }
        return cls(
            market=market,
            allocation_market=allocation_market,
            sessions=sessions,
            decision_indices=decision_indices,
            decision_index_by_time=decision_index_by_time,
            allocation_decision_index_by_time=allocation_decision_index_by_time,
            execution_index_by_decision=execution_index_by_decision,
            allocation_window=allocation_window,
            artifacts=artifacts,
            initial_portfolio=initial_portfolio,
            cache_bytes=int(replay_frame.estimated_size()),
        )

    def scatter_overlay(self, oos_scored: pl.DataFrame) -> np.ndarray:
        """Scatter the candidate's narrow OOS scores into a NaN float64 overlay."""
        from src.stocks.backtesting.engine import PreparedReplayMarket

        market = self.market
        if not isinstance(market, PreparedReplayMarket):
            raise TypeError("PreparedSelectionRoute market must be a PreparedReplayMarket")
        overlay = np.full(market.row_count, np.nan, dtype=np.float64)
        rows_by_key = market.rows_by_key
        for row in oos_scored.select("instrument_id", "session", "pred_score").iter_rows(
            named=True
        ):
            key = (str(row["instrument_id"]), _session_as_datetime(row["session"]))
            prepared_row = rows_by_key.get(key)
            if prepared_row is not None:
                overlay[int(prepared_row.index)] = float(row["pred_score"])
        return overlay

    def scatter_overlays(
        self, oos_scored: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scatter the candidate's OOS scores into execution and allocation overlays.

        The execution overlay is aligned to the OOS execution market; the
        allocation overlay is aligned to the complete-panel allocation market
        and carries ``NaN`` on every row outside the OOS interval.
        """
        execution_overlay = self.scatter_overlay(oos_scored)
        allocation_overlay = np.full(
            self.allocation_market.row_count, np.nan, dtype=np.float64
        )
        oos_session_set = set(self.sessions)
        rows_by_key = self.allocation_market.rows_by_key
        for row in oos_scored.select(
            "instrument_id", "session", "pred_score"
        ).iter_rows(named=True):
            session = _session_as_datetime(row["session"])
            if session not in oos_session_set:
                continue
            allocation_index = rows_by_key.get(
                (str(row["instrument_id"]), session)
            )
            if allocation_index is not None:
                allocation_overlay[int(allocation_index)] = float(row["pred_score"])
        return execution_overlay, allocation_overlay

    def decision_index_for(self, decision_time: datetime) -> int | None:
        """OOS execution session index of the decision owning ``decision_time``."""
        position = self.decision_index_by_time.get(decision_time)
        if position is None or position >= len(self.decision_indices):
            return None
        return int(self.decision_indices[position])

    def allocation_decision_index_for(self, decision_time: datetime) -> int | None:
        """Full-panel allocation session index owning ``decision_time``."""
        return self.allocation_decision_index_by_time.get(decision_time)

    def window_indices(self, allocation_decision_index: int) -> np.ndarray:
        """Row indices of the bounded allocation window ending at a decision."""
        allocation_market = self.allocation_market
        if allocation_decision_index < 0 or allocation_decision_index >= len(
            allocation_market.sessions
        ):
            raise ValueError(
                f"allocation decision index {allocation_decision_index} "
                f"outside allocation sessions"
            )
        start = max(0, allocation_decision_index - self.allocation_window + 1)
        return np.concatenate(
            [
                np.arange(
                    allocation_market.session_ranges[i][0],
                    allocation_market.session_ranges[i][1],
                )
                for i in range(start, allocation_decision_index + 1)
            ]
        )

    def window_frame(
        self,
        allocation_decision_index: int,
        allocation_overlay: np.ndarray,
    ) -> pl.DataFrame:
        """Assemble the bounded decision-window frame from static arrays.

        Produces exactly the ``(instrument_id, session, pred_score, sector,
        adtv, close)`` cross-section the reference ``_bounded_replay_history``
        would expose for the same decision, including pre-OOS warm-up history.
        Historical overlay rows normalize ``NaN`` to ``null`` so calibration
        treats them as unscored rather than non-finite. No market-wide join,
        ``partition_by``, or per-row dict materialization is performed.
        """
        allocation_market = self.allocation_market
        if allocation_decision_index < 0 or allocation_decision_index >= len(
            allocation_market.sessions
        ):
            raise ValueError(
                f"allocation decision index {allocation_decision_index} "
                f"outside allocation sessions"
            )
        start = max(0, allocation_decision_index - self.allocation_window + 1)
        indices = np.concatenate(
            [
                np.arange(
                    allocation_market.session_ranges[i][0],
                    allocation_market.session_ranges[i][1],
                )
                for i in range(start, allocation_decision_index + 1)
            ]
        )
        if indices.size == 0:
            return pl.DataFrame(
                schema={
                    "instrument_id": pl.Utf8,
                    "session": pl.Datetime("us", "UTC"),
                    "pred_score": pl.Float64,
                    "sector": pl.Utf8,
                    "adtv": pl.Float64,
                    "close": pl.Float64,
                }
            )
        return pl.DataFrame(
            {
                "instrument_id": allocation_market.instrument_ids[indices],
                "session": pl.Series(
                    allocation_market.row_sessions[indices].tolist(),
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "pred_score": np.asarray(allocation_overlay)[indices],
                "sector": allocation_market.sector[indices],
                "adtv": allocation_market.adtv[indices],
                "close": allocation_market.close[indices],
            }
        ).with_columns(pl.col("pred_score").fill_nan(None))



def train_model(
    snapshot: DatasetSnapshot,
    registry: ModelArtifactRegistry,
    request: TrainingRequest,
) -> ModelManifest:
    """Derive and publish a promoted champion or an immutable ``NO_TRADE`` artifact."""
    manifest = snapshot.manifest
    if manifest.asset_kind is not AssetKind.STOCK:
        raise ValueError(
            f"train_model only accepts stock datasets, got {manifest.asset_kind.value}"
        )

    frame = research_eligible_frame(snapshot.frame)
    decision_time = frame["available_time"].max()
    if not isinstance(decision_time, datetime):
        raise ValueError("panel must carry a datetime available_time")
    validate_stock_rows_available(frame, decision_time)
    frame = _restrict_labels_available(frame, decision_time)

    feature_columns = v2_feature_columns(frame)
    if not feature_columns:
        raise ValueError("composed snapshot exposes no stock_alpha_v2 feature columns")
    _reject_predictor_target_columns(frame, feature_columns)

    route_specs = _resolve_route_specs(frame, request.candidate_horizons)
    if not route_specs:
        raise ValueError(
            "composed snapshot exposes no complete route label/relevance/availability "
            f"column set for candidate horizons {request.candidate_horizons}"
        )

    panel = _index_sessions(frame)
    n_sessions = int(panel["session_index"].n_unique())
    control_route = route_specs[0]
    eligible_from, eligible_to = _eligibility_from_panel(panel, control_route.horizon)

    base_manifest = ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v2",
        feature_schema_hash=manifest.schema_hash,
        universe_policy_hash=manifest.universe_policy_hash,
        label_definition=control_route.label_column,
        label_horizon_sessions=control_route.horizon,
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="lambdarank_blend",
    )

    if n_sessions < _MIN_TRAIN_SESSIONS:
        return _publish_no_trade(
            registry, request, base_manifest, panel, control_route.label_column,
            control_route.relevance_column, "insufficient-history",
            details=f"n_sessions={n_sessions}",
        )

    holdout_fold, training_panel = _reserve_forward_holdout(
        panel, request, control_route.label_span_sessions,
        control_route.label_available_column,
    )

    base = request.base_cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()

    run_store = TrainingRunStore.resolve(
        snapshot,
        request,
        feature_columns,
        route_specs,
        base,
        stress,
        registry_root=registry.root if request.run_root is not None else None,
    )

    champion_config, n_optuna_trials, champion_route = _tune_champion(
        training_panel, request, base_manifest, feature_columns, route_specs,
        dataset_manifest=snapshot.manifest,
        registry=registry,
        base_schedule=base,
        stress_schedule=stress,
        run_store=run_store,
    )
    if champion_config is None or champion_route is None:
        return _publish_no_trade(
            registry, request, base_manifest, panel, control_route.label_column,
            control_route.relevance_column, "no-champion-trial",
            tuning_telemetry=LambdaRankConfig._tuning_telemetry,
        )

    route = champion_route
    route_manifest = _route_manifest(base_manifest, route)
    tuning_telemetry = getattr(champion_config, "_tuning_telemetry", None) or {}
    selected_policy_id = str(tuning_telemetry.get("selected_policy_id", ""))
    selected_growth_risk_aversion = float(
        tuning_telemetry.get("selected_growth_risk_aversion", 1.0)
    )
    selected_turnover_budget = float(
        tuning_telemetry.get("selected_turnover_budget", 0.20)
    )
    selected_compounding = (
        CompoundingPolicyConfig(growth_risk_aversion=selected_growth_risk_aversion)
        if selected_policy_id
        else None
    )
    splitter = PurgedWalkForward(
        n_folds=request.n_folds,
        label_horizon_sessions=route.label_span_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
        validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
        min_train_sessions=_MIN_TRAIN_SESSIONS,
    )
    folds = splitter.split(training_panel)
    if not folds:
        return _publish_no_trade(
            registry, request, base_manifest, panel, route.label_column,
            route.relevance_column, "no-eligible-folds",
        )

    fold_models, scored_frames, fold_rank_ic = _fit_and_score_folds(
        training_panel, folds, request, route_manifest, feature_columns,
        route.label_column, route.relevance_column, champion_config,
    )
    if not fold_models:
        return _publish_no_trade(
            registry, request, base_manifest, panel, route.label_column,
            route.relevance_column, "no-fit-folds",
        )

    oos = pl.concat(scored_frames)
    _reject_non_finite_economic_inputs(oos)
    champion_oos_ledger = _build_calibration_ledger(
        oos, training_panel, route.label_column, route.label_available_column,
    )

    replay_guard = ReplayResourceGuard(
        request, replay_mode=ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS
    )
    replay_context = _prepare_replay_static_context(
        training_panel, request, holding_horizon_sessions=route.horizon,
        guard=replay_guard,
        turnover_budget=selected_turnover_budget,
        compounding=selected_compounding,
    )
    prepared_route = PreparedSelectionRoute.build(
        training_panel,
        [_session_as_datetime(oos["session"].min())],
        request,
        route,
        guard=replay_guard,
    )
    try:
        replay = _event_ledger_evaluation(
            training_panel, oos, request, snapshot.manifest, registry, base,
            stress, replay_context=replay_context,
            calibration_ledger=champion_oos_ledger,
            holding_horizon_sessions=route.horizon,
            label_column=route.label_column,
            label_available_column=route.label_available_column,
            replay_mode=ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
            replay_guard=replay_guard,
            prepared_route=prepared_route,
        )
    except TrainingCapacityError as exc:
        return _publish_no_trade(
            registry, request, base_manifest, panel, route.label_column,
            route.relevance_column, "replay-capacity-exceeded",
            details=str(exc),
            tuning_telemetry=replay_guard.telemetry(),
        )

    reasons: list[str] = []
    budget = PromotionRiskBudget()
    gates = _evaluate_gates(
        replay, fold_rank_ic, budget, request, n_trials=n_optuna_trials,
    )
    reasons.extend(cast(list[str], gates["reasons"]))

    holdout_ok, holdout_reason, _holdout_evidence = _evaluate_forward_holdout(
        registry, request, route_manifest, panel, holdout_fold, champion_config,
        feature_columns, route.label_column, route.relevance_column,
        snapshot.manifest, base, stress,
        calibration_ledger=champion_oos_ledger,
        label_span_sessions=route.label_span_sessions,
        label_available_column=route.label_available_column,
        holding_horizon_sessions=route.horizon,
        selected_policy_id=selected_policy_id,
        compounding=selected_compounding,
        turnover_budget=selected_turnover_budget,
    )
    reasons.append(holdout_reason)
    passed = bool(gates["passed"]) and holdout_ok and bool(fold_rank_ic)

    model = fold_models[-1] if passed else _no_trade_model(
        route_manifest, feature_columns, route.label_column, route.relevance_column,
        champion_config,
    )
    champion_calibration_state = replay.calibration_evidence.get("calibration_state")
    if champion_calibration_state is not None:
        model.set_calibration_state(
            cast(dict[str, object], champion_calibration_state)
        )
    published_manifest = model.manifest()
    registry.publish(model, published_manifest)
    registry.write_metrics(
        request.artifact_id,
        _build_metrics(
            request, replay, fold_rank_ic, gates, reasons, published_manifest,
            tuning_telemetry=getattr(champion_config, "_tuning_telemetry", {}),
        ),
    )
    logger.info(
        "published %s artifact %s (promoted=%s)",
        "champion" if passed else "NO_TRADE",
        request.artifact_id,
        passed,
    )
    return published_manifest


def _restrict_labels_available(frame: pl.DataFrame, decision_time: datetime) -> pl.DataFrame:
    availability_columns = [
        c for c in frame.columns if c.startswith("label_available_time")
    ]
    if not availability_columns:
        return frame
    condition = pl.lit(True)
    for column in availability_columns:
        condition = condition & (
            pl.col(column).is_null() | (pl.col(column) <= decision_time)
        )
    return frame.filter(condition)


def _index_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if "session_index" not in frame.columns:
        frame = frame.with_columns(
            pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
        )
    return frame.with_columns(
        pl.col("session_index").rank("dense").cast(pl.Int64).alias("session_index")
    )


def _eligibility_from_panel(
    panel: pl.DataFrame,
    horizon_sessions: int,
) -> tuple[str, str]:
    sessions = sorted(panel["session"].unique().to_list())
    if not sessions:
        raise ValueError("no sessions available for eligibility")
    first = sessions[0]
    last = sessions[-1]
    end = last if isinstance(last, datetime) else datetime.combine(last, datetime.min.time(), tzinfo=UTC)
    return first.isoformat(), end.isoformat()


def _reject_predictor_target_columns(frame: pl.DataFrame, feature_columns: tuple[str, ...]) -> None:
    offending = [c for c in feature_columns if c.startswith(("target_", "label_"))]
    if offending:
        raise ValueError(f"v2 predictors must not be target/label columns: {offending}")

def _resolve_route_specs(
    frame: pl.DataFrame,
    candidate_horizons: tuple[int, ...],
) -> tuple[RouteSpec, ...]:
    """Resolve active route specs from the composed snapshot columns.

    A configured horizon is active only when all three route columns
    (``residual_o2o_{h}d``, ``relevance_{h}d``, ``label_available_time_{h}d``)
    are present. The legacy five-day names ``relevance`` and
    ``label_available_time`` are accepted for the control horizon so v2
    single-horizon composed frames remain trainable. In a multi-horizon (v3)
    composed frame a configured horizon with a missing or non-finite route
    column fails closed with ``ValueError``; inactive legacy routes are logged
    and excluded and are never silently substituted with another horizon's
    labels.
    """
    multi_horizon = any(
        column.startswith("residual_o2o_") and column != "residual_o2o_5d"
        for column in frame.columns
    )
    routes: list[RouteSpec] = []
    invalid: list[tuple[int, str]] = []
    for horizon in candidate_horizons:
        label_column = f"residual_o2o_{horizon}d"
        relevance_column = f"relevance_{horizon}d"
        availability_column = f"label_available_time_{horizon}d"
        if (
            label_column not in frame.columns
            or relevance_column not in frame.columns
            or availability_column not in frame.columns
        ):
            if horizon == 5 and RELEVANCE_COLUMN in frame.columns:
                relevance_column = RELEVANCE_COLUMN
            if horizon == 5 and LABEL_AVAILABLE_COLUMN in frame.columns:
                availability_column = LABEL_AVAILABLE_COLUMN
        missing = [
            column
            for column in (label_column, relevance_column, availability_column)
            if column not in frame.columns
        ]
        if missing:
            if multi_horizon:
                invalid.append((horizon, f"missing columns {missing}"))
            else:
                logger.info(
                    "route horizon %s invalid: missing columns %s",
                    horizon,
                    missing,
                )
            continue
        if multi_horizon and not frame.filter(
            pl.col(label_column).is_not_null()
            & ~pl.col(label_column).is_finite()
        ).is_empty():
            invalid.append((horizon, f"non-finite {label_column}"))
            continue
        routes.append(
            RouteSpec(
                horizon=horizon,
                label_column=label_column,
                relevance_column=relevance_column,
                label_available_column=availability_column,
            )
        )
    if invalid:
        raise ValueError(
            "candidate route horizons fail closed: "
            + "; ".join(f"{h}d ({reason})" for h, reason in invalid)
        )
    return tuple(routes)


def _merge_replay_telemetry(
    route_attrs: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Aggregate per-route replay resource telemetry into one summary."""
    summaries = [
        cast(dict[str, object], attrs["replay_resource"])
        for attrs in route_attrs.values()
        if isinstance(attrs.get("replay_resource"), dict)
    ]
    if not summaries:
        return {}
    peak = max(
        float(cast(float, s.get("replay_peak_rss_mib", 0.0))) for s in summaries
    )
    prepared_decision_count = sum(
        int(cast(int, s.get("prepared_decision_count", 0))) for s in summaries
    )
    stage_seconds = cast(
        dict[str, object],
        next(
            (
                s.get("replay_stage_seconds")
                for s in summaries
                if isinstance(s.get("replay_stage_seconds"), dict)
            ),
            {},
        ),
    )
    replay_seconds = float(cast(float, stage_seconds.get("replay", 0.0)))
    return {
        "replay_stage_seconds": {
            "replay": round(replay_seconds, 3),
        },
        "replay_peak_rss_mib": round(peak, 3),
        "replay_operational_limit_mib": round(
            float(
                cast(float, summaries[0].get("replay_operational_limit_mib", 0.0))
            ),
            3,
        ),
        "replay_limit_mib": round(
            float(cast(float, summaries[0].get("replay_limit_mib", 0.0))), 3
        ),
        "prepared_decision_count": prepared_decision_count,
        "inner_stress_replay": False,
        "capacity_failure_reason": next(
            (
                s.get("capacity_failure_reason")
                for s in summaries
                if s.get("capacity_failure_reason") is not None
            ),
            None,
        ),
    }


def _route_manifest(base_manifest: ModelManifest, route: RouteSpec) -> ModelManifest:
    """Route-specific model manifest binding the label and holding horizon."""
    return ModelManifest(
        artifact_id=base_manifest.artifact_id,
        asset_kind=base_manifest.asset_kind,
        feature_set=base_manifest.feature_set,
        feature_schema_hash=base_manifest.feature_schema_hash,
        universe_policy_hash=base_manifest.universe_policy_hash,
        label_definition=route.label_column,
        label_horizon_sessions=route.horizon,
        eligible_from=base_manifest.eligible_from,
        eligible_to=base_manifest.eligible_to,
        model_type=base_manifest.model_type,
    )


def _tune_champion(
    tuning_panel: pl.DataFrame,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    route_specs: tuple[RouteSpec, ...],
    *,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    run_store: TrainingRunStore | None = None,
) -> tuple[LambdaRankConfig | None, int, RouteSpec | None]:
    """Run a temporally isolated proxy screen -> promotion -> exact selection.

    The tuning panel is the last purged-and-embargoed data available before the
    first outer validation decision, so no outer validation row influences
    candidate selection. Each active route gets an equal, explicit screen budget
    of ``request.optuna_trials // len(route_specs)`` serial seeded TPE
    configurations, all scored against one fixed-stride ``session_stride_proxy``
    context built from the fold-0 context (every ``ceil(sqrt(route_budget))``-th
    in-split session). A LightGBM NDCG callback drives Optuna median pruning,
    and pruned candidates remain terminal trials for Deflated Sharpe. A pruned
    or invalid route stays terminal evidence and is never silently reallocated
    to another horizon. The top ``ceil(sqrt(route_budget))`` positive-screen
    candidates per horizon (six of 27 for the 81/3/3 profile) are promoted to a
    full fold-0 refit with an adaptive continuation budget derived from the
    proxy best iteration, then the single top all-positive candidate (ranked by
    ``(-full_fold0_ic, -proxy_ic, trial_number)``) is refit over the remaining
    folds and replayed exactly once through the event ledger; a rejected
    finalist is never backfilled. A champion is selected across all routes
    lexicographically by ``(bootstrap lower bound, strategy IR, -max drawdown,
    -turnover, median Rank-IC, -holding horizon, -trial number)``. Returns
    ``(config, n_trials, route)`` where ``n_trials`` is the selected route's
    terminal screen trial count fed to Deflated Sharpe.
    """
    LambdaRankConfig._tuning_telemetry = None
    per_route_trials = max(1, request.optuna_trials // len(route_specs))
    total_terminal_screen_trials = per_route_trials * len(route_specs)
    guard = TrialResourceGuard(request, predictor_count=len(feature_columns) * 3)
    lgb_threads = _resolve_workflow_threads(request)

    route_champions: list[
        tuple[
            tuple[float, float, float, float, float, float, int, int, str],
            int,
            RouteSpec,
            optuna.Study,
            LambdaRankConfig,
        ]
    ] = []
    route_attrs: dict[str, dict[str, object]] = {}
    shortlist_evidence_all: list[dict[str, object]] = []
    n_terminal_total = 0
    screened_total = 0
    pruned_total = 0
    shortlisted_total = 0
    eligible_total = 0
    replays_evaluated_total = 0
    best_screen_proxy_lower_bound: float | None = None
    screen_seconds_total = 0.0
    refit_seconds_total = 0.0
    replay_seconds_total = 0.0
    early_rejected_total = 0
    early_rejected_seconds_total = 0.0

    for route in route_specs:
        route_manifest = _route_manifest(base_manifest, route)
        tuning_folds = PurgedWalkForward(
            n_folds=max(1, min(3, request.n_folds)),
            label_horizon_sessions=route.label_span_sessions,
            embargo_sessions=request.embargo_sessions,
            session_column="session_index",
            min_train_sessions=_MIN_TRAIN_SESSIONS // 2,
        ).split(tuning_panel)
        if not tuning_folds:
            route_attrs[str(route.horizon)] = {
                "selection_status": "no-eligible-tuning-folds",
                "holding_horizon_sessions": route.horizon,
            }
            logger.info("[EVAL] route=%sd stage=no-eligible-tuning-folds", route.horizon)
            continue
        policy = ScreenFidelityPolicy.for_budget(
            total_trials=request.optuna_trials,
            route_count=max(1, len(route_specs)),
            fold_count=max(1, len(tuning_folds)),
        )
        timings = _SelectionTimings()
        fold0_context, proxy_contexts, fold_context = _fit_stable_contexts(
            tuning_panel, tuning_folds, route_manifest, feature_columns,
            route.label_column, route.relevance_column,
            proxy_session_stride=policy.proxy_session_stride,
            timings=timings,
        )
        if fold0_context is None or any(
            proxy is None for proxy in proxy_contexts
        ):
            route_attrs[str(route.horizon)] = {
                "selection_status": "no-prepared-fold-0",
                "holding_horizon_sessions": route.horizon,
            }
            logger.info(
                "[EVAL] route=%sd stage=no-prepared-fold-0", route.horizon
            )
            continue
        proxy_reference = next(
            (proxy for proxy in proxy_contexts if proxy is not None),
            None,
        )
        cache_bytes = _fold_cache_bytes(fold0_context.prepared)
        logger.info(
            "[EVAL] route=%sd stage=proxy_built stride=%d proxy_folds=%d "
            "proxy_train_rows=%s proxy_validation_rows=%s "
            "full_train_rows=%d full_validation_rows=%d",
            route.horizon,
            policy.proxy_session_stride,
            len(proxy_contexts),
            [p.train_rows if p is not None else 0 for p in proxy_contexts],
            [p.validation_rows if p is not None else 0 for p in proxy_contexts],
            fold0_context.train_rows,
            fold0_context.validation_rows,
        )

        screen_phase = f"screen_h{route.horizon}"
        screen_evidence = {
            "route_horizon": route.horizon,
            "optuna_trials": per_route_trials,
            "study_name": f"lambdarank_v2_{request.artifact_id}_h{route.horizon}",
            "seed": request.seed,
            "screen_boosting_rounds": _SCREEN_BOOSTING_ROUNDS,
            "screen_early_stopping_rounds": _SCREEN_EARLY_STOPPING_ROUNDS,
            "screen_fidelity": "session_stride_proxy",
            "proxy_session_stride": policy.proxy_session_stride,
            "proxy_train_rows": (
                proxy_reference.train_rows if proxy_reference is not None else 0
            ),
            "proxy_validation_rows": (
                proxy_reference.validation_rows if proxy_reference is not None else 0
            ),
        }
        screen_phase_hash = content_hash(screen_evidence)
        study_name = str(screen_evidence["study_name"])
        storage: optuna.storages.BaseStorage
        if run_store is not None:
            storage = optuna.storages.RDBStorage(url=run_store.optuna_storage_url(route.horizon))
            resumed_screen = bool(
                run_store.resume
                and run_store.completed_phase(screen_phase, screen_phase_hash)
            )
        else:
            storage = optuna.storages.InMemoryStorage()
            resumed_screen = False

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=resumed_screen,
            sampler=optuna.samplers.TPESampler(
                seed=request.seed + route.horizon, n_startup_trials=10
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=max(1, per_route_trials // 5),
                n_warmup_steps=0,
            ),
        )

        def screen_objective(
            trial: optuna.Trial,
            _route: RouteSpec = route,
            _tuning_folds: list[Fold] = tuning_folds,
            _proxy_contexts: tuple[
                PreparedCandidateContext | None, ...
            ] = proxy_contexts,
            _route_manifest: ModelManifest = route_manifest,
            _lgb_threads: int = lgb_threads,
        ) -> float:
            config = _config_from_trial(trial, num_threads=_lgb_threads)
            screen_config = _screen_config(config)
            if any(proxy is None for proxy in _proxy_contexts):
                raise optuna.TrialPruned()
            fold_lower_bounds = [
                _score_trial_fold(
                    tuning_panel, _tuning_folds[fold_index],
                    cast(PreparedCandidateContext, proxy_context),
                    request, _route_manifest, feature_columns,
                    _route.label_column, _route.relevance_column,
                    screen_config, guard, trial, fold_index,
                    callbacks=(), report_progress=False,
                    key_prefix=f"h{_route.horizon}_",
                )
                for fold_index, proxy_context in enumerate(_proxy_contexts)
            ]
            if any(bound is None for bound in fold_lower_bounds):
                raise optuna.TrialPruned()
            valid_bounds = [float(bound) for bound in fold_lower_bounds if bound is not None]
            trial.set_user_attr("proxy_economic_lower_bounds", valid_bounds)
            objective = min(valid_bounds)
            logger.info(
                "[EVAL] route=%sd trial=%s stage=screen_proxy_lower_bound "
                "bounds=%s objective=%.8f",
                _route.horizon, trial.number,
                [round(bound, 8) for bound in valid_bounds],
                objective,
            )
            return objective

        if not resumed_screen:
            screen_started = time.perf_counter()
            study.optimize(
                screen_objective,
                n_trials=per_route_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
            screen_seconds = time.perf_counter() - screen_started
            if run_store is not None:
                run_store.checkpoint_phase(screen_phase, screen_evidence)
        else:
            screen_seconds = 0.0
            logger.info(
                "[SYS] route=%sd stage=screen_resumed from=%s",
                route.horizon,
                run_store.phase_path(screen_phase) if run_store is not None else "",
            )
        screen_seconds_total += screen_seconds
        logger.info(
            "[SYS] route=%sd stage=screen elapsed_ms=%.1f rss=%.1f",
            route.horizon,
            screen_seconds * 1000.0,
            TrialResourceGuard._rss_mib(),
        )

        n_terminal = sum(
            1
            for t in study.trials
            if t.state
            in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        )
        if n_terminal != per_route_trials:
            route_attrs[str(route.horizon)] = {
                "n_terminal_trials": n_terminal,
                "optuna_trials": per_route_trials,
                "selection_status": "incomplete",
                "holding_horizon_sessions": route.horizon,
            }
            continue
        complete = [
            t
            for t in study.trials
            if t.state is optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        completed: list[tuple[float, int]] = []
        for trial in complete:
            trial_value = trial.value
            if trial_value is None:
                continue
            completed.append((float(trial_value), trial.number))
        best_route_lb = max((value for value, _ in completed), default=None)
        if best_route_lb is not None and (
            best_screen_proxy_lower_bound is None
            or best_route_lb > best_screen_proxy_lower_bound
        ):
            best_screen_proxy_lower_bound = best_route_lb
        screen_scores = sorted(
            ((value, number) for value, number in completed if value > 0.0),
            key=lambda pair: (-pair[0], pair[1]),
        )
        shortlist = screen_scores[: policy.promotion_width]
        for _screen_lb, trial_number in shortlist:
            logger.info(
                "[EVAL] route=%sd trial=%s stage=promoted", route.horizon, trial_number
            )

        champion, selection = _select_economic_champion(
            study, shortlist, tuning_panel, tuning_folds, fold_context, request,
            route_manifest, feature_columns, route, guard, dataset_manifest,
            registry, base_schedule, stress_schedule,
            terminal_trial_count=n_terminal,
            total_terminal_screen_trials=total_terminal_screen_trials,
            policy=policy,
            proxy_best_iteration_by_trial={
                trial.number: int(cast(int, trial.user_attrs.get("proxy_best_iteration", 0)))
                for trial in study.trials
                if isinstance(trial.user_attrs.get("proxy_best_iteration"), int)
            },
            lgb_threads=lgb_threads,
            timings=timings,
        )
        if run_store is not None and selection is not None:
            run_store.checkpoint_phase(
                f"selection_h{route.horizon}",
                {
                    "route_horizon": route.horizon,
                    "promotion_width": policy.promotion_width,
                    "economic_finalist_width": policy.economic_finalist_width,
                    "selected_trial_number": selection.get("selected_trial_number"),
                    "promoted_trials": selection.get("promoted_trials"),
                    "all_positive_finalists": selection.get("all_positive_finalists"),
                    "economically_eligible_trials": selection.get(
                        "economically_eligible_trials"
                    ),
                    "economic_replay_seconds": selection.get(
                        "economic_replay_seconds"
                    ),
                },
            )

        telemetry = guard.telemetry()
        for name, value in telemetry.items():
            study.set_user_attr(name, value)
        study.set_user_attr("n_terminal_trials", n_terminal)
        study.set_user_attr("optuna_trials", per_route_trials)
        study.set_user_attr("screened_trials", len(complete))
        study.set_user_attr("pruned_trials", n_terminal - len(complete))
        study.set_user_attr("shortlisted_trials", len(shortlist))
        study.set_user_attr("selection_policy_version", SELECTION_POLICY_VERSION)
        for name, value in policy.to_json_safe().items():
            study.set_user_attr(name, value)
        study.set_user_attr("screen_fidelity", "session_stride_proxy")
        study.set_user_attr("proxy_session_stride", policy.proxy_session_stride)
        study.set_user_attr(
            "proxy_train_rows",
            proxy_reference.train_rows if proxy_reference is not None else 0,
        )
        study.set_user_attr(
            "proxy_validation_rows",
            proxy_reference.validation_rows if proxy_reference is not None else 0,
        )
        study.set_user_attr("cache_bytes", cache_bytes)
        study.set_user_attr("screen_seconds", screen_seconds)
        study.set_user_attr("resolved_lgb_threads", lgb_threads)
        study.set_user_attr("compute_plan_version", COMPUTE_PLAN_VERSION)
        study.set_user_attr("holding_horizon_sessions", route.horizon)
        study.set_user_attr("label_column", route.label_column)
        study.set_user_attr("relevance_column", route.relevance_column)
        study.set_user_attr("label_available_column", route.label_available_column)
        if best_route_lb is not None:
            study.set_user_attr("best_screen_proxy_lower_bound", best_route_lb)
        selection_telemetry = selection
        if selection_telemetry is None:
            selection_telemetry = _selection_telemetry(
                0.0,
                timings,
                lgb_threads,
                [],
                [],
                [],
                [],
                replay_guard=None,
                total_terminal_screen_trials=total_terminal_screen_trials,
                route_terminal_screen_trials=n_terminal,
                exact_compounding_policy_replays=0,
                configured_compounding_policy_cells=1,
                selection_multiplicity_version=SELECTION_MULTIPLICITY_VERSION,
            )
        for name, value in selection_telemetry.items():
            study.set_user_attr(name, value)
        if selection is not None:
            for name, value in selection.items():
                study.set_user_attr(name, value)
        if champion is None or selection is None:
            study.set_user_attr(
                "selection_status",
                "no_complete_screen_candidate"
                if not shortlist
                else "no_economically_eligible_candidate",
            )
            logger.info(
                "[EVAL] route=%sd stage=selection_status %s",
                route.horizon,
                study.user_attrs["selection_status"],
            )
        else:
            study.set_user_attr("selection_status", "selected")
            logger.info(
                "[EVAL] route=%sd trial=%s stage=selected",
                route.horizon,
                int(cast(int, selection["selected_trial_number"])),
            )

        n_terminal_total += n_terminal
        screened_total += len(complete)
        pruned_total += n_terminal - len(complete)
        shortlisted_total += len(shortlist)
        route_evidence = cast(
            list[dict[str, object]],
            selection.get("shortlist_candidate_evidence", []) if selection else [],
        )
        shortlist_evidence_all.extend(route_evidence)
        eligible_route = int(
            cast(int, selection.get("economically_eligible_trials", 0)) if selection else 0
        )
        eligible_total += eligible_route
        refit_seconds_total += float(
            cast(float, selection.get("full_refit_seconds", 0.0)) if selection else 0.0
        )
        replay_seconds_total += float(
            cast(float, selection.get("economic_replay_seconds", 0.0)) if selection else 0.0
        )
        early_rejected_total += int(
            cast(int, selection.get("early_rejected_full_refits", 0)) if selection else 0
        )
        early_rejected_seconds_total += float(
            cast(
                float,
                selection.get("early_rejected_full_refit_seconds", 0.0) if selection else 0.0,
            )
        )
        replays_evaluated_total += int(
            cast(int, selection.get("exact_compounding_policy_replays", 0)) if selection else 0
        )
        route_attrs[str(route.horizon)] = dict(study.user_attrs)

        proxy_min_by_trial: dict[int, float] = {}
        for trial in study.trials:
            bounds = trial.user_attrs.get("proxy_economic_lower_bounds")
            if isinstance(bounds, list) and bounds:
                proxy_min_by_trial[int(trial.number)] = float(min(bounds))
        concordance_x: list[float] = []
        concordance_y: list[float] = []
        for row in route_evidence:
            trial_number = int(cast(int, row.get("trial_number", -1)))
            if trial_number in proxy_min_by_trial:
                replay_lb = row.get("bootstrap_lower_bound")
                if isinstance(replay_lb, (int, float)) and math.isfinite(float(replay_lb)):
                    concordance_x.append(proxy_min_by_trial[trial_number])
                    concordance_y.append(float(replay_lb))
        if concordance_x:
            route_attrs[str(route.horizon)]["proxy_exact_spearman_concordance"] = round(
                _spearman_concordance(concordance_x, concordance_y), 8
            )

        if champion is None or selection is None:
            continue
        key = (
            float(cast(float, selection["selected_inner_compounding_lower_bound"])),
            float(cast(float, selection["selected_inner_dsr_probability"])),
            float(cast(float, selection["selected_inner_geometric_excess_growth"])),
            -float(cast(float, selection["selected_inner_max_drawdown"])),
            -float(cast(float, selection["selected_inner_turnover"])),
            float(cast(float, selection["selected_inner_median_rank_ic"])),
            -route.horizon,
            -int(cast(int, selection["selected_trial_number"])),
            str(cast(str, selection["selected_policy_id"])),
        )
        route_champions.append((key, n_terminal, route, study, champion))

    merged: dict[str, object] = {
        "candidate_horizons": list(request.candidate_horizons),
        "active_routes": [route.horizon for route in route_specs],
        "per_route_trial_budget": per_route_trials,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "compute_plan_version": COMPUTE_PLAN_VERSION,
        "selection_multiplicity_version": SELECTION_MULTIPLICITY_VERSION,
        "resolved_lgb_threads": lgb_threads,
        **guard.telemetry(),
        "n_terminal_trials": n_terminal_total,
        "optuna_trials": n_terminal_total,
        "total_terminal_screen_trials": total_terminal_screen_trials,
        "global_multiplicity_count": total_terminal_screen_trials,
        "configured_compounding_policy_cells": 1,
        "exact_compounding_policy_replays": replays_evaluated_total,
        "screened_trials": screened_total,
        "pruned_trials": pruned_total,
        "shortlisted_trials": shortlisted_total,
        "economically_eligible_trials": eligible_total,
        "cache_bytes": sum(
            int(cast(int, attrs.get("cache_bytes", 0)))
            for attrs in route_attrs.values()
        ),
        "screen_seconds": screen_seconds_total,
        "full_refit_seconds": refit_seconds_total,
        "economic_replay_seconds": replay_seconds_total,
        "early_rejected_full_refits": early_rejected_total,
        "early_rejected_full_refit_seconds": early_rejected_seconds_total,
        "full_refit_boosting_rounds": (
            _SCREEN_BOOSTING_ROUNDS + 2 * _SCREEN_EARLY_STOPPING_ROUNDS
        ),
        "full_refit_early_stopping_rounds": 2 * _SCREEN_EARLY_STOPPING_ROUNDS,
        "shortlist_candidate_evidence": shortlist_evidence_all,
        "replay_resource": _merge_replay_telemetry(route_attrs),
        "routes": route_attrs,
    }
    for policy_key in (
        "route_budget",
        "proxy_session_stride",
        "promotion_width",
        "economic_finalist_width",
        "fold_count",
        "all_positive_finalists",
        "promoted_trials",
    ):
        if policy_key not in merged:
            for attrs in route_attrs.values():
                if policy_key in attrs:
                    merged[policy_key] = attrs[policy_key]
                    break
    for telemetry_key in (
        "screen_fidelity",
        "proxy_train_rows",
        "proxy_validation_rows",
    ):
        if telemetry_key not in merged:
            for attrs in route_attrs.values():
                if telemetry_key in attrs:
                    merged[telemetry_key] = attrs[telemetry_key]
                    break
    if best_screen_proxy_lower_bound is not None:
        merged["best_screen_proxy_lower_bound"] = best_screen_proxy_lower_bound

    if not route_champions:
        statuses = [
            str(attrs.get("selection_status"))
            for attrs in route_attrs.values()
            if attrs.get("selection_status")
        ]
        if not statuses:
            merged["selection_status"] = "no-eligible-route"
        elif "no_complete_screen_candidate" in statuses:
            merged["selection_status"] = "no_complete_screen_candidate"
        elif "no_economically_eligible_candidate" in statuses:
            merged["selection_status"] = "no_economically_eligible_candidate"
        else:
            merged["selection_status"] = statuses[0]
        LambdaRankConfig._tuning_telemetry = merged
        logger.info(
            "[EVAL] stage=selection_status %s",
            merged["selection_status"],
        )
        return None, n_terminal_total, None

    route_champions.sort(key=lambda row: row[0], reverse=True)
    _winner_key, _winner_n_terminal, winner_route, winner_study, _ = route_champions[0]
    winner_config = _config_from_params(
        dict(winner_study.trials[
            int(cast(int, winner_study.user_attrs["selected_trial_number"]))
        ].params),
        num_threads=lgb_threads,
    )
    winner_selection = {
        name: value
        for name, value in winner_study.user_attrs.items()
        if name.startswith("selected_")
    }
    winner_evidence = {
        name: value
        for name, value in winner_study.user_attrs.items()
        if name.startswith("selected_inner_")
    }
    merged["selection_status"] = "selected"
    merged["selected_horizon"] = winner_route.horizon
    merged["selected_label_column"] = winner_route.label_column
    merged["selected_relevance_column"] = winner_route.relevance_column
    merged["selected_label_available_column"] = winner_route.label_available_column
    merged.update(winner_selection)
    merged.update(winner_evidence)
    for policy_key in (
        "selection_policy_version",
        "route_budget",
        "proxy_session_stride",
        "promotion_width",
        "economic_finalist_width",
        "fold_count",
        "all_positive_finalists",
        "promoted_trials",
        "early_rejected_full_refits",
        "compounding_policy_replays",
        "total_terminal_screen_trials",
        "route_terminal_screen_trials",
        "configured_compounding_policy_cells",
        "selection_multiplicity_version",
    ):
        if policy_key in winner_study.user_attrs:
            merged[policy_key] = winner_study.user_attrs[policy_key]
    for telemetry_key in (
        "screen_fidelity",
        "proxy_train_rows",
        "proxy_validation_rows",
        "best_screen_proxy_lower_bound",
        "proxy_exact_spearman_concordance",
        "fold_retention_telemetry",
    ):
        if telemetry_key in winner_study.user_attrs:
            merged[telemetry_key] = winner_study.user_attrs[telemetry_key]
    winner_config._tuning_telemetry = merged
    logger.info(
        "[EVAL] route=%sd trial=%s stage=selected",
        winner_route.horizon,
        int(cast(int, winner_selection["selected_trial_number"])),
    )
    return winner_config, total_terminal_screen_trials, winner_route


def _visible_cpu_counts() -> tuple[int, int]:
    """Visible logical CPUs (cgroup-affine) and physical cores.

    ``logical`` respects the cgroup CPU affinity mask; ``physical`` falls back
    to half the logical count (minimum one) when detection is unavailable.
    """
    try:
        logical = max(1, len(os.sched_getaffinity(0)))
    except OSError:
        logical = max(1, os.cpu_count() or 1)
    physical = psutil.cpu_count(logical=False)
    if physical is None or physical < 1:
        physical = max(1, logical // 2)
    return physical, logical


def _resolve_workflow_threads(request: TrainingRequest) -> int:
    """Resolve the deterministic workflow LightGBM thread plan once."""
    physical, logical = _visible_cpu_counts()
    return resolve_lgb_num_threads(request.lgb_threads, physical, logical)


def _fit_stable_context(
    tuning_panel: pl.DataFrame,
    fold: Fold,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
) -> _StableTrialContext:
    """Fit one StableRankComposite and prepare one immutable fold context.

    The composite's fitted weights/orientations/winsors and the validation
    stable scores are invariant across every LambdaRank search parameter, so
    they are computed once; the prepared fold matrices are derived in the same
    call. The returned rich context is transient and is slimmed by
    :func:`_prepare_candidate_context` before retention.
    """
    allowlist = stock_alpha_v2_allowlist()
    train_frame = tuning_panel[fold.train_mask]
    validation_frame = tuning_panel[fold.validation_mask]
    quantiles = fit_v2_winsor_quantiles(train_frame, feature_columns)
    train_processed = apply_v2_transforms(
        train_frame, feature_columns, winsor_quantiles=quantiles
    )
    validation_processed = apply_v2_transforms(
        validation_frame, feature_columns, winsor_quantiles=quantiles
    )
    stable = StableRankComposite(
        factors=allowlist,
        manifest=base_manifest,
        label_column=label_column,
        block_length=base_manifest.label_horizon_sessions,
        session_column="session",
    )
    stable.fit(train_processed, validation_processed)
    predict_input = _drop_target_columns(validation_processed, label_column)
    stable_scores = stable.predict(predict_input).select(
        "session", "instrument_id", "pred_score"
    )
    prepared: PreparedLambdaRankFold | None = None
    try:
        prepared = LambdaRankBlendModel(
            base_manifest,
            allowlist,
            label_column,
            config=LambdaRankConfig(),
            session_column="session",
            relevance_column=relevance_column or RELEVANCE_COLUMN,
        ).prepare_fold(train_processed, validation_processed)
    except ValueError:
        prepared = None
    return _StableTrialContext(
        train_processed=train_processed,
        validation_processed=validation_processed,
        validation_frame=validation_frame,
        stable_scores=stable_scores,
        prepared=prepared,
    )


def _prepare_candidate_context(
    rich: _StableTrialContext,
    label_column: str,
    relevance_column: str | None,
) -> PreparedCandidateContext | None:
    """Slim a transient fold context into the retained prepared candidate state.

    The validation keys are aligned row-for-row to the prepared validation
    matrix (the same null-relevance filter and session sort the matrix used);
    stable scores and labels are narrowed to the retained keys. The rich
    transformed frames are dropped after this call. Returns ``None`` when the
    fold cannot be prepared (a degenerate fold is fail-closed and later folds
    are never built).
    """
    prepared = rich.prepared
    if prepared is None or prepared.validation_matrix is None:
        return None
    relevance = relevance_column or RELEVANCE_COLUMN
    val_used = (
        rich.validation_processed.filter(pl.col(relevance).is_not_null())
        .sort("session")
        .select("session", "instrument_id")
    )
    if val_used.height != prepared.validation_matrix.shape[0]:
        raise ValueError(
            "prepared validation rows must align with the prepared validation "
            f"matrix ({val_used.height} != {prepared.validation_matrix.shape[0]})"
        )
    validation_index = val_used
    stable_scores = rich.stable_scores.join(
        validation_index, on=["session", "instrument_id"], how="inner"
    )
    labels = rich.validation_frame.select(
        "session_index", "session", "instrument_id", pl.col(label_column)
    )
    return PreparedCandidateContext(
        prepared=prepared,
        validation_index=validation_index,
        stable_scores=stable_scores,
        labels=labels,
        train_rows=int(prepared.train_matrix.shape[0]),
        validation_rows=int(prepared.validation_matrix.shape[0]),
    )


class _FoldContextProvider:
    """Lazily materialize per-fold prepared candidate contexts on demand.

    Fold 0 is seeded by the caller because the proxy and the six fold-0
    promotions share it. Any remaining fold is built exactly once when a
    finalist requests it and is never cached, so a remaining-fold matrix cannot
    overlap the replay phase or a later fold.
    """

    def __init__(
        self,
        tuning_panel: pl.DataFrame,
        tuning_folds: list[Fold],
        base_manifest: ModelManifest,
        feature_columns: tuple[str, ...],
        label_column: str,
        relevance_column: str | None,
        *,
        timings: _SelectionTimings | None = None,
    ) -> None:
        self._panel = tuning_panel
        self._folds = tuning_folds
        self._manifest = base_manifest
        self._features = feature_columns
        self._label = label_column
        self._relevance = relevance_column
        self._timings = timings
        self._seeded: dict[int, PreparedCandidateContext | None] = {}

    def seed(self, index: int, context: PreparedCandidateContext | None) -> None:
        self._seeded[index] = context

    def __call__(self, index: int) -> PreparedCandidateContext | None:
        if index in self._seeded:
            return self._seeded[index]
        if index < 0 or index >= len(self._folds):
            raise IndexError(f"fold index {index} outside the tuning fold set")
        started = time.perf_counter()
        rich = _fit_stable_context(
            self._panel, self._folds[index], self._manifest, self._features,
            self._label, self._relevance,
        )
        slim = _prepare_candidate_context(rich, self._label, self._relevance)
        del rich
        if self._timings is not None:
            self._timings.context_prepare_seconds += time.perf_counter() - started
        return slim

    def release(self) -> None:
        """Drop every seeded context before replay materialization."""
        self._seeded.clear()


def _fit_stable_contexts(
    tuning_panel: pl.DataFrame,
    tuning_folds: list[Fold],
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    *,
    proxy_session_stride: int,
    timings: _SelectionTimings | None = None,
) -> tuple[
    PreparedCandidateContext | None,
    tuple[PreparedCandidateContext | None, ...],
    _FoldContextProvider,
]:
    """Build the full fold-0 context, per-fold proxy contexts, and a lazy provider.

    Fold 0 is materialized and retained in full because every promoted
    full-refit and the fold-0 proxy share it. One slim fixed-stride proxy
    context is built for every tuning fold so every Optuna trial can be scored
    on all purged proxy folds; the rich full contexts for folds 1..n are
    discarded after their proxy preparation. Later full folds are materialized
    lazily by the returned provider, one at a time and only after a finalist is
    known. Returns ``(fold0, proxy_contexts, provider)`` where ``proxy_contexts``
    is one entry per tuning fold; either a full fold-0 or a proxy entry is
    ``None`` when its fold cannot be prepared, in which case the route stays
    fail-closed and later folds are never built.
    """
    if not tuning_folds:
        raise ValueError("tuning_folds must not be empty")
    fold0_rich = _fit_stable_context(
        tuning_panel, tuning_folds[0], base_manifest, feature_columns,
        label_column, relevance_column,
    )
    started = time.perf_counter()
    fold0 = _prepare_candidate_context(fold0_rich, label_column, relevance_column)
    proxy_contexts: list[PreparedCandidateContext | None] = []
    for index, fold in enumerate(tuning_folds):
        if index == 0:
            rich = fold0_rich
        else:
            rich = _fit_stable_context(
                tuning_panel, fold, base_manifest, feature_columns,
                label_column, relevance_column,
            )
        proxy_rich = _build_proxy_context(
            rich, proxy_session_stride, base_manifest, feature_columns,
            label_column, relevance_column,
        )
        proxy = _prepare_candidate_context(proxy_rich, label_column, relevance_column)
        proxy_contexts.append(proxy)
        if timings is not None:
            timings.fold_telemetry[f"fold_{index}"] = _fold_retention_telemetry(
                rich.train_processed,
                rich.prepared,
                relevance_column or RELEVANCE_COLUMN,
            )
        del proxy_rich
        if index != 0:
            del rich
    del fold0_rich
    if timings is not None:
        timings.context_prepare_seconds += time.perf_counter() - started
    provider = _FoldContextProvider(
        tuning_panel, tuning_folds, base_manifest, feature_columns,
        label_column, relevance_column, timings=timings,
    )
    provider.seed(0, fold0)
    return fold0, tuple(proxy_contexts), provider


def _proxy_session_filter(
    frame: pl.DataFrame,
    stride: int,
    session_column: str = "session",
) -> pl.DataFrame:
    """Keep sessions whose ordinal within the frame is congruent to zero.

    The ordinal is the 0-based position of the session in the frame's own
    chronological session order, so the same fixed rule is applied to the train
    and validation splits independently. All retained rows keep their original
    columns, groups, relevance, label availability, and stable score values; the
    selection is deterministic, causal, and never stratifies by labels or picks
    a candidate-specific sample.
    """
    if stride <= 0:
        raise ValueError("proxy session stride must be positive")
    sessions = (
        frame.select(pl.col(session_column).unique().alias(session_column))
        .sort(session_column)
        .with_row_index("__ordinal")
        .filter(pl.col("__ordinal") % stride == 0)
        .select(session_column)
    )
    return frame.join(sessions, on=session_column, how="inner")


def _build_proxy_context(
    full_context: _StableTrialContext,
    stride: int,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
) -> _StableTrialContext:
    """Build the fixed-stride proxy screen context from the fold-0 context.

    The proxy reuses the already-transformed fold-0 context, filtering both the
    train and validation splits by the same ``session ordinal % stride == 0``
    rule. The cached stable scores are filtered to the retained validation
    sessions unchanged (the composite score is a per-session cross-sectional
    percentile rank, so retained rows keep identical values), and the prepared
    fold matrices are rebuilt once from the filtered frames so the screen fast
    path still avoids per-trial Polars work. Raises ``ValueError`` when the
    proxy exposes no qualifying group (a degenerate proxy is never silently
    accepted as a screen signal).
    """
    proxy_train = _proxy_session_filter(full_context.train_processed, stride)
    proxy_validation = _proxy_session_filter(
        full_context.validation_processed, stride
    )
    proxy_validation_frame = _proxy_session_filter(
        full_context.validation_frame, stride
    )
    proxy_stable_scores = _proxy_session_filter(
        full_context.stable_scores, stride
    )
    prepared: PreparedLambdaRankFold | None = None
    try:
        prepared = LambdaRankBlendModel(
            base_manifest,
            stock_alpha_v2_allowlist(),
            label_column,
            config=LambdaRankConfig(),
            session_column="session",
            relevance_column=relevance_column or RELEVANCE_COLUMN,
        ).prepare_fold(proxy_train, proxy_validation)
    except ValueError:
        prepared = None
    return _StableTrialContext(
        train_processed=proxy_train,
        validation_processed=proxy_validation,
        validation_frame=proxy_validation_frame,
        stable_scores=proxy_stable_scores,
        prepared=prepared,
    )


def _score_trial_fold(
    tuning_panel: pl.DataFrame,
    fold: Fold,
    context: PreparedCandidateContext,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig,
    guard: TrialResourceGuard,
    trial: optuna.Trial,
    fold_index: int,
    *,
    callbacks: Sequence[Callable[..., object]] = (),
    report_progress: bool = True,
    key_prefix: str = "",
) -> float | None:
    """Score one tuning fold with the cached slim context; ``None`` prunes.

    The resource guard admits the allocation (including the prepared-fold cache
    bytes) before fitting and records elapsed/RSS telemetry in a ``finally`` so
    a callback-raised :class:`optuna.TrialPruned` still leaves timing evidence.
    ``key_prefix`` qualifies the recorded fold key with the route horizon so
    cross-route keys never collide. The returned value is the cost-aware
    non-overlapping proxy compounding lower bound, never a Rank-IC; a missing,
    non-finite, insufficient-block, or non-positive fold fails closed to
    ``None``. A breach raises :class:`TrainingCapacityError`. Fold model,
    prediction frame, and LightGBM datasets are local to this call and released
    on return, so a trial never retains every fold's artifacts.
    """
    del tuning_panel, fold, feature_columns
    key = f"{key_prefix}trial_{trial.number}_fold_{fold_index}"
    guard.admit(
        context.train_rows,
        extra_bytes=_fold_cache_bytes(context.prepared),
    )
    started = time.perf_counter()
    try:
        result = _score_context_model(
            context, request, base_manifest, label_column, relevance_column, config,
            callbacks=callbacks,
        )
        if result is None:
            return None
        _ic, scored, outcome = result
        if outcome.best_iteration is not None:
            trial.set_user_attr("proxy_best_iteration", outcome.best_iteration)
        bound = _economic_screen_score(
            context.labels,
            scored,
            label_column=label_column,
            top_k=request.top_k,
            holding_horizon_sessions=int(base_manifest.label_horizon_sessions),
            cost_schedule=request.stress_cost_schedule or default_stress_schedule(),
            n_bootstrap=request.n_bootstrap,
            bootstrap_alpha=request.bootstrap_alpha,
            seed=request.seed,
        )
        if not math.isfinite(bound) or bound <= 0.0:
            return None
        if report_progress:
            trial.report(float(bound), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(bound)
    finally:
        guard.record_fold(
            key,
            time.perf_counter() - started,
            guard.estimate_mib(context.train_rows),
        )
        guard.check_after()


def _score_context_model(
    context: PreparedCandidateContext,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig,
    *,
    callbacks: Sequence[Callable[..., object]] = (),
    initial_rounds: int | None = None,
    timings: _SelectionTimings | None = None,
) -> tuple[float, pl.DataFrame, FitTrialOutcome] | None:
    """Fit one candidate on a cached prepared fold and return ``(rank_ic, scored, outcome)``.

    ``None`` signals a fail-closed fold (missing columns, unusable groups, or
    invalid inputs). The prepared-fold fast path trains on the immutable
    matrices and predicts directly from ``validation_matrix``, returning a slim
    ``(session, instrument_id, pred_score)`` frame; no transformed predictor
    frame is reconstructed or retained. ``initial_rounds`` selects the adaptive
    continuation driver for promoted full refits. ``timings``, when supplied,
    accumulates the exclusive train/predict durations of a full refit.
    """
    model = LambdaRankBlendModel(
        base_manifest,
        stock_alpha_v2_allowlist(),
        label_column,
        config=config,
        session_column="session",
        relevance_column=relevance_column or RELEVANCE_COLUMN,
    )
    try:
        train_started = time.perf_counter()
        outcome = model.fit_trial_prepared(
            context.prepared,
            context.stable_scores,
            callbacks=callbacks,
            initial_rounds=initial_rounds,
        )
        if timings is not None:
            timings.refit_train_seconds += time.perf_counter() - train_started
    except ValueError:
        return None
    if not outcome.fit_ok or model.no_trade:
        return None
    try:
        predict_started = time.perf_counter()
        scored = model.predict_prepared_scores(
            context.prepared,
            context.validation_index,
            context.stable_scores,
        )
        if timings is not None:
            timings.refit_predict_seconds += time.perf_counter() - predict_started
    except ValueError:
        return None
    ic = _median_rank_ic(context.labels, scored, label_column)
    return float(ic), scored, outcome


def _frame_bytes(frame: pl.DataFrame) -> int:
    """Estimated resident bytes of an immutable replay market input."""
    return int(frame.estimated_size())


def _fold_cache_bytes(prepared: PreparedLambdaRankFold | None) -> int:
    """Byte size of the immutable per-fold matrices held in the search cache."""
    if prepared is None:
        return 0
    total = (
        int(prepared.train_matrix.nbytes)
        + int(prepared.train_relevance.nbytes)
        + int(prepared.train_weights.nbytes)
    )
    if (
        prepared.validation_matrix is not None
        and prepared.validation_relevance is not None
    ):
        total += int(prepared.validation_matrix.nbytes)
        total += int(prepared.validation_relevance.nbytes)
    return total


def _fold_retention_telemetry(
    train: pl.DataFrame,
    prepared: PreparedLambdaRankFold | None,
    relevance_column: str,
) -> dict[str, object]:
    """Flat per-fold retention telemetry for one prepared fold.

    ``labeled_rows`` is the relevance-eligible row count before preparation;
    ``retained_rows`` is the prepared train matrix height (rows with null
    predictors are retained, so the retention ratio is exactly the null
    predictor drop, if any). ``oldest_session_weight`` and
    ``newest_session_weight`` are the per-session recency weight sums of the
    first and last chronological sessions in the prepared fold.
    """
    if relevance_column in train.columns:
        labeled = int(train.filter(pl.col(relevance_column).is_not_null()).height)
    else:
        labeled = 0
    retained = int(prepared.train_matrix.shape[0]) if prepared is not None else 0
    oldest_weight = 0.0
    newest_weight = 0.0
    if prepared is not None and prepared.train_group_sizes:
        sizes = prepared.train_group_sizes
        weights = prepared.train_weights
        start = 0
        for size in sizes:
            start += int(size)
        end = start
        start = 0
        first = sizes[0]
        oldest_weight = float(np.sum(weights[start : start + int(first)]))
        last = sizes[-1]
        newest_weight = float(np.sum(weights[end - int(last) : end]))
    retention_ratio = float(retained / labeled) if labeled else 0.0
    return {
        "labeled_rows": labeled,
        "retained_rows": retained,
        "retention_ratio": round(retention_ratio, 6),
        "oldest_session_weight": round(oldest_weight, 8),
        "newest_session_weight": round(newest_weight, 8),
    }


def _screen_config(config: LambdaRankConfig) -> LambdaRankConfig:
    """Frozen reduced-budget screen profile; not a candidate parameter."""
    params = {
        name: getattr(config, name)
        for name in (
            "objective",
            "metric",
            "label_gain",
            "eval_at",
            "seed",
            "num_leaves",
            "learning_rate",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "max_bin",
            "min_group_size",
            "half_life_sessions",
            "num_threads",
        )
    }
    return LambdaRankConfig(
        **params,
        n_estimators=_SCREEN_BOOSTING_ROUNDS,
        early_stopping_rounds=_SCREEN_EARLY_STOPPING_ROUNDS,
    )


def _screen_informed_full_refit_config(config: LambdaRankConfig) -> LambdaRankConfig:
    """Frozen full-refit budget for shortlisted candidates only.

    Every shortlisted candidate keeps its sampled hyperparameters and pinned
    seeds, but the boosting budget is bounded by the structural screen
    relation (the whole screen horizon plus two screen-patience windows)
    instead of the unpinned 5,000-round default. Returns a new config; the
    frozen Optuna-sampled config is never mutated.
    """
    n_estimators = _SCREEN_BOOSTING_ROUNDS + 2 * _SCREEN_EARLY_STOPPING_ROUNDS
    patience = 2 * _SCREEN_EARLY_STOPPING_ROUNDS
    if n_estimators < 1:
        raise ValueError("derived full-refit boosting rounds must be positive")
    if patience < 1:
        raise ValueError("derived full-refit early-stopping patience must be positive")
    if patience >= n_estimators:
        raise ValueError(
            "derived full-refit early-stopping patience must be smaller "
            "than the boosting-round budget"
        )
    params = {
        name: getattr(config, name)
        for name in (
            "objective",
            "metric",
            "label_gain",
            "eval_at",
            "seed",
            "num_leaves",
            "learning_rate",
            "max_depth",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "max_bin",
            "min_group_size",
            "half_life_sessions",
            "num_threads",
        )
    }
    return LambdaRankConfig(
        **params,
        n_estimators=n_estimators,
        early_stopping_rounds=patience,
    )


def _fit_and_score_candidate(
    tuning_panel: pl.DataFrame,
    tuning_folds: list[Fold],
    fold_context: Callable[[int], PreparedCandidateContext | None],
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig,
    guard: TrialResourceGuard,
    candidate_key: str,
    *,
    static_cache_bytes: int = 0,
    fold_indices: tuple[int, ...] | None = None,
    initial_rounds: int | None = None,
    key_prefix: str = "",
    timings: _SelectionTimings | None = None,
) -> tuple[list[float], pl.DataFrame | None] | None:
    """Full-budget refit of one promoted candidate over selected inner folds.

    ``fold_context`` materializes each requested fold lazily (fold 0 is already
    built and seeded; remaining folds are built on demand and released). The
    ``initial_rounds`` supplies the adaptive continuation first-pass budget
    derived from the candidate's proxy screen best iteration. Returns
    ``(fold_rank_ic, concatenated slim validation scores)`` for the requested
    folds, ``(fold_rank_ic, None)`` when the first non-positive full-refit
    Rank-IC rejects the candidate early, or ``None`` when any fold fails closed.
    The early stop is equivalence-preserving: eligibility already requires every
    full-refit IC to be strictly positive, so the skipped later folds and replay
    cannot make that candidate eligible. Per-fold models, prediction frames,
    and LightGBM datasets are released before the next fold.
    """
    del tuning_panel, feature_columns
    fold_indices = fold_indices or tuple(range(len(tuning_folds)))
    fold_rank_ic: list[float] = []
    scored_frames: list[pl.DataFrame] = []
    full_refit_rounds = config.n_estimators
    full_refit_patience = config.early_stopping_rounds
    for fold_index in fold_indices:
        context = fold_context(fold_index)
        if context is None:
            return None
        key = f"{key_prefix}refit_{candidate_key}_fold_{fold_index}"
        guard.admit(
            context.train_rows,
            extra_bytes=(
                _fold_cache_bytes(context.prepared)
                + static_cache_bytes
            ),
        )
        started = time.perf_counter()
        logger.info(
            "[EVAL] %s stage=full_refit_fold_start fold=%d rounds=%d patience=%d rss=%.1f",
            candidate_key,
            fold_index,
            full_refit_rounds,
            full_refit_patience,
            TrialResourceGuard._rss_mib(),
        )
        try:
            result = _score_context_model(
                context, request, base_manifest, label_column, relevance_column, config,
                initial_rounds=initial_rounds,
                timings=timings,
            )
            if result is None:
                return None
            ic, scored, outcome = result
            if outcome.best_iteration is not None:
                logger.info(
                    "[EVAL] %s stage=full_refit_best_iteration fold=%d best=%d "
                    "stopped_early=%s rounds=%d continuation=%s",
                    candidate_key,
                    fold_index,
                    outcome.best_iteration,
                    outcome.stopped_early,
                    outcome.rounds_trained,
                    outcome.used_continuation,
                )
            if timings is not None:
                timings.actual_refit_rounds[
                    f"{key_prefix}refit_{candidate_key}_fold_{fold_index}"
                ] = outcome.rounds_trained
                timings.actual_best_iterations[
                    f"{key_prefix}refit_{candidate_key}_fold_{fold_index}"
                ] = outcome.best_iteration
        finally:
            guard.record_fold(
                key,
                time.perf_counter() - started,
                guard.estimate_mib(context.train_rows),
            )
            guard.check_after()
        fold_rank_ic.append(ic)
        logger.info(
            "[EVAL] %s stage=full_refit_fold_done fold=%d rounds=%d patience=%d "
            "elapsed_ms=%.1f rss=%.1f",
            candidate_key,
            fold_index,
            full_refit_rounds,
            full_refit_patience,
            (time.perf_counter() - started) * 1000.0,
            TrialResourceGuard._rss_mib(),
        )
        if ic <= 0.0:
            logger.info(
                "[EVAL] %s stage=early_rejected fold_rank_ic=%.6f",
                candidate_key,
                ic,
            )
            return fold_rank_ic, None
        scored_frames.append(scored)
    return fold_rank_ic, pl.concat(scored_frames) if scored_frames else pl.DataFrame()


def _prepare_replay_static_context(
    panel: pl.DataFrame,
    request: TrainingRequest,
    *,
    holding_horizon_sessions: int = 5,
    guard: ReplayResourceGuard | None = None,
    turnover_budget: float | None = None,
    compounding: CompoundingPolicyConfig | None = None,
) -> ReplayStaticContext:
    """Build the candidate-invariant replay inputs once per selection panel.

    The compact market index (carrying the immutable point-in-time execution
    columns and the causal 20-session ADTV), instrument map, and risk policy
    are shared by every shortlisted candidate. ``holding_horizon_sessions``
    fixes the policy's rebalance cadence so a route replays at its own session
    frequency. ``turnover_budget`` and ``compounding`` override the frozen risk
    policy so a selected compounding grid cell replays unchanged. Raises
    ``ValueError`` for a missing required replay column or a non-finite cached
    market input.
    """
    from src.stocks.backtesting.engine import REQUIRED_BACKTEST_COLUMNS

    frame = panel.drop("session_index")
    missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"replay panel must carry {', '.join(missing)}")
    _reject_non_finite_economic_inputs(frame)
    frame = frame.sort("session").with_columns(
        pl.col("trading_value")
        .rolling_mean(20, min_samples=1)
        .over("instrument_id")
        .alias("adtv")
    )
    market_index = _build_replay_market_index(frame, guard=guard)
    instruments = _instruments_from_frame(market_index)
    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
        rebalance_frequency_sessions=holding_horizon_sessions,
        turnover_budget=turnover_budget if turnover_budget is not None else 0.20,
        compounding=compounding if compounding is not None else CompoundingPolicyConfig(),
    )
    return ReplayStaticContext(
        market_index=market_index,
        instruments=instruments,
        policy=policy,
        cache_bytes=_frame_bytes(market_index),
    )


_ECONOMIC_FAILURE_CODES = (
    "non_positive_fold_rank_ic",
    "no_attempted_orders",
    "no_filled_orders",
    "non_finite_replay",
    "non_positive_bootstrap_lower_bound",
    "dsr_below_threshold",
    "replay_capacity_exceeded",
)


def _evaluate_economic_candidate(
    fold_rank_ic: list[float],
    replay: ReplayResult,
    request: TrainingRequest,
    trial_number: int,
    screen_economic_lower_bound: float,
    *,
    holding_horizon_sessions: int = 5,
    label_column: str = "",
    relevance_column: str = "",
    label_available_column: str = "",
    terminal_trial_count: int = 0,
    policy_id: str = "",
    total_terminal_screen_trials: int = 0,
    route_terminal_screen_trials: int = 0,
    exact_compounding_policy_replays: int = 0,
    configured_compounding_policy_cells: int = 0,
    selection_multiplicity_version: str = "",
) -> EconomicCandidateEvidence:
    """Evaluate every economic predicate and emit immutable candidate evidence.

    All predicates are evaluated without early return so the exact failed
    reason codes stay recoverable for every shortlisted (trial, policy) replay.
    ``eligible`` requires positive Rank-IC folds, attempted and filled orders,
    a finite replay, a strictly positive holding-period block-log compounding
    bootstrap lower bound, and a Deflated Sharpe probability at or above the
    frozen risk-budget threshold. The legacy daily arithmetic excess bootstrap
    is retained only as ``legacy_daily_excess_lower_bound`` diagnostic.
    """
    failures: list[str] = []
    if not fold_rank_ic or not all(ic > 0.0 for ic in fold_rank_ic):
        failures.append("non_positive_fold_rank_ic")
    if replay.attempted_orders <= 0:
        failures.append("no_attempted_orders")
    if replay.filled_orders <= 0:
        failures.append("no_filled_orders")
    replay_finite = _replay_is_finite(replay)
    if not replay_finite:
        failures.append("non_finite_replay")
    budget = PromotionRiskBudget()
    dsr_count = total_terminal_screen_trials or terminal_trial_count
    compounding = _compounding_evidence(
        replay.strategy_returns,
        replay.benchmark_returns,
        replay.decision_boundaries,
        holding_horizon_sessions,
        request,
        budget,
        dsr_count,
    )
    bootstrap_lower_bound = compounding.bootstrap_lower_bound
    if bootstrap_lower_bound <= 0.0:
        failures.append("non_positive_bootstrap_lower_bound")
    dsr_probability = compounding.dsr_probability
    if dsr_probability < budget.deflated_sharpe_probability:
        failures.append("dsr_below_threshold")
    calibration_evidence = replay.calibration_evidence or {}
    return EconomicCandidateEvidence(
        trial_number=trial_number,
        screen_economic_lower_bound=screen_economic_lower_bound,
        fold_rank_ic=list(fold_rank_ic),
        median_rank_ic=float(np.median(fold_rank_ic)) if fold_rank_ic else 0.0,
        attempted_orders=replay.attempted_orders,
        filled_orders=replay.filled_orders,
        planned_cycles=replay.planned_cycles,
        no_trade_reason_counts=dict(replay.no_trade_reason_counts),
        unfilled_order_reason_counts=dict(replay.unfilled_order_reason_counts),
        replay_finite=replay_finite,
        bootstrap_lower_bound=bootstrap_lower_bound,
        strategy_ir=_information_ratio(replay.strategy_returns),
        max_drawdown=float(replay.metrics.get("max_drawdown", 1.0)),
        turnover=float(replay.metrics.get("turnover", 0.0)),
        failure_reasons=tuple(
            code for code in _ECONOMIC_FAILURE_CODES if code in failures
        ),
        eligible=not failures,
        calibration_history_sessions=int(
            cast(int, calibration_evidence.get("history_sessions", 0))
        ),
        eligible_bucket_count=int(
            cast(int, calibration_evidence.get("eligible_bucket_count", 0))
        ),
        average_expected_net_alpha=float(
            cast(float, calibration_evidence.get("average_expected_net_alpha", 0.0))
        ),
        cash_cycles=int(replay.no_trade_reason_counts.get("no-feasible-allocation", 0)),
        cost_drag=float(replay.metrics.get("cost_drag", 0.0)),
        calibration_state=(
            cast(dict[str, object], calibration_evidence.get("calibration_state"))
            if calibration_evidence.get("calibration_state") is not None
            else None
        ),
        holding_horizon_sessions=holding_horizon_sessions,
        label_column=label_column,
        relevance_column=relevance_column,
        label_available_column=label_available_column,
        terminal_trial_count=terminal_trial_count,
        policy_id=policy_id,
        dsr_probability=dsr_probability,
        geometric_excess_growth=(
            float(np.mean(compounding.block_log_excess))
            if compounding.block_log_excess
            else 0.0
        ),
        compounding_block_count=compounding.complete_block_count,
        legacy_daily_excess_lower_bound=_inner_bootstrap_lower_bound(replay, request),
        replay_resource=dict(replay.replay_resource or {}),
        compounding_overlay=dict(replay.compounding_overlay or {}),
        total_terminal_screen_trials=total_terminal_screen_trials,
        route_terminal_screen_trials=route_terminal_screen_trials,
        exact_compounding_policy_replays=exact_compounding_policy_replays,
        configured_compounding_policy_cells=configured_compounding_policy_cells,
        selection_multiplicity_version=selection_multiplicity_version,
    )


def _economically_eligible(
    fold_rank_ic: list[float],
    replay: ReplayResult,
    request: TrainingRequest,
) -> bool:
    """Gate every economic candidate on temporally isolated evidence only."""
    if not fold_rank_ic or not all(ic > 0.0 for ic in fold_rank_ic):
        return False
    if replay.attempted_orders <= 0 or replay.filled_orders <= 0:
        return False
    if not _replay_is_finite(replay):
        return False
    return _inner_bootstrap_lower_bound(replay, request) > 0.0


def _inner_bootstrap_lower_bound(
    replay: ReplayResult,
    request: TrainingRequest,
) -> float:
    """Moving-block bootstrap excess lower bound matching the outer gate."""
    if not replay.excess_returns:
        return 0.0
    budget = PromotionRiskBudget()
    return _moving_block_bootstrap_lower_bound(
        replay.excess_returns,
        block_length=max(5, 1),
        n_bootstrap=max(request.n_bootstrap, 2),
        seed=request.seed,
        alpha=budget.bootstrap_alpha,
    )


def _replay_is_finite(replay: ReplayResult) -> bool:
    """Reject any non-finite numeric evidence from an economic replay."""
    samples: list[float] = (
        list(replay.excess_returns)
        + list(replay.strategy_returns)
        + list(replay.benchmark_returns)
        + list(replay.metrics.values())
        + [replay.final_value, replay.base_total_return, replay.benchmark_total_return]
    )
    if replay.stress_total_return is not None:
        samples.append(replay.stress_total_return)
    if replay.stress_metrics is not None:
        samples.extend(v for v in replay.stress_metrics.values() if v is not None)
    return all(math.isfinite(float(v)) for v in samples)


def _select_economic_champion(
    study: optuna.Study,
    shortlist: list[tuple[float, int]],
    tuning_panel: pl.DataFrame,
    tuning_folds: list[Fold],
    fold_context: _FoldContextProvider,
    request: TrainingRequest,
    route_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    route: RouteSpec,
    guard: TrialResourceGuard,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    *,
    terminal_trial_count: int,
    total_terminal_screen_trials: int = 0,
    policy: ScreenFidelityPolicy,
    proxy_best_iteration_by_trial: dict[int, int] | None = None,
    lgb_threads: int = 1,
    timings: _SelectionTimings | None = None,
) -> tuple[LambdaRankConfig | None, dict[str, object] | None]:
    """Promote, refit, and exactly replay the route's economic finalists.

    The promoted ``shortlist`` (already limited to ``policy.promotion_width``)
    is full-refit on fold 0 with an adaptive continuation budget derived from
    each candidate's proxy screen best iteration and ranked by
    ``(-full_fold0_ic, -proxy_lower_bound, trial_number)``. Up to
    ``policy.economic_finalist_width`` all-positive candidates are then refit
    over the remaining folds and each is replayed through the exact event
    ledger exactly once under the single pre-registered default
    ``StockRiskPolicy`` (``CompoundingPolicyConfig()`` and its frozen turnover
    budget). The former six-policy compounding grid is removed from active
    selection. A non-positive fold or a failed replay is never backfilled: a
    route without an eligible finalist has no champion, which is intentionally
    conservative in the financial sense. Replay inputs are built once per route
    and reused; only the candidate score overlay differs.
    ``ReplayResourceGuard`` admits and releases every replay, and a capacity
    failure makes that candidate ineligible rather than permitting an unguarded
    replay. Only candidates with positive Rank-IC folds, actual fills, a finite
    replay, a strictly positive block-log compounding bootstrap lower bound,
    and a Deflated Sharpe probability at or above the frozen threshold are
    ordered by ``(compounding lower bound, DSR probability, geometric excess
    growth, -max drawdown, -turnover, median Rank-IC, -holding horizon,
    -trial number, policy id)``. The DSR input count for every route and final
    promotion is ``total_terminal_screen_trials`` (the global terminal screen
    count, including completed and pruned/failed trials). Returns ``(config,
    selection telemetry)`` or ``(None, None)`` when no promoted candidate
    survives the funnel.
    """
    replay_guard = ReplayResourceGuard(
        request, replay_mode=ReplayMode.INNER_SELECTION_BASE_ONLY
    )
    route_terminal_screen_trials = terminal_trial_count
    configured_compounding_policy_cells = 1
    exact_compounding_policy_replays = 0
    if not shortlist:
        return None, _selection_telemetry(
            0.0,
            timings,
            lgb_threads,
            [],
            [],
            [],
            [],
            replay_guard=replay_guard,
            total_terminal_screen_trials=total_terminal_screen_trials,
            route_terminal_screen_trials=route_terminal_screen_trials,
            exact_compounding_policy_replays=exact_compounding_policy_replays,
            configured_compounding_policy_cells=configured_compounding_policy_cells,
            selection_multiplicity_version=SELECTION_MULTIPLICITY_VERSION,
        )
    key_prefix = f"h{route.horizon}_"
    refit_started = time.perf_counter()
    candidate_rows: list[
        tuple[
            tuple[float, float, float, float, float, float, int, int, str],
            tuple[int, str],
        ]
    ] = []
    best_cell_by_trial: dict[
        int,
        tuple[
            tuple[float, float, float, float, float, float, int, int, str],
            EconomicCandidateEvidence,
        ],
    ] = {}
    shortlist_evidence: list[dict[str, object]] = []
    early_rejected_seconds: list[float] = []
    replay_seconds = 0.0
    screen_lb_by_trial = {trial_number: screen_lb for screen_lb, trial_number in shortlist}
    proxy_best_by_trial = dict(proxy_best_iteration_by_trial or {})

    remaining_indices = tuple(range(1, len(tuning_folds)))
    fold0_ranked: list[tuple[float, float, int, LambdaRankConfig, pl.DataFrame]] = []
    for _screen_lb, trial_number in shortlist:
        frozen = study.trials[trial_number]
        config = _config_from_params(dict(frozen.params), num_threads=lgb_threads)
        full_refit_config = _screen_informed_full_refit_config(config)
        initial_rounds = adaptive_refit_rounds(proxy_best_by_trial.get(trial_number))
        refit_started_trial = time.perf_counter()
        refit = _fit_and_score_candidate(
            tuning_panel, tuning_folds, fold_context, request, route_manifest,
            feature_columns, route.label_column, route.relevance_column,
            full_refit_config,
            guard, f"trial{trial_number}",
            fold_indices=(0,),
            initial_rounds=initial_rounds,
            key_prefix=key_prefix,
            timings=timings,
        )
        if refit is None:
            logger.info("[EVAL] trial=%s stage=refit_failed", trial_number)
            continue
        fold_rank_ic, oos0 = refit
        if oos0 is None or not fold_rank_ic or fold_rank_ic[0] <= 0.0:
            early_rejected_seconds.append(time.perf_counter() - refit_started_trial)
            logger.info("[EVAL] trial=%s stage=fold0_rejected", trial_number)
            continue
        fold0_ranked.append(
            (float(fold_rank_ic[0]), _screen_lb, trial_number, full_refit_config, oos0)
        )

    fold0_ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    finalists: list[tuple[list[float], pl.DataFrame, int]] = []
    for fold0_ic, _screen_lb, trial_number, full_refit_config, oos0 in fold0_ranked:
        if len(finalists) >= policy.economic_finalist_width:
            break
        initial_rounds = adaptive_refit_rounds(proxy_best_by_trial.get(trial_number))
        if not remaining_indices:
            finalists.append(([fold0_ic], oos0, trial_number))
            logger.info(
                "[EVAL] trial=%s stage=all_positive_finalist fold_rank_ic=%s",
                trial_number, [fold0_ic],
            )
            continue
        refit_started_trial = time.perf_counter()
        refit = _fit_and_score_candidate(
            tuning_panel, tuning_folds, fold_context, request, route_manifest,
            feature_columns, route.label_column, route.relevance_column,
            full_refit_config,
            guard, f"trial{trial_number}",
            fold_indices=remaining_indices,
            initial_rounds=initial_rounds,
            key_prefix=key_prefix,
            timings=timings,
        )
        if refit is None:
            logger.info("[EVAL] trial=%s stage=remaining_folds_refit_failed", trial_number)
            continue
        remaining_ic, oos_rest = refit
        if oos_rest is None:
            early_rejected_seconds.append(time.perf_counter() - refit_started_trial)
            logger.info("[EVAL] trial=%s stage=finalist_rejected", trial_number)
            continue
        full_ic = [fold0_ic, *list(remaining_ic)]
        full_oos = pl.concat([oos0, oos_rest]) if not oos0.is_empty() or not oos_rest.is_empty() else oos0
        finalists.append((full_ic, full_oos, trial_number))
        logger.info(
            "[EVAL] trial=%s stage=all_positive_finalist fold_rank_ic=%s",
            trial_number, full_ic,
        )

    if not finalists:
        refit_seconds = time.perf_counter() - refit_started
        logger.info(
            "[SYS] route=%sd stage=shortlist elapsed_ms=%.1f rss=%.1f",
            route.horizon,
            refit_seconds * 1000.0,
            TrialResourceGuard._rss_mib(),
        )
        return None, _selection_telemetry(
            replay_seconds, timings, lgb_threads,
            early_rejected_seconds, shortlist_evidence, fold0_ranked, finalists,
            replay_guard=replay_guard,
            total_terminal_screen_trials=total_terminal_screen_trials,
            route_terminal_screen_trials=route_terminal_screen_trials,
            exact_compounding_policy_replays=exact_compounding_policy_replays,
            configured_compounding_policy_cells=configured_compounding_policy_cells,
            selection_multiplicity_version=SELECTION_MULTIPLICITY_VERSION,
        )

    fold_context.release()
    gc.collect()

    replay_started = time.perf_counter()
    replay_context = _prepare_replay_static_context(
        tuning_panel, request, holding_horizon_sessions=route.horizon,
        guard=replay_guard,
    )
    validation_sessions = sorted(
        tuning_panel.filter(
            pl.col("session_index").is_in(tuning_folds[0].validation_mask)
        )["session"].unique().to_list()
    )
    prepared_route = (
        PreparedSelectionRoute.build(
            tuning_panel,
            [_session_as_datetime(validation_sessions[0])],
            request,
            route,
            guard=replay_guard,
        )
        if validation_sessions
        else None
    )
    if timings is not None:
        timings.replay_prepare_seconds += time.perf_counter() - replay_started

    for fold_rank_ic, oos, trial_number in finalists:
        _screen_lb = screen_lb_by_trial[trial_number]
        causal_oos_ledger = _build_calibration_ledger(
            oos, tuning_panel, route.label_column, route.label_available_column,
        )
        policy_id = _DEFAULT_COMPOUNDING_POLICY_ID
        replay_started = time.perf_counter()
        try:
            replay = _event_ledger_evaluation(
                tuning_panel, oos, request, dataset_manifest, registry,
                base_schedule, stress_schedule, replay_context=replay_context,
                calibration_ledger=causal_oos_ledger,
                holding_horizon_sessions=route.horizon,
                label_column=route.label_column,
                label_available_column=route.label_available_column,
                replay_mode=ReplayMode.INNER_SELECTION_BASE_ONLY,
                replay_guard=replay_guard,
                prepared_route=prepared_route,
            )
        except TrainingCapacityError as exc:
            replay_guard.capacity_failure_reason = "replay_capacity_exceeded"
            replay_seconds += time.perf_counter() - replay_started
            logger.warning(
                "[EVAL] trial=%s policy=%s stage=replay_capacity_exceeded %s",
                trial_number, policy_id, exc,
            )
            shortlist_evidence.append(
                {
                    "trial_number": int(trial_number),
                    "policy_id": policy_id,
                    "eligible": False,
                    "failure_reasons": ["replay_capacity_exceeded"],
                    "attempted_orders": 0,
                    "filled_orders": 0,
                    "replay_finite": False,
                    "replay_resource": replay_guard.telemetry(),
                    "total_terminal_screen_trials": int(
                        total_terminal_screen_trials
                    ),
                    "route_terminal_screen_trials": int(
                        route_terminal_screen_trials
                    ),
                    "exact_compounding_policy_replays": 1,
                    "configured_compounding_policy_cells": int(
                        configured_compounding_policy_cells
                    ),
                    "selection_multiplicity_version": str(
                        SELECTION_MULTIPLICITY_VERSION
                    ),
                }
            )
            continue
        replay_seconds += time.perf_counter() - replay_started
        exact_compounding_policy_replays += 1
        evidence = _evaluate_economic_candidate(
            fold_rank_ic, replay, request, trial_number, _screen_lb,
            holding_horizon_sessions=route.horizon,
            label_column=route.label_column,
            relevance_column=route.relevance_column,
            label_available_column=route.label_available_column,
            terminal_trial_count=total_terminal_screen_trials,
            policy_id=policy_id,
            total_terminal_screen_trials=total_terminal_screen_trials,
            route_terminal_screen_trials=route_terminal_screen_trials,
            exact_compounding_policy_replays=1,
            configured_compounding_policy_cells=configured_compounding_policy_cells,
            selection_multiplicity_version=SELECTION_MULTIPLICITY_VERSION,
        )
        shortlist_evidence.append(evidence.to_json_safe())
        logger.info(
            "[EVAL] trial=%s policy=%s stage=evidence eligible=%s failure_reasons=%s",
            trial_number,
            policy_id,
            evidence.eligible,
            list(evidence.failure_reasons),
        )
        if not evidence.eligible:
            continue
        cell_key = (
            evidence.bootstrap_lower_bound,
            evidence.dsr_probability,
            evidence.geometric_excess_growth,
            -evidence.max_drawdown,
            -evidence.turnover,
            evidence.median_rank_ic,
            -route.horizon,
            -trial_number,
            policy_id,
        )
        candidate_rows.append((cell_key, (trial_number, policy_id)))
        current_best = best_cell_by_trial.get(trial_number)
        if current_best is None or cell_key > current_best[0]:
            best_cell_by_trial[trial_number] = (cell_key, evidence)
        logger.info(
            "[EVAL] trial=%s policy=%s stage=economically_eligible "
            "compounding_lower_bound=%.8f dsr_probability=%.6f",
            trial_number,
            policy_id,
            evidence.bootstrap_lower_bound,
            evidence.dsr_probability,
        )
        gc.collect()
    refit_seconds = time.perf_counter() - refit_started
    logger.info(
        "[SYS] route=%sd stage=shortlist elapsed_ms=%.1f rss=%.1f",
        route.horizon,
        refit_seconds * 1000.0,
        TrialResourceGuard._rss_mib(),
    )
    if timings is not None:
        timings.economic_replay_seconds = replay_seconds
    selection_tail = _selection_telemetry(
        replay_seconds, timings, lgb_threads,
        early_rejected_seconds, shortlist_evidence, fold0_ranked, finalists,
        replay_guard=replay_guard,
        total_terminal_screen_trials=total_terminal_screen_trials,
        route_terminal_screen_trials=route_terminal_screen_trials,
        exact_compounding_policy_replays=exact_compounding_policy_replays,
        configured_compounding_policy_cells=configured_compounding_policy_cells,
        selection_multiplicity_version=SELECTION_MULTIPLICITY_VERSION,
    )
    if not candidate_rows:
        return None, {
            "economically_eligible_trials": 0,
            **selection_tail,
        }
    candidate_rows.sort(key=lambda row: row[0], reverse=True)
    _winner_key, (winner_number, winner_policy_id) = candidate_rows[0]
    champion = _config_from_params(
        dict(study.trials[winner_number].params), num_threads=lgb_threads
    )
    _winner_cell_key, winner_evidence = best_cell_by_trial[int(winner_number)]
    winner_calibration_state = winner_evidence.calibration_state
    if winner_calibration_state is not None:
        champion._calibration_state = dict(winner_calibration_state)
    selected_growth_risk_aversion = _DEFAULT_GROWTH_RISK_AVERSION
    selected_turnover_budget = _DEFAULT_TURNOVER_BUDGET
    return champion, {
        "economically_eligible_trials": len(candidate_rows),
        "selected_trial_number": winner_number,
        "selected_policy_id": winner_policy_id,
        "selected_growth_risk_aversion": float(selected_growth_risk_aversion),
        "selected_turnover_budget": float(selected_turnover_budget),
        "selected_inner_compounding_lower_bound": winner_evidence.bootstrap_lower_bound,
        "selected_inner_dsr_probability": winner_evidence.dsr_probability,
        "selected_inner_geometric_excess_growth": winner_evidence.geometric_excess_growth,
        "selected_inner_max_drawdown": winner_evidence.max_drawdown,
        "selected_inner_turnover": winner_evidence.turnover,
        "selected_inner_median_rank_ic": winner_evidence.median_rank_ic,
        "selected_inner_strategy_ir": winner_evidence.strategy_ir,
        "selected_inner_holding_horizon_sessions": winner_evidence.holding_horizon_sessions,
        "selected_calibration_state": winner_calibration_state,
        "compounding_policy_replays": shortlist_evidence,
        **selection_tail,
    }


def _selection_telemetry(
    replay_seconds: float,
    timings: _SelectionTimings | None,
    lgb_threads: int,
    early_rejected_seconds: list[float],
    shortlist_evidence: list[dict[str, object]],
    fold0_ranked: list[tuple[float, float, int, LambdaRankConfig, pl.DataFrame]],
    finalists: list[tuple[list[float], pl.DataFrame, int]],
    *,
    replay_guard: ReplayResourceGuard | None,
    total_terminal_screen_trials: int = 0,
    route_terminal_screen_trials: int = 0,
    exact_compounding_policy_replays: int = 0,
    configured_compounding_policy_cells: int = 0,
    selection_multiplicity_version: str = "",
) -> dict[str, object]:
    """Route-qualified exclusive stage telemetry for one selection funnel.

    ``full_refit_seconds`` excludes replay preparation and replay execution by
    construction: it is the sum of the exclusive per-fold train and predict
    durations, so the overlapping legacy timer double-count is removed.
    """
    train_seconds = timings.refit_train_seconds if timings is not None else 0.0
    predict_seconds = timings.refit_predict_seconds if timings is not None else 0.0
    best_iterations: dict[str, int | None] = (
        timings.actual_best_iterations if timings is not None else {}
    )
    return {
        "full_refit_boosting_rounds": (
            _SCREEN_BOOSTING_ROUNDS + 2 * _SCREEN_EARLY_STOPPING_ROUNDS
        ),
        "full_refit_early_stopping_rounds": 2 * _SCREEN_EARLY_STOPPING_ROUNDS,
        "early_rejected_full_refits": len(early_rejected_seconds),
        "early_rejected_full_refit_seconds": round(
            float(sum(early_rejected_seconds)), 3
        ),
        "shortlist_candidate_evidence": shortlist_evidence,
        "replay_resource": (
            replay_guard.telemetry() if replay_guard is not None else {}
        ),
        "promoted_trials": len(fold0_ranked),
        "all_positive_finalists": len(finalists),
        "context_prepare_seconds": (
            round(timings.context_prepare_seconds, 3) if timings is not None else 0.0
        ),
        "refit_train_seconds": round(train_seconds, 3),
        "refit_predict_seconds": round(predict_seconds, 3),
        "replay_prepare_seconds": (
            round(timings.replay_prepare_seconds, 3) if timings is not None else 0.0
        ),
        "economic_replay_seconds": round(replay_seconds, 3),
        "full_refit_seconds": round(train_seconds + predict_seconds, 3),
        "resolved_lgb_threads": int(lgb_threads),
        "actual_refit_rounds": (
            dict(timings.actual_refit_rounds) if timings is not None else {}
        ),
        "actual_best_iterations": {
            key: (int(value) if value is not None else None)
            for key, value in best_iterations.items()
        },
        "total_terminal_screen_trials": int(total_terminal_screen_trials),
        "route_terminal_screen_trials": int(route_terminal_screen_trials),
        "exact_compounding_policy_replays": int(exact_compounding_policy_replays),
        "configured_compounding_policy_cells": int(
            configured_compounding_policy_cells
        ),
        "selection_multiplicity_version": str(selection_multiplicity_version),
        "fold_retention_telemetry": (
            dict(timings.fold_telemetry) if timings is not None else {}
        ),
    }


def _config_from_trial(
    trial: optuna.Trial,
    num_threads: int | None = None,
) -> LambdaRankConfig:
    return _config_from_params(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 500, 5000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", (127, 255)),
        },
        num_threads=num_threads,
    )


def _config_from_params(
    params: dict[str, Any],
    num_threads: int | None = None,
) -> LambdaRankConfig:
    return LambdaRankConfig(
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=float(params["feature_fraction"]),
        bagging_fraction=float(params["bagging_fraction"]),
        lambda_l1=float(params["lambda_l1"]),
        lambda_l2=float(params["lambda_l2"]),
        max_bin=int(params["max_bin"]),
        num_threads=num_threads or 1,
    )


def _fit_and_score_folds(
    panel: pl.DataFrame,
    folds: list[Fold],
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig,
) -> tuple[list[LambdaRankBlendModel], list[pl.DataFrame], list[float]]:
    fold_models: list[LambdaRankBlendModel] = []
    scored_frames: list[pl.DataFrame] = []
    fold_rank_ic: list[float] = []
    allowlist = stock_alpha_v2_allowlist()
    for fold in folds:
        train_frame = panel[fold.train_mask]
        validation_frame = panel[fold.validation_mask]
        quantiles = fit_v2_winsor_quantiles(train_frame, feature_columns)
        train_processed = apply_v2_transforms(
            train_frame, feature_columns, winsor_quantiles=quantiles
        )
        validation_processed = apply_v2_transforms(
            validation_frame, feature_columns, winsor_quantiles=quantiles
        )
        model = LambdaRankBlendModel(
            base_manifest,
            allowlist,
            label_column,
            config=config,
            session_column="session",
            relevance_column=relevance_column or RELEVANCE_COLUMN,
        )
        try:
            model.fit(train_processed, validation_processed)
        except ValueError:
            logger.info("fold model could not fit; fold skipped")
            continue
        predict_input = _drop_target_columns(validation_processed, label_column)
        scored = model.predict(predict_input)
        fold_models.append(model)
        scored_frames.append(scored)
        fold_rank_ic.append(_median_rank_ic(validation_frame, scored, label_column))
    return fold_models, scored_frames, fold_rank_ic


def _event_ledger_evaluation(
    panel: pl.DataFrame,
    oos_scored: pl.DataFrame,
    request: TrainingRequest,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    *,
    replay_context: ReplayStaticContext | None = None,
    calibration_ledger: pl.DataFrame | None = None,
    holding_horizon_sessions: int = 5,
    label_column: str = "residual_o2o_5d",
    label_available_column: str = "label_available_time",
    replay_mode: ReplayMode = ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
    replay_guard: ReplayResourceGuard | None = None,
    prepared_route: PreparedSelectionRoute | None = None,
) -> ReplayResult:
    """Replay the out-of-sample scored panel through the event-driven backtester.

    A scored planner constructs constrained target allocations directly from the
    frozen fold predictions, so promotion metrics come from the same event
    ledger used by paper/live paths without needing a pre-published artifact.
    When ``replay_context`` is supplied its compact market index, risk policy,
    and instruments are reused instead of being rebuilt; only ``pred_score``
    is joined per replay. The policy's ``rebalance_frequency_sessions`` (built
    from ``holding_horizon_sessions``) drives the decision cadence, and the
    causal calibrator is bound to the route's label and availability columns so
    a 10/15-day route is cost-amortized over its own horizon. The replay-window
    causal 20-session ADTV is computed once and passed to the backtester so the
    base and stress execution ledgers reuse the same validated column instead of
    recomputing it twice.

    ``replay_mode`` selects the evidence scope: inner shortlist selection runs
    only a base-cost ledger (``INNER_SELECTION_BASE_ONLY``), while final
    promotion runs the paired base/stress replay with one prepared decision per
    decision timestamp (``FINAL_PROMOTION_BASE_AND_STRESS``). Every replay
    materialization is admitted through ``replay_guard`` when supplied, and an
    unsafe admission or observed RSS breach raises ``TrainingCapacityError``.
    """
    from src.core.portfolio import PortfolioSnapshot
    from src.stocks.backtesting.engine import (
        ArtifactSchedule,
        ArtifactSlot,
        BacktestRequest,
        PreparedReplayDecision,
        PreparedReplayMarket,
        StockBacktester,
    )
    from src.stocks.data.costs import CostEvidence
    from src.stocks.trading.portfolio_constructor import construct_target_allocations
    from src.stocks.workflows.trading_cycle import (
        CycleStatus,
        TradingCycleRequest,
        TradingCycleResult,
        _build_intents,
    )

    if replay_context is None:
        replay_context = _prepare_replay_static_context(
            panel, request, holding_horizon_sessions=holding_horizon_sessions,
            guard=replay_guard,
        )
    frame = replay_context.market_index
    instruments = replay_context.instruments
    policy = replay_context.policy
    policy.compounding_evidence.clear()

    if prepared_route is not None:
        overlay, allocation_overlay = prepared_route.scatter_overlays(oos_scored)
        if replay_guard is not None:
            replay_guard.admit(
                int(oos_scored.estimated_size()), stage="candidate_overlay"
            )
        sessions = prepared_route.sessions
        decision_indices = prepared_route.decision_indices
        decision_times = tuple(sessions[i] for i in decision_indices)
        start_time = sessions[0]
        end_time = sessions[-1]
        cadence = max(1, int(policy.rebalance_frequency_sessions))
        replay_frame = panel.filter(pl.col("session") >= start_time)
        if replay_guard is not None:
            replay_guard.admit(
                int(replay_frame.estimated_size()), stage="replay_adtv"
            )
        market = prepared_route.market
    else:
        overlay = np.zeros(0, dtype=np.float64)
        if replay_guard is not None:
            replay_guard.admit(
                int(oos_scored.estimated_size()), stage="candidate_score_join"
            )
        scored_for_replay = frame.join(
            oos_scored.select("instrument_id", "session", "pred_score"),
            on=["instrument_id", "session"],
            how="left",
        )
        if replay_guard is not None:
            replay_guard.check_after(stage="candidate_score_join")

        scored_sessions = sorted(
            scored_for_replay.filter(pl.col("pred_score").is_not_null())["session"].unique().to_list()
        )
        if not scored_sessions:
            raise ValueError("scored OOS panel exposes no scored session")
        replay_frame = frame.filter(pl.col("session") >= scored_sessions[0])
        sessions = tuple(
            _session_as_datetime(s)
            for s in sorted(replay_frame["session"].unique().to_list())
        )
        if replay_guard is not None:
            replay_guard.admit(
                int(replay_frame.estimated_size()), stage="replay_adtv"
            )

        start_time = sessions[0]
        end_time = sessions[-1]
        cadence = max(1, int(policy.rebalance_frequency_sessions))
        decision_indices = tuple(
            i for i in range(len(sessions)) if i % cadence == 0
        )
        decision_times = tuple(sessions[i] for i in decision_indices)

    calibrator = (
        CausalAlphaCalibrator(
            bucket_count=request.calibration_bucket_count,
            min_calibration_sessions=request.min_calibration_sessions,
            seed=request.seed,
            n_bootstrap=request.n_bootstrap,
            bootstrap_alpha=request.bootstrap_alpha,
            label_column=label_column,
            label_available_column=label_available_column,
        )
        if calibration_ledger is not None and not calibration_ledger.is_empty()
        else None
    )
    calibration_schedule: CausalCalibrationSchedule | None = None
    session_cluster_schedule: SessionClusterCalibrationSchedule | None = None
    if calibrator is not None:
        assert calibration_ledger is not None
        if replay_mode is ReplayMode.INNER_SELECTION_BASE_ONLY:
            session_cluster_schedule = SessionClusterCalibrationSchedule.build(
                calibration_ledger,
                decision_times,
                calibrator,
                base_schedule,
                block_length=holding_horizon_sessions,
                max_workspace_bytes=_bootstrap_workspace_bytes(
                    request.n_bootstrap, int(calibration_ledger.height)
                ),
            )
        else:
            calibration_schedule = CausalCalibrationSchedule.build(
                calibration_ledger,
                decision_times,
                calibrator,
                base_schedule,
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
        """Frozen calibration state at a decision, or ``None`` without a ledger."""
        if calibrator is None:
            return None
        assert calibration_ledger is not None
        workspace_cap: int | None = None
        active_schedule = (
            session_cluster_schedule
            if session_cluster_schedule is not None
            else calibration_schedule
        )
        if replay_guard is not None:
            assert active_schedule is not None
            prefix_rows = active_schedule.eligible_prefix_rows(decision_time)
            if prefix_rows > 0:
                workspace_cap = replay_guard.bootstrap_workspace_cap(
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
                base_schedule,
                max_bootstrap_workspace_bytes=workspace_cap,
            )
        calibration_tracker["state"] = calibrator.calibration_state()
        return state

    def _prepare_decision(
        decision_time: datetime,
        execution_time: datetime,
    ) -> PreparedReplayDecision:
        """Prepare immutable market/planning inputs once for a decision."""
        if replay_guard is not None:
            replay_guard.record_prepared_decision()
        try:
            if prepared_route is not None:
                allocation_decision_index = prepared_route.allocation_decision_index_for(
                    decision_time
                )
                if allocation_decision_index is None:
                    return PreparedReplayDecision(
                        decision_time, execution_time, pl.DataFrame(),
                        calibration_state=None, reason="no-decision-session",
                    )
                visible = prepared_route.window_frame(
                    allocation_decision_index, allocation_overlay
                )
            else:
                visible = _bounded_replay_history(
                    scored_for_replay, decision_time, policy
                )
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
            calibration_state = _state_at(decision_time, visible)
        except ValueError as exc:
            return PreparedReplayDecision(
                decision_time, execution_time, pl.DataFrame(),
                calibration_state=None, reason=f"constraint:{exc}",
            )
        if prepared_route is None and calibration_state is not None:
            visible = CausalAlphaCalibrator.apply_prepared(calibration_state, visible)
        return PreparedReplayDecision(
            decision_time, execution_time, visible,
            calibration_state=calibration_state,
        )

    def _scenario_planner(
        prepared: PreparedReplayDecision,
        portfolio: PortfolioSnapshot,
        cycle_request: TradingCycleRequest,
    ) -> TradingCycleResult:
        """Construct scenario-specific allocations against the prepared inputs."""
        if prepared.reason is not None:
            return _scored_no_trade(portfolio, cycle_request, prepared.reason)
        try:
            if prepared_route is not None:
                allocation_decision_index = prepared_route.allocation_decision_index_for(
                    prepared.decision_time
                )
                if allocation_decision_index is None:
                    return _scored_no_trade(
                        portfolio, cycle_request, "no-decision-session"
                    )
                allocations = construct_target_allocations_prepared(
                    prepared_route.allocation_market,
                    allocation_decision_index,
                    allocation_overlay,
                    prepared.calibration_state,
                    instruments,
                    portfolio,
                    policy,
                )
            else:
                visible = prepared.visible
                if visible.is_empty():
                    return _scored_no_trade(
                        portfolio, cycle_request, "empty-scored-cross-section"
                    )
                allocations = construct_target_allocations(
                    visible, instruments, portfolio, policy
                )
        except ValueError as exc:
            return _scored_no_trade(portfolio, cycle_request, f"constraint:{exc}")
        if not allocations:
            return _scored_no_trade(portfolio, cycle_request, "no-feasible-allocation")
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

    initial_portfolio = PortfolioSnapshot(
        account_snapshot_id="promotion",
        as_of=datetime(2000, 1, 1, tzinfo=UTC),
        settled_cash=request.initial_cash,
        unsettled_cash=0.0,
        positions=(),
    )
    backtest_request = BacktestRequest(
        strategy_id=request.artifact_id,
        start_time=start_time,
        end_time=end_time,
        decision_session_indices=decision_indices,
        cost_schedule=base_schedule,
        stress_cost_schedule=stress_schedule,
        risk_policy=policy,
        seed=request.seed,
    )
    artifacts = ArtifactSchedule(
        slots=(
            ArtifactSlot(
                eligible_from=start_time,
                eligible_to=end_time,
                artifact_id=request.artifact_id,
            ),
        )
    )
    evidence: CostEvidence | None = getattr(request, "cost_evidence", None)
    if replay_mode is ReplayMode.INNER_SELECTION_BASE_ONLY:
        backtester = StockBacktester(
            registry=registry,
            instruments=instruments,
            manifest=dataset_manifest,
            cost_schedule=base_schedule,
            stress_cost_schedule=None,
            cost_evidence=evidence,
            seed=request.seed,
            decision_provider=_prepare_decision,
            scenario_planner=_scenario_planner,
        )
    else:
        backtester = StockBacktester(
            registry=registry,
            instruments=instruments,
            manifest=dataset_manifest,
            cost_schedule=base_schedule,
            stress_cost_schedule=stress_schedule,
            cost_evidence=evidence,
            seed=request.seed,
            decision_provider=_prepare_decision,
            scenario_planner=_scenario_planner,
        )
    if prepared_route is not None:
        score_overlay = overlay
    else:
        market = PreparedReplayMarket.build(
            replay_frame,
            backtester.adtv_window,
            instruments=instruments,
            artifacts=artifacts,
            initial_portfolio=initial_portfolio,
        )
        score_overlay = (
            replay_frame.sort(["session", "instrument_id"])
            .select("instrument_id", "session")
            .join(
                scored_for_replay.select("instrument_id", "session", "pred_score"),
                on=["instrument_id", "session"],
                how="left",
            )["pred_score"]
            .to_numpy()
            .astype(np.float64)
        )
    result = backtester.run_prepared(backtest_request, market, score_overlay)
    if replay_guard is not None:
        replay_guard.check_after(stage="replay")
        replay_guard.record_stage("replay", 0.0)
    benchmark = _benchmark_return_series(replay_frame)
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
    if calibration_state is not None and isinstance(calibration_state, dict):
        buckets = calibration_state.get("buckets") or []
        net_alphas = [
            float(row["expected_active_alpha"]) - float(calibration_state["round_trip_cost"])
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
    else:
        calibration_evidence = {}
    if session_cluster_schedule is not None:
        calibration_evidence["bootstrap_unit"] = "session_cluster"
        calibration_evidence.update(session_cluster_schedule.telemetry())
    replay_resource: dict[str, object] = {}
    if replay_guard is not None:
        replay_resource = replay_guard.telemetry()
        if session_cluster_schedule is not None:
            replay_resource.update(session_cluster_schedule.telemetry())
    compounding_overlay = _compounding_overlay_summary(
        list(policy.compounding_evidence),
        include_records=replay_mode is not ReplayMode.INNER_SELECTION_BASE_ONLY,
    )
    return ReplayResult(
        ledger=tuple(result.ledger),
        trades=tuple(result.trades),
        metrics=result.metrics,
        stress_metrics=result.stress_metrics,
        final_value=result.final_value,
        excess_returns=excess,
        benchmark_returns=benchmark,
        strategy_returns=strategy_returns,
        base_total_return=(
            (result.final_value - initial_cash) / initial_cash if initial_cash > 0 else 0.0
        ),
        stress_total_return=stress_total,
        benchmark_total_return=benchmark_total,
        planned_cycles=result.planned_cycles,
        attempted_orders=result.attempted_orders,
        filled_orders=result.filled_orders,
        no_trade_reason_counts=no_trade_reason_counts,
        unfilled_order_reason_counts=dict(result.unfilled_order_reason_counts),
        calibration_evidence=calibration_evidence,
        replay_mode=replay_mode.value,
        replay_resource=replay_resource,
        prepared_decision_count=backtester.prepared_decision_count,
        decision_boundaries=list(decision_indices),
        holding_horizon_sessions=int(holding_horizon_sessions),
        compounding_overlay=compounding_overlay,
    )


def _bounded_replay_history(
    scored: pl.DataFrame,
    decision_time: datetime,
    policy: StockRiskPolicy,
) -> pl.DataFrame:
    """Select the latest scored cross-section plus bounded point-in-time history.

    Returns at most ``max(volatility_lookback_sessions,
    covariance_lookback_sessions) + 1`` sessions ending at the latest scored
    session at or before ``decision_time``. Historical rows before the first
    OOS score keep ``pred_score = null``; the selection cross-section remains
    the current frozen OOS score. No row after the decision is included.
    Raises ``ValueError`` for a missing ``session``/``pred_score`` column or
    when no scored cross-section exists at or before the decision time.
    """
    if "session" not in scored.columns or "pred_score" not in scored.columns:
        raise ValueError("scored frame must carry session and pred_score columns")
    visible = scored.filter(pl.col("session") <= decision_time)
    latest_scored = visible.filter(pl.col("pred_score").is_not_null())
    if latest_scored.is_empty():
        raise ValueError(
            f"no scored cross-section at or before decision_time={decision_time}"
        )
    decision_session = latest_scored["session"].max()
    window = (
        max(policy.volatility_lookback_sessions, policy.covariance_lookback_sessions)
        + 1
    )
    window_sessions = (
        visible.filter(pl.col("session") <= decision_session)
        .select(pl.col("session").unique())
        .sort("session", descending=True)
        .head(window)
    )
    return visible.join(window_sessions, on="session", how="inner")


def _session_as_datetime(session: object) -> datetime:
    if isinstance(session, datetime):
        return session
    return datetime.combine(cast(date, session), datetime.min.time(), tzinfo=UTC)


def _decision_time_at(
    market: object,
    session_index: int,
) -> datetime:
    """Max point-in-time ``available_time`` at a prepared-market session.

    Mirrors ``StockBacktester._prepared_decision_time`` so the prepared route's
    decision-time map resolves exactly the decision timestamps the engine passes
    to its decision provider.
    """
    from src.stocks.backtesting.engine import PreparedReplayMarket

    if not isinstance(market, PreparedReplayMarket):
        raise TypeError("decision-time lookup requires a PreparedReplayMarket")
    start, stop = market.session_ranges[session_index]
    values = [
        value
        for value in market.available_time[start:stop]
        if value is not None
    ]
    if not values:
        raise ValueError("no available_time at decision session")
    return max(cast(list[datetime], values))


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


def _strategy_return_series(ledger: list[BacktestLedgerRow]) -> list[float]:
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


def _instruments_from_frame(frame: pl.DataFrame) -> dict[str, Instrument]:
    return {
        str(row["instrument_id"]): Instrument(
            instrument_id=str(row["instrument_id"]),
            asset_kind=AssetKind.STOCK,
            exchange="KRX",
            symbol=str(row["instrument_id"]).split(":")[-1],
            currency="KRW",
        )
        for row in frame.select("instrument_id").unique().iter_rows(named=True)
    }


def _drop_target_columns(
    frame: pl.DataFrame,
    label_column: str | None = None,
) -> pl.DataFrame:
    drops = [
        c
        for c in frame.columns
        if c.startswith(("target_", "label_", "residual_o2o_", "relevance_"))
        or c in (label_column, RELEVANCE_COLUMN, LABEL_AVAILABLE_COLUMN)
    ]
    return frame.drop(drops)


def _build_calibration_ledger(
    oos_scored: pl.DataFrame,
    panel: pl.DataFrame,
    label_column: str,
    label_available_column: str = LABEL_AVAILABLE_COLUMN,
) -> pl.DataFrame:
    """Join OOS predictions with their point-in-time labels for calibration.

    The ledger carries ``(session, instrument_id, score, label,
    label_available_column)`` so ``CausalAlphaCalibrator`` can consume only
    prior label-available OOS observations. ``panel`` is the parent frame
    carrying the route's label columns; every ledger row is a real historical
    OOS score.
    """
    if label_column not in panel.columns:
        raise ValueError(f"panel has no calibration label column {label_column!r}")
    if label_available_column not in panel.columns:
        raise ValueError(
            f"panel has no {label_available_column!r} for calibration"
        )
    required = ("session", "instrument_id")
    if oos_scored.is_empty() or not all(
        c in oos_scored.columns for c in required
    ):
        return pl.DataFrame(
            schema={
                "session": pl.Datetime("us", "UTC"),
                "instrument_id": pl.Utf8,
                "score": pl.Float64,
                label_column: pl.Float64,
                label_available_column: pl.Datetime("us", "UTC"),
            }
        )
    if "pred_score" not in oos_scored.columns:
        raise ValueError("scored frame must carry pred_score for calibration")
    return oos_scored.select("session", "instrument_id", "pred_score").join(
        panel.select(
            "session",
            "instrument_id",
            label_column,
            label_available_column,
        ),
        on=["session", "instrument_id"],
        how="inner",
    ).rename({"pred_score": "score"})


def _median_rank_ic(
    labeled: pl.DataFrame,
    scored: pl.DataFrame,
    label_column: str,
) -> float:
    sub = labeled.select(
        pl.col("session"),
        pl.col("instrument_id"),
        pl.col(label_column),
    ).join(
        scored.select("session", "instrument_id", "pred_score"),
        on=["session", "instrument_id"],
    ).filter(
        pl.col(label_column).is_not_null() & pl.col("pred_score").is_not_null()
    )
    if sub.is_empty() or "session" not in sub.columns:
        return 0.0
    ics: list[float] = []
    for rows in sub.sort("session").partition_by("session"):
        scores = rows["pred_score"].to_numpy().astype(float)
        labels = rows[label_column].to_numpy().astype(float)
        if len(scores) < 2 or np.std(scores) == 0.0 or np.std(labels) == 0.0:
            continue
        rs = np.argsort(np.argsort(scores)) - np.argsort(np.argsort(scores)).mean()
        rl = np.argsort(np.argsort(labels)) - np.argsort(np.argsort(labels)).mean()
        denom = math.sqrt(float(np.sum(rs * rs)) * float(np.sum(rl * rl)))
        ics.append(float(np.sum(rs * rl) / denom) if denom > 0.0 else 0.0)
    return float(np.median(ics)) if ics else 0.0


def _spearman_concordance(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float:
    """Spearman rank concordance between two paired series, ``0.0`` on failure.

    Used as a diagnostic between the proxy minimum economic lower bound and the
    exact replay lower bound. Fewer than two valid pairs, or zero rank variance
    in either series, fails closed to ``0.0``.
    """
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=False)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return 0.0
    xs_arr = np.asarray([p[0] for p in pairs], dtype=float)
    ys_arr = np.asarray([p[1] for p in pairs], dtype=float)
    rx = np.argsort(np.argsort(xs_arr)).astype(float)
    ry = np.argsort(np.argsort(ys_arr)).astype(float)
    denom = math.sqrt(float(np.sum((rx - rx.mean()) ** 2) * np.sum((ry - ry.mean()) ** 2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum((rx - rx.mean()) * (ry - ry.mean())) / denom)


def _reject_non_finite_economic_inputs(frame: pl.DataFrame) -> None:
    for column in _ECONOMIC_COLUMNS:
        if column in frame.columns:
            non_finite = frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
            if not non_finite.is_empty():
                raise ValueError(f"non-finite economic input in {column}")


def _compounding_evidence(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    decision_boundaries: list[int],
    holding_horizon_sessions: int,
    request: TrainingRequest,
    budget: PromotionRiskBudget,
    n_trials: int,
) -> CompoundingEvidence:
    """Build holding-period-consistent compounding evidence.

    The aligned daily return series are split into complete route-length
    intervals of ``holding_horizon_sessions`` consecutive returns anchored at
    ``decision_boundaries`` (defaulting to a uniform cadence from index zero).
    For every complete interval the block log-excess wealth difference
    ``sum(log1p(strategy)) - sum(log1p(benchmark))`` is computed; an interval
    with a non-finite value or a return ``<= -1`` is rejected, not clipped.
    Incomplete leading/trailing intervals are dropped and fewer than three
    complete blocks fail closed to a zero lower bound and zero DSR probability.
    The seeded moving-block bootstrap (block length two rebalances) yields the
    lower bound, and Deflated Sharpe is applied to the same block series.
    """
    horizon = max(1, int(holding_horizon_sessions))
    common = min(len(strategy_returns), len(benchmark_returns))
    boundaries = sorted(
        int(boundary)
        for boundary in decision_boundaries
        if 0 <= int(boundary) < common
    )
    if not boundaries:
        boundaries = list(range(0, common - horizon + 1, horizon))
    blocks: list[float] = []
    rejected = 0
    for start in boundaries:
        if start + horizon > common:
            continue
        strat = strategy_returns[start : start + horizon]
        bench = benchmark_returns[start : start + horizon]
        strat_simple = [math.expm1(float(value)) for value in strat]
        bench_simple = [math.expm1(float(value)) for value in bench]
        if (
            not all(math.isfinite(float(value)) for value in strat_simple)
            or not all(math.isfinite(float(value)) for value in bench_simple)
            or any(value <= -1.0 for value in strat_simple)
            or any(value <= -1.0 for value in bench_simple)
        ):
            rejected += 1
            continue
        block_excess = sum(math.log1p(value) for value in strat_simple) - sum(
            math.log1p(value) for value in bench_simple
        )
        blocks.append(float(block_excess))
    if len(blocks) < 3:
        return CompoundingEvidence(
            block_log_excess=[],
            bootstrap_lower_bound=0.0,
            dsr_probability=0.0,
            complete_block_count=0,
            rejected_block_count=rejected,
        )
    lower_bound = _moving_block_bootstrap_lower_bound(
        blocks,
        block_length=2,
        n_bootstrap=max(request.n_bootstrap, 2),
        seed=request.seed,
        alpha=budget.bootstrap_alpha,
    )
    dsr_probability = _deflated_sharpe_probability(
        blocks,
        annualization=252,
        n_trials=n_trials,
    )
    return CompoundingEvidence(
        block_log_excess=blocks,
        bootstrap_lower_bound=lower_bound,
        dsr_probability=dsr_probability,
        complete_block_count=len(blocks),
        rejected_block_count=rejected,
    )


def _compounding_overlay_summary(
    records: Sequence[dict[str, object]],
    *,
    include_records: bool = False,
) -> dict[str, object]:
    """Compact JSON-safe compounding-overlay summary from per-decision records.

    The summary is the single source of truth propagated to the final
    ``metrics`` payload and to every ``(trial, policy)`` inner candidate
    evidence row. Per-decision records are preserved only when
    ``include_records`` is true, i.e. only for the final artifact.
    """
    decision_count = len(records)
    scales = [
        float(cast(float, record["confidence_scale"]))
        for record in records
        if record.get("confidence_scale") is not None
    ]
    cash_reasons = Counter(
        str(record["cash_reason"])
        for record in records
        if record.get("cash_reason") is not None
    )
    gross_before = [
        float(cast(float, record["gross_before_compounding"])) for record in records
    ]
    gross_after = [
        float(cast(float, record["gross_after_compounding"])) for record in records
    ]
    lambdas = [float(cast(float, record["turnover_lambda"])) for record in records]
    summary: dict[str, object] = {
        "decision_count": decision_count,
        "cash_count": sum(cash_reasons.values()),
        "cash_reasons": dict(sorted(cash_reasons.items())),
        "mean_confidence_scale": float(np.mean(scales)) if scales else 0.0,
        "p10_confidence_scale": (
            float(np.percentile(scales, 10)) if scales else 0.0
        ),
        "positive_scale_fraction": (
            sum(1 for scale in scales if scale > 0.0) / max(decision_count, 1)
        ),
        "mean_gross_before_compounding": (
            float(np.mean(gross_before)) if gross_before else 0.0
        ),
        "mean_gross_after_compounding": (
            float(np.mean(gross_after)) if gross_after else 0.0
        ),
        "mean_turnover_lambda": float(np.mean(lambdas)) if lambdas else 0.0,
    }
    if include_records:
        summary["records"] = [dict(record) for record in records]
    return summary


def _moving_block_bootstrap_lower_bound(
    values: list[float],
    block_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    block = max(block_length, 1)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < arr.size:
            start = int(rng.integers(0, max(1, arr.size - block + 1)))
            sample.extend(arr[start : start + block])
        means[b] = float(np.mean(sample[: arr.size]))
    return float(np.quantile(means, alpha))


def _economic_screen_score(
    labeled: pl.DataFrame,
    scored: pl.DataFrame,
    *,
    label_column: str,
    top_k: int,
    holding_horizon_sessions: int,
    cost_schedule: CostSchedule,
    n_bootstrap: int,
    bootstrap_alpha: float,
    seed: int,
) -> float:
    """Cost-aware non-overlapping proxy compounding lower bound for one fold.

    Joins the validation labels ``(session_index, session, instrument_id,
    label_column)`` to the finite model predictions, greedily retains decision
    sessions at least ``holding_horizon_sessions`` apart so forward label
    windows never overlap, then for each retained decision selects the top
    ``min(top_k, available_names)`` names at equal weight. One-way portfolio
    turnover is half the L1 distance from the prior equal weights (the first
    decision has turnover ``1.0``), the effective stress cost point is resolved
    at the decision time, and each block's net log excess is
    ``mean(label) - turnover * round_trip_rate`` with
    ``round_trip_rate = 2*commission + tax + 2*slippage_bps/10000``. The
    seeded moving-block bootstrap (block length two rebalances) lower bound is
    returned; fewer than three valid blocks, a non-finite block, or an invalid
    group returns ``0.0`` fail-closed. Missing identity/session/label/score
    columns and invalid scalar arguments raise ``ValueError``.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if holding_horizon_sessions < 1:
        raise ValueError("holding_horizon_sessions must be positive")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if not 0.0 < bootstrap_alpha < 1.0:
        raise ValueError("bootstrap_alpha must be in (0, 1)")
    required = ("session", "instrument_id")
    for column in required:
        if column not in labeled.columns or column not in scored.columns:
            raise ValueError(
                f"economic screen inputs must carry {column} in both labeled and scored"
            )
    if "session_index" not in labeled.columns:
        raise ValueError("economic screen labeled frame must carry session_index")
    if label_column not in labeled.columns:
        raise ValueError(f"economic screen labeled frame must carry {label_column}")
    if "pred_score" not in scored.columns:
        raise ValueError("economic screen scored frame must carry pred_score")

    joined = labeled.select(
        "session_index", "session", "instrument_id", pl.col(label_column)
    ).join(
        scored.select("session", "instrument_id", "pred_score"),
        on=["session", "instrument_id"],
        how="inner",
    ).filter(
        pl.col(label_column).is_not_null()
        & pl.col("pred_score").is_not_null()
        & pl.col(label_column).is_finite()
        & pl.col("pred_score").is_finite()
    )
    if joined.is_empty():
        return 0.0
    decision_sessions: list[tuple[int, object]] = []
    last_kept: int | None = None
    for row in joined.select("session_index", "session").unique().sort("session_index").iter_rows():
        index = int(row[0])
        if last_kept is None or index - last_kept >= holding_horizon_sessions:
            decision_sessions.append((index, row[1]))
            last_kept = index
    if len(decision_sessions) < 3:
        return 0.0
    blocks: list[float] = []
    previous_names: list[str] = []
    for index, decision_time in decision_sessions:
        cross = joined.filter(pl.col("session_index") == index)
        k = min(top_k, int(cross.height))
        if k < 1:
            continue
        top = cross.sort("pred_score", descending=True).head(k)
        names = top["instrument_id"].to_list()
        if previous_names:
            union = set(previous_names) | set(names)
            turnover = 0.5 * sum(
                abs(
                    (1.0 / k if name in names else 0.0)
                    - (1.0 / len(previous_names) if name in previous_names else 0.0)
                )
                for name in union
            )
        else:
            turnover = 1.0
        previous_names = names
        decision = cost_schedule.cost_for(_session_as_datetime(decision_time))
        round_trip_rate = (
            2.0 * decision.commission_rate
            + decision.tax_rate
            + 2.0 * decision.slippage_bps / 10000.0
        )
        mean_value = top[label_column].mean()
        mean_label = float(cast(float, mean_value)) if mean_value is not None else 0.0
        block = mean_label - turnover * round_trip_rate
        if not math.isfinite(block):
            return 0.0
        blocks.append(block)
    if len(blocks) < 3:
        return 0.0
    return _moving_block_bootstrap_lower_bound(
        blocks,
        block_length=2,
        n_bootstrap=max(n_bootstrap, 2),
        seed=seed,
        alpha=bootstrap_alpha,
    )


def _evaluate_gates(
    replay: ReplayResult,
    fold_rank_ic: list[float],
    budget: PromotionRiskBudget,
    request: TrainingRequest,
    *,
    n_trials: int,
) -> dict[str, object]:
    """Lexicographic fail-closed promotion gates over the event ledger.

    A replay with zero attempted orders is evidence-incomplete and must fail
    promotion instead of being read as a flat strategy. Gate 2 requires a
    strictly positive moving-block bootstrap lower bound of complete
    holding-period block log-excess returns, and Gate 5 evaluates Deflated
    Sharpe on that same block series with the supplied terminal trial count.
    The legacy daily arithmetic excess bootstrap survives only as the
    ``legacy_daily_excess_lower_bound`` diagnostic.
    """
    reasons: list[str] = []
    passed = True

    positive_fraction = (
        sum(1 for ic in fold_rank_ic if ic > 0.0) / len(fold_rank_ic)
        if fold_rank_ic
        else 0.0
    )
    gate1_ok = positive_fraction >= budget.min_positive_refit_fraction
    reasons.append(f"gate1_positive_rank_ic_fraction={positive_fraction:.4f}")
    passed = passed and gate1_ok

    if replay.attempted_orders <= 0:
        passed = False
        reasons.append("evidence_incomplete_no_attempted_orders=true")
    else:
        reasons.append(f"attempted_orders={replay.attempted_orders}")
    if replay.filled_orders <= 0:
        passed = False
        reasons.append("evidence_incomplete_no_filled_orders=true")
    else:
        reasons.append(f"filled_orders={replay.filled_orders}")

    compounding = _compounding_evidence(
        replay.strategy_returns,
        replay.benchmark_returns,
        replay.decision_boundaries,
        replay.holding_horizon_sessions,
        request,
        budget,
        n_trials,
    )
    lower_bound = compounding.bootstrap_lower_bound
    gate2_ok = lower_bound > 0.0
    reasons.append(f"gate2_compounding_lower_bound={lower_bound:.8f}")
    reasons.append(f"gate2_compounding_block_count={compounding.complete_block_count}")
    passed = passed and gate2_ok

    legacy_daily = (
        _moving_block_bootstrap_lower_bound(
            replay.excess_returns,
            max(5, 1),
            max(request.n_bootstrap, 2),
            request.seed,
            budget.bootstrap_alpha,
        )
        if replay.excess_returns
        else 0.0
    )
    reasons.append(f"legacy_daily_excess_lower_bound={legacy_daily:.8f}")

    strategy_returns = replay.strategy_returns
    benchmark_returns = replay.benchmark_returns
    aligned_excess = _aligned_excess(strategy_returns, benchmark_returns)
    active_ir = _active_information_ratio(aligned_excess)
    gate3_ok = math.isfinite(active_ir) and active_ir > 0.0
    reasons.append(f"gate3_active_information_ratio={active_ir:.6f}")
    reasons.append(
        f"gate3_aligned_observations={len(aligned_excess)}"
    )
    passed = passed and gate3_ok

    benchmark_total = replay.benchmark_total_return
    gate4_ok = bool(
        replay.stress_total_return is not None
        and math.isfinite(replay.stress_total_return)
        and math.isfinite(benchmark_total)
        and replay.stress_total_return > benchmark_total
    )
    stress_value = (
        f"{replay.stress_total_return:.8f}"
        if replay.stress_total_return is not None
        else "nan"
    )
    reasons.append(f"gate4_stress_cost_excess={gate4_ok}")
    reasons.append(f"gate4_stress_total_return={stress_value}")
    reasons.append(f"gate4_benchmark_total_return={benchmark_total:.8f}")
    passed = passed and gate4_ok

    deflated_prob = compounding.dsr_probability
    gate5_ok = deflated_prob >= budget.deflated_sharpe_probability
    reasons.append(f"gate5_deflated_sharpe_probability={deflated_prob:.6f}")
    passed = passed and gate5_ok

    strategy_drawdown = float(replay.metrics.get("max_drawdown", 1.0))
    benchmark_drawdown = _drawdown_from_returns(benchmark_returns)
    gate6_ok = (
        benchmark_drawdown <= 0.0
        or strategy_drawdown <= budget.max_benchmark_drawdown_ratio * benchmark_drawdown
    )
    reasons.append(f"gate6_drawdown_ratio={strategy_drawdown:.4f}/{benchmark_drawdown:.4f}")
    passed = passed and gate6_ok

    return {"passed": passed, "reasons": reasons}


def _information_ratio(returns: list[float]) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2 or np.std(arr, ddof=0) <= 0.0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=0)) * math.sqrt(252.0)


def _active_information_ratio(excess_returns: list[float]) -> float:
    """Annualized active information ratio of aligned strategy-minus-benchmark returns.

    Computed as ``mean(excess) / std(excess) * sqrt(252)`` on the aligned
    excess series. Fewer than two observations, zero variance, or a non-finite
    value fails closed to ``0.0`` so the caller's ``> 0`` predicate rejects it.
    """
    arr = np.asarray(excess_returns, dtype=float)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return 0.0
    std = float(np.std(arr, ddof=0))
    if std <= 0.0:
        return 0.0
    return float(np.mean(arr) / std) * math.sqrt(252.0)


def _deflated_sharpe_probability(
    returns: list[float],
    annualization: int,
    n_trials: int,
    *,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Deterministic trial-count-adjusted Deflated Sharpe probability.

    Returns the probability that the observed non-annualized Sharpe exceeds
    the expected maximum Sharpe under a null of ``n_trials`` independent
    zero-mean trials, using the sample skewness and excess kurtosis of the OOS
    return sample. The annualization factor cancels between the observed Sharpe
    and its variance estimate, so the result is annualization-invariant. Empty,
    non-finite, too-short, or zero-variance samples fail closed to 0.0.
    """
    from scipy import stats

    arr = np.asarray(returns, dtype=float)
    if arr.size < 2 or n_trials < 1 or annualization < 1:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return 0.0
    std = float(arr.std(ddof=1))
    if std <= 0.0:
        return 0.0
    sharpe = float(arr.mean() / std)
    if skewness is not None and kurtosis is not None:
        skew = float(skewness)
        kurt = float(kurtosis)
    else:
        skew, kurt = _sample_central_moments(arr)
    variance = (1.0 - skew * sharpe + (kurt / 4.0) * sharpe * sharpe) / (arr.size - 1)
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0
    expected_max_sharpe = _expected_maximum_null_sharpe(n_trials) * math.sqrt(variance)
    return float(stats.norm.cdf((sharpe - expected_max_sharpe) / math.sqrt(variance)))


def _sample_central_moments(arr: np.ndarray) -> tuple[float, float]:
    """Vectorized sample skewness and excess kurtosis without scipy warnings."""
    mean = float(np.mean(arr))
    centered = arr - mean
    variance = float(np.mean(centered * centered))
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0, 0.0
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / variance**1.5
    kurt = m4 / variance**2 - 3.0
    if not (math.isfinite(skew) and math.isfinite(kurt)):
        return 0.0, 0.0
    return skew, kurt


def _expected_maximum_null_sharpe(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` independent standard normal trials."""
    if n_trials < 1:
        return 0.0
    from scipy import stats

    euler_mascheroni = 0.5772156649015329
    inv_n = 1.0 / n_trials
    inv_ne = 1.0 / (n_trials * math.e)
    return float(
        (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - inv_n)
        + euler_mascheroni * stats.norm.ppf(1.0 - inv_ne)
    )


def _drawdown_from_returns(returns: list[float]) -> float:
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / np.where(peaks > 0, peaks, 1.0)
    return float(np.max(dd)) if dd.size else 0.0


def _reserve_forward_holdout(
    panel: pl.DataFrame,
    request: TrainingRequest,
    label_span_sessions: int,
    label_available_column: str = LABEL_AVAILABLE_COLUMN,
) -> tuple[Fold | None, pl.DataFrame]:
    """Reserve the dated forward holdout before tuning and outer folds.

    When the immutable snapshot holds at least the required number of
    label-available sessions on or after ``2026-03-10``, the newest
    ``holdout_sessions`` are pinned as a locked ``PurgedWalkForward.holdout``
    and the returned training panel contains only sessions before that block.
    Otherwise returns ``(None, panel)`` unchanged and promotion stays fail
    closed. ``label_available_column`` is the control route's availability
    column so both legacy five-day and multi-horizon panels reserve the block
    against the right labels.
    """
    holdout_sessions = (
        request.holdout_sessions
        if request.holdout_sessions > 0
        else _FORWARD_HOLDOUT_SESSIONS
    )
    if holdout_sessions < 1 or label_available_column not in panel.columns:
        return None, panel
    holdout_start = datetime.combine(_FORWARD_HOLDOUT_START, datetime.min.time(), tzinfo=UTC)
    post_start_sessions = panel.filter(
        (pl.col("session") >= holdout_start)
        & pl.col(label_available_column).is_not_null()
    )["session_index"].unique().to_list()
    if len(post_start_sessions) < holdout_sessions:
        return None, panel
    splitter = PurgedWalkForward(
        n_folds=1,
        label_horizon_sessions=label_span_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
        min_train_sessions=0,
    )
    fold = splitter.holdout(panel, holdout_sessions)
    block_start = _session_as_datetime(
        panel.filter(
            pl.col("session_index").is_in(fold.validation_mask)
        )["session"].min()
    )
    if block_start < holdout_start:
        return None, panel
    training_panel = panel.filter(pl.col("session_index") < fold.validation_decision_start)
    return fold, training_panel


def _evaluate_forward_holdout(
    registry: ModelArtifactRegistry,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    panel: pl.DataFrame,
    holdout_fold: Fold | None,
    champion_config: LambdaRankConfig,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    dataset_manifest: DatasetManifest,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    *,
    calibration_ledger: pl.DataFrame | None = None,
    label_span_sessions: int = 6,
    label_available_column: str = LABEL_AVAILABLE_COLUMN,
    holding_horizon_sessions: int = 5,
    selected_policy_id: str = "",
    compounding: CompoundingPolicyConfig | None = None,
    turnover_budget: float | None = None,
) -> tuple[bool, str, dict[str, object] | None]:
    """Fit the frozen candidate on pre-holdout data and replay the block once.

    The holdout block is the same newest ``holdout_sessions`` pinned by
    ``_reserve_forward_holdout``, but the fold is rebuilt with the route's
    ``label_span_sessions`` purge/embargo so a 10/15-day route never trains on
    a label that overlaps the holdout decisions. The selected compounding
    policy id and risk fields bind the holdout fingerprint and evidence, and
    the frozen policy is replayed unchanged. Returns ``(ready, reason,
    evidence)``. A candidate fingerprint may inspect the holdout exactly once;
    a reused fingerprint raises ``ValueError``. Incomplete data leaves the
    candidate ``NO_TRADE`` with ``forward_holdout_ready=false``.
    """
    if holdout_fold is None:
        return (
            False,
            "gate8_forward_holdout_ready=false:insufficient-label-available-sessions-on-or-after-2026-03-10",
            None,
        )
    holdout_sessions = (
        request.holdout_sessions
        if request.holdout_sessions > 0
        else _FORWARD_HOLDOUT_SESSIONS
    )
    route_splitter = PurgedWalkForward(
        n_folds=1,
        label_horizon_sessions=label_span_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
        min_train_sessions=0,
    )
    try:
        route_fold = route_splitter.holdout(panel, holdout_sessions)
    except ValueError:
        return (
            False,
            "gate8_forward_holdout_ready=false:insufficient-label-available-sessions-on-or-after-2026-03-10",
            None,
        )
    block_session_indexes = sorted(
        int(v)
        for v in panel["session_index"][route_fold.validation_mask].unique().to_list()
    )
    holdout_session_range = (
        block_session_indexes[0],
        block_session_indexes[-1],
    )
    fingerprint = _forward_holdout_fingerprint(
        base_manifest, request, dataset_manifest, holdout_session_range,
        champion_config, base_schedule, stress_schedule, holding_horizon_sessions,
        selected_policy=selected_policy_id,
    )
    existing = registry.read_forward_holdout(request.artifact_id)
    if existing is not None and existing.get("fingerprint") == fingerprint:
        raise ValueError(
            f"forward holdout for candidate fingerprint {fingerprint!r} "
            f"was already inspected for {request.artifact_id!r}"
        )
    models, scored, _fold_ic = _fit_and_score_folds(
        panel, [route_fold], request, base_manifest, feature_columns, label_column,
        relevance_column, champion_config,
    )
    if not models:
        return False, "gate8_forward_holdout_ready=false:no-fit", None
    holdout_oos = pl.concat(scored)
    replay_guard = ReplayResourceGuard(
        request, replay_mode=ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS
    )
    replay_context = _prepare_replay_static_context(
        panel, request, holding_horizon_sessions=holding_horizon_sessions,
        guard=replay_guard,
        turnover_budget=turnover_budget,
        compounding=compounding,
    )
    prepared_route = PreparedSelectionRoute.build(
        panel,
        [_session_as_datetime(holdout_oos["session"].min())],
        request,
        RouteSpec(
            horizon=holding_horizon_sessions,
            label_column=label_column,
            relevance_column=relevance_column or RELEVANCE_COLUMN,
            label_available_column=label_available_column,
        ),
        guard=replay_guard,
    )
    try:
        replay = _event_ledger_evaluation(
            panel, holdout_oos, request, dataset_manifest, registry,
            base_schedule, stress_schedule, replay_context=replay_context,
            calibration_ledger=calibration_ledger,
            holding_horizon_sessions=holding_horizon_sessions,
            label_column=label_column,
            label_available_column=label_available_column,
            replay_mode=ReplayMode.FINAL_PROMOTION_BASE_AND_STRESS,
            replay_guard=replay_guard,
            prepared_route=prepared_route,
        )
    except TrainingCapacityError:
        return (
            False,
            "gate8_forward_holdout_ready=false:replay-capacity-exceeded",
            None,
        )
    if replay.attempted_orders <= 0:
        return False, "gate8_forward_holdout_ready=false:no-attempted-orders", None
    if replay.filled_orders <= 0:
        return False, "gate8_forward_holdout_ready=false:no-filled-orders", None
    risk_policy: dict[str, object] = {
        name: getattr(request, name)
        for name in (
            "top_k",
            "max_exposure",
            "max_single_weight",
            "participation_limit",
        )
    }
    if turnover_budget is not None:
        risk_policy["turnover_budget"] = float(turnover_budget)
    if compounding is not None:
        risk_policy["growth_risk_aversion"] = float(compounding.growth_risk_aversion)
    evidence: dict[str, object] = {
        "feature_schema_hash": base_manifest.feature_schema_hash,
        "universe_policy_hash": base_manifest.universe_policy_hash,
        "label_dataset_hash": _label_dataset_hash(dataset_manifest),
        "holdout_session_range": holdout_session_range,
        "model_config": _config_snapshot(champion_config),
        "holding_horizon_sessions": holding_horizon_sessions,
        "label_column": label_column,
        "label_available_column": label_available_column,
        "selected_policy_id": selected_policy_id,
        "risk_policy": risk_policy,
        "cost_schedules": (base_schedule.name, stress_schedule.name),
        "seed": request.seed,
        "planned_cycles": replay.planned_cycles,
        "attempted_orders": replay.attempted_orders,
        "filled_orders": replay.filled_orders,
        "base_total_return": replay.base_total_return,
        "stress_total_return": replay.stress_total_return,
        "ledger_metrics": replay.metrics,
    }
    registry.write_forward_holdout(request.artifact_id, fingerprint, evidence)
    return True, "gate8_forward_holdout_ready=true", evidence


def _forward_holdout_fingerprint(
    base_manifest: ModelManifest,
    request: TrainingRequest,
    dataset_manifest: DatasetManifest,
    holdout_session_range: tuple[int, int],
    config: LambdaRankConfig,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
    holding_horizon_sessions: int = 5,
    selected_policy: str = "",
) -> str:
    """SHA-256 identity binding one forward-holdout evaluation to its inputs."""
    config_fields = "|".join(
        f"{name}={value}" for name, value in sorted(_config_snapshot(config).items())
    )
    policy_fields = "|".join(
        f"{name}={getattr(request, name)}"
        for name in ("top_k", "max_exposure", "max_single_weight", "participation_limit")
    )
    key = "|".join(
        (
            base_manifest.feature_schema_hash,
            base_manifest.universe_policy_hash,
            _label_dataset_hash(dataset_manifest),
            f"holdout_range={holdout_session_range[0]}..{holdout_session_range[1]}",
            f"model={config_fields}",
            f"policy={policy_fields}",
            base_schedule.name,
            stress_schedule.name,
            str(request.seed),
            f"holding_horizon_sessions={holding_horizon_sessions}",
            f"selected_policy={selected_policy}",
        )
    )
    return sha256(key.encode("utf-8")).hexdigest()


def _label_dataset_hash(manifest: DatasetManifest) -> str:
    return manifest.content_hash or manifest.schema_hash


def _config_snapshot(config: LambdaRankConfig) -> dict[str, object]:
    return {
        name: value
        for name, value in vars(config).items()
        if not name.startswith("_")
    }


def _no_trade_model(
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig | None,
) -> LambdaRankBlendModel:
    del feature_columns, relevance_column
    return LambdaRankBlendModel(
        base_manifest,
        stock_alpha_v2_allowlist(),
        label_column,
        config=config or LambdaRankConfig(),
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )


def _publish_no_trade(
    registry: ModelArtifactRegistry,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    panel: pl.DataFrame,
    label_column: str,
    relevance_column: str | None,
    reason: str,
    *,
    details: str = "",
    tuning_telemetry: dict[str, object] | None = None,
) -> ModelManifest:
    del panel
    model = _no_trade_model(
        base_manifest,
        stock_alpha_v2_allowlist(),
        label_column,
        relevance_column,
        None,
    )
    published_manifest = model.manifest()
    registry.publish(model, published_manifest)
    registry.write_metrics(
        request.artifact_id,
        {
            "artifact_id": request.artifact_id,
            "model_type": published_manifest.model_type,
            "promoted": False,
            "no_trade": True,
            "n_folds_evaluated": 0,
            "median_rank_ic": 0.0,
            "promotion_reasons": [f"{reason}:{details}".rstrip(":")],
            "ledger_metrics": {},
            "stress_metrics": None,
            "gates": {"passed": False},
            "optuna_trials": request.optuna_trials,
            "resource": tuning_telemetry or {},
            "total_terminal_screen_trials": (tuning_telemetry or {}).get(
                "total_terminal_screen_trials", 0
            ),
            "route_terminal_screen_trials": (tuning_telemetry or {}).get(
                "route_terminal_screen_trials", 0
            ),
            "exact_compounding_policy_replays": (tuning_telemetry or {}).get(
                "exact_compounding_policy_replays", 0
            ),
            "configured_compounding_policy_cells": (tuning_telemetry or {}).get(
                "configured_compounding_policy_cells", 0
            ),
            "selection_multiplicity_version": (tuning_telemetry or {}).get(
                "selection_multiplicity_version", ""
            ),
        },
    )
    logger.info("published NO_TRADE artifact %s (%s)", request.artifact_id, reason)
    return published_manifest


def _build_metrics(
    request: TrainingRequest,
    replay: ReplayResult,
    fold_rank_ic: list[float],
    gates: dict[str, object],
    reasons: list[str],
    manifest: ModelManifest,
    *,
    tuning_telemetry: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "artifact_id": request.artifact_id,
        "model_type": manifest.model_type,
        "promoted": bool(gates["passed"]),
        "no_trade": not bool(gates["passed"]),
        "n_folds_evaluated": len(fold_rank_ic),
        "median_rank_ic": float(np.median(fold_rank_ic)) if fold_rank_ic else 0.0,
        "rank_ic_coverage": sum(1 for ic in fold_rank_ic if ic != 0.0)
        / max(len(fold_rank_ic), 1),
        "ledger_metrics": replay.metrics,
        "stress_metrics": replay.stress_metrics,
        "planned_cycles": replay.planned_cycles,
        "attempted_orders": replay.attempted_orders,
        "filled_orders": replay.filled_orders,
        "no_trade_reason_counts": replay.no_trade_reason_counts,
        "unfilled_order_reason_counts": replay.unfilled_order_reason_counts,
        "base_total_return": replay.base_total_return,
        "stress_total_return": replay.stress_total_return,
        "benchmark_total_return": replay.benchmark_total_return,
        "promotion_reasons": reasons,
        "gates": gates,
        "optuna_trials": request.optuna_trials,
        "resource": tuning_telemetry or {},
        "total_terminal_screen_trials": (tuning_telemetry or {}).get(
            "total_terminal_screen_trials", 0
        ),
        "route_terminal_screen_trials": (tuning_telemetry or {}).get(
            "route_terminal_screen_trials", 0
        ),
        "exact_compounding_policy_replays": (tuning_telemetry or {}).get(
            "exact_compounding_policy_replays", 0
        ),
        "configured_compounding_policy_cells": (tuning_telemetry or {}).get(
            "configured_compounding_policy_cells", 0
        ),
        "selection_multiplicity_version": (tuning_telemetry or {}).get(
            "selection_multiplicity_version", ""
        ),
        "replay_mode": replay.replay_mode,
        "replay_resource": replay.replay_resource or {},
        "prepared_decision_count": replay.prepared_decision_count,
        "compounding_overlay": replay.compounding_overlay,
        "selected_policy_id": (tuning_telemetry or {}).get("selected_policy_id"),
    }
