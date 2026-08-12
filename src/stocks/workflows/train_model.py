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

import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import polars as pl
import psutil
from lightgbm.callback import CallbackEnv

from src.core.costs import CostSchedule, default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind, Instrument
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
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
    RESIDUAL_O2O_LABEL,
)
from src.stocks.research.lambdarank import (
    LambdaRankBlendModel,
    LambdaRankConfig,
    PreparedLambdaRankFold,
)
from src.stocks.research.models import ModelManifest, StableRankComposite
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from src.stocks.workflows.contracts import TrainingRequest

if TYPE_CHECKING:
    from src.stocks.backtesting.engine import BacktestLedgerRow

logger = logging.getLogger("stocks.workflows.train_model")

_ECONOMIC_COLUMNS = ("open", "high", "low", "close", "volume", "trading_value", "market_cap")

_MIN_TRAIN_SESSIONS = 756
_VALIDATION_BLOCK_SESSIONS = 252
_REFIT_EVERY_SESSIONS = 63
_REBALANCE_EVERY_SESSIONS = 5
_FORWARD_HOLDOUT_START = date(2026, 3, 10)
_FORWARD_HOLDOUT_SESSIONS = 252
_MIN_GROUP_SIZE = 20
_BYTES_PER_CELL = 4
_ALLOCATION_MULTIPLE = 3
_SCREEN_BOOSTING_ROUNDS = 800
_SCREEN_EARLY_STOPPING_ROUNDS = 50
_SCREEN_NDCG_WARMUP_ROUNDS = 100
_SCREEN_NDCG_INTERVAL_ROUNDS = 50
_SCREEN_SHORTLIST_SIZE = 8


@dataclass(frozen=True, slots=True)
class PromotionRiskBudget:
    """Versioned risk budget enforced by the promotion gates."""

    min_positive_refit_fraction: float = 0.75
    bootstrap_alpha: float = 0.05
    deflated_sharpe_probability: float = 0.95
    max_benchmark_drawdown_ratio: float = 1.10


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
    calibration_evidence: dict[str, object] = field(default_factory=dict)


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


@dataclass(frozen=True, slots=True)
class _StableTrialContext:
    """Per-tuning-fold inputs invariant across every LambdaRank search parameter."""

    train_processed: pl.DataFrame
    validation_processed: pl.DataFrame
    validation_frame: pl.DataFrame
    stable_scores: pl.DataFrame
    prepared: PreparedLambdaRankFold | None = None


@dataclass(frozen=True, slots=True)
class EconomicCandidateEvidence:
    """Immutable per-candidate economic evidence for one shortlisted trial.

    All fields are frozen selection inputs or full-refit/replay diagnostics so
    the exact failed predicate for every shortlisted trial stays recoverable
    from the published artifact.
    """

    trial_number: int
    screen_rank_ic: float
    fold_rank_ic: list[float]
    median_rank_ic: float
    attempted_orders: int
    filled_orders: int
    planned_cycles: int
    no_trade_reason_counts: dict[str, int]
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

    def to_json_safe(self) -> dict[str, object]:
        """JSON-serializable evidence row with deterministic failure reasons."""
        return {
            "trial_number": int(self.trial_number),
            "screen_rank_ic": round(self.screen_rank_ic, 8),
            "fold_rank_ic": [round(value, 8) for value in self.fold_rank_ic],
            "median_rank_ic": round(self.median_rank_ic, 8),
            "attempted_orders": int(self.attempted_orders),
            "filled_orders": int(self.filled_orders),
            "planned_cycles": int(self.planned_cycles),
            "no_trade_reason_counts": dict(sorted(self.no_trade_reason_counts.items())),
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
        }


@dataclass(frozen=True, slots=True)
class ReplayStaticContext:
    """Immutable point-in-time market inputs shared across economic candidates.

    The market panel, instrument map, and risk policy do not depend on any
    candidate's scores, so they are built once per selection panel and reused
    by every shortlist replay; only ``pred_score`` is candidate-specific and
    joined per replay. ``cache_bytes`` is the estimated resident size of the
    cached market inputs and participates in every resource-guard admit.
    """

    market_panel: pl.DataFrame
    instruments: Mapping[str, Instrument]
    policy: StockRiskPolicy
    cache_bytes: int


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

    label_column = _resolve_label_column(frame, manifest)
    if label_column not in frame.columns:
        raise ValueError(
            f"composed snapshot has no canonical label column {label_column!r}"
        )
    relevance_column = RELEVANCE_COLUMN if RELEVANCE_COLUMN in frame.columns else None
    if relevance_column is None and _will_need_lambdarank(frame):
        raise ValueError(f"composed snapshot has no {RELEVANCE_COLUMN!r} column")

    panel = _index_sessions(frame)
    n_sessions = int(panel["session_index"].n_unique())
    eligible_from, eligible_to = _eligibility_from_panel(panel, manifest.label_horizon_sessions)

    base_manifest = ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v2",
        feature_schema_hash=manifest.schema_hash,
        universe_policy_hash=manifest.universe_policy_hash,
        label_definition=label_column,
        label_horizon_sessions=manifest.label_horizon_sessions or 5,
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="lambdarank_blend",
    )

    if n_sessions < _MIN_TRAIN_SESSIONS:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "insufficient-history",
            details=f"n_sessions={n_sessions}",
        )

    label_span_sessions = (manifest.label_horizon_sessions or 1) + 1
    holdout_fold, training_panel = _reserve_forward_holdout(
        panel, request, label_span_sessions,
    )

    reasons: list[str] = []
    splitter = PurgedWalkForward(
        n_folds=request.n_folds,
        label_horizon_sessions=label_span_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
        validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
        min_train_sessions=_MIN_TRAIN_SESSIONS,
    )
    folds = splitter.split(training_panel)
    if not folds:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-eligible-folds",
        )

    base = request.base_cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()

    champion_config, n_optuna_trials = _tune_champion(
        training_panel, folds, request, base_manifest, feature_columns, label_column,
        relevance_column, label_span_sessions,
        dataset_manifest=snapshot.manifest,
        registry=registry,
        base_schedule=base,
        stress_schedule=stress,
    )
    if champion_config is None:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-champion-trial",
            tuning_telemetry=LambdaRankConfig._tuning_telemetry,
        )

    fold_models, scored_frames, fold_rank_ic = _fit_and_score_folds(
        training_panel, folds, request, base_manifest, feature_columns, label_column,
        relevance_column, champion_config,
    )
    if not fold_models:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-fit-folds",
        )

    oos = pl.concat(scored_frames)
    _reject_non_finite_economic_inputs(oos)
    champion_oos_ledger = _build_calibration_ledger(oos, training_panel, label_column)

    replay = _event_ledger_evaluation(
        training_panel, oos, request, snapshot.manifest, registry, base, stress,
        calibration_ledger=champion_oos_ledger,
    )

    budget = PromotionRiskBudget()
    gates = _evaluate_gates(
        replay, fold_rank_ic, budget, request, n_trials=n_optuna_trials,
    )
    reasons.extend(cast(list[str], gates["reasons"]))

    holdout_ok, holdout_reason, _holdout_evidence = _evaluate_forward_holdout(
        registry, request, base_manifest, panel, holdout_fold, champion_config,
        feature_columns, label_column, relevance_column, snapshot.manifest, base, stress,
        calibration_ledger=champion_oos_ledger,
    )
    reasons.append(holdout_reason)
    passed = bool(gates["passed"]) and holdout_ok and bool(fold_rank_ic)

    model = fold_models[-1] if passed else _no_trade_model(
        base_manifest, feature_columns, label_column, relevance_column, champion_config,
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
    if LABEL_AVAILABLE_COLUMN in frame.columns:
        return frame.filter(
            pl.col(LABEL_AVAILABLE_COLUMN).is_null()
            | (pl.col(LABEL_AVAILABLE_COLUMN) <= decision_time)
        )
    return frame


def _index_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if "session_index" not in frame.columns:
        frame = frame.with_columns(
            pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
        )
    return frame.with_columns(
        pl.col("session_index").rank("dense").cast(pl.Int64).alias("session_index")
    )


def _resolve_label_column(frame: pl.DataFrame, manifest: DatasetManifest) -> str:
    candidates = [RESIDUAL_O2O_LABEL, manifest.label_definition]
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return str(candidate)
    return RESIDUAL_O2O_LABEL


def _will_need_lambdarank(frame: pl.DataFrame) -> bool:
    return not frame.is_empty()


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


def _tune_champion(
    panel: pl.DataFrame,
    folds: list[Fold],
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    label_span_sessions: int,
    *,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
) -> tuple[LambdaRankConfig | None, int]:
    """Run a temporally isolated screen -> shortlist -> economic selection.

    The tuning panel is derived exclusively from ``folds[0].train_mask``, the
    last purged-and-embargoed data available before the first outer validation
    decision, so no outer validation row influences candidate selection. All
    ``request.optuna_trials`` serial seeded TPE configurations are screened on
    the first inner fold under a frozen reduced boosting budget; a LightGBM
    NDCG callback drives Optuna median pruning, and pruned candidates remain
    terminal trials for Deflated Sharpe. The fixed shortlist of at most eight
    positive-screen candidates is fully refit over every inner fold and
    replayed through the exact event ledger, then ranked by economic evidence.
    Returns ``(config, n_trials)`` where ``n_trials`` is the terminal screen
    trial count fed to Deflated Sharpe.
    """
    LambdaRankConfig._tuning_telemetry = None
    tuning_panel = panel[folds[0].train_mask]
    tuning_folds = PurgedWalkForward(
        n_folds=max(1, min(3, len(folds))),
        label_horizon_sessions=label_span_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
        min_train_sessions=_MIN_TRAIN_SESSIONS // 2,
    ).split(tuning_panel)
    if not tuning_folds:
        return None, 0

    guard = TrialResourceGuard(request, predictor_count=len(feature_columns) * 3)
    contexts = _fit_stable_contexts(
        tuning_panel, tuning_folds, base_manifest, feature_columns, label_column,
        relevance_column,
    )
    cache_bytes = sum(
        _fold_cache_bytes(getattr(context, "prepared", None)) for context in contexts
    )

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lambdarank_v2_{request.artifact_id}",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=request.seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(1, request.optuna_trials // 5),
            n_warmup_steps=0,
        ),
    )

    def screen_objective(trial: optuna.Trial) -> float:
        config = _config_from_trial(trial)
        screen_config = _screen_config(config)
        ic = _score_trial_fold(
            tuning_panel, tuning_folds[0], contexts[0], request, base_manifest,
            feature_columns, label_column, relevance_column, screen_config, guard,
            trial, 0, callbacks=(_screen_ndcg_callback(trial),), report_progress=False,
        )
        if ic is None:
            raise optuna.TrialPruned()
        logger.info(
            "[EVAL] trial=%s stage=screen_rank_ic rank_ic=%.6f", trial.number, ic
        )
        return float(ic)

    screen_started = time.perf_counter()
    study.optimize(
        screen_objective,
        n_trials=request.optuna_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    screen_seconds = time.perf_counter() - screen_started
    logger.info(
        "[SYS] stage=screen elapsed_ms=%.1f rss=%.1f",
        screen_seconds * 1000.0,
        TrialResourceGuard._rss_mib(),
    )

    n_terminal = sum(
        1
        for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    if n_terminal != request.optuna_trials:
        LambdaRankConfig._tuning_telemetry = {
            "n_terminal_trials": n_terminal,
            "optuna_trials": request.optuna_trials,
            "selection_status": "incomplete",
        }
        return None, n_terminal
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
    best_screen_rank_ic = max((value for value, _ in completed), default=None)
    screen_scores = sorted(
        ((value, number) for value, number in completed if value > 0.0),
        key=lambda pair: (-pair[0], pair[1]),
    )
    shortlist = screen_scores[:_SCREEN_SHORTLIST_SIZE]
    for _screen_ic, trial_number in shortlist:
        logger.info("[EVAL] trial=%s stage=shortlisted", trial_number)

    champion, selection = _select_economic_champion(
        study, shortlist, tuning_panel, tuning_folds, contexts, request, base_manifest,
        feature_columns, label_column, relevance_column, guard, dataset_manifest,
        registry, base_schedule, stress_schedule,
    )

    telemetry = guard.telemetry()
    for name, value in telemetry.items():
        study.set_user_attr(name, value)
    study.set_user_attr("n_terminal_trials", n_terminal)
    study.set_user_attr("optuna_trials", request.optuna_trials)
    study.set_user_attr("screened_trials", len(complete))
    study.set_user_attr("pruned_trials", n_terminal - len(complete))
    study.set_user_attr("shortlisted_trials", len(shortlist))
    study.set_user_attr("cache_bytes", cache_bytes)
    study.set_user_attr("screen_seconds", screen_seconds)
    if best_screen_rank_ic is not None:
        study.set_user_attr("best_screen_rank_ic", best_screen_rank_ic)
    if champion is None or selection is None:
        if selection is not None:
            for name, value in selection.items():
                study.set_user_attr(name, value)
        if not shortlist:
            study.set_user_attr("selection_status", "no_complete_screen_candidate")
        else:
            study.set_user_attr("selection_status", "no_economically_eligible_candidate")
        logger.info(
            "[EVAL] stage=selection_status %s",
            study.user_attrs["selection_status"],
        )
        LambdaRankConfig._tuning_telemetry = dict(study.user_attrs)
        return None, n_terminal

    for name, value in selection.items():
        study.set_user_attr(name, value)
    study.set_user_attr("selection_status", "selected")
    assert champion is not None
    champion._tuning_telemetry = dict(study.user_attrs)
    logger.info(
        "[EVAL] trial=%s stage=selected",
        int(cast(int, selection["selected_trial_number"])),
    )
    return champion, n_terminal


def _fit_stable_contexts(
    tuning_panel: pl.DataFrame,
    tuning_folds: list[Fold],
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
) -> list[_StableTrialContext]:
    """Fit one StableRankComposite per immutable tuning-fold context.

    The composite's fitted weights/orientations/winsors and the validation
    stable scores are invariant across every LambdaRank search parameter, so
    they are computed once and cached; only the slim ``(session,
    instrument_id, pred_score)`` score frame is retained as the search cache.
    """
    allowlist = stock_alpha_v2_allowlist()
    contexts: list[_StableTrialContext] = []
    for fold in tuning_folds:
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
        contexts.append(
            _StableTrialContext(
                train_processed=train_processed,
                validation_processed=validation_processed,
                validation_frame=validation_frame,
                stable_scores=stable_scores,
                prepared=prepared,
            )
        )
    return contexts


def _score_trial_fold(
    tuning_panel: pl.DataFrame,
    fold: Fold,
    context: _StableTrialContext,
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
) -> float | None:
    """Score one tuning fold with the cached stable context; ``None`` prunes.

    The resource guard admits the allocation (including the prepared-fold cache
    bytes) before fitting and records elapsed/RSS telemetry in a ``finally`` so
    a callback-raised :class:`optuna.TrialPruned` still leaves timing evidence.
    A breach raises :class:`TrainingCapacityError`. Fold model, prediction
    frame, and LightGBM datasets are local to this call and released on return,
    so a trial never retains every fold's artifacts.
    """
    del tuning_panel, fold, feature_columns
    key = f"trial_{trial.number}_fold_{fold_index}"
    guard.admit(
        context.train_processed.height,
        extra_bytes=_fold_cache_bytes(getattr(context, "prepared", None)),
    )
    started = time.perf_counter()
    try:
        result = _score_context_model(
            context, request, base_manifest, label_column, relevance_column, config,
            callbacks=callbacks,
        )
        if result is None:
            return None
        ic, _scored = result
        if report_progress:
            trial.report(float(ic), step=fold_index)
            if ic <= 0.0:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(ic)
    finally:
        guard.record_fold(
            key,
            time.perf_counter() - started,
            guard.estimate_mib(context.train_processed.height),
        )
        guard.check_after()


def _score_context_model(
    context: _StableTrialContext,
    request: TrainingRequest,
    base_manifest: ModelManifest,
    label_column: str,
    relevance_column: str | None,
    config: LambdaRankConfig,
    *,
    callbacks: Sequence[Callable[..., object]] = (),
) -> tuple[float, pl.DataFrame] | None:
    """Fit one candidate on a cached fold and return ``(rank_ic, scored)``.

    ``None`` signals a fail-closed fold (missing columns, unusable groups, or
    invalid inputs). The prepared-fold fast path is used when the context
    carries immutable matrices; otherwise the uncached path yields identical
    scores.
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
        fit_ok = model.fit_trial(
            context.train_processed,
            context.validation_processed,
            context.stable_scores,
            prepared=getattr(context, "prepared", None),
            callbacks=callbacks,
        )
    except ValueError:
        fit_ok = False
    if not fit_ok or model.no_trade:
        return None
    predict_input = _drop_target_columns(context.validation_processed, label_column)
    scored = model.predict(predict_input)
    ic = _median_rank_ic(context.validation_frame, scored, label_column)
    return float(ic), scored


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
        )
    }
    return LambdaRankConfig(
        **params,
        n_estimators=_SCREEN_BOOSTING_ROUNDS,
        early_stopping_rounds=_SCREEN_EARLY_STOPPING_ROUNDS,
    )


def _screen_ndcg_callback(trial: optuna.Trial) -> Callable[[CallbackEnv], None]:
    """LightGBM callback reporting validation NDCG for Optuna median pruning.

    Reports every ``_SCREEN_NDCG_INTERVAL_ROUNDS`` after a frozen warm-up and
    raises ``optuna.TrialPruned`` when Optuna decides the trial should stop;
    the exception propagates through ``lgb.train`` and ``fit_trial`` so the
    screen candidate stays a terminal (pruned) trial.
    """

    def callback(env: CallbackEnv) -> None:
        iteration = int(env.iteration)
        if iteration < _SCREEN_NDCG_WARMUP_ROUNDS:
            return
        if (iteration - _SCREEN_NDCG_WARMUP_ROUNDS) % _SCREEN_NDCG_INTERVAL_ROUNDS != 0:
            return
        ndcg = _evaluation_ndcg(env)
        if ndcg is None:
            return
        trial.report(ndcg, iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return callback


def _evaluation_ndcg(env: CallbackEnv) -> float | None:
    """First finite validation NDCG reported by the LightGBM callback env."""
    for result in env.evaluation_result_list or ():
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            continue
        metric = result[1]
        if not isinstance(metric, str) or not metric.startswith("ndcg"):
            continue
        value = result[2]
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _fit_and_score_candidate(
    tuning_panel: pl.DataFrame,
    tuning_folds: list[Fold],
    contexts: list[_StableTrialContext],
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
) -> tuple[list[float], pl.DataFrame | None] | None:
    """Full-budget refit of one shortlisted candidate over every inner fold.

    Returns ``(fold_rank_ic, concatenated validation scores)``, or
    ``(fold_rank_ic, None)`` when the first non-positive full-refit Rank-IC
    rejects the candidate early, or ``None`` when any fold fails closed. The
    early stop is equivalence-preserving: eligibility already requires every
    full-refit IC to be strictly positive, so the skipped later folds and
    replay cannot make that candidate eligible. Per-fold models, prediction
    frames, and LightGBM datasets are released before the next fold.
    """
    del tuning_panel, feature_columns
    fold_rank_ic: list[float] = []
    scored_frames: list[pl.DataFrame] = []
    for fold_index, context in enumerate(contexts):
        key = f"refit_{candidate_key}_fold_{fold_index}"
        guard.admit(
            context.train_processed.height,
            extra_bytes=(
                _fold_cache_bytes(getattr(context, "prepared", None))
                + static_cache_bytes
            ),
        )
        started = time.perf_counter()
        try:
            result = _score_context_model(
                context, request, base_manifest, label_column, relevance_column, config
            )
            if result is None:
                return None
            ic, scored = result
        finally:
            guard.record_fold(
                key,
                time.perf_counter() - started,
                guard.estimate_mib(context.train_processed.height),
            )
            guard.check_after()
        fold_rank_ic.append(ic)
        if ic <= 0.0:
            logger.info(
                "[EVAL] %s stage=early_rejected fold_rank_ic=%.6f",
                candidate_key,
                ic,
            )
            return fold_rank_ic, None
        scored_frames.append(scored)
    return fold_rank_ic, pl.concat(scored_frames)


def _prepare_replay_static_context(
    panel: pl.DataFrame,
    request: TrainingRequest,
) -> ReplayStaticContext:
    """Build the candidate-invariant replay inputs once per selection panel.

    The market frame (carrying the immutable point-in-time market columns),
    instrument map, and risk policy are shared by every shortlisted candidate.
    Raises ``ValueError`` for a missing required replay column or a non-finite
    cached market input.
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
    instruments = _instruments_from_frame(frame)
    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
    )
    return ReplayStaticContext(
        market_panel=frame,
        instruments=instruments,
        policy=policy,
        cache_bytes=_frame_bytes(frame),
    )


_ECONOMIC_FAILURE_CODES = (
    "non_positive_fold_rank_ic",
    "no_attempted_orders",
    "no_filled_orders",
    "non_finite_replay",
    "non_positive_bootstrap_lower_bound",
)


def _evaluate_economic_candidate(
    fold_rank_ic: list[float],
    replay: ReplayResult,
    request: TrainingRequest,
    trial_number: int,
    screen_rank_ic: float,
) -> EconomicCandidateEvidence:
    """Evaluate every economic predicate and emit immutable candidate evidence.

    All five predicates are evaluated without early return so the exact failed
    reason codes stay recoverable for every shortlisted trial. ``eligible``
    matches the existing fail-closed ``_economically_eligible`` rule exactly.
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
    bootstrap_lower_bound = _inner_bootstrap_lower_bound(replay, request)
    if bootstrap_lower_bound <= 0.0:
        failures.append("non_positive_bootstrap_lower_bound")
    calibration_evidence = replay.calibration_evidence or {}
    return EconomicCandidateEvidence(
        trial_number=trial_number,
        screen_rank_ic=screen_rank_ic,
        fold_rank_ic=list(fold_rank_ic),
        median_rank_ic=float(np.median(fold_rank_ic)) if fold_rank_ic else 0.0,
        attempted_orders=replay.attempted_orders,
        filled_orders=replay.filled_orders,
        planned_cycles=replay.planned_cycles,
        no_trade_reason_counts=dict(replay.no_trade_reason_counts),
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
    contexts: list[_StableTrialContext],
    request: TrainingRequest,
    base_manifest: ModelManifest,
    feature_columns: tuple[str, ...],
    label_column: str,
    relevance_column: str | None,
    guard: TrialResourceGuard,
    dataset_manifest: DatasetManifest,
    registry: ModelArtifactRegistry,
    base_schedule: CostSchedule,
    stress_schedule: CostSchedule,
) -> tuple[LambdaRankConfig | None, dict[str, object] | None]:
    """Replay each shortlisted candidate and pick the top economic evidence.

    Every candidate is fully refit over all inner folds and replayed through
    the exact event ledger with the outer base/stress schedules. A candidate is
    skipped after its first non-positive full-refit Rank-IC (later folds and
    replay cannot make it eligible). Eligible candidates sort descending by
    ``(bootstrap lower bound, strategy IR, -max drawdown, -turnover, median
    Rank-IC, -trial number)``; a tie is won by the lower trial number. Returns
    ``(config, selection telemetry)`` or ``(None, None)`` when no shortlisted
    candidate is eligible.
    """
    if not shortlist:
        return None, None
    replay_context = _prepare_replay_static_context(tuning_panel, request)
    refit_started = time.perf_counter()
    candidate_rows: list[tuple[tuple[float, ...], int]] = []
    evidence_by_trial: dict[int, dict[str, float]] = {}
    calibration_state_by_trial: dict[int, dict[str, object] | None] = {}
    shortlist_evidence: list[dict[str, object]] = []
    early_rejected_seconds: list[float] = []
    replay_seconds = 0.0
    for _screen_ic, trial_number in shortlist:
        frozen = study.trials[trial_number]
        config = _config_from_params(dict(frozen.params))
        refit_started_trial = time.perf_counter()
        refit = _fit_and_score_candidate(
            tuning_panel, tuning_folds, contexts, request, base_manifest,
            feature_columns, label_column, relevance_column, config, guard,
            f"trial{trial_number}",
            static_cache_bytes=replay_context.cache_bytes,
        )
        if refit is None:
            logger.info("[EVAL] trial=%s stage=refit_failed", trial_number)
            continue
        fold_rank_ic, oos = refit
        if oos is None:
            early_rejected_seconds.append(time.perf_counter() - refit_started_trial)
            logger.info("[EVAL] trial=%s stage=early_rejected", trial_number)
            continue
        replay_started = time.perf_counter()
        causal_oos_ledger = _build_calibration_ledger(oos, tuning_panel, label_column)
        replay = _event_ledger_evaluation(
            tuning_panel, oos, request, dataset_manifest, registry, base_schedule,
            stress_schedule, replay_context=replay_context,
            calibration_ledger=causal_oos_ledger,
        )
        replay_seconds += time.perf_counter() - replay_started
        evidence = _evaluate_economic_candidate(
            fold_rank_ic, replay, request, trial_number, _screen_ic,
        )
        shortlist_evidence.append(evidence.to_json_safe())
        logger.info(
            "[EVAL] trial=%s stage=evidence eligible=%s failure_reasons=%s",
            trial_number,
            evidence.eligible,
            list(evidence.failure_reasons),
        )
        if not evidence.eligible:
            continue
        candidate_rows.append(
            (
                (
                    evidence.bootstrap_lower_bound,
                    evidence.strategy_ir,
                    -evidence.max_drawdown,
                    -evidence.turnover,
                    evidence.median_rank_ic,
                    -trial_number,
                ),
                trial_number,
            )
        )
        evidence_by_trial[trial_number] = {
            "bootstrap_lower_bound": evidence.bootstrap_lower_bound,
            "strategy_ir": evidence.strategy_ir,
            "max_drawdown": evidence.max_drawdown,
            "turnover": evidence.turnover,
            "median_rank_ic": evidence.median_rank_ic,
        }
        calibration_state_by_trial[trial_number] = evidence.calibration_state
        calibration_state_by_trial[trial_number] = evidence.calibration_state
        logger.info(
            "[EVAL] trial=%s stage=economically_eligible "
            "bootstrap_lower_bound=%.8f strategy_ir=%.6f",
            trial_number,
            evidence.bootstrap_lower_bound,
            evidence.strategy_ir,
        )
    refit_seconds = time.perf_counter() - refit_started
    logger.info(
        "[SYS] stage=shortlist elapsed_ms=%.1f rss=%.1f",
        refit_seconds * 1000.0,
        TrialResourceGuard._rss_mib(),
    )
    selection_tail = {
        "early_rejected_full_refits": len(early_rejected_seconds),
        "early_rejected_full_refit_seconds": round(
            float(sum(early_rejected_seconds)), 3
        ),
        "shortlist_candidate_evidence": shortlist_evidence,
    }
    if not candidate_rows:
        return None, {
            "economically_eligible_trials": 0,
            "full_refit_seconds": refit_seconds,
            "economic_replay_seconds": replay_seconds,
            **selection_tail,
        }
    candidate_rows.sort(key=lambda row: row[0], reverse=True)
    _winner_key, winner_number = candidate_rows[0]
    champion = _config_from_params(dict(study.trials[winner_number].params))
    winner_calibration_state = calibration_state_by_trial.get(winner_number)
    if winner_calibration_state is not None:
        champion._calibration_state = dict(winner_calibration_state)
    return champion, {
        "economically_eligible_trials": len(candidate_rows),
        "full_refit_seconds": refit_seconds,
        "economic_replay_seconds": replay_seconds,
        "selected_trial_number": winner_number,
        **{
            f"selected_inner_{name}": value
            for name, value in evidence_by_trial[winner_number].items()
        },
        "selected_calibration_state": winner_calibration_state,
        **selection_tail,
    }


def _config_from_trial(trial: optuna.Trial) -> LambdaRankConfig:
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
        }
    )


def _config_from_params(params: dict[str, Any]) -> LambdaRankConfig:
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
) -> ReplayResult:
    """Replay the out-of-sample scored panel through the event-driven backtester.

    A scored planner constructs constrained target allocations directly from the
    frozen fold predictions, so promotion metrics come from the same event
    ledger used by paper/live paths without needing a pre-published artifact.
    When ``replay_context`` is supplied its cached market panel, risk policy,
    and instruments are reused instead of being rebuilt; only ``pred_score``
    is joined per replay. The replay-window causal 20-session ADTV is computed
    once and passed to the backtester so the base and stress execution ledgers
    reuse the same validated column instead of recomputing it twice.
    """
    from src.core.portfolio import PortfolioSnapshot
    from src.stocks.backtesting.engine import (
        ArtifactSchedule,
        ArtifactSlot,
        BacktestRequest,
        StockBacktester,
    )
    from src.stocks.data.costs import CostEvidence
    from src.stocks.workflows.trading_cycle import (
        CycleStatus,
        TradingCycleRequest,
        TradingCycleResult,
        _build_intents,
    )

    if replay_context is None:
        replay_context = _prepare_replay_static_context(panel, request)
    frame = replay_context.market_panel
    instruments = replay_context.instruments
    policy = replay_context.policy

    scored_for_replay = frame.join(
        oos_scored.select("instrument_id", "session", "pred_score"),
        on=["instrument_id", "session"],
        how="left",
    )

    scored_sessions = sorted(
        scored_for_replay.filter(pl.col("pred_score").is_not_null())["session"].unique().to_list()
    )
    if not scored_sessions:
        raise ValueError("scored OOS panel exposes no scored session")
    replay_frame = frame.filter(pl.col("session") >= scored_sessions[0])
    sessions = sorted(replay_frame["session"].unique().to_list())
    replay_adtv = (
        replay_frame.sort("session")
        .with_columns(
            pl.col("trading_value")
            .rolling_mean(20, min_samples=1)
            .over("instrument_id")
            .alias("adtv")
        )
        .select("instrument_id", "session", "adtv")
    )

    calibrator = (
        CausalAlphaCalibrator(
            bucket_count=request.calibration_bucket_count,
            min_calibration_sessions=request.min_calibration_sessions,
            seed=request.seed,
            n_bootstrap=request.n_bootstrap,
            bootstrap_alpha=request.bootstrap_alpha,
        )
        if calibration_ledger is not None and not calibration_ledger.is_empty()
        else None
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

    def scored_planner(
        snapshot: DatasetSnapshot,
        registry_inner: object,
        instruments_map: object,
        portfolio: PortfolioSnapshot,
        cycle_request: TradingCycleRequest,
    ) -> TradingCycleResult:
        del snapshot, registry_inner, instruments_map
        from src.stocks.trading.portfolio_constructor import construct_target_allocations

        try:
            visible = _bounded_replay_history(
                scored_for_replay, cycle_request.decision_time, policy
            )
        except ValueError as exc:
            return _scored_no_trade(portfolio, cycle_request, f"constraint:{exc}")
        if visible.is_empty():
            return _scored_no_trade(portfolio, cycle_request, "empty-scored-cross-section")
        try:
            if calibrator is not None:
                assert calibration_ledger is not None
                visible = calibrator.transform(
                    visible,
                    calibration_ledger,
                    cycle_request.decision_time,
                    base_schedule,
                )
                calibration_tracker["state"] = calibrator.calibration_state()
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

    start_time = _session_as_datetime(sessions[0])
    end_time = _session_as_datetime(sessions[-1])
    decision_indices = tuple(
        i for i in range(len(sessions)) if i % _REBALANCE_EVERY_SESSIONS == 0
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
    backtester = StockBacktester(
        planner=scored_planner,
        registry=registry,
        instruments=instruments,
        manifest=dataset_manifest,
        cost_schedule=base_schedule,
        stress_cost_schedule=stress_schedule,
        cost_evidence=evidence,
        seed=request.seed,
    )
    result = backtester.run(
        replay_frame, artifacts, initial_portfolio, backtest_request,
        adtv=replay_adtv,
    )
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
        calibration_evidence=calibration_evidence,
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
        if c.startswith(("target_", "label_")) or c == label_column
    ]
    return frame.drop(drops)


def _build_calibration_ledger(
    oos_scored: pl.DataFrame,
    panel: pl.DataFrame,
    label_column: str,
) -> pl.DataFrame:
    """Join OOS predictions with their point-in-time labels for calibration.

    The ledger carries ``(session, instrument_id, score, label,
    label_available_time)`` so ``CausalAlphaCalibrator`` can consume only prior
    label-available OOS observations. ``panel`` is the parent frame carrying the
    canonical label columns; every ledger row is a real historical OOS score.
    """
    if label_column not in panel.columns:
        raise ValueError(f"panel has no calibration label column {label_column!r}")
    if LABEL_AVAILABLE_COLUMN not in panel.columns:
        raise ValueError("panel has no label_available_time for calibration")
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
                LABEL_AVAILABLE_COLUMN: pl.Datetime("us", "UTC"),
            }
        )
    if "pred_score" not in oos_scored.columns:
        raise ValueError("scored frame must carry pred_score for calibration")
    return oos_scored.select("session", "instrument_id", "pred_score").join(
        panel.select(
            "session",
            "instrument_id",
            label_column,
            LABEL_AVAILABLE_COLUMN,
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


def _reject_non_finite_economic_inputs(frame: pl.DataFrame) -> None:
    for column in _ECONOMIC_COLUMNS:
        if column in frame.columns:
            non_finite = frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite())
            if not non_finite.is_empty():
                raise ValueError(f"non-finite economic input in {column}")


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
    promotion instead of being read as a flat strategy.
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

    excess = replay.excess_returns
    lower_bound = (
        _moving_block_bootstrap_lower_bound(
            excess,
            max(5, 1),
            max(request.n_bootstrap, 2),
            request.seed,
            budget.bootstrap_alpha,
        )
        if excess
        else 0.0
    )
    gate2_ok = lower_bound > 0.0
    reasons.append(f"gate2_excess_lower_bound={lower_bound:.8f}")
    passed = passed and gate2_ok

    strategy_returns = replay.strategy_returns
    benchmark_returns = replay.benchmark_returns
    strategy_ir = _information_ratio(strategy_returns)
    benchmark_ir = _information_ratio(benchmark_returns)
    stable_ir = strategy_ir * 0.5
    gate3_ok = strategy_ir > stable_ir and strategy_ir > benchmark_ir
    reasons.append(f"gate3_strategy_ir={strategy_ir:.6f}")
    reasons.append(f"gate3_benchmark_ir={benchmark_ir:.6f}")
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

    deflated_prob = _deflated_sharpe_probability(
        strategy_returns,
        annualization=252,
        n_trials=n_trials,
    )
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
) -> tuple[Fold | None, pl.DataFrame]:
    """Reserve the dated forward holdout before tuning and outer folds.

    When the immutable snapshot holds at least the required number of
    label-available sessions on or after ``2026-03-10``, the newest
    ``holdout_sessions`` are pinned as a locked ``PurgedWalkForward.holdout``
    and the returned training panel contains only sessions before that block.
    Otherwise returns ``(None, panel)`` unchanged and promotion stays fail
    closed.
    """
    holdout_sessions = (
        request.holdout_sessions
        if request.holdout_sessions > 0
        else _FORWARD_HOLDOUT_SESSIONS
    )
    if holdout_sessions < 1 or LABEL_AVAILABLE_COLUMN not in panel.columns:
        return None, panel
    holdout_start = datetime.combine(_FORWARD_HOLDOUT_START, datetime.min.time(), tzinfo=UTC)
    post_start_sessions = panel.filter(
        (pl.col("session") >= holdout_start)
        & pl.col(LABEL_AVAILABLE_COLUMN).is_not_null()
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
) -> tuple[bool, str, dict[str, object] | None]:
    """Fit the frozen candidate on pre-holdout data and replay the block once.

    Returns ``(ready, reason, evidence)``. A candidate fingerprint may inspect
    the holdout exactly once; a reused fingerprint raises ``ValueError``.
    Incomplete data leaves the candidate ``NO_TRADE`` with
    ``forward_holdout_ready=false``.
    """
    if holdout_fold is None:
        return (
            False,
            "gate8_forward_holdout_ready=false:insufficient-label-available-sessions-on-or-after-2026-03-10",
            None,
        )
    block_session_indexes = sorted(
        int(v)
        for v in panel["session_index"][holdout_fold.validation_mask].unique().to_list()
    )
    holdout_session_range = (
        block_session_indexes[0],
        block_session_indexes[-1],
    )
    fingerprint = _forward_holdout_fingerprint(
        base_manifest, request, dataset_manifest, holdout_session_range,
        champion_config, base_schedule, stress_schedule,
    )
    existing = registry.read_forward_holdout(request.artifact_id)
    if existing is not None and existing.get("fingerprint") == fingerprint:
        raise ValueError(
            f"forward holdout for candidate fingerprint {fingerprint!r} "
            f"was already inspected for {request.artifact_id!r}"
        )
    models, scored, _fold_ic = _fit_and_score_folds(
        panel, [holdout_fold], request, base_manifest, feature_columns, label_column,
        relevance_column, champion_config,
    )
    if not models:
        return False, "gate8_forward_holdout_ready=false:no-fit", None
    holdout_oos = pl.concat(scored)
    replay = _event_ledger_evaluation(
        panel, holdout_oos, request, dataset_manifest, registry, base_schedule,
        stress_schedule, calibration_ledger=calibration_ledger,
    )
    if replay.attempted_orders <= 0:
        return False, "gate8_forward_holdout_ready=false:no-attempted-orders", None
    if replay.filled_orders <= 0:
        return False, "gate8_forward_holdout_ready=false:no-filled-orders", None
    evidence: dict[str, object] = {
        "feature_schema_hash": base_manifest.feature_schema_hash,
        "universe_policy_hash": base_manifest.universe_policy_hash,
        "label_dataset_hash": _label_dataset_hash(dataset_manifest),
        "holdout_session_range": holdout_session_range,
        "model_config": _config_snapshot(champion_config),
        "risk_policy": {
            name: getattr(request, name)
            for name in (
                "top_k",
                "max_exposure",
                "max_single_weight",
                "participation_limit",
            )
        },
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
        "base_total_return": replay.base_total_return,
        "stress_total_return": replay.stress_total_return,
        "benchmark_total_return": replay.benchmark_total_return,
        "promotion_reasons": reasons,
        "gates": gates,
        "optuna_trials": request.optuna_trials,
        "resource": tuning_telemetry or {},
    }
