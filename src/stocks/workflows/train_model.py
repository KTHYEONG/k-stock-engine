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
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
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
from src.stocks.research.datasets import (
    research_eligible_frame,
    validate_stock_rows_available,
)
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
from src.stocks.research.lambdarank import LambdaRankBlendModel, LambdaRankConfig
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

    def admit(self, rows: int) -> None:
        """Reject the run before allocation when the increment cannot fit."""
        estimate = self.estimate_mib(rows)
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

    champion_config, n_optuna_trials = _tune_champion(
        training_panel, folds, request, base_manifest, feature_columns, label_column,
        relevance_column, label_span_sessions,
    )
    if champion_config is None:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-champion-trial",
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

    base = request.base_cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()
    replay = _event_ledger_evaluation(
        training_panel, oos, request, snapshot.manifest, registry, base, stress,
    )

    budget = PromotionRiskBudget()
    gates = _evaluate_gates(
        replay, fold_rank_ic, budget, request, n_trials=n_optuna_trials,
    )
    reasons.extend(cast(list[str], gates["reasons"]))

    holdout_ok, holdout_reason, _holdout_evidence = _evaluate_forward_holdout(
        registry, request, base_manifest, panel, holdout_fold, champion_config,
        feature_columns, label_column, relevance_column, snapshot.manifest, base, stress,
    )
    reasons.append(holdout_reason)
    passed = bool(gates["passed"]) and holdout_ok and bool(fold_rank_ic)

    model = fold_models[-1] if passed else _no_trade_model(
        base_manifest, feature_columns, label_column, relevance_column, champion_config,
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
) -> tuple[LambdaRankConfig | None, int]:
    """Run a temporally isolated serial Optuna TPE search and select lexicographically.

    The tuning panel is derived exclusively from ``folds[0].train_mask``, the
    last purged-and-embargoed data available before the first outer validation
    decision, so no outer validation row influences hyperparameter selection.
    Trials run serially (``n_jobs=1``), one inner fold at a time, pruning
    immediately on non-positive fold Rank-IC or an Optuna median-pruner
    decision; per-fold models, predictions, and LightGBM datasets are released
    before the next fold. Returns ``(config, n_trials)`` where ``n_trials`` is
    the completed plus pruned terminal trial count fed to Deflated Sharpe; a
    count that does not equal ``request.optuna_trials`` never selects a
    champion.
    """
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

    def objective(trial: optuna.Trial) -> float:
        config = _config_from_trial(trial)
        fold_rank_ic: list[float] = []
        for fold_index, fold in enumerate(tuning_folds):
            ic = _score_trial_fold(
                tuning_panel, fold, contexts[fold_index], request, base_manifest,
                feature_columns, label_column, relevance_column, config, guard,
                trial, fold_index,
            )
            if ic is None:
                raise optuna.TrialPruned()
            trial.report(float(ic), step=fold_index)
            if ic <= 0.0:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()
            fold_rank_ic.append(float(ic))
        return float(np.median(fold_rank_ic))

    study.optimize(objective, n_trials=request.optuna_trials, n_jobs=1, show_progress_bar=False)
    n_terminal = sum(
        1
        for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    telemetry = guard.telemetry()
    for name, value in telemetry.items():
        study.set_user_attr(name, value)
    study.set_user_attr("n_terminal_trials", n_terminal)
    study.set_user_attr("optuna_trials", request.optuna_trials)
    best_trial = _completed_best_trial(study)
    if n_terminal != request.optuna_trials or not study.trials or best_trial is None:
        return None, n_terminal
    champion = _config_from_params(dict(best_trial.params))
    champion._tuning_telemetry = dict(study.user_attrs)
    return champion, n_terminal


def _completed_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Best completed trial without Optuna's all-pruned ``ValueError``."""
    best: optuna.trial.FrozenTrial | None = None
    best_value = float("-inf")
    for trial in study.trials:
        if trial.state is not optuna.trial.TrialState.COMPLETE:
            continue
        value = trial.value
        if value is None:
            continue
        if best is None or value > best_value:
            best = trial
            best_value = value
    return best


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
    del relevance_column
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
        contexts.append(
            _StableTrialContext(
                train_processed=train_processed,
                validation_processed=validation_processed,
                validation_frame=validation_frame,
                stable_scores=stable_scores,
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
) -> float | None:
    """Score one tuning fold with the cached stable context; ``None`` prunes.

    The resource guard admits the allocation before fitting, samples process
    RSS afterward, and a breach raises :class:`TrainingCapacityError`. Fold
    model, prediction frame, and LightGBM datasets are local to this call and
    released on return, so a trial never retains every fold's artifacts.
    """
    del tuning_panel, fold, feature_columns
    key = f"trial_{trial.number}_fold_{fold_index}"
    guard.admit(context.train_processed.height)
    started = time.perf_counter()
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
            context.train_processed, context.validation_processed, context.stable_scores
        )
    except ValueError:
        fit_ok = False
    if not fit_ok or model.no_trade:
        guard.record_fold(key, time.perf_counter() - started, guard.estimate_mib(context.train_processed.height))
        return None
    predict_input = _drop_target_columns(context.validation_processed, label_column)
    scored = model.predict(predict_input)
    ic = _median_rank_ic(context.validation_frame, scored, label_column)
    guard.record_fold(
        key, time.perf_counter() - started, guard.estimate_mib(context.train_processed.height)
    )
    guard.check_after()
    return float(ic)


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
) -> ReplayResult:
    """Replay the out-of-sample scored panel through the event-driven backtester.

    A scored planner constructs constrained target allocations directly from the
    frozen fold predictions, so promotion metrics come from the same event
    ledger used by paper/live paths without needing a pre-published artifact.
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

    frame = panel.drop("session_index")
    adtv_lookup = (
        frame.sort("session")
        .with_columns(
            pl.col("trading_value")
            .rolling_mean(20, min_samples=1)
            .over("instrument_id")
            .alias("adtv")
        )
        .select("instrument_id", "session", "adtv")
    )
    scored_for_replay = (
        frame.join(
            oos_scored.select("instrument_id", "session", "pred_score"),
            on=["instrument_id", "session"],
            how="left",
        ).join(adtv_lookup, on=["instrument_id", "session"], how="left")
    )
    instruments = _instruments_from_frame(frame)

    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
    )

    scored_sessions = sorted(
        scored_for_replay.filter(pl.col("pred_score").is_not_null())["session"].unique().to_list()
    )
    if not scored_sessions:
        raise ValueError("scored OOS panel exposes no scored session")
    replay_frame = frame.filter(pl.col("session") >= scored_sessions[0])
    sessions = sorted(replay_frame["session"].unique().to_list())

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
    result = backtester.run(replay_frame, artifacts, initial_portfolio, backtest_request)
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
        stress_schedule,
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
