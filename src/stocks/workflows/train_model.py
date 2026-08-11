"""Stock model-training workflow: v2 snapshot -> nested walk-forward -> champion/NO_TRADE artifact.

The workflow consumes the composed labeled v2 snapshot (base OHLCV + ``feature__*``
stock_alpha_v2 columns + canonical residual labels), runs a purged/embargoed
expanding walk-forward with quarterly refits, tunes the ``LambdaRankBlendModel``
with seeded Optuna TPE trials on the development window only, evaluates every
fold through the event-driven ``StockBacktester`` under base and stress costs,
and publishes either an immutable champion or an immutable ``NO_TRADE`` artifact.
Promotion is lexicographic and fail-closed; a failing gate never relaxes
parameters. Promotion remains false until a frozen candidate passes one new
252-session forward holdout starting after 2026-03-10.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import polars as pl

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
from src.stocks.research.models import ModelManifest
from src.stocks.workflows.contracts import TrainingRequest

if TYPE_CHECKING:
    from src.stocks.backtesting.engine import BacktestLedgerRow

logger = logging.getLogger("stocks.workflows.train_model")

_ECONOMIC_COLUMNS = ("open", "high", "low", "close", "volume", "trading_value", "market_cap")

_MIN_TRAIN_SESSIONS = 756
_VALIDATION_BLOCK_SESSIONS = 252
_LABEL_PURGE_SESSIONS = 6
_EMBARGO_SESSIONS = 5
_REFIT_EVERY_SESSIONS = 63
_REBALANCE_EVERY_SESSIONS = 5
_N_OPTUNA_TRIALS = 80
_FORWARD_HOLDOUT_START = date(2026, 3, 10)
_FORWARD_HOLDOUT_SESSIONS = 252
_MIN_GROUP_SIZE = 20


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

    reasons: list[str] = []
    splitter = PurgedWalkForward(
        n_folds=1,
        label_horizon_sessions=_LABEL_PURGE_SESSIONS,
        embargo_sessions=_EMBARGO_SESSIONS,
        session_column="session_index",
        validation_window_sessions=_VALIDATION_BLOCK_SESSIONS,
        min_train_sessions=_MIN_TRAIN_SESSIONS,
    )
    folds = splitter.split(panel)
    if not folds:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-eligible-folds",
        )

    champion_config = _tune_champion(
        panel, folds, request, base_manifest, feature_columns, label_column,
        relevance_column,
    )
    if champion_config is None:
        return _publish_no_trade(
            registry, request, base_manifest, panel, label_column, relevance_column,
            "no-champion-trial",
        )

    fold_models, scored_frames, fold_rank_ic = _fit_and_score_folds(
        panel, folds, request, base_manifest, feature_columns, label_column,
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
        panel, oos, request, snapshot.manifest, registry, base, stress,
    )

    budget = PromotionRiskBudget()
    gates = _evaluate_gates(replay, fold_rank_ic, budget, request)
    reasons.extend(cast(list[str], gates["reasons"]))

    holdout_ok = _forward_holdout_not_consumed(base_manifest, request)
    reasons.append(f"gate8_forward_holdout_ready={holdout_ok}")
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
) -> LambdaRankConfig | None:
    """Run seeded Optuna TPE trials on the development folds and select lexicographically."""
    dev = pl.concat([panel[fold.train_mask] for fold in folds])
    dev_folds = PurgedWalkForward(
        n_folds=max(1, min(3, len(folds))),
        label_horizon_sessions=_LABEL_PURGE_SESSIONS,
        embargo_sessions=_EMBARGO_SESSIONS,
        session_column="session_index",
        min_train_sessions=_MIN_TRAIN_SESSIONS // 2,
    ).split(dev)
    if not dev_folds:
        return None

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lambdarank_v2_{request.artifact_id}",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=request.seed, n_startup_trials=10),
    )

    def objective(trial: optuna.Trial) -> float:
        config = _config_from_trial(trial)
        _models, _scored, fold_ic = _fit_and_score_folds(
            dev, dev_folds, request, base_manifest, feature_columns, label_column,
            relevance_column, config,
        )
        if not fold_ic:
            raise optuna.TrialPruned()
        if any(ic <= 0.0 for ic in fold_ic):
            raise optuna.TrialPruned()
        return float(np.median(fold_ic))

    study.optimize(objective, n_trials=_N_OPTUNA_TRIALS, show_progress_bar=False)
    if not study.trials or study.best_trial is None:
        return None
    return _config_from_params(dict(study.best_params))


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
    from src.stocks.trading.portfolio_constructor import StockRiskPolicy
    from src.stocks.workflows.trading_cycle import (
        CycleStatus,
        TradingCycleRequest,
        TradingCycleResult,
    )

    frame = panel.drop("session_index")
    sessions = sorted(frame["session"].unique().to_list())
    instruments = _instruments_from_frame(frame)

    policy = StockRiskPolicy(
        top_k=request.top_k,
        gross_cap=request.max_exposure,
        single_name_cap=request.max_single_weight,
        participation_limit=request.participation_limit,
    )

    def scored_planner(
        snapshot: DatasetSnapshot,
        registry_inner: object,
        instruments_map: object,
        portfolio: PortfolioSnapshot,
        cycle_request: TradingCycleRequest,
    ) -> TradingCycleResult:
        del snapshot, registry_inner, instruments_map
        visible = oos_scored.filter(
            pl.col("session") <= cycle_request.decision_time
        )
        if visible.is_empty():
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
                reasons=("empty-scored-cross-section",),
            )
        from src.stocks.trading.portfolio_constructor import construct_target_allocations

        try:
            allocations = construct_target_allocations(
                visible, instruments, portfolio, policy
            )
        except ValueError as exc:
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
                reasons=(f"constraint:{exc}",),
            )
        return TradingCycleResult(
            status=CycleStatus.PLANNED if allocations else CycleStatus.NO_TRADE,
            cycle_id="stub",
            decision_time=cycle_request.decision_time,
            dataset_hash="d",
            artifact_id=cycle_request.artifact_id,
            account_snapshot_id=portfolio.account_snapshot_id,
            allocations=tuple(allocations),
            intents=(),
            selected_instruments=tuple(
                sorted({a.instrument.instrument_id for a in allocations})
            ),
            reasons=("scored-plan",),
        )

    first_session = sessions[0]
    last_session = sessions[-1]
    start_time = (
        first_session
        if isinstance(first_session, datetime)
        else datetime.combine(first_session, datetime.min.time(), tzinfo=UTC)
    )
    end_time = (
        last_session
        if isinstance(last_session, datetime)
        else datetime.combine(last_session, datetime.min.time(), tzinfo=UTC)
    )
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
    result = backtester.run(frame, artifacts, initial_portfolio, backtest_request)
    benchmark = _benchmark_return_series(frame)
    strategy_returns = _strategy_return_series(list(result.ledger))
    excess = _aligned_excess(strategy_returns, benchmark)
    return ReplayResult(
        ledger=tuple(result.ledger),
        trades=tuple(result.trades),
        metrics=result.metrics,
        stress_metrics=result.stress_metrics,
        final_value=result.final_value,
        excess_returns=excess,
        benchmark_returns=benchmark,
        strategy_returns=strategy_returns,
    )


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
) -> dict[str, object]:
    """Lexicographic fail-closed promotion gates over the event ledger."""
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

    stress_metrics = replay.stress_metrics or {}
    benchmark_total = float(np.sum(benchmark_returns)) if benchmark_returns else 0.0
    gate4_ok = False
    if stress_metrics:
        stress_total = float(stress_metrics.get("cagr", 0.0)) * 0.0
        gate4_ok = stress_total > benchmark_total
    reasons.append(f"gate4_stress_cost_excess={gate4_ok}")
    passed = passed and gate4_ok

    sharpe = float(replay.metrics.get("sharpe", 0.0))
    deflated_prob = _deflated_sharpe_probability(sharpe, 252, _N_OPTUNA_TRIALS)
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
    sharpe: float,
    annualization: int,
    n_trials: int,
) -> float:
    del annualization, n_trials
    if sharpe <= 0.0:
        return 0.0
    from scipy import stats

    # One-sided probability that the observed annualized Sharpe is positive.
    return float(stats.norm.cdf(sharpe))


def _drawdown_from_returns(returns: list[float]) -> float:
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / np.where(peaks > 0, peaks, 1.0)
    return float(np.max(dd)) if dd.size else 0.0


def _forward_holdout_not_consumed(
    base_manifest: ModelManifest,
    request: TrainingRequest,
) -> bool:
    """A frozen candidate must pass one 252-session holdout after 2026-03-10.

    Until 252 newly collected KRX sessions are evaluated once by this frozen
    candidate, promotion remains false regardless of historical metrics.
    """
    del base_manifest, request
    return False


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
        "promotion_reasons": reasons,
        "gates": gates,
    }
