"""Stock model-training workflow: snapshot -> folds -> champion/NO_TRADE artifact.

The workflow evaluates every eligible outer fold, fits factor directions,
weights, and portfolio parameters on training rows only, computes ranking and
ledger metrics under base and stress costs, and publishes either an immutable
champion or an immutable ``NO_TRADE`` artifact. Promotion is fail-closed and
lexicographic; a failing gate never relaxes parameters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.instruments import AssetKind
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.research.datasets import research_eligible_frame, validate_stock_rows_available
from src.stocks.research.features import build_features, phase1_allowlist
from src.stocks.research.folds import Fold, PurgedWalkForward
from src.stocks.research.labels import LabelDefinition
from src.stocks.research.models import (
    ModelManifest,
    RankICConfig,
    StableRankComposite,
)
from src.stocks.trading.allocation_policy import AllocationPolicy
from src.stocks.trading.simulator import SimResult, StockSimulator
from src.stocks.workflows.contracts import TrainingRequest

logger = logging.getLogger("stocks.workflows.train_model")

_ECONOMIC_COLUMNS = ("open", "high", "low", "close", "volume", "trading_value", "market_cap")


@dataclass(frozen=True, slots=True)
class PromotionRiskBudget:
    """Versioned risk budget enforced by the promotion gates."""

    max_drawdown: float = 0.35
    max_turnover: float = 20.0
    max_cost_drag: float = 0.05
    min_exposure: float = 0.1
    holdout_deviation_ratio: float = 2.0


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

    base = request.base_cost_schedule or default_base_schedule()
    stress = request.stress_cost_schedule or default_stress_schedule()

    frame = research_eligible_frame(snapshot.frame)
    decision_time = frame["available_time"].max()
    if not isinstance(decision_time, datetime):
        raise ValueError("panel must carry a datetime available_time")
    validate_stock_rows_available(frame, decision_time)

    features = phase1_allowlist()
    feature_frame = build_features(frame, features)

    label = LabelDefinition(
        name=manifest.label_definition,
        entry_field="open",
        exit_field="close",
        horizon_sessions=manifest.label_horizon_sessions,
    )
    labeled = label.apply(feature_frame).with_columns(
        pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
    )

    splitter = PurgedWalkForward(
        n_folds=request.n_folds,
        label_horizon_sessions=manifest.label_horizon_sessions,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
    )
    folds = splitter.split(labeled)
    if not folds:
        raise ValueError("no folds available for training")

    eligible_from, eligible_to = _eligibility_from_folds(labeled, folds, manifest.label_horizon_sessions)
    base_manifest = ModelManifest(
        artifact_id=request.artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set=manifest.feature_set,
        feature_schema_hash=manifest.schema_hash,
        universe_policy_hash=manifest.universe_policy_hash,
        label_definition=manifest.label_definition,
        label_horizon_sessions=manifest.label_horizon_sessions,
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="stable_rank_composite",
    )

    config = RankICConfig(
        seed=request.seed,
        n_bootstrap=request.n_bootstrap,
        alpha=request.bootstrap_alpha,
    )
    factor_names = tuple(f.name for f in features)
    fold_models: list[StableRankComposite] = []
    scored_frames: list[pl.DataFrame] = []
    fold_rank_ic: list[float] = []
    for fold in folds:
        train_frame = labeled[fold.train_mask]
        validation_frame = labeled[fold.validation_mask]
        inner_folds = splitter.inner_folds(train_frame)
        model = StableRankComposite(
            factors=factor_names,
            manifest=base_manifest,
            label_column=label.name,
            config=config,
            block_length=manifest.label_horizon_sessions,
            session_column="session",
        )
        model.fit(train_frame, validation_frame, inner_folds=inner_folds)
        predict_input = _drop_target_columns(validation_frame, label.name)
        scored = model.predict(predict_input)
        scored_frames.append(scored)
        fold_models.append(model)
        fold_rank_ic.append(_median_rank_ic(validation_frame, scored, label.name))

    oos = pl.concat(scored_frames)
    _reject_non_finite_economic_inputs(oos)

    policy = AllocationPolicy(
        top_k=request.top_k,
        max_single_weight=request.max_single_weight,
        max_exposure=request.max_exposure,
        participation_limit=request.participation_limit,
        portfolio_value=request.portfolio_value,
    )
    simulator = StockSimulator(
        cost_schedule=base,
        initial_cash=request.initial_cash,
        adtv_participation_limit=request.participation_limit,
        stress_schedule=stress,
    )
    sim_result = simulator.simulate(oos, policy, AssetKind.STOCK)
    benchmark_returns = _benchmark_return_series(oos)
    strategy_returns = _strategy_return_series(sim_result.ledger)
    excess = _aligned_excess(strategy_returns, benchmark_returns)

    budget = PromotionRiskBudget()
    reasons: list[str] = []
    gates = _evaluate_gates(
        sim_result,
        excess,
        benchmark_returns,
        budget,
        request,
        labeled,
        fold_rank_ic,
        manifest.label_horizon_sessions,
    )
    reasons.extend(cast(list[str], gates["reasons"]))

    holdout_ok, holdout_ic = _evaluate_holdout(
        labeled, request, config, factor_names, base_manifest, label.name, fold_rank_ic, budget
    )
    reasons.append(f"gate5_holdout_ic={holdout_ic:.8f}")
    passed = bool(gates["passed"]) and holdout_ok

    model = (
        fold_models[-1]
        if passed and fold_models
        else StableRankComposite(
            factors=factor_names,
            manifest=base_manifest,
            label_column=label.name,
            config=config,
            block_length=manifest.label_horizon_sessions,
            session_column="session",
        )
    )
    published_manifest = model.manifest()
    registry.publish(model, published_manifest)
    registry.write_metrics(request.artifact_id, _build_metrics(
        request,
        sim_result,
        fold_rank_ic,
        gates,
        reasons,
        benchmark_returns,
        excess,
        published_manifest,
    ))
    logger.info(
        "published %s artifact %s (promoted=%s)",
        "champion" if passed else "NO_TRADE",
        request.artifact_id,
        passed,
    )
    return published_manifest


def _eligibility_from_folds(
    labeled: pl.DataFrame,
    folds: list[Fold],
    horizon_sessions: int,
) -> tuple[str, str]:
    sessions = sorted(labeled["session"].unique().to_list())
    if not sessions:
        raise ValueError("no sessions available for eligibility")
    val_sessions: list[datetime] = []
    for fold in folds:
        rows = labeled[fold.validation_mask]
        if rows.is_empty():
            continue
        val_sessions.extend(rows["session"].to_list())
    if not val_sessions:
        raise ValueError("no validation sessions for eligibility")
    first = min(val_sessions)
    last = max(val_sessions)
    position = sessions.index(last)
    end = sessions[min(len(sessions) - 1, position + horizon_sessions)]
    return first.isoformat(), end.isoformat()


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
            returns.append(_as_float(value))
    return returns


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"non-numeric benchmark value: {value!r}")


def _strategy_return_series(ledger: list[dict[str, object]]) -> list[float]:
    returns: list[float] = []
    for i in range(1, len(ledger)):
        prev = _as_float(ledger[i - 1]["equity"])
        if prev > 0:
            returns.append(math.log(_as_float(ledger[i]["equity"]) / prev))
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
    sim_result: SimResult,
    excess: list[float],
    benchmark_returns: list[float],
    budget: PromotionRiskBudget,
    request: TrainingRequest,
    labeled: pl.DataFrame,
    fold_rank_ic: list[float],
    label_horizon_sessions: int,
) -> dict[str, object]:
    reasons: list[str] = []
    passed = True

    median_excess = float(np.median(excess)) if excess else 0.0
    lower_bound = (
        _moving_block_bootstrap_lower_bound(
            excess,
            max(label_horizon_sessions, 1),
            request.n_bootstrap,
            request.seed,
            request.bootstrap_alpha,
        )
        if excess
        else 0.0
    )
    gate2_ok = median_excess > 0.0 and lower_bound > 0.0
    reasons.append(f"gate2_median_excess={median_excess:.8f}")
    reasons.append(f"gate2_lower_bound={lower_bound:.8f}")
    passed = passed and gate2_ok

    m = sim_result.metrics
    gate3_ok = (
        bool(fold_rank_ic)
        and m["max_drawdown"] <= budget.max_drawdown
        and m["turnover"] <= budget.max_turnover
        and m["cost_drag"] <= budget.max_cost_drag
        and m["exposure"] >= budget.min_exposure
        and min(fold_rank_ic) > 0.0
    )
    reasons.append(f"gate3_drawdown={m['max_drawdown']:.4f}")
    reasons.append(f"gate3_turnover={m['turnover']:.4f}")
    reasons.append(f"gate3_cost_drag={m['cost_drag']:.4f}")
    reasons.append(f"gate3_exposure={m['exposure']:.4f}")
    passed = passed and gate3_ok

    benchmark_total = float(np.sum(benchmark_returns)) if benchmark_returns else 0.0
    stress_ok = False
    if sim_result.stress_metrics is not None:
        if sim_result.stress_final_value is None:
            raise RuntimeError("stress simulation did not produce a final value")
        stress_total = math.log(sim_result.stress_final_value / request.initial_cash)
        stress_ok = stress_total > benchmark_total
        reasons.append(f"gate4_stress_total={stress_total:.8f}")
        reasons.append(f"gate4_benchmark_total={benchmark_total:.8f}")
    passed = passed and stress_ok

    return {
        "passed": passed,
        "reasons": reasons,
        "median_excess": median_excess,
        "excess_lower_bound": lower_bound,
        "benchmark_total_return": benchmark_total,
    }


def _evaluate_holdout(
    labeled: pl.DataFrame,
    request: TrainingRequest,
    config: RankICConfig,
    factor_names: tuple[str, ...],
    base_manifest: ModelManifest,
    label_name: str,
    fold_rank_ic: list[float],
    budget: PromotionRiskBudget,
) -> tuple[bool, float]:
    if request.holdout_sessions <= 0:
        return True, 0.0
    splitter = PurgedWalkForward(
        n_folds=request.n_folds,
        label_horizon_sessions=5,
        embargo_sessions=request.embargo_sessions,
        session_column="session_index",
    )
    splitter.pin_holdout(request.artifact_id)
    holdout_fold = splitter.holdout(labeled, request.holdout_sessions)
    model = StableRankComposite(
        factors=factor_names,
        manifest=base_manifest,
        label_column=label_name,
        config=config,
        block_length=5,
        session_column="session",
    )
    model.fit(labeled[holdout_fold.train_mask], labeled[holdout_fold.validation_mask])
    scored = model.predict(_drop_target_columns(labeled[holdout_fold.validation_mask], label_name))
    holdout_ic = _median_rank_ic(labeled[holdout_fold.validation_mask], scored, label_name)
    splitter.mark_holdout_inspected(request.artifact_id)
    median_oos = float(np.median(fold_rank_ic)) if fold_rank_ic else 0.0
    consistent = (
        holdout_ic * median_oos >= 0.0
        and abs(holdout_ic - median_oos)
        <= budget.holdout_deviation_ratio * max(abs(median_oos), 1e-12)
    )
    return consistent, holdout_ic


def _build_metrics(
    request: TrainingRequest,
    sim_result: SimResult,
    fold_rank_ic: list[float],
    gates: dict[str, object],
    reasons: list[str],
    benchmark_returns: list[float],
    excess: list[float],
    manifest: ModelManifest,
) -> dict[str, object]:
    return {
        "artifact_id": request.artifact_id,
        "model_type": manifest.model_type,
        "promoted": bool(gates["passed"]),
        "no_trade": not bool(gates["passed"]),
        "eligible_from": manifest.eligible_from,
        "eligible_to": manifest.eligible_to,
        "n_folds_evaluated": len(fold_rank_ic),
        "median_rank_ic": float(np.median(fold_rank_ic)) if fold_rank_ic else 0.0,
        "rank_ic_coverage": sum(1 for ic in fold_rank_ic if ic != 0.0) / max(len(fold_rank_ic), 1),
        "median_excess_return": gates.get("median_excess", 0.0),
        "excess_lower_bound": gates.get("excess_lower_bound", 0.0),
        "benchmark_total_return": gates.get("benchmark_total_return", 0.0),
        "ledger_metrics": sim_result.metrics,
        "stress_metrics": sim_result.stress_metrics,
        "promotion_reasons": reasons,
        "gates": gates,
    }
