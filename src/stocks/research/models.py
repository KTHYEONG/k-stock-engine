"""Model protocol and deterministic ranking composite.

A model is an interchangeable implementation behind a protocol, not the
architecture. ``StableRankComposite`` is the phase-1 strategy derivation model:
it fits factor directions and weights on training rows only and turns weak or
unstable evidence into zero exposure. It may validly produce a ``NO_TRADE``
model whose predictions are all-zero scores.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl

from src.core.instruments import AssetKind

TARGET_PREFIXES = ("target_", "label_")
_V2_FEATURE_PREFIX = "feature__"


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Immutable model metadata; no ``latest`` alias is permitted."""

    artifact_id: str
    asset_kind: AssetKind
    feature_set: str
    feature_schema_hash: str
    universe_policy_hash: str
    label_definition: str
    label_horizon_sessions: int
    eligible_from: str
    eligible_to: str
    model_type: str = "baseline"
    params: dict[str, str] | None = None

    @property
    def eligible_time_range(self) -> tuple[str, str]:
        return (self.eligible_from, self.eligible_to)


@runtime_checkable
class Model(Protocol):
    """Interchangeable model implementation contract."""

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None: ...

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    def manifest(self) -> ModelManifest: ...


class DeterministicBaseline:
    """Deterministic, data-independent ranking model.

    Scores instruments by a configured ranking feature so the full pipeline
    (dataset -> folds -> artifact -> score -> portfolio) can be validated without
    any fitted parameters.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        ranking_feature: str = "feature_momentum_5d",
        descending: bool = True,
    ):
        self._manifest = manifest
        self._ranking_feature = ranking_feature
        self._descending = descending

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        if self._ranking_feature not in train.columns:
            raise ValueError(f"missing ranking feature {self._ranking_feature!r} in training fold")

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._ranking_feature not in frame.columns:
            raise ValueError(f"missing ranking feature {self._ranking_feature!r} in frame")
        score_expr = (
            pl.col(self._ranking_feature)
            if self._descending
            else -pl.col(self._ranking_feature)
        )
        return frame.with_columns(score_expr.alias("pred_score"))

    def manifest(self) -> ModelManifest:
        return self._manifest


@dataclass(frozen=True, slots=True)
class RankICConfig:
    """Moving-block-bootstrap and winsorization settings for factor derivation."""

    seed: int = 42
    n_bootstrap: int = 200
    alpha: float = 0.05
    winsor_low: float = 0.01
    winsor_high: float = 0.99

    def __post_init__(self) -> None:
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        if not (0.0 < self.alpha < 0.5):
            raise ValueError("alpha must be in (0, 0.5)")


class StableRankComposite:
    """Factor-direction and weight estimation from training Rank-IC evidence.

    Fits on training rows only: daily Spearman Rank-IC per factor, orientation
    by median inner-fold IC sign, and a moving-block-bootstrap lower confidence
    bound (block length at least the label horizon) drives retention and
    weights. If no factor survives, the model is ``NO_TRADE``.
    """

    def __init__(
        self,
        factors: tuple[str, ...],
        manifest: ModelManifest,
        label_column: str,
        config: RankICConfig | None = None,
        block_length: int | None = None,
        session_column: str = "session",
    ):
        if not factors:
            raise ValueError("factors must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        self.factors = factors
        self._manifest = manifest
        self.label_column = label_column
        self.session_column = session_column
        self.config = config or RankICConfig()
        self._block_length = max(block_length or manifest.label_horizon_sessions, 1)
        self._factor_weights: dict[str, float] = {}
        self._orientation: dict[str, float] = {}
        self._winsor: dict[str, tuple[float, float]] = {}
        self._resolved_factors: dict[str, str | None] = {}
        self._no_trade = True

    @property
    def no_trade(self) -> bool:
        return self._no_trade

    @property
    def factor_weights(self) -> dict[str, float]:
        return dict(self._factor_weights)

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        inner_folds: Sequence[object] | None = None,
    ) -> None:
        """Fit factor directions and weights on ``train`` rows only."""
        del validation
        if self.label_column not in train.columns:
            raise ValueError(f"missing label column {self.label_column!r} in training fold")
        self._resolved_factors = {
            factor: self._resolve_factor_column(train, factor) for factor in self.factors
        }
        missing = [f for f in self.factors if self._resolved_factors[f] is None]
        if missing:
            raise ValueError(f"missing factor columns {missing} in training fold")

        daily_ics = {
            factor: self._daily_rank_ic(train, factor) for factor in self.factors
        }
        fold_ics: dict[str, list[float]] = {f: [] for f in self.factors}
        for fold in inner_folds or []:
            mask = getattr(fold, "train_mask", None)
            if mask is None:
                continue
            sub = train[mask]
            for factor in self.factors:
                fold_ics[factor].extend(self._daily_rank_ic(sub, factor))

        orientations: dict[str, float] = {}
        for factor in self.factors:
            pool = fold_ics[factor] if fold_ics[factor] else daily_ics[factor]
            median_ic = float(np.median(pool)) if pool else 0.0
            orientations[factor] = 1.0 if median_ic > 0.0 else -1.0

        lower_bounds: dict[str, float] = {}
        for factor in self.factors:
            oriented = [orientations[factor] * ic for ic in daily_ics[factor]]
            lower_bounds[factor] = self._bootstrap_lower_bound(oriented)

        weights: dict[str, float] = {}
        for factor in self.factors:
            if lower_bounds[factor] > 0.0:
                weights[factor] = lower_bounds[factor]
        total = sum(weights.values())
        self._no_trade = total <= 0.0
        self._factor_weights = (
            {f: w / total for f, w in weights.items()} if total > 0.0 else {}
        )
        self._orientation = orientations
        self._winsor = {
            factor: self._winsor_quantiles(train, factor) for factor in self.factors
        }

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Score a panel; reject frames carrying target or label columns."""
        self._reject_target_columns(frame)
        if self.no_trade:
            return frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("pred_score"))
        if self.session_column not in frame.columns:
            raise ValueError(f"missing session column {self.session_column!r}")
        weighted_factors = [
            f for f in self.factors if self._factor_weights.get(f, 0.0) != 0.0
        ]
        if not weighted_factors:
            return frame.with_columns(pl.lit(0.0).alias("pred_score"))
        rank_exprs: list[pl.Expr] = []
        for factor in weighted_factors:
            column = self._factor_column(factor)
            lo, hi = self._winsor[factor]
            clipped = pl.col(column).clip(lo, hi)
            within = pl.col(column).count().over(self.session_column)
            rank = (
                ((clipped.rank("average").over(self.session_column) - 1.0) / (within - 1.0))
                .fill_null(0.5)
                .alias(f"__rank_{factor}")
            )
            rank_exprs.append(rank)
        ranked = frame.with_columns(rank_exprs)
        score = pl.sum_horizontal(
            [
                pl.col(f"__rank_{factor}")
                * self._orientation[factor]
                * self._factor_weights[factor]
                for factor in weighted_factors
            ]
        )
        result = ranked.with_columns(score.alias("pred_score"))
        return result.drop([c for c in result.columns if c.startswith("__rank_")])

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "factors": ",".join(self.factors),
            "factor_weights": ",".join(
                f"{factor}={self._factor_weights.get(factor, 0.0):.9f}"
                for factor in self.factors
            ),
            "seed": str(self.config.seed),
            "no_trade": str(self.no_trade).lower(),
        }
        return ModelManifest(
            artifact_id=self._manifest.artifact_id,
            asset_kind=self._manifest.asset_kind,
            feature_set=self._manifest.feature_set,
            feature_schema_hash=self._manifest.feature_schema_hash,
            universe_policy_hash=self._manifest.universe_policy_hash,
            label_definition=self._manifest.label_definition,
            label_horizon_sessions=self._manifest.label_horizon_sessions,
            eligible_from=self._manifest.eligible_from,
            eligible_to=self._manifest.eligible_to,
            model_type="stable_rank_composite",
            params=params,
        )

    def _reject_target_columns(self, frame: pl.DataFrame) -> None:
        offending = [
            c
            for c in frame.columns
            if c.startswith(TARGET_PREFIXES) or c == self.label_column
        ]
        if offending:
            raise ValueError(f"predict rejects target/label columns: {offending}")

    def _daily_rank_ic(self, frame: pl.DataFrame, factor: str) -> list[float]:
        column = self._factor_column(factor)
        sub = frame.filter(pl.col(column).is_not_null() & pl.col(self.label_column).is_not_null())
        ics: list[float] = []
        if sub.is_empty() or self.session_column not in sub.columns:
            return ics
        for rows in sub.sort(self.session_column).partition_by(self.session_column):
            scores = rows[column].to_numpy().astype(float)
            labels = rows[self.label_column].to_numpy().astype(float)
            if len(scores) < 2 or np.std(scores) == 0.0 or np.std(labels) == 0.0:
                continue
            rs = np.argsort(np.argsort(scores))
            rl = np.argsort(np.argsort(labels))
            rs = rs - rs.mean()
            rl = rl - rl.mean()
            denom = np.sqrt(np.sum(rs * rs) * np.sum(rl * rl))
            ics.append(float(np.sum(rs * rl) / denom) if denom > 0.0 else 0.0)
        return ics

    def _bootstrap_lower_bound(self, values: list[float]) -> float:
        values_arr = np.asarray(values, dtype=float)
        n = values_arr.size
        if n == 0:
            return 0.0
        rng = np.random.default_rng(self.config.seed)
        block = min(max(self._block_length, 1), n)
        n_blocks = max(1, math.ceil(n / block))
        means = np.empty(self.config.n_bootstrap)
        for b in range(self.config.n_bootstrap):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            sample = np.concatenate(
                [values_arr[start : start + block] for start in starts]
            )[:n]
            means[b] = np.mean(sample[:n])
        return float(np.quantile(means, self.config.alpha))

    def _winsor_quantiles(self, frame: pl.DataFrame, factor: str) -> tuple[float, float]:
        column = self._factor_column(factor)
        values = frame[column].drop_nulls().to_numpy().astype(float)
        if values.size == 0:
            return (0.0, 0.0)
        return (
            float(np.quantile(values, self.config.winsor_low)),
            float(np.quantile(values, self.config.winsor_high)),
        )

    def _factor_column(self, factor: str) -> str:
        column = self._resolved_factors.get(factor, factor)
        if column is None:
            raise ValueError(f"factor {factor!r} has no resolved column")
        return column

    @staticmethod
    def _resolve_factor_column(frame: pl.DataFrame, factor: str) -> str | None:
        if factor in frame.columns:
            return factor
        prefixed = f"{_V2_FEATURE_PREFIX}{factor}"
        if prefixed in frame.columns:
            return prefixed
        return None
