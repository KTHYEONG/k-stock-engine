"""Deterministic LightGBM LambdaRank blended with the stable rank composite.

``LambdaRankBlendModel`` is the v2 champion candidate: a frozen 50/50 blend of
the cross-sectional percentile rank of a LightGBM ``lambdarank`` ranking and the
cross-sectional percentile rank of the train-only :class:`StableRankComposite`.
Weights are frozen before evaluation and never silently redistributed; if either
component cannot fit, the artifact is ``NO_TRADE``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from src.core.instruments import AssetKind
from src.stocks.research.labels import (
    LAMBDARANK_GAIN,
    MIN_LAMBDARANK_GROUP,
    RELEVANCE_COLUMN,
)
from src.stocks.research.models import (
    TARGET_PREFIXES,
    ModelManifest,
    StableRankComposite,
)

logger = logging.getLogger("stocks.research.lambdarank")

LAMBDARANK_WEIGHT = 0.50
STABLE_WEIGHT = 0.50
_V2_FEATURE_PREFIX = "feature__"


@dataclass(frozen=True, slots=True)
class PreparedLambdaRankFold:
    """Immutable per-tuning-fold inputs shared by every search candidate.

    All matrices and relevance labels are prepared once per fold and reused
    verbatim by every ``fit_trial`` call so a candidate never repeats Polars
    filtering, sorting, matrix conversion, group construction, or
    observation-weight construction. Arrays are C-contiguous with the pinned
    dtype and are read-only after preparation.
    """

    train_matrix: np.ndarray
    train_relevance: np.ndarray
    train_group_sizes: list[int]
    train_weights: np.ndarray
    validation_matrix: np.ndarray | None
    validation_relevance: np.ndarray | None
    validation_group_sizes: list[int] | None
    predictor_columns: list[str]


def _as_readonly(array: np.ndarray) -> np.ndarray:
    """Pin an array read-only so no later candidate can mutate cached inputs."""
    array.setflags(write=False)
    return array


class LambdaRankConfig:
    """Frozen deterministic LambdaRank training and search contract.

    All LightGBM seeds are pinned to 42; training is deterministic column-wise
    CPU. ``label_gain`` and ``eval_at`` are the ranking evaluation contract.
    """

    _tuning_telemetry: dict[str, object] | None = None
    _calibration_state: dict[str, object] | None = None

    def __init__(
        self,
        *,
        objective: str = "lambdarank",
        metric: str = "ndcg",
        label_gain: tuple[int, ...] = LAMBDARANK_GAIN,
        eval_at: tuple[int, int] = (10, 20),
        seed: int = 42,
        num_leaves: int = 31,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        min_child_samples: int = 500,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.9,
        bagging_freq: int = 1,
        lambda_l1: float = 0.0,
        lambda_l2: float = 1.0,
        max_bin: int = 255,
        n_estimators: int = 5000,
        early_stopping_rounds: int = 200,
        min_group_size: int = MIN_LAMBDARANK_GROUP,
        half_life_sessions: int = 504,
    ):
        if objective != "lambdarank":
            raise ValueError("objective must be lambdarank")
        if tuple(label_gain) != tuple(LAMBDARANK_GAIN):
            raise ValueError("label_gain must be (0, 1, 3, 7, 15)")
        if num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")
        if half_life_sessions < 1:
            raise ValueError("half_life_sessions must be positive")
        self.objective = objective
        self.metric = metric
        self.label_gain = tuple(label_gain)
        self.eval_at = tuple(eval_at)
        self.seed = seed
        self.data_random_seed = seed
        self.feature_fraction_seed = seed
        self.bagging_seed = seed
        self.deterministic = True
        self.force_col_wise = True
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.max_bin = max_bin
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.min_group_size = min_group_size
        self.half_life_sessions = half_life_sessions

    def lgb_params(self) -> dict[str, object]:
        """Deterministic LightGBM parameters with every seed pinned."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "label_gain": list(self.label_gain),
            "eval_at": list(self.eval_at),
            "seed": self.seed,
            "data_random_seed": self.data_random_seed,
            "feature_fraction_seed": self.feature_fraction_seed,
            "bagging_seed": self.bagging_seed,
            "deterministic": self.deterministic,
            "force_col_wise": self.force_col_wise,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "verbosity": -1,
            "num_threads": 1,
        }


class LambdaRankBlendModel:
    """Frozen 50/50 LambdaRank + stable composite blend."""

    def __init__(
        self,
        manifest: ModelManifest,
        features: tuple[str, ...],
        label_column: str,
        config: LambdaRankConfig | None = None,
        session_column: str = "session",
        relevance_column: str = RELEVANCE_COLUMN,
    ):
        if not features:
            raise ValueError("features must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        self._manifest = manifest
        self.features = tuple(features)
        self.label_column = label_column
        self.config = config or LambdaRankConfig()
        self.session_column = session_column
        self.relevance_column = relevance_column
        self._stable = StableRankComposite(
            factors=self.features,
            manifest=manifest,
            label_column=label_column,
            block_length=manifest.label_horizon_sessions,
            session_column=session_column,
        )
        self._booster: lgb.Booster | None = None
        self._feature_gains: dict[str, float] = {}
        self._missing_rates: dict[str, float] = {}
        self._excluded_features: list[str] = []
        self._predictor_columns: list[str] = []
        self._train_group_count = 0
        self._no_trade = True
        self._stable_scores_cache: pl.DataFrame | None = None
        self._calibration_state: dict[str, object] | None = None

    @property
    def no_trade(self) -> bool:
        return self._no_trade

    @property
    def stable(self) -> StableRankComposite:
        return self._stable

    @property
    def calibration_state(self) -> dict[str, object] | None:
        """Frozen causal net-alpha calibration evidence, or ``None``."""
        return self._calibration_state

    def set_calibration_state(self, state: dict[str, object] | None) -> None:
        """Bind the JSON-safe calibration snapshot used by prediction columns."""
        self._calibration_state = None if state is None else dict(state)

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        """Fit both components on training rows only."""
        self._stable_scores_cache = None
        self._stable.fit(train, validation)
        lambda_ok = self._fit_lambdarank(train, validation)
        self._no_trade = not (lambda_ok and not self._stable.no_trade)

    def fit_trial(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        stable_scores: pl.DataFrame,
        *,
        prepared: PreparedLambdaRankFold | None = None,
        callbacks: Sequence[Callable[..., object]] = (),
    ) -> bool:
        """Fit only the LambdaRank booster, reusing cached stable scores.

        ``stable_scores`` must carry ``(session, instrument_id, pred_score)``
        and is invariant across every LambdaRank search parameter, so the trial
        fast path never refits the StableRank composite. When ``prepared`` is
        supplied, the immutable fold matrices are reused and no Polars
        filtering, sorting, group, weight, or matrix construction is repeated.
        Supplied ``callbacks`` are appended to the LightGBM early-stopping
        callback and a callback-raised ``optuna.TrialPruned`` propagates.
        Returns whether the booster fit succeeded; the legacy public
        ``fit``/``predict`` path remains the final-model path and yields
        identical blend scores for identical data and configuration.
        """
        self._stable_scores_cache = stable_scores
        lambda_ok = self._fit_lambdarank(
            train, validation, prepared=prepared, callbacks=callbacks
        )
        self._no_trade = not lambda_ok
        return lambda_ok

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Target-free cross-sectional 50/50 percentile-rank blend."""
        self._reject_target_columns(frame)
        if self._no_trade or self._booster is None:
            return frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("pred_score"))
        if self.session_column not in frame.columns:
            raise ValueError(f"missing session column {self.session_column!r}")

        matrix = self._feature_matrix(frame)
        lambda_pred = self._booster.predict(matrix)
        lambda_col = pl.Series("__lambda_score", np.asarray(lambda_pred, dtype=float))
        scored = frame.with_columns(lambda_col)
        if self._stable_scores_cache is not None:
            stable_scored = self._stable_scores_cache
        elif not self._stable.no_trade:
            stable_scored = self._stable.predict(frame).select(
                self.session_column, "instrument_id", "pred_score"
            )
        else:
            stable_scored = None
        if stable_scored is not None:
            scored = scored.join(
                stable_scored,
                on=[self.session_column, "instrument_id"],
                how="left",
            ).rename({"pred_score": "__stable_score"})
        else:
            scored = scored.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("__stable_score"))

        within = pl.col("__lambda_score").count().over(self.session_column)
        lambda_rank = (
            (pl.col("__lambda_score").rank("average").over(self.session_column) - 1.0)
            / (within - 1.0)
        ).fill_null(0.5)
        stable_within = pl.col("__stable_score").count().over(self.session_column)
        stable_rank = (
            (pl.col("__stable_score").rank("average").over(self.session_column) - 1.0)
            / (stable_within - 1.0)
        ).fill_null(0.5)
        blend = (
            LAMBDARANK_WEIGHT * lambda_rank + STABLE_WEIGHT * stable_rank
        ).alias("pred_score")
        result = scored.with_columns(blend)
        result = result.drop("__lambda_score", "__stable_score")
        if self._calibration_state is not None:
            from src.stocks.research.economic_alpha import CausalAlphaCalibrator

            result = CausalAlphaCalibrator.from_state(
                self._calibration_state
            ).apply_frozen(result)
        return result

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "objective": self.config.objective,
            "metric": self.config.metric,
            "label_gain": ",".join(str(g) for g in self.config.label_gain),
            "eval_at": ",".join(str(v) for v in self.config.eval_at),
            "seed": str(self.config.seed),
            "blend_weight_lambdarank": f"{LAMBDARANK_WEIGHT:.6f}",
            "blend_weight_stable": f"{STABLE_WEIGHT:.6f}",
            "feature_list": ",".join(self.features),
            "feature_gains": json.dumps(self._feature_gains, sort_keys=True),
            "missing_rates": json.dumps(self._missing_rates, sort_keys=True),
            "excluded_features": ",".join(self._excluded_features),
            "num_leaves": str(self.config.num_leaves),
            "learning_rate": str(self.config.learning_rate),
            "n_estimators": str(self.config.n_estimators),
            "early_stopping_rounds": str(self.config.early_stopping_rounds),
            "train_group_count": str(self._train_group_count),
            "no_trade": str(self._no_trade).lower(),
            "calibration_state": (
                json.dumps(self._calibration_state, sort_keys=True, default=str)
                if self._calibration_state is not None
                else ""
            ),
        }
        return ModelManifest(
            artifact_id=self._manifest.artifact_id,
            asset_kind=AssetKind.STOCK,
            feature_set=self._manifest.feature_set,
            feature_schema_hash=self._manifest.feature_schema_hash,
            universe_policy_hash=self._manifest.universe_policy_hash,
            label_definition=self._manifest.label_definition,
            label_horizon_sessions=self._manifest.label_horizon_sessions,
            eligible_from=self._manifest.eligible_from,
            eligible_to=self._manifest.eligible_to,
            model_type="lambdarank_blend",
            params=params,
        )

    def _fit_lambdarank(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        *,
        prepared: PreparedLambdaRankFold | None = None,
        callbacks: Sequence[Callable[..., object]] = (),
    ) -> bool:
        if prepared is not None:
            return self._fit_lambdarank_prepared(prepared, callbacks)
        missing = [c for c in self.features if not self._resolve_column(train, c)]
        if missing:
            self._excluded_features = missing
            logger.info("lambda component missing feature columns %s", missing)
            return False
        if self.relevance_column not in train.columns:
            logger.info("lambda component missing relevance column")
            return False

        self._predictor_columns = self._resolve_predictor_columns(train)
        if not self._predictor_columns:
            return False

        usable = train.filter(
            pl.col(self.relevance_column).is_not_null()
        )
        for column in self._predictor_columns:
            usable = usable.filter(pl.col(column).is_not_null())
        if usable.is_empty():
            return False

        group_sizes, session_order = self._group_sizes(usable)
        if not group_sizes:
            return False
        self._train_group_count = len(group_sizes)
        usable = usable.filter(pl.col(self.session_column).is_in(session_order))
        ordered = usable.sort(self.session_column)
        matrix = self._float32_matrix(ordered, self._predictor_columns)
        labels = ordered[self.relevance_column].cast(pl.Int32).to_numpy()
        weights = self._observation_weights(ordered, group_sizes)

        train_set = lgb.Dataset(
            matrix,
            label=labels,
            group=group_sizes,
            weight=weights,
            params={"verbosity": -1},
        )
        valid_set: lgb.Dataset | None = None
        if validation is not None and not validation.is_empty():
            val_used = validation.filter(
                pl.col(self.relevance_column).is_not_null()
            )
            val_group_sizes, _ = self._group_sizes(val_used)
            if val_group_sizes:
                val_ordered = val_used.sort(self.session_column)
                valid_set = lgb.Dataset(
                    self._float32_matrix(val_ordered, self._predictor_columns),
                    label=val_ordered[self.relevance_column].cast(pl.Int32).to_numpy(),
                    group=val_group_sizes,
                    params={"verbosity": -1},
                )

        callback_chain: list[Callable[..., object]] = [
            lgb.early_stopping(self.config.early_stopping_rounds)
        ]
        callback_chain.extend(callbacks)
        self._booster = lgb.train(
            self.config.lgb_params(),
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[valid_set] if valid_set is not None else [train_set],
            callbacks=callback_chain,
        )
        importance = self._booster.feature_importance("gain")
        self._feature_gains = dict(
            zip(
                self._predictor_columns,
                (float(v) for v in importance),
                strict=True,
            )
        )
        self._missing_rates = {
            name: float(train[col].null_count()) / train.height
            for name, col in zip(self.features, self._resolve_predictor_columns(train), strict=False)
        }
        return True

    def _fit_lambdarank_prepared(
        self,
        prepared: PreparedLambdaRankFold,
        callbacks: Sequence[Callable[..., object]],
    ) -> bool:
        """Train the booster on pre-prepared immutable fold matrices."""
        if not prepared.predictor_columns:
            return False
        self._predictor_columns = list(prepared.predictor_columns)
        self._train_group_count = len(prepared.train_group_sizes)
        train_set = lgb.Dataset(
            prepared.train_matrix,
            label=prepared.train_relevance,
            group=prepared.train_group_sizes,
            weight=prepared.train_weights,
            params={"verbosity": -1},
        )
        valid_set: lgb.Dataset | None = None
        if prepared.validation_group_sizes:
            valid_set = lgb.Dataset(
                prepared.validation_matrix,
                label=prepared.validation_relevance,
                group=prepared.validation_group_sizes,
                params={"verbosity": -1},
            )
        callback_chain: list[Callable[..., object]] = [
            lgb.early_stopping(self.config.early_stopping_rounds)
        ]
        callback_chain.extend(callbacks)
        self._booster = lgb.train(
            self.config.lgb_params(),
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[valid_set] if valid_set is not None else [train_set],
            callbacks=callback_chain,
        )
        importance = self._booster.feature_importance("gain")
        self._feature_gains = dict(
            zip(
                self._predictor_columns,
                (float(v) for v in importance),
                strict=True,
            )
        )
        return True

    def prepare_fold(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
    ) -> PreparedLambdaRankFold | None:
        """Derive the immutable cached fold inputs, or ``None`` when unusable.

        Replicates the legacy ``_fit_lambdarank`` derivation exactly so the
        prepared and uncached trial paths produce identical boosters. A fold
        missing features, relevance, predictor rows, or qualifying groups
        returns ``None``; the group-sum invariant still raises ``ValueError``
        to keep the fail-closed semantics of the uncached path.
        """
        missing = [c for c in self.features if not self._resolve_column(train, c)]
        if missing:
            return None
        if self.relevance_column not in train.columns:
            return None
        predictor_columns = self._resolve_predictor_columns(train)
        if not predictor_columns:
            return None

        usable = train.filter(pl.col(self.relevance_column).is_not_null())
        for column in predictor_columns:
            usable = usable.filter(pl.col(column).is_not_null())
        if usable.is_empty():
            return None
        group_sizes, session_order = self._group_sizes(usable)
        if not group_sizes:
            return None
        ordered = (
            usable.filter(pl.col(self.session_column).is_in(session_order))
            .sort(self.session_column)
        )
        train_matrix = _as_readonly(self._float32_matrix(ordered, predictor_columns))
        train_relevance = _as_readonly(
            ordered[self.relevance_column].cast(pl.Int32).to_numpy()
        )
        train_weights = _as_readonly(self._observation_weights(ordered, group_sizes))

        validation_matrix: np.ndarray | None = None
        validation_relevance: np.ndarray | None = None
        validation_group_sizes: list[int] | None = None
        if validation is not None and not validation.is_empty():
            val_used = validation.filter(pl.col(self.relevance_column).is_not_null())
            val_group_sizes, _ = self._group_sizes(val_used)
            if val_group_sizes:
                val_ordered = val_used.sort(self.session_column)
                validation_matrix = _as_readonly(
                    self._float32_matrix(val_ordered, predictor_columns)
                )
                validation_relevance = _as_readonly(
                    val_ordered[self.relevance_column].cast(pl.Int32).to_numpy()
                )
                validation_group_sizes = val_group_sizes

        return PreparedLambdaRankFold(
            train_matrix=train_matrix,
            train_relevance=train_relevance,
            train_group_sizes=group_sizes,
            train_weights=train_weights,
            validation_matrix=validation_matrix,
            validation_relevance=validation_relevance,
            validation_group_sizes=validation_group_sizes,
            predictor_columns=predictor_columns,
        )

    def _resolve_predictor_columns(self, frame: pl.DataFrame) -> list[str]:
        """Manifest-ordered raw plus rank and sector-demeaned rank columns."""
        columns: list[str] = []
        for name in self.features:
            raw = self._resolve_column(frame, name)
            if raw is None:
                continue
            for suffix in ("", "__rank", "__sector_rank"):
                candidate = raw + suffix
                if candidate in frame.columns:
                    columns.append(candidate)
        return columns

    def _group_sizes(
        self, frame: pl.DataFrame
    ) -> tuple[list[int], list[object]]:
        counts = (
            frame.group_by(self.session_column)
            .len()
            .sort(self.session_column)
            .filter(pl.col("len") >= self.config.min_group_size)
        )
        sizes = counts["len"].to_list()
        sessions = counts[self.session_column].to_list()
        total = int(counts["len"].sum()) if sizes else 0
        if total and total != frame.height:
            raise ValueError(
                "lambda group sizes must sum exactly to the row count "
                f"({total} != {frame.height})"
            )
        return [int(s) for s in sizes], sessions

    def _observation_weights(
        self,
        ordered: pl.DataFrame,
        group_sizes: list[int],
    ) -> np.ndarray:
        session_positions = (
            ordered.select(pl.col(self.session_column).unique().alias(self.session_column))
            .sort(self.session_column)
            .with_row_index("__pos")
        )
        joined = ordered.join(session_positions, on=self.session_column)
        positions = joined["__pos"].to_numpy().astype(float)
        half_life = self.config.half_life_sessions
        decay = np.exp2(-positions / float(half_life))
        weights = np.empty(len(ordered), dtype=float)
        start = 0
        for size in group_sizes:
            if size > 0:
                weights[start : start + size] = 1.0 / size
            start += size
        return (weights * decay).astype(float)

    def _feature_matrix(self, frame: pl.DataFrame) -> np.ndarray:
        columns = self._predictor_columns or self._resolve_predictor_columns(frame)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"predict missing feature columns {missing}")
        selected = frame.select(columns)
        for column in columns:
            bad = selected.filter(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
            )
            if not bad.is_empty():
                raise ValueError(f"non-finite feature value in {column}")
        return self._float32_matrix(selected, columns)

    @staticmethod
    def _float32_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
        """Contiguous Float32 design matrix without redundant ``astype`` copies.

        Columns are cast to Float32 in Polars before a single contiguous
        ``to_numpy`` conversion; ``ascontiguousarray`` only copies when the
        resulting array does not already carry the required dtype/contiguity.
        """
        cast_exprs = [
            pl.col(column).cast(pl.Float32)
            if frame.schema[column] != pl.Float32
            else pl.col(column)
            for column in columns
        ]
        array = frame.select(cast_exprs).to_numpy()
        if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array, dtype=np.float32)
        return array

    def _resolve_column(self, frame: pl.DataFrame, name: str) -> str | None:
        if name in frame.columns:
            return name
        prefixed = f"{_V2_FEATURE_PREFIX}{name}"
        if prefixed in frame.columns:
            return prefixed
        return None

    def _reject_target_columns(self, frame: pl.DataFrame) -> None:
        offending = [
            c
            for c in frame.columns
            if c.startswith(TARGET_PREFIXES) or c == self.label_column
        ]
        if offending:
            raise ValueError(f"predict rejects target/label columns: {offending}")
