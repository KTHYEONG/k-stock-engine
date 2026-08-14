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
from lightgbm.callback import CallbackEnv

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
STABLE_WEIGHT = 1.0 - LAMBDARANK_WEIGHT
LAMBDARANK_ABLATION_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
_V2_FEATURE_PREFIX = "feature__"

_FULL_REFIT_ROUND_CAP = 900
_FULL_REFIT_PATIENCE = 100


@dataclass(frozen=True, slots=True)
class FitTrialOutcome:
    """Typed outcome of one ``fit_trial`` call.

    ``fit_ok`` mirrors the legacy boolean contract. ``best_iteration`` is the
    LightGBM 1-based best validation iteration (or ``None`` when the fit did not
    produce a booster), ``stopped_early`` reports whether training terminated on
    the early-stopping rule before the round cap, and ``rounds_trained`` is the
    number of boosting rounds actually executed. ``used_continuation`` is set by
    the adaptive full-refit path and ``fallback_to_one_shot`` records that a
    parity check failed and the one-shot reference result was used instead.
    """

    fit_ok: bool
    best_iteration: int | None = None
    stopped_early: bool = False
    rounds_trained: int = 0
    used_continuation: bool = False
    fallback_to_one_shot: bool = False


def adaptive_refit_rounds(proxy_best_iteration: int | None) -> int:
    """First-pass full-refit round budget derived from the proxy best iteration.

    ``initial_rounds = min(900, max(2 * 100, proxy_best_iteration + 100))`` per
    the v2 selection contract: a promoted candidate always starts with at least
    two full early-stopping patience windows so the first pass can itself stop
    early, and never beyond the 900-round cap.
    """
    proxy = proxy_best_iteration if proxy_best_iteration is not None else 0
    return min(
        _FULL_REFIT_ROUND_CAP,
        max(2 * _FULL_REFIT_PATIENCE, proxy + _FULL_REFIT_PATIENCE),
    )


def resolve_lgb_num_threads(
    requested_threads: int | None,
    physical_cores: int,
    logical_cores: int,
) -> int:
    """Resolve the deterministic LightGBM thread plan.

    ``None`` selects every cgroup-visible physical core (the measured 3.46x
    sweet spot for the production fold). An explicit value must be positive and
    must not exceed the visible logical CPUs; a non-positive value or an
    oversubscribed value raises ``ValueError``. There is no silent
    oversubscription and the resolution never samples the value from Optuna.
    """
    if physical_cores < 1:
        raise ValueError("physical_cores must be positive")
    if logical_cores < 1:
        raise ValueError("logical_cores must be positive")
    if requested_threads is not None:
        if requested_threads < 1:
            raise ValueError("lgb_threads must be positive")
        if requested_threads > logical_cores:
            raise ValueError(
                f"lgb_threads {requested_threads} exceeds the visible logical "
                f"CPU count {logical_cores}"
            )
        return requested_threads
    return min(physical_cores, logical_cores)


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
        num_threads: int = 1,
        lambdarank_weight: float = LAMBDARANK_WEIGHT,
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
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        if not 0.0 <= float(lambdarank_weight) <= 1.0:
            raise ValueError("lambdarank_weight must be within [0.0, 1.0]")
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
        self.num_threads = num_threads
        self.lambdarank_weight = float(lambdarank_weight)

    @property
    def candidate_family(self) -> str:
        """The registered ablation family for this blend weight.

        Endpoints ``0.00`` and ``1.00`` classify exactly as ``stable_only`` and
        ``ml_only``; interior weights map to their explicit blend family (for
        example ``0.25`` -> ``blend_25``). Classification is provenance only and
        never alters prediction mathematics.
        """
        weight = self.lambdarank_weight
        if weight == 0.0:
            return "stable_only"
        if weight == 1.0:
            return "ml_only"
        return f"blend_{round(weight * 100):02d}"

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
            "num_threads": self.num_threads,
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
        self._close_error: str | None = None

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

    def close(self) -> None:
        """Idempotently release native LightGBM resources held by a fitted model.

        Frees the fitted Booster's Datasets through the supported
        ``Booster.free_dataset`` API when a booster is fitted, then clears the
        booster and cached stable-score references so a temporary model never
        retains a native allocator or a duplicate prediction frame. Predictions
        already materialized by a caller are never altered and prepared
        immutable fold matrices (:class:`PreparedLambdaRankFold`) are never
        mutated. A native cleanup error on a live fit is never silently
        suppressed: it is recorded as a deterministic caller-visible reason in
        :attr:`close_error`. Calling ``close()`` again, including after an
        error, is safe.
        """
        booster = self._booster
        if booster is not None:
            try:
                booster.free_dataset()
            except Exception as exc:  # noqa: BLE001
                self._close_error = (
                    f"native_cleanup_failed:{type(exc).__name__}:{exc}"
                )
                return
            self._booster = None
        self._stable_scores_cache = None
        self._close_error = None

    @property
    def close_error(self) -> str | None:
        """Deterministic native-cleanup failure reason, or ``None`` after a clean close."""
        return self._close_error

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
        initial_rounds: int | None = None,
    ) -> FitTrialOutcome:
        """Fit only the LambdaRank booster, reusing cached stable scores.

        ``stable_scores`` must carry ``(session, instrument_id, pred_score)``
        and is invariant across every LambdaRank search parameter, so the trial
        fast path never refits the StableRank composite. When ``prepared`` is
        supplied, the immutable fold matrices are reused and no Polars
        filtering, sorting, group, weight, or matrix construction is repeated.
        Supplied ``callbacks`` are appended to the LightGBM early-stopping
        callback and a callback-raised ``optuna.TrialPruned`` propagates.

        When ``initial_rounds`` is supplied the booster is trained adaptively:
        a first pass runs up to ``initial_rounds`` rounds and, if that pass
        reaches its round budget without satisfying early stopping, the identical
        Booster is continued with ``init_model`` in deterministic
        ``early_stopping_rounds`` chunks until the configured ``n_estimators``
        cap or the early-stopping rule fires. The continuation never restarts
        from a random state, alters seeds, lowers folds, or accepts a capped
        model as converged. Returns a :class:`FitTrialOutcome`; the legacy
        public ``fit``/``predict`` path remains the final-model path and yields
        identical blend scores for identical data and configuration.
        """
        self._stable_scores_cache = stable_scores
        outcome = self._fit_lambdarank(
            train,
            validation,
            prepared=prepared,
            callbacks=callbacks,
            initial_rounds=initial_rounds,
        )
        self._no_trade = not outcome.fit_ok
        return outcome

    def fit_trial_prepared(
        self,
        prepared: PreparedLambdaRankFold,
        stable_scores: pl.DataFrame,
        *,
        callbacks: Sequence[Callable[..., object]] = (),
        initial_rounds: int | None = None,
    ) -> FitTrialOutcome:
        """Fit only the LambdaRank booster on the immutable prepared fold.

        Identical to :meth:`fit_trial` with ``prepared`` supplied: the cached
        stable scores are bound and the booster is trained from the prepared
        matrices. The prepared prediction fast path consumes the same validation
        matrix, so no Polars filtering, matrix conversion, or predictor-frame
        construction is repeated for scoring.
        """
        self._stable_scores_cache = stable_scores
        outcome = self._fit_lambdarank_prepared(
            prepared,
            callbacks,
            initial_rounds=initial_rounds,
        )
        self._no_trade = not outcome.fit_ok
        return outcome

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
            self.config.lambdarank_weight * lambda_rank
            + (1.0 - self.config.lambdarank_weight) * stable_rank
        ).alias("pred_score")
        result = scored.with_columns(blend)
        result = result.drop("__lambda_score", "__stable_score")
        if self._calibration_state is not None:
            from src.stocks.research.economic_alpha import CausalAlphaCalibrator

            result = CausalAlphaCalibrator.from_state(
                self._calibration_state
            ).apply_frozen(result)
        return result

    def predict_prepared_scores(
        self,
        prepared: PreparedLambdaRankFold,
        validation_index: pl.DataFrame,
        stable_scores: pl.DataFrame,
    ) -> pl.DataFrame:
        """Cross-sectional 50/50 percentile-rank blend on the prepared validation matrix.

        ``validation_index`` must carry ``(session, instrument_id)`` aligned
        row-for-row with ``prepared.validation_matrix``; ``stable_scores`` must
        carry ``(session, instrument_id, pred_score)`` for the same keys. The
        booster is called directly on the immutable matrix and the blend is
        computed over exactly those rows, returning a slim frame containing only
        ``session``, ``instrument_id``, and ``pred_score``. Misaligned rows,
        duplicated keys, non-finite scores, or an un-fitted booster raise
        ``ValueError``. Parity with the public :meth:`predict` is exact on
        aligned inputs.
        """
        if self._no_trade or self._booster is None:
            raise ValueError("no fitted booster is available for prepared prediction")
        if prepared.validation_matrix is None:
            raise ValueError("prepared fold exposes no validation matrix")
        required_index = (self.session_column, "instrument_id")
        if not all(c in validation_index.columns for c in required_index):
            raise ValueError(f"validation_index must carry {required_index}")
        required_stable = (*required_index, "pred_score")
        if not all(c in stable_scores.columns for c in required_stable):
            raise ValueError(f"stable_scores must carry {required_stable}")
        if validation_index.height != prepared.validation_matrix.shape[0]:
            raise ValueError(
                "validation_index row count must match the prepared "
                f"validation matrix ({validation_index.height} != "
                f"{prepared.validation_matrix.shape[0]})"
            )
        duplicated = int(validation_index.select(required_index).is_duplicated().sum())
        if duplicated:
            raise ValueError("validation_index keys must be unique")
        stable_dup = int(
            stable_scores.select(required_index).is_duplicated().sum()
        )
        if stable_dup:
            raise ValueError("stable_scores keys must be unique")
        lambda_pred = np.asarray(
            self._booster.predict(prepared.validation_matrix), dtype=float
        )
        if not np.all(np.isfinite(lambda_pred)):
            raise ValueError("non-finite LambdaRank predictions")
        scored = (
            validation_index.with_columns(
                pl.Series("__lambda_score", lambda_pred)
            )
            .join(
                stable_scores.select(*required_stable),
                on=required_index,
                how="left",
            )
            .rename({"pred_score": "__stable_score"})
            .with_columns(pl.col("__stable_score").fill_null(0.0))
        )
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
            self.config.lambdarank_weight * lambda_rank
            + (1.0 - self.config.lambdarank_weight) * stable_rank
        ).alias("pred_score")
        result = scored.select(self.session_column, "instrument_id", blend)
        if result["pred_score"].null_count() or not np.all(
            np.isfinite(result["pred_score"].to_numpy())
        ):
            raise ValueError("non-finite or missing blended prediction")
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
            "blend_weight_lambdarank": f"{self.config.lambdarank_weight:.6f}",
            "blend_weight_stable": f"{1.0 - self.config.lambdarank_weight:.6f}",
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
        initial_rounds: int | None = None,
    ) -> FitTrialOutcome:
        if prepared is not None:
            return self._fit_lambdarank_prepared(
                prepared, callbacks, initial_rounds=initial_rounds
            )
        missing = [c for c in self.features if not self._resolve_column(train, c)]
        if missing:
            self._excluded_features = missing
            logger.info("lambda component missing feature columns %s", missing)
            return FitTrialOutcome(fit_ok=False)
        if self.relevance_column not in train.columns:
            logger.info("lambda component missing relevance column")
            return FitTrialOutcome(fit_ok=False)

        self._predictor_columns = self._resolve_predictor_columns(train)
        if not self._predictor_columns:
            return FitTrialOutcome(fit_ok=False)

        usable = train.filter(
            pl.col(self.relevance_column).is_not_null()
        )
        for column in self._predictor_columns:
            bad = usable.filter(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
            )
            if not bad.is_empty():
                raise ValueError(f"non-finite predictor value in {column}")
        if usable.is_empty():
            return FitTrialOutcome(fit_ok=False)

        group_sizes, session_order = self._group_sizes(usable)
        if not group_sizes:
            return FitTrialOutcome(fit_ok=False)
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
            free_raw_data=False,
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
                    free_raw_data=False,
                )

        outcome = self._train_booster(
            train_set,
            valid_set,
            callbacks=callbacks,
            initial_rounds=initial_rounds,
        )
        if not outcome.fit_ok:
            return outcome
        assert self._booster is not None
        importance = self._booster.feature_importance("gain")
        self._feature_gains = dict(
            zip(
                self._predictor_columns,
                (float(v) for v in importance),
                strict=True,
            )
        )
        self._missing_rates = {}
        for name in self.features:
            raw = self._resolve_column(train, name)
            if raw is None:
                self._missing_rates[name] = 0.0
                continue
            self._missing_rates[name] = float(train[raw].null_count()) / train.height
        return outcome

    def _fit_lambdarank_prepared(
        self,
        prepared: PreparedLambdaRankFold,
        callbacks: Sequence[Callable[..., object]],
        *,
        initial_rounds: int | None = None,
    ) -> FitTrialOutcome:
        """Train the booster on pre-prepared immutable fold matrices."""
        if not prepared.predictor_columns:
            return FitTrialOutcome(fit_ok=False)
        self._predictor_columns = list(prepared.predictor_columns)
        self._train_group_count = len(prepared.train_group_sizes)
        train_set = lgb.Dataset(
            prepared.train_matrix,
            label=prepared.train_relevance,
            group=prepared.train_group_sizes,
            weight=prepared.train_weights,
            params={"verbosity": -1},
            free_raw_data=False,
        )
        valid_set: lgb.Dataset | None = None
        if prepared.validation_group_sizes:
            valid_set = lgb.Dataset(
                prepared.validation_matrix,
                label=prepared.validation_relevance,
                group=prepared.validation_group_sizes,
                params={"verbosity": -1},
                free_raw_data=False,
            )
        outcome = self._train_booster(
            train_set,
            valid_set,
            callbacks=callbacks,
            initial_rounds=initial_rounds,
        )
        if not outcome.fit_ok:
            return outcome
        assert self._booster is not None
        importance = self._booster.feature_importance("gain")
        self._feature_gains = dict(
            zip(
                self._predictor_columns,
                (float(v) for v in importance),
                strict=True,
            )
        )
        return outcome

    def _train_booster(
        self,
        train_set: lgb.Dataset,
        valid_set: lgb.Dataset | None,
        *,
        callbacks: Sequence[Callable[..., object]],
        initial_rounds: int | None,
    ) -> FitTrialOutcome:
        """Deterministic adaptive boosting driver with a global early-stop rule.

        The non-adaptive path mirrors the legacy ``lgb.train`` with the
        LightGBM early-stopping callback. The adaptive path (``initial_rounds``
        supplied) runs a first pass of ``initial_rounds`` rounds with early
        stopping, then continues the identical Booster through ``init_model``
        in ``early_stopping_rounds`` chunks until the configured
        ``n_estimators`` cap or the global early-stopping rule fires. The best
        iteration is always recomputed from the recorded per-round validation
        metric, so chunk re-baselining can never shift the terminal best round.
        """
        total_rounds = self.config.n_estimators
        patience = self.config.early_stopping_rounds
        evaluation = _EvalTracker()

        def _first_pass(init_model: lgb.Booster | None, num_rounds: int) -> lgb.Booster:
            callback_chain: list[Callable[..., object]] = [
                lgb.early_stopping(patience, verbose=False),
                evaluation.record,
            ]
            callback_chain.extend(callbacks)
            return lgb.train(
                self.config.lgb_params(),
                train_set,
                num_boost_round=num_rounds,
                valid_sets=[valid_set] if valid_set is not None else [train_set],
                init_model=init_model,
                callbacks=callback_chain,
            )

        def _continuation_chunk(init_model: lgb.Booster, num_rounds: int) -> lgb.Booster:
            callback_chain: list[Callable[..., object]] = [
                evaluation.record,
            ]
            callback_chain.extend(callbacks)
            return lgb.train(
                self.config.lgb_params(),
                train_set,
                num_boost_round=num_rounds,
                valid_sets=[valid_set] if valid_set is not None else [train_set],
                init_model=init_model,
                callbacks=callback_chain,
            )

        if initial_rounds is None:
            self._booster = _first_pass(None, total_rounds)
            best_iteration = evaluation.best_iteration_1based(total_rounds)
            self._booster = _truncate_booster(self._booster, best_iteration)
            return FitTrialOutcome(
                fit_ok=True,
                best_iteration=best_iteration,
                stopped_early=bool(
                    best_iteration is not None and best_iteration < total_rounds
                ),
                rounds_trained=total_rounds,
            )

        used_continuation = False
        rounds_trained = 0
        first = max(1, min(initial_rounds, total_rounds))
        booster = _first_pass(None, first)
        rounds_trained = first
        best_iteration = evaluation.best_iteration_1based(rounds_trained)
        stopped_early = bool(
            best_iteration is not None and rounds_trained - best_iteration >= patience
        )
        while (
            not stopped_early
            and rounds_trained < total_rounds
            and best_iteration is not None
        ):
            used_continuation = True
            chunk = min(patience, total_rounds - rounds_trained)
            booster = _continuation_chunk(booster, chunk)
            rounds_trained += chunk
            best_iteration = evaluation.best_iteration_1based(rounds_trained)
            stopped_early = bool(
                best_iteration is not None and rounds_trained - best_iteration >= patience
            )
        self._booster = _truncate_booster(booster, best_iteration)
        return FitTrialOutcome(
            fit_ok=True,
            best_iteration=best_iteration,
            stopped_early=stopped_early,
            rounds_trained=rounds_trained,
            used_continuation=used_continuation,
        )

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
        to keep the fail-closed semantics of the uncached path. Rows are
        filtered only by relevance and minimum group eligibility; null
        predictors are never deleted.
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
            bad = usable.filter(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
            )
            if not bad.is_empty():
                raise ValueError(f"non-finite predictor value in {column}")
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
        """Manifest-ordered rank, sector-rank, and missing-indicator columns.

        Raw source feature levels never enter the LightGBM design matrix; for
        every manifest feature the resolved source column must expose the three
        derived predictors (``__rank``, ``__sector_rank``, ``__missing``). A
        missing derived column fails closed with ``ValueError``.
        """
        columns: list[str] = []
        for name in self.features:
            raw = self._resolve_column(frame, name)
            if raw is None:
                continue
            for suffix in ("__rank", "__sector_rank", "__missing"):
                candidate = raw + suffix
                if candidate not in frame.columns:
                    raise ValueError(
                        f"missing derived predictor column {candidate!r} "
                        f"for feature {name!r}"
                    )
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
        max_position = float(positions.max()) if positions.size else 0.0
        half_life = self.config.half_life_sessions
        age_from_newest = max_position - positions
        recency = np.exp2(-age_from_newest / float(half_life))
        weights = np.empty(len(ordered), dtype=float)
        start = 0
        for size in group_sizes:
            if size > 0:
                weights[start : start + size] = 1.0 / size
            start += size
        result = (weights * recency).astype(float)
        if len(result) != len(ordered):
            raise ValueError("observation weight length must match the ordered row count")
        if not np.all(np.isfinite(result)):
            raise ValueError("observation weights must be finite")
        return result

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


class _EvalTracker:
    """Cumulative per-round validation-metric collector across boosting chunks.

    The first non-training metric of each round (the validation ``ndcg@10`` for
    the pinned LambdaRank contract) is appended in training order so the global
    best iteration can be recovered deterministically across ``init_model``
    continuation chunks, where LightGBM's own early-stopping callback would
    otherwise re-baseline its best score at every chunk boundary.
    """

    def __init__(self) -> None:
        self._series: list[float] = []

    def record(self, env: CallbackEnv) -> None:
        for result in env.evaluation_result_list or ():
            data_name = result[0]
            if not isinstance(data_name, str):
                continue
            if data_name.startswith("train"):
                continue
            value = result[2]
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                self._series.append(number)
                return

    def best_iteration_1based(self, rounds_trained: int) -> int | None:
        """1-based global best iteration over the recorded rounds.

        Falls back to ``rounds_trained`` when no valid metric was recorded so a
        degenerate fit never claims a ``None`` best round.
        """
        if not self._series:
            return None
        best_index = int(np.argmax(self._series))
        return best_index + 1


def _truncate_booster(
    booster: lgb.Booster,
    best_iteration: int | None,
) -> lgb.Booster:
    """Rebuild the booster with exactly ``best_iteration`` trees.

    Mirrors the LightGBM early-stopping contract where the trained model is
    rolled back to the best round so ``predict`` defaults to ``best_iteration``.
    A missing best iteration returns the full booster unchanged.
    """
    if best_iteration is None or best_iteration < 1:
        return booster
    total = booster.num_trees()
    if best_iteration >= total:
        return booster
    return lgb.Booster(
        model_str=booster.model_to_string(num_iteration=best_iteration)
    )


def verify_adaptive_parity(
    continuation: FitTrialOutcome,
    one_shot: FitTrialOutcome,
    *,
    booster: lgb.Booster | None,
    reference_booster: lgb.Booster | None,
    predict_input: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> bool:
    """True when adaptive continuation reproduces the one-shot reference.

    Compares the terminal best iteration, early-stop status, and the raw
    booster predictions at the reference best iteration within the pinned
    float tolerance. Used by the parity proof on fixtures and the fixed
    production sample; a mismatch means the caller must rerun the one-shot
    reference and emit ``adaptive_refit_fallback=true``.
    """
    if continuation.best_iteration != one_shot.best_iteration:
        return False
    if continuation.stopped_early != one_shot.stopped_early:
        return False
    if booster is None or reference_booster is None or one_shot.best_iteration is None:
        return True
    predicted = booster.predict(predict_input, num_iteration=one_shot.best_iteration)
    reference = reference_booster.predict(
        predict_input, num_iteration=one_shot.best_iteration
    )
    delta = float(np.max(np.abs(np.asarray(predicted) - np.asarray(reference))))
    return delta <= atol + rtol * float(np.max(np.abs(reference)))
