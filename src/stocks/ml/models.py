"""Deterministic continuous net-alpha models and out-of-fold calibration.

The learner stack is an explicit baseline/challenger pair:

- ``ElasticNetNetAlpha`` fits a deterministic ``ElasticNet`` on fold-local
  standardized rank features — the stable linear benchmark.
- ``LightGbmNetAlpha`` is the L1-regression challenger with pinned seed and
  deterministic thread/leaf settings.
- ``NetAlphaCalibrator`` fits a monotone calibration on OOF predictions only,
  converting a score into an expected net alpha and a decimal lower bound,
  without ever reading an in-fold target. Calibration targets are decimal
  realized returns (``risk_residual - reference_cost``), never the
  MAD-standardized ``net_alpha_target``.
- ``CalibratedNetAlphaModel`` wraps a fitted learner plus the fitted
  calibration so the persisted artifact applies the decimal lower bound to
  every production prediction.

A challenger is accepted only when its paired OOF incremental policy-utility
block-bootstrap lower bound over the baseline is strictly positive; otherwise
the baseline is retained or both are ``NO_TRADE``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet

from src.stocks.research.models import Model, ModelManifest

TARGET_PREFIXES = ("target_", "label_")
SCORE_COLUMN = "predicted_net_alpha"
MIN_BUCKET_OBSERVATIONS = 5


@dataclass(frozen=True, slots=True)
class NetAlphaModelConfig:
    """Deterministic training configuration shared by baseline and challenger."""

    seed: int = 42
    num_leaves: int = 31
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    n_estimators: int = 500
    early_stopping_rounds: int = 50
    elastic_alpha: float = 0.5
    elastic_l1_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if not 0.0 < self.feature_fraction <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1]")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.elastic_alpha < 0:
            raise ValueError("elastic_alpha must be non-negative")
        if not 0.0 <= self.elastic_l1_ratio <= 1.0:
            raise ValueError("elastic_l1_ratio must be in [0, 1]")


def _float32_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Contiguous Float32 design matrix from declared columns."""
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


def _reject_target_columns(frame: pl.DataFrame, label_column: str) -> None:
    offending = [
        c
        for c in frame.columns
        if c.startswith(TARGET_PREFIXES) or c == label_column
    ]
    if offending:
        raise ValueError(f"predict rejects target/label columns: {offending}")


class ElasticNetNetAlpha:
    """Deterministic ElasticNet baseline on fold-standardized rank features."""

    def __init__(
        self,
        manifest: ModelManifest,
        feature_columns: tuple[str, ...],
        label_column: str,
        config: NetAlphaModelConfig | None = None,
    ):
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        self._manifest = manifest
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._config = config or NetAlphaModelConfig()
        self._model: ElasticNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        del validation
        if self._label_column not in train.columns:
            raise ValueError(
                f"missing label column {self._label_column!r} in training fold"
            )
        missing = [c for c in self._feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in training fold")
        features = _float32_matrix(train, self._feature_columns)
        targets = train[self._label_column].cast(pl.Float64).to_numpy()
        finite = np.isfinite(targets)
        valid = np.isfinite(features).all(axis=1) & finite
        if not valid.any():
            raise ValueError("training fold has no finite feature/target rows")
        features = features[valid]
        targets = targets[valid]
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0.0] = 1.0
        standardized = (features - self._mean) / self._std
        self._model = ElasticNet(
            alpha=self._config.elastic_alpha,
            l1_ratio=self._config.elastic_l1_ratio,
            max_iter=2000,
            random_state=self._config.seed,
        )
        self._model.fit(standardized, targets)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._model is None:
            raise ValueError("model is not fitted")
        _reject_target_columns(frame, self._label_column)
        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in predict frame")
        features = _float32_matrix(frame, self._feature_columns)
        if self._mean is None or self._std is None:
            raise ValueError("model has no frozen fold statistics")
        standardized = (features - self._mean) / self._std
        standardized[~np.isfinite(standardized)] = 0.0
        scores = self._model.predict(standardized)
        return frame.with_columns(
            pl.Series(SCORE_COLUMN, scores.astype(np.float64))
        )

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "family": "elastic_net",
            "alpha": f"{self._config.elastic_alpha:.6g}",
            "l1_ratio": f"{self._config.elastic_l1_ratio:.6g}",
            "seed": str(self._config.seed),
            "feature_columns": ",".join(self._feature_columns),
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
            model_type="net_alpha_elastic_net",
            params=params,
        )


class LightGbmNetAlpha:
    """Deterministic L1 LightGBM regressor challenger on rank features."""

    def __init__(
        self,
        manifest: ModelManifest,
        feature_columns: tuple[str, ...],
        label_column: str,
        config: NetAlphaModelConfig | None = None,
        num_threads: int = 1,
    ):
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not label_column:
            raise ValueError("label_column must not be empty")
        if num_threads < 1:
            raise ValueError("num_threads must be positive")
        self._manifest = manifest
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._config = config or NetAlphaModelConfig()
        self._num_threads = num_threads
        self._booster: lgb.Booster | None = None
        self._best_iteration: int | None = None

    def _params(self) -> dict[str, object]:
        config = self._config
        return {
            "objective": "regression_l1",
            "metric": "mae",
            "num_leaves": config.num_leaves,
            "learning_rate": config.learning_rate,
            "max_depth": config.max_depth,
            "min_child_samples": config.min_child_samples,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 1.0,
            "max_bin": 255,
            "num_threads": self._num_threads,
            "seed": config.seed,
            "deterministic": True,
            "force_col_wise": True,
            "data_random_seed": config.seed,
            "feature_fraction_seed": config.seed,
            "bagging_seed": config.seed,
            "verbosity": -1,
        }

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        if self._label_column not in train.columns:
            raise ValueError(
                f"missing label column {self._label_column!r} in training fold"
            )
        missing = [c for c in self._feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in training fold")
        train_features = _float32_matrix(train, self._feature_columns)
        train_targets = train[self._label_column].cast(pl.Float64).to_numpy()
        finite = np.isfinite(train_targets)
        if not finite.any():
            raise ValueError("training fold has no finite target rows")
        train_features = train_features[finite]
        train_targets = train_targets[finite]
        train_set = lgb.Dataset(
            train_features, label=train_targets, params={"verbosity": -1}
        )
        valid_set = None
        if (
            not validation.is_empty()
            and self._label_column in validation.columns
            and all(c in validation.columns for c in self._feature_columns)
        ):
            valid_features = _float32_matrix(validation, self._feature_columns)
            valid_targets = validation[self._label_column].cast(pl.Float64).to_numpy()
            vfinite = np.isfinite(valid_targets)
            if vfinite.any():
                valid_set = lgb.Dataset(
                    valid_features[vfinite],
                    label=valid_targets[vfinite],
                    reference=train_set,
                    params={"verbosity": -1},
                )
        self._booster = lgb.train(
            self._params(),
            train_set,
            num_boost_round=self._config.n_estimators,
            valid_sets=[valid_set] if valid_set is not None else None,
            callbacks=[
                lgb.early_stopping(
                    self._config.early_stopping_rounds,
                    verbose=False,
                    min_delta=0.0,
                )
            ],
        )
        if self._booster.best_iteration > 0:
            self._best_iteration = self._booster.best_iteration

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._booster is None:
            raise ValueError("model is not fitted")
        _reject_target_columns(frame, self._label_column)
        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns {missing} in predict frame")
        features = _float32_matrix(frame, self._feature_columns)
        features[~np.isfinite(features)] = 0.0
        scores = np.asarray(
            self._booster.predict(features, num_iteration=self._best_iteration),
            dtype=np.float64,
        )
        return frame.with_columns(pl.Series(SCORE_COLUMN, scores))

    def manifest(self) -> ModelManifest:
        params: dict[str, str] = {
            "family": "lightgbm_l1",
            "objective": "regression_l1",
            "num_leaves": str(self._config.num_leaves),
            "learning_rate": f"{self._config.learning_rate:.6g}",
            "min_child_samples": str(self._config.min_child_samples),
            "seed": str(self._config.seed),
            "num_threads": str(self._num_threads),
            "feature_columns": ",".join(self._feature_columns),
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
            model_type="net_alpha_lightgbm_l1",
            params=params,
        )


@dataclass(frozen=True, slots=True)
class BucketEvidence:
    """One calibration bucket's OOF evidence."""

    bucket: int
    sample_size: int
    expected_alpha: float
    alpha_lower_bound: float


@dataclass(frozen=True, slots=True)
class CalibrationState:
    """Frozen calibration table for prediction-time application."""

    buckets: tuple[BucketEvidence, ...]
    bucket_count: int
    boundaries: tuple[float, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "bucket_count": self.bucket_count,
            "boundaries": list(self.boundaries),
            "buckets": [
                {
                    "bucket": b.bucket,
                    "sample_size": b.sample_size,
                    "expected_alpha": b.expected_alpha,
                    "alpha_lower_bound": b.alpha_lower_bound,
                }
                for b in self.buckets
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> CalibrationState:
        """Rehydrate a frozen calibration table from its JSON serialization."""
        raw_buckets = payload.get("buckets") or []
        if not isinstance(raw_buckets, list):
            raise ValueError("calibration state 'buckets' must be a list")
        buckets = tuple(
            BucketEvidence(
                bucket=int(item["bucket"]),
                sample_size=int(item["sample_size"]),
                expected_alpha=float(item["expected_alpha"]),
                alpha_lower_bound=float(item["alpha_lower_bound"]),
            )
            for item in raw_buckets
        )
        raw_boundaries = payload.get("boundaries") or ()
        boundaries = tuple(
            float(value)
            for value in (raw_boundaries if isinstance(raw_boundaries, list) else ())
        )
        raw_count = payload.get("bucket_count", 0)
        bucket_count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
        return cls(
            buckets=buckets,
            bucket_count=bucket_count,
            boundaries=boundaries,
        )


class NetAlphaCalibrator:
    """Monotone OOF-only calibration of scores to expected net alpha.

    Bucket boundaries and statistics are fit on out-of-fold predictions only;
    the in-fold targets are never read at calibration time. Each bucket is
    usable only when it holds at least ``MIN_BUCKET_OBSERVATIONS`` observations
    and its seeded block-bootstrap lower bound is strictly positive; otherwise
    the bucket is treated as a non-trade signal.

    The calibration label is a decimal realized return (``risk_residual -
    reference_cost``), never the MAD-standardized ``net_alpha_target``, so the
    ``net_alpha_lower_bound`` it emits is an economic return in compatible
    units with replay transaction costs.
    """

    def __init__(
        self,
        bucket_count: int,
        seed: int = 42,
        n_bootstrap: int = 200,
        bootstrap_alpha: float = 0.05,
        block_length: int = 5,
        label_column: str = "realized_net_return",
    ):
        if bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")
        self._bucket_count = bucket_count
        self._seed = seed
        self._n_bootstrap = n_bootstrap
        self._bootstrap_alpha = bootstrap_alpha
        self._block_length = block_length
        self._label_column = label_column
        self._state: CalibrationState | None = None

    @property
    def state(self) -> CalibrationState:
        """The frozen calibration table; raises until ``fit`` has run."""
        if self._state is None:
            raise ValueError("calibrator has no frozen state; call fit first")
        return self._state

    def state_json(self) -> dict[str, object]:
        """JSON serialization of the frozen calibration table."""
        return self.state.to_json()

    def load_state(self, payload: dict[str, object]) -> NetAlphaCalibrator:
        """Restore a frozen calibration table from its JSON serialization."""
        self._state = CalibrationState.from_json(payload)
        return self

    def fit(self, oof: pl.DataFrame) -> CalibrationState:
        """Fit monotone bucket statistics on OOF predictions only."""
        if oof.is_empty():
            raise ValueError("cannot calibrate an empty OOF panel")
        required = (SCORE_COLUMN, self._label_column, "session")
        missing = [c for c in required if c not in oof.columns]
        if missing:
            raise ValueError(f"OOF panel missing columns {missing}")
        scored = oof.filter(
            pl.col(SCORE_COLUMN).is_not_null()
            & pl.col(self._label_column).is_not_null()
            & pl.col(SCORE_COLUMN).is_finite()
            & pl.col(self._label_column).is_finite()
        )
        if scored.is_empty():
            raise ValueError("OOF panel has no finite score/label rows")

        score_values = scored[SCORE_COLUMN].to_numpy().astype(float)
        quantiles = np.linspace(0.0, 1.0, self._bucket_count + 1)[1:-1]
        boundaries = tuple(float(v) for v in np.quantile(score_values, quantiles))
        boundaries = tuple(sorted(set(boundaries)))
        bucket_expr = (
            pl.col(SCORE_COLUMN)
            .cut(boundaries, left_closed=False)
            .to_physical()
            .cast(pl.Int32)
        )
        joined = scored.with_columns(bucket_expr.alias("__bucket"))
        buckets: list[BucketEvidence] = []
        for bucket in range(self._bucket_count):
            rows = joined.filter(pl.col("__bucket") == bucket)
            if rows.height < MIN_BUCKET_OBSERVATIONS:
                continue
            labels = rows[self._label_column].to_numpy().astype(float)
            if not np.all(np.isfinite(labels)):
                continue
            bound = self._lower_bound(labels)
            if bound <= 0.0:
                continue
            buckets.append(
                BucketEvidence(
                    bucket=bucket,
                    sample_size=rows.height,
                    expected_alpha=float(np.mean(labels)),
                    alpha_lower_bound=bound,
                )
            )
        if not buckets:
            state = CalibrationState(
                buckets=(), bucket_count=self._bucket_count, boundaries=boundaries
            )
        else:
            buckets.sort(key=lambda b: b.expected_alpha)
            state = CalibrationState(
                buckets=tuple(buckets),
                bucket_count=self._bucket_count,
                boundaries=boundaries,
            )
        self._state = state
        return state

    def _lower_bound(self, labels: np.ndarray) -> float:
        n = labels.size
        block = min(max(self._block_length, 1), n)
        n_blocks = int(np.ceil(n / block))
        max_start = max(1, n - block + 1)
        rng = np.random.default_rng(self._seed)
        starts = rng.integers(0, max_start, size=(self._n_bootstrap, n_blocks))
        offsets = np.arange(block)
        index = (starts[:, :, None] + offsets[None, None, :]).reshape(
            self._n_bootstrap, n_blocks * block
        )[:, :n]
        means = labels[index].mean(axis=1)
        return float(np.quantile(means, self._bootstrap_alpha))

    def apply(self, scored: pl.DataFrame) -> pl.DataFrame:
        """Apply the frozen calibration to a scored panel."""
        if self._state is None:
            raise ValueError("calibrator has no frozen state; call fit first")
        return _apply_calibration(scored, self._state)


def _apply_calibration(
    scored: pl.DataFrame, state: CalibrationState
) -> pl.DataFrame:
    """Attach expected net alpha and lower bound columns from the bucket table."""
    if not state.buckets:
        return scored.with_columns(
            pl.lit(0.0, dtype=pl.Float64).alias("expected_net_alpha"),
            pl.lit(0.0, dtype=pl.Float64).alias("net_alpha_lower_bound"),
        )
    boundaries = list(state.boundaries) or sorted(
        {b.expected_alpha for b in state.buckets}
    )
    bucket_expr = (
        pl.col(SCORE_COLUMN)
        .cut(boundaries, left_closed=False)
        .to_physical()
        .cast(pl.Int32)
    )
    table = pl.DataFrame(
        {
            "bucket": [b.bucket for b in state.buckets],
            "expected_net_alpha": [b.expected_alpha for b in state.buckets],
            "net_alpha_lower_bound": [b.alpha_lower_bound for b in state.buckets],
        }
    )
    augmented = scored.with_columns(bucket_expr.alias("bucket_idx"))
    merged = augmented.join(
        table, left_on="bucket_idx", right_on="bucket", how="left"
    ).with_columns(
        pl.col("expected_net_alpha").fill_null(0.0),
        pl.col("net_alpha_lower_bound").fill_null(0.0),
    )
    return merged.drop("bucket_idx")


class CalibratedNetAlphaModel:
    """Learner plus fitted calibration persisted as one production artifact.

    ``predict`` delegates to the wrapped learner and then attaches the frozen
    decimal ``net_alpha_lower_bound``, so production planning consumes the same
    economic score the replay gated on. The wrapper is joblib-serialized with
    the calibration table; ``manifest`` also records the serialized calibration
    for auditability.
    """

    def __init__(self, model: Model, calibrator: NetAlphaCalibrator):
        if model is None:
            raise ValueError("CalibratedNetAlphaModel requires a base model")
        if calibrator is None:
            raise ValueError("CalibratedNetAlphaModel requires a fitted calibrator")
        self._model = model
        self._calibrator = calibrator

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        self._model.fit(train, validation)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        scored = self._model.predict(frame)
        return self._calibrator.apply(scored)

    def manifest(self) -> ModelManifest:
        base = self._model.manifest()
        params = dict(base.params or {})
        params["calibration_state"] = json.dumps(
            self._calibrator.state.to_json(), sort_keys=True
        )
        return replace(base, params=params)
