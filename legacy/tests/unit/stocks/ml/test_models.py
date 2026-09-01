"""Net-alpha models: decimal OOF calibration serialization and champion wrapper."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import polars as pl

from src.core.instruments import AssetKind
from legacy.stocks.ml.models import (
    SCORE_COLUMN,
    CalibrationState,
    CalibratedNetAlphaModel,
    ElasticNetNetAlpha,
    NetAlphaCalibrator,
    NetAlphaModelConfig,
)
from legacy.stocks.research.models import ModelManifest


def _manifest(artifact_id: str = "na_cal") -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_net_alpha_v1",
        feature_schema_hash="net-alpha-v1",
        universe_policy_hash="net-alpha-v1",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )


def _oof_panel(n_rows: int = 120, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    scores = np.sort(rng.uniform(-1.0, 1.0, n_rows))
    realized = 0.02 * scores + rng.normal(0.0, 0.001, n_rows)
    sessions = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 10)]
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i % 8 + 1:05d}" for i in range(n_rows)],
            "session": [sessions[i % len(sessions)] for i in range(n_rows)],
            SCORE_COLUMN: scores,
            "realized_net_return": realized,
        }
    )


def test_calibration_state_json_round_trip() -> None:
    oof = _oof_panel()
    calibrator = NetAlphaCalibrator(
        bucket_count=10,
        seed=42,
        n_bootstrap=50,
        bootstrap_alpha=0.05,
        block_length=5,
        label_column="realized_net_return",
    )
    state = calibrator.fit(oof)
    restored = CalibrationState.from_json(state.to_json())
    assert restored == state
    assert restored.bucket_count == state.bucket_count


def test_calibration_fits_decimal_realized_returns_and_applies() -> None:
    oof = _oof_panel()
    calibrator = NetAlphaCalibrator(
        bucket_count=10,
        seed=42,
        n_bootstrap=50,
        bootstrap_alpha=0.05,
        block_length=5,
        label_column="realized_net_return",
    )
    state = calibrator.fit(oof)
    assert state.buckets
    calibrated = calibrator.apply(oof)
    assert "net_alpha_lower_bound" in calibrated.columns
    assert "expected_net_alpha" in calibrated.columns
    assert calibrated["net_alpha_lower_bound"].min() >= 0.0


def test_calibrated_model_predict_applies_lower_bound() -> None:
    manifest = _manifest()
    oof = _oof_panel().with_columns(
        pl.col("realized_net_return").alias("net_alpha_target"),
        pl.col(SCORE_COLUMN).alias("feature_momentum_5d"),
    )
    base = ElasticNetNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=NetAlphaModelConfig(seed=42),
    )
    base.fit(oof, oof.head(0))
    calibrator = NetAlphaCalibrator(
        bucket_count=5, seed=42, n_bootstrap=50, block_length=3,
        label_column="realized_net_return",
    )
    calibrator.fit(oof)
    wrapped = CalibratedNetAlphaModel(base, calibrator)
    prediction = wrapped.predict(oof.select("instrument_id", "session", "feature_momentum_5d"))
    assert SCORE_COLUMN in prediction.columns
    assert "net_alpha_lower_bound" in prediction.columns
    assert "calibration_state" in (wrapped.manifest().params or {})


def _lightgbm_frame(n_rows: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 12)]
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i % 8 + 1:05d}" for i in range(n_rows)],
            "session": [sessions[i % len(sessions)] for i in range(n_rows)],
            "feature_momentum_5d": rng.normal(0.0, 1.0, n_rows),
            "net_alpha_target": rng.normal(0.0, 1.0, n_rows),
        }
    )


def test_lightgbm_target_free_fit_is_deterministic_and_no_early_stopping(monkeypatch) -> None:
    """A target-free validation frame trains the fixed budget without early stopping."""
    from legacy.stocks.ml import models as models_module
    from legacy.stocks.ml.models import LightGbmNetAlpha

    manifest = _manifest()
    train = _lightgbm_frame()
    empty = train.head(0)
    captured: dict[str, object] = {}
    real_train = models_module.lgb.train

    def spy(*args, **kwargs):
        captured["valid_sets"] = kwargs.get("valid_sets")
        captured["has_early_stopping"] = any(
            (
                "early" in type(cb).__name__.lower()
                and "stopping" in type(cb).__name__.lower()
            )
            for cb in (kwargs.get("callbacks") or [])
        )
        return real_train(*args, **kwargs)

    monkeypatch.setattr(models_module.lgb, "train", spy)
    config = NetAlphaModelConfig(seed=42, n_estimators=20, min_child_samples=10)
    model = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=config, num_threads=1,
    )
    model.fit(train, empty)
    assert captured["valid_sets"] is None
    assert captured["has_early_stopping"] is False
    assert model._best_iteration is None

    predictions = model.predict(
        train.select("instrument_id", "session", "feature_momentum_5d")
    )
    assert predictions[SCORE_COLUMN].n_unique() > 1

    second = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=config, num_threads=1,
    )
    second.fit(train, empty)
    assert np.array_equal(
        predictions[SCORE_COLUMN].to_numpy(),
        second.predict(train.select("instrument_id", "session", "feature_momentum_5d"))[
            SCORE_COLUMN
        ].to_numpy(),
    )


def test_lightgbm_labeled_validation_requests_early_stopping(monkeypatch) -> None:
    """A finite labeled validation set keeps the early-stopping path."""
    from legacy.stocks.ml import models as models_module
    from legacy.stocks.ml.models import LightGbmNetAlpha

    manifest = _manifest()
    train = _lightgbm_frame()
    valid = _lightgbm_frame(seed=8)
    captured: dict[str, object] = {}
    real_train = models_module.lgb.train

    def spy(*args, **kwargs):
        captured["valid_sets"] = kwargs.get("valid_sets")
        captured["has_early_stopping"] = any(
            (
                "early" in type(cb).__name__.lower()
                and "stopping" in type(cb).__name__.lower()
            )
            for cb in (kwargs.get("callbacks") or [])
        )
        return real_train(*args, **kwargs)

    monkeypatch.setattr(models_module.lgb, "train", spy)
    model = LightGbmNetAlpha(
        manifest, ("feature_momentum_5d",), "net_alpha_target",
        config=NetAlphaModelConfig(seed=42, n_estimators=20, min_child_samples=10),
        num_threads=1,
    )
    model.fit(train, valid)
    assert captured["valid_sets"] is not None
    assert captured["has_early_stopping"] is True
    assert model._best_iteration is not None


def test_causal_calibration_adapter_preserves_public_contract() -> None:
    """The causal adapter preserves predicted_net_alpha and appends decimal columns."""
    from legacy.stocks.research.economic_alpha import CausalAlphaCalibrator

    from legacy.stocks.ml.models import CausalCalibrationAdapter

    scored = _oof_panel().select("instrument_id", "session", SCORE_COLUMN)
    state: dict[str, object] = {
        "bucket_count": 3,
        "history_sessions": 10,
        "round_trip_cost": 0.001,
        "exit_cost_rate": 0.0005,
        "buckets": [],
    }
    adapter = CausalCalibrationAdapter(
        CausalAlphaCalibrator(bucket_count=3, min_calibration_sessions=1, seed=1), state
    )
    augmented = adapter.apply(scored)
    assert SCORE_COLUMN in augmented.columns
    assert "expected_net_alpha" in augmented.columns
    assert "net_alpha_lower_bound" in augmented.columns
    assert "calibration_state" in CalibratedNetAlphaModel(
        _DummyModel(_manifest()), adapter
    ).manifest().params


class _DummyModel:
    def __init__(self, manifest: ModelManifest):
        self._manifest = manifest

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame) -> None:
        del train, validation

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.lit(0.0, dtype=pl.Float64).alias(SCORE_COLUMN))

    def manifest(self) -> ModelManifest:
        return self._manifest


def test_session_balanced_weights_equalize_session_totals() -> None:
    from legacy.stocks.ml.models import (
        normalize_session_weights,
        session_balanced_weights,
    )

    frame = pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "instrument_id": ["a", "b", "c", "d", "e"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    raw = session_balanced_weights(frame)
    assert raw[0] == pytest.approx(1.0 / 3.0)
    assert raw[3] == pytest.approx(1.0 / 2.0)
    normalized = normalize_session_weights(raw)
    assert normalized.sum() == pytest.approx(frame.height)
    # Each session carries equal total training weight despite different sizes.
    assert normalized[:3].sum() == pytest.approx(normalized[3:].sum())


def test_session_balanced_weights_reject_malformed() -> None:
    from legacy.stocks.ml.models import normalize_session_weights

    with pytest.raises(ValueError, match="malformed"):
        normalize_session_weights(np.asarray([0.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="malformed"):
        normalize_session_weights(np.asarray([np.nan, 1.0]))


def test_weighted_path_selection_matches_independent_elastic_fits() -> None:
    from scipy.stats import spearmanr
    from sklearn.linear_model import ElasticNet

    from legacy.stocks.ml.models import (
        _float32_matrix,
        fit_weighted_elastic_path,
        normalize_session_weights,
        session_balanced_weights,
        weighted_fold_statistics,
    )

    rng = np.random.default_rng(11)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for s in range(80):
        for t in range(12):
            x1 = float(rng.normal(0.0, 1.0))
            x2 = float(rng.normal(0.0, 1.0))
            x3 = float(rng.normal(0.0, 1.0))
            target = 0.3 * x1 - 0.2 * x2 + 0.1 * x3 + rng.normal(0.0, 0.01)
            rows.append(
                {
                    "session_index": s,
                    "session": start,
                    "instrument_id": f"KRX:{t:05d}",
                    "feature__a": x1,
                    "feature__b": x2,
                    "feature__c": x3,
                    "net_alpha_target": target,
                    "realized_net_return": target,
                }
            )
    frame = pl.DataFrame(rows)
    train_mask = frame["session_index"] < 60
    validation = frame.filter(~train_mask)
    columns = ("feature__a", "feature__b", "feature__c")
    fractions = (0.01, 0.03, 0.10, 0.30)

    features_full = _float32_matrix(frame, columns)
    targets_full = frame["net_alpha_target"].cast(pl.Float64).to_numpy()
    codes_full = np.asarray(frame["session_index"].to_numpy(), dtype=np.int64)
    solution = fit_weighted_elastic_path(
        features_full[np.asarray(train_mask)],
        targets_full[np.asarray(train_mask)],
        codes_full[np.asarray(train_mask)],
        fractions,
        seed=42,
    )
    assert solution is not None
    alpha_max = solution.alpha_max

    def _rank_ic_for(scores: np.ndarray, val: pl.DataFrame) -> float:
        sub = val.with_columns(pl.Series("__score", scores)).select(
            "session", "__score", "realized_net_return"
        ).filter(
            pl.col("__score").is_not_null()
            & pl.col("realized_net_return").is_not_null()
        )
        ics = []
        for rows_ in sub.partition_by("session"):
            if rows_.height < 2:
                continue
            rho, _ = spearmanr(
                rows_["__score"].to_numpy(), rows_["realized_net_return"].to_numpy()
            )
            ics.append(float(rho))
        return float(np.mean(ics)) if ics else 0.0

    path_scores = solution.predict(validation, columns)
    path_ics = {
        fraction: _rank_ic_for(path_scores[fraction], validation)
        for fraction in fractions
    }
    best_path = max(fractions, key=lambda f: path_ics[f])

    train = frame.filter(train_mask)
    features = _float32_matrix(train, columns)
    targets = train["net_alpha_target"].cast(pl.Float64).to_numpy()
    valid = np.isfinite(features).all(axis=1) & np.isfinite(targets)
    weights = normalize_session_weights(
        session_balanced_weights(train), total=int(valid.sum())
    )[valid]
    mean, std = weighted_fold_statistics(features, weights, valid)
    standardized = (features[valid] - mean) / std
    independent_ics: dict[float, float] = {}
    for fraction in fractions:
        model = ElasticNet(
            alpha=fraction * alpha_max,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42,
        )
        model.fit(standardized, targets[valid], sample_weight=weights)
        val_features = _float32_matrix(validation, columns)
        val_std = (val_features - mean) / std
        val_std[~np.isfinite(val_std)] = 0.0
        independent_ics[fraction] = _rank_ic_for(
            model.predict(val_std), validation
        )
    best_independent = max(fractions, key=lambda f: independent_ics[f])
    assert best_path == best_independent
    assert path_ics[best_path] == pytest.approx(independent_ics[best_path], abs=1e-6)


WEIGHTS_INVALID_01 = "WEIGHTS-INVALID-01"


def test_weights_invalid_01_mixed_non_finite_rows_keep_aligned_lengths() -> None:
    """WEIGHTS-INVALID-01: valid-only counts keep rows and weights aligned.

    Mixed non-finite rows produce equal selected row/weight lengths; every
    valid session owns an equal raw total and the normalized weights sum to 1
    within 1e-12.
    """
    from legacy.stocks.ml.models import session_balanced_weights_from_codes

    codes = np.asarray([0, 0, 0, 1, 1, 1, 2], dtype=np.int64)
    # Session 0: one invalid row; session 2: two invalid rows.
    valid = np.asarray([True, True, False, True, True, True, False])
    weights = session_balanced_weights_from_codes(codes, valid)
    selected_rows = int(valid.sum())
    assert weights.shape == codes.shape  # full length preserved
    selected = weights[valid]
    assert selected.size == selected_rows

    for code in (0, 1):
        session_total = float(selected[codes[valid] == code].sum())
        assert session_total == pytest.approx(0.5)  # each session: 1 / n_sessions
    total = float(selected.sum())
    assert abs(total - 1.0) <= 1e-12


ELASTIC_PARITY_01 = "ELASTIC-PARITY-01"


def test_elastic_parity_01_prepared_path_matches_frame_reference() -> None:
    """ELASTIC-PARITY-01: prepared arrays reproduce the frame-based path.

    alpha_max, four fractions, coefficients, intercepts, and predictions match
    the legacy reference within 1e-12 on an all-finite fixture.
    """
    from legacy.stocks.ml.models import (
        _fit_weighted_elastic_path_reference,
        fit_weighted_elastic_path,
    )

    rng = np.random.default_rng(23)
    columns = ("feature__a", "feature__b", "feature__c")
    rows: list[dict[str, object]] = []
    for s in range(40):
        for t in range(8):
            a, b, c = (float(rng.normal()) for _ in range(3))
            rows.append(
                {
                    "session": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s),
                    "instrument_id": f"KRX:{t:05d}",
                    "feature__a": a,
                    "feature__b": b,
                    "feature__c": c,
                    "net_alpha_target": 0.4 * a - 0.3 * b + rng.normal(scale=0.01),
                }
            )
    frame = pl.DataFrame(rows)
    fractions = (0.01, 0.03, 0.10, 0.30)

    reference = _fit_weighted_elastic_path_reference(
        frame, columns, fractions, seed=42
    )
    features = np.stack([frame[c].to_numpy() for c in columns], axis=1).astype(np.float32)
    targets = frame["net_alpha_target"].to_numpy().astype(np.float64)
    codes = np.arange(frame.height, dtype=np.int64) // 8
    prepared = fit_weighted_elastic_path(features, targets, codes, fractions, seed=42)

    assert reference is not None
    assert prepared is not None
    assert prepared.fractions == reference.fractions
    assert prepared.alpha_max == pytest.approx(reference.alpha_max, abs=1e-12)
    assert prepared.coefficients == pytest.approx(reference.coefficients, abs=1e-12)
    assert prepared.intercepts == pytest.approx(reference.intercepts, abs=1e-12)
    ref_scores = reference.predict(frame, columns)
    prep_scores = prepared.predict_array(features.astype(np.float64))
    for fraction in fractions:
        diff = float(np.max(np.abs(prep_scores[fraction] - ref_scores[fraction])))
        assert diff <= 1e-12


def _indexed_design_fixture(
    n_sessions: int = 24,
    n_instruments: int = 6,
    seed: int = 29,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ML_FULL_EXECUTION_P0_ELASTIC_PARITY_01 deterministic fixture."""
    rng = np.random.default_rng(seed)
    n_rows = n_sessions * n_instruments
    features = rng.normal(size=(n_rows, 4)).astype(np.float32)
    features[:, 3] = 2.5  # zero-variance column exercises unit std handling
    codes = np.repeat(np.arange(n_sessions, dtype=np.int64), n_instruments)
    targets = (
        0.5 * features[:, 0].astype(np.float64)
        - 0.25 * features[:, 1].astype(np.float64)
        + rng.normal(scale=0.01, size=n_rows)
    )
    return features, targets, codes


def _indexed_design_frame(
    features: np.ndarray, targets: np.ndarray, codes: np.ndarray
) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    n_rows = features.shape[0]
    return pl.DataFrame(
        {
            "session": [start + timedelta(days=int(c)) for c in codes],
            "instrument_id": [f"KRX:{i % 6:05d}" for i in range(n_rows)],
            "feature__a": features[:, 0],
            "feature__b": features[:, 1],
            "feature__c": features[:, 2],
            "feature__d": features[:, 3],
            "net_alpha_target": targets,
        }
    )


def test_elastic_parity_01_indexed_design_matches_frame_reference() -> None:
    """ELASTIC-PARITY-01: indexed workspace reproduces the frame oracle.

    One chunked Fortran-order float64 design yields alpha_max, every
    fraction's coefficients/intercepts, and predictions within 1e-10 of the
    frame-based reference; the zero-variance column keeps std == 1.0.
    """
    from legacy.stocks.ml.models import (
        _fit_weighted_elastic_path_reference,
        fit_prepared_elastic_path,
        prepare_indexed_elastic_design,
    )

    columns = ("feature__a", "feature__b", "feature__c", "feature__d")
    fractions = (0.01, 0.03, 0.10, 0.30)
    features, targets, codes = _indexed_design_fixture()
    frame = _indexed_design_frame(features, targets, codes)

    reference = _fit_weighted_elastic_path_reference(
        frame, columns, fractions, seed=42
    )
    assert reference is not None

    design = prepare_indexed_elastic_design(
        features,
        np.arange(features.shape[0], dtype=np.int64),
        targets,
        codes,
        chunk_rows=9,
    )
    assert design is not None
    assert design.standardized.flags["F_CONTIGUOUS"]
    assert design.standardized.dtype == np.float64
    # The zero-variance column standardizes with unit std.
    assert design.std[3] == pytest.approx(1.0)

    solution = fit_prepared_elastic_path(design, fractions, seed=42)
    assert solution is not None
    assert solution.fractions == reference.fractions
    assert abs(solution.alpha_max - reference.alpha_max) <= 1e-10
    assert np.max(np.abs(solution.coefficients - reference.coefficients)) <= 1e-10
    assert np.max(np.abs(solution.intercepts - reference.intercepts)) <= 1e-10

    prep_scores = solution.predict_array(features.astype(np.float64))
    ref_scores = reference.predict(frame, columns)
    for fraction in fractions:
        diff = float(np.max(np.abs(prep_scores[fraction] - ref_scores[fraction])))
        assert diff <= 1e-10


def test_elastic_parity_01_mixed_non_finite_rows_keep_aligned_valid_rows() -> None:
    """ELASTIC-PARITY-01: mixed non-finite rows retain aligned rows/weights.

    Invalid feature/target rows are excluded exactly once; the surviving
    target and weight arrays stay row-aligned, weights sum to one, and every
    valid session carries an equal total share matching the array route
    within 1e-10.
    """
    from legacy.stocks.ml.models import (
        fit_prepared_elastic_path,
        fit_weighted_elastic_path,
        prepare_indexed_elastic_design,
    )

    fractions = (0.03, 0.30)
    features, targets, codes = _indexed_design_fixture()
    invalid_rows = np.asarray([3, 40, 41, 100])
    features[invalid_rows[:3], 0] = np.nan
    targets[invalid_rows[3]] = np.nan
    expected_valid = (
        np.isfinite(features).all(axis=1) & np.isfinite(targets)
    )

    design = prepare_indexed_elastic_design(
        features,
        np.arange(features.shape[0], dtype=np.int64),
        targets,
        codes,
        chunk_rows=7,
    )
    assert design is not None
    assert np.array_equal(design.row_indices, expected_valid.nonzero()[0])
    assert design.target.size == design.weights.size == int(expected_valid.sum())
    assert abs(float(design.weights.sum()) - 1.0) <= 1e-12
    valid_codes = codes[expected_valid]
    for code in np.unique(valid_codes):
        session_total = float(design.weights[valid_codes == code].sum())
        assert session_total == pytest.approx(1.0 / 24.0)

    route = fit_weighted_elastic_path(
        features[expected_valid],
        targets[expected_valid],
        codes[expected_valid],
        fractions,
        seed=42,
    )
    solution = fit_prepared_elastic_path(design, fractions, seed=42)
    assert route is not None
    assert solution is not None
    assert abs(solution.alpha_max - route.alpha_max) <= 1e-10
    assert np.max(np.abs(solution.coefficients - route.coefficients)) <= 1e-10
    # Non-finite prediction inputs map to zero standardized values: a fully
    # invalid row collapses to the intercept alone.
    block = features[:4].astype(np.float64).copy()
    block[0, :] = np.nan
    scores = solution.predict_array(block)[fractions[-1]]
    assert scores[0] == pytest.approx(float(solution.intercepts[-1]))
    assert np.isfinite(scores[1:4]).all()
