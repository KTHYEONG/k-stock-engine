"""LambdaRank blend model: config contract, blend math, and NO_TRADE behavior."""
from __future__ import annotations

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.research.features import (
    apply_v2_transforms,
    fit_v2_winsor_quantiles,
    stock_alpha_v2_allowlist,
    v2_feature_columns,
)
from src.stocks.research.labels import (
    RESIDUAL_O2O_LABEL,
    RELEVANCE_COLUMN,
    residual_open_to_open_label,
)
from src.stocks.research.lambdarank import (
    LAMBDARANK_WEIGHT,
    STABLE_WEIGHT,
    LambdaRankBlendModel,
    LambdaRankConfig,
)
from src.stocks.research.models import ModelManifest, StableRankComposite


def build_panel(n_sessions: int = 40, n_tickers: int = 40, seed: int = 7) -> pl.DataFrame:
    from datetime import UTC, datetime, timedelta

    import numpy as np

    allowlist = stock_alpha_v2_allowlist()
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        price = 100.0
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            price = max(10.0, price * (1.0 + float(rng.normal(0.0, 0.02))))
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "sector": f"S{t % 4}",
                    "open": price,
                    "close": price * 1.001,
                    **{f"feature__{name}": float(rng.normal(t * 0.01, 1.0)) for name in allowlist},
                }
            )
    df = pl.DataFrame(rows)
    labels = residual_open_to_open_label(df.select(["instrument_id", "session", "open"]))
    return df.join(labels, on=["instrument_id", "session"], how="inner")


def make_manifest(artifact_id: str = "blend_test") -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v2",
        feature_schema_hash="hash",
        universe_policy_hash="universe",
        label_definition=RESIDUAL_O2O_LABEL,
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
    )


def test_lambdarank_config_objective_and_gain_contract() -> None:
    config = LambdaRankConfig()
    assert config.objective == "lambdarank"
    assert config.label_gain == (0, 1, 3, 7, 15)
    params = config.lgb_params()
    assert params["objective"] == "lambdarank"
    assert params["metric"] == "ndcg"
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True


def test_blend_weights_are_frozen() -> None:
    assert LAMBDARANK_WEIGHT == 0.50
    assert STABLE_WEIGHT == 0.50
    assert LambdaRankConfig().lambdarank_weight == 0.50
    assert LambdaRankConfig().lambdarank_weight == LAMBDARANK_WEIGHT


def test_blend_weight_validation_and_manifest_provenance() -> None:
    with pytest.raises(ValueError, match="lambdarank_weight"):
        LambdaRankConfig(lambdarank_weight=-0.1)
    with pytest.raises(ValueError, match="lambdarank_weight"):
        LambdaRankConfig(lambdarank_weight=1.1)
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        config = LambdaRankConfig(lambdarank_weight=weight)
        assert config.lambdarank_weight == weight

    df = build_panel()
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    val = df.filter(pl.col("session_index") >= 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    model = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=LambdaRankConfig(learning_rate=0.03, num_leaves=31, lambdarank_weight=0.75),
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    model.fit(train_t, val_t)
    assert model.no_trade is False
    predict_input = val_t.drop(
        [RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"]
    )
    scored = model.predict(predict_input)
    assert scored["pred_score"].null_count() == 0
    params = model.manifest().params
    assert params["blend_weight_lambdarank"] == "0.750000"
    assert params["blend_weight_stable"] == "0.250000"


def test_model_fits_and_blends_percentile_ranks() -> None:
    df = build_panel()
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    val = df.filter(pl.col("session_index") >= 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)

    model = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=LambdaRankConfig(learning_rate=0.03, num_leaves=31),
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    model.fit(train_t, val_t)
    assert model.no_trade is False

    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    scored = model.predict(predict_input)
    assert "pred_score" in scored.columns
    scores = scored["pred_score"].to_numpy()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert scored["pred_score"].mean() == pytest.approx(0.5, abs=0.05)


def test_small_group_renders_no_trade_zero_scores() -> None:
    df = build_panel(n_sessions=20, n_tickers=3)
    feature_columns = v2_feature_columns(df)
    quantiles = fit_v2_winsor_quantiles(df, feature_columns)
    transformed = apply_v2_transforms(df, feature_columns, winsor_quantiles=quantiles)
    model = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        relevance_column=RELEVANCE_COLUMN,
    )
    model.fit(transformed, transformed)
    assert model.no_trade is True
    scored = model.predict(
        transformed.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    )
    assert scored["pred_score"].to_list() == [0.0] * scored.height


def test_trial_fast_path_matches_legacy_blend() -> None:
    import numpy as np

    df = build_panel()
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    val = df.filter(pl.col("session_index") >= 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    config = LambdaRankConfig(learning_rate=0.03, num_leaves=31)

    legacy = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    legacy.fit(train_t, val_t)
    legacy_scored = legacy.predict(predict_input)
    assert legacy.no_trade is False

    stable = StableRankComposite(
        factors=stock_alpha_v2_allowlist(),
        manifest=make_manifest(),
        label_column=RESIDUAL_O2O_LABEL,
        block_length=5,
        session_column="session",
    )
    stable.fit(train_t, val_t)
    stable_scores = stable.predict(predict_input).select("session", "instrument_id", "pred_score")

    prepared = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    ).prepare_fold(train_t, val_t)
    assert prepared is not None
    assert prepared.train_matrix.dtype == np.float32
    assert prepared.train_matrix.flags["C_CONTIGUOUS"]
    assert prepared.train_matrix.flags["WRITEABLE"] is False
    assert prepared.train_relevance.dtype == np.int32
    assert prepared.train_relevance.flags["WRITEABLE"] is False
    assert prepared.train_weights.dtype == np.float64
    assert prepared.train_weights.flags["WRITEABLE"] is False
    assert prepared.validation_matrix is not None
    assert prepared.validation_matrix.flags["WRITEABLE"] is False
    assert prepared.predictor_columns

    trial = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    outcome = trial.fit_trial(train_t, val_t, stable_scores)
    assert outcome.fit_ok is True
    assert outcome.best_iteration is not None
    trial_scored = trial.predict(predict_input)
    assert (trial_scored["pred_score"].to_numpy() == pytest.approx(legacy_scored["pred_score"].to_numpy()))
    assert (trial_scored["pred_score"].to_numpy() - legacy_scored["pred_score"].to_numpy()).max() < 1e-12

    prepared_trial = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    prepared_outcome = prepared_trial.fit_trial(
        train_t, val_t, stable_scores, prepared=prepared
    )
    assert prepared_outcome.fit_ok is True
    assert prepared_outcome.best_iteration == outcome.best_iteration
    prepared_scored = prepared_trial.predict(predict_input)
    assert (prepared_scored["pred_score"].to_numpy() == pytest.approx(legacy_scored["pred_score"].to_numpy()))
    assert (prepared_scored["pred_score"].to_numpy() - legacy_scored["pred_score"].to_numpy()).max() < 1e-12
    assert (prepared_scored["pred_score"].to_numpy() == pytest.approx(trial_scored["pred_score"].to_numpy()))

    matrix = LambdaRankBlendModel._float32_matrix(predict_input, list(feature_columns))
    assert matrix.dtype == np.float32
    assert matrix.flags["C_CONTIGUOUS"]
    relevance = val_t[RELEVANCE_COLUMN].cast(pl.Int32).to_numpy()
    assert relevance.dtype == np.int32


def test_trial_fast_path_preserves_labeled_rows_with_predictor_nulls() -> None:
    """Null predictors never delete a labeled row; prepared and uncached agree."""
    import numpy as np

    df = build_panel(n_sessions=60, n_tickers=40)
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 45)
    val = df.filter(pl.col("session_index") >= 45)

    rng = np.random.default_rng(11)
    null_feature = feature_columns[0]
    train = train.with_columns(
        pl.when(pl.Series(rng.random(train.height)) < 0.3)
        .then(None)
        .otherwise(pl.col(null_feature))
        .alias(null_feature)
    )

    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])

    eligible = int(train.filter(pl.col(RELEVANCE_COLUMN).is_not_null()).height)
    config = LambdaRankConfig(
        learning_rate=0.05, num_leaves=15, n_estimators=200,
        early_stopping_rounds=20,
    )

    model = LambdaRankBlendModel(
        make_manifest("null_retention"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    stable = StableRankComposite(
        factors=stock_alpha_v2_allowlist(),
        manifest=make_manifest("null_retention_stable"),
        label_column=RESIDUAL_O2O_LABEL,
        block_length=5,
        session_column="session",
    )
    stable.fit(train_t, val_t)
    stable_scores = stable.predict(predict_input).select(
        "session", "instrument_id", "pred_score"
    )
    outcome = model.fit_trial(train_t, val_t, stable_scores=stable_scores)
    assert outcome.fit_ok is True

    prepared = LambdaRankBlendModel(
        make_manifest("null_retention_prepared"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    ).prepare_fold(train_t, val_t)
    assert prepared is not None
    assert prepared.train_matrix.shape[0] == eligible
    assert model._train_group_count == len(prepared.train_group_sizes)
    assert prepared.predictor_columns

    reference = LambdaRankBlendModel(
        make_manifest("null_retention_ref"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    reference.fit(train_t, val_t)
    reference_scored = reference.predict(predict_input).select(
        "session", "instrument_id", "pred_score"
    )
    trial_model = LambdaRankBlendModel(
        make_manifest("null_retention_prepared_trial"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    trial_outcome = trial_model.fit_trial_prepared(
        prepared, stable_scores=stable_scores
    )
    assert trial_outcome.fit_ok is True
    val_used = (
        val_t.filter(pl.col(RELEVANCE_COLUMN).is_not_null())
        .sort("session")
        .select("session", "instrument_id")
    )
    assert val_used.height == prepared.validation_matrix.shape[0]
    joined = reference_scored.join(
        trial_model.predict_prepared_scores(prepared, val_used, stable_scores),
        on=["session", "instrument_id"],
        suffix="_prepared",
    )
    assert (
        joined["pred_score"].to_numpy() - joined["pred_score_prepared"].to_numpy()
    ).max() < 1e-12


def test_observation_weights_are_newest_anchored_and_session_sums_are_recency() -> None:
    """Newest session sums to 1.0; older sessions receive exp2 decay from newest."""
    import numpy as np

    df = build_panel(n_sessions=40, n_tickers=40)
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    model = LambdaRankBlendModel(
        make_manifest("recency"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=LambdaRankConfig(half_life_sessions=504),
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    usable = train_t.filter(pl.col(RELEVANCE_COLUMN).is_not_null())
    group_sizes, _ = model._group_sizes(usable)
    ordered = usable.filter(
        pl.col("session").is_in(
            usable.group_by("session").len().filter(
                pl.col("len") >= model.config.min_group_size
            )["session"]
        )
    ).sort("session")
    weights = model._observation_weights(ordered, group_sizes)
    assert len(weights) == ordered.height
    assert np.all(np.isfinite(weights))
    start = 0
    session_sums: list[float] = []
    for size in group_sizes:
        session_sums.append(float(np.sum(weights[start : start + size])))
        start += size
    assert session_sums[-1] == pytest.approx(1.0, abs=1e-9)
    assert session_sums[0] <= session_sums[-1]
    assert session_sums[0] == pytest.approx(
        np.exp2(-(len(session_sums) - 1) / 504.0), rel=1e-9
    )
    assert session_sums == sorted(session_sums)


def test_resolve_predictor_columns_returns_rank_sector_rank_and_missing_only() -> None:
    """Raw feature levels never enter the booster design matrix."""
    df = build_panel(n_sessions=40, n_tickers=40)
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    model = LambdaRankBlendModel(
        make_manifest("predictors"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    columns = model._resolve_predictor_columns(train_t)
    expected = [
        column
        for feature in model.features
        for column in (
            f"feature__{feature}__rank",
            f"feature__{feature}__sector_rank",
            f"feature__{feature}__missing",
        )
    ]
    assert columns == expected
    assert not any(
        column in train_t.columns and column not in expected for column in columns
    )


def test_trial_callbacks_propagate_trial_pruned() -> None:
    import optuna

    df = build_panel()
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 30)
    val = df.filter(pl.col("session_index") >= 30)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    config = LambdaRankConfig(learning_rate=0.03, num_leaves=31)

    stable = StableRankComposite(
        factors=stock_alpha_v2_allowlist(),
        manifest=make_manifest(),
        label_column=RESIDUAL_O2O_LABEL,
        block_length=5,
        session_column="session",
    )
    stable.fit(train_t, val_t)
    stable_scores = stable.predict(predict_input).select("session", "instrument_id", "pred_score")
    prepared = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    ).prepare_fold(train_t, val_t)
    assert prepared is not None

    def prune_on_first_round(env: object) -> None:
        del env
        raise optuna.TrialPruned()

    model = LambdaRankBlendModel(
        make_manifest(),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    with pytest.raises(optuna.TrialPruned):
        model.fit_trial(
            train_t, val_t, stable_scores, prepared=prepared, callbacks=(prune_on_first_round,)
        )


def test_resolve_lgb_num_threads_contract() -> None:
    from src.stocks.research.lambdarank import resolve_lgb_num_threads

    assert resolve_lgb_num_threads(None, physical_cores=4, logical_cores=8) == 4
    assert resolve_lgb_num_threads(None, physical_cores=4, logical_cores=4) == 4
    assert resolve_lgb_num_threads(None, physical_cores=1, logical_cores=8) == 1
    assert resolve_lgb_num_threads(2, physical_cores=4, logical_cores=8) == 2
    assert resolve_lgb_num_threads(8, physical_cores=4, logical_cores=8) == 8
    with pytest.raises(ValueError, match="positive"):
        resolve_lgb_num_threads(0, physical_cores=4, logical_cores=8)
    with pytest.raises(ValueError, match="positive"):
        resolve_lgb_num_threads(-2, physical_cores=4, logical_cores=8)
    with pytest.raises(ValueError, match="logical CPU"):
        resolve_lgb_num_threads(9, physical_cores=4, logical_cores=8)
    with pytest.raises(ValueError, match="physical_cores"):
        resolve_lgb_num_threads(None, physical_cores=0, logical_cores=8)


def test_prepared_scoring_matches_reference_and_threads_are_bit_identical() -> None:
    from src.stocks.research.lambdarank import (
        LambdaRankBlendModel,
        LambdaRankConfig,
    )

    df = build_panel(n_sessions=80, n_tickers=40)
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 60)
    val = df.filter(pl.col("session_index") >= 60)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    config = LambdaRankConfig(
        learning_rate=0.05, num_leaves=15, min_child_samples=500,
        n_estimators=400, early_stopping_rounds=30,
    )

    stable = StableRankComposite(
        factors=stock_alpha_v2_allowlist(),
        manifest=make_manifest(),
        label_column=RESIDUAL_O2O_LABEL,
        block_length=5,
        session_column="session",
    )
    stable.fit(train_t, val_t)
    stable_scores = stable.predict(predict_input).select("session", "instrument_id", "pred_score")

    def _fit(num_threads: int) -> tuple[LambdaRankBlendModel, pl.DataFrame]:
        fold_config = LambdaRankConfig(
            learning_rate=0.05, num_leaves=15, min_child_samples=500,
            n_estimators=400, early_stopping_rounds=30, num_threads=num_threads,
        )
        prepared = LambdaRankBlendModel(
            make_manifest("prepared_parity"),
            stock_alpha_v2_allowlist(),
            RESIDUAL_O2O_LABEL,
            config=fold_config,
            session_column="session",
            relevance_column=RELEVANCE_COLUMN,
        ).prepare_fold(train_t, val_t)
        assert prepared is not None
        model = LambdaRankBlendModel(
            make_manifest("prepared_parity"),
            stock_alpha_v2_allowlist(),
            RESIDUAL_O2O_LABEL,
            config=fold_config,
            session_column="session",
            relevance_column=RELEVANCE_COLUMN,
        )
        outcome = model.fit_trial_prepared(prepared, stable_scores)
        assert outcome.fit_ok is True
        val_used = (
            val_t.filter(pl.col(RELEVANCE_COLUMN).is_not_null())
            .sort("session")
            .select("session", "instrument_id")
        )
        assert val_used.height == prepared.validation_matrix.shape[0]
        slim = model.predict_prepared_scores(prepared, val_used, stable_scores)
        assert list(slim.columns) == ["session", "instrument_id", "pred_score"]
        return model, slim

    single, single_scored = _fit(1)
    four, four_scored = _fit(4)
    assert single._booster is not None
    assert four._booster is not None
    assert (
        single._booster.num_trees() == four._booster.num_trees()
    )
    assert (single_scored["pred_score"].to_numpy() == four_scored["pred_score"].to_numpy()).all()
    assert single_scored.equals(four_scored)

    reference = LambdaRankBlendModel(
        make_manifest("prepared_parity_ref"),
        stock_alpha_v2_allowlist(),
        RESIDUAL_O2O_LABEL,
        config=config,
        session_column="session",
        relevance_column=RELEVANCE_COLUMN,
    )
    reference.fit(train_t, val_t)
    reference_scored = reference.predict(predict_input).select(
        "session", "instrument_id", "pred_score"
    )
    joined = reference_scored.join(single_scored, on=["session", "instrument_id"], suffix="_prepared")
    assert (joined["pred_score"].to_numpy() - joined["pred_score_prepared"].to_numpy()).max() < 1e-12


def test_adaptive_continuation_parity_with_one_shot() -> None:
    from src.stocks.research.lambdarank import (
        FitTrialOutcome,
        LambdaRankBlendModel,
        adaptive_refit_rounds,
        verify_adaptive_parity,
    )

    df = build_panel(n_sessions=80, n_tickers=40)
    feature_columns = v2_feature_columns(df)
    train = df.filter(pl.col("session_index") < 60)
    val = df.filter(pl.col("session_index") >= 60)
    quantiles = fit_v2_winsor_quantiles(train, feature_columns)
    train_t = apply_v2_transforms(train, feature_columns, winsor_quantiles=quantiles)
    val_t = apply_v2_transforms(val, feature_columns, winsor_quantiles=quantiles)
    predict_input = val_t.drop([RESIDUAL_O2O_LABEL, RELEVANCE_COLUMN, "label_available_time"])
    config = LambdaRankConfig(
        learning_rate=0.05, num_leaves=15, min_child_samples=500,
        n_estimators=400, early_stopping_rounds=30,
    )

    stable = StableRankComposite(
        factors=stock_alpha_v2_allowlist(),
        manifest=make_manifest(),
        label_column=RESIDUAL_O2O_LABEL,
        block_length=5,
        session_column="session",
    )
    stable.fit(train_t, val_t)
    stable_scores = stable.predict(predict_input).select("session", "instrument_id", "pred_score")

    def _fit(initial_rounds: int | None) -> tuple[FitTrialOutcome, LambdaRankBlendModel]:
        model = LambdaRankBlendModel(
            make_manifest("adaptive_parity"),
            stock_alpha_v2_allowlist(),
            RESIDUAL_O2O_LABEL,
            config=config,
            session_column="session",
            relevance_column=RELEVANCE_COLUMN,
        )
        outcome = model.fit_trial(
            train_t, val_t, stable_scores, initial_rounds=initial_rounds
        )
        return outcome, model

    one_shot, one_shot_model = _fit(None)
    assert one_shot.fit_ok is True
    assert adaptive_refit_rounds(one_shot.best_iteration) <= config.n_estimators
    assert adaptive_refit_rounds(500) == 600
    assert adaptive_refit_rounds(50) == 200
    assert adaptive_refit_rounds(10_000) == 900

    continuation, continuation_model = _fit(40)
    assert continuation.fit_ok is True
    assert continuation.used_continuation is True
    assert continuation.best_iteration == one_shot.best_iteration
    assert continuation.stopped_early == one_shot.stopped_early
    assert continuation.rounds_trained <= config.n_estimators

    matrix = LambdaRankBlendModel._float32_matrix(
        predict_input, list(continuation_model._predictor_columns)
    )
    assert (
        verify_adaptive_parity(
            continuation,
            one_shot,
            booster=continuation_model._booster,
            reference_booster=one_shot_model._booster,
            predict_input=matrix,
        )
        is True
    )
    continued_scored = continuation_model.predict(predict_input)
    reference_scored = one_shot_model.predict(predict_input)
    assert (
        continued_scored["pred_score"].to_numpy()
        - reference_scored["pred_score"].to_numpy()
    ).max() < 1e-12
    assert rank_ic(val, continued_scored, RESIDUAL_O2O_LABEL) == pytest.approx(
        rank_ic(val, reference_scored, RESIDUAL_O2O_LABEL), rel=1e-12, abs=1e-12
    )


def rank_ic(labeled: pl.DataFrame, scored: pl.DataFrame, label_column: str) -> float:
    import numpy as np

    sub = labeled.select(
        pl.col("session"), pl.col("instrument_id"), pl.col(label_column)
    ).join(
        scored.select("session", "instrument_id", "pred_score"),
        on=["session", "instrument_id"],
    ).filter(pl.col(label_column).is_not_null() & pl.col("pred_score").is_not_null())
    ics: list[float] = []
    for rows in sub.sort("session").partition_by("session"):
        scores = rows["pred_score"].to_numpy().astype(float)
        labels = rows[label_column].to_numpy().astype(float)
        if len(scores) < 2 or np.std(scores) == 0.0 or np.std(labels) == 0.0:
            continue
        rs = np.argsort(np.argsort(scores)) - np.argsort(np.argsort(scores)).mean()
        rl = np.argsort(np.argsort(labels)) - np.argsort(np.argsort(labels)).mean()
        denom = float(np.sqrt(float(np.sum(rs * rs)) * float(np.sum(rl * rl))))
        ics.append(float(np.sum(rs * rl) / denom) if denom > 0.0 else 0.0)
    return float(np.median(ics)) if ics else 0.0


def test_manifest_binds_v2_contract() -> None:
    manifest = make_manifest("blend_manifest")
    model = LambdaRankBlendModel(
        manifest, stock_alpha_v2_allowlist(), RESIDUAL_O2O_LABEL
    )
    published = model.manifest()
    assert published.model_type == "lambdarank_blend"
    assert published.feature_set == "stock_alpha_v2"
    assert published.params is not None
    assert published.params["objective"] == "lambdarank"
    assert published.params["label_gain"] == "0,1,3,7,15"
