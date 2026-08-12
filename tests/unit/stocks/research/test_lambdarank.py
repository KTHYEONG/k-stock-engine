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
