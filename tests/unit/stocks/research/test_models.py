"""StableRankComposite factor-derivation model tests."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.core.instruments import AssetKind
from src.stocks.research.folds import PurgedWalkForward
from src.stocks.research.labels import LabelDefinition
from src.stocks.research.models import (
    ModelManifest,
    RankICConfig,
    StableRankComposite,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        artifact_id="composite_v1",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v1",
        feature_schema_hash="hash",
        universe_policy_hash="universe",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=2,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="stable_rank_composite",
    )


def _panel(n_instruments: int = 8, n_sessions: int = 14) -> pl.DataFrame:
    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_sessions)]
    rows = [
        {
            "instrument_id": f"KRX:{t:05d}",
            "session": sessions[s],
            "rev_5d": float((t + s) % 5) / 5.0,
            "ln_mktcap": math.log(1e11 + t * 1e9),
            "open": 100.0 + float(t),
            "close": 100.0 + float(t) + float(s % 3) * 0.1,
        }
        for t in range(n_instruments)
        for s in range(n_sessions)
    ]
    frame = pl.DataFrame(rows)
    labeled = LabelDefinition("fwd_ret_5d", "open", "close", 2).apply(frame)
    return labeled.with_columns(
        pl.col("session").rank("dense").cast(pl.Int64).alias("session_index")
    )


def test_composite_fits_and_normalizes_or_no_trade() -> None:
    labeled = _panel()
    splitter = PurgedWalkForward(
        n_folds=3, label_horizon_sessions=2, embargo_sessions=1, session_column="session_index"
    )
    folds = splitter.split(labeled)
    assert folds
    fold = folds[-1]
    train = labeled[fold.train_mask]
    val = labeled[fold.validation_mask]
    model = StableRankComposite(
        factors=("rev_5d", "ln_mktcap"),
        manifest=_manifest(),
        label_column="fwd_ret_5d",
        config=RankICConfig(seed=11, n_bootstrap=50),
        block_length=2,
    )
    model.fit(train, val, inner_folds=splitter.inner_folds(train))
    if not model.no_trade:
        assert sum(abs(w) for w in model.factor_weights.values()) == pytest.approx(1.0)
    scored = model.predict(val.drop(["fwd_ret_5d"]))
    assert "pred_score" in scored.columns


def test_composite_rejects_labels_at_predict() -> None:
    labeled = _panel()
    model = StableRankComposite(
        factors=("rev_5d",),
        manifest=_manifest(),
        label_column="fwd_ret_5d",
        config=RankICConfig(seed=1, n_bootstrap=20),
        block_length=2,
    )
    with pytest.raises(ValueError, match="target"):
        model.predict(labeled)


def test_no_trade_model_scores_zero() -> None:
    model = StableRankComposite(
        factors=("rev_5d", "ln_mktcap"),
        manifest=_manifest(),
        label_column="fwd_ret_5d",
        config=RankICConfig(seed=1, n_bootstrap=20),
        block_length=2,
    )
    frame = _panel().drop(["fwd_ret_5d", "session_index"])
    out = model.predict(frame)
    assert model.no_trade
    assert out["pred_score"].to_list() == [0.0] * frame.height
