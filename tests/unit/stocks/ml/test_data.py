"""NetAlphaResearchData composition: realized-outcome retention and clean predictors."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.data import compose_net_alpha_training_data
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def _decision_time() -> datetime:
    return datetime(2024, 12, 31, tzinfo=UTC)


def test_wide_composition_retains_decimal_realized_outcomes() -> None:
    df = stock_net_alpha_composed_df(n_sessions=30, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for horizon in (3, 5):
        label_frame = data.labels_by_horizon[horizon]
        assert "risk_residual" in label_frame.columns
        assert "reference_cost" in label_frame.columns
        assert "open" in label_frame.columns


def test_long_composition_retains_decimal_realized_outcomes() -> None:
    wide = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, candidate_horizon_sessions=(3, 5)
    )
    parts = [
        wide.select(
            pl.col("session_index"),
            pl.col("session"),
            pl.col("instrument_id"),
            pl.col("open"),
            pl.lit(horizon).alias("horizon_sessions"),
            pl.col(f"net_alpha_{horizon}d_target").alias("net_alpha_target"),
            pl.col(f"label_available_time_{horizon}d").alias("label_available_time"),
            pl.col(f"risk_residual_{horizon}d").alias("risk_residual"),
            pl.col(f"reference_cost_{horizon}d").alias("reference_cost"),
            pl.lit(0.0).alias("gross_return"),
        )
        for horizon in (3, 5)
    ]
    long_frame = pl.concat(parts)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=long_frame.columns),
        frame=long_frame,
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for horizon in (3, 5):
        label_frame = data.labels_by_horizon[horizon]
        assert "risk_residual" in label_frame.columns
        assert "reference_cost" in label_frame.columns
        assert "open" in label_frame.columns


def test_feature_frame_is_target_free() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, audit_clean=True, label_scale=50.0
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for column in data.feature_frame.columns:
        assert not column.startswith(
            ("net_alpha_", "label_available_time_", "risk_residual_", "reference_cost_")
        )
        assert column not in ("horizon_sessions", "net_alpha_target", "risk_residual", "reference_cost")


def test_horizon_universes_are_independent() -> None:
    df = stock_net_alpha_composed_df(n_sessions=40, n_tickers=6, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert set(data.labels_by_horizon) == {3, 5}
    assert data.labels_by_horizon[3].height > 0
    assert data.labels_by_horizon[5].height > 0
