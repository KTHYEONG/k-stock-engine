"""NetAlphaResearchData composition: realized-outcome retention and clean predictors."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.ml.data import compose_net_alpha_training_data
from src.stocks.ml.training import _build_label_join
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)

_NARROW_LABEL_COLUMNS = {
    "instrument_id",
    "session",
    "net_alpha_target",
    "label_available_time",
    "risk_residual",
    "reference_cost",
}


def _decision_time() -> datetime:
    return datetime(2024, 12, 31, tzinfo=UTC)


def _assert_narrow_labels_restored(data, horizon: int) -> None:
    label_frame = data.labels_by_horizon[horizon]
    assert set(label_frame.columns) == _NARROW_LABEL_COLUMNS
    join = _build_label_join(data, horizon)
    assert {
        "instrument_id",
        "session",
        "net_alpha_target",
        "label_available_time",
        "risk_residual",
        "reference_cost",
        "open",
        "adtv_20d",
        "volatility_20d",
        "realized_net_return",
    } <= set(join.columns)
    realized = (join["risk_residual"] - join["reference_cost"]).to_list()
    assert join["realized_net_return"].to_list() == pytest.approx(realized)


def test_wide_composition_retains_decimal_realized_outcomes() -> None:
    df = stock_net_alpha_composed_df(n_sessions=30, n_tickers=4, audit_clean=True)
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    for horizon in (3, 5):
        _assert_narrow_labels_restored(data, horizon)


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
            pl.col("adtv_20d"),
            pl.col("volatility_20d"),
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
        _assert_narrow_labels_restored(data, horizon)


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


def test_compose_builds_status_coverage_and_join_evidence() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=40, n_tickers=6, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert set(data.coverage_by_horizon) == {3, 5}
    for horizon in (3, 5):
        coverage = data.coverage_by_horizon[horizon]
        assert coverage.horizon_sessions == horizon
        assert coverage.decision_rows == data.feature_frame.height
        assert coverage.realized_rows == len(data.labels_by_horizon[horizon])
        assert coverage.status_counts.realized > 0
        assert "outcome_status" in coverage.status_projection.columns
        # Every decision key resolves to exactly one typed state.
        assert (
            coverage.status_counts.realized
            + coverage.status_counts.partial_tail
            + coverage.status_counts.unresolved
            == coverage.decision_rows
        )
    evidence = {e.horizon_sessions: e for e in data.join_evidence}
    for horizon in (3, 5):
        assert evidence[horizon].decision_rows == data.feature_frame.height
        assert evidence[horizon].realized_rows == len(data.labels_by_horizon[horizon])
        assert evidence[horizon].status_counts is not None
        assert evidence[horizon].status_counts.realized == evidence[horizon].realized_rows


def test_feature_frame_never_carries_outcome_status() -> None:
    df = stock_net_alpha_composed_df(
        n_sessions=10, n_tickers=2, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    data = compose_net_alpha_training_data(
        snapshot, _decision_time(), (3, 5)
    )
    assert "outcome_status" not in data.feature_frame.columns
    for frame in data.labels_by_horizon.values():
        assert "outcome_status" not in frame.columns


def test_future_label_or_status_never_changes_decision_eligibility() -> None:
    """Acceptance 4: removing a future label/status row cannot change the decision frame."""
    df = stock_net_alpha_composed_df(
        n_sessions=30, n_tickers=4, audit_clean=True
    )
    snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=df.columns), frame=df
    )
    decision_time = _decision_time()
    full = compose_net_alpha_training_data(snapshot, decision_time, (3, 5))
    # Drop the last session's label/status rows for every instrument: those are
    # the newest (future) rows relative to the decision-time panel of earlier
    # sessions. The decision feature universe and per-horizon realised labels
    # must be identical because decision eligibility never depends on future rows.
    last_session = df["session"].max()
    trimmed = df.filter(pl.col("session") != last_session)
    trimmed_snapshot = DatasetSnapshot(
        manifest=stock_net_alpha_manifest(columns=trimmed.columns),
        frame=trimmed,
    )
    trimmed_data = compose_net_alpha_training_data(
        trimmed_snapshot, decision_time, (3, 5)
    )
    assert set(full.labels_by_horizon) == set(trimmed_data.labels_by_horizon)
    for horizon in (3, 5):
        full_labels = full.labels_by_horizon[horizon]
        trimmed_labels = trimmed_data.labels_by_horizon[horizon]
        assert (
            full_labels.filter(pl.col("session") != last_session)
            .sort(["instrument_id", "session"])["session"]
            .to_list()
            == trimmed_labels.sort(["instrument_id", "session"])["session"].to_list()
        )
    # The decision feature universe is identical once the newest session is
    # excluded from both panels.
    assert (
        full.feature_frame.filter(pl.col("session") != last_session)
        .sort(["instrument_id", "session"])["session"]
        .to_list()
        == trimmed_data.feature_frame.sort(["instrument_id", "session"])["session"].to_list()
    )


def test_horizon_outcome_coverage_maps_score_keys_and_segments() -> None:
    from datetime import timedelta

    from src.stocks.ml.data import HorizonOutcomeCoverage

    start = datetime(2024, 1, 1, tzinfo=UTC)
    sessions = [start + timedelta(days=i) for i in range(6)]
    score_keys = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{t:05d}" for t in range(3)] * 6,
            "session": [s for s in sessions for _ in range(3)],
            "oof_segment_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    status_frame = pl.DataFrame(
        {
            "instrument_id": [f"KRX:{t:05d}" for t in range(3)] * 6,
            "session": [s for s in sessions for _ in range(3)],
            "outcome_status": ["REALIZED"] * 17 + ["MISSING_EXIT_PRICE"],
        }
    )
    coverage = HorizonOutcomeCoverage.build(
        3, score_keys, status_frame, segment_column="oof_segment_id"
    )
    assert coverage.decision_rows == 18
    assert coverage.realized_rows == 17
    assert coverage.status_counts.realized == 17
    assert coverage.status_counts.unresolved == 1
    assert len(coverage.segment_projection) == 2
    segment_counts = {s.segment_id: s.counts for s in coverage.segment_projection}
    assert segment_counts[0].realized == 9
    assert segment_counts[1].unresolved == 1
    assert "outcome_status" in coverage.status_projection.columns
    assert coverage.to_json()["realized_rows"] == 17


def test_horizon_outcome_coverage_fails_closed_on_unclassified_key() -> None:
    from datetime import timedelta

    from src.stocks.ml.data import HorizonOutcomeCoverage

    start = datetime(2024, 1, 1, tzinfo=UTC)
    score_keys = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001", "KRX:00002"],
            "session": [start, start + timedelta(days=1)],
        }
    )
    # KRX:00002 has no status row: the sidecar must cover the whole universe.
    status_frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:00001"],
            "session": [start],
            "outcome_status": ["REALIZED"],
        }
    )
    with pytest.raises(ValueError, match="absent from the outcome status sidecar"):
        HorizonOutcomeCoverage.build(3, score_keys, status_frame)
