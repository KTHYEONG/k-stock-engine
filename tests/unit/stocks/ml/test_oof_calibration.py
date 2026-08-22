"""Causal session-cluster OOF calibration isolation.

A later label can never change an earlier decision session's calibrated score,
the ``min_calibration_sessions`` history gate keeps early sessions in cash, and
changing the labels of the decision session itself is invariant to its own
calibrated score.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from src.stocks.ml import training
from src.stocks.ml.contracts import NetAlphaTrainingRequest, RiskSettings


def _causal_fixture(
    n_sessions: int = 24, n_tickers: int = 20
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic scored OOF panel plus its label ledger."""
    rng = np.random.default_rng(11)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for s in range(n_sessions):
        for t in range(n_tickers):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": start + timedelta(days=s),
                    "predicted_net_alpha": float(t) + 0.5 * (s % 3),
                    "risk_residual": 0.003 * t + rng.normal(0.0, 0.0002),
                    "label_available_time": start + timedelta(days=s + 3),
                }
            )
    labels = pl.DataFrame(rows)
    scored = labels.select("instrument_id", "session", "predicted_net_alpha")
    return scored, labels


def _request() -> NetAlphaTrainingRequest:
    return NetAlphaTrainingRequest(
        artifact_id="na_cal",
        risk=RiskSettings(
            calibration_bucket_count=3, min_calibration_sessions=2
        ),
        bootstrap_resamples=50,
        bootstrap_alpha=0.05,
    )


def _lower_bounds(frame: pl.DataFrame, at: datetime) -> np.ndarray:
    return (
        frame.filter(pl.col("session") == at)
        .sort(["session", "instrument_id"])["net_alpha_lower_bound"]
        .fill_null(0.0)
        .to_numpy()
    )


def test_later_label_cannot_change_earlier_session_calibrated_score() -> None:
    scored, labels = _causal_fixture()
    cutoff = datetime(2024, 1, 15, tzinfo=UTC)
    calibrated = training._causal_oof_calibrate(scored, labels, _request(), 3)
    flipped = labels.with_columns(
        pl.when(pl.col("session") >= cutoff)
        .then(pl.lit(0.99))
        .otherwise(pl.col("risk_residual"))
        .alias("risk_residual")
    )
    recalibrated = training._causal_oof_calibrate(scored, flipped, _request(), 3)

    early = calibrated.filter(pl.col("session") < cutoff).sort(
        ["session", "instrument_id"]
    )
    early_flipped = recalibrated.filter(pl.col("session") < cutoff).sort(
        ["session", "instrument_id"]
    )
    assert np.array_equal(
        early["net_alpha_lower_bound"].fill_null(0.0).to_numpy(),
        early_flipped["net_alpha_lower_bound"].fill_null(0.0).to_numpy(),
    )
    assert np.array_equal(
        early["expected_net_alpha"].fill_null(0.0).to_numpy(),
        early_flipped["expected_net_alpha"].fill_null(0.0).to_numpy(),
    )


def test_minimum_calibration_history_stays_cash() -> None:
    scored, labels = _causal_fixture()
    calibrated = training._causal_oof_calibrate(scored, labels, _request(), 3)
    # At day 3 only session day 0 is label-available (0 + 3 <= 3), one session
    # below min_calibration_sessions=2, so the decision stays all cash.
    cash_day = datetime(2024, 1, 4, tzinfo=UTC)
    cash = _lower_bounds(calibrated, cash_day)
    assert cash.size > 0
    assert np.all(cash == 0.0)
    # A later session with >= 2 revealed label sessions may produce evidence.
    late_day = datetime(2024, 1, 20, tzinfo=UTC)
    late = _lower_bounds(calibrated, late_day)
    assert late.size > 0
    assert np.all(late >= 0.0)


def test_same_session_label_change_is_invariant() -> None:
    scored, labels = _causal_fixture()
    at_session = datetime(2024, 1, 13, tzinfo=UTC)
    calibrated = training._causal_oof_calibrate(scored, labels, _request(), 3)
    flipped = labels.with_columns(
        pl.when(pl.col("session") == at_session)
        .then(pl.lit(0.99))
        .otherwise(pl.col("risk_residual"))
        .alias("risk_residual")
    )
    recalibrated = training._causal_oof_calibrate(scored, flipped, _request(), 3)
    assert np.array_equal(
        _lower_bounds(calibrated, at_session),
        _lower_bounds(recalibrated, at_session),
    )


def test_causal_oof_calibration_is_deterministic() -> None:
    scored, labels = _causal_fixture()
    first = training._causal_oof_calibrate(scored, labels, _request(), 3)
    second = training._causal_oof_calibrate(scored, labels, _request(), 3)
    assert np.array_equal(
        first.sort(["session", "instrument_id"])["net_alpha_lower_bound"]
        .fill_null(0.0)
        .to_numpy(),
        second.sort(["session", "instrument_id"])["net_alpha_lower_bound"]
        .fill_null(0.0)
        .to_numpy(),
    )


def test_streaming_output_preserves_row_order_and_five_float64_columns() -> None:
    """ML_FULL_EXECUTION_P0_CALIBRATION_STREAM_03.

    Streaming calibration keeps input order, scores, and column types.

    The scored panel is reversed so decision sessions arrive interleaved;
    output rows must stay in the caller's original order with unchanged
    score values, the five economic columns must be Float64, and labels
    changed at or after a decision session must not move that session's or
    earlier calibrated values.
    """
    scored, labels = _causal_fixture()
    shuffled_scored = scored.reverse()
    shuffled_labels = labels.reverse()

    calibrated = training._causal_oof_calibrate(
        shuffled_scored, shuffled_labels, _request(), 3
    )
    assert calibrated.height == shuffled_scored.height
    for column in (
        "expected_active_alpha",
        "alpha_lower_bound",
        "expected_net_alpha",
        "net_alpha_lower_bound",
        "exit_cost_rate",
    ):
        assert calibrated.schema[column] == pl.Float64
    assert (
        calibrated["instrument_id"].to_list()
        == shuffled_scored["instrument_id"].to_list()
    )
    assert calibrated["session"].to_list() == shuffled_scored["session"].to_list()
    assert np.array_equal(
        calibrated["predicted_net_alpha"].to_numpy(),
        shuffled_scored["predicted_net_alpha"].to_numpy(),
    )

    # Causal isolation holds under interleaved input order as well.
    cutoff = datetime(2024, 1, 15, tzinfo=UTC)
    flipped_labels = shuffled_labels.with_columns(
        pl.when(pl.col("session") >= cutoff)
        .then(pl.lit(0.99))
        .otherwise(pl.col("risk_residual"))
        .alias("risk_residual")
    )
    recalibrated = training._causal_oof_calibrate(
        shuffled_scored, flipped_labels, _request(), 3
    )
    key = ["session", "instrument_id"]
    early = calibrated.filter(pl.col("session") < cutoff).sort(key)
    early_flipped = recalibrated.filter(pl.col("session") < cutoff).sort(key)
    assert np.array_equal(
        early["net_alpha_lower_bound"].fill_null(0.0).to_numpy(),
        early_flipped["net_alpha_lower_bound"].fill_null(0.0).to_numpy(),
    )
    assert np.array_equal(
        early["expected_net_alpha"].fill_null(0.0).to_numpy(),
        early_flipped["expected_net_alpha"].fill_null(0.0).to_numpy(),
    )
