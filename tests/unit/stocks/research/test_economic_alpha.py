"""Causal net-alpha calibration: time isolation, fail-closed buckets, costs."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.core.costs import default_base_schedule
from src.stocks.research.economic_alpha import (
    ALPHA_COLUMN,
    LOWER_BOUND_COLUMN,
    NET_ALPHA_COLUMN,
    CausalAlphaCalibrator,
)


def _observations(
    n_sessions: int = 40,
    n_tickers: int = 20,
    start: datetime | None = None,
    horizon: int = 5,
) -> pl.DataFrame:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "instrument_id": f"KRX:{t:06d}",
            "session": start + timedelta(days=s),
            "score": float(t) + (s % 3),
            "residual_o2o_5d": 0.01 if t % 4 == 0 else -0.005,
            "label_available_time": start + timedelta(days=s + horizon + 1),
        }
        for s in range(n_sessions)
        for t in range(n_tickers)
    ]
    return pl.DataFrame(rows)

def _scored(
    n_tickers: int = 20,
    decision: datetime | None = None,
) -> pl.DataFrame:
    decision = decision or datetime(2024, 2, 10, tzinfo=UTC)
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{t:06d}" for t in range(n_tickers)],
            "session": [decision] * n_tickers,
            "pred_score": [float(t) for t in range(n_tickers)],
            "close": [50_000.0] * n_tickers,
            "adtv": [1e9] * n_tickers,
            "sector": ["S0"] * n_tickers,
        }
    )


def _positive_observations(
    n_sessions: int = 40,
    n_tickers: int = 20,
) -> pl.DataFrame:
    """Residuals strongly correlated with score so top buckets clear the bound."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for s in range(n_sessions):
        for t in range(n_tickers):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": start + timedelta(days=s),
                    "score": float(t) + (s % 3),
                    "residual_o2o_5d": float(0.004 * t + rng.normal(0.0, 0.0002)),
                    "label_available_time": start + timedelta(days=s + 6),
                }
            )  # noqa: PERF401
    return pl.DataFrame(rows)


def test_rejects_missing_and_non_finite_inputs() -> None:
    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10)
    scored = _scored()
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    observations = _observations()
    with pytest.raises(ValueError, match="instrument_id"):
        cal.transform(scored.drop("instrument_id"), observations, decision, default_base_schedule())
    with pytest.raises(ValueError, match="residual_o2o_5d"):
        cal.transform(scored, observations.drop("residual_o2o_5d"), decision, default_base_schedule())
    bad = observations.with_columns(
        pl.when(pl.col("instrument_id") == "KRX:000000")
        .then(float("inf"))
        .otherwise(pl.col("score"))
        .alias("score")
    )
    with pytest.raises(ValueError, match="non-finite"):
        cal.transform(scored, bad, decision, default_base_schedule())
    bad_score = scored.with_columns(
        pl.when(pl.col("instrument_id") == "KRX:000000")
        .then(float("nan"))
        .otherwise(pl.col("pred_score"))
        .alias("pred_score")
    )
    with pytest.raises(ValueError, match="non-finite"):
        cal.transform(bad_score, observations, decision, default_base_schedule())


def test_never_uses_future_or_current_labels() -> None:
    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=50)
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    observations = _observations()
    future_flipped = observations.with_columns(
        pl.when(pl.col("session") >= datetime(2024, 2, 9, tzinfo=UTC))
        .then(pl.lit(0.99))
        .otherwise(pl.col("residual_o2o_5d"))
        .alias("residual_o2o_5d")
    )
    baseline = cal.transform(_scored(), observations, decision, default_base_schedule())
    flipped = cal.transform(_scored(), future_flipped, decision, default_base_schedule())
    assert baseline.select(ALPHA_COLUMN, NET_ALPHA_COLUMN).to_dicts() == flipped.select(
        ALPHA_COLUMN, NET_ALPHA_COLUMN
    ).to_dicts()

    visible = _scored().with_columns(
        (pl.col("session") + timedelta(days=60)).alias("session")
    )
    transformed = cal.transform(visible, observations, decision, default_base_schedule())
    assert transformed.filter(pl.col("session") > decision).is_empty()


def test_insufficient_calibration_history_fails_closed_to_null() -> None:
    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10)
    decision = datetime(2024, 1, 8, tzinfo=UTC)
    observations = _observations(n_sessions=3)
    out = cal.transform(_scored(), observations, decision, default_base_schedule())
    assert out[ALPHA_COLUMN].null_count() == out.height
    assert out[NET_ALPHA_COLUMN].null_count() == out.height


def test_bucket_evidence_is_deterministic_for_same_seed() -> None:
    cal_a = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=40)
    cal_b = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=40)
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    observations = _observations()
    a = cal_a.transform(_scored(), observations, decision, default_base_schedule())
    b = cal_b.transform(_scored(), observations, decision, default_base_schedule())
    assert a[ALPHA_COLUMN].to_list() == b[ALPHA_COLUMN].to_list()
    assert a[LOWER_BOUND_COLUMN].to_list() == b[LOWER_BOUND_COLUMN].to_list()
    assert [e.to_json_safe() for e in cal_a.bucket_evidence] == [
        e.to_json_safe() for e in cal_b.bucket_evidence
    ]


def test_net_alpha_uses_round_trip_cost_not_a_fixed_bps() -> None:
    from src.core.costs import CostPoint, CostSchedule

    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=80)
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    observations = _positive_observations()
    cheap = CostSchedule(
        name="cheap",
        points=(CostPoint(decision, 0.0, 0.0, 0.0),),
    )
    expensive = CostSchedule(
        name="expensive",
        points=(CostPoint(decision, 0.001, 0.01, 50.0),),
    )
    cheap_out = cal.transform(_scored(), observations, decision, cheap)
    expensive_out = cal.transform(_scored(), observations, decision, expensive)
    cheap_nets = cheap_out[NET_ALPHA_COLUMN].drop_nulls().to_list()
    expensive_nets = expensive_out[NET_ALPHA_COLUMN].drop_nulls().to_list()
    assert cheap_nets
    assert expensive_nets
    assert max(expensive_nets) < max(cheap_nets)
    assert min(expensive_nets) < min(cheap_nets)


def test_frozen_state_reproduces_transform_evidence() -> None:
    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=50)
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    observations = _observations()
    scored = _scored()
    live = cal.transform(scored, observations, decision, default_base_schedule())
    state = cal.calibration_state()
    frozen = CausalAlphaCalibrator.from_state(state)
    frozen_out = frozen.apply_frozen(scored)
    assert frozen_out.select(ALPHA_COLUMN, NET_ALPHA_COLUMN, LOWER_BOUND_COLUMN).to_dicts() == live.select(
        ALPHA_COLUMN, NET_ALPHA_COLUMN, LOWER_BOUND_COLUMN
    ).to_dicts()
    assert state["round_trip_cost"] > 0.0
    assert state["exit_cost_rate"] > 0.0


def test_negative_bootstrap_lower_bound_bucket_is_null() -> None:
    cal = CausalAlphaCalibrator(bucket_count=5, min_calibration_sessions=10, n_bootstrap=200)
    decision = datetime(2024, 2, 10, tzinfo=UTC)
    rows: list[dict] = []
    for s in range(30):
        for t in range(10):
            rows.append(  # noqa: PERF401
                {
                    "instrument_id": f"KRX:{t:06d}",
                    "session": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=s),
                    "score": float(t),
                    "residual_o2o_5d": -0.02 if t >= 5 else 0.001,
                    "label_available_time": datetime(2024, 1, 1, tzinfo=UTC)
                    + timedelta(days=s + 6),
                }
            )  # noqa: PERF401
    observations = pl.DataFrame(rows)
    out = cal.transform(_scored(), observations, decision, default_base_schedule())
    evidence = cal.bucket_evidence
    assert evidence
    assert all(
        e.expected_active_alpha is None or e.alpha_lower_bound is not None
        for e in evidence
    )
    assert out[LOWER_BOUND_COLUMN].null_count() > 0
