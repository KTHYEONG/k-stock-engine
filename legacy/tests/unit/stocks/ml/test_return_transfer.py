"""Contract tests for the return-transfer research primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import math
import polars as pl
import pytest

from legacy.stocks.ml.contracts import NetAlphaResearchData
from legacy.stocks.ml.execution_replay import ExecutionReplayEvidence
from legacy.stocks.ml.return_transfer import (
    ReturnDistributionLabels,
    ReturnTransferSettings,
    build_prequential_transition_ledger,
    build_return_distribution_labels,
    build_return_transfer_panel,
    fit_return_distribution_oof,
)


def _manifest() -> SimpleNamespace:
    return SimpleNamespace()


def _labels(sessions: tuple[datetime, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["A", "B"] * len(sessions),
            "session": [session for session in sessions for _ in range(2)],
            "gross_return": [0.10, -0.05] * len(sessions),
            "risk_residual": [0.08, -0.08] * len(sessions),
            "reference_cost": [0.01, 0.01] * len(sessions),
        }
    )


def test_RETURN_TRANSFER_01_PIT_FEATURE_VIEWS() -> None:
    """RETURN_TRANSFER_01_PIT_FEATURE_VIEWS: certified view excludes fundamentals."""
    now = datetime(2024, 1, 1, tzinfo=UTC)
    feature = pl.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "session": [now, now],
            "bp_ratio": [1.0, 2.0],
            "ep_ratio": [0.5, 0.7],
            "momentum": [0.1, 0.2],
        }
    )
    data = NetAlphaResearchData(
        feature_frame=feature,
        labels_by_horizon={10: _labels((now,))},
        manifest=_manifest(),
    )
    certified = build_return_transfer_panel(data, certified=True)
    assert "bp_ratio" not in certified.columns
    assert "ep_ratio" not in certified.columns
    assert "net_alpha_target" not in certified.columns
    assert (certified["available_at"] <= certified["session"]).all()


def test_RETURN_TRANSFER_02_LABEL_SEPARATION() -> None:
    """RETURN_TRANSFER_02_LABEL_SEPARATION: absolute log return and costs."""
    now = datetime(2024, 1, 1, tzinfo=UTC)
    labels = _labels((now,)).with_columns(pl.lit(99.0).alias("reference_notional"))
    result = build_return_distribution_labels(labels, horizon_sessions=10)
    first = result.frame.row(0, named=True)
    assert math.isclose(first["log_return"], math.log1p(0.10))
    assert math.isclose(first["downside"], 0.0)
    assert math.isclose(first["cost_enter"], 0.005)
    changed = build_return_distribution_labels(
        labels.with_columns(pl.lit(1.0).alias("reference_notional")),
        horizon_sessions=10,
    )
    assert result.frame.select("log_return", "downside").equals(
        changed.frame.select("log_return", "downside")
    )


def test_RETURN_TRANSFER_03_OOF_ISOLATION() -> None:
    """RETURN_TRANSFER_03_OOF_ISOLATION: validation mutation is isolated."""
    sessions = tuple(
        datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)
    )
    panel = pl.DataFrame(
        {
            "instrument_id": ["A", "B"] * len(sessions),
            "session": [session for session in sessions for _ in range(2)],
            "signal": [0.1, 0.2] * len(sessions),
        }
    )
    labels = build_return_distribution_labels(_labels(sessions), horizon_sessions=10)
    fold = SimpleNamespace(
        train_sessions=sessions[:2],
        validation_sessions=sessions[2:],
    )
    settings = ReturnTransferSettings()
    baseline = fit_return_distribution_oof(panel, labels, (fold,), settings)
    mutated_frame = labels.frame.with_columns(
        pl.when(pl.col("session") >= sessions[2])
        .then(100.0)
        .otherwise(pl.col("log_return"))
        .alias("log_return")
    )
    mutated = fit_return_distribution_oof(
        panel,
        ReturnDistributionLabels(horizon_sessions=10, frame=mutated_frame),
        (fold,),
        settings,
    )
    assert baseline.select("mu", "q20", "residual_rank_pred").equals(
        mutated.select("mu", "q20", "residual_rank_pred")
    )
    assert (baseline["feature_timestamp"] <= baseline["decision_time"]).all()


def test_RETURN_TRANSFER_04_TRANSITION_ACCOUNTING() -> None:
    """RETURN_TRANSFER_04_TRANSITION_ACCOUNTING: impossible state fails."""
    evidence = ExecutionReplayEvidence(
        base_log_growth=(0.01,),
        stress_log_growth=(0.01,),
        segment_ids=(1,),
        planned_cycles=1,
        filled_orders=0,
        cash_session_fraction=0.0,
        turnover=0.1,
        observed_interval_count=1,
        invested_interval_count=1,
    )
    scores = pl.DataFrame(
        {
            "session": [datetime(2024, 1, 1, tzinfo=UTC)],
            "mu": [0.01],
            "q20": [-0.01],
        }
    )
    with pytest.raises(ValueError, match="reconciliation failure"):
        build_prequential_transition_ledger(scores, evidence)
