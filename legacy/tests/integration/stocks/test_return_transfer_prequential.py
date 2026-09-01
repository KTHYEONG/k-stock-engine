"""Integration contracts for common-calendar return-transfer studies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from legacy.stocks.ml.contracts import NetAlphaResearchData, NetAlphaTrainingRequest
from legacy.stocks.ml.return_transfer import ReturnTransferSettings, evaluate_return_transfer_study
from legacy.stocks.research.artifacts import ModelArtifactRegistry


def _data() -> NetAlphaResearchData:
    sessions = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8))
    frame = pl.DataFrame(
        {
            "instrument_id": ["A", "B"] * len(sessions),
            "session": [s for s in sessions for _ in range(2)],
            "signal": [0.1, 0.2] * len(sessions),
        }
    )
    labels = frame.select(
        "instrument_id", "session"
    ).with_columns(
        pl.lit(0.01).alias("net_alpha_target"),
        pl.col("session").alias("label_available_time"),
        pl.lit(0.02).alias("gross_return"),
        pl.lit(0.0).alias("risk_residual"),
        pl.lit(0.001).alias("reference_cost"),
    )
    return NetAlphaResearchData(
        feature_frame=frame,
        labels_by_horizon={10: labels},
        manifest=SimpleNamespace(),
    )


def test_RETURN_TRANSFER_06_COMMON_PREQUENTIAL_CALENDAR() -> None:
    """RETURN_TRANSFER_06_COMMON_PREQUENTIAL_CALENDAR: common decision keys."""
    payload = evaluate_return_transfer_study(
        _data(),
        NetAlphaTrainingRequest(artifact_id="rt06"),
        ReturnTransferSettings(),
        registry=ModelArtifactRegistry(Path("tmp")),
    )
    candidates = payload["per_candidate"]
    assert len(candidates) == 8
    hashes = {item["decision_key_hash"] for item in candidates.values()}
    assert len(hashes) == 1
    assert len(payload["forward_holdout_sessions"]) >= 1


def test_RETURN_TRANSFER_07_INCUMBENT_IMPROVEMENT() -> None:
    """RETURN_TRANSFER_07_INCUMBENT_IMPROVEMENT: paired deltas are explicit."""
    payload = evaluate_return_transfer_study(
        _data(),
        NetAlphaTrainingRequest(artifact_id="rt07"),
        ReturnTransferSettings(),
        registry=ModelArtifactRegistry(Path("tmp")),
    )
    evaluation = payload["EVAL"]
    assert evaluation["base_delta"] > 0.0
    assert evaluation["stress_delta"] > 0.0
    assert evaluation["mdd_worsening"] <= 0.02
