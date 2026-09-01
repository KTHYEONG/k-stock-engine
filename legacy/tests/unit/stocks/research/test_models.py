"""Model protocol and deterministic baseline tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.core.instruments import AssetKind
from legacy.stocks.research.models import (
    DeterministicBaseline,
    Model,
    ModelManifest,
)


def _manifest() -> ModelManifest:
    return ModelManifest(
        artifact_id="baseline_v1",
        asset_kind=AssetKind.STOCK,
        feature_set="stock_alpha_v1",
        feature_schema_hash="hash",
        universe_policy_hash="universe",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        eligible_from="2024-01-01T00:00:00+00:00",
        eligible_to="2024-12-31T00:00:00+00:00",
        model_type="deterministic_baseline",
    )


def _panel() -> pl.DataFrame:
    sessions = [datetime(2024, 1, 1, tzinfo=UTC) + __import__("datetime").timedelta(days=i) for i in range(5)]
    return pl.DataFrame(
        {
            "instrument_id": [f"KRX:{i:05d}" for i in range(4)],
            "session": [sessions[0]] * 4,
            "feature_momentum_5d": [0.1, 0.4, 0.9, 0.2],
            "open": [100.0] * 4,
        }
    )


def test_model_protocol_is_runtime_checkable() -> None:
    model = DeterministicBaseline(_manifest())
    assert isinstance(model, Model)


def test_deterministic_baseline_scores_by_ranking_feature() -> None:
    frame = _panel()
    model = DeterministicBaseline(_manifest(), ranking_feature="feature_momentum_5d")
    model.fit(frame, frame.head(0))
    scored = model.predict(frame)
    assert "pred_score" in scored.columns
    assert scored["pred_score"].to_list() == [0.1, 0.4, 0.9, 0.2]


def test_deterministic_baseline_missing_feature_rejected() -> None:
    model = DeterministicBaseline(_manifest(), ranking_feature="missing_feature")
    with pytest.raises(ValueError, match="missing_feature"):
        model.predict(_panel())


def test_manifest_eligible_time_range() -> None:
    manifest = _manifest()
    assert manifest.eligible_time_range == (
        "2024-01-01T00:00:00+00:00",
        "2024-12-31T00:00:00+00:00",
    )
