"""Score-model workflow wiring tests."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.workflows.contracts import ScoringRequest
from src.stocks.workflows.score_model import score_model
from src.stocks.workflows.train_model import train_model
from src.stocks.ml.contracts import NetAlphaTrainingRequest
from tests.fixtures.stocks.helpers import (
    stock_net_alpha_composed_df,
    stock_net_alpha_manifest,
)


def test_score_model_loads_artifact_and_scores(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=120, n_tickers=8)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    train_model(
        snapshot,
        registry,
        NetAlphaTrainingRequest(
            artifact_id="stock_net_alpha_20240101",
            fold_count=2,
            candidate_horizon_sessions=(5,),
            bootstrap_resamples=50,
        ),
    )
    decision = datetime(2024, 4, 29, 0, 0, tzinfo=UTC)
    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(
            artifact_id="stock_net_alpha_20240101", decision_time=decision
        ),
    )
    assert not scored.is_empty()
    assert "predicted_net_alpha" in scored.columns


def test_score_model_rejects_unavailable_artifact(tmp_path) -> None:
    df = stock_net_alpha_composed_df(n_sessions=60, n_tickers=6)
    manifest = stock_net_alpha_manifest(columns=df.columns)
    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    snapshot = DatasetSnapshot(manifest=manifest, frame=df)
    with pytest.raises(FileNotFoundError):
        score_model(
            snapshot,
            registry,
            ScoringRequest(
                artifact_id="missing_net_alpha",
                decision_time=datetime(2024, 2, 20, tzinfo=UTC),
            ),
        )


import json

from src.stocks.ml.features import (
    fit_model_feature_schema,
)
from src.stocks.research.models import AssetKind, ModelManifest
from src.stocks.ml.contracts import CANONICAL_FEATURE_SET


class _V6Model:
    """Minimal model that records the columns it was asked to score."""

    def __init__(self, manifest: ModelManifest, learner_columns: tuple[str, ...]):
        self._manifest = manifest
        self.learner_columns = learner_columns
        self.no_trade = False
        self.seen_columns: tuple[str, ...] = ()

    def fit(self, train: object, validation: object) -> None:  # pragma: no cover
        return None

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.seen_columns = tuple(frame.columns)
        return frame.with_columns(pl.lit(0.5).alias("predicted_net_alpha"))

    def manifest(self) -> ModelManifest:
        return self._manifest


def _v6_roles() -> dict[str, str]:
    return {"mom_5d": "ALPHA", "vol_20d": "ALPHA"}


def _feature_frame(with_nulls: bool) -> pl.DataFrame:
    session = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(4):
        mom = None if (with_nulls and i == 0) else float(i + 1)
        rows.append(
            {
                "instrument_id": f"KRX:0000{i + 1}",
                "session": session,
                "sector": f"S{i % 2}",
                "mom_5d": mom,
                "vol_20d": float(i + 2),
            }
        )
    return pl.DataFrame(rows)


def _v6_manifest(
    artifact_id: str, schema_json: str, feature_schema_hash: str = "h"
) -> ModelManifest:
    return ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set=CANONICAL_FEATURE_SET,
        feature_schema_hash=feature_schema_hash,
        universe_policy_hash="u",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=10,
        eligible_from=datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        eligible_to=datetime(2024, 12, 31, tzinfo=UTC).isoformat(),
        model_type="net_alpha_elastic_net",
        params={"feature_transform_schema": schema_json},
    )


def test_feature_contract_v6_frozen_transform(tmp_path) -> None:
    """FEATURE_CONTRACT_V6_FROZEN_TRANSFORM.

    A v6 artifact applies its persisted schema and emits exactly its ordered
    learner columns even when the scoring frame's null pattern differs from the
    fit frame. Malformed, fingerprint-mismatched, or source-column-missing
    payloads raise ValueError before model.predict.
    """
    fit_frame = _feature_frame(with_nulls=True)
    schema = fit_model_feature_schema(fit_frame, _v6_roles())
    schema_json = json.dumps(schema.to_json())

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    score_frame = _feature_frame(with_nulls=False)
    snapshot_manifest = stock_net_alpha_manifest(columns=score_frame.columns)
    model = _V6Model(
        _v6_manifest(
            "v6_artifact", schema_json, feature_schema_hash=snapshot_manifest.schema_hash
        ),
        schema.learner_columns,
    )
    registry.publish(model, model.manifest())
    snapshot = DatasetSnapshot(manifest=snapshot_manifest, frame=score_frame)

    scored = score_model(
        snapshot,
        registry,
        ScoringRequest(artifact_id="v6_artifact", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
    )
    assert "predicted_net_alpha" in scored.columns
    emitted = [c for c in scored.columns if c in schema.learner_columns]
    assert emitted == list(schema.learner_columns)

    # Malformed payload.
    bad_model = _V6Model(
        _v6_manifest(
            "v6_bad", "not-json", feature_schema_hash=snapshot_manifest.schema_hash
        ),
        schema.learner_columns,
    )
    registry.publish(bad_model, bad_model.manifest())
    with pytest.raises(ValueError, match="malformed JSON"):
        score_model(
            snapshot,
            registry,
            ScoringRequest(artifact_id="v6_bad", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
        )

    # Fingerprint mismatch.
    tampered = dict(schema.to_json())
    tampered["fingerprint"] = "wrong-fingerprint"
    bad_fp_model = _V6Model(
        _v6_manifest(
            "v6_fp", json.dumps(tampered), feature_schema_hash=snapshot_manifest.schema_hash
        ),
        schema.learner_columns,
    )
    registry.publish(bad_fp_model, bad_fp_model.manifest())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        score_model(
            snapshot,
            registry,
            ScoringRequest(artifact_id="v6_fp", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
        )

    # Missing source column in the scoring frame.
    missing_col_frame = score_frame.drop("vol_20d")
    missing_snapshot = DatasetSnapshot(manifest=snapshot_manifest, frame=missing_col_frame)
    with pytest.raises(ValueError, match="sources missing from frame"):
        score_model(
            missing_snapshot,
            registry,
            ScoringRequest(artifact_id="v6_artifact", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
        )


def test_feature_alias_growth_recovery(tmp_path) -> None:
    """GROWTH_RECOVERY_FEATURE_ALIAS_01.

    A manifest whose schema requires ep_ratio and bp_ratio scores a frame that
    only exposes the prefixed feature__ep_ratio/feature__bp_ratio columns via the
    feature-source binding, and raises ValueError before model.predict when the
    canonical and prefixed columns are both present with conflicting values.
    """
    roles = {"ep_ratio": "ALPHA", "bp_ratio": "ALPHA"}
    fit_frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:0001", "KRX:0002"],
            "session": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            "sector": ["S1", "S2"],
            "ep_ratio": [1.0, 2.0],
            "bp_ratio": [3.0, 4.0],
        }
    )
    schema = fit_model_feature_schema(fit_frame, roles)
    schema_json = json.dumps(schema.to_json())

    prefixed_frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:0001", "KRX:0002"],
            "session": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            "sector": ["S1", "S2"],
            "feature__ep_ratio": [1.0, 2.0],
            "feature__bp_ratio": [3.0, 4.0],
        }
    )
    prefixed_manifest = stock_net_alpha_manifest(columns=prefixed_frame.columns)

    registry = ModelArtifactRegistry(tmp_path / "artifacts")
    model = _V6Model(
        _v6_manifest(
            "alias_artifact", schema_json, feature_schema_hash=prefixed_manifest.schema_hash
        ),
        schema.learner_columns,
    )
    registry.publish(model, model.manifest())

    prefixed_snapshot = DatasetSnapshot(manifest=prefixed_manifest, frame=prefixed_frame)
    scored = score_model(
        prefixed_snapshot,
        registry,
        ScoringRequest(artifact_id="alias_artifact", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
    )
    assert "predicted_net_alpha" in scored.columns

    conflict_frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:0001"],
            "session": [datetime(2024, 1, 1, tzinfo=UTC)],
            "sector": ["S1"],
            "ep_ratio": [1.0],
            "feature__ep_ratio": [9.0],
        }
    )
    conflict_manifest = stock_net_alpha_manifest(columns=conflict_frame.columns)
    conflict_model = _V6Model(
        _v6_manifest(
            "alias_conflict", schema_json, feature_schema_hash=conflict_manifest.schema_hash
        ),
        schema.learner_columns,
    )
    registry.publish(conflict_model, conflict_model.manifest())
    conflict_snapshot = DatasetSnapshot(
        manifest=conflict_manifest,
        frame=conflict_frame,
    )
    with pytest.raises(ValueError, match="conflict"):
        score_model(
            conflict_snapshot,
            registry,
            ScoringRequest(artifact_id="alias_conflict", decision_time=datetime(2024, 6, 1, tzinfo=UTC)),
        )
