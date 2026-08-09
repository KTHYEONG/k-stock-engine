"""Immutable model artifact registry.

There is no ``latest`` alias: callers pass a full artifact ID, and the registry
validates kind, feature schema, universe policy, label contract, and eligibility
time before returning a loaded model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src.core.instruments import AssetKind
from src.stocks.research.models import Model, ModelManifest

ARTIFACT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,128}$")

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """Consumer request that binds a loaded artifact to an eligibility window."""

    asset_kind: AssetKind
    feature_set: str
    feature_schema_hash: str
    decision_time: datetime


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Model
    manifest: ModelManifest


class ModelArtifactRegistry:
    """Filesystem-backed immutable artifact store with fail-closed validation."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _artifact_dir(self, artifact_id: str) -> Path:
        if not ARTIFACT_ID_RE.match(artifact_id):
            raise ValueError(f"invalid artifact_id {artifact_id!r}")
        return self.root / artifact_id

    def publish(self, model: Model, manifest: ModelManifest) -> str:
        artifact_dir = self._artifact_dir(manifest.artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / MANIFEST_FILENAME
        if manifest_path.exists():
            raise ValueError(f"artifact already exists: {manifest.artifact_id}")
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(_manifest_to_dict(manifest), fh, indent=2, default=str)
        joblib.dump(model, artifact_dir / MODEL_FILENAME)
        return manifest.artifact_id

    def load(self, artifact_id: str, request: PredictionRequest) -> LoadedModel:
        artifact_dir = self._artifact_dir(artifact_id)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} not found at {artifact_dir}")
        manifest_path = artifact_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} has no manifest")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = _manifest_from_dict(json.load(fh))

        self._validate_manifest(manifest, request)
        model = joblib.load(artifact_dir / MODEL_FILENAME)
        return LoadedModel(model=model, manifest=manifest)

    def _validate_manifest(self, manifest: ModelManifest, request: PredictionRequest) -> None:
        if manifest.asset_kind is not request.asset_kind:
            raise ValueError(
                f"asset kind mismatch: artifact {manifest.asset_kind.value}, "
                f"request {request.asset_kind.value}"
            )
        if manifest.feature_set != request.feature_set:
            raise ValueError(
                f"feature-set mismatch: artifact {manifest.feature_set!r}, "
                f"request {request.feature_set!r}"
            )
        if manifest.feature_schema_hash != request.feature_schema_hash:
            raise ValueError("feature schema hash mismatch")
        if not (
            _parse_iso(manifest.eligible_from)
            <= request.decision_time
            <= _parse_iso(manifest.eligible_to)
        ):
            raise ValueError(
                f"artifact not eligible at {request.decision_time.isoformat()} "
                f"(eligible {manifest.eligible_from} .. {manifest.eligible_to})"
            )

    def write_metrics(self, artifact_id: str, metrics: dict[str, object]) -> None:
        artifact_dir = self._artifact_dir(artifact_id)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} not found")
        with (artifact_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _manifest_to_dict(manifest: ModelManifest) -> dict[str, object]:
    return {
        "artifact_id": manifest.artifact_id,
        "asset_kind": manifest.asset_kind.value,
        "feature_set": manifest.feature_set,
        "feature_schema_hash": manifest.feature_schema_hash,
        "universe_policy_hash": manifest.universe_policy_hash,
        "label_definition": manifest.label_definition,
        "label_horizon_sessions": manifest.label_horizon_sessions,
        "eligible_from": manifest.eligible_from,
        "eligible_to": manifest.eligible_to,
        "model_type": manifest.model_type,
        "params": manifest.params or {},
    }


def _manifest_from_dict(data: dict[str, object]) -> ModelManifest:
    label_horizon = int(str(data["label_horizon_sessions"]))
    raw_params = data.get("params") or {}
    params = dict(raw_params) if isinstance(raw_params, dict) else {}
    return ModelManifest(
        artifact_id=str(data["artifact_id"]),
        asset_kind=AssetKind(str(data["asset_kind"])),
        feature_set=str(data["feature_set"]),
        feature_schema_hash=str(data["feature_schema_hash"]),
        universe_policy_hash=str(data["universe_policy_hash"]),
        label_definition=str(data["label_definition"]),
        label_horizon_sessions=label_horizon,
        eligible_from=str(data["eligible_from"]),
        eligible_to=str(data["eligible_to"]),
        model_type=str(data.get("model_type", "baseline")),
        params=params,
    )
