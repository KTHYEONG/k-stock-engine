"""Immutable model artifact registry.

There is no ``latest`` alias: callers pass a full artifact ID, and the registry
validates kind, feature schema, universe policy, label contract, and eligibility
time before returning a loaded model.
"""
from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib

from src.core.instruments import AssetKind
from src.stocks.research.models import Model, ModelManifest

ARTIFACT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,128}$")

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"
FORWARD_HOLDOUT_FILENAME = "forward_holdout.json"


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


@dataclass(frozen=True, slots=True)
class ArtifactRetentionCandidate:
    """An artifact eligible for explicit, operator-approved pruning."""

    artifact_id: str
    reason: str


class ModelArtifactRegistry:
    """Filesystem-backed immutable artifact store with fail-closed validation."""

    def __init__(self, root: Path | None = None, *, in_memory: bool = False):
        self.root = Path(root) if root is not None else Path("in-memory")
        self._in_memory = in_memory
        self._models: dict[str, Model] = {}
        self._manifests: dict[str, ModelManifest] = {}
        self._metrics: dict[str, dict[str, object]] = {}
        self._forward_holdouts: dict[str, dict[str, object]] = {}

    @classmethod
    def in_memory(cls) -> ModelArtifactRegistry:
        """Create a process-local registry that never writes model artifacts."""
        return cls(in_memory=True)

    def list_artifact_ids(self) -> tuple[str, ...]:
        """List immutable artifact directories with a manifest, deterministically."""
        if self._in_memory:
            return tuple(sorted(self._manifests))
        if not self.root.exists():
            return ()
        return tuple(
            sorted(
                path.name
                for path in self.root.iterdir()
                if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
            )
        )

    def retention_candidates(
        self,
        keep_ids: Iterable[str] = (),
        *,
        retain_promoted: bool = True,
    ) -> tuple[ArtifactRetentionCandidate, ...]:
        """Return candidates while protecting explicit and promoted artifacts."""
        keep = frozenset(keep_ids)
        candidates: list[ArtifactRetentionCandidate] = []
        for artifact_id in self.list_artifact_ids():
            if artifact_id in keep:
                continue
            if retain_promoted and self.is_promoted(artifact_id):
                continue
            if self._in_memory:
                reason = "unpromoted" if artifact_id in self._metrics else "missing-metrics"
                candidates.append(ArtifactRetentionCandidate(artifact_id, reason))
                continue
            metrics_path = self._artifact_dir(artifact_id) / METRICS_FILENAME
            reason = "unpromoted" if metrics_path.is_file() else "missing-metrics"
            candidates.append(ArtifactRetentionCandidate(artifact_id, reason))
        return tuple(candidates)

    def prune(
        self,
        candidates: Iterable[ArtifactRetentionCandidate],
        *,
        apply: bool = False,
    ) -> int:
        """Delete only supplied candidates when ``apply`` is explicitly true."""
        if not apply:
            return 0
        deleted = 0
        for candidate in candidates:
            if self._in_memory:
                if candidate.artifact_id in self._manifests:
                    self._models.pop(candidate.artifact_id, None)
                    self._manifests.pop(candidate.artifact_id, None)
                    self._metrics.pop(candidate.artifact_id, None)
                    self._forward_holdouts.pop(candidate.artifact_id, None)
                    deleted += 1
                continue
            artifact_dir = self._artifact_dir(candidate.artifact_id)
            if artifact_dir.is_dir():
                shutil.rmtree(artifact_dir)
                deleted += 1
        return deleted

    def _artifact_dir(self, artifact_id: str) -> Path:
        if not ARTIFACT_ID_RE.match(artifact_id):
            raise ValueError(f"invalid artifact_id {artifact_id!r}")
        return self.root / artifact_id

    def publish(self, model: Model, manifest: ModelManifest) -> str:
        if self._in_memory:
            if manifest.artifact_id in self._manifests:
                raise ValueError(f"artifact already exists: {manifest.artifact_id}")
            self._models[manifest.artifact_id] = model
            self._manifests[manifest.artifact_id] = manifest
            return manifest.artifact_id
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
        if self._in_memory:
            manifest = self.read_manifest(artifact_id)
            self._validate_manifest(manifest, request)
            return LoadedModel(model=self._models[artifact_id], manifest=manifest)
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

    def read_manifest(self, artifact_id: str) -> ModelManifest:
        """Return the frozen manifest of ``artifact_id`` without loading the model.

        Unlike ``load`` this performs no eligibility/schema validation; it is
        used by replay and scheduling code that must inspect the eligibility
        window before binding a decision time.
        """
        if self._in_memory:
            self._artifact_dir(artifact_id)
            try:
                return self._manifests[artifact_id]
            except KeyError as exc:
                raise FileNotFoundError(f"artifact {artifact_id!r} not found") from exc
        artifact_dir = self._artifact_dir(artifact_id)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} not found at {artifact_dir}")
        manifest_path = artifact_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} has no manifest")
        with manifest_path.open("r", encoding="utf-8") as fh:
            return _manifest_from_dict(json.load(fh))

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
        if self._in_memory:
            self.read_manifest(artifact_id)
            self._metrics[artifact_id] = dict(metrics)
            return
        artifact_dir = self._artifact_dir(artifact_id)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} not found")
        with (artifact_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)

    def read_metrics(self, artifact_id: str) -> dict[str, object]:
        """Return the metrics stored for an artifact."""
        if self._in_memory:
            self.read_manifest(artifact_id)
            try:
                return dict(self._metrics[artifact_id])
            except KeyError as exc:
                raise FileNotFoundError(f"artifact {artifact_id!r} has no metrics") from exc
        artifact_dir = self._artifact_dir(artifact_id)
        path = artifact_dir / METRICS_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} has no metrics")
        with path.open("r", encoding="utf-8") as fh:
            return cast(dict[str, object], json.load(fh))

    def is_promoted(self, artifact_id: str) -> bool:
        """Return whether immutable promotion evidence marks an artifact promoted."""
        if self._in_memory:
            return self._metrics.get(artifact_id, {}).get("promoted") is True
        artifact_dir = self._artifact_dir(artifact_id)
        metrics_path = artifact_dir / METRICS_FILENAME
        if not metrics_path.exists():
            return False
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        return isinstance(metrics, dict) and metrics.get("promoted") is True

    def write_forward_holdout(
        self,
        artifact_id: str,
        fingerprint: str,
        evidence: dict[str, object],
    ) -> str:
        """Persist one candidate's forward-holdout evidence atomically.

        A candidate fingerprint may be inspected once; writing a second
        evaluation for the same fingerprint is rejected with ``ValueError``.
        """
        if self._in_memory:
            self.read_manifest(artifact_id)
            memory_existing = self._forward_holdouts.get(artifact_id, {})
            if memory_existing.get("fingerprint") == fingerprint:
                raise ValueError(
                    f"forward holdout for candidate fingerprint {fingerprint!r} "
                    f"was already inspected for {artifact_id!r}"
                )
            self._forward_holdouts[artifact_id] = {
                "fingerprint": fingerprint,
                "evidence": evidence,
            }
            return fingerprint
        artifact_dir = self._artifact_dir(artifact_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / FORWARD_HOLDOUT_FILENAME
        existing: dict[str, object] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        if existing.get("fingerprint") == fingerprint:
            raise ValueError(
                f"forward holdout for candidate fingerprint {fingerprint!r} "
                f"was already inspected for {artifact_id!r}"
            )
        payload: dict[str, object] = {"fingerprint": fingerprint, "evidence": evidence}
        temp = path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        temp.replace(path)
        return fingerprint

    def read_forward_holdout(self, artifact_id: str) -> dict[str, object] | None:
        """Return the persisted forward-holdout evidence for ``artifact_id``."""
        if self._in_memory:
            record = self._forward_holdouts.get(artifact_id)
            return dict(record) if record is not None else None
        path = self._artifact_dir(artifact_id) / FORWARD_HOLDOUT_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return cast(dict[str, object], json.load(fh))


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
