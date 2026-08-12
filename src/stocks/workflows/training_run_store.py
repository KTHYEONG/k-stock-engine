"""Durable, fingerprinted recovery for multi-stage training runs.

``TrainingRunStore`` gives the 81-trial training job a durable unit of progress
under the repository's project-local data root: every completed stage (screen
study, promotion order, fold refits, replay evidence, resource telemetry) is
atomically checkpointed under ``<run_root>/.training/<artifact-id>/`` and the
run identity is fingerprinted from the snapshot content, request values
(excluding ``resume``), feature schema, route policy, cost schedules, and the
``SELECTION_POLICY_VERSION``. A fingerprint mismatch raises ``ValueError`` so
stale evidence is never silently reused; a resumed run skips only units whose
identity and content hashes validate and reruns an interrupted in-progress fit.
Artifact publication remains the last, atomic step and the completed artifact
stays the sole terminal truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.core.costs import CostSchedule
from src.stocks.data.contracts import DatasetSnapshot
from src.stocks.workflows.contracts import TrainingRequest
from src.stocks.workflows.economic_selection import SELECTION_POLICY_VERSION

_RUN_DIR_NAME = ".training"
_IDENTITY_FILENAME = "run_identity.json"


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def content_hash(value: object) -> str:
    """Public canonical SHA-256 of any JSON-serializable unit of evidence."""
    return _digest(value)


def _schedule_digest(schedule: CostSchedule) -> str:
    return _digest(
        {
            "name": schedule.name,
            "cost_points": [
                {
                    "effective_from": point.effective_from.isoformat(),
                    "commission_rate": point.commission_rate,
                    "tax_rate": point.tax_rate,
                    "slippage_bps": point.slippage_bps,
                }
                for point in schedule.points
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Fingerprinted identity of one training run's durable state."""

    run_root: Path
    artifact_id: str
    fingerprint: str

    def to_json_safe(self) -> dict[str, str]:
        return {
            "run_root": str(self.run_root),
            "artifact_id": self.artifact_id,
            "fingerprint": self.fingerprint,
            "selection_policy_version": SELECTION_POLICY_VERSION,
        }


class TrainingRunStore:
    """Atomic phase checkpointing and validated resume for a training run."""

    def __init__(
        self,
        identity: RunIdentity,
        *,
        resume: bool = False,
    ) -> None:
        self.identity = identity
        self.resume = resume
        self.root = identity.run_root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def fingerprint(
        cls,
        snapshot: DatasetSnapshot,
        request: TrainingRequest,
        feature_columns: tuple[str, ...],
        route_specs: tuple[Any, ...],
        base_schedule: CostSchedule,
        stress_schedule: CostSchedule,
    ) -> str:
        """Content hash of every input that can change the selected champion."""
        request_values = {
            field.name: getattr(request, field.name)
            for field in fields(TrainingRequest)
            if field.name not in ("resume", "run_root")
        }
        return _digest(
            {
                "snapshot_content_hash": snapshot.manifest.content_hash,
                "snapshot_schema_hash": snapshot.manifest.schema_hash,
                "request": request_values,
                "feature_columns": list(feature_columns),
                "route_specs": [
                    {
                        "horizon": route.horizon,
                        "label_column": route.label_column,
                        "relevance_column": route.relevance_column,
                        "label_available_column": route.label_available_column,
                    }
                    for route in route_specs
                ],
                "base_cost_schedule": _schedule_digest(base_schedule),
                "stress_cost_schedule": _schedule_digest(stress_schedule),
                "selection_policy_version": SELECTION_POLICY_VERSION,
            }
        )

    @classmethod
    def resolve(
        cls,
        snapshot: DatasetSnapshot,
        request: TrainingRequest,
        feature_columns: tuple[str, ...],
        route_specs: tuple[Any, ...],
        base_schedule: CostSchedule,
        stress_schedule: CostSchedule,
        *,
        registry_root: Path | None = None,
    ) -> TrainingRunStore | None:
        """Resolve (or create) the run store for the request.

        Returns ``None`` when the request carries no durable ``run_root``, so
        the legacy in-memory study path is preserved for callers that opt out.
        A mismatched persisted identity raises ``ValueError``.
        """
        if request.run_root is None:
            return None
        run_root = request.run_root
        if registry_root is not None and run_root.resolve() == registry_root.resolve():
            run_root = registry_root / _RUN_DIR_NAME / request.artifact_id
        identity = RunIdentity(
            run_root=run_root,
            artifact_id=request.artifact_id,
            fingerprint=cls.fingerprint(
                snapshot, request, feature_columns, route_specs,
                base_schedule, stress_schedule,
            ),
        )
        store = cls(identity, resume=request.resume)
        identity_path = run_root / _IDENTITY_FILENAME
        if identity_path.exists():
            persisted = json.loads(identity_path.read_text())
            if persisted.get("fingerprint") != identity.fingerprint:
                raise ValueError(
                    "training run identity mismatch: inputs changed since the "
                    "last run; stale evidence must not be silently reused "
                    f"(persisted {persisted.get('fingerprint')} != {identity.fingerprint})"
                )
        else:
            identity_path.write_text(
                json.dumps(identity.to_json_safe(), indent=2, sort_keys=True)
            )
        return store

    def optuna_storage_url(self, route_horizon: int) -> str:
        """Per-route SQLite study storage under the run root."""
        return f"sqlite:///{self.root / f'study_h{route_horizon}.db'!s}"

    def phase_path(self, phase: str) -> Path:
        return self.root / f"phase_{phase}.json"

    def completed_phase(self, phase: str, content_hash: str) -> bool:
        """True only when the phase checkpoint exists with the same content."""
        path = self.phase_path(phase)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return bool(payload.get("content_hash") == content_hash)

    def checkpoint_phase(self, phase: str, evidence: dict[str, object]) -> Path:
        """Atomically write a phase checkpoint (write-then-rename)."""
        payload = {
            "phase": phase,
            "content_hash": _digest(evidence),
            "evidence": evidence,
            "selection_policy_version": SELECTION_POLICY_VERSION,
        }
        path = self.phase_path(phase)
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        tmp.replace(path)
        return path

    def phase_evidence(self, phase: str) -> dict[str, object]:
        """Return the persisted phase evidence, or ``{}`` when absent."""
        path = self.phase_path(phase)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        evidence = payload.get("evidence")
        return evidence if isinstance(evidence, dict) else {}
