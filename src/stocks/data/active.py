"""Snapshotless active research data pipeline."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.stocks.data.catalog import CatalogKind, CatalogStore
from src.stocks.data.contracts import CoverageRange
from src.stocks.data.direct import DirectDataRequest
from src.stocks.ml.contracts import CANONICAL_FEATURE_SET
from src.storage.parquet_datasets import ParquetDatasetStore

logger = logging.getLogger("stocks.data.active")

_ACTIVE_OPS_KINDS = (CatalogKind.BASE_PANEL, CatalogKind.FEATURES, CatalogKind.LABELS, CatalogKind.COSTS)


@dataclass(frozen=True, slots=True)
class ActiveResearchDataRequest:
    start: date
    end: date
    candidate_horizon_sessions: tuple[int, ...]
    feature_set: str = CANONICAL_FEATURE_SET

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("start must not be after end")
        if not self.candidate_horizon_sessions:
            raise ValueError("candidate_horizon_sessions must be non-empty")
        if not self.feature_set:
            raise ValueError("feature_set must be non-empty")


@dataclass(frozen=True, slots=True)
class ActiveResearchDataSelection:
    direct_request: DirectDataRequest
    cost_evidence_path: Path
    data_inputs: Mapping[str, object]


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def resolve_active_research_data(
    *,
    catalog_root: Path,
    base_root: Path,
    feature_root: Path,
    label_root: Path,
    request: ActiveResearchDataRequest,
) -> ActiveResearchDataSelection:
    """Resolve one active selection from policy, validating hashes and coverage."""
    store = CatalogStore(Path(catalog_root))
    policy = store.load_active_policy()
    entries = policy.require_operational_entries(store)
    req_range = CoverageRange(start=request.start, end=request.end)
    for kind in _ACTIVE_OPS_KINDS:
        ent = entries[kind]
        if not ent.content_hash:
            raise ValueError(f"{kind.value}:{ent.name} has empty content_hash")
        if ent.coverage is not None and not ent.coverage.contains(req_range):
            raise ValueError(f"{kind.value}:{ent.name} coverage {ent.coverage.start}..{ent.coverage.end} does not contain requested {request.start}..{request.end}")
    base_entry = entries[CatalogKind.BASE_PANEL]
    feature_entry = entries[CatalogKind.FEATURES]
    label_entry = entries[CatalogKind.LABELS]
    costs_entry = entries[CatalogKind.COSTS]
    costs_path = Path(costs_entry.path) if costs_entry.path else Path("")
    if not costs_path.is_absolute() and costs_entry.path:
        costs_path = Path(costs_entry.path)
    if not costs_path.exists() or not costs_path.is_file():
        raise ValueError(f"costs:{costs_entry.name} cost evidence missing at {costs_path}")
    base_store = ParquetDatasetStore(Path(base_root))
    feature_store = ParquetDatasetStore(Path(feature_root))
    label_store = ParquetDatasetStore(Path(label_root))
    for store_obj, ent, kind_name in [
        (base_store, base_entry, "base_panel"),
        (feature_store, feature_entry, "features"),
        (label_store, label_entry, "labels"),
    ]:
        try:
            manifest = store_obj.read_manifest(ent.name)
        except FileNotFoundError as exc:
            raise ValueError(f"{kind_name}:{ent.name} manifest missing on disk") from exc
        if manifest.content_hash != ent.content_hash:
            raise ValueError(f"{kind_name}:{ent.name} hash mismatch catalog {ent.content_hash} vs manifest {manifest.content_hash}")
        if manifest.schema_hash != ent.schema_hash and ent.schema_hash:
            raise ValueError(f"{kind_name}:{ent.name} schema hash mismatch")
    feature_manifest = feature_store.read_manifest(feature_entry.name)
    label_manifest = label_store.read_manifest(label_entry.name)
    base_manifest = base_store.read_manifest(base_entry.name)
    cost_hash = _hash_file(costs_path)
    if not cost_hash:
        raise ValueError(f"costs:{costs_entry.name} cost evidence hash unavailable")
    if cost_hash != costs_entry.content_hash:
        raise ValueError(
            f"costs:{costs_entry.name} hash mismatch catalog {costs_entry.content_hash} vs file {cost_hash}"
        )
    direct_request = DirectDataRequest(
        base_dataset_id=base_entry.name,
        feature_dataset_id=feature_entry.name,
        label_dataset_id=label_entry.name,
        start=request.start,
        end=request.end,
        feature_set=request.feature_set,
        candidate_horizon_sessions=request.candidate_horizon_sessions,
    )
    policy_version = 1
    try:
        import json
        payload = json.loads((Path(catalog_root) / "active_datasets.json").read_text(encoding="utf-8"))
        policy_version = int(payload.get("active_datasets_version", 1))
    except Exception:
        policy_version = 1
    data_inputs: dict[str, object] = {
        "base_dataset_id": base_entry.name,
        "base_content_hash": base_entry.content_hash,
        "base_schema_hash": base_manifest.schema_hash,
        "feature_dataset_id": feature_entry.name,
        "feature_content_hash": feature_entry.content_hash,
        "feature_schema_hash": feature_manifest.schema_hash,
        "label_dataset_id": label_entry.name,
        "label_content_hash": label_entry.content_hash,
        "label_schema_hash": label_manifest.schema_hash,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "active_policy_version": policy_version,
        "cost_evidence_path": str(costs_path),
        "cost_evidence_hash": cost_hash,
        "candidate_horizon_sessions": list(request.candidate_horizon_sessions),
        "feature_set": request.feature_set,
    }
    logger.debug(
        "active selection resolved base=%s features=%s labels=%s costs=%s range=%s..%s",
        base_entry.name, feature_entry.name, label_entry.name, costs_entry.name, request.start, request.end,
    )
    return ActiveResearchDataSelection(
        direct_request=direct_request,
        cost_evidence_path=costs_path,
        data_inputs=data_inputs,
    )
