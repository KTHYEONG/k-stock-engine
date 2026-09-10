"""Immutable content-addressed backtest run manifest."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.data.schemas import PITDataError, SilverTable

SCHEMA_VERSION = "backtest-run-v1"


def _check_dataset_id(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise PITDataError(f"invalid {field}: must be a string")
    if not value.strip():
        raise PITDataError(f"invalid {field}: must be non-empty")
    if value.strip() != value or not value.strip():
        raise PITDataError(f"invalid {field}: must be non-empty")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise PITDataError(f"invalid {field}: must be a single path component")
    return value


def _normalize_silver_ids(
    silver_dataset_ids: Mapping[SilverTable | str, str],
) -> dict[str, str]:
    if not isinstance(silver_dataset_ids, Mapping):
        raise PITDataError("silver_dataset_ids must contain exactly every SilverTable value once")
    normalized: dict[str, str] = {}
    for key, value in silver_dataset_ids.items():
        if isinstance(key, SilverTable):
            name = key.value
        elif isinstance(key, str):
            name = key
        else:
            raise PITDataError("silver_dataset_ids must contain exactly every SilverTable value once")
        try:
            table = SilverTable(name)
        except ValueError as exc:
            raise PITDataError(f"silver_dataset_ids must contain exactly every SilverTable value once: unknown {name!r}") from exc
        normalized[table.value] = _check_dataset_id(value, field=f"silver_dataset_ids[{table.value}]")
    expected = sorted(t.value for t in SilverTable)
    if sorted(normalized) != expected:
        raise PITDataError("silver_dataset_ids must contain exactly every SilverTable value once")
    return normalized


def _canonical_payload(
    *,
    silver_root: str,
    gold_root: str,
    silver_dataset_ids: dict[str, str],
    gold_dataset_id: str,
    validation_start: date,
    validation_end: date,
    strategy_id: str,
    policy_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "gold_dataset_id": gold_dataset_id,
        "gold_root": gold_root,
        "policy_versions": {k: policy_versions[k] for k in sorted(policy_versions)},
        "schema_version": SCHEMA_VERSION,
        "silver_dataset_ids": {k: silver_dataset_ids[k] for k in sorted(silver_dataset_ids)},
        "silver_root": silver_root,
        "strategy_id": strategy_id,
        "validation_end": validation_end.isoformat(),
        "validation_start": validation_start.isoformat(),
    }


def _content_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class BacktestRunManifest:
    schema_version: str
    content_hash: str
    silver_root: str
    gold_root: str
    silver_dataset_ids: Mapping[str, str]
    gold_dataset_id: str
    validation_start: date
    validation_end: date
    strategy_id: str
    policy_versions: Mapping[str, str]


def build_backtest_run_manifest(
    *,
    silver_root: Path,
    gold_root: Path,
    silver_dataset_ids: Mapping[SilverTable | str, str],
    gold_dataset_id: str,
    validation_start: date,
    validation_end: date,
    strategy_id: str,
    policy_versions: Mapping[str, str],
) -> BacktestRunManifest:
    if validation_start > validation_end:
        raise PITDataError("validation_start must be on or before validation_end")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise PITDataError("strategy_id must be non-empty")
    if not isinstance(policy_versions, Mapping) or not policy_versions:
        raise PITDataError("policy_versions must be non-empty")
    policies: dict[str, str] = {}
    for key, value in policy_versions.items():
        if not isinstance(key, str) or not key.strip():
            raise PITDataError("policy_versions key must be non-empty")
        if not isinstance(value, str) or not value.strip():
            raise PITDataError("policy_versions value must be non-empty")
        policies[key] = value
    normalized_ids = _normalize_silver_ids(silver_dataset_ids)
    gid = _check_dataset_id(gold_dataset_id, field="gold_dataset_id")
    payload = _canonical_payload(
        silver_root=str(silver_root),
        gold_root=str(gold_root),
        silver_dataset_ids=normalized_ids,
        gold_dataset_id=gid,
        validation_start=validation_start,
        validation_end=validation_end,
        strategy_id=strategy_id,
        policy_versions=policies,
    )
    digest = _content_hash(payload)
    return BacktestRunManifest(
        schema_version=SCHEMA_VERSION,
        content_hash=digest,
        silver_root=str(silver_root),
        gold_root=str(gold_root),
        silver_dataset_ids=dict(normalized_ids),
        gold_dataset_id=gid,
        validation_start=validation_start,
        validation_end=validation_end,
        strategy_id=strategy_id,
        policy_versions=dict(policies),
    )


def _manifest_to_json(manifest: BacktestRunManifest) -> dict[str, Any]:
    payload = _canonical_payload(
        silver_root=manifest.silver_root,
        gold_root=manifest.gold_root,
        silver_dataset_ids=dict(manifest.silver_dataset_ids),
        gold_dataset_id=manifest.gold_dataset_id,
        validation_start=manifest.validation_start,
        validation_end=manifest.validation_end,
        strategy_id=manifest.strategy_id,
        policy_versions=dict(manifest.policy_versions),
    )
    payload["content_hash"] = manifest.content_hash
    return payload


def write_backtest_run_manifest(*, manifest: BacktestRunManifest, artifact_root: Path) -> Path:
    if manifest.schema_version != SCHEMA_VERSION:
        raise PITDataError("invalid backtest run manifest schema_version")
    expected = _content_hash(
        _canonical_payload(
            silver_root=manifest.silver_root,
            gold_root=manifest.gold_root,
            silver_dataset_ids=dict(manifest.silver_dataset_ids),
            gold_dataset_id=manifest.gold_dataset_id,
            validation_start=manifest.validation_start,
            validation_end=manifest.validation_end,
            strategy_id=manifest.strategy_id,
            policy_versions=dict(manifest.policy_versions),
        )
    )
    if expected != manifest.content_hash:
        raise PITDataError("invalid backtest run manifest content_hash")
    runs_dir = Path(artifact_root) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{manifest.content_hash}.json"
    if path.exists():
        existing = load_backtest_run_manifest(path)
        if existing != manifest:
            raise PITDataError("existing backtest run manifest differs from requested manifest")
        return path
    staging = runs_dir / f".{manifest.content_hash}.{uuid.uuid4().hex}.tmp"
    staging.write_text(
        json.dumps(_manifest_to_json(manifest), sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(staging, path)
    return path


def load_backtest_run_manifest(path: Path) -> BacktestRunManifest:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PITDataError(f"missing backtest run manifest: {path}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"malformed backtest run manifest: {path}") from exc
    if not isinstance(data, dict):
        raise PITDataError("malformed backtest run manifest: top-level object required")
    expected_fields = {
        "schema_version",
        "content_hash",
        "silver_root",
        "gold_root",
        "silver_dataset_ids",
        "gold_dataset_id",
        "validation_start",
        "validation_end",
        "strategy_id",
        "policy_versions",
    }
    if set(data) != expected_fields:
        raise PITDataError("invalid backtest run manifest: unknown/missing fields")
    if data["schema_version"] != SCHEMA_VERSION:
        raise PITDataError("invalid backtest run manifest: wrong schema_version")
    for field in (
        "content_hash",
        "silver_root",
        "gold_root",
        "gold_dataset_id",
        "validation_start",
        "validation_end",
        "strategy_id",
    ):
        if not isinstance(data[field], str):
            raise PITDataError("invalid backtest run manifest: wrong scalar types")
    for field in ("silver_dataset_ids", "policy_versions"):
        if not isinstance(data[field], dict):
            raise PITDataError("invalid backtest run manifest: wrong scalar types")
    try:
        validation_start = date.fromisoformat(data["validation_start"])
        validation_end = date.fromisoformat(data["validation_end"])
    except ValueError as exc:
        raise PITDataError("invalid backtest run manifest: wrong scalar types") from exc
    rebuilt = build_backtest_run_manifest(
        silver_root=Path(data["silver_root"]),
        gold_root=Path(data["gold_root"]),
        silver_dataset_ids=dict(data["silver_dataset_ids"]),
        gold_dataset_id=data["gold_dataset_id"],
        validation_start=validation_start,
        validation_end=validation_end,
        strategy_id=data["strategy_id"],
        policy_versions=dict(data["policy_versions"]),
    )
    if rebuilt.content_hash != data["content_hash"]:
        raise PITDataError("invalid backtest run manifest: content_hash mismatch")
    return rebuilt
