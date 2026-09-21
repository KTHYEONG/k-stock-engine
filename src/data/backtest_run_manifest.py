"""Immutable content-addressed backtest run manifest."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePath
from typing import Any, Literal

from src.data.research_scope import ResearchScope
from src.data.runtime import DataRuntime
from src.data.schemas import PITDataError, SilverTable

LEGACY_SCHEMA_VERSION = "backtest-run-v2"
SCOPE_SCHEMA_VERSION: Literal["backtest-run-v3"] = "backtest-run-v3"
_MIN_PERIOD_START = date(2020, 1, 1)

BacktestSegment = Literal["development", "validation", "holdout"]


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
        "schema_version": LEGACY_SCHEMA_VERSION,
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
class LegacyBacktestRunManifest:
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


def build_legacy_backtest_run_manifest(
    *,
    silver_root: Path,
    gold_root: Path,
    silver_dataset_ids: Mapping[SilverTable | str, str],
    gold_dataset_id: str,
    validation_start: date,
    validation_end: date,
    strategy_id: str,
    policy_versions: Mapping[str, str],
) -> LegacyBacktestRunManifest:
    if validation_start > validation_end:
        raise PITDataError("validation_start must be on or before validation_end")
    if strategy_id == "compounding-v2":
        from src.data.research_period import earliest_evaluable_decision_date

        floor = earliest_evaluable_decision_date()
        if validation_start < floor:
            raise PITDataError(
                f"compounding-v2 validation_start {validation_start} precedes fundamentals lookback floor {floor}"
            )
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
    return LegacyBacktestRunManifest(
        schema_version=LEGACY_SCHEMA_VERSION,
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


def _manifest_to_json(manifest: LegacyBacktestRunManifest) -> dict[str, Any]:
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


def write_legacy_backtest_run_manifest(*, manifest: LegacyBacktestRunManifest, artifact_root: Path) -> Path:
    if manifest.schema_version != LEGACY_SCHEMA_VERSION:
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
        existing = load_legacy_backtest_run_manifest(path)
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


def load_legacy_backtest_run_manifest(path: Path) -> LegacyBacktestRunManifest:
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
    if data["schema_version"] != LEGACY_SCHEMA_VERSION:
        if data["schema_version"] == "backtest-run-v1":
            raise PITDataError("invalid backtest run manifest: backtest-run-v1 requires rebuild-required")
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
    rebuilt = build_legacy_backtest_run_manifest(
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


@dataclass(frozen=True, slots=True)
class BacktestRunManifest:
    """Immutable identity for one backtest against one completed Scope segment."""

    schema_version: Literal["backtest-run-v3"]
    content_hash: str
    scope_id: str
    scope_hash: str
    segment: BacktestSegment
    period_start: date
    period_end: date
    silver_dataset_ids: Mapping[str, str]
    gold_dataset_id: str
    strategy_id: str
    strategy_policy_hash: str
    execution_policy_hash: str
    universe_policy_hash: str


def _segment_period(scope: ResearchScope, segment: BacktestSegment) -> tuple[date, date]:
    if segment == "development":
        return (scope.development_start, scope.development_end)
    if segment == "validation":
        return (scope.validation_start, scope.validation_end)
    if segment == "holdout":
        return (scope.holdout_start, scope.holdout_end)
    raise PITDataError(f"unknown backtest segment {segment!r}")


def _check_scope_dataset_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise PITDataError(f"invalid {field}: must be a non-empty single path component")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise PITDataError(f"invalid {field}: must be a non-empty single path component")
    return value


def _resolve_workspace_dir(*, root: Path, relative: PurePath, field: str) -> Path:
    candidate = Path(root) / relative
    if candidate.resolve() != Path(root).resolve() / relative:
        raise PITDataError(f"{field} points outside the workspace")
    return candidate


def _read_scope_stamp(manifest_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _verified_silver_dir(*, runtime: DataRuntime, table: str, dataset_id: str) -> Path:
    dataset_dir = _resolve_workspace_dir(
        root=runtime.workspace.silver_root,
        relative=PurePath(table, dataset_id),
        field=f"silver dataset {table}/{dataset_id}",
    )
    stamp = _read_scope_stamp(dataset_dir / "dataset_manifest.json")
    if stamp is None or stamp.get("scope_hash") != runtime.scope.content_hash:
        raise PITDataError(f"silver dataset {table}/{dataset_id} is outside the workspace or has a foreign Scope hash")
    return dataset_dir


def _verified_gold_release(*, runtime: DataRuntime, dataset_id: str) -> dict[str, Any]:
    release_dir = _resolve_workspace_dir(
        root=runtime.workspace.gold_root,
        relative=PurePath("releases", dataset_id),
        field=f"gold dataset {dataset_id}",
    )
    stamp = _read_scope_stamp(release_dir / "release.json")
    if stamp is None or stamp.get("scope_hash") != runtime.scope.content_hash:
        raise PITDataError(f"gold dataset {dataset_id} is outside the workspace or has a foreign Scope hash")
    decision_raw = stamp.get("decision_time")
    if isinstance(decision_raw, str) and decision_raw.strip():
        try:
            decision_day = date.fromisoformat(decision_raw.strip()[:10])
        except ValueError as exc:
            raise PITDataError(f"gold dataset {dataset_id} has invalid release metadata") from exc
        if decision_day > runtime.scope.completed_end:
            raise PITDataError(f"gold dataset {dataset_id} was resolved outside the completed scope")
    return stamp


def _check_policy_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PITDataError(f"invalid {field}: must be non-empty")
    return value


def _scope_canonical_payload(
    *,
    scope_id: str,
    scope_hash: str,
    segment: BacktestSegment,
    period_start: date,
    period_end: date,
    silver_dataset_ids: dict[str, str],
    gold_dataset_id: str,
    strategy_id: str,
    strategy_policy_hash: str,
    execution_policy_hash: str,
    universe_policy_hash: str,
) -> dict[str, Any]:
    return {
        "execution_policy_hash": execution_policy_hash,
        "gold_dataset_id": gold_dataset_id,
        "period_end": period_end.isoformat(),
        "period_start": period_start.isoformat(),
        "schema_version": SCOPE_SCHEMA_VERSION,
        "scope_hash": scope_hash,
        "scope_id": scope_id,
        "segment": segment,
        "silver_dataset_ids": {key: silver_dataset_ids[key] for key in sorted(silver_dataset_ids)},
        "strategy_id": strategy_id,
        "strategy_policy_hash": strategy_policy_hash,
        "universe_policy_hash": universe_policy_hash,
    }


def build_backtest_run_manifest(
    *,
    runtime: DataRuntime,
    segment: BacktestSegment,
    silver_dataset_ids: Mapping[str, str],
    gold_dataset_id: str,
    strategy_id: str,
    strategy_policy_hash: str,
    execution_policy_hash: str,
    universe_policy_hash: str,
) -> BacktestRunManifest:
    """Bind all data and policies before the runner reads a price or feature row."""
    scope = runtime.scope
    period_start, period_end = _segment_period(scope, segment)
    if period_start < _MIN_PERIOD_START or period_end > scope.completed_end:
        raise PITDataError("backtest period is outside the completed scope range")
    if not isinstance(silver_dataset_ids, Mapping) or not silver_dataset_ids:
        raise PITDataError("silver_dataset_ids must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for name, ident in silver_dataset_ids.items():
        table = str(name) if isinstance(name, str) else ""
        if not table:
            raise PITDataError("silver_dataset_ids table name must be non-empty")
        normalized[table] = _check_scope_dataset_id(ident, field=f"silver_dataset_ids[{table}]")
    if "daily_market" not in normalized:
        raise PITDataError("silver_dataset_ids must bind daily_market bars")
    for table, ident in normalized.items():
        _verified_silver_dir(runtime=runtime, table=table, dataset_id=ident)
    gold_id = _check_scope_dataset_id(gold_dataset_id, field="gold_dataset_id")
    _verified_gold_release(runtime=runtime, dataset_id=gold_id)
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise PITDataError("strategy_id must be non-empty")
    strategy_hash = _check_policy_hash(strategy_policy_hash, field="strategy_policy_hash")
    execution_hash = _check_policy_hash(execution_policy_hash, field="execution_policy_hash")
    universe_hash = _check_policy_hash(universe_policy_hash, field="universe_policy_hash")
    payload = _scope_canonical_payload(
        scope_id=scope.scope_id,
        scope_hash=scope.content_hash,
        segment=segment,
        period_start=period_start,
        period_end=period_end,
        silver_dataset_ids=normalized,
        gold_dataset_id=gold_id,
        strategy_id=strategy_id,
        strategy_policy_hash=strategy_hash,
        execution_policy_hash=execution_hash,
        universe_policy_hash=universe_hash,
    )
    content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = BacktestRunManifest(
        schema_version=SCOPE_SCHEMA_VERSION,
        content_hash=content_hash,
        scope_id=scope.scope_id,
        scope_hash=scope.content_hash,
        segment=segment,
        period_start=period_start,
        period_end=period_end,
        silver_dataset_ids=dict(normalized),
        gold_dataset_id=gold_id,
        strategy_id=strategy_id,
        strategy_policy_hash=strategy_hash,
        execution_policy_hash=execution_hash,
        universe_policy_hash=universe_hash,
    )
    run_dir = runtime.workspace.runs_root / "backtests" / content_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    record = dict(payload)
    record["content_hash"] = content_hash
    record["created_at"] = datetime.now(UTC).isoformat()
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise PITDataError("existing backtest manifest is unreadable")
        prior = dict(existing)
        prior.pop("created_at", None)
        current = dict(record)
        current.pop("created_at", None)
        if prior != current:
            raise PITDataError("existing backtest manifest differs from requested inputs")
        return manifest
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
