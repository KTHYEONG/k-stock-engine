#!/usr/bin/env python3
"""One-shot migration of scoped Silver/Gold datasets to the v2 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl

from src.core.krx_calendar import xkrx_session_calendar
from src.core.market_rules import load_krx_market_rules
from src.core.time import KRX_TZ
from src.data.bronze_aggregation import discover_verified_bronze_receipts
from src.data.dataset_registry import REGISTRY_NAME, DatasetRegistry
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_digest,
    load_manifest,
    publish_dataset,
    verify_dataset,
)
from src.data.market_panel import MarketPanelPolicy, _rules_fingerprint, materialize_market_panel
from src.data.reference_benchmarks import (
    BenchmarkDefinition,
    Weighting,
    load_benchmark_definitions,
    materialize_reference_benchmarks,
)
from src.data.runtime import DataRuntime, load_data_runtime
from src.data.schemas import EvidenceKind, PITDataError
from src.data.workspace import ResearchWorkspace

_MIGRATION_ORDER = (
    "ordinary_universe",
    "daily_market",
    "investor_flow_ls",
    "industry",
    "dividend_events",
    "market_panel",
    "investor_flow_kis_supplement",
    "investor_flow",
    "reference_benchmarks",
)


@dataclass(frozen=True, slots=True)
class LegacyDataset:
    layer: DatasetLayer
    directory: Path
    dataset_id: str
    manifest: Mapping[str, Any]
    frames: Mapping[str, pl.DataFrame]
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationResult:
    kind: str
    old_id: str
    new_id: str
    path: Path
    rows: int


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rules", type=Path, default=Path("config/market/krx_market_rules.toml"))
    parser.add_argument(
        "--definitions", type=Path, default=Path("config/data/reference_benchmarks.toml")
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PITDataError(f"unreadable migration manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise PITDataError(f"invalid migration manifest: {path}")
    return raw


def _legacy_kind(dataset_id: str, manifest: Mapping[str, Any]) -> str:
    if dataset_id.startswith("ordinary_universe_"):
        return "ordinary_universe"
    if dataset_id.startswith("daily_market_"):
        return "daily_market"
    if dataset_id.startswith("investor_flow_kis_supplement_"):
        return "investor_flow_kis_supplement"
    if dataset_id.startswith("investor_flow_"):
        return "investor_flow_ls" if "universe_dataset_id" in manifest else "investor_flow"
    if dataset_id.startswith("industry_"):
        return "industry"
    if dataset_id.startswith("dividend_events_"):
        return "dividend_events"
    if dataset_id.startswith("market_panel_"):
        return "market_panel"
    if dataset_id.startswith("reference_benchmarks_"):
        return "reference_benchmarks"
    if dataset_id.startswith("financial_facts") or dataset_id.startswith("financial_quality"):
        return dataset_id.rsplit("_", 1)[0]
    raise PITDataError(f"unknown legacy dataset kind: {dataset_id}")


def _partition_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = manifest.get("partitions", [])
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise PITDataError("legacy manifest has no partition list")
    result: list[Mapping[str, Any]] = [entry for entry in entries if isinstance(entry, dict)]
    for key in ("exits", "table"):
        special = manifest.get(key)
        if isinstance(special, dict) and isinstance(special.get("path"), str):
            result.append(special)
    if not result:
        raise PITDataError("legacy manifest has no readable partitions")
    return result


def _read_legacy_frames(directory: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, pl.DataFrame], tuple[str, ...]]:
    frames: dict[str, pl.DataFrame] = {}
    keys: list[str] = []
    for entry in _partition_entries(manifest):
        relative = entry.get("path")
        digest = entry.get("parquet_sha256", entry.get("sha256"))
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PITDataError(f"legacy dataset partition metadata is invalid: {directory}")
        posix = PurePosixPath(relative)
        if (
            "\\" in relative
            or posix.is_absolute()
            or posix.as_posix() != relative
            or any(part in ("", ".", "..") for part in posix.parts)
        ):
            raise PITDataError(f"legacy dataset partition path is invalid: {relative}")
        if relative in frames:
            continue
        path = directory / relative
        try:
            frame = pl.read_parquet(path)
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise PITDataError(f"legacy dataset partition is unreadable: {path}") from exc
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PITDataError(f"legacy dataset partition is unreadable: {path}") from exc
        if actual != digest:
            raise PITDataError(f"legacy dataset partition hash mismatch: {path}")
        frames[relative] = frame
        if "session" in frame.columns:
            keys.extend(str(value)[:10] for value in frame["session"].unique().to_list())
    if not frames:
        raise PITDataError(f"legacy dataset has no readable partitions: {directory}")
    return frames, tuple(sorted(set(keys)))


def _discover(*, silver_root: Path, gold_root: Path) -> tuple[LegacyDataset, ...]:
    """Freeze direct and one-level legacy dataset directories before any action."""

    found: list[LegacyDataset] = []

    def add_direct(layer: DatasetLayer, directory: Path) -> None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _read_json(manifest_path)
        if manifest.get("schema") == "dataset-manifest-v2":
            return
        dataset_id = str(manifest.get("dataset_id") or directory.name)
        if dataset_id != directory.name:
            raise PITDataError(f"legacy dataset id does not match directory: {directory}")
        frames, keys = _read_legacy_frames(directory, manifest)
        found.append(LegacyDataset(layer, directory, dataset_id, manifest, frames, keys))

    for layer, root in ((DatasetLayer.SILVER, silver_root), (DatasetLayer.GOLD, gold_root)):
        if not root.is_dir():
            continue
        directories = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
        )
        for directory in directories:
            add_direct(layer, directory)
            for generation in sorted(
                path for path in directory.iterdir()
                if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
            ):
                generation_manifest = generation / "dataset_manifest.json"
                generation_content = generation / "content_manifest.json"
                if not generation_manifest.is_file() or not generation_content.is_file():
                    continue
                merged = dict(_read_json(generation_content))
                merged.update(_read_json(generation_manifest))
                merged["dataset_id"] = f"{directory.name}_{generation.name[:16]}"
                frames, keys = _read_legacy_frames(generation, merged)
                found.append(
                    LegacyDataset(layer, generation, str(merged["dataset_id"]), merged, frames, keys)
                )
    return tuple(found)


def _old_id_value(manifest: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = manifest.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _frame_digests(frames: Mapping[str, pl.DataFrame]) -> list[str]:
    values: list[str] = []
    for frame in frames.values():
        for column in ("source_hash", "ksic_source_hash"):
            if column in frame.columns:
                values.extend(str(value) for value in frame[column].drop_nulls().unique().to_list())
    if values:
        return values
    return [hashlib.sha256(frame.to_pandas().to_csv(index=False).encode("utf-8")).hexdigest() for frame in frames.values()]


def _manifest_hashes(manifest: Mapping[str, Any], name: str) -> list[str]:
    value = manifest.get(name)
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _investor_page_hashes(bronze_root: Path, *, provider: str) -> list[str]:
    root = Path(bronze_root) / "investor_flow"
    if not root.is_dir():
        return []
    hashes: list[str] = []
    for path in sorted(root.glob("*/payload.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PITDataError(f"legacy investor-flow source is unreadable: {path}") from exc
        if not isinstance(payload, dict):
            raise PITDataError(f"legacy investor-flow source is invalid: {path}")
        if provider == "LS":
            if str(payload.get("provider") or "").strip().upper() != "LS":
                continue
            rows = payload.get("rows")
            status = str(payload.get("status") or "").strip()
            if (isinstance(rows, list) and rows) or status in {"provider_error", "missing_sessions"}:
                hashes.append(path.parent.name)
        elif provider == "KIS" and payload.get("provider") == "KIS" and isinstance(payload.get("rows"), list) and payload["rows"]:
            hashes.append(path.parent.name)
    return hashes


def _industry_hashes(bronze_root: Path, frames: Mapping[str, pl.DataFrame]) -> list[str]:
    try:
        grouped = discover_verified_bronze_receipts(
            bronze_root=Path(bronze_root), kinds=frozenset({EvidenceKind.INDUSTRY})
        )
    except PITDataError:
        grouped = {}
    receipts = grouped.get(EvidenceKind.INDUSTRY, ())
    if receipts:
        return [receipt.content_hash for receipt in receipts]
    values: list[str] = []
    for frame in frames.values():
        for column in ("source_hash", "ksic_source_hash"):
            if column in frame.columns:
                values.extend(str(value) for value in frame[column].drop_nulls().unique().to_list())
    return values


def _dividend_hashes(bronze_root: Path, manifest: Mapping[str, Any], frames: Mapping[str, pl.DataFrame]) -> list[str]:
    try:
        from src.data.dividend_events import _iter_decision_envelopes

        envelopes = _iter_decision_envelopes(Path(bronze_root))
    except PITDataError:
        envelopes = []
    if envelopes:
        return [
            hashlib.sha256(
                json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()
            for envelope in envelopes
        ]
    explicit = _manifest_hashes(manifest, "decision_hashes")
    if explicit:
        return explicit
    return _frame_digests(frames)


def _corp_bridge_hash(bronze_root: Path, manifest: Mapping[str, Any]) -> str:
    explicit = manifest.get("corp_code_bridge")
    if isinstance(explicit, str) and explicit:
        return explicit
    paths = sorted((Path(bronze_root) / "dart_corp_codes").glob("*/payload.json"))
    if not paths:
        raise PITDataError("corp-code bridge is missing; migration cannot certify dividend lineage")
    return paths[-1].parent.name


def _calendar_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _param_value(manifest: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = manifest.get(name, default)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _identity_for(
    dataset: LegacyDataset,
    *,
    bronze_root: Path,
    mapped: Mapping[str, str],
    rules: Any,
    source_hashes: Mapping[str, Sequence[str]] | None = None,
    calendar_keys: Sequence[str] = (),
    dividend_calendar_keys: Sequence[str] = (),
    reference_definitions: Sequence[BenchmarkDefinition] | None = None,
) -> DatasetIdentity:
    manifest = dataset.manifest
    kind = _legacy_kind(dataset.dataset_id, manifest)
    sources = source_hashes or {}
    inputs: dict[str, str]
    params: dict[str, str | int | float | bool | None] = {}

    def reference(old_id: str, *, role_kind: str) -> str:
        value = mapped.get(old_id, old_id)
        from src.data.datasets import dataset_reference

        return dataset_reference(value, kind=role_kind)

    if kind == "ordinary_universe":
        hashes = list(sources.get("ordinary_universe") or _manifest_hashes(manifest, "source_hashes"))
        if not hashes:
            hashes = _frame_digests(dataset.frames)
        inputs = {"bronze_master": dataset_digest(hashes)}
        params["calendar_digest"] = _calendar_digest(calendar_keys or dataset.keys)
    elif kind == "daily_market":
        universe = _old_id_value(manifest, "universe_dataset_id") or ""
        hashes = list(sources.get("daily_market") or _manifest_hashes(manifest, "source_hashes"))
        if not hashes:
            hashes = _frame_digests(dataset.frames)
        inputs = {
            "universe": reference(universe, role_kind="ordinary_universe"),
            "bronze_daily": dataset_digest(hashes),
        }
        params.update(
            available_time=_param_value(manifest, "available_time", "18:00:00"),
            fluc_tolerance_pct=_param_value(manifest, "fluc_tolerance_pct", 0.01),
        )
    elif kind == "investor_flow_ls":
        universe = _old_id_value(manifest, "universe_dataset_id") or ""
        hashes = list(sources.get("investor_flow_ls") or _manifest_hashes(manifest, "raw_page_hashes"))
        if not hashes:
            hashes = _frame_digests(dataset.frames)
        inputs = {
            "universe": reference(universe, role_kind="ordinary_universe"),
            "bronze_flow": dataset_digest(hashes),
        }
        params.update(
            available_session_lag=_param_value(manifest, "available_session_lag", 1),
            available_time=_param_value(manifest, "available_time", "08:00:00"),
        )
    elif kind == "investor_flow_kis_supplement":
        ls = _old_id_value(manifest, "ls_dataset_id") or ""
        panel = _old_id_value(manifest, "market_panel_dataset_id") or ""
        hashes = list(sources.get("investor_flow_kis_supplement") or _manifest_hashes(manifest, "kis_page_hashes"))
        if not hashes:
            hashes = _frame_digests(dataset.frames)
        inputs = {
            "ls": reference(ls, role_kind="investor_flow_ls"),
            "market_panel": reference(panel, role_kind="market_panel"),
            "bronze_kis": dataset_digest(hashes),
        }
        params.update(
            available_session_lag=_param_value(manifest, "available_session_lag", 1),
            available_time=_param_value(manifest, "available_time", "08:00:00"),
        )
    elif kind == "investor_flow":
        ls = _old_id_value(manifest, "ls_dataset_id") or ""
        kis = _old_id_value(manifest, "kis_supplement_dataset_id") or ""
        inputs = {
            "ls": reference(ls, role_kind="investor_flow_ls"),
            "kis_supplement": reference(kis, role_kind="investor_flow_kis_supplement"),
        }
    elif kind == "industry":
        hashes = list(sources.get("industry") or _industry_hashes(bronze_root, dataset.frames))
        if not hashes:
            hashes = _frame_digests(dataset.frames)
        inputs = {"bronze_classification": dataset_digest(hashes)}
        raw_symbols = manifest.get("symbols")
        if isinstance(raw_symbols, list):
            params["symbols"] = ",".join(sorted(str(symbol) for symbol in raw_symbols))
        else:
            params["symbols"] = None
    elif kind == "dividend_events":
        hashes = list(sources.get("dividend_events") or _dividend_hashes(bronze_root, manifest, dataset.frames))
        inputs = {
            "bronze_dividend_decisions": dataset_digest(hashes),
            "corp_code_bridge": dataset_digest([_corp_bridge_hash(bronze_root, manifest)]),
        }
        params["calendar_digest"] = _calendar_digest(
            dividend_calendar_keys or calendar_keys or dataset.keys
        )
    elif kind == "market_panel":
        daily = _old_id_value(manifest, "daily_market_dataset_id") or ""
        universe = _old_id_value(manifest, "universe_dataset_id") or ""
        inputs = {
            "daily_market": reference(daily, role_kind="daily_market"),
            "universe": reference(universe, role_kind="ordinary_universe"),
        }
        params.update(
            adtv_short_sessions=_param_value(manifest, "adtv_short_sessions", 20),
            adtv_long_sessions=_param_value(manifest, "adtv_long_sessions", 60),
            return_vol_sessions=_param_value(manifest, "return_vol_sessions", 60),
            rules_version=rules.version,
            rules_fingerprint=_rules_fingerprint(rules),
        )
    elif kind == "reference_benchmarks":
        panel = _old_id_value(manifest, "market_panel_dataset_id") or ""
        inputs = {"market_panel": reference(panel, role_kind="market_panel")}
        if reference_definitions is None:
            definitions = manifest.get("definitions", [])
            if isinstance(definitions, list):
                definitions = sorted(
                    (item for item in definitions if isinstance(item, dict)),
                    key=lambda item: str(item.get("benchmark_id", "")),
                )
        else:
            definitions = [
                {
                    "benchmark_id": definition.benchmark_id,
                    "weighting": definition.weighting.value,
                    "min_adtv20_krw": definition.min_adtv20_krw,
                }
                for definition in sorted(reference_definitions, key=lambda item: item.benchmark_id)
            ]
        params.update(
            definitions_version=_param_value(manifest, "definitions_version", "reference-benchmarks-v1"),
            definitions=json.dumps(definitions, sort_keys=True, separators=(",", ":")),
        )
    else:
        raise PITDataError(f"legacy dataset kind is not migratable: {kind}")
    policy_version = str(manifest.get("policy_version") or _policy_version(kind))
    if kind == "reference_benchmarks":
        policy_version = str(params["definitions_version"])
    return DatasetIdentity(
        kind=kind,
        layer=dataset.layer,
        policy_version=policy_version,
        inputs=inputs,
        params=params,
    )


def _policy_version(kind: str) -> str:
    return {
        "ordinary_universe": "krx-ordinary-equity-v1",
        "daily_market": "krx-daily-market-v1",
        "investor_flow_ls": "ls-t1702-net-shares-v1",
        "investor_flow_kis_supplement": "kis-investor-trade-net-shares-supplement-v1",
        "investor_flow": "investor-flow-ls-kis-union-v1",
        "industry": "kis-industry-classification-v2",
        "dividend_events": "dividend-events-v2",
        "market_panel": "krx-market-panel-v2",
        "reference_benchmarks": "reference-benchmarks-v1",
    }[kind]


def _row_key(kind: str, frame: pl.DataFrame) -> list[str]:
    candidates = {
        "ordinary_universe": ["session", "instrument_id"],
        "daily_market": ["session", "instrument_id"],
        "investor_flow_ls": ["session", "ticker"],
        "investor_flow_kis_supplement": ["session", "ticker"],
        "investor_flow": ["session", "ticker"],
        "industry": ["ticker"],
        "dividend_events": ["ticker", "record_date"],
        "market_panel": ["session", "instrument_id"],
        "reference_benchmarks": ["benchmark_id", "session"],
    }[kind]
    return [column for column in candidates if column in frame.columns]


def _assert_rows_equal(old: Mapping[str, pl.DataFrame], new_path: Path, kind: str) -> None:
    _assert_concatenated_rows_equal(old, new_path, kind)


def _definitions_from_manifest(manifest: Mapping[str, Any]) -> tuple[BenchmarkDefinition, ...]:
    raw_definitions = manifest.get("definitions")
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise PITDataError("legacy reference benchmark manifest has no definitions")
    definitions: list[BenchmarkDefinition] = []
    seen: set[str] = set()
    for raw in raw_definitions:
        if not isinstance(raw, dict):
            raise PITDataError("legacy reference benchmark definition is invalid")
        benchmark_id = raw.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not benchmark_id or benchmark_id in seen:
            raise PITDataError("legacy reference benchmark definition has an invalid or duplicate id")
        seen.add(benchmark_id)
        try:
            weighting = Weighting(str(raw.get("weighting")))
        except ValueError as exc:
            raise PITDataError(f"legacy reference benchmark weighting is invalid: {benchmark_id}") from exc
        threshold = raw.get("min_adtv20_krw")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
                raise PITDataError(f"legacy reference benchmark threshold is invalid: {benchmark_id}")
            threshold = float(threshold)
        definitions.append(
            BenchmarkDefinition(
                benchmark_id=benchmark_id,
                weighting=weighting,
                min_adtv20_krw=threshold,
            )
        )
    return tuple(sorted(definitions, key=lambda item: item.benchmark_id))


def _definition_payload(definitions: Sequence[BenchmarkDefinition]) -> list[dict[str, object]]:
    return [
        {
            "benchmark_id": definition.benchmark_id,
            "weighting": definition.weighting.value,
            "min_adtv20_krw": definition.min_adtv20_krw,
        }
        for definition in sorted(definitions, key=lambda item: item.benchmark_id)
    ]


def _resolve_reference_definitions(
    manifest: Mapping[str, Any], definitions_path: Path
) -> tuple[str, tuple[BenchmarkDefinition, ...]]:
    """Use the configured definitions only when they certify the legacy output."""

    legacy_version = str(manifest.get("definitions_version", "reference-benchmarks-v1"))
    legacy_definitions = _definitions_from_manifest(manifest)
    try:
        configured_version, configured = load_benchmark_definitions(definitions_path)
    except (OSError, ValueError, PITDataError):
        if not definitions_path.exists():
            return legacy_version, legacy_definitions
        raise
    if configured_version != legacy_version or _definition_payload(configured) != _definition_payload(legacy_definitions):
        raise PITDataError("reference benchmark definitions do not match the legacy dataset")
    return configured_version, configured


def _migrate_one(
    dataset: LegacyDataset,
    *,
    bronze_root: Path,
    target_root: Path,
    mapped: dict[str, str],
    rules: Any,
    source_hashes: Mapping[str, Sequence[str]] | None = None,
    calendar_keys: Sequence[str] = (),
    dividend_calendar_keys: Sequence[str] = (),
    reference_definitions: Sequence[BenchmarkDefinition] | None = None,
) -> MigrationResult:
    identity = _identity_for(
        dataset,
        bronze_root=bronze_root,
        mapped=mapped,
        rules=rules,
        source_hashes=source_hashes,
        calendar_keys=calendar_keys,
        dividend_calendar_keys=dividend_calendar_keys,
        reference_definitions=reference_definitions,
    )
    details = {
        key: value
        for key, value in dataset.manifest.items()
        if key not in {"dataset_id", "partitions", "exits", "table"}
    }
    details["legacy_dataset_id"] = dataset.dataset_id
    published = publish_dataset(
        layer_root=target_root,
        identity=identity,
        partitions=dataset.frames,
        details=details,
    )
    mapped[dataset.dataset_id] = published.dataset_id
    _assert_rows_equal(dataset.frames, published.path, identity.kind)
    return MigrationResult(identity.kind, dataset.dataset_id, published.dataset_id, published.path, published.rows)


def _rebuild_gold(
    *,
    runtime: Any,
    migrated: Mapping[str, MigrationResult],
    old_by_kind: Mapping[str, LegacyDataset],
    rules_path: Path,
    definitions_path: Path,
) -> None:
    panel_old = old_by_kind.get("market_panel")
    if panel_old is not None:
        required = {"daily_market", "ordinary_universe", "market_panel"}
        missing = required - set(migrated)
        if missing:
            raise PITDataError(f"market panel rebuild lacks migrated inputs: {sorted(missing)}")
        panel_new = migrated["market_panel"]
        panel_policy = MarketPanelPolicy(
            adtv_short_sessions=int(panel_old.manifest.get("adtv_short_sessions", 20)),
            adtv_long_sessions=int(panel_old.manifest.get("adtv_long_sessions", 60)),
            return_vol_sessions=int(panel_old.manifest.get("return_vol_sessions", 60)),
        )
        with tempfile.TemporaryDirectory(prefix="market-panel-rebuild-") as temporary_gold:
            rebuilt_panel = materialize_market_panel(
                daily_market_path=runtime.workspace.silver_root / migrated["daily_market"].new_id,
                universe_path=runtime.workspace.silver_root / migrated["ordinary_universe"].new_id,
                rules=load_krx_market_rules(rules_path),
                gold_root=Path(temporary_gold),
                policy=panel_policy,
            )
        if rebuilt_panel.dataset_id != panel_new.new_id:
            raise PITDataError("market panel rebuild did not reproduce migrated id")

    benchmark_old = old_by_kind.get("reference_benchmarks")
    if benchmark_old is not None:
        required = {"market_panel", "reference_benchmarks"}
        missing = required - set(migrated)
        if missing:
            raise PITDataError(f"reference benchmark rebuild lacks migrated inputs: {sorted(missing)}")
        version, definitions = _resolve_reference_definitions(benchmark_old.manifest, definitions_path)
        benchmark_new = migrated["reference_benchmarks"]
        with tempfile.TemporaryDirectory(prefix="reference-benchmark-rebuild-") as temporary_gold:
            rebuilt_benchmark = materialize_reference_benchmarks(
                market_panel_path=runtime.workspace.gold_root / migrated["market_panel"].new_id,
                definitions=definitions,
                definitions_version=version,
                gold_root=Path(temporary_gold),
            )
        if rebuilt_benchmark.dataset_id != benchmark_new.new_id:
            raise PITDataError("reference benchmark rebuild did not reproduce migrated id")


def _verify_registered_scope(runtime: Any, registry: DatasetRegistry) -> None:
    """Run the same intrinsic and lineage verification as ``verify-datasets``."""

    current = registry.snapshot()
    if not current:
        raise PITDataError("migration produced no registered datasets")
    directories: dict[str, list[Path]] = {}
    for root in (runtime.workspace.silver_root, runtime.workspace.gold_root):
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir() or path.is_symlink() or not (path / "manifest.json").is_file():
                continue
            directories.setdefault(path.name, []).append(path)
            if path.name.startswith("."):
                continue
            for nested in path.iterdir():
                if nested.is_dir() and not nested.is_symlink() and (nested / "manifest.json").is_file():
                    directories.setdefault(nested.name, []).append(nested)
    known_ids = set(directories) | set(registry.retired())
    for kind, dataset_id in sorted(current.items()):
        matches = directories.get(dataset_id, [])
        if len(matches) != 1:
            raise PITDataError(f"registered dataset is missing or ambiguous: {dataset_id}")
        verification = verify_dataset(matches[0], known_ids=known_ids.__contains__)
        if not verification.passed:
            raise PITDataError(f"registered dataset failed verification: {dataset_id}: {verification.failures}")
        manifest = load_manifest(matches[0])
        if manifest.kind != kind:
            raise PITDataError(f"registered dataset kind mismatch: {dataset_id}")


def _concatenated_frames(frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    return pl.concat(list(frames.values()), how="vertical_relaxed")


def _assert_concatenated_rows_equal(
    old: Mapping[str, pl.DataFrame], new_path: Path, kind: str
) -> None:
    """Compare all rows after the declared key sort, independent of partitioning."""

    new_frames: dict[str, pl.DataFrame] = {}
    for partition in load_manifest(new_path).partitions:
        path = new_path / partition.path
        try:
            new_frames[partition.path] = pl.read_parquet(path)
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise PITDataError(f"migrated partition is unreadable: {kind}:{partition.path}") from exc
    left = _concatenated_frames(old)
    right = _concatenated_frames(new_frames)
    if list(left.columns) != list(right.columns):
        raise PITDataError(f"migrated schema changed: {kind}")
    keys = {
        "financial_facts": ["company_id", "fiscal_period", "filing_id", "fact", "restatement_id", "consolidated"],
        "financial_quality": ["company_id", "fiscal_period", "accounting_basis", "available_at"],
    }.get(kind) or _row_key(kind, left)
    if not keys or any(key not in left.columns for key in keys):
        raise PITDataError(f"migrated row key is unavailable: {kind}")
    left = left.sort(keys)
    right = right.sort(keys)
    if not left.equals(right):
        raise PITDataError(f"row content changed during migration: {kind}")


def _legacy_decision_time(dataset: LegacyDataset) -> datetime:
    raw = dataset.manifest.get("time_end")
    if isinstance(raw, str):
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise PITDataError(f"legacy financial decision time is invalid: {dataset.dataset_id}") from exc
        if value.tzinfo is not None:
            return value
    frame = _concatenated_frames(dataset.frames)
    if "available_at" in frame.columns and frame.height:
        frame_value = frame["available_at"].max()
        if isinstance(frame_value, datetime):
            return frame_value
    raise PITDataError(f"legacy financial decision time is unavailable: {dataset.dataset_id}")


def _rebuild_financial_datasets(
    *,
    legacy: Mapping[str, LegacyDataset],
    bronze_root: Path,
    stage_runtime: DataRuntime,
    artifact_root: Path,
    results: list[MigrationResult],
    mapped: dict[str, str],
) -> None:
    """Rebuild financial facts/quality and assert legacy row equality."""

    facts_old = legacy.get("financial_facts")
    if facts_old is None:
        return
    from src.data.incremental_normalization import refresh_dart_financial_facts

    decision_time = _legacy_decision_time(facts_old)
    calendar = xkrx_session_calendar()
    facts_artifact = refresh_dart_financial_facts(
        bronze_root=bronze_root,
        silver_root=stage_runtime.workspace.silver_root,
        artifact_root=artifact_root,
        decision_time=decision_time,
        calendar=calendar,
    )
    facts_path = Path(facts_artifact.dataset_path)
    _assert_concatenated_rows_equal(facts_old.frames, facts_path, "financial_facts")
    facts_result = MigrationResult(
        "financial_facts", facts_old.dataset_id, facts_path.name, facts_path, facts_artifact.row_count
    )
    results.append(facts_result)
    mapped[facts_old.dataset_id] = facts_result.new_id

    quality_old = legacy.get("financial_quality")
    if quality_old is None:
        return
    from src.data.datasets import dataset_digest, read_dataset
    from src.data.financial_quality import (
        FinancialQualityEvent,
        build_financial_quality_events,
        materialize_financial_quality,
    )

    facts = read_dataset(facts_path).collect()
    unresolved: list[FinancialQualityEvent] = []
    old_quality = _concatenated_frames(quality_old.frames)
    if not old_quality.is_empty():
        for row in old_quality.iter_rows(named=True):
            if row.get("accounting_basis") != "unknown" or not row.get("exclusion_reason"):
                continue
            filing_ids = json.loads(str(row.get("source_filing_ids_json") or "[]"))
            filing_id = str(filing_ids[0]) if filing_ids else "legacy-unresolved"
            unresolved.append(
                FinancialQualityEvent(
                    company_id=str(row["company_id"]),
                    fiscal_period=str(row["fiscal_period"]),
                    filing_id=filing_id,
                    published_at=row["published_at"],
                    available_at=row["available_at"],
                    reason=str(row["exclusion_reason"]),
                )
            )
    events = build_financial_quality_events(
        facts,
        unresolved_events=tuple(unresolved),
        decision_time=decision_time,
    )
    quality_path = materialize_financial_quality(
        events,
        layer_root=stage_runtime.workspace.silver_root,
        decision_time=decision_time,
        facts_dataset_id=facts_result.new_id,
        quarantine_digest=dataset_digest([]),
        unresolved_events_digest=dataset_digest(
            [f"{event.company_id}:{event.fiscal_period}:{event.filing_id}" for event in unresolved]
        ),
    )
    _assert_concatenated_rows_equal(quality_old.frames, quality_path, "financial_quality")
    quality_result = MigrationResult(
        "financial_quality", quality_old.dataset_id, quality_path.name, quality_path, events.height
    )
    results.append(quality_result)
    mapped[quality_old.dataset_id] = quality_result.new_id


def _stage_runtime(runtime: DataRuntime, root: Path) -> DataRuntime:
    return DataRuntime(
        scope=runtime.scope,
        workspace=ResearchWorkspace(root=root, scope=runtime.scope),
    )


def _commit_dataset(source: Path, target_root: Path) -> Path:
    target = target_root / source.name
    target_root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_manifest = load_manifest(source)
        target_manifest = load_manifest(target)
        if source_manifest.dataset_id != target_manifest.dataset_id or source_manifest.partitions != target_manifest.partitions:
            raise PITDataError(f"existing v2 dataset differs during migration commit: {source.name}")
        shutil.rmtree(source)
        return target
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}.", suffix=".commit", dir=target_root))
    staged = temporary / source.name
    try:
        shutil.copytree(source, staged)
        os.replace(staged, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target


def _write_registry_document(state_root: Path, current: Mapping[str, str], retired: Mapping[str, str]) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    payload = {"current": dict(sorted(current.items())), "retired": dict(sorted(retired.items()))}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = state_root / f".{REGISTRY_NAME}.migration.tmp"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, state_root / REGISTRY_NAME)


def migrate_datasets_v2(
    *,
    scope_config: Path,
    data_root: Path,
    apply: bool = False,
    rules_path: Path = Path("config/market/krx_market_rules.toml"),
    definitions_path: Path = Path("config/data/reference_benchmarks.toml"),
    discovered: tuple[LegacyDataset, ...] | None = None,
) -> tuple[MigrationResult, ...]:
    """Migrate the frozen legacy scope and return the published results."""

    runtime = load_data_runtime(scope_config=scope_config, data_root=data_root)
    # Discovery happens before any publication, registry update, or deletion.
    legacy = discovered if discovered is not None else _discover(
        silver_root=runtime.workspace.silver_root,
        gold_root=runtime.workspace.gold_root,
    )
    by_kind: dict[str, LegacyDataset] = {}
    for dataset in legacy:
        kind = _legacy_kind(dataset.dataset_id, dataset.manifest)
        if kind in {"financial_facts", "financial_quality"}:
            continue
        if kind in by_kind:
            raise PITDataError(f"multiple legacy datasets have kind {kind}")
        by_kind[kind] = dataset

    ordered = [by_kind[kind] for kind in _MIGRATION_ORDER if kind in by_kind]
    rules = load_krx_market_rules(rules_path)
    bronze_root = runtime.workspace.bronze_root
    source_hashes: dict[str, list[str]] = {
        "ordinary_universe": [
            str(entry["source_hash"])
            for dataset in (by_kind.get("ordinary_universe"),)
            if dataset is not None
            for entry in _partition_entries(dataset.manifest)
            if isinstance(entry.get("source_hash"), str)
        ],
        "daily_market": [
            str(entry["source_hash"])
            for dataset in (by_kind.get("daily_market"),)
            if dataset is not None
            for entry in _partition_entries(dataset.manifest)
            if isinstance(entry.get("source_hash"), str)
        ],
        "investor_flow_ls": (
            _manifest_hashes(by_kind["investor_flow_ls"].manifest, "raw_page_hashes")
            or _manifest_hashes(by_kind["investor_flow_ls"].manifest, "source_hashes")
            or _frame_digests(by_kind["investor_flow_ls"].frames)
            if "investor_flow_ls" in by_kind
            else []
        ),
        "investor_flow_kis_supplement": (
            _manifest_hashes(by_kind["investor_flow_kis_supplement"].manifest, "kis_page_hashes")
            or _manifest_hashes(by_kind["investor_flow_kis_supplement"].manifest, "source_hashes")
            or _frame_digests(by_kind["investor_flow_kis_supplement"].frames)
            if "investor_flow_kis_supplement" in by_kind
            else []
        ),
        "industry": (
            _manifest_hashes(by_kind["industry"].manifest, "source_hashes")
            or _frame_digests(by_kind["industry"].frames)
            if "industry" in by_kind
            else []
        ),
        "dividend_events": (
            _manifest_hashes(by_kind["dividend_events"].manifest, "decision_hashes")
            or _frame_digests(by_kind["dividend_events"].frames)
            if "dividend_events" in by_kind
            else []
        ),
    }
    calendar_keys: list[str] = []
    for kind in ("daily_market", "ordinary_universe"):
        calendar_dataset = by_kind.get(kind)
        if calendar_dataset is None:
            continue
        for frame in calendar_dataset.frames.values():
            if "session" in frame.columns:
                calendar_keys.extend(str(value)[:10] for value in frame["session"].unique().to_list())
    calendar_keys = sorted(set(calendar_keys))
    dividend_calendar_keys = [
        session.astimezone(KRX_TZ).date().isoformat() for session in xkrx_session_calendar().sessions
    ]
    reference_definitions: tuple[BenchmarkDefinition, ...] | None = None
    if "reference_benchmarks" in by_kind:
        _reference_version, reference_definitions = _resolve_reference_definitions(
            by_kind["reference_benchmarks"].manifest, definitions_path
        )

    retirements: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="dataset-v2-migration-") as temporary:
        stage_runtime = _stage_runtime(runtime, Path(temporary) / "data")
        stage_runtime.workspace.initialize()
        mapped: dict[str, str] = {}
        results: list[MigrationResult] = []
        for dataset in ordered:
            kind = _legacy_kind(dataset.dataset_id, dataset.manifest)
            if kind == "investor_flow_ls":
                old_universe = _old_id_value(dataset.manifest, "universe_dataset_id")
                if old_universe and old_universe not in mapped and not (runtime.workspace.silver_root / old_universe).is_dir():
                    retirements[old_universe] = (
                        "universe generation superseded before hash-verified lineage; LS flow kept for union lineage"
                    )
            target_root = stage_runtime.workspace.gold_root if dataset.layer is DatasetLayer.GOLD else stage_runtime.workspace.silver_root
            result = _migrate_one(
                dataset,
                bronze_root=bronze_root,
                target_root=target_root,
                mapped=mapped,
                rules=rules,
                source_hashes=source_hashes,
                calendar_keys=calendar_keys,
                dividend_calendar_keys=dividend_calendar_keys,
                reference_definitions=reference_definitions,
            )
            results.append(result)

        _rebuild_financial_datasets(
            legacy={
                kind: dataset
                for dataset in legacy
                if (kind := _legacy_kind(dataset.dataset_id, dataset.manifest))
                in {"financial_facts", "financial_quality"}
            },
            bronze_root=bronze_root,
            stage_runtime=stage_runtime,
            artifact_root=Path(temporary) / "artifacts",
            results=results,
            mapped=mapped,
        )
        result_by_kind = {result.kind: result for result in results}
        if "market_panel" in result_by_kind:
            _rebuild_gold(
                runtime=stage_runtime,
                migrated=result_by_kind,
                old_by_kind=by_kind,
                rules_path=rules_path,
                definitions_path=definitions_path,
            )
        stage_registry = DatasetRegistry(
            stage_runtime.workspace.state_root,
            data_root=stage_runtime.workspace.root,
        )
        known_ids = {path.name for path in stage_registry._dataset_directories()} | set(retirements)
        for result in results:
            verification = verify_dataset(result.path, known_ids=known_ids.__contains__)
            if not verification.passed:
                raise PITDataError(f"migrated dataset failed verification: {result.new_id}: {verification.failures}")
            if load_manifest(result.path).kind != result.kind:
                raise PITDataError(f"migrated dataset kind mismatch: {result.new_id}")
        for dataset_id, reason in retirements.items():
            stage_registry.retire(dataset_id, reason)
        for result in results:
            stage_registry.register(result.kind, result.new_id)
        _verify_registered_scope(stage_runtime, stage_registry)

        committed_paths: dict[str, Path] = {}
        created_targets: list[Path] = []
        try:
            for result in results:
                target_root = runtime.workspace.gold_root if result.path.parent == stage_runtime.workspace.gold_root else runtime.workspace.silver_root
                target = target_root / result.path.name
                existed = target.exists()
                committed_paths[result.new_id] = _commit_dataset(result.path, target_root)
                if not existed:
                    created_targets.append(target)
            actual_registry = DatasetRegistry(runtime.workspace.state_root)
            current = dict(actual_registry.snapshot())
            current.update({result.kind: result.new_id for result in results})
            retired = {dataset_id: item.reason for dataset_id, item in actual_registry.retired().items()}
            retired.update(retirements)
            _write_registry_document(runtime.workspace.state_root, current, retired)
        except Exception:
            for target in created_targets:
                shutil.rmtree(target, ignore_errors=True)
            raise
        results = [replace(result, path=committed_paths[result.new_id]) for result in results]

    if apply:
        # Every legacy generation, including rebuilt financial tables, is
        # removed only after the staged v2 set and registry have passed.
        removable = list(legacy)
        for dataset in removable:
            if dataset.directory.exists() and not dataset.directory.is_symlink():
                shutil.rmtree(dataset.directory)
    return tuple(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite existing Silver/Gold datasets into v2 directories and register them."""

    args = _parse_args(argv)
    try:
        preview_runtime = load_data_runtime(scope_config=args.scope_config, data_root=args.data_root)
        preview = _discover(
            silver_root=preview_runtime.workspace.silver_root,
            gold_root=preview_runtime.workspace.gold_root,
        )
        for dataset in preview:
            sys.stdout.write(
                json.dumps(
                    {
                        "status": "discovered",
                        "dataset_id": dataset.dataset_id,
                        "layer": dataset.layer.value,
                        "path": str(dataset.directory),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        results = migrate_datasets_v2(
            scope_config=args.scope_config,
            data_root=args.data_root,
            apply=bool(args.apply),
            rules_path=args.rules,
            definitions_path=args.definitions,
            discovered=tuple(preview),
        )
    except (PITDataError, ValueError, KeyError, TypeError, OSError) as exc:
        sys.stdout.write(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True) + "\n")
        return 1
    for result in results:
        sys.stdout.write(
            json.dumps(
                {
                    "kind": result.kind,
                    "old_dataset_id": result.old_id,
                    "dataset_id": result.new_id,
                    "dataset_path": str(result.path),
                    "rows": result.rows,
                },
                sort_keys=True,
            )
            + "\n"
        )
    sys.stdout.write(json.dumps({"status": "ok", "migrated": len(results), "applied": bool(args.apply)}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
