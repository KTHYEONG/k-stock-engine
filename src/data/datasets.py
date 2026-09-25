"""Identity-bound Parquet dataset publication, verification, and reads."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import polars as pl

from src.core.pit import PITDataError

MANIFEST_NAME: Final = "manifest.json"
MANIFEST_SCHEMA: Final = "dataset-manifest-v2"

_DATASET_ID_RE: Final = re.compile(r"(?P<kind>[A-Za-z0-9][A-Za-z0-9_-]*)_(?P<digest>[0-9a-f]{16})\Z")
_DATASET_INPUT_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*_[0-9a-f]{16}\Z")
_BRONZE_INPUT_RE: Final = re.compile(r"bronze:[0-9a-f]{64}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_DATASET_PARAM = str | int | float | bool | None
_DATASET_PARAM_VALUE = str | int | float | bool | None | datetime


class DatasetLayer(StrEnum):
    SILVER = "silver"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class DatasetCheck:
    """One quality gate recorded at build time."""

    name: str
    value: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Content-determining inputs for a dataset.

    ``inputs`` maps a role name to an upstream dataset id or a digest of Bronze
    receipt hashes. ``params`` contains only builder parameters that affect
    content, so equal identities are required to produce equal partitions.
    """

    kind: str
    layer: DatasetLayer
    policy_version: str
    inputs: Mapping[str, str]
    params: Mapping[str, _DATASET_PARAM_VALUE]

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if not isinstance(self.layer, DatasetLayer):
            raise PITDataError("dataset layer must be a DatasetLayer")
        inputs = dict(self.inputs)
        params = dict(self.params)
        for role, input_value in inputs.items():
            if not isinstance(role, str) or not role:
                raise PITDataError("dataset input roles must be non-empty strings")
            if not isinstance(input_value, str) or not (
                _DATASET_INPUT_RE.fullmatch(input_value) or _BRONZE_INPUT_RE.fullmatch(input_value)
            ):
                raise PITDataError(f"invalid dataset input {role!r}: {input_value!r}")
        for name, param_value in params.items():
            if not isinstance(name, str) or not name:
                raise PITDataError("dataset parameter names must be non-empty strings")
            _canonical_param(param_value)
        object.__setattr__(self, "inputs", MappingProxyType(inputs))
        object.__setattr__(self, "params", MappingProxyType(params))


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    """Physical Parquet partition declared by a dataset manifest."""

    path: str
    rows: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Validated metadata for one identity-bound Parquet dataset."""

    schema: str
    dataset_id: str
    kind: str
    layer: DatasetLayer
    policy_version: str
    inputs: Mapping[str, str]
    params: Mapping[str, _DATASET_PARAM_VALUE]
    partitions: tuple[DatasetPartition, ...]
    rows: int
    checks: tuple[DatasetCheck, ...]
    details: Mapping[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DatasetVerification:
    """Verification result with all independently detectable failures retained."""

    dataset_id: str
    path: Path
    rows: int
    failures: tuple[str, ...]
    failed_checks: tuple[DatasetCheck, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class PublishedDataset:
    dataset_id: str
    path: Path
    rows: int


def dataset_reference(dataset_id: str, *, kind: str) -> str:
    """Return a valid identity reference for a v2 or pre-v2 dataset name."""

    _validate_kind(kind)
    if isinstance(dataset_id, str) and _DATASET_ID_RE.fullmatch(dataset_id):
        actual_kind = dataset_kind_from_id(dataset_id)
        if actual_kind != kind:
            raise PITDataError(
                f"dataset reference kind mismatch: expected={kind!r} actual={actual_kind!r}"
            )
        return dataset_id
    if not isinstance(dataset_id, str) or not dataset_id:
        raise PITDataError(f"invalid dataset reference for {kind}")
    return f"{kind}_{hashlib.sha256(dataset_id.encode('utf-8')).hexdigest()[:16]}"


def dataset_kind_from_id(dataset_id: str) -> str:
    """Return the kind encoded in a valid dataset id."""

    match = _DATASET_ID_RE.fullmatch(dataset_id)
    if match is None:
        raise PITDataError(f"invalid dataset id: {dataset_id!r}")
    return cast("str", match.group("kind"))


def dataset_id_for(identity: DatasetIdentity) -> str:
    """Return ``<kind>_<first 16 hex of identity SHA-256>``."""

    payload = {
        "kind": identity.kind,
        "layer": identity.layer.value,
        "policy_version": identity.policy_version,
        "inputs": dict(identity.inputs),
        "params": {name: _canonical_param(value) for name, value in identity.params.items()},
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{identity.kind}_{digest[:16]}"


def publish_dataset(
    *,
    layer_root: Path,
    identity: DatasetIdentity,
    partitions: Mapping[str, pl.DataFrame],
    checks: Sequence[DatasetCheck] = (),
    details: Mapping[str, object] | None = None,
) -> PublishedDataset:
    """Atomically publish a v2 manifest and all declared Parquet partitions.

    The staged directory is a sibling of the final dataset directory. An
    existing dataset is accepted only when its identity and partition table are
    identical, making deterministic republication a true no-op.

    Args:
        layer_root: Silver or Gold directory receiving the dataset.
        identity: Content-determining dataset identity.
        partitions: Relative POSIX paths mapped to in-memory Parquet frames.
        checks: Build-time quality gates recorded in the manifest.
        details: Builder-specific counters excluded from dataset identity.

    Returns:
        The published dataset id, final path, and total row count.

    Raises:
        PITDataError: The identity or partition paths are invalid, a write
            fails, or an existing target describes different content.
    """

    root = Path(layer_root)
    dataset_id = dataset_id_for(identity)
    dataset_dir = root / dataset_id
    staging: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        _validate_checks(checks)
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", suffix=".staging", dir=root))
        entries: list[DatasetPartition] = []
        total_rows = 0
        for relative_path in sorted(partitions):
            _validate_partition_path(relative_path)
            frame = partitions[relative_path]
            output_path = staging / PurePosixPath(relative_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                frame.write_parquet(output_path)
            except (OSError, pl.exceptions.PolarsError) as exc:
                raise PITDataError(f"cannot write dataset partition {relative_path!r}") from exc
            entries.append(
                DatasetPartition(
                    path=relative_path,
                    rows=frame.height,
                    sha256=file_sha256(output_path),
                )
            )
            total_rows += frame.height

        manifest = DatasetManifest(
            schema=MANIFEST_SCHEMA,
            dataset_id=dataset_id,
            kind=identity.kind,
            layer=identity.layer,
            policy_version=identity.policy_version,
            inputs=identity.inputs,
            params=identity.params,
            partitions=tuple(entries),
            rows=total_rows,
            checks=tuple(checks),
            details=MappingProxyType(dict(details or {})),
            created_at=datetime.now(UTC),
        )
        _write_manifest(staging / MANIFEST_NAME, manifest)

        if dataset_dir.exists():
            _require_identical_partitions(dataset_dir, manifest)
            return PublishedDataset(dataset_id=dataset_id, path=dataset_dir, rows=total_rows)

        try:
            os.rename(staging, dataset_dir)
        except OSError as exc:
            if dataset_dir.exists():
                _require_identical_partitions(dataset_dir, manifest)
                return PublishedDataset(dataset_id=dataset_id, path=dataset_dir, rows=total_rows)
            raise PITDataError(f"cannot publish dataset {dataset_id}") from exc
        return PublishedDataset(dataset_id=dataset_id, path=dataset_dir, rows=total_rows)
    except OSError as exc:
        raise PITDataError(f"cannot stage or publish dataset {dataset_id}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    """Read and validate one v2 dataset manifest without reading Parquet data."""

    directory = Path(dataset_dir)
    path = directory / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PITDataError(f"unreadable dataset manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise PITDataError(f"invalid dataset manifest: {path}")
    try:
        schema = _required_str(raw, "schema")
        if schema != MANIFEST_SCHEMA:
            raise PITDataError(f"unsupported dataset manifest schema: {schema!r}")
        dataset_id = _required_dataset_id(raw, "dataset_id")
        if dataset_id != directory.name:
            raise PITDataError(f"dataset id does not match directory: {directory.name}")
        kind = _required_str(raw, "kind")
        _validate_kind(kind)
        layer = DatasetLayer(_required_str(raw, "layer"))
        policy_version = _required_str(raw, "policy_version")
        inputs = _mapping(raw, "inputs")
        params = _mapping(raw, "params")
        identity = DatasetIdentity(
            kind=kind,
            layer=layer,
            policy_version=policy_version,
            inputs=cast("Mapping[str, str]", inputs),
            params=cast("Mapping[str, _DATASET_PARAM_VALUE]", params),
        )
        if dataset_id_for(identity) != dataset_id:
            raise PITDataError(f"dataset id does not match identity: {dataset_id}")
        partitions = _parse_partitions(directory, raw.get("partitions"))
        rows = _nonnegative_int(raw.get("rows"), "rows")
        if rows != sum(partition.rows for partition in partitions):
            raise PITDataError(f"dataset row total does not match partitions: {dataset_id}")
        checks = _parse_checks(raw.get("checks"))
        details = _mapping(raw, "details")
        created_at = _created_at(raw.get("created_at"))
        return DatasetManifest(
            schema=schema,
            dataset_id=dataset_id,
            kind=kind,
            layer=layer,
            policy_version=policy_version,
            inputs=identity.inputs,
            params=identity.params,
            partitions=partitions,
            rows=rows,
            checks=checks,
            details=MappingProxyType(details),
            created_at=created_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PITDataError):
            raise
        raise PITDataError(f"invalid dataset manifest: {path}") from exc


def verify_dataset(dataset_dir: Path, *, known_ids: Callable[[str], bool]) -> DatasetVerification:
    """Verify hashes, row counts, identity, lineage, and recorded quality gates."""

    directory = Path(dataset_dir)
    try:
        manifest = load_manifest(directory)
    except PITDataError as exc:
        return DatasetVerification(
            dataset_id=directory.name,
            path=directory,
            rows=0,
            failures=(str(exc),),
            failed_checks=(),
        )

    failed_checks = tuple(check for check in manifest.checks if not check.passed)
    failures = [
        f"quality check failed: {check.name}={check.value!r} limit={check.limit!r}"
        for check in failed_checks
    ]
    for role, dataset_id in sorted(manifest.inputs.items()):
        if _DATASET_INPUT_RE.fullmatch(dataset_id) and not known_ids(dataset_id):
            failures.append(f"missing dataset input: {role}={dataset_id}")

    for partition in manifest.partitions:
        path = _safe_partition_path(directory, partition.path)
        if not path.is_file():
            failures.append(f"missing dataset partition: {partition.path}")
            continue
        try:
            actual_sha256 = file_sha256(path)
        except OSError:
            failures.append(f"unreadable dataset partition: {partition.path}")
            continue
        if actual_sha256 != partition.sha256:
            failures.append(
                f"dataset partition hash mismatch: {partition.path} expected={partition.sha256} actual={actual_sha256}"
            )
            continue
        try:
            actual_rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        except (OSError, pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            failures.append(f"unreadable dataset partition: {partition.path}")
            _ = exc
            continue
        if actual_rows != partition.rows:
            failures.append(
                f"dataset partition row mismatch: {partition.path} expected={partition.rows} actual={actual_rows}"
            )

    return DatasetVerification(
        dataset_id=manifest.dataset_id,
        path=directory,
        rows=manifest.rows,
        failures=tuple(failures),
        failed_checks=failed_checks,
    )


def read_dataset(dataset_dir: Path, *, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """Verify intrinsic dataset integrity and lazily scan all partitions."""

    directory = Path(dataset_dir)
    verification = verify_dataset(directory, known_ids=lambda _dataset_id: True)
    if not verification.passed:
        raise PITDataError(f"dataset verification failed: {'; '.join(verification.failures)}")
    manifest = load_manifest(directory)
    if not manifest.partitions:
        return pl.LazyFrame()
    frames: list[pl.LazyFrame] = []
    for partition in manifest.partitions:
        path = _safe_partition_path(directory, partition.path)
        try:
            schema_frame = pl.read_parquet(path, n_rows=0)
            source = pl.scan_parquet(path)
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            raise PITDataError(f"unreadable dataset partition: {partition.path}") from exc
        if columns is None:
            frames.append(source)
            continue
        expressions: list[pl.Expr] = []
        for column in columns:
            if column in schema_frame.columns:
                expressions.append(pl.col(column))
            else:
                expressions.append(pl.lit(None, dtype=pl.String).alias(column))
        frames.append(source.select(expressions))
    return pl.concat(frames, how="diagonal_relaxed")


def dataset_partition_paths(
    dataset_dir: Path,
    *,
    known_ids: Callable[[str], bool] | None = None,
    allow_legacy: bool = True,
) -> tuple[Path, ...]:
    """Return verified partition paths for v2 and pre-v2 dataset directories.

    R4b builders consume the v2 contract exclusively.  The small legacy reader
    is retained at this boundary so a migration can be performed without
    making every builder understand the old manifest spelling.  A manifest
    that declares the v2 schema is never downgraded to the compatibility path
    when verification fails; otherwise a tampered v2 partition could be hidden
    by the legacy fallback.
    """

    directory = Path(dataset_dir)
    try:
        raw = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PITDataError(f"unreadable dataset manifest: {directory}") from exc
    if not isinstance(raw, dict) or raw.get("dataset_id") != directory.name:
        raise PITDataError(f"invalid dataset manifest: {directory}")

    if raw.get("schema") == MANIFEST_SCHEMA:
        verification = verify_dataset(
            directory,
            known_ids=(known_ids or (lambda _dataset_id: True)),
        )
        if not verification.passed:
            raise PITDataError(f"invalid v2 dataset: {'; '.join(verification.failures)}")
        manifest = load_manifest(directory)
        return tuple(_safe_partition_path(directory, partition.path) for partition in manifest.partitions)

    if not allow_legacy:
        raise PITDataError(f"legacy dataset is not accepted by this consumer: {directory}")

    raw_parts = raw.get("partitions")
    if not isinstance(raw_parts, list):
        raise PITDataError(f"invalid dataset partitions: {directory}")
    paths: list[Path] = []
    seen: set[str] = set()
    for item in raw_parts:
        if not isinstance(item, dict):
            raise PITDataError(f"invalid dataset partition: {directory}")
        relative = item.get("path")
        digest = item.get("parquet_sha256", item.get("sha256"))
        if not isinstance(relative, str) or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise PITDataError(f"invalid dataset partition metadata: {directory}")
        if relative in seen:
            raise PITDataError(f"duplicate dataset partition path: {relative}")
        seen.add(relative)
        _validate_partition_path(relative)
        path = _safe_partition_path(directory, relative)
        try:
            if file_sha256(path) != digest:
                raise PITDataError(f"dataset partition hash mismatch: {relative}")
        except OSError as exc:
            raise PITDataError(f"unreadable dataset partition: {relative}") from exc
        paths.append(path)
    return tuple(paths)


def read_dataset_compat(
    dataset_dir: Path,
    *,
    columns: Sequence[str] | None = None,
    known_ids: Callable[[str], bool] | None = None,
) -> pl.LazyFrame:
    """Read a verified v2 dataset or a verified pre-v2 migration source.

    New code should use :func:`read_dataset`.  This adapter exists only at
    migration and compatibility edges; it deliberately has no write path.
    """

    directory = Path(dataset_dir)
    paths = dataset_partition_paths(directory, known_ids=known_ids)
    if not paths:
        return pl.LazyFrame()
    frame = pl.scan_parquet([str(path) for path in paths], hive_partitioning=True)
    return frame if columns is None else frame.select(*columns)


def dataset_digest(values: Sequence[str]) -> str:
    """Return a stable Bronze-style digest for a set of source identities."""

    normalized = sorted(dict.fromkeys(str(value) for value in values))
    return f"bronze:{hashlib.sha256(chr(0).join(normalized).encode('utf-8')).hexdigest()}"


def resolve_bronze_digest(
    value: str | None, values: Sequence[str], *, label: str = "Bronze source"
) -> str:
    """Validate an optional source digest against the pages actually read."""

    actual = dataset_digest(values)
    if value is None:
        return actual
    supplied = normalize_bronze_digest(value, values)
    if supplied != actual:
        raise PITDataError(f"{label} digest does not match verified source pages")
    return supplied


def normalize_bronze_digest(value: str | None, values: Sequence[str]) -> str:
    """Normalize an optional caller-supplied Bronze digest deterministically."""

    if value is None:
        return dataset_digest(values)
    if not isinstance(value, str) or not value:
        raise PITDataError("Bronze digest override must be a non-empty string")
    text = value.strip()
    if text.startswith("bronze:"):
        if not _BRONZE_INPUT_RE.fullmatch(text):
            raise PITDataError("Bronze digest override has an invalid bronze: value")
        return text
    if not _SHA256_RE.fullmatch(text):
        raise PITDataError("Bronze digest override must be a SHA-256 digest")
    return f"bronze:{text}"


def _legacy_partition_sessions(raw_partitions: object) -> list[date]:
    if not isinstance(raw_partitions, list):
        raise PITDataError("invalid ordinary-universe partitions")
    sessions: list[date] = []
    for item in raw_partitions:
        if not isinstance(item, dict):
            raise PITDataError("invalid ordinary-universe partition")
        raw_session = item.get("session")
        if not isinstance(raw_session, str) and raw_session is not None:
            raise PITDataError("invalid ordinary-universe partition session")
        if raw_session is None:
            path = item.get("path")
            if not isinstance(path, str) or "session=" not in path:
                raise PITDataError("ordinary-universe partition lacks session")
            raw_session = path.split("session=", 1)[1].split("/", 1)[0]
        try:
            session = date.fromisoformat(str(raw_session)[:10])
        except ValueError as exc:
            raise PITDataError("invalid ordinary-universe partition session") from exc
        if sessions and session <= sessions[-1]:
            raise PITDataError("ordinary-universe partitions are not strictly ordered")
        sessions.append(session)
    return sessions


def canonical_content_hash(frame: pl.DataFrame, ordered_columns: list[str]) -> str:
    """Return a row-order-invariant content fingerprint for migration audits."""

    rows = frame.select(ordered_columns).sort(ordered_columns).hash_rows(seed=0).to_numpy().tobytes()
    schema = "\n".join(ordered_columns).encode("utf-8")
    return hashlib.sha256(schema + b"\x00" + rows).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def universe_sessions(
    silver_root: Path, universe_id: str, *, allow_legacy: bool = True
) -> tuple[str, tuple[date, ...]]:
    """Return the explicit ordinary-universe dataset id and ordered sessions."""

    dataset = Path(silver_root) / universe_id
    try:
        raw = json.loads((dataset / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PITDataError("invalid ordinary-universe manifest") from exc
    if not isinstance(raw, dict) or raw.get("dataset_id") != dataset.name:
        raise PITDataError("invalid ordinary-universe manifest")

    if raw.get("schema") == MANIFEST_SCHEMA:
        try:
            verification = verify_dataset(dataset, known_ids=lambda _dataset_id: True)
        except PITDataError as exc:
            raise PITDataError("invalid ordinary-universe manifest") from exc
        if not verification.passed:
            raise PITDataError(f"invalid ordinary-universe manifest: {'; '.join(verification.failures)}")
        manifest = load_manifest(dataset)
        if manifest.kind != "ordinary_universe":
            raise PITDataError("invalid ordinary-universe kind")
        sessions: list[date] = []
        for partition in manifest.partitions:
            path = _safe_partition_path(dataset, partition.path)
            try:
                frame = pl.read_parquet(path, columns=["session"])
                values = frame.get_column("session").unique().to_list()
            except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
                raise PITDataError("invalid ordinary-universe partition") from exc
            if not values:
                match = re.search(r"(?:^|/)session=([^/]+)(?:/|$)", partition.path)
                if match is None:
                    raise PITDataError("ordinary-universe partition lacks session")
                try:
                    values = [date.fromisoformat(match.group(1))]
                except ValueError as exc:
                    raise PITDataError("invalid ordinary-universe partition session") from exc
            for value in values:
                if not isinstance(value, date):
                    try:
                        value = date.fromisoformat(str(value)[:10])
                    except ValueError as exc:
                        raise PITDataError("invalid ordinary-universe partition session") from exc
                if sessions and value <= sessions[-1]:
                    raise PITDataError("ordinary-universe partitions are not strictly ordered")
                sessions.append(value)
        return universe_id, tuple(sessions)

    if not allow_legacy:
        raise PITDataError("legacy ordinary-universe manifest is not accepted by this consumer")
    parts = raw.get("partitions")
    if not isinstance(parts, list) or not parts:
        raise PITDataError("invalid ordinary-universe manifest")
    return universe_id, tuple(_legacy_partition_sessions(parts))


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", kind):
        raise PITDataError("dataset kind must be a non-empty path-safe identifier")


def _canonical_param(value: object) -> _DATASET_PARAM_VALUE:
    if value is None or isinstance(value, (str, bool, int)):
        return cast("_DATASET_PARAM_VALUE", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PITDataError("dataset parameter floats must be finite")
        return value
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise PITDataError("dataset parameter datetimes require an offset")
        return value.isoformat()
    raise PITDataError(f"unsupported dataset parameter value: {value!r}")


def _validate_partition_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise PITDataError(f"invalid dataset partition path: {relative_path!r}")
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or posix_path.as_posix() != relative_path:
        raise PITDataError(f"invalid dataset partition path: {relative_path!r}")
    if any(part in ("", ".", "..") for part in posix_path.parts) or not posix_path.name:
        raise PITDataError(f"invalid dataset partition path: {relative_path!r}")
    if relative_path == MANIFEST_NAME:
        raise PITDataError("dataset partition path conflicts with the manifest")


def _safe_partition_path(dataset_dir: Path, relative_path: str) -> Path:
    _validate_partition_path(relative_path)
    root = dataset_dir.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise PITDataError(f"dataset partition escapes its directory: {relative_path!r}")
    return path


def _require_identical_partitions(dataset_dir: Path, expected: DatasetManifest) -> None:
    try:
        existing = load_manifest(dataset_dir)
    except PITDataError as exc:
        raise PITDataError(f"existing dataset has an invalid manifest (unreadable or differs): {dataset_dir}") from exc
    if existing.partitions != expected.partitions:
        raise PITDataError(f"non-deterministic rebuild differs for dataset {expected.dataset_id}")
    existing_identity = DatasetIdentity(
        kind=existing.kind,
        layer=existing.layer,
        policy_version=existing.policy_version,
        inputs=existing.inputs,
        params=existing.params,
    )
    expected_identity = DatasetIdentity(
        kind=expected.kind,
        layer=expected.layer,
        policy_version=expected.policy_version,
        inputs=expected.inputs,
        params=expected.params,
    )
    if dataset_id_for(existing_identity) != dataset_id_for(expected_identity):
        raise PITDataError(f"non-deterministic rebuild differs for dataset {expected.dataset_id}")
    manifest_path = dataset_dir / MANIFEST_NAME
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise PITDataError(f"existing dataset manifest is unreadable: {dataset_dir}") from exc
    if raw_manifest != raw_manifest.rstrip() + b"\n":
        raise PITDataError(f"non-deterministic rebuild differs for dataset {expected.dataset_id}")
    verification = verify_dataset(dataset_dir, known_ids=lambda _dataset_id: True)
    if not verification.passed:
        raise PITDataError(
            f"existing dataset failed verification: {'; '.join(verification.failures)}"
        )


def _write_manifest(path: Path, manifest: DatasetManifest) -> None:
    payload: dict[str, object] = {
        "schema": manifest.schema,
        "dataset_id": manifest.dataset_id,
        "kind": manifest.kind,
        "layer": manifest.layer.value,
        "policy_version": manifest.policy_version,
        "inputs": dict(manifest.inputs),
        "params": {name: _canonical_param(value) for name, value in manifest.params.items()},
        "partitions": [
            {
                "path": partition.path,
                "rows": partition.rows,
                "sha256": partition.sha256,
                "row_count": partition.rows,
                "parquet_sha256": partition.sha256,
            }
            for partition in manifest.partitions
        ],
        "rows": manifest.rows,
        "checks": [
            {"name": check.name, "value": check.value, "limit": check.limit, "passed": check.passed}
            for check in manifest.checks
        ],
        "details": dict(manifest.details),
        "created_at": manifest.created_at.isoformat(),
    }
    # Keep scalar builder diagnostics visible to pre-v2 operational readers;
    # details remains the canonical v2 location and the identity never uses them.
    for name, value in manifest.details.items():
        if name not in {"partitions", "table", "rows", "schema", "dataset_id"}:
            payload.setdefault(name, value)
    try:
        encoded = json.dumps(payload, allow_nan=False, default=str, indent=2, sort_keys=True) + "\n"
        path.write_text(encoded, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise PITDataError(f"cannot write dataset manifest: {path}") from exc


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise PITDataError(f"dataset manifest field {key!r} must be a non-empty string")
    return value


def _required_dataset_id(raw: Mapping[str, object], key: str) -> str:
    value = _required_str(raw, key)
    if not _DATASET_ID_RE.fullmatch(value):
        raise PITDataError(f"invalid dataset id: {value!r}")
    return value


def _mapping(raw: Mapping[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict) or any(not isinstance(item_key, str) for item_key in value):
        raise PITDataError(f"dataset manifest field {key!r} must be an object")
    return dict(value)


def _parse_partitions(dataset_dir: Path, raw: object) -> tuple[DatasetPartition, ...]:
    if not isinstance(raw, list):
        raise PITDataError("dataset manifest partitions must be a list")
    partitions: list[DatasetPartition] = []
    paths: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PITDataError("dataset manifest partition must be an object")
        path = _required_str(item, "path")
        _safe_partition_path(dataset_dir, path)
        if path in paths:
            raise PITDataError(f"duplicate dataset partition path: {path}")
        paths.add(path)
        rows = _nonnegative_int(item.get("rows"), "partition rows")
        sha256 = _required_str(item, "sha256")
        if not _SHA256_RE.fullmatch(sha256):
            raise PITDataError(f"invalid partition sha256: {sha256!r}")
        partitions.append(DatasetPartition(path=path, rows=rows, sha256=sha256))
    return tuple(partitions)


def _validate_checks(checks: Sequence[DatasetCheck]) -> None:
    seen: set[str] = set()
    for check in checks:
        if not isinstance(check, DatasetCheck):
            raise PITDataError("dataset checks must contain DatasetCheck values")
        if not isinstance(check.name, str) or not check.name:
            raise PITDataError("dataset check name must be non-empty")
        if check.name in seen:
            raise PITDataError(f"duplicate dataset check: {check.name}")
        seen.add(check.name)
        if not math.isfinite(float(check.value)) or not math.isfinite(float(check.limit)):
            raise PITDataError(f"dataset check values must be finite: {check.name}")
        if not isinstance(check.passed, bool):
            raise PITDataError(f"dataset check passed must be boolean: {check.name}")


def _parse_checks(raw: object) -> tuple[DatasetCheck, ...]:
    if not isinstance(raw, list):
        raise PITDataError("dataset manifest checks must be a list")
    checks: list[DatasetCheck] = []
    for item in raw:
        if not isinstance(item, dict):
            raise PITDataError("dataset manifest check must be an object")
        name = _required_str(item, "name")
        value = item.get("value")
        limit = item.get("limit")
        passed = item.get("passed")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PITDataError(f"check {name!r} value must be numeric")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise PITDataError(f"check {name!r} limit must be numeric")
        if not isinstance(passed, bool):
            raise PITDataError(f"check {name!r} passed must be boolean")
        checks.append(DatasetCheck(name=name, value=float(value), limit=float(limit), passed=passed))
    return tuple(checks)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PITDataError(f"dataset manifest {label} must be a non-negative integer")
    return value


def _created_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise PITDataError("dataset manifest created_at must be an ISO-8601 string")
    try:
        created_at = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PITDataError("dataset manifest created_at must be an ISO-8601 string") from exc
    if created_at.utcoffset() is None:
        raise PITDataError("dataset manifest created_at requires an offset")
    return created_at
