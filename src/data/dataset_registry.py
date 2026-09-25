"""Scope-local current-dataset registry with verified atomic updates."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from src.core.pit import PITDataError
from src.data.datasets import (
    DatasetLayer,
    dataset_kind_from_id,
    load_manifest,
    verify_dataset,
)

REGISTRY_NAME: Final = "datasets.json"


@dataclass(frozen=True, slots=True)
class RetiredInput:
    """An intentionally absent upstream dataset retained as lineage."""

    dataset_id: str
    reason: str


class DatasetRegistry:
    """Per-scope pointers from dataset kinds to verified current dataset ids.

    The registry is stored at ``state/<scope>/datasets.json``. Exclusive
    advisory locking and same-directory replacement ensure a completed,
    verified build is the only event that moves a current pointer.
    """

    def __init__(
        self,
        state_root: Path,
        *,
        data_root: Path | None = None,
        scope_id: str | None = None,
    ) -> None:
        self._state_root = Path(state_root)
        self._path = self._state_root / REGISTRY_NAME
        self._lock_path = self._state_root / f".{REGISTRY_NAME}.lock"
        self._data_root = Path(data_root) if data_root is not None else self._state_root.parent.parent
        self._scope_id = self._state_root.name if scope_id is None else scope_id

    def current(self, kind: str) -> str | None:
        """Return the current dataset id for ``kind`` when registered."""

        current, _ = self._read()
        return current.get(kind)

    def require(self, kind: str) -> str:
        """Return the current dataset id or fail closed when it is unset."""

        dataset_id = self.current(kind)
        if dataset_id is None:
            raise PITDataError(f"dataset kind is not registered: {kind}")
        return dataset_id

    def register(self, kind: str, dataset_id: str) -> None:
        """Register a physically present, fully verified dataset atomically.

        Args:
            kind: Logical dataset kind; it must match the id prefix.
            dataset_id: Identity-bound id to make current.

        Raises:
            PITDataError: The id kind is mismatched, the dataset is absent or
                ambiguous, verification fails, or the registry cannot be read
                or replaced. The registry remains unchanged on every failure.
        """

        if dataset_kind_from_id(dataset_id) != kind:
            raise PITDataError(f"dataset kind mismatch: kind={kind!r} dataset_id={dataset_id!r}")
        candidates = self._candidate_directories(dataset_id)
        if not candidates:
            raise PITDataError(f"dataset directory does not exist: {dataset_id}")
        if len(candidates) != 1:
            raise PITDataError(f"dataset id is ambiguous across Silver and Gold: {dataset_id}")

        dataset_dir = candidates[0]
        manifest = load_manifest(dataset_dir)
        try:
            relative = dataset_dir.resolve().relative_to(self._data_root.resolve())
            expected_layer = DatasetLayer(relative.parts[0])
        except (ValueError, IndexError) as exc:
            raise PITDataError(f"dataset directory is outside the data root: {dataset_dir}") from exc
        if manifest.layer is not expected_layer:
            raise PITDataError(
                f"dataset layer does not match its root: expected={expected_layer.value} manifest={manifest.layer.value}"
            )

        _current, retired = self._read()
        known_ids = set(retired)
        for path in self._dataset_directories():
            try:
                known_ids.add(load_manifest(path).dataset_id)
            except PITDataError:
                continue
        verification = verify_dataset(dataset_dir, known_ids=known_ids.__contains__)
        if not verification.passed:
            raise PITDataError(f"dataset verification failed: {'; '.join(verification.failures)}")

        with self._locked():
            current, retired = self._read_unlocked()
            updated = dict(current)
            updated[kind] = dataset_id
            self._write_unlocked(current=updated, retired=retired)

    def retire(self, dataset_id: str, reason: str) -> None:
        """Record an intentionally absent lineage id without changing current pointers.

        This is used only by the one-shot migration tool when a historical input
        was superseded before hash-verified lineage could be retained on disk.
        The reason is mandatory so a later verifier never has to guess whether
        absence is intentional.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise PITDataError("retired dataset reason must be non-empty")
        dataset_kind_from_id(dataset_id)
        with self._locked():
            current, retired = self._read_unlocked()
            updated = dict(retired)
            previous = updated.get(dataset_id)
            if previous is not None and previous != reason:
                raise PITDataError(f"retired dataset already has a different reason: {dataset_id}")
            updated[dataset_id] = reason
            self._write_unlocked(current=current, retired=updated)

    def retired(self) -> Mapping[str, RetiredInput]:
        """Return intentionally absent lineage ids and their reasons."""

        _current, retired = self._read()
        return MappingProxyType(
            {dataset_id: RetiredInput(dataset_id=dataset_id, reason=reason) for dataset_id, reason in retired.items()}
        )

    def snapshot(self) -> Mapping[str, str]:
        """Return an immutable snapshot of all current dataset pointers."""

        current, _ = self._read()
        return MappingProxyType(dict(current))

    def _layer_roots(self) -> tuple[Path, Path]:
        if self._scope_id:
            return (
                self._data_root / "silver" / self._scope_id,
                self._data_root / "gold" / self._scope_id,
            )
        return (self._data_root / "silver", self._data_root / "gold")

    def _candidate_directories(self, dataset_id: str) -> tuple[Path, ...]:
        """Find flat and one-level table layouts without trusting symlinks."""

        candidates: list[Path] = []
        for root in self._layer_roots():
            direct = root / dataset_id
            if direct.is_dir() and not direct.is_symlink():
                candidates.append(direct)
            if not root.is_dir() or root.is_symlink():
                continue
            for table in sorted(root.iterdir()):
                if not table.is_dir() or table.is_symlink() or table.name == dataset_id:
                    continue
                nested = table / dataset_id
                if nested.is_dir() and not nested.is_symlink():
                    candidates.append(nested)
        return tuple(candidates)

    def _dataset_directories(self) -> tuple[Path, ...]:
        """Return physically present v2 dataset directories for lineage checks."""

        directories: list[Path] = []
        for root in self._layer_roots():
            if not root.is_dir() or root.is_symlink():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                    continue
                if (path / "manifest.json").is_file():
                    directories.append(path)
                    continue
                directories.extend(
                    nested
                    for nested in sorted(path.iterdir())
                    if nested.is_dir() and not nested.is_symlink() and (nested / "manifest.json").is_file()
                )
        return tuple(directories)

    def _read(self) -> tuple[dict[str, str], dict[str, str]]:
        return self._read_unlocked()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._state_root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def _read_unlocked(self) -> tuple[dict[str, str], dict[str, str]]:
        if not self._path.exists():
            return {}, {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PITDataError(f"unreadable dataset registry: {self._path}") from exc
        if not isinstance(raw, dict) or set(raw) != {"current", "retired"}:
            raise PITDataError(f"invalid dataset registry: {self._path}")
        current = self._parse_current(raw.get("current"))
        retired = self._parse_retired(raw.get("retired"))
        return current, retired

    def _write_unlocked(self, *, current: Mapping[str, str], retired: Mapping[str, str]) -> None:
        payload = {"current": dict(sorted(current.items())), "retired": dict(sorted(retired.items()))}
        try:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{REGISTRY_NAME}.", suffix=".tmp", dir=self._state_root
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self._path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise PITDataError(f"cannot write dataset registry: {self._path}") from exc

    @staticmethod
    def _parse_current(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise PITDataError("dataset registry current must be an object")
        current: dict[str, str] = {}
        for kind, dataset_id in raw.items():
            if not isinstance(kind, str) or not isinstance(dataset_id, str):
                raise PITDataError("dataset registry current entries must map strings to strings")
            if dataset_kind_from_id(dataset_id) != kind:
                raise PITDataError(f"dataset registry kind mismatch: kind={kind!r} dataset_id={dataset_id!r}")
            current[kind] = dataset_id
        return current

    @staticmethod
    def _parse_retired(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise PITDataError("dataset registry retired must be an object")
        retired: dict[str, str] = {}
        for dataset_id, reason in raw.items():
            if not isinstance(dataset_id, str) or not isinstance(reason, str) or not reason:
                raise PITDataError("dataset registry retired entries must map ids to non-empty reasons")
            dataset_kind_from_id(dataset_id)
            retired[dataset_id] = reason
        return retired
