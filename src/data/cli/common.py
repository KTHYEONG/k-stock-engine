"""Shared runtime resolution, JSON-line emission and the uniform error envelope."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.dataset_registry import DatasetRegistry
    from src.data.receipt_catalog import ReceiptCatalog
    from src.data.runtime import DataRuntime

__all__ = [
    "CommandFailed",
    "add_scoped_args",
    "emit",
    "register_dataset",
    "registered_id",
    "resolve_input_id",
    "scoped_catalog",
    "scoped_runtime",
]

_LOG = logging.getLogger(__name__)

_KIS_CLASSIFICATION_PACE_SECONDS = 1.0


class CommandFailed(Exception):  # noqa: N818 - carries an exit code, not just an error message
    """A command that already emitted its lines but must exit non-zero."""

    def __init__(self, exit_code: int, payload: Mapping[str, object] | None = None) -> None:
        super().__init__(f"command failed with exit code {exit_code}")
        self.exit_code = exit_code
        self.payload = dict(payload) if payload is not None else None


def emit(payload: Mapping[str, object]) -> None:
    """Write one JSON line to stdout."""
    sys.stdout.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def add_scoped_args(parser: argparse.ArgumentParser) -> None:
    """Scope-config and data-root flags shared by every scoped command."""
    parser.add_argument("--scope-config", type=Path, required=False, default=None)
    parser.add_argument("--data-root", type=Path, required=False, default=None)


def scoped_runtime(args: argparse.Namespace) -> DataRuntime:
    """Scoped runtime for scoped commands, defaulting to ``config/runtime.toml``."""
    from src.data.runtime import resolve_data_runtime

    return resolve_data_runtime(scope_config=args.scope_config, data_root=args.data_root)


def scoped_catalog(runtime: DataRuntime) -> ReceiptCatalog:
    """Scope-local receipt catalog under the workspace Bronze root."""
    from src.data.receipt_catalog import ReceiptCatalog

    return ReceiptCatalog(runtime.workspace.bronze_root / "catalog")


def registered_id(registry: DatasetRegistry, kind: str, explicit: str | None) -> str:
    """Explicit dataset id or fail closed through the registry."""
    return str(explicit) if explicit is not None else registry.require(kind)


def resolve_input_id(
    runtime: DataRuntime, registry: DatasetRegistry, kind: str, explicit: str | None
) -> str:
    """One dataset input exclusively through an explicit id or registry."""
    _ = runtime
    return registered_id(registry, kind, explicit)


def register_dataset(runtime: DataRuntime, kind: str, dataset_id: str) -> None:
    """Record one built dataset as the scope's current pointer."""
    from src.data.dataset_registry import DatasetRegistry

    DatasetRegistry(runtime.workspace.state_root).register(kind, dataset_id)
