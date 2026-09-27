"""Resolved immutable scope and workspace for production data commands."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.runtime import RuntimeConfig, load_runtime_config
from src.data.research_scope import ResearchScope, load_research_scope
from src.data.workspace import ResearchWorkspace, build_workspace

__all__ = ["DataRuntime", "load_data_runtime", "resolve_data_runtime"]


@dataclass(frozen=True, slots=True)
class DataRuntime:
    """Resolved immutable scope and workspace supplied to every data and backtest command."""

    scope: ResearchScope
    workspace: ResearchWorkspace


def load_data_runtime(*, scope_config: Path, data_root: Path) -> DataRuntime:
    """Load the sole production scope and bind all command paths to its workspace."""
    scope = load_research_scope(scope_config)
    workspace = build_workspace(data_root=data_root, scope=scope)
    return DataRuntime(scope=scope, workspace=workspace)


def resolve_data_runtime(
    *,
    scope_config: Path | None = None,
    data_root: Path | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> DataRuntime:
    """Load the scoped runtime, defaulting paths to ``config/runtime.toml``."""
    resolved = runtime_config if runtime_config is not None else load_runtime_config()
    return load_data_runtime(
        scope_config=Path(scope_config) if scope_config is not None else resolved.default_scope,
        data_root=Path(data_root) if data_root is not None else resolved.data_root,
    )
