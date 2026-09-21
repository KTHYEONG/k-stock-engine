"""Resolved immutable scope and workspace for production data commands."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.research_scope import ResearchScope, load_research_scope
from src.data.workspace import ResearchWorkspace, build_workspace

__all__ = ["DataRuntime", "load_data_runtime"]


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
