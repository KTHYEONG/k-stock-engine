"""Scope-namespaced paths for raw evidence, derived releases, and runtime state."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.research_scope import ResearchScope

__all__ = ["ResearchWorkspace", "build_workspace"]


@dataclass(frozen=True, slots=True)
class ResearchWorkspace:
    """Scope-namespaced paths for raw evidence, derived releases, runtime state, and immutable runs."""

    root: Path
    scope: ResearchScope

    @property
    def bronze_root(self) -> Path:
        return self.root / "bronze" / self.scope.scope_id

    @property
    def silver_root(self) -> Path:
        return self.root / "silver" / self.scope.scope_id

    @property
    def gold_root(self) -> Path:
        return self.root / "gold" / self.scope.scope_id

    @property
    def state_root(self) -> Path:
        return self.root / "state" / self.scope.scope_id

    @property
    def runs_root(self) -> Path:
        return self.root / "runs" / self.scope.scope_id

    def initialize(self) -> None:
        for layer in (self.bronze_root, self.silver_root, self.gold_root, self.state_root, self.runs_root):
            layer.mkdir(parents=True, exist_ok=True)


def build_workspace(*, data_root: Path, scope: ResearchScope) -> ResearchWorkspace:
    """Bind a validated scope to one data root without reading or selecting a previous release."""
    if not data_root.is_absolute():
        raise ValueError(f"invalid data_root {str(data_root)!r}: must be absolute")
    scope_id = scope.scope_id
    resolved_root = data_root.resolve()
    for layer in ("bronze", "silver", "gold", "state", "runs"):
        candidate = (resolved_root / layer / scope_id).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError(f"invalid scope_id {scope_id!r}: escapes data root")
    if "/" in scope_id or "\\" in scope_id or ".." in scope_id:
        raise ValueError(f"invalid scope_id {scope_id!r}")
    return ResearchWorkspace(root=data_root, scope=scope)
