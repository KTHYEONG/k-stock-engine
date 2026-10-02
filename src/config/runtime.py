"""Repository-anchored runtime paths loaded from ``config/runtime.toml``."""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.config.errors import ConfigError

__all__ = ["RuntimeConfig", "load_runtime_config"]

_MUTABLE_ROOTS = frozenset({"data_root", "logs_root"})


class RuntimeConfig(BaseModel):
    """Absolute repository-anchored paths shared by every CLI and tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_root: Path
    data_root: Path
    default_scope: Path
    logs_root: Path
    market_rules: Path
    reference_benchmarks: Path
    hedge_series: Path
    cash_series: Path
    engine: Path
    research_protocol: Path
    strategies_root: Path


def _repo_root_for(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _resolve(root: Path, key: str, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"runtime config {key!r} must be a non-empty path string")
    candidate = Path(value.strip())
    return candidate if candidate.is_absolute() else root / candidate


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    """Load ``config/runtime.toml`` and resolve every path to an absolute path.

    The repository root is the directory that holds ``config/runtime.toml``, found
    from this module's location rather than the working directory, so every CLI
    and tool resolves the same files regardless of where it is launched.

    Raises:
        ConfigError: file missing, unknown key, or a resolved path that does not exist
            (``data_root`` and ``logs_root`` may be created on demand).
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "runtime.toml"
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"runtime config is missing: {path}") from exc
    except ValueError as exc:
        raise ConfigError(f"runtime config is invalid TOML: {path}") from exc
    root = _repo_root_for(Path(path))
    resolved: dict[str, Path] = {"repo_root": root}
    for key, value in raw.items():
        resolved[key] = _resolve(root, key, value)
    try:
        config = RuntimeConfig.model_validate(resolved)
    except ValueError as exc:
        raise ConfigError(f"runtime config has an unknown key or invalid path: {exc}") from exc
    for key, candidate in resolved.items():
        if key in ("repo_root", *sorted(_MUTABLE_ROOTS)):
            continue
        if key == "strategies_root":
            if not candidate.is_dir():
                raise ConfigError(f"runtime config {key!r} does not exist: {candidate}")
            continue
        if not candidate.is_file():
            raise ConfigError(f"runtime config {key!r} does not exist: {candidate}")
    for key in sorted(_MUTABLE_ROOTS):
        resolved[key].mkdir(parents=True, exist_ok=True)
    return config
