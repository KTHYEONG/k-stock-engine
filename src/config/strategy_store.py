"""Locations inside the private strategy store, selected by ``KSE_STRATEGY_STORE``."""

from __future__ import annotations

from pathlib import Path

from src.config.errors import ConfigError
from src.config.secrets import read_secret

__all__ = ["champion_path", "feature_sets_path", "store_revision", "strategy_store_root"]

_STORE_ENV = "KSE_STRATEGY_STORE"
_REVISION_FILE = "STORE_REVISION"
_STRATEGY_DIR = Path("k-stock-engine") / "kr_swing_2019_v1"
_FEATURE_SETS_FILE = Path("k-stock-engine") / "feature_sets.toml"


def strategy_store_root() -> Path | None:
    """Absolute store directory from ``KSE_STRATEGY_STORE``, or None when unset or blank.

    Raises:
        ConfigError: the variable is set to a relative path.
    """
    raw = read_secret(_STORE_ENV, default="").strip()
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute():
        raise ConfigError(f"{_STORE_ENV} must be an absolute directory: {raw!r}")
    return root


def store_revision(root: Path) -> str | None:
    """Stripped content of ``root/STORE_REVISION``, or None when the file is absent or blank."""
    try:
        text = (Path(root) / _REVISION_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def champion_path(root: Path) -> Path:
    """Production champion strategy file inside the store."""
    return Path(root) / _STRATEGY_DIR / "champion.toml"


def feature_sets_path(root: Path) -> Path:
    """Feature-set table file inside the store."""
    return Path(root) / _FEATURE_SETS_FILE
