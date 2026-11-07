"""Strategy store location helpers resolved from ``KSE_STRATEGY_STORE``."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError
from src.config.strategy_store import champion_path, feature_sets_path, store_revision, strategy_store_root


def test_store_root_is_none_when_unset_or_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KSE_STRATEGY_STORE", raising=False)
    assert strategy_store_root() is None
    monkeypatch.setenv("KSE_STRATEGY_STORE", "   ")
    assert strategy_store_root() is None


def test_store_root_must_be_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KSE_STRATEGY_STORE", "relative")
    with pytest.raises(ConfigError):
        strategy_store_root()


def test_store_paths_and_revision(tmp_path: Path) -> None:
    assert champion_path(tmp_path) == tmp_path / "k-stock-engine" / "kr_swing_2019_v1" / "champion.toml"
    assert feature_sets_path(tmp_path) == tmp_path / "k-stock-engine" / "feature_sets.toml"
    assert store_revision(tmp_path) is None
    (tmp_path / "STORE_REVISION").write_text("  abc123\n", encoding="utf-8")
    assert store_revision(tmp_path) == "abc123"
    (tmp_path / "STORE_REVISION").write_text("\n", encoding="utf-8")
    assert store_revision(tmp_path) is None
