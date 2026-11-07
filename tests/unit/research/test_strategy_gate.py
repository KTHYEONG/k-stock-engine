"""Production fail-closed gate for champion strategy files."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError

_STORE = Path("tests/fixtures/strategy_store/k-stock-engine/kr_swing_2019_v1")
_EXAMPLE = Path("config/research/champion.example.toml")


def test_example_file_is_rejected_for_production() -> None:
    from src.research.strategy_file import require_production_spec

    with pytest.raises(ConfigError, match="example"):
        require_production_spec(_EXAMPLE)


def test_unset_store_is_rejected_even_for_store_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research.strategy_file import require_production_spec

    monkeypatch.delenv("KSE_STRATEGY_STORE")
    with pytest.raises(ConfigError, match="KSE_STRATEGY_STORE"):
        require_production_spec(_STORE / "champion.toml")


def test_store_file_is_accepted_for_production() -> None:
    from src.research.strategy_file import require_production_spec

    require_production_spec(_STORE / "champion.toml")


def test_example_parent_in_extends_chain_is_rejected(tmp_path: Path) -> None:
    from src.research.strategy_file import require_production_spec

    child = tmp_path / "child.toml"
    child.write_text(f'extends = "{_EXAMPLE.resolve()}"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        require_production_spec(child)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    from src.research.strategy_file import require_production_spec

    with pytest.raises(ConfigError, match="missing"):
        require_production_spec(tmp_path / "absent.toml")
