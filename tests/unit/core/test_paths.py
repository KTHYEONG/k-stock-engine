"""Modern filesystem paths are repository-local and independent of .env."""
from __future__ import annotations

from pathlib import Path

from src.core.paths import DATA_ROOT, PROJECT_ROOT


def test_active_paths_are_under_repository_root() -> None:
    assert Path(__file__).resolve().parents[3] == PROJECT_ROOT
    assert DATA_ROOT == PROJECT_ROOT / "data"


def test_modern_paths_do_not_reference_legacy_environment_names() -> None:
    source = Path("src/core/paths.py").read_text(encoding="utf-8")
    assert "STOCK_" not in source
    assert "DATA_DIR" not in source
    assert "FEATURE_STORE_PATH" not in source
