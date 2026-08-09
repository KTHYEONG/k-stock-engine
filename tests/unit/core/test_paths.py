"""Modern filesystem paths are repository-local and independent of .env."""
from __future__ import annotations

from pathlib import Path

from src.core.paths import (
    DATA_ROOT,
    PROJECT_ROOT,
    STOCK_ARTIFACT_ROOT,
    STOCK_DATASET_ROOT,
    STOCK_FEATURE_SOURCE_ROOT,
)


def test_stock_paths_are_under_repository_data_root() -> None:
    assert Path(__file__).resolve().parents[3] == PROJECT_ROOT
    assert DATA_ROOT == PROJECT_ROOT / "data"
    assert STOCK_DATASET_ROOT == DATA_ROOT / "curated" / "stocks"
    assert STOCK_ARTIFACT_ROOT == DATA_ROOT / "artifacts" / "stocks"
    assert STOCK_FEATURE_SOURCE_ROOT == DATA_ROOT / "processed" / "features"


def test_modern_paths_do_not_reference_legacy_environment_names() -> None:
    source = Path("src/core/paths.py").read_text(encoding="utf-8")
    assert "DATA_DIR" not in source
    assert "FEATURE_STORE_PATH" not in source
