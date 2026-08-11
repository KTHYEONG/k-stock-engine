"""Curate CLI resolves repository-local paths without legacy environment names."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.core.paths import STOCK_ARTIFACT_ROOT, STOCK_DATASET_ROOT, STOCK_FEATURE_SOURCE_ROOT
from src.stocks.cli import curate
from src.stocks.cli.curate import main

DATES = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]


def legacy_row(day_index: int) -> dict[str, object]:
    return {
        "date": DATES[day_index],
        "ticker": "000050",
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1_000_000.0,
        "trading_value": 1.05e8,
        "market_cap": 1e12,
        "sector": "S1",
        "log_return_5d": 0.1,
        "volatility_20d": 0.2,
        "target_return_5d": 0.05,
    }


def write_source(root: Path) -> None:
    for i, day in enumerate(DATES):
        year_dir = root / f"year={day.year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([legacy_row(i)]).write_parquet(year_dir / f"{day.isoformat()}_feat.parquet")


def test_curate_cli_defaults_to_repository_local_roots(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    dataset_root = tmp_path / "datasets"
    write_source(source_root)
    monkeypatch.setattr(curate, "STOCK_FEATURE_SOURCE_ROOT", source_root)
    monkeypatch.setattr(curate, "STOCK_DATASET_ROOT", dataset_root)

    assert main(["--dataset-id", "krx_daily_research_v1_20240102_20240104",
                 "--start-date", "2024-01-01", "--end-date", "2024-01-31"]) == 0

    manifest_path = dataset_root / "krx_daily_research_v1_20240102_20240104" / "dataset_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "v2"
    assert manifest["content_hash"]
    assert (dataset_root / "krx_daily_research_v1_20240102_20240104" / "partitions").exists()


def test_curate_cli_requires_explicit_dataset_id(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    write_source(source_root)
    monkeypatch.setattr(curate, "STOCK_FEATURE_SOURCE_ROOT", source_root)
    monkeypatch.setattr(curate, "STOCK_DATASET_ROOT", tmp_path / "datasets")
    with pytest.raises(SystemExit):
        main(["--start-date", "2024-01-01", "--end-date", "2024-01-31"])


def test_modern_stock_clis_do_not_reference_legacy_environment_paths() -> None:
    for path in (
        Path("src/core/paths.py"),
        Path("src/stocks/cli/curate.py"),
        Path("src/stocks/cli/train.py"),
        Path("src/stocks/cli/simulate.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "DATA_DIR" not in source
        assert "FEATURE_STORE_PATH" not in source
    assert STOCK_FEATURE_SOURCE_ROOT.name == "features"
    assert STOCK_DATASET_ROOT.name == "stocks"
    assert STOCK_ARTIFACT_ROOT.name == "stocks"


def test_train_and_simulate_clis_default_to_canonical_roots() -> None:
    from src.core.paths import (
        STOCK_BASE_PANEL_ROOT,
        STOCK_CATALOG_ROOT,
        STOCK_FEATURE_PANEL_ROOT,
        STOCK_LABEL_ROOT,
    )
    from src.stocks.cli import simulate, train

    assert train.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert train.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert train.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert train.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT
    assert train.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert simulate.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert simulate.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert simulate.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert simulate.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT
    assert simulate.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
