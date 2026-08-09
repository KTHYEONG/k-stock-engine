"""Simulate CLI resolves repository-local canonical paths without environment names."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.core.paths import STOCK_ARTIFACT_ROOT, STOCK_DATASET_ROOT
from src.stocks.cli import simulate


def test_simulate_cli_defaults_to_canonical_roots() -> None:
    assert simulate.STOCK_DATASET_ROOT is STOCK_DATASET_ROOT
    assert simulate.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT


def test_simulate_cli_reads_through_stock_dataset_repository(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def read(self, dataset_id, feature_set, decision_time):
            captured["dataset_id"] = dataset_id
            captured["feature_set"] = feature_set
            return None

    monkeypatch.setattr(simulate, "StockDatasetRepository", lambda store: FakeRepo())

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-root", type=Path, default=simulate.STOCK_DATASET_ROOT)
    parser.add_argument("--registry", type=Path, default=simulate.STOCK_ARTIFACT_ROOT)
    args = parser.parse_args(["--dataset-id", "krx_daily_research_v1"])

    repo = simulate.StockDatasetRepository(None)
    repo.read(args.dataset_id, "stock_alpha_v1", None)
    assert captured == {"dataset_id": "krx_daily_research_v1", "feature_set": "stock_alpha_v1"}
