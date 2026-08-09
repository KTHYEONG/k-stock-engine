"""Train CLI resolves repository-local canonical paths without environment names."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.core.paths import STOCK_ARTIFACT_ROOT, STOCK_DATASET_ROOT
from src.stocks.cli import train


def test_train_cli_defaults_to_canonical_roots() -> None:
    assert train.STOCK_DATASET_ROOT is STOCK_DATASET_ROOT
    assert train.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT


def test_train_cli_reads_through_stock_dataset_repository(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def read(self, dataset_id, feature_set, decision_time):
            captured["dataset_id"] = dataset_id
            captured["feature_set"] = feature_set
            return None

    monkeypatch.setattr(train, "StockDatasetRepository", lambda store: FakeRepo())

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-root", type=Path, default=train.STOCK_DATASET_ROOT)
    parser.add_argument("--registry", type=Path, default=train.STOCK_ARTIFACT_ROOT)
    args = parser.parse_args(["--dataset-id", "krx_daily_research_v1"])

    repo = train.StockDatasetRepository(None)
    repo.read(args.dataset_id, "stock_alpha_v1", None)
    assert captured == {"dataset_id": "krx_daily_research_v1", "feature_set": "stock_alpha_v1"}
