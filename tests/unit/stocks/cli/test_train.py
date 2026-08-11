"""Train CLI requires an explicit snapshot id and resolves it through the catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.core.paths import (
    STOCK_ARTIFACT_ROOT,
    STOCK_BASE_PANEL_ROOT,
    STOCK_CATALOG_ROOT,
    STOCK_FEATURE_PANEL_ROOT,
    STOCK_LABEL_ROOT,
)
from src.stocks.cli import train


def test_train_cli_defaults_to_canonical_roots() -> None:
    assert train.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert train.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert train.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert train.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert train.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT


def test_train_cli_rejects_missing_snapshot_id() -> None:
    with pytest.raises(SystemExit):
        train.main(["--artifact-id", "a1"])


def test_train_cli_resolves_snapshot_and_composes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshot:
        pass

    class FakeRepository:
        def __init__(self, **kwargs):
            captured["roots"] = kwargs

        def compose_training_snapshot(self, snapshot, **kwargs):
            captured["snapshot_id"] = snapshot.snapshot_id
            captured["compose_kwargs"] = kwargs
            return None

    def fake_resolve(catalog_root, snapshot_id, *, mode):
        captured["catalog_root"] = catalog_root
        captured["mode"] = mode
        fake = FakeSnapshot()
        fake.snapshot_id = snapshot_id
        return fake

    monkeypatch.setattr(train, "resolve_snapshot_for_mode", fake_resolve)
    monkeypatch.setattr(train, "ResearchDataRepository", FakeRepository)

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id")
    parser.add_argument("--catalog-root", type=Path, default=train.STOCK_CATALOG_ROOT)
    args = parser.parse_args(["--snapshot-id", "research_snap_1"])

    snapshot = train.resolve_snapshot_for_mode(args.catalog_root, args.snapshot_id, mode="research")
    repo = train.ResearchDataRepository(base_root=args.catalog_root)
    repo.compose_training_snapshot(snapshot, feature_set="stock_alpha_v1", decision_time=None)
    assert captured["snapshot_id"] == "research_snap_1"
    assert captured["mode"] == "research"
