"""Simulate CLI requires an explicit snapshot id and resolves it through the catalog."""
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
from src.stocks.cli import simulate


def test_simulate_cli_defaults_to_canonical_roots() -> None:
    assert simulate.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert simulate.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert simulate.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert simulate.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert simulate.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT


def test_simulate_cli_rejects_missing_snapshot_id() -> None:
    with pytest.raises(SystemExit):
        simulate.main(["--artifact-id", "a1"])


def test_simulate_cli_rejects_provisional_for_paper_mode(monkeypatch) -> None:
    def fake_resolve(catalog_root, snapshot_id, *, mode):
        raise ValueError(f"snapshot {snapshot_id} is provisional and cannot drive {mode} mode")

    monkeypatch.setattr(simulate, "resolve_snapshot_for_mode", fake_resolve)

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-id")
    parser.add_argument("--catalog-root", type=Path, default=simulate.STOCK_CATALOG_ROOT)
    parser.add_argument("--mode", default="paper")
    args = parser.parse_args(["--snapshot-id", "prov_snap_1", "--mode", "paper"])

    with pytest.raises(ValueError, match="provisional"):
        simulate.resolve_snapshot_for_mode(args.catalog_root, args.snapshot_id, mode=args.mode)
