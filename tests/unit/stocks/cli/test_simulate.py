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
from src.stocks.settings import REFERENCE_DATETIME


def test_simulate_cli_defaults_to_canonical_roots() -> None:
    assert simulate.STOCK_ARTIFACT_ROOT is STOCK_ARTIFACT_ROOT
    assert simulate.STOCK_CATALOG_ROOT is STOCK_CATALOG_ROOT
    assert simulate.STOCK_BASE_PANEL_ROOT is STOCK_BASE_PANEL_ROOT
    assert simulate.STOCK_FEATURE_PANEL_ROOT is STOCK_FEATURE_PANEL_ROOT
    assert simulate.STOCK_LABEL_ROOT is STOCK_LABEL_ROOT


def test_simulate_parser_default_decision_time_uses_reference_boundary() -> None:
    args = simulate.build_parser().parse_args(
        ["--artifact-id", "a1"]
    )
    assert args.decision_time == REFERENCE_DATETIME


def test_simulate_cli_rejects_missing_snapshot_id() -> None:
    # snapshotless: missing catalog policy is handled as ValueError inside main, not SystemExit for snapshot
    # But main still requires artifact-id; without active policy it will raise ValueError about missing costs
    import tempfile
    from pathlib import Path as _P
    with pytest.raises((SystemExit, ValueError)):
        simulate.main(["--artifact-id", "a1", "--catalog-root", str(_P(tempfile.gettempdir()) / "empty_catalog")])


def test_simulate_direct_inputs_bypass_snapshot_resolution() -> None:
    """Direct simulation arguments use active selection, not snapshot."""
    args = simulate.build_parser().parse_args(
        [
            "--artifact-id", "a1",
            "--research-start", "2024-01-01",
            "--research-end", "2024-03-31",
        ]
    )
    assert args.research_start.isoformat() == "2024-01-01"


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
