"""Silver flow-target gap tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))


def _write_ls(root: Path, cells: list[tuple[date, str]]) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    frame = pl.DataFrame(
        {"session": [s for s, _ in cells], "ticker": [t for _, t in cells]},
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return publish_dataset(
        layer_root=root.parent,
        identity=DatasetIdentity(
            kind="investor_flow_ls", layer=DatasetLayer.SILVER,
            policy_version="test-v1", inputs={}, params={},
        ),
        partitions={"year=2024/part.parquet": frame},
    ).path


def _targets() -> object:
    from src.data.flow_targets import FlowTargets

    cells = pl.DataFrame(
        {
            "session": [SESSIONS[0], SESSIONS[1], SESSIONS[0]],
            "ticker": ["000001", "000001", "000002"],
        },
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return FlowTargets(
        universe_dataset_id="ordinary_universe_0123456789abcdef",
        daily_market_dataset_id="daily_market_0123456789abcdef",
        cells=cells,
    )


def test_gap_is_targets_minus_ls_coverage(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _write_ls(tmp_path / "silver" / "investor_flow_test", [(SESSIONS[0], "000001")])
    gap = compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    found = {(s, sym.ticker) for sym in gap.symbols for s in sym.sessions}
    assert (SESSIONS[0], "000001") not in found
    assert (SESSIONS[1], "000001") in found
    assert (SESSIONS[0], "000002") in found
    assert gap.universe_dataset_id.startswith("ordinary_universe_")
    assert gap.daily_market_dataset_id.startswith("daily_market_")
    assert gap.ls_dataset_id == ls_flow.name
    assert gap.total_cells == 2


def test_gap_sorted_and_totaled(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _write_ls(tmp_path / "silver" / "investor_flow_test", [(SESSIONS[0], "000001")])
    gap = compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    tickers = [s.ticker for s in gap.symbols]
    assert tickers == sorted(tickers)
    assert gap.total_cells == sum(len(s.sessions) for s in gap.symbols)


def test_gap_rejects_bad_ls_input(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _write_ls(tmp_path / "silver" / "investor_flow_test", [(SESSIONS[0], "000001")])
    (ls_flow / "year=2024" / "part.parquet").write_bytes(b"tampered")
    with pytest.raises(PITDataError):
        compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    with pytest.raises(PITDataError, match="invalid"):
        compute_missing_investor_flow_cells(
            targets=_targets(), ls_flow_silver_path=tmp_path / "silver" / "missing",
        )
    manifest = json.loads((ls_flow / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "other"
    (ls_flow / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PITDataError, match="invalid"):
        compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
