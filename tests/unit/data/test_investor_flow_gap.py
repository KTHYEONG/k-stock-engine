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


def _wide_frame(
    cells: list[tuple[date, str]],
    *,
    include_session: bool = True,
    include_ticker: bool = True,
    width: int = 10,
) -> pl.DataFrame:
    data: dict[str, list[object]] = {}
    if include_session:
        data["session"] = [s for s, _ in cells]
    if include_ticker:
        data["ticker"] = [t for _, t in cells]
    n = len(cells)
    for i in range(width):
        data[f"pad{i}"] = [i] * n
    return pl.DataFrame(data)


def _publish_wide_ls(root: Path, partitions: dict[str, pl.DataFrame]) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    return publish_dataset(
        layer_root=root.parent,
        identity=DatasetIdentity(
            kind="investor_flow_ls", layer=DatasetLayer.SILVER,
            policy_version="test-v1", inputs={}, params={},
        ),
        partitions=partitions,
    ).path


def _wide_targets() -> object:
    from src.data.flow_targets import FlowTargets

    cells = pl.DataFrame(
        {
            "session": [SESSIONS[2], SESSIONS[2], SESSIONS[0], SESSIONS[1], SESSIONS[0], SESSIONS[1]],
            "ticker": ["000002", "000001", "000001", "000001", "000002", "000002"],
        },
        schema={"session": pl.Date, "ticker": pl.String},
    )
    return FlowTargets(
        universe_dataset_id="ordinary_universe_0123456789abcdef",
        daily_market_dataset_id="daily_market_0123456789abcdef",
        cells=cells,
    )


def test_gap_parity_three_wide_partitions(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import (
        InvestorFlowGapSymbol,
        MissingInvestorFlowCells,
        compute_missing_investor_flow_cells,
    )

    ls_flow = _publish_wide_ls(
        tmp_path / "silver" / "investor_flow_test",
        {
            "a/part.parquet": _wide_frame([(SESSIONS[0], "000001"), (SESSIONS[0], "000002")]),
            "b/part.parquet": _wide_frame([(SESSIONS[1], "000001"), (SESSIONS[0], "000002")]),
            "c/part.parquet": _wide_frame([(SESSIONS[0], "000001")]),
        },
    )
    gap = compute_missing_investor_flow_cells(targets=_wide_targets(), ls_flow_silver_path=ls_flow)
    assert gap == MissingInvestorFlowCells(
        universe_dataset_id="ordinary_universe_0123456789abcdef",
        daily_market_dataset_id="daily_market_0123456789abcdef",
        ls_dataset_id=ls_flow.name,
        symbols=(
            InvestorFlowGapSymbol("000001", (SESSIONS[2],)),
            InvestorFlowGapSymbol("000002", (SESSIONS[1], SESSIONS[2])),
        ),
        total_cells=3,
    )


def test_gap_skips_legacy_partition_without_row_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _publish_wide_ls(
        tmp_path / "silver" / "investor_flow_test",
        {
            "dense/part.parquet": _wide_frame([(SESSIONS[0], "000001")]),
            "legacy/part.parquet": _wide_frame(
                [(SESSIONS[0], "000001")], include_ticker=False
            ),
        },
    )
    read_paths: list[str] = []
    orig = pl.read_parquet

    def _spy(source: object, **kwargs: object) -> pl.DataFrame:
        read_paths.append(str(source))
        return orig(source, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pl, "read_parquet", _spy)
    gap = compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    assert not any("legacy/part.parquet" in path for path in read_paths)
    assert gap.total_cells == 2


def test_gap_rejects_ls_without_dense_partitions(tmp_path: Path) -> None:
    from src.core.pit import PITDataError
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _publish_wide_ls(
        tmp_path / "silver" / "investor_flow_test",
        {"legacy/part.parquet": _wide_frame([(SESSIONS[0], "000001")], include_session=False)},
    )
    with pytest.raises(PITDataError, match="LS flow has no dense partitions"):
        compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)


def test_gap_dedupes_cross_partition_cells(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import _coverage_frame, compute_missing_investor_flow_cells

    ls_flow = _publish_wide_ls(
        tmp_path / "silver" / "investor_flow_test",
        {
            "a/part.parquet": _wide_frame([(SESSIONS[0], "000001")]),
            "b/part.parquet": _wide_frame([(SESSIONS[0], "000001")]),
        },
    )
    gap = compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    found = {(s, sym.ticker) for sym in gap.symbols for s in sym.sessions}
    assert (SESSIONS[0], "000001") not in found
    assert gap.total_cells == 2
    coverage = _coverage_frame([str(ls_flow / p / "part.parquet") for p in ("a", "b")])
    assert coverage.rows() == [(SESSIONS[0], "000001")]


def test_coverage_casts_before_deduplication(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import _coverage_frame

    path = tmp_path / "part.parquet"
    pl.DataFrame({"session": [19724, 19724], "ticker": [1, 1]}).write_parquet(path)
    coverage = _coverage_frame([str(path)])
    assert coverage.schema == {"session": pl.Date, "ticker": pl.String}
    assert coverage.rows() == [(SESSIONS[0], "1")]


def test_empty_dense_partition_is_valid_coverage(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _write_ls(tmp_path / "silver" / "investor_flow_test", [])
    gap = compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    assert gap.total_cells == 3


@pytest.mark.parametrize("exists", [False, True])
def test_coverage_propagates_unreadable_partition(tmp_path: Path, exists: bool) -> None:
    from src.data.investor_flow_gap import _coverage_frame

    path = tmp_path / "part.parquet"
    if exists:
        path.write_bytes(b"not parquet")
    with pytest.raises((OSError, pl.exceptions.PolarsError)) as previous:
        pl.read_parquet(path)
    with pytest.raises(type(previous.value)):
        _coverage_frame([str(path)])


def test_gap_projects_only_session_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    ls_flow = _publish_wide_ls(
        tmp_path / "silver" / "investor_flow_test",
        {
            "a/part.parquet": _wide_frame([(SESSIONS[0], "000001")]),
            "b/part.parquet": _wide_frame([(SESSIONS[1], "000001")]),
        },
    )
    seen: list[object] = []
    orig = pl.read_parquet

    def _spy(source: object, **kwargs: object) -> pl.DataFrame:
        seen.append(kwargs.get("columns"))
        return orig(source, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pl, "read_parquet", _spy)
    compute_missing_investor_flow_cells(targets=_targets(), ls_flow_silver_path=ls_flow)
    assert seen
    assert all(columns == ["session", "ticker"] for columns in seen)


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
