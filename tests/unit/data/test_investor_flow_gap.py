"""Invariant guards for the Gold panel vs LS Silver coverage-requirement diff."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))


def _write_dataset(directory: Path, frame: pl.DataFrame, *, year: int = 2024, params: dict[str, object] | None = None) -> Path:
    from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset

    kind = "market_panel" if "gold" in directory.parts else "investor_flow_ls"
    return publish_dataset(
        layer_root=directory.parent,
        identity=DatasetIdentity(
            kind=kind,
            layer=DatasetLayer.GOLD if kind == "market_panel" else DatasetLayer.SILVER,
            policy_version="test-v1",
            inputs={},
            params=params or {},
        ),
        partitions={f"year={year}/part.parquet": frame},
    ).path


def _panel_frame() -> pl.DataFrame:
    rows = [
        (SESSIONS[0], "KRX:000001", "000001", True, "tradable"),
        (SESSIONS[1], "KRX:000001", "000001", True, "tradable"),
        (SESSIONS[0], "KRX:000002", "000002", True, "tradable"),
        (SESSIONS[2], "KRX:000002", "000002", True, "tradable"),
        (SESSIONS[0], "KRX:000003", "000003", True, "halted"),
        (SESSIONS[1], "KRX:000003", "000003", False, "tradable"),
    ]
    return pl.DataFrame(
        {
            "session": [row[0] for row in rows],
            "instrument_id": [row[1] for row in rows],
            "ticker": [row[2] for row in rows],
            "eligible": [row[3] for row in rows],
            "price_state": [row[4] for row in rows],
        },
        schema={
            "session": pl.Date,
            "instrument_id": pl.String,
            "ticker": pl.String,
            "eligible": pl.Boolean,
            "price_state": pl.String,
        },
    )


def _ls_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"session": [SESSIONS[0], SESSIONS[2]], "ticker": ["000001", "000002"]},
        schema={"session": pl.Date, "ticker": pl.String},
    )


def _gap_inputs(tmp_path: Path) -> tuple[Path, Path]:
    panel = _write_dataset(tmp_path / "gold" / "market_panel_test", _panel_frame())
    ls_flow = _write_dataset(tmp_path / "silver" / "investor_flow_test", _ls_frame())
    return panel, ls_flow


def test_compute_missing_cells_excludes_ls_covered_cells(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    panel, ls_flow = _gap_inputs(tmp_path)
    gap = compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)
    covered = {(SESSIONS[0], "000001"), (SESSIONS[2], "000002")}
    found = {(session, symbol.ticker) for symbol in gap.symbols for session in symbol.sessions}

    assert found.isdisjoint(covered)


def test_compute_missing_cells_includes_uncovered_tradable_cells(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    panel, ls_flow = _gap_inputs(tmp_path)
    gap = compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)
    by_ticker = {symbol.ticker: symbol.sessions for symbol in gap.symbols}

    assert by_ticker["000001"] == (SESSIONS[1],)
    assert by_ticker["000002"] == (SESSIONS[0],)


def test_compute_missing_cells_ignores_ineligible_or_non_tradable(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    panel, ls_flow = _gap_inputs(tmp_path)
    gap = compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)

    assert "000003" not in {symbol.ticker for symbol in gap.symbols}


def test_compute_missing_cells_pins_both_dataset_ids(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    panel, ls_flow = _gap_inputs(tmp_path)
    gap = compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)

    assert gap.market_panel_dataset_id == panel.name
    assert gap.ls_dataset_id == ls_flow.name


def test_compute_missing_cells_sorted_and_totaled(tmp_path: Path) -> None:
    from src.data.investor_flow_gap import compute_missing_investor_flow_cells

    panel, ls_flow = _gap_inputs(tmp_path)
    gap = compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)
    tickers = [symbol.ticker for symbol in gap.symbols]

    assert tickers == sorted(tickers)
    assert all(list(symbol.sessions) == sorted(symbol.sessions) for symbol in gap.symbols)
    assert gap.total_cells == sum(len(symbol.sessions) for symbol in gap.symbols) == 2


def test_compute_missing_cells_rejects_tampered_partition(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_gap import compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)
    (panel / "year=2024" / "part.parquet").write_bytes(b"tampered")

    with pytest.raises(PITDataError, match="hash mismatch"):
        compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)


def test_compute_missing_cells_rejects_invalid_manifest(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_gap import compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)
    manifest = json.loads((ls_flow / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "other"
    (ls_flow / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PITDataError, match="invalid"):
        compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)


def test_compute_missing_cells_rejects_unreadable_partition(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_gap import compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)
    (ls_flow / "year=2024" / "part.parquet").unlink()

    with pytest.raises(PITDataError, match="missing dataset partition"):
        compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)


def test_compute_missing_cells_rejects_missing_dataset(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_gap import compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)

    with pytest.raises(PITDataError, match="invalid"):
        compute_missing_investor_flow_cells(
            market_panel_path=tmp_path / "gold" / "market_panel_absent",
            ls_flow_silver_path=ls_flow,
        )


def test_compute_missing_cells_rejects_malformed_partition_entry(tmp_path: Path) -> None:
    import pytest

    from src.data.investor_flow_gap import compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)
    (panel / "manifest.json").write_text(
        json.dumps({"dataset_id": panel.name, "partitions": [{"path": "year=2024/part.parquet"}]}),
        encoding="utf-8",
    )

    with pytest.raises(PITDataError, match="invalid"):
        compute_missing_investor_flow_cells(market_panel_path=panel, ls_flow_silver_path=ls_flow)


def test_gap_reader_handles_manifest_and_dense_partition_boundaries(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.data.investor_flow_gap import _read_verified_partitions, compute_missing_investor_flow_cells
    from src.data.schemas import PITDataError

    panel, ls_flow = _gap_inputs(tmp_path)
    import src.data.investor_flow_gap as gap_module

    monkeypatch.setattr(
        gap_module,
        "dataset_partition_paths",
        lambda *_args, **_kwargs: (panel / "year=2024" / "part.parquet",),
    )
    monkeypatch.setattr(gap_module, "load_manifest", lambda _path: (_ for _ in ()).throw(PITDataError("bad manifest")))
    assert _read_verified_partitions(panel, label="panel", expected_kind="market_panel")[0] == panel.name
    monkeypatch.undo()

    with pytest.raises(PITDataError, match="kind"):
        _read_verified_partitions(ls_flow, label="flow", expected_kind="market_panel")

    panel_only = _write_dataset(
        tmp_path / "gold" / "panel_no_dense", pl.DataFrame({"value": [1]}), params={"case": "panel-no-dense"}
    )
    ls_valid = _write_dataset(tmp_path / "silver" / "ls_valid", _ls_frame())
    with pytest.raises(PITDataError, match="no dense"):
        compute_missing_investor_flow_cells(market_panel_path=panel_only, ls_flow_silver_path=ls_valid)

    panel_valid = _write_dataset(tmp_path / "gold" / "panel_valid", _panel_frame())
    ls_no_dense = _write_dataset(
        tmp_path / "silver" / "ls_no_dense", pl.DataFrame({"value": [1]}), params={"case": "ls-no-dense"}
    )
    with pytest.raises(PITDataError, match="LS flow has no dense"):
        compute_missing_investor_flow_cells(market_panel_path=panel_valid, ls_flow_silver_path=ls_no_dense)
