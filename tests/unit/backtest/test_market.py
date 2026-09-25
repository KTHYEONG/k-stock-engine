"""Dense market-array layout, read-only cache, and manifest validation tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.backtest.market import MarketArrays, load_market_arrays
from src.core.pit import PITDataError

DAY0 = date(2020, 1, 6)
DAY1 = date(2020, 1, 7)
DAY2 = date(2020, 1, 8)

_PANEL_SCHEMA: dict[str, Any] = {
    "session": pl.Date,
    "instrument_id": pl.String,
    "market": pl.String,
    "open": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "close": pl.Int64,
    "base_price": pl.Int64,
    "volume": pl.Int64,
    "tick_size": pl.Int64,
    "upper_limit": pl.Int64,
    "lower_limit": pl.Int64,
    "sell_tax_rate": pl.Float64,
    "adtv20": pl.Float64,
    "ret_vol60": pl.Float64,
    "share_factor": pl.Float64,
    "eligible": pl.Boolean,
    "open_at_upper": pl.Boolean,
    "open_at_lower": pl.Boolean,
}


def _mrow(session: date, instrument_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session": session,
        "instrument_id": instrument_id,
        "market": "KOSPI",
        "open": 100,
        "high": 110,
        "low": 90,
        "close": 105,
        "base_price": 100,
        "volume": 1000,
        "tick_size": 1,
        "upper_limit": 130,
        "lower_limit": 70,
        "sell_tax_rate": 0.002,
        "adtv20": 1_000_000_000.0,
        "ret_vol60": 0.02,
        "share_factor": 1.0,
        "eligible": True,
        "open_at_upper": False,
        "open_at_lower": False,
    }
    row.update(overrides)
    return row


def _write_panel(root: Path, name: str, rows: list[dict[str, Any]]) -> Path:
    dataset = root / name
    by_year: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        session = row["session"]
        assert isinstance(session, date)
        by_year.setdefault(session.year, []).append(row)
    partitions: list[dict[str, Any]] = []
    for year in sorted(by_year):
        rel = f"year={year}/part.parquet"
        out_path = dataset / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(by_year[year], schema=_PANEL_SCHEMA)
        frame.write_parquet(out_path)
        partitions.append({
            "year": year,
            "path": rel,
            "row_count": frame.height,
            "parquet_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        })
    (dataset / "manifest.json").write_text(
        json.dumps({"dataset_id": name, "partitions": partitions}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset


def _load(panel: Path, cache: Path) -> MarketArrays:
    return load_market_arrays(panel_dir=panel, cache_root=cache)


def _assert_equal(first: MarketArrays, second: MarketArrays) -> None:
    assert first.dataset_id == second.dataset_id
    assert first.sessions == second.sessions
    assert first.instrument_ids == second.instrument_ids
    for name, arr in first.int_fields.items():
        assert np.array_equal(arr, second.int_fields[name])
    for name, arr in first.float_fields.items():
        assert np.array_equal(arr, second.float_fields[name], equal_nan=True)
    for name, arr in first.bool_fields.items():
        assert np.array_equal(arr, second.bool_fields[name])
    assert np.array_equal(first.market, second.market)


def test_load_market_arrays_maps_cells_and_missing(tmp_path: Path) -> None:
    rows = [
        _mrow(DAY0, "KRX:A", close=100),
        _mrow(DAY0, "KRX:B", close=200),
        _mrow(DAY1, "KRX:A", close=110),
        _mrow(DAY2, "KRX:A", close=120),
        _mrow(DAY2, "KRX:B", close=220),
    ]
    panel = _write_panel(tmp_path / "gold", "market_panel_test", rows)
    arrays = _load(panel, tmp_path / "cache")

    assert arrays.sessions == (DAY0, DAY1, DAY2)
    assert arrays.instrument_ids == ("KRX:A", "KRX:B")
    close = arrays.int_fields["close"]
    assert close.tolist() == [[100, 200], [110, 0], [120, 220]]
    present = arrays.bool_fields["present"]
    assert present.tolist() == [[True, True], [True, False], [True, True]]
    assert arrays.market.tolist() == [[1, 1], [1, 0], [1, 1]]


def test_market_arrays_are_read_only(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    arrays = _load(panel, tmp_path / "cache")
    with pytest.raises(ValueError, match="read-only"):
        arrays.int_fields["close"][0, 0] = 999


def test_cache_reused_without_parquet_reads(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    cache = tmp_path / "cache"
    first = _load(panel, cache)
    for part in panel.glob("**/*.parquet"):
        part.unlink()
    with pytest.raises(PITDataError):
        _load(panel, cache)


def test_cache_with_foreign_dataset_id_rebuilt(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    cache = tmp_path / "cache"
    fresh = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "fresh")
    _load(panel, cache)
    cache_file = cache / "market_panel_test.npz"
    with np.load(str(cache_file), allow_pickle=False) as store:
        payload = {key: store[key] for key in store.files}
    payload["dataset_id"] = np.asarray("market_panel_other")
    np.savez(str(cache_file), **payload)
    rebuilt = _load(panel, cache)
    _assert_equal(fresh, rebuilt)


def test_tampered_partition_rejected(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    part = panel / "year=2020/part.parquet"
    with part.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_duplicate_rows_rejected(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold",
        "market_panel_test",
        [_mrow(DAY0, "KRX:A"), _mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")],
    )
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_manifest_dataset_id_mismatch_rejected(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A")])
    (panel / "manifest.json").write_text(
        json.dumps({"dataset_id": "market_panel_other", "partitions": []}) + "\n", encoding="utf-8"
    )
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    with pytest.raises(PITDataError):
        _load(tmp_path / "gold" / "market_panel_missing", tmp_path / "cache")


def test_invalid_partition_entry_rejected(tmp_path: Path) -> None:
    panel = tmp_path / "gold" / "market_panel_test"
    panel.mkdir(parents=True)
    (panel / "manifest.json").write_text(
        json.dumps({"dataset_id": "market_panel_test", "partitions": [{"path": "a.parquet"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_unreadable_partition_rejected(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A")])
    (panel / "year=2020/part.parquet").unlink()
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_missing_column_rejected(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A")])
    frame = pl.read_parquet(panel / "year=2020/part.parquet").drop("close")
    out_path = panel / "year=2020/part.parquet"
    frame.write_parquet(out_path)
    manifest = json.loads((panel / "manifest.json").read_text(encoding="utf-8"))
    manifest["partitions"][0]["parquet_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (panel / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PITDataError):
        _load(panel, tmp_path / "cache")


def test_corrupt_cache_rebuilt(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    cache = tmp_path / "cache"
    fresh = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "fresh")
    _load(panel, cache)
    (cache / "market_panel_test.npz").write_bytes(b"not a zip file")
    rebuilt = _load(panel, cache)
    _assert_equal(fresh, rebuilt)


def test_market_panel_with_only_exit_partition_is_rejected(tmp_path: Path) -> None:
    panel = tmp_path / "gold" / "market_panel_test"
    panel.mkdir(parents=True)
    exit_path = panel / "instrument_exits.parquet"
    pl.DataFrame(
        {"instrument_id": ["KRX:A"], "last_session": [DAY0], "last_close": [100], "exit_kind": ["traded_exit"]}
    ).write_parquet(exit_path)
    (panel / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": panel.name,
                "partitions": [{"path": exit_path.name, "parquet_sha256": hashlib.sha256(exit_path.read_bytes()).hexdigest()}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PITDataError, match="no dense partitions"):
        _load(panel, tmp_path / "cache")


def test_cache_shape_and_checksum_failures_rebuild_from_source(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "gold", "market_panel_test", [_mrow(DAY0, "KRX:A"), _mrow(DAY1, "KRX:A")]
    )
    fresh = load_market_arrays(panel_dir=panel, cache_root=tmp_path / "fresh")

    for index, mutation in enumerate(("field_shape", "market_shape", "checksum")):
        cache = tmp_path / f"cache-{index}"
        load_market_arrays(panel_dir=panel, cache_root=cache)
        cache_file = cache / "market_panel_test.npz"
        with np.load(cache_file, allow_pickle=False) as store:
            payload = {key: store[key] for key in store.files}
        if mutation == "field_shape":
            payload["int_close"] = np.asarray([1], dtype=np.int64)
        elif mutation == "market_shape":
            payload["market"] = np.asarray([1], dtype=np.int8)
        else:
            payload["checksum"] = np.asarray("wrong")
        np.savez(cache_file, **payload)
        rebuilt = load_market_arrays(panel_dir=panel, cache_root=cache)
        _assert_equal(fresh, rebuilt)
